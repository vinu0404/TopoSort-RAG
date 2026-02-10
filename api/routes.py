"""
REST API routes (non-streaming).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import db_session, get_current_user_id
from config.model_list import AgentRegistry
from config.settings import config
from core.composer_agent import ComposerAgent
from core.master_agent import MasterAgent
from core.memory_extractor import MemoryExtractor
from core.memory_manager import MemoryManager
from core.agent_factory import build_agent_instances
from core.orchestrator import Orchestrator
from database.helpers import (
    ensure_user_exists,
    ensure_session_exists,
    get_or_create_conversation,
    save_agent_executions,
    save_document_record,
    save_messages,
)
from document_pipeline.document_processor import process_document
from tools.registry import ToolRegistry
from utils.schemas import Source

from utils.llm_providers import get_llm_provider
from utils.schemas import ComposerInput, QueryRequest

logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/query")
async def handle_query(
    request: QueryRequest,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Non-streaming query endpoint."""
    user_id = auth_user_id  
    await ensure_user_exists(session, user_id)
    sess_id = await ensure_session_exists(session, user_id, request.session_id)
    conv_id = await get_or_create_conversation(
        session, user_id, sess_id, title=request.query[:120],
    )
    master_llm = get_llm_provider(config.master_model_provider, default_model=config.master_model)
    memory_mgr = MemoryManager(llm_provider=master_llm)
    long_term = await memory_mgr.get_long_term_memory(user_id, db_session=session)
    conversation_history = await memory_mgr.get_conversation_history_for_agents(
        user_id, db_session=session, conversation_id=conv_id,
    )

    agent_registry = AgentRegistry()
    master = MasterAgent(agent_registry=agent_registry, llm_provider=master_llm)
    extractor = MemoryExtractor(llm_provider=master_llm)

    plan_coro = master.plan(
        request.query,
        user_id,
        conversation_history=conversation_history,
        long_term_memory=long_term.model_dump(),
    )
    extract_coro = extractor.extract_and_store(
        query=request.query,
        user_id=user_id,
        memory_manager=memory_mgr,
        db_session=session,
    )

    plan, _ = await asyncio.gather(plan_coro, extract_coro)
    # Skip orchestration for conversation_memory queries (no agents needed)
    if plan.execution_plan.agents:
        registry = ToolRegistry()
        agent_instances = build_agent_instances(registry)
        orchestrator = Orchestrator(agent_instances=agent_instances)
        context = {
            "user_id": user_id,
            "query_id": plan.query_id,
            "session_id": sess_id,
            "long_term_memory": long_term.model_dump(),
            "conversation_history": conversation_history,
        }
        results = await orchestrator.execute_plan(plan.execution_plan, context)
        await save_agent_executions(session, conv_id, results)
    else:
        logger.info("[Query] No agents in plan (intent=%s) — skipping orchestration", plan.analysis.intent)
        results = {}

    composer_llm = get_llm_provider(config.composer_model_provider, default_model=config.composer_model)
    composer = ComposerAgent(llm_provider=composer_llm)

    agent_outputs = [v for v in results.values() if hasattr(v, "agent_id")]

    all_sources: list[Source] = []
    for ao in agent_outputs:
        for s in (ao.metadata or {}).get("sources", []):
            if isinstance(s, dict):
                all_sources.append(Source(agent=ao.agent_id, **{k: v for k, v in s.items() if k in Source.model_fields}))
            elif isinstance(s, Source):
                all_sources.append(s)

    composer_input = ComposerInput(
        query_id=plan.query_id,
        original_query=request.query,
        user_id=user_id,
        agent_results=agent_outputs,
        all_sources=all_sources,
        long_term_memory=long_term,
        conversation_history=memory_mgr._turns.get(user_id, []),
    )
    output = await composer.compose(composer_input)

    await memory_mgr.add_turn(
        user_id, request.query, output.answer,
        db_session=session, conversation_id=conv_id,
    )
    await save_messages(
        session, conv_id, request.query, output.answer,
        metadata={"sources": [s.model_dump() for s in output.sources]},
    )

    return {
        "answer": output.answer,
        "sources": [s.model_dump() for s in output.sources],
        "agents_used": list(results.keys()),
    }


@router.post("/documents/upload")
async def upload_document(
    user_id: str = Form(...),
    file: UploadFile = File(...),
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Upload and index a document."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    user_id = auth_user_id
    await ensure_user_exists(session, user_id)

    file_bytes = await file.read()
    result = await process_document(
        user_id=user_id,
        file_path=file.filename,
        file_bytes=file_bytes,
    )
    await save_document_record(
        session,
        user_id=user_id,
        doc_id=result["doc_id"],
        filename=result["filename"],
        doc_type=result.get("doc_type"),
        description=result.get("description"),
        total_chunks=result.get("total_chunks"),
        qdrant_collection=f"user_{user_id}_documents",
    )

    return result


@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}
