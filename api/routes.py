"""
REST API routes (non-streaming).
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import db_session, get_current_user_id
from config.model_list import AgentRegistry
from config.settings import config
from core.composer_agent import ComposerAgent
from core.master_agent import MasterAgent
from core.memory_extractor import MemoryExtractor
from core.memory_manager import MemoryManager
from core.agent_factory import build_agent_instances
from core.orchestrator import Orchestrator
from database.helpers import (
    bg_save_agent_executions,
    bg_save_messages,
    ensure_user_exists,
    ensure_session_exists,
    get_or_create_conversation,
    list_user_conversations,
    load_conversation_messages_full,
    resolve_hitl_request,
    save_document_record,
    update_document_status,
    get_document_statuses,
)
from tasks.document_tasks import process_document_task
from tools.registry import ToolRegistry
from utils.schemas import HitlUserResponse, Source

from utils.llm_providers import get_llm_provider
from utils.schemas import ComposerInput, QueryRequest

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/agents")
async def list_agents():
    """Return all registered agents and their capabilities."""
    registry = AgentRegistry()
    return {"agents": registry.get_agent_capabilities()}


@router.post("/query")
async def handle_query(
    request: QueryRequest,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Non-streaming query endpoint."""
    await ensure_user_exists(session, user_id)
    sess_id = await ensure_session_exists(session, user_id, request.session_id)
    conv_id = await get_or_create_conversation(
        session, user_id, sess_id,
        title=request.query[:120],
        conversation_id=request.conversation_id,
    )
    await session.commit()
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
    if plan.execution_plan.agents:
        registry = ToolRegistry()
        agent_instances = build_agent_instances(registry)
        orchestrator = Orchestrator(
            agent_instances=agent_instances,
            tool_registry=registry,
        )
        context = {
            "user_id": user_id,
            "query_id": plan.query_id,
            "session_id": sess_id,
            "long_term_memory": long_term.model_dump(),
            "conversation_history": conversation_history,
        }
        # Non-streaming: no HITL callback → HITL agents are auto-skipped
        results = await orchestrator.execute_plan(
            plan.execution_plan, context, on_hitl_needed=None,
        )
        asyncio.create_task(bg_save_agent_executions(conv_id, results))
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
        conversation_history=memory_mgr._turns.get(conv_id, []),
    )
    output = await composer.compose(composer_input)

    await memory_mgr.add_turn(
        user_id, request.query, output.answer,
        db_session=session, conversation_id=conv_id,
    )
    sources_data = [s.model_dump() for s in output.sources]
    asyncio.create_task(
        bg_save_messages(conv_id, request.query, output.answer, {"sources": sources_data}),
    )

    total_tokens = sum(
        getattr(o, "resource_usage", {}).get("tokens_used", 0)
        for o in results.values()
        if hasattr(o, "resource_usage") and isinstance(getattr(o, "resource_usage", None), dict)
    )

    return {
        "answer": output.answer,
        "sources": sources_data,
        "agents_used": list(results.keys()),
        "tokens_used": total_tokens,
        "session_id": sess_id,
        "conversation_id": conv_id,
    }


@router.post("/documents/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Accept one or more files.  Creates a DB record for each with
    status='pending', enqueues a Celery task per file, and returns
    immediately so the client is not blocked.
    """
    import uuid as _uuid

    await ensure_user_exists(session, user_id)

    accepted: List[Dict[str, str]] = []

    for f in files:
        if not f.filename:
            continue
        doc_id = str(_uuid.uuid4())
        file_bytes = await f.read()

        # Persist a 'pending' record in the DB
        await save_document_record(
            session,
            user_id=user_id,
            doc_id=doc_id,
            filename=f.filename,
            qdrant_collection=f"user_{user_id}_documents",
            processing_status="pending",
        )
        await session.commit()

        # Enqueue Celery task to process the document asynchronously
        process_document_task.delay(
            user_id=user_id,
            doc_id=doc_id,
            filename=f.filename,
            file_bytes_hex=file_bytes.hex(),
        )

        accepted.append({"doc_id": doc_id, "filename": f.filename, "status": "pending"})

    if not accepted:
        raise HTTPException(status_code=400, detail="No valid files provided")

    return {"documents": accepted}


@router.get("/documents/status")
async def document_status(
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Return processing status for all of this user's documents."""
    docs = await get_document_statuses(session, auth_user_id)
    return {"documents": docs}


@router.get("/documents/status/stream")
async def document_status_stream(
    request: Request,
    auth_user_id: str = Depends(get_current_user_id),
):
    """
    SSE endpoint that pushes real-time document processing status.

    Subscribes to the Redis Pub/Sub channel ``doc_status:{user_id}``
    and forwards events to the client as SSE.  The connection stays
    open until the client disconnects.
    """
    import redis.asyncio as aioredis

    channel_name = f"doc_status:{auth_user_id}"

    async def _event_stream():
        r = aioredis.from_url(config.redis_url, decode_responses=True)
        pubsub = r.pubsub()
        await pubsub.subscribe(channel_name)

        try:
            while True:
                # Check if client disconnected
                if await request.is_disconnected():
                    break

                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if msg and msg["type"] == "message":
                    yield f"event: doc_status\ndata: {msg['data']}\n\n"
                else:
                    # Send keepalive comment every ~2s to detect disconnects
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(2)
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()
            await r.close()

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/health")
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


# ── HITL response endpoint ──────────────────────────────────────────────


@router.post("/hitl/respond")
async def hitl_respond(
    payload: HitlUserResponse,
    _auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Accept the user's approval or denial for a HITL request.

    The streaming endpoint's polling loop will pick up the change
    from the DB and resume orchestration.
    """
    final_status = await resolve_hitl_request(
        request_id=payload.request_id,
        decision=payload.decision.value,
        instructions=payload.instructions,
    )

    if final_status == "not_found":
        raise HTTPException(status_code=404, detail="HITL request not found")

    if final_status != payload.decision.value:
        raise HTTPException(
            status_code=409,
            detail=f"Request already resolved as '{final_status}'",
        )

    return {
        "request_id": payload.request_id,
        "status": final_status,
    }


# ── Conversation list / load endpoints ──────────────────────────────────


@router.get("/conversations")
async def get_conversations(
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """List all conversations for the authenticated user (newest first)."""
    convos = await list_user_conversations(session, auth_user_id)
    return {"conversations": convos}


@router.get("/conversations/{conversation_id}/messages")
async def get_conversation_messages(
    conversation_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Load all messages for a specific conversation owned by the authenticated user."""
    from database.models import Conversation
    from sqlalchemy import select as sa_select

    conv = (await session.execute(
        sa_select(Conversation).where(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == auth_user_id,
        )
    )).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    messages = await load_conversation_messages_full(session, conversation_id)
    return {"conversation_id": conversation_id, "messages": messages}
