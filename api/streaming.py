"""
SSE-style streaming endpoint.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
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
    save_messages,
)
from tools.registry import ToolRegistry
from utils.schemas import Source

from utils.llm_providers import get_llm_provider
from utils.schemas import ComposerInput, QueryRequest

logger = logging.getLogger(__name__)

router = APIRouter(tags=["streaming"])

@router.post("/query/stream")
async def stream_query(
    request: QueryRequest,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
):
    request.user_id = auth_user_id
    return StreamingResponse(
        _stream_events(request, session),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


def _sse_event(event: str, data: Any) -> str:
    """Format a single SSE message."""
    payload = json.dumps(data) if not isinstance(data, str) else data
    return f"event: {event}\ndata: {payload}\n\n"


async def _stream_events(request: QueryRequest, session: AsyncSession) -> AsyncIterator[str]:
    """
    Full pipeline:  plan (|| memory extraction) → orchestrate → compose (streamed).
    """
    start_time = time.perf_counter()
    user_id = request.user_id

    try:
        await ensure_user_exists(session, user_id)
        sess_id = await ensure_session_exists(session, user_id, request.session_id)
        conv_id = await get_or_create_conversation(
            session, user_id, sess_id, title=request.query[:120],
        )
        yield _sse_event("status", {"phase": "planning"})

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

        yield _sse_event("plan", {
            "intent": plan.analysis.intent,
            "agents": [t.model_dump() for t in plan.execution_plan.agents],
        })

        yield _sse_event("status", {"phase": "executing"})

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
            logger.info("[Stream] No agents in plan (intent=%s) — skipping orchestration", plan.analysis.intent)
            results = {}

        for agent_id, output in results.items():
            yield _sse_event("agent_result", {
                "agent": agent_id,
                "done": getattr(output, "task_done", False),
            })

        yield _sse_event("status", {"phase": "composing"})

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

        composer_answer = ""
        async for chunk in composer.stream(composer_input):
            composer_answer += chunk
            yield _sse_event("token", {"text": chunk})
        await memory_mgr.add_turn(
            user_id, request.query, composer_answer,
            db_session=session, conversation_id=conv_id,
        )
        await save_messages(session, conv_id, request.query, composer_answer)
        elapsed = time.perf_counter() - start_time
        yield _sse_event("done", {
            "total_time": round(elapsed, 3),
            "answer_length": len(composer_answer),
        })

    except Exception as exc:
        logger.exception("Streaming error")
        elapsed = time.perf_counter() - start_time
        yield _sse_event("error", {
            "message": str(exc),
            "elapsed": round(elapsed, 3),
        })
