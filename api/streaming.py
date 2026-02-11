"""
SSE-style streaming endpoint with HITL (Human-in-the-Loop) support.

Architecture
------------
While the orchestrator runs agents, it may encounter tools that require
human approval.  The orchestrator calls an ``on_hitl_needed`` callback
which:

1. Writes a ``hitl_requests`` row to the DB  (status='pending').
2. Pushes an SSE-formatted event onto an ``asyncio.Queue`` so the
   generator can emit it to the client.
3. Polls the DB until the user responds or the request times out.

The event generator runs the orchestrator as an ``asyncio.Task`` and
reads from the queue in parallel, yielding SSE events to the client as
they arrive.  The queue is **per-request** and dies with the connection
— it is NOT persistent state.  All real state lives in the DB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any, AsyncIterator, Dict, List, Optional

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
    bg_save_agent_executions,
    bg_save_messages,
    create_hitl_request,
    ensure_user_exists,
    ensure_session_exists,
    get_or_create_conversation,
    poll_hitl_decision,
)
from tools.registry import ToolRegistry
from utils.schemas import HitlResolvedDecision, Source

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


# ── HITL callback factory ──────────────────────────────────────────────


def _make_hitl_callback(
    session: AsyncSession,
    conversation_id: str,
    sse_queue: asyncio.Queue,
):
    """
    Return an async callback that the Orchestrator invokes when an agent
    has tools requiring HITL approval.

    The callback:
        1. Persists a ``hitl_requests`` row (status=pending).
        2. Pushes a ``hitl_required`` SSE event onto *sse_queue*.
        3. Polls the DB every ``config.hitl_poll_interval`` seconds.
        4. Returns a ``HitlResolvedDecision`` once the user responds or
           the request times out.
    """

    async def _on_hitl_needed(
        agent_cfg: Dict[str, Any],
        hitl_tool_names: List[str],
    ) -> HitlResolvedDecision:
        agent_id = agent_cfg["agent_id"]
        agent_name = agent_cfg["agent_name"]
        task = agent_cfg.get("task", "")

        # 1. Persist to DB
        request_id = await create_hitl_request(
            session=session,
            conversation_id=conversation_id,
            agent_id=agent_id,
            agent_name=agent_name,
            tool_names=hitl_tool_names,
            task_description=task,
            timeout_seconds=config.hitl_timeout_seconds,
        )

        # 2. Push SSE event so the client can render an approval dialog
        await sse_queue.put(_sse_event("hitl_required", {
            "request_id": request_id,
            "agent_id": agent_id,
            "agent_name": agent_name,
            "tool_names": hitl_tool_names,
            "task_description": task,
            "timeout_seconds": config.hitl_timeout_seconds,
        }))

        # 3. Poll DB until resolved or timed out
        deadline = time.monotonic() + config.hitl_timeout_seconds
        while time.monotonic() < deadline:
            decision = await poll_hitl_decision(request_id)
            if decision is not None:
                status = decision["status"]
                if status == "approved":
                    await sse_queue.put(_sse_event("hitl_approved", {
                        "request_id": request_id,
                        "agent_id": agent_id,
                        "instructions": decision.get("user_instructions"),
                    }))
                    return HitlResolvedDecision(
                        approved=True,
                        instructions=decision.get("user_instructions"),
                        tool_names=hitl_tool_names,
                    )
                elif status == "denied":
                    await sse_queue.put(_sse_event("hitl_denied", {
                        "request_id": request_id,
                        "agent_id": agent_id,
                    }))
                    return HitlResolvedDecision(
                        approved=False,
                        tool_names=hitl_tool_names,
                        reason="denied_by_user",
                    )
                else:
                    # timed_out / expired
                    await sse_queue.put(_sse_event("hitl_timeout", {
                        "request_id": request_id,
                        "agent_id": agent_id,
                    }))
                    return HitlResolvedDecision(
                        approved=False,
                        tool_names=hitl_tool_names,
                        reason=f"hitl_{status}",
                    )
            await asyncio.sleep(config.hitl_poll_interval)

        # Deadline passed — timeout
        await sse_queue.put(_sse_event("hitl_timeout", {
            "request_id": request_id,
            "agent_id": agent_id,
        }))
        return HitlResolvedDecision(
            approved=False,
            tool_names=hitl_tool_names,
            reason="hitl_timeout",
        )

    return _on_hitl_needed


# ── SSE stream generator ───────────────────────────────────────────────


async def _stream_events(request: QueryRequest, session: AsyncSession) -> AsyncIterator[str]:
    """
    Full pipeline:  plan (|| memory extraction) → orchestrate → compose (streamed).

    If the plan includes agents with HITL tools the generator will emit
    ``hitl_required`` events and pause orchestration until the user
    responds via ``POST /hitl/respond``.
    """
    start_time = time.perf_counter()
    user_id = request.user_id

    try:
        await ensure_user_exists(session, user_id)
        sess_id = await ensure_session_exists(session, user_id, request.session_id)
        conv_id = await get_or_create_conversation(
            session, user_id, sess_id, title=request.query[:120],
        )
        await session.commit()
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

            # ── HITL-aware orchestration ────────────────────────────
            sse_queue: asyncio.Queue = asyncio.Queue()
            on_hitl = _make_hitl_callback(session, conv_id, sse_queue)

            orch_task = asyncio.create_task(
                orchestrator.execute_plan(plan.execution_plan, context, on_hitl_needed=on_hitl)
            )

            # Drain SSE queue while orchestrator runs (HITL events)
            while not orch_task.done():
                try:
                    event_str = await asyncio.wait_for(sse_queue.get(), timeout=0.3)
                    yield event_str
                except asyncio.TimeoutError:
                    pass

            # Drain any remaining events queued just before task finished
            while not sse_queue.empty():
                yield sse_queue.get_nowait()

            # Retrieve results (will re-raise if orchestrator failed)
            results = orch_task.result()

            asyncio.create_task(bg_save_agent_executions(conv_id, results))
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
        metadata = {"sources": [s.model_dump() for s in all_sources]} if all_sources else {}
        asyncio.create_task(bg_save_messages(conv_id, request.query, composer_answer, metadata))
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
