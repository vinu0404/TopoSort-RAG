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

from auth.dependencies import db_session, get_current_user_id
from config.model_list import AgentRegistry
from config.settings import config, model_provider_for, _VALID_MODELS
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
    get_conversation_persona,
    get_or_create_conversation,
    load_agent_executions_for_conversation,
    seed_default_personas,
    poll_hitl_decision,
    update_conversation_title,
)
from tools.registry import ToolRegistry
from utils.schemas import HitlResolvedDecision, PersonaContext, Source

from utils.llm_providers import get_llm_provider
from utils.schemas import ComposerInput, QueryRequest, RecomposeRequest
from core.title_generator import generate_title

logger = logging.getLogger(__name__)


async def _build_voice_audio(
    composer: ComposerAgent,
    composer_input: ComposerInput,
) -> tuple[str, str]:
    """
    Run voice summary LLM + TTS in one shot.

    Returns (base64_audio, voice_summary_text).
    Designed to run as an asyncio.Task in parallel with text streaming.
    """
    import base64
    from voice.tts import get_tts_provider

    voice_text = await composer.generate_voice_summary_from_input(composer_input)
    tts = get_tts_provider()
    tts_result = await tts.synthesize(voice_text)
    audio_b64 = base64.b64encode(tts_result.audio_bytes).decode("ascii")
    return audio_b64, voice_text


router = APIRouter(tags=["streaming"])

@router.post("/query/stream")
async def stream_query(
    request: QueryRequest,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    return StreamingResponse(
        _stream_events(request, session, user_id),
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

        request_id = await create_hitl_request(
            session=session,
            conversation_id=conversation_id,
            agent_id=agent_id,
            agent_name=agent_name,
            tool_names=hitl_tool_names,
            task_description=task,
            timeout_seconds=config.hitl_timeout_seconds,
        )

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


async def _stream_events(request: QueryRequest, session: AsyncSession, user_id: str) -> AsyncIterator[str]:
    """
    Full pipeline:  plan (|| memory extraction) → orchestrate → compose (streamed).

    If the plan includes agents with HITL tools the generator will emit
    ``hitl_required`` events and pause orchestration until the user
    responds via ``POST /hitl/respond``.
    """
    start_time = time.perf_counter()

    try:
        await ensure_user_exists(session, user_id)
        await seed_default_personas(session, user_id)
        sess_id = await ensure_session_exists(session, user_id, request.session_id)
        conv_id = await get_or_create_conversation(
            session, user_id, sess_id,
            title=request.query[:120],
            conversation_id=request.conversation_id,
            persona_id=request.persona_id,
        )
        is_new_conversation = not request.conversation_id
        await session.commit()
        yield _sse_event("status", {"phase": "planning"})

        # Use user-selected model if valid, otherwise fall back to config defaults
        if request.model and request.model in _VALID_MODELS:
            _provider = model_provider_for(request.model)
            master_llm = get_llm_provider(_provider, default_model=request.model)
        else:
            master_llm = get_llm_provider(config.master_model_provider, default_model=config.master_model)

        # Launch title generation in background for new conversations
        title_task = None
        if is_new_conversation and not request.compare:
            title_task = asyncio.create_task(
                generate_title(master_llm, request.query, model=request.model or config.master_model)
            )

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
        if request.compare:
            plan = await plan_coro
        else:
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

        # Load persona early so it can be passed to agents (like code_agent)
        persona_data = await get_conversation_persona(session, conv_id)
        persona_ctx = PersonaContext(**persona_data) if persona_data else None

        if plan.execution_plan.agents:
            registry = ToolRegistry()
            agent_instances = build_agent_instances(
                registry,
                model_override=request.model if request.model and request.model in _VALID_MODELS else None,
            )
            orchestrator = Orchestrator(
                agent_instances=agent_instances,
                tool_registry=registry,
            )
            context = {
                "user_id": user_id,
                "query_id": plan.query_id,
                "session_id": sess_id,
                "conversation_id": str(conv_id),
                "long_term_memory": long_term.model_dump(),
                "conversation_history": conversation_history,
                "active_web_collection_ids": request.active_web_collection_ids,
                "selected_doc_ids": request.selected_doc_ids,
                "persona": persona_data,  # Pass persona dict to agents
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
            results = orch_task.result()

            if not request.compare:
                asyncio.create_task(bg_save_agent_executions(conv_id, results))
        else:
            logger.info("[Stream] No agents in plan (intent=%s) — skipping orchestration", plan.analysis.intent)
            results = {}

        for agent_id, output in results.items():
            agent_sources = []
            if hasattr(output, "metadata") and isinstance(output.metadata, dict):
                for s in output.metadata.get("sources", []):
                    if isinstance(s, dict):
                        agent_sources.append(s)
                    elif isinstance(s, Source):
                        agent_sources.append(s.model_dump())
            yield _sse_event("agent_result", {
                "agent": agent_id,
                "done": getattr(output, "task_done", False),
                "sources": agent_sources,
            })

            # Emit artifact preview events (before composition)
            if hasattr(output, "artifacts") and output.artifacts:
                for art in output.artifacts:
                    yield _sse_event("artifact_preview", {
                        "agent_id": agent_id,
                        "filename": art.filename,
                        "artifact_type": art.artifact_type,
                        "content_type": art.content_type,
                        "file_size_bytes": art.file_size_bytes,
                        "preview_data": art.preview_data,
                        "base64_data": art.base64_data,
                    })

        yield _sse_event("status", {"phase": "composing"})

        if request.model and request.model in _VALID_MODELS:
            _provider = model_provider_for(request.model)
            composer_llm = get_llm_provider(_provider, default_model=request.model)
        else:
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
            persona=persona_ctx,
            source=request.source,
        )

        # Launch voice summary + TTS in parallel with text streaming
        voice_task = None
        if request.source == "voice" and not request.compare:
            voice_task = asyncio.create_task(
                _build_voice_audio(composer, composer_input)
            )

        composer_answer = ""
        voice_audio_b64 = None
        voice_summary = None
        voice_emitted = False

        async for chunk in composer.stream(composer_input, model_override=request.model):
            composer_answer += chunk
            yield _sse_event("token", {"text": chunk})

            # Emit voice audio as soon as it's ready, while text is still streaming
            if voice_task and not voice_emitted and voice_task.done():
                try:
                    voice_audio_b64, voice_summary = voice_task.result()
                    yield _sse_event("voice_audio", {
                        "audio": voice_audio_b64,
                        "content_type": "audio/mpeg",
                    })
                except Exception:
                    logger.exception("Voice audio generation failed")
                voice_emitted = True

        # If voice task didn't finish during streaming, await it now
        if voice_task and not voice_emitted:
            try:
                voice_audio_b64, voice_summary = await voice_task
                yield _sse_event("voice_audio", {
                    "audio": voice_audio_b64,
                    "content_type": "audio/mpeg",
                })
            except Exception:
                logger.exception("Voice audio generation failed")

        if not request.compare:
            await memory_mgr.add_turn(
                user_id, request.query, composer_answer,
                db_session=session, conversation_id=conv_id,
            )
        metadata = {"sources": [s.model_dump() for s in all_sources]} if all_sources else {}
        if voice_summary:
            metadata["voice_summary"] = voice_summary

        # ── Build per-component token breakdown ─────────────────────
        token_details: dict = {}

        # Master agent tokens
        if plan.usage:
            token_details["master_agent"] = plan.usage

        # Per-agent tokens (from resource_usage populated by each agent)
        for agent_id, output in results.items():
            if hasattr(output, "resource_usage") and isinstance(output.resource_usage, dict):
                agent_tokens = output.resource_usage.get("tokens_used", 0)
                if agent_tokens:
                    token_details[agent_id] = {"total_tokens": agent_tokens}

        # Composer tokens (from stream_with_usage)
        if composer.last_stream_usage:
            token_details["composer"] = composer.last_stream_usage

        total_tokens = sum(
            entry.get("total_tokens", 0) for entry in token_details.values()
        )

        if not request.compare:
            await bg_save_messages(
                conv_id, request.query, composer_answer, metadata,
                model_used=request.model,
                total_tokens=total_tokens, token_details=token_details,
            )

        # Resolve auto-generated title for new conversations
        generated_title = None
        if title_task:
            try:
                generated_title = await title_task
                await update_conversation_title(session, conv_id, generated_title)
                await session.commit()
            except Exception:
                logger.warning("Title generation task failed", exc_info=True)

        if not request.compare:
            # ── Generate follow-up suggestions ───────────────────
            try:
                followup_prompt = (
                    "Based on this Q&A, suggest exactly 3 short follow-up questions "
                    "the user might ask next. Return ONLY the 3 questions, one per line, "
                    "no numbering, no bullet points, no extra text.\n\n"
                    f"User question: {request.query[:300]}\n\n"
                    f"Assistant answer: {composer_answer[:500]}"
                )
                followup_result = await composer_llm.generate(
                    prompt=followup_prompt,
                    temperature=0.7,
                    model=request.model if request.model and request.model in _VALID_MODELS else None,
                )
                questions = [
                    q.strip().lstrip("0123456789.-) ") for q in followup_result.text.strip().split("\n")
                    if q.strip()
                ][:3]
                if questions:
                    yield _sse_event("suggestions", {"questions": questions})
            except Exception:
                logger.debug("Follow-up suggestion generation failed", exc_info=True)

        elapsed = time.perf_counter() - start_time
        done_payload = {
            "total_time": round(elapsed, 3),
            "tokens_used": total_tokens,
            "token_details": token_details,
            "answer_length": len(composer_answer),
            "session_id": sess_id,
            "conversation_id": conv_id,
        }
        if generated_title:
            done_payload["conversation_title"] = generated_title
        if request.compare:
            done_payload["compare"] = True
        yield _sse_event("done", done_payload)

    except Exception as exc:
        logger.exception("Streaming error")
        elapsed = time.perf_counter() - start_time
        yield _sse_event("error", {
            "message": str(exc),
            "elapsed": round(elapsed, 3),
        })


# ── Recompose-only stream (for Compare feature) ───────────────────────────


@router.post("/query/recompose-stream")
async def recompose_stream(
    request: RecomposeRequest,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    return StreamingResponse(
        _recompose_events(request, session, user_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


async def _recompose_events(
    request: RecomposeRequest,
    session: AsyncSession,
    user_id: str,
) -> AsyncIterator[str]:
    """
    Reconstruct ComposerInput from DB and re-run only the Composer
    with a different model.  Nothing is saved to the DB.
    """
    import uuid as _uuid
    from sqlalchemy import select as sa_select
    from database.models import Conversation, Message
    from utils.schemas import AgentOutput

    start_time = time.perf_counter()

    try:
        # 1. Validate model
        if request.model not in _VALID_MODELS:
            yield _sse_event("error", {"message": f"Invalid model: {request.model}"})
            return

        # 2. Verify conversation ownership
        conv = (await session.execute(
            sa_select(Conversation).where(
                Conversation.conversation_id == request.conversation_id,
                Conversation.user_id == user_id,
            )
        )).scalar_one_or_none()
        if conv is None:
            yield _sse_event("error", {"message": "Conversation not found"})
            return

        yield _sse_event("status", {"phase": "reconstructing"})

        # 3. Load last user message (original query)
        last_user_msg = (await session.execute(
            sa_select(Message)
            .where(Message.conversation_id == conv.conversation_id, Message.role == "user")
            .order_by(Message.created_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if last_user_msg is None:
            yield _sse_event("error", {"message": "No user message found"})
            return

        original_query = last_user_msg.content

        # 4. Reconstruct AgentOutput objects from DB
        raw_payloads = await load_agent_executions_for_conversation(
            session, str(conv.conversation_id),
        )
        agent_results = [AgentOutput(**p) for p in raw_payloads]

        # 5. Extract sources from agent results
        all_sources: list[Source] = []
        for ao in agent_results:
            for s in (ao.metadata or {}).get("sources", []):
                if isinstance(s, dict):
                    all_sources.append(Source(
                        agent=ao.agent_id,
                        **{k: v for k, v in s.items() if k in Source.model_fields},
                    ))

        # 6. Load long-term memory
        memory_mgr = MemoryManager()
        long_term = await memory_mgr.get_long_term_memory(user_id, db_session=session)

        # 7. Load conversation history (hydrate if cache cold)
        conv_id_str = str(conv.conversation_id)
        conversation_history = memory_mgr._turns.get(conv_id_str, [])
        if not conversation_history:
            await memory_mgr._hydrate_from_db(user_id, session, conv_id_str)
            conversation_history = memory_mgr._turns.get(conv_id_str, [])

        # 8. Load persona
        persona_data = await get_conversation_persona(session, conv_id_str)
        persona_ctx = PersonaContext(**persona_data) if persona_data else None

        # 9. Build ComposerInput
        composer_input = ComposerInput(
            query_id=str(_uuid.uuid4()),
            original_query=original_query,
            user_id=user_id,
            agent_results=agent_results,
            all_sources=all_sources,
            long_term_memory=long_term,
            conversation_history=conversation_history,
            persona=persona_ctx,
            source="text",
        )

        # 10. Create composer with the requested model
        yield _sse_event("status", {"phase": "composing"})
        _provider = model_provider_for(request.model)
        composer_llm = get_llm_provider(_provider, default_model=request.model)
        composer = ComposerAgent(llm_provider=composer_llm)

        # 11. Stream the response
        composer_answer = ""
        async for chunk in composer.stream(composer_input, model_override=request.model):
            composer_answer += chunk
            yield _sse_event("token", {"text": chunk})

        # 12. Done (no DB save)
        elapsed = time.perf_counter() - start_time
        yield _sse_event("done", {
            "total_time": round(elapsed, 3),
            "tokens_used": composer.last_stream_usage.get("total_tokens", 0) if composer.last_stream_usage else 0,
            "answer_length": len(composer_answer),
            "model": request.model,
        })

    except Exception as exc:
        logger.exception("Recompose streaming error")
        elapsed = time.perf_counter() - start_time
        yield _sse_event("error", {"message": str(exc), "elapsed": round(elapsed, 3)})
