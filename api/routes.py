"""
REST API routes (non-streaming).
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import db_session, get_current_user_id
from config.model_list import AgentRegistry
from config.settings import config, model_provider_for, _VALID_MODELS, MODEL_CATALOG


_key_pool = ThreadPoolExecutor(max_workers=4)


def _get_provider_key(provider: str) -> str | None:
    """Return the raw API key string for a provider, or None."""
    key_map = {
        "openai": config.openai_api_key,
        "anthropic": config.anthropic_api_key,
        "google": config.google_api_key,
    }
    val = key_map.get(provider)
    return val if val and val.strip() else None


def _validate_openai(key: str) -> bool:
    """Lightweight OpenAI key check — list-models call."""
    try:
        from openai import OpenAI
        c = OpenAI(api_key=key, timeout=8)
        c.models.list()  # minimal authenticated call
        return True
    except Exception:
        return False


def _validate_anthropic(key: str) -> bool:
    """Lightweight Anthropic key check — tiny message."""
    try:
        from anthropic import Anthropic
        c = Anthropic(api_key=key, timeout=8)
        c.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=1,
            messages=[{"role": "user", "content": "hi"}],
        )
        return True
    except Exception:
        return False


def _validate_google(key: str) -> bool:
    """Lightweight Google key check — list-models call."""
    try:
        import google.generativeai as genai
        genai.configure(api_key=key)
        list(genai.list_models())  # authenticated call
        return True
    except Exception:
        return False


_VALIDATORS = {
    "openai": _validate_openai,
    "anthropic": _validate_anthropic,
    "google": _validate_google,
}
from core.composer_agent import ComposerAgent
from core.master_agent import MasterAgent
from core.memory_extractor import MemoryExtractor
from core.memory_manager import MemoryManager
from core.agent_factory import build_agent_instances
from core.orchestrator import Orchestrator
from database.helpers import (
    bg_save_agent_executions,
    bg_save_messages,
    create_persona,
    create_share_token,
    create_web_scrape_collection,
    delete_document_record,
    delete_persona,
    delete_web_collection_record,
    ensure_user_exists,
    ensure_session_exists,
    get_conversation_persona,
    get_user_web_collections,
    get_web_collection_for_user,
    seed_default_personas,
    get_or_create_conversation,
    get_document_for_user,
    get_user_personas,
    toggle_web_collection_active,
    update_persona,
    list_user_conversations,
    load_conversation_messages_full,
    resolve_hitl_request,
    revoke_share_token,
    save_document_record,
    get_document_statuses,
    update_conversation_title,
)
from tasks.document_tasks import process_document_task
from tools.registry import ToolRegistry
from utils.schemas import HitlUserResponse, Source
from utils.llm_providers import get_llm_provider
from utils.schemas import (
    ComposerInput, PersonaCreate, PersonaContext, PersonaUpdate, QueryRequest,
    WebScrapeCollectionCreate, WebScrapeToggle,
)
from core.title_generator import generate_title

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models", tags=["models"])
async def list_models():
    """Return available models grouped by provider, with availability flag.

    API keys are validated in parallel via threads so the endpoint stays
    responsive even if a provider is slow to respond.
    """
    loop = asyncio.get_running_loop()

    # Launch one validation per provider in parallel
    futures: dict[str, asyncio.Future] = {}
    for provider in MODEL_CATALOG:
        key = _get_provider_key(provider)
        if not key:
            futures[provider] = loop.create_future()
            futures[provider].set_result(False)
        else:
            validator = _VALIDATORS.get(provider)
            if validator:
                futures[provider] = loop.run_in_executor(_key_pool, validator, key)
            else:
                futures[provider] = loop.create_future()
                futures[provider].set_result(bool(key))

    # Await all results together
    availability = {}
    for provider, fut in futures.items():
        try:
            availability[provider] = await fut
        except Exception:
            availability[provider] = False

    result = {}
    for provider, models in MODEL_CATALOG.items():
        result[provider] = {
            "available": availability.get(provider, False),
            "models": models,
        }
    return {"models": result}


@router.get("/agents", tags=["agents"])
async def list_agents():
    """Return all registered agents and their capabilities."""
    registry = AgentRegistry()
    return {"agents": registry.get_agent_capabilities()}


@router.post("/query", tags=["query"])
async def handle_query(
    request: QueryRequest,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Non-streaming query endpoint."""
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
    # Use user-selected model if valid, otherwise fall back to config defaults
    if request.model and request.model in _VALID_MODELS:
        _provider = model_provider_for(request.model)
        master_llm = get_llm_provider(_provider, default_model=request.model)
    else:
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
            "active_web_collection_ids": request.active_web_collection_ids,
            "selected_doc_ids": request.selected_doc_ids,
        }
        # Non-streaming: no HITL callback → HITL agents are auto-skipped
        results = await orchestrator.execute_plan(
            plan.execution_plan, context, on_hitl_needed=None,
        )
        asyncio.create_task(bg_save_agent_executions(conv_id, results))
    else:
        logger.info("[Query] No agents in plan (intent=%s) — skipping orchestration", plan.analysis.intent)
        results = {}

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

    persona_data = await get_conversation_persona(session, conv_id)
    composer_input = ComposerInput(
        query_id=plan.query_id,
        original_query=request.query,
        user_id=user_id,
        agent_results=agent_outputs,
        all_sources=all_sources,
        long_term_memory=long_term,
        conversation_history=memory_mgr._turns.get(conv_id, []),
        persona=PersonaContext(**persona_data) if persona_data else None,
    )
    output = await composer.compose(composer_input)

    await memory_mgr.add_turn(
        user_id, request.query, output.answer,
        db_session=session, conversation_id=conv_id,
    )
    sources_data = [s.model_dump() for s in output.sources]

    # ── Build per-component token breakdown ─────────────────────
    token_details: dict = {}

    # Master agent tokens
    if plan.usage:
        token_details["master_agent"] = plan.usage

    # Per-agent tokens
    for agent_id, agent_out in results.items():
        if hasattr(agent_out, "resource_usage") and isinstance(agent_out.resource_usage, dict):
            agent_tokens = agent_out.resource_usage.get("tokens_used", 0)
            if agent_tokens:
                token_details[agent_id] = {"total_tokens": agent_tokens}

    # Composer tokens (from generate)
    if output.usage:
        token_details["composer"] = output.usage

    total_tokens = sum(
        entry.get("total_tokens", 0) for entry in token_details.values()
    )

    asyncio.create_task(
        bg_save_messages(
            conv_id, request.query, output.answer, {"sources": sources_data},
            model_used=request.model,
            total_tokens=total_tokens, token_details=token_details,
        ),
    )

    # Auto-generate title for new conversations
    if is_new_conversation:
        async def _bg_title():
            try:
                title = await generate_title(composer_llm, request.query, model=request.model or config.composer_model)
                from database.session import async_session_factory
                async with async_session_factory() as s:
                    await update_conversation_title(s, conv_id, title)
                    await s.commit()
            except Exception:
                logger.warning("Background title generation failed", exc_info=True)
        asyncio.create_task(_bg_title())

    return {
        "answer": output.answer,
        "sources": sources_data,
        "agents_used": list(results.keys()),
        "tokens_used": total_tokens,
        "token_details": token_details,
        "session_id": sess_id,
        "conversation_id": conv_id,
    }


@router.post("/documents/upload", tags=["documents"])
async def upload_documents(
    files: List[UploadFile] = File(...),
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Accept one or more files.  Creates a DB record for each with
    status='pending' and enqueues a Celery task per file.  The Celery
    worker handles S3 upload + document processing — so the API
    response is instant regardless of how many files are uploaded.
    """
    import uuid as _uuid
    from storage.s3 import build_storage_key

    await ensure_user_exists(session, user_id)

    accepted: List[Dict[str, str]] = []

    for f in files:
        if not f.filename:
            continue
        doc_id = str(_uuid.uuid4())
        file_bytes = await f.read()
        ct = f.content_type or "application/octet-stream"
        s_key = build_storage_key(user_id, doc_id, f.filename)

        # Persist a 'pending' record in the DB (no S3 upload here)
        await save_document_record(
            session,
            user_id=user_id,
            doc_id=doc_id,
            filename=f.filename,
            qdrant_collection=f"user_{user_id}_documents",
            processing_status="pending",
            storage_key=s_key,
            storage_bucket=config.s3_bucket,
            file_size_bytes=len(file_bytes),
            content_type=ct,
        )
        await session.commit()

        # Enqueue Celery task — worker does S3 upload + processing
        process_document_task.delay(
            user_id=user_id,
            doc_id=doc_id,
            filename=f.filename,
            file_bytes_hex=file_bytes.hex(),
            content_type=ct,
            storage_key=s_key,
        )

        accepted.append({"doc_id": doc_id, "filename": f.filename, "status": "pending"})

    if not accepted:
        raise HTTPException(status_code=400, detail="No valid files provided")

    return {"documents": accepted}


@router.get("/documents/status", tags=["documents"])
async def document_status(
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Return processing status for all of this user's documents."""
    docs = await get_document_statuses(session, auth_user_id)
    return {"documents": docs}


@router.get("/documents/{doc_id}/view", tags=["documents"])
async def document_view(
    doc_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Generate a short-lived pre-signed URL for viewing a document.

    Security:
      1. JWT verified → user_id
      2. DB ownership check: ``WHERE doc_id = ? AND user_id = ?``
      3. S3 path scoped to ``uploads/{user_id}/...``
      4. URL expires in ``s3_presign_expiry`` seconds (default 5 min)
    """
    from storage.s3 import generate_presigned_url

    doc = await get_document_for_user(session, doc_id, auth_user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    if not doc.storage_key:
        raise HTTPException(status_code=404, detail="File not available in cloud storage")

    url = generate_presigned_url(
        storage_key=doc.storage_key,
        bucket=doc.storage_bucket,
    )

    return {
        "url": url,
        "filename": doc.filename,
        "content_type": doc.content_type,
        "expires_in": config.s3_presign_expiry,
    }


@router.delete("/documents/{doc_id}", tags=["documents"])
async def delete_document(
    doc_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Delete a document: S3 object, Qdrant vectors, and DB record.
    """
    doc = await get_document_for_user(session, doc_id, auth_user_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # 1. Delete from S3
    if doc.storage_key:
        from storage.s3 import delete_file
        try:
            delete_file(doc.storage_key, bucket=doc.storage_bucket)
        except Exception:
            logger.warning("S3 delete failed for %s — continuing", doc.storage_key)

    # 2. Delete from Qdrant
    try:
        from document_pipeline.vector_store import get_vector_store
        store = get_vector_store()
        await store.delete_document(auth_user_id, doc_id)
    except Exception:
        logger.warning("Qdrant delete failed for doc %s — continuing", doc_id)

    # 3. Delete from DB
    await delete_document_record(session, doc_id, auth_user_id)
    await session.commit()

    return {"deleted": True, "doc_id": doc_id}


@router.get("/documents/status/stream", tags=["documents"])
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


@router.get("/health", tags=["health"])
async def health_check() -> Dict[str, str]:
    return {"status": "ok"}


# ── HITL response endpoint ──────────────────────────────────────────────


@router.post("/hitl/respond", tags=["hitl"])
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


@router.get("/conversations", tags=["conversations"])
async def get_conversations(
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """List conversations for the authenticated user (newest first, paginated)."""
    limit = max(1, min(limit, 100))
    offset = max(0, offset)
    convos, total = await list_user_conversations(session, auth_user_id, limit=limit, offset=offset)
    return {
        "conversations": convos,
        "total": total,
        "limit": limit,
        "offset": offset,
        "has_more": offset + limit < total,
    }


@router.get("/conversations/{conversation_id}/messages", tags=["conversations"])
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


@router.post("/conversations/{conversation_id}/share", tags=["conversations"])
async def share_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Generate (or return existing) share link for a conversation."""
    try:
        token = await create_share_token(session, conversation_id, auth_user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await session.commit()
    return {"share_token": token, "share_url": f"/shared/{token}"}


@router.delete("/conversations/{conversation_id}/share", tags=["conversations"])
async def unshare_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, str]:
    """Revoke the share link for a conversation."""
    try:
        await revoke_share_token(session, conversation_id, auth_user_id)
    except ValueError:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await session.commit()
    return {"status": "unshared"}


@router.delete("/conversations/{conversation_id}", tags=["conversations"])
async def delete_conversation(
    conversation_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Delete a conversation and all its messages, executions, and summaries."""
    from database.helpers import delete_user_conversation

    deleted = await delete_user_conversation(session, conversation_id, auth_user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    await session.commit()
    return {"deleted": True, "conversation_id": conversation_id}


@router.put("/conversations/{conversation_id}/messages/{message_id}/edit", tags=["conversations"])
async def edit_message(
    conversation_id: str,
    message_id: str,
    body: Dict[str, Any],
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Edit a user message and truncate all messages after it."""
    content = body.get("content", "").strip()
    if not content:
        raise HTTPException(status_code=400, detail="Content cannot be empty")

    from database.helpers import edit_message_and_truncate

    try:
        result = await edit_message_and_truncate(
            session, conversation_id, message_id, auth_user_id, content,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    await session.commit()
    return {
        "message_id": message_id,
        "conversation_id": conversation_id,
        "deleted_messages": result["deleted_messages"],
        "deleted_summaries": result["deleted_summaries"],
    }


@router.put("/conversations/{conversation_id}/swap-response", tags=["conversations"])
async def swap_response(
    conversation_id: str,
    request: Request,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Replace the last assistant message content with a new response.
    Used by the Compare feature when the user clicks 'Use This'.
    """
    from database.models import Conversation, Message

    body = await request.json()
    new_content = body.get("new_content", "").strip()
    model = body.get("model", "")
    if not new_content:
        raise HTTPException(status_code=400, detail="new_content is required")

    # Verify ownership
    conv = (await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == auth_user_id,
        )
    )).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    # Find last assistant message
    last_asst = (await session.execute(
        select(Message)
        .where(Message.conversation_id == conv.conversation_id, Message.role == "assistant")
        .order_by(Message.created_at.desc())
        .limit(1)
    )).scalar_one_or_none()
    if last_asst is None:
        raise HTTPException(status_code=404, detail="No assistant message found")

    last_asst.content = new_content
    if model:
        last_asst.model_used = model
    await session.commit()

    return {"message_id": str(last_asst.message_id), "swapped": True}


# ── Persona CRUD ────────────────────────────────────────────────────────


@router.post("/personas", tags=["personas"])
async def create_persona_endpoint(
    body: PersonaCreate,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Create a new persona for the authenticated user."""
    persona = await create_persona(session, auth_user_id, body.name, body.description)
    await session.commit()
    return persona


@router.get("/personas", tags=["personas"])
async def list_personas(
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """List all personas for the authenticated user."""
    await seed_default_personas(session, auth_user_id)
    personas = await get_user_personas(session, auth_user_id)
    await session.commit()
    return {"personas": personas}


@router.put("/personas/{persona_id}", tags=["personas"])
async def update_persona_endpoint(
    persona_id: str,
    body: PersonaUpdate,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Update a persona owned by the authenticated user."""
    updated = await update_persona(
        session, persona_id, auth_user_id,
        name=body.name, description=body.description,
    )
    if updated is None:
        raise HTTPException(status_code=404, detail="Persona not found")
    await session.commit()
    return updated


@router.delete("/personas/{persona_id}", tags=["personas"])
async def delete_persona_endpoint(
    persona_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Delete a persona owned by the authenticated user."""
    ok = await delete_persona(session, persona_id, auth_user_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Persona not found")
    await session.commit()
    return {"deleted": True}


# ── Analytics ────────────────────────────────────────────────────────────


@router.get("/analytics", tags=["analytics"])
async def get_analytics(
    days: int = 30,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Per-user analytics: agent usage, tokens, response times, patterns."""
    from database.helpers import (
        get_agent_usage_stats,
        get_token_usage_over_time,
        get_agent_response_times,
        get_query_patterns,
        get_token_breakdown_by_agent,
        get_web_scrape_stats,
    )

    agent_usage = await get_agent_usage_stats(session, auth_user_id, days)
    token_timeline = await get_token_usage_over_time(session, auth_user_id, days)
    response_times = await get_agent_response_times(session, auth_user_id, days)
    patterns = await get_query_patterns(session, auth_user_id, days)
    token_breakdown = await get_token_breakdown_by_agent(session, auth_user_id, days)
    web_scrape = await get_web_scrape_stats(session, auth_user_id)

    return {
        "agent_usage": agent_usage,
        "token_timeline": token_timeline,
        "response_times": response_times,
        "query_patterns": patterns,
        "token_breakdown": token_breakdown,
        "web_scrape_collections": web_scrape,
    }


# ── Web Scrape Collection CRUD ─────────────────────────────────────────


@router.post("/web-scrape/collections", tags=["web-scrape"])
async def create_web_scrape(
    body: WebScrapeCollectionCreate,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Create a web scrape collection and enqueue the Celery task."""
    from tasks.web_scrape_tasks import scrape_web_collection_task

    await ensure_user_exists(session, auth_user_id)

    urls_with_depth = [{"url": u.url, "depth": u.depth} for u in body.urls]
    collection = await create_web_scrape_collection(
        session, auth_user_id, body.name, urls_with_depth,
    )
    await session.commit()

    # Build task payload
    urls_data = [
        {"url_id": u["url_id"], "url": u["url"], "depth": u["depth"]}
        for u in collection["urls"]
    ]
    scrape_web_collection_task.delay(
        user_id=auth_user_id,
        collection_id=collection["collection_id"],
        urls_data=urls_data,
    )

    return collection


@router.get("/web-scrape/collections", tags=["web-scrape"])
async def list_web_scrape_collections(
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """List all web scrape collections for the authenticated user."""
    collections = await get_user_web_collections(session, auth_user_id)
    return {"collections": collections}


@router.put("/web-scrape/collections/{collection_id}/toggle", tags=["web-scrape"])
async def toggle_web_scrape_collection(
    collection_id: str,
    body: WebScrapeToggle,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Toggle the is_active flag for a web scrape collection."""
    result = await toggle_web_collection_active(
        session, collection_id, auth_user_id, body.is_active,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Collection not found")
    await session.commit()
    return result


@router.delete("/web-scrape/collections/{collection_id}", tags=["web-scrape"])
async def delete_web_scrape_collection(
    collection_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Delete a web scrape collection: Qdrant vectors + DB records."""
    coll = await get_web_collection_for_user(session, collection_id, auth_user_id)
    if coll is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Delete from Qdrant
    try:
        from document_pipeline.vector_store import get_vector_store
        store = get_vector_store()
        await store.delete_web_collection(auth_user_id, collection_id)
    except Exception:
        logger.warning("Qdrant delete failed for web collection %s — continuing", collection_id)

    # Delete from DB (cascades URLs)
    await delete_web_collection_record(session, collection_id, auth_user_id)
    await session.commit()
    return {"deleted": True, "collection_id": collection_id}


@router.post("/web-scrape/collections/{collection_id}/rescrape", tags=["web-scrape"])
async def rescrape_web_collection(
    collection_id: str,
    session: AsyncSession = Depends(db_session),
    auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Re-scrape an existing web scrape collection."""
    from database.helpers import update_web_collection_status, update_web_url_status
    from tasks.web_scrape_tasks import scrape_web_collection_task

    coll = await get_web_collection_for_user(session, collection_id, auth_user_id)
    if coll is None:
        raise HTTPException(status_code=404, detail="Collection not found")

    # Delete old Qdrant data
    try:
        from document_pipeline.vector_store import get_vector_store
        store = get_vector_store()
        await store.delete_web_collection(auth_user_id, collection_id)
    except Exception:
        logger.warning("Qdrant delete failed during rescrape for %s", collection_id)

    # Reset statuses
    await update_web_collection_status(session, collection_id, "pending", total_pages=0, total_chunks=0)
    from sqlalchemy import select
    from database.models import WebScrapeUrl
    url_rows = (await session.execute(
        select(WebScrapeUrl).where(WebScrapeUrl.collection_id == coll.collection_id)
    )).scalars().all()
    urls_data = []
    for u in url_rows:
        await update_web_url_status(session, str(u.url_id), "pending", pages_scraped=0, chunks_created=0)
        urls_data.append({"url_id": str(u.url_id), "url": u.url, "depth": u.depth})
    await session.commit()

    scrape_web_collection_task.delay(
        user_id=auth_user_id,
        collection_id=collection_id,
        urls_data=urls_data,
    )
    return {"collection_id": collection_id, "status": "pending"}


@router.get("/web-scrape/status/stream", tags=["web-scrape"])
async def web_scrape_status_stream(
    request: Request,
    auth_user_id: str = Depends(get_current_user_id),
):
    """SSE endpoint for real-time web scrape status updates."""
    import redis.asyncio as aioredis

    channel_name = f"web_scrape_status:{auth_user_id}"

    async def _event_stream():
        r = aioredis.from_url(config.redis_url, decode_responses=True)
        pubsub = r.pubsub()
        await pubsub.subscribe(channel_name)

        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if msg and msg["type"] == "message":
                    yield f"event: web_scrape_status\ndata: {msg['data']}\n\n"
                else:
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


# ── Voice endpoints ─────────────────────────────────────────────────────


@router.post("/voice/transcribe", tags=["voice"])
async def voice_transcribe(
    file: UploadFile = File(...),
    _auth_user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Transcribe an audio file via the configured STT provider.

    Accepts audio from the browser's MediaRecorder (typically audio/webm).
    Returns the transcribed text.
    """
    max_bytes = config.voice_max_audio_size_mb * 1024 * 1024
    audio_bytes = await file.read()
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Audio file too large (max {config.voice_max_audio_size_mb} MB)",
        )

    content_type = file.content_type or "audio/webm"

    from voice.stt import get_stt_provider

    stt = get_stt_provider()
    result = await stt.transcribe(audio_bytes, content_type=content_type)

    return {
        "text": result.text,
        "confidence": result.confidence,
        "language": result.language,
        "duration_ms": result.duration_ms,
        "provider": result.provider,
    }


@router.post("/voice/synthesize", tags=["voice"])
async def voice_synthesize(
    request: Request,
    _auth_user_id: str = Depends(get_current_user_id),
) -> StreamingResponse:
    """
    Synthesize text to speech via the configured TTS provider.

    Expects JSON body: { "text": "...", "voice": "Matthew" (optional) }
    Returns audio/mpeg stream.
    """
    body = await request.json()
    text = body.get("text", "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="'text' field is required")

    if len(text) > 3000:
        raise HTTPException(status_code=400, detail="Text too long (max 3000 chars)")

    voice = body.get("voice")

    from voice.tts import get_tts_provider

    tts = get_tts_provider()
    result = await tts.synthesize(text, voice=voice)

    return StreamingResponse(
        iter([result.audio_bytes]),
        media_type=result.content_type,
        headers={
            "Content-Disposition": "inline; filename=\"speech.mp3\"",
        },
    )

@router.post("/tts/speak", tags=["voice"])
async def tts_speak(
    request: Request,
    _auth_user_id: str = Depends(get_current_user_id),
) -> StreamingResponse:
    """
    Text-to-speech for assistant responses.

    Accepts raw assistant message text, strips sources/citations/markdown,
    then synthesizes clean text via AWS Polly.

    Expects JSON body: { "text": "...", "voice": "Matthew" (optional) }
    Returns audio/mpeg stream.
    """
    import re

    body = await request.json()
    raw = body.get("text", "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="'text' field is required")

    # 1. Strip trailing "Sources:\n[1] ..." block
    raw = re.sub(r"\n\n?Sources:\n[\s\S]*$", "", raw)
    # 2. Strip inline citation references [1], [2], etc.
    raw = re.sub(r"\[\d+\]", "", raw)
    # 3. Strip markdown bold/italic markers
    raw = re.sub(r"\*{1,3}(.+?)\*{1,3}", r"\1", raw)
    # 4. Strip markdown headers
    raw = re.sub(r"^#{1,6}\s*", "", raw, flags=re.MULTILINE)
    # 5. Strip markdown links [text](url) → text
    raw = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", raw)
    # 6. Strip code fences
    raw = re.sub(r"```[\s\S]*?```", "", raw)
    raw = re.sub(r"`([^`]+)`", r"\1", raw)
    # 7. Collapse whitespace
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    text = raw.strip()

    if not text:
        raise HTTPException(status_code=400, detail="No speakable text after cleaning")

    # Polly has a 3000 char limit for neural engine
    if len(text) > 3000:
        text = text[:3000]

    voice = body.get("voice")

    from voice.tts import get_tts_provider

    tts = get_tts_provider()
    result = await tts.synthesize(text, voice=voice)

    return StreamingResponse(
        iter([result.audio_bytes]),
        media_type=result.content_type,
        headers={
            "Content-Disposition": "inline; filename=\"speech.mp3\"",
        },
    )


@router.get("/demo-credentials")
async def demo_credentials() -> Dict[str, Any]:
    """Expose demo credentials for the login UI when explicitly enabled."""
    if not config.demo_user_enabled or not config.show_demo_credentials_on_login:
        return {"enabled": False}

    return {
        "enabled": True,
        "email": config.demo_user_email,
        "password": config.demo_user_password,
        "display_name": config.demo_user_display_name,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Artifact endpoints
# ═══════════════════════════════════════════════════════════════════════════════

@router.post("/artifacts/save", tags=["artifacts"])
async def save_artifact_endpoint(
    request: Request,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """
    Save an artifact from the frontend (user clicked Save on a preview).
    Accepts base64-encoded file data, uploads to S3, persists DB row.
    """
    import base64
    import uuid as _uuid
    from storage.s3 import upload_file, build_artifact_storage_key, generate_presigned_url
    from database.helpers import save_artifact
    from database.models import Conversation

    body = await request.json()
    conversation_id = body.get("conversation_id")
    filename = body.get("filename", "artifact")
    artifact_type = body.get("artifact_type", "text")
    content_type = body.get("content_type", "application/octet-stream")
    base64_data = body.get("base64_data", "")

    if not conversation_id or not base64_data:
        raise HTTPException(status_code=400, detail="conversation_id and base64_data required")

    # Verify conversation ownership
    from sqlalchemy import select
    conv = (await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == user_id,
        )
    )).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    file_bytes = base64.b64decode(base64_data)
    artifact_id = str(_uuid.uuid4())
    storage_key = build_artifact_storage_key(user_id, conversation_id, artifact_id, filename)
    upload_file(file_bytes, storage_key, content_type=content_type)

    await save_artifact(
        session,
        artifact_id=artifact_id,
        conversation_id=conversation_id,
        agent_id=body.get("agent_id", ""),
        agent_name=body.get("agent_name", "code_agent"),
        filename=filename,
        artifact_type=artifact_type,
        content_type=content_type,
        file_size_bytes=len(file_bytes),
        storage_key=storage_key,
        preview_data=body.get("preview_data", {}),
    )
    await session.commit()

    download_url = generate_presigned_url(storage_key)
    return {
        "artifact_id": artifact_id,
        "download_url": download_url,
    }


@router.get("/artifacts/{artifact_id}/download", tags=["artifacts"])
async def download_artifact(
    artifact_id: str,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """Generate a presigned download URL for an artifact (verifies ownership)."""
    from storage.s3 import generate_presigned_url
    from database.helpers import get_artifact_for_user

    artifact = await get_artifact_for_user(session, artifact_id, user_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail="Artifact not found")

    url = generate_presigned_url(storage_key=artifact.storage_key)
    return {
        "url": url,
        "filename": artifact.filename,
        "content_type": artifact.content_type,
        "file_size_bytes": artifact.file_size_bytes,
        "expires_in": config.s3_presign_expiry,
    }


@router.get("/conversations/{conversation_id}/artifacts", tags=["artifacts"])
async def list_conversation_artifacts(
    conversation_id: str,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, Any]:
    """List all saved artifacts for a conversation (for history reload)."""
    from database.helpers import list_artifacts_for_conversation
    from database.models import Conversation
    from sqlalchemy import select

    conv = (await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == conversation_id,
            Conversation.user_id == user_id,
        )
    )).scalar_one_or_none()
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    artifacts = await list_artifacts_for_conversation(session, conversation_id)
    return {"artifacts": artifacts}
