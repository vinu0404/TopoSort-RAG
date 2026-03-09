"""
REST API routes (non-streaming).
"""

from __future__ import annotations

import asyncio
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
    create_persona,
    delete_document_record,
    delete_persona,
    ensure_user_exists,
    ensure_session_exists,
    get_conversation_persona,
    seed_default_personas,
    get_or_create_conversation,
    get_document_for_user,
    get_user_personas,
    update_persona,
    list_user_conversations,
    load_conversation_messages_full,
    resolve_hitl_request,
    save_document_record,
    get_document_statuses,
)
from tasks.document_tasks import process_document_task
from tools.registry import ToolRegistry
from utils.schemas import HitlUserResponse, Source
from utils.llm_providers import get_llm_provider
from utils.schemas import ComposerInput, PersonaCreate, PersonaContext, PersonaUpdate, QueryRequest

logger = logging.getLogger(__name__)

router = APIRouter()


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
