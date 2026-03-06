"""
Database helper functions — ensure parent records exist and persist data.

"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    AgentExecution,
    Conversation,
    ConversationSummary,
    Document,
    HitlRequest,
    Message,
    Persona,
    Session,
    User,
)
from database.session import async_session_factory

logger = logging.getLogger(__name__)



def _to_uuid(value: str | uuid.UUID) -> uuid.UUID:
    return uuid.UUID(value) if isinstance(value, str) else value



async def ensure_user_exists(session: AsyncSession, user_id: str) -> None:
    """Create a ``User`` row if one does not already exist (idempotent)."""
    uid = _to_uuid(user_id)
    stmt = (
        pg_insert(User)
        .values(
            user_id=uid,
            email=f"{uid}@mrag.local",
            display_name=f"User {str(uid)[:8]}",
        )
        .on_conflict_do_nothing(index_elements=["user_id"])
    )
    await session.execute(stmt)
    await session.flush()


async def ensure_session_exists(
    session: AsyncSession,
    user_id: str,
    session_id: str | None = None,
) -> str:
    """Return an existing or newly-created ``session_id`` for the user."""
    uid = _to_uuid(user_id)

    if session_id:
        sid = _to_uuid(session_id)
        result = await session.execute(
            select(Session).where(Session.session_id == sid)
        )
        if result.scalar_one_or_none() is None:
            session.add(Session(session_id=sid, user_id=uid))
            await session.flush()
        return str(sid)

    result = await session.execute(
        select(Session)
        .where(Session.user_id == uid, Session.is_active.is_(True))
        .order_by(Session.started_at.desc())
        .limit(1)
    )
    existing = result.scalar_one_or_none()
    if existing:
        return str(existing.session_id)

    sid = uuid.uuid4()
    session.add(Session(session_id=sid, user_id=uid))
    await session.flush()
    return str(sid)


async def get_or_create_conversation(
    session: AsyncSession,
    user_id: str,
    session_id: str,
    title: str | None = None,
    conversation_id: str | None = None,
    persona_id: str | None = None,
) -> str:
    """Return an existing or newly-created ``conversation_id``.

    * If *conversation_id* is given, validate it exists and return it.
    * Otherwise create a **new** conversation (supports "New Chat").
    * If *persona_id* is given on a new conversation, attach it.
    """
    uid = _to_uuid(user_id)
    sid = _to_uuid(session_id)

    if conversation_id:
        cid = _to_uuid(conversation_id)
        result = await session.execute(
            select(Conversation).where(
                Conversation.conversation_id == cid,
                Conversation.user_id == uid,
            )
        )
        existing = result.scalar_one_or_none()
        if existing:
            existing.updated_at = datetime.now(timezone.utc)
            await session.flush()
            return str(existing.conversation_id)
    cid = uuid.uuid4()
    pid = _to_uuid(persona_id) if persona_id else None
    session.add(
        Conversation(
            conversation_id=cid,
            session_id=sid,
            user_id=uid,
            persona_id=pid,
            title=title or "New conversation",
        )
    )
    await session.flush()
    return str(cid)


async def close_user_sessions(
    session: AsyncSession,
    user_id: str,
) -> int:
    """Mark all active sessions for *user_id* as inactive.

    Returns the number of sessions closed.
    """
    from sqlalchemy import update

    uid = _to_uuid(user_id)
    now = datetime.now(timezone.utc)
    result = await session.execute(
        update(Session)
        .where(Session.user_id == uid, Session.is_active.is_(True))
        .values(is_active=False, ended_at=now)
    )
    await session.flush()
    return result.rowcount  


async def list_user_conversations(
    session: AsyncSession,
    user_id: str,
    limit: int = 50,
) -> list[dict]:
    """Return recent conversations for a user (newest first).

    Each dict contains ``conversation_id``, ``title``, ``created_at``,
    ``updated_at``, and ``message_count``.
    """
    from sqlalchemy import func

    uid = _to_uuid(user_id)
    msg_count = (
        select(func.count(Message.message_id))
        .where(Message.conversation_id == Conversation.conversation_id)
        .correlate(Conversation)
        .scalar_subquery()
    )
    stmt = (
        select(
            Conversation.conversation_id,
            Conversation.title,
            Conversation.persona_id,
            Conversation.created_at,
            Conversation.updated_at,
            msg_count.label("message_count"),
        )
        .where(Conversation.user_id == uid)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
    )
    rows = await session.execute(stmt)
    return [
        {
            "conversation_id": str(r.conversation_id),
            "title": r.title,
            "persona_id": str(r.persona_id) if r.persona_id else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "message_count": r.message_count or 0,
        }
        for r in rows
    ]


async def load_conversation_messages_full(
    session: AsyncSession,
    conversation_id: str,
    limit: int = 100,
) -> list[dict]:
    """Load messages for a conversation (for frontend display), oldest first.

    Unlike ``load_conversation_messages`` (used internally for LLM context),
    this returns full metadata including ``message_id`` and ``created_at``.
    """
    from sqlalchemy import case

    cid = _to_uuid(conversation_id)
    role_order = case(
        (Message.role == "user", 0),
        (Message.role == "system", 1),
        else_=2,
    )
    stmt = (
        select(Message)
        .where(Message.conversation_id == cid)
        .order_by(Message.created_at.asc(), role_order)
        .limit(limit)
    )
    rows = await session.execute(stmt)
    return [
        {
            "message_id": str(m.message_id),
            "role": m.role,
            "content": m.content,
            "created_at": m.created_at.isoformat() if m.created_at else None,
        }
        for m in rows.scalars()
    ]


async def save_messages(
    session: AsyncSession,
    conversation_id: str,
    user_query: str,
    assistant_answer: str,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Insert a *user* message and an *assistant* message.

    The assistant row gets ``created_at = now + 1 µs`` so that a simple
    ``ORDER BY created_at`` always returns user-before-assistant — even
    though both belong to the same logical turn.
    """
    cid = _to_uuid(conversation_id)
    now = datetime.now(timezone.utc)

    session.add(
        Message(conversation_id=cid, role="user", content=user_query, created_at=now)
    )
    session.add(
        Message(
            conversation_id=cid,
            role="assistant",
            content=assistant_answer,
            metadata_=metadata or {},
            created_at=now + timedelta(microseconds=1),
        )
    )
    await session.flush()


# ── Background (fire-and-forget) wrappers ───────────────────────────


async def bg_save_messages(
    conversation_id: str,
    user_query: str,
    assistant_answer: str,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget: persist messages with an independent DB session."""
    try:
        async with async_session_factory() as session:
            await save_messages(session, conversation_id, user_query, assistant_answer, metadata)
            await session.commit()
    except Exception:
        logger.exception(
            "Background save_messages failed for conversation %s", conversation_id,
        )


async def bg_save_agent_executions(
    conversation_id: str,
    agent_results: Dict[str, Any],
) -> None:
    """Fire-and-forget: persist agent executions with an independent DB session."""
    try:
        async with async_session_factory() as session:
            await save_agent_executions(session, conversation_id, agent_results)
            await session.commit()
    except Exception:
        logger.exception(
            "Background save_agent_executions failed for conversation %s", conversation_id,
        )


async def save_agent_executions(
    session: AsyncSession,
    conversation_id: str,
    agent_results: Dict[str, Any],
) -> None:
    """Persist one ``AgentExecution`` row per agent that ran."""
    cid = _to_uuid(conversation_id)
    now = datetime.now(timezone.utc)

    for agent_name, output in agent_results.items():
        status = "success" if getattr(output, "task_done", False) else "failed"
        payload = (
            output.model_dump() if hasattr(output, "model_dump") else {"raw": str(output)}
        )
        session.add(
            AgentExecution(
                conversation_id=cid,
                agent_name=agent_name,
                task_description=getattr(output, "task_description", None),
                status=status,
                output_payload=payload,
                started_at=now,
                completed_at=now,
            )
        )
    await session.flush()


async def save_document_record(
    session: AsyncSession,
    user_id: str,
    doc_id: str,
    filename: str,
    doc_type: str | None = None,
    description: str | None = None,
    total_chunks: int | None = None,
    qdrant_collection: str | None = None,
    processing_status: str = "pending",
    storage_key: str | None = None,
    storage_bucket: str | None = None,
    file_size_bytes: int | None = None,
    content_type: str | None = None,
) -> None:
    """Persist document metadata to the ``documents`` table."""
    uid = _to_uuid(user_id)
    did = _to_uuid(doc_id)

    session.add(
        Document(
            doc_id=did,
            user_id=uid,
            filename=filename,
            doc_type=doc_type,
            description=description,
            total_chunks=total_chunks,
            qdrant_collection=qdrant_collection,
            processing_status=processing_status,
            storage_key=storage_key,
            storage_bucket=storage_bucket,
            file_size_bytes=file_size_bytes,
            content_type=content_type,
        )
    )
    await session.flush()


async def get_document_for_user(
    session: AsyncSession,
    doc_id: str,
    user_id: str,
) -> Document | None:
    """
    Fetch a document only if it belongs to the given user.

    Returns ``None`` when the doc doesn't exist or belongs to
    another user — callers should treat that as 403/404.
    """
    did = _to_uuid(doc_id)
    uid = _to_uuid(user_id)
    result = await session.execute(
        select(Document).where(Document.doc_id == did, Document.user_id == uid)
    )
    return result.scalar_one_or_none()


async def update_document_status(
    session: AsyncSession,
    doc_id: str,
    status: str,
    description: str | None = None,
    total_chunks: int | None = None,
    error_message: str | None = None,
) -> None:
    """Update document processing status in the DB."""
    from sqlalchemy import update

    did = _to_uuid(doc_id)
    values: dict = {"processing_status": status}
    if description is not None:
        values["description"] = description
    if total_chunks is not None:
        values["total_chunks"] = total_chunks
    if error_message is not None:
        values["error_message"] = error_message

    await session.execute(
        update(Document).where(Document.doc_id == did).values(**values)
    )
    await session.flush()


async def get_document_statuses(
    session: AsyncSession,
    user_id: str,
    doc_ids: list[str] | None = None,
) -> list[dict]:
    """
    Query document processing statuses from the DB.
    If doc_ids is provided, filter to those; otherwise return all for the user.
    """
    from sqlalchemy import select

    uid = _to_uuid(user_id)
    stmt = select(Document).where(Document.user_id == uid)
    if doc_ids:
        uuid_ids = [_to_uuid(d) for d in doc_ids]
        stmt = stmt.where(Document.doc_id.in_(uuid_ids))
    stmt = stmt.order_by(Document.uploaded_at.desc())

    result = await session.execute(stmt)
    rows = result.scalars().all()
    return [
        {
            "doc_id": str(r.doc_id),
            "filename": r.filename,
            "processing_status": r.processing_status,
            "description": r.description,
            "total_chunks": r.total_chunks,
            "error_message": r.error_message,
        }
        for r in rows
    ]


async def load_conversation_messages(
    session: AsyncSession,
    conversation_id: str,
    limit: int = 50,
) -> list[dict[str, str]]:
    """
    Load the most recent messages for a conversation from PostgreSQL.
    """
    from sqlalchemy import case

    cid = _to_uuid(conversation_id)
    role_order = case(
        (Message.role == "user", 0),
        (Message.role == "system", 1),
        else_=2,
    )
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == cid)
        .order_by(Message.created_at.asc(), role_order)
        .limit(limit)
    )
    rows = result.scalars().all()
    return [
        {"role": row.role, "content": row.content}
        for row in rows
    ]


async def load_conversation_summaries(
    session: AsyncSession,
    conversation_id: str,
) -> list[dict[str, Any]]:
    """
    Load persisted conversation summaries for a conversation.

    Returns dicts with 'summary_text' and 'turns_covered', ordered by
    creation time.
    """
    cid = _to_uuid(conversation_id)
    result = await session.execute(
        select(ConversationSummary)
        .where(ConversationSummary.conversation_id == cid)
        .order_by(ConversationSummary.created_at.asc())
    )
    rows = result.scalars().all()
    return [
        {
            "summary_text": row.summary_text,
            "turns_covered": row.turns_covered,
        }
        for row in rows
    ]


async def save_conversation_summary(
    session: AsyncSession,
    conversation_id: str,
    summary_text: str,
    turns_covered: int,
) -> None:
    """Persist a conversation summary row."""
    cid = _to_uuid(conversation_id)
    session.add(
        ConversationSummary(
            conversation_id=cid,
            summary_text=summary_text,
            turns_covered=turns_covered,
        )
    )
    await session.flush()


# ── HITL (Human-in-the-Loop) helpers ────────────────────────────────


async def create_hitl_request(
    session: AsyncSession,
    conversation_id: str,
    agent_id: str,
    agent_name: str,
    tool_names: List[str],
    task_description: str,
    timeout_seconds: int = 120,
) -> str:
    """
    Insert a pending HITL approval request.

    Returns the ``request_id`` (UUID string) so the caller can poll for a
    decision later.
    """
    cid = _to_uuid(conversation_id)
    now = datetime.now(timezone.utc)
    row = HitlRequest(
        conversation_id=cid,
        agent_id=agent_id,
        agent_name=agent_name,
        tool_names=tool_names,
        task_description=task_description,
        status="pending",
        created_at=now,
        expires_at=now + timedelta(seconds=timeout_seconds),
    )
    session.add(row)
    await session.commit()
    logger.info(
        "HITL request created: request_id=%s  agent=%s  tools=%s",
        row.request_id, agent_name, tool_names,
    )
    return str(row.request_id)


async def poll_hitl_decision(
    request_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Check the DB for a resolved HITL request (approved / denied / timed_out).

    Uses an **independent session** so the caller can poll in a loop
    without holding the request's main session open.

    Returns
    -------
    None           – still pending
    dict           – keys: status, user_instructions
    """
    rid = _to_uuid(request_id)
    async with async_session_factory() as session:
        result = await session.execute(
            select(HitlRequest).where(HitlRequest.request_id == rid)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return {"status": "expired", "user_instructions": None}
        if row.status == "pending":
            # Check if it expired while pending
            if datetime.now(timezone.utc) >= row.expires_at:
                row.status = "timed_out"
                row.responded_at = datetime.now(timezone.utc)
                await session.commit()
                return {"status": "timed_out", "user_instructions": None}
            return None  # still waiting
        return {
            "status": row.status,
            "user_instructions": row.user_instructions,
        }


async def resolve_hitl_request(
    request_id: str,
    decision: str,
    instructions: Optional[str] = None,
) -> str:
    """
    Update a pending HITL request with the user's decision.

    Returns the final status.  If the request was already resolved
    (timed_out, expired, etc.) the update is skipped and the current
    status is returned.
    """
    rid = _to_uuid(request_id)
    async with async_session_factory() as session:
        result = await session.execute(
            select(HitlRequest).where(HitlRequest.request_id == rid)
        )
        row = result.scalar_one_or_none()
        if row is None:
            return "not_found"
        if row.status != "pending":
            return row.status  # already resolved
        row.status = decision
        row.user_instructions = instructions
        row.responded_at = datetime.now(timezone.utc)
        await session.commit()
        logger.info("HITL resolved: request_id=%s  decision=%s", request_id, decision)
        return decision


async def expire_stale_hitl_requests() -> int:
    """
    Mark all pending HITL requests whose ``expires_at`` has passed as
    ``expired``.  Called on application startup to clean up orphans.

    Returns the number of rows affected.
    """
    async with async_session_factory() as session:
        result = await session.execute(
            select(HitlRequest).where(
                HitlRequest.status == "pending",
                HitlRequest.expires_at < datetime.now(timezone.utc),
            )
        )
        rows = result.scalars().all()
        for row in rows:
            row.status = "expired"
            row.responded_at = datetime.now(timezone.utc)
        await session.commit()
        if rows:
            logger.info("Expired %d stale HITL requests on startup", len(rows))
        return len(rows)


# ── Persona helpers ─────────────────────────────────────────────────


async def create_persona(
    session: AsyncSession,
    user_id: str,
    name: str,
    description: str,
) -> dict:
    """Create a new persona for a user. Returns the serialised row."""
    uid = _to_uuid(user_id)
    row = Persona(user_id=uid, name=name, description=description)
    session.add(row)
    await session.flush()
    return _persona_to_dict(row)


async def get_user_personas(
    session: AsyncSession,
    user_id: str,
) -> list[dict]:
    """Return all personas belonging to a user (newest first)."""
    uid = _to_uuid(user_id)
    result = await session.execute(
        select(Persona)
        .where(Persona.user_id == uid)
        .order_by(Persona.created_at.desc())
    )
    return [_persona_to_dict(r) for r in result.scalars()]


async def get_persona(
    session: AsyncSession,
    persona_id: str,
    user_id: str,
) -> Persona | None:
    """Fetch a single persona — only if owned by *user_id*."""
    pid = _to_uuid(persona_id)
    uid = _to_uuid(user_id)
    result = await session.execute(
        select(Persona).where(Persona.persona_id == pid, Persona.user_id == uid)
    )
    return result.scalar_one_or_none()


async def update_persona(
    session: AsyncSession,
    persona_id: str,
    user_id: str,
    name: str | None = None,
    description: str | None = None,
) -> dict | None:
    """Update name/description of a persona. Returns updated dict or None."""
    row = await get_persona(session, persona_id, user_id)
    if row is None:
        return None
    if name is not None:
        row.name = name
    if description is not None:
        row.description = description
    row.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return _persona_to_dict(row)


async def delete_persona(
    session: AsyncSession,
    persona_id: str,
    user_id: str,
) -> bool:
    """Delete a persona owned by *user_id*. Returns True if deleted."""
    row = await get_persona(session, persona_id, user_id)
    if row is None:
        return False
    await session.delete(row)
    await session.flush()
    return True


async def get_conversation_persona(
    session: AsyncSession,
    conversation_id: str,
) -> dict | None:
    """Load the persona attached to a conversation (if any).

    Returns ``{"name": ..., "description": ...}`` or None.
    """
    cid = _to_uuid(conversation_id)
    result = await session.execute(
        select(Conversation).where(Conversation.conversation_id == cid)
    )
    conv = result.scalar_one_or_none()
    if conv is None or conv.persona_id is None:
        return None
    result2 = await session.execute(
        select(Persona).where(Persona.persona_id == conv.persona_id)
    )
    persona = result2.scalar_one_or_none()
    if persona is None:
        return None
    return {"name": persona.name, "description": persona.description}


def _persona_to_dict(row: Persona) -> dict:
    return {
        "persona_id": str(row.persona_id),
        "name": row.name,
        "description": row.description,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }