"""
Database helper functions — ensure parent records exist and persist data.

"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import delete as sa_delete, func, select
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
    WebScrapeCollection,
    WebScrapeUrl,
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

async def ensure_demo_user(session: AsyncSession) -> dict | None:
    """Ensure the configured demo user exists (idempotent)."""
    from auth.password import hash_password
    from config.settings import config

    if not config.demo_user_enabled:
        return None

    email = (config.demo_user_email or "").strip().lower()
    password = config.demo_user_password or ""
    display_name = (config.demo_user_display_name or "Demo User").strip()

    if not email or not password:
        logger.warning(
            "DEMO_USER_ENABLED=true but DEMO_USER_EMAIL or DEMO_USER_PASSWORD is missing; skipping demo user seed."
        )
        return None

    existing = (
        await session.execute(
            select(User).where(User.email == email).limit(1)
        )
    ).scalar_one_or_none()

    if existing is None:
        existing = User(
            user_id=uuid.uuid4(),
            email=email,
            display_name=display_name,
            password_hash=hash_password(password),
        )
        session.add(existing)
        await session.flush()
        logger.info("Seeded demo user: %s", email)
    elif display_name and existing.display_name != display_name:
        existing.display_name = display_name
        await session.flush()

    await seed_default_personas(session, str(existing.user_id))

    return {
        "user_id": str(existing.user_id),
        "email": existing.email,
        "display_name": existing.display_name,
    }

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
            # Update persona if caller sent a (different) persona_id
            new_pid = _to_uuid(persona_id) if persona_id else None
            if existing.persona_id != new_pid:
                existing.persona_id = new_pid
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


async def update_conversation_title(
    session: AsyncSession,
    conversation_id: str,
    title: str,
) -> None:
    """Update the title of an existing conversation."""
    cid = _to_uuid(conversation_id)
    result = await session.execute(
        select(Conversation).where(Conversation.conversation_id == cid)
    )
    conv = result.scalar_one_or_none()
    if conv:
        conv.title = title
        await session.flush()


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
    limit: int = 20,
    offset: int = 0,
) -> tuple[list[dict], int]:
    """Return recent conversations for a user (newest first).

    Returns ``(rows, total_count)`` to support pagination.
    Each dict contains ``conversation_id``, ``title``, ``created_at``,
    ``updated_at``, and ``message_count``.
    """
    from sqlalchemy import func

    uid = _to_uuid(user_id)

    # Total count
    count_stmt = select(func.count(Conversation.conversation_id)).where(
        Conversation.user_id == uid
    )
    total = (await session.execute(count_stmt)).scalar() or 0

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
            Conversation.share_token,
            Conversation.created_at,
            Conversation.updated_at,
            msg_count.label("message_count"),
        )
        .where(Conversation.user_id == uid)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    rows = await session.execute(stmt)
    items = [
        {
            "conversation_id": str(r.conversation_id),
            "title": r.title,
            "persona_id": str(r.persona_id) if r.persona_id else None,
            "share_token": str(r.share_token) if r.share_token else None,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
            "message_count": r.message_count or 0,
        }
        for r in rows
    ]
    return items, total


async def delete_user_conversation(
    session: AsyncSession, conversation_id: str, user_id: str,
) -> bool:
    """Delete a conversation owned by user_id. Returns True if deleted."""
    uid = _to_uuid(user_id)
    cid = _to_uuid(conversation_id)
    result = await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == cid,
            Conversation.user_id == uid,
        )
    )
    conv = result.scalar_one_or_none()
    if not conv:
        return False
    await session.delete(conv)
    await session.flush()
    return True


async def edit_message_and_truncate(
    session: AsyncSession,
    conversation_id: str,
    message_id: str,
    user_id: str,
    new_content: str,
) -> dict:
    """Edit a user message and delete everything after it.

    1. Verify ownership and that the message is a user message.
    2. Delete all messages created after the edited message.
    3. Delete agent_executions started after the edited message.
    4. Delete HITL requests created after the edited message.
    5. Delete conversation summaries that cover turns beyond the edit point.
    6. Update the message content.

    Returns ``{"deleted_messages": int, "deleted_summaries": int}``
    or raises ``ValueError`` if not found / not a user message.
    """
    uid = _to_uuid(user_id)
    cid = _to_uuid(conversation_id)
    mid = _to_uuid(message_id)

    # ── 1. Verify conversation ownership ──
    conv = (await session.execute(
        select(Conversation).where(
            Conversation.conversation_id == cid,
            Conversation.user_id == uid,
        )
    )).scalar_one_or_none()
    if not conv:
        raise ValueError("Conversation not found")

    # ── 2. Verify message exists and is a user message ──
    msg = (await session.execute(
        select(Message).where(
            Message.message_id == mid,
            Message.conversation_id == cid,
        )
    )).scalar_one_or_none()
    if not msg:
        raise ValueError("Message not found")
    if msg.role != "user":
        raise ValueError("Only user messages can be edited")

    edit_timestamp = msg.created_at

    # ── 3. Count the turn number of this message ──
    # A "turn" = one user message. Count user messages up to and including this one.
    turn_number = (await session.execute(
        select(func.count(Message.message_id)).where(
            Message.conversation_id == cid,
            Message.role == "user",
            Message.created_at <= edit_timestamp,
        )
    )).scalar() or 0

    # ── 4. Update the message content BEFORE bulk deletes ──
    # (bulk sa_delete can expire ORM objects in the same session)
    msg.content = new_content
    await session.flush()

    # ── 5. Delete all messages AFTER the edited message ──
    del_msgs = await session.execute(
        sa_delete(Message).where(
            Message.conversation_id == cid,
            Message.created_at > edit_timestamp,
        )
    )
    deleted_messages = del_msgs.rowcount

    # ── 6. Delete agent executions after the edit point ──
    await session.execute(
        sa_delete(AgentExecution).where(
            AgentExecution.conversation_id == cid,
            AgentExecution.started_at > edit_timestamp,
        )
    )

    # ── 7. Delete HITL requests after the edit point ──
    await session.execute(
        sa_delete(HitlRequest).where(
            HitlRequest.conversation_id == cid,
            HitlRequest.created_at > edit_timestamp,
        )
    )

    # ── 8. Trim conversation summaries ──
    # Each summary covers N turns. Walk through summaries in order and
    # delete any whose range extends into or beyond `turn_number`.
    # (turn_number is the turn being re-run, so keep only turns < turn_number)
    summaries = (await session.execute(
        select(ConversationSummary)
        .where(ConversationSummary.conversation_id == cid)
        .order_by(ConversationSummary.created_at.asc())
    )).scalars().all()

    deleted_summaries = 0
    cumulative_turns = 0
    for s in summaries:
        cumulative_turns += s.turns_covered
        # This summary's range ends at cumulative_turns.
        # If it extends to turn_number or beyond, delete it.
        if cumulative_turns >= turn_number:
            await session.delete(s)
            deleted_summaries += 1

    await session.flush()

    return {
        "deleted_messages": deleted_messages,
        "deleted_summaries": deleted_summaries,
    }


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
            "model_used": m.model_used,
            "total_tokens": m.total_tokens or 0,
            "token_details": m.token_details if m.token_details else {},
            "metadata": m.metadata_ if m.metadata_ else {},
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
    model_used: str | None = None,
    total_tokens: int = 0,
    token_details: Dict[str, Any] | None = None,
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
            model_used=model_used,
            total_tokens=total_tokens,
            token_details=token_details or {},
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
    model_used: str | None = None,
    total_tokens: int = 0,
    token_details: Dict[str, Any] | None = None,
) -> None:
    """Fire-and-forget: persist messages with an independent DB session."""
    try:
        async with async_session_factory() as session:
            await save_messages(
                session, conversation_id, user_query, assistant_answer,
                metadata, model_used=model_used,
                total_tokens=total_tokens, token_details=token_details,
            )
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
            "file_size_bytes": r.file_size_bytes,
            "content_type": r.content_type,
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
        }
        for r in rows
    ]


async def delete_document_record(
    session: AsyncSession,
    doc_id: str,
    user_id: str,
) -> bool:
    """Delete a document record. Returns True if a row was deleted."""
    from sqlalchemy import delete as sa_delete

    did = _to_uuid(doc_id)
    uid = _to_uuid(user_id)
    result = await session.execute(
        sa_delete(Document).where(Document.doc_id == did, Document.user_id == uid)
    )
    await session.flush()
    return result.rowcount > 0


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


async def seed_default_personas(session: AsyncSession, user_id: str) -> None:
    """Create the default personas for a user if they have none yet."""
    from config.settings import DEFAULT_PERSONAS

    uid = _to_uuid(user_id)
    result = await session.execute(
        select(Persona).where(Persona.user_id == uid).limit(1)
    )
    if result.scalar_one_or_none() is not None:
        return  # user already has personas

    for p in DEFAULT_PERSONAS:
        session.add(Persona(user_id=uid, name=p["name"], description=p["description"]))
    await session.flush()


# ── Shared conversations ────────────────────────────────────────────────


async def create_share_token(
    session: AsyncSession,
    conversation_id: str,
    user_id: str,
) -> str:
    """Generate a share token for a conversation owned by user_id.

    Returns the token string.  If the conversation already has a token,
    returns the existing one (idempotent).
    """
    cid = _to_uuid(conversation_id)
    uid = _to_uuid(user_id)

    conv = (
        await session.execute(
            select(Conversation).where(
                Conversation.conversation_id == cid,
                Conversation.user_id == uid,
            )
        )
    ).scalar_one_or_none()
    if conv is None:
        raise ValueError("Conversation not found or not owned by user")

    if conv.share_token:
        return str(conv.share_token)

    token = uuid.uuid4()
    conv.share_token = token
    await session.flush()
    return str(token)


async def revoke_share_token(
    session: AsyncSession,
    conversation_id: str,
    user_id: str,
) -> None:
    """Remove the share token (revoke the public link)."""
    cid = _to_uuid(conversation_id)
    uid = _to_uuid(user_id)

    conv = (
        await session.execute(
            select(Conversation).where(
                Conversation.conversation_id == cid,
                Conversation.user_id == uid,
            )
        )
    ).scalar_one_or_none()
    if conv is None:
        raise ValueError("Conversation not found or not owned by user")

    conv.share_token = None
    await session.flush()


async def load_shared_conversation(
    session: AsyncSession,
    share_token: str,
) -> dict | None:
    """Load a conversation + messages by share token (no auth required).

    Returns None if the token doesn't match any conversation.
    """
    token = _to_uuid(share_token)

    conv = (
        await session.execute(
            select(Conversation).where(Conversation.share_token == token)
        )
    ).scalar_one_or_none()
    if conv is None:
        return None

    messages = await load_conversation_messages_full(session, str(conv.conversation_id))
    return {
        "conversation_id": str(conv.conversation_id),
        "title": conv.title,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "messages": messages,
    }


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


# ── Web Scrape Collection helpers ──────────────────────────────────


def _web_collection_to_dict(row: WebScrapeCollection, urls: list | None = None) -> dict:
    d = {
        "collection_id": str(row.collection_id),
        "name": row.name,
        "is_active": row.is_active,
        "status": row.status,
        "total_pages": row.total_pages or 0,
        "total_chunks": row.total_chunks or 0,
        "error_message": row.error_message,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    if urls is not None:
        d["urls"] = urls
    return d


def _web_url_to_dict(row: WebScrapeUrl) -> dict:
    return {
        "url_id": str(row.url_id),
        "url": row.url,
        "depth": row.depth,
        "status": row.status,
        "pages_scraped": row.pages_scraped or 0,
        "chunks_created": row.chunks_created or 0,
        "error_message": row.error_message,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


async def create_web_scrape_collection(
    session: AsyncSession,
    user_id: str,
    name: str,
    urls_with_depth: List[Dict[str, Any]],
) -> dict:
    """Create a web scrape collection with its URL entries.

    ``urls_with_depth`` is a list of ``{"url": str, "depth": int}``.
    Returns the serialised collection dict including URL rows.
    """
    uid = _to_uuid(user_id)
    collection = WebScrapeCollection(user_id=uid, name=name, status="pending")
    session.add(collection)
    await session.flush()  # populate collection_id

    url_rows: list[dict] = []
    for entry in urls_with_depth:
        url_row = WebScrapeUrl(
            collection_id=collection.collection_id,
            url=entry["url"],
            depth=entry.get("depth", 1),
        )
        session.add(url_row)
        await session.flush()
        url_rows.append(_web_url_to_dict(url_row))

    return _web_collection_to_dict(collection, urls=url_rows)


async def get_user_web_collections(
    session: AsyncSession,
    user_id: str,
) -> list[dict]:
    """List all web scrape collections for a user with their URLs."""
    uid = _to_uuid(user_id)
    result = await session.execute(
        select(WebScrapeCollection)
        .where(WebScrapeCollection.user_id == uid)
        .order_by(WebScrapeCollection.created_at.desc())
    )
    collections = result.scalars().all()

    items = []
    for c in collections:
        url_result = await session.execute(
            select(WebScrapeUrl)
            .where(WebScrapeUrl.collection_id == c.collection_id)
            .order_by(WebScrapeUrl.created_at.asc())
        )
        urls = [_web_url_to_dict(u) for u in url_result.scalars()]
        items.append(_web_collection_to_dict(c, urls=urls))
    return items


async def get_web_collection_for_user(
    session: AsyncSession,
    collection_id: str,
    user_id: str,
) -> WebScrapeCollection | None:
    """Ownership-checked fetch of a web scrape collection."""
    cid = _to_uuid(collection_id)
    uid = _to_uuid(user_id)
    result = await session.execute(
        select(WebScrapeCollection).where(
            WebScrapeCollection.collection_id == cid,
            WebScrapeCollection.user_id == uid,
        )
    )
    return result.scalar_one_or_none()


async def update_web_collection_status(
    session: AsyncSession,
    collection_id: str,
    status: str,
    total_pages: int | None = None,
    total_chunks: int | None = None,
    error_message: str | None = None,
) -> None:
    """Update web scrape collection status and aggregates."""
    from sqlalchemy import update

    cid = _to_uuid(collection_id)
    values: dict = {"status": status, "updated_at": datetime.now(timezone.utc)}
    if total_pages is not None:
        values["total_pages"] = total_pages
    if total_chunks is not None:
        values["total_chunks"] = total_chunks
    if error_message is not None:
        values["error_message"] = error_message
    await session.execute(
        update(WebScrapeCollection)
        .where(WebScrapeCollection.collection_id == cid)
        .values(**values)
    )
    await session.flush()


async def update_web_url_status(
    session: AsyncSession,
    url_id: str,
    status: str,
    pages_scraped: int | None = None,
    chunks_created: int | None = None,
    error_message: str | None = None,
) -> None:
    """Update individual web scrape URL status."""
    from sqlalchemy import update

    uid = _to_uuid(url_id)
    values: dict = {"status": status}
    if pages_scraped is not None:
        values["pages_scraped"] = pages_scraped
    if chunks_created is not None:
        values["chunks_created"] = chunks_created
    if error_message is not None:
        values["error_message"] = error_message
    await session.execute(
        update(WebScrapeUrl).where(WebScrapeUrl.url_id == uid).values(**values)
    )
    await session.flush()


async def toggle_web_collection_active(
    session: AsyncSession,
    collection_id: str,
    user_id: str,
    is_active: bool,
) -> dict | None:
    """Toggle is_active flag for a web scrape collection. Returns updated dict or None."""
    row = await get_web_collection_for_user(session, collection_id, user_id)
    if row is None:
        return None
    row.is_active = is_active
    row.updated_at = datetime.now(timezone.utc)
    await session.flush()
    return _web_collection_to_dict(row)


async def delete_web_collection_record(
    session: AsyncSession,
    collection_id: str,
    user_id: str,
) -> bool:
    """Delete a web scrape collection (cascades to URLs). Returns True if deleted."""
    row = await get_web_collection_for_user(session, collection_id, user_id)
    if row is None:
        return False
    await session.delete(row)
    await session.flush()
    return True


async def get_active_web_collection_ids(
    session: AsyncSession,
    user_id: str,
) -> list[str]:
    """Return collection_ids of all active (toggled ON) web scrape collections."""
    uid = _to_uuid(user_id)
    result = await session.execute(
        select(WebScrapeCollection.collection_id).where(
            WebScrapeCollection.user_id == uid,
            WebScrapeCollection.is_active.is_(True),
            WebScrapeCollection.status.in_(["ready", "partial"]),
        )
    )
    return [str(r[0]) for r in result]


# ═══════════════════════════════════════════════════════════════
# Analytics helpers
# ═══════════════════════════════════════════════════════════════

async def get_agent_usage_stats(
    session: AsyncSession, user_id: str, days: int = 30,
) -> list[dict]:
    """Agent execution counts grouped by agent_name, ordered by frequency."""
    from sqlalchemy import func

    uid = _to_uuid(user_id)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    stmt = (
        select(
            AgentExecution.agent_name,
            func.count(AgentExecution.execution_id).label("count"),
            func.count(AgentExecution.execution_id).filter(
                AgentExecution.status == "success"
            ).label("success_count"),
            func.count(AgentExecution.execution_id).filter(
                AgentExecution.status == "failed"
            ).label("failed_count"),
        )
        .join(Conversation, Conversation.conversation_id == AgentExecution.conversation_id)
        .where(Conversation.user_id == uid, AgentExecution.started_at >= cutoff)
        .group_by(AgentExecution.agent_name)
        .order_by(func.count(AgentExecution.execution_id).desc())
    )
    rows = await session.execute(stmt)
    return [
        {"agent_name": r.agent_name, "count": r.count,
         "success_count": r.success_count, "failed_count": r.failed_count}
        for r in rows
    ]


async def get_token_usage_over_time(
    session: AsyncSession, user_id: str, days: int = 30,
) -> list[dict]:
    """Total tokens aggregated by day."""
    from sqlalchemy import func

    uid = _to_uuid(user_id)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    date_trunc = func.date_trunc("day", Message.created_at)

    stmt = (
        select(
            date_trunc.label("period"),
            func.sum(Message.total_tokens).label("total_tokens"),
            func.count(Message.message_id).label("message_count"),
        )
        .join(Conversation, Conversation.conversation_id == Message.conversation_id)
        .where(
            Conversation.user_id == uid,
            Message.role == "assistant",
            Message.created_at >= cutoff,
        )
        .group_by("period")
        .order_by("period")
    )
    rows = await session.execute(stmt)
    return [
        {"period": r.period.isoformat() if r.period else None,
         "total_tokens": r.total_tokens or 0,
         "message_count": r.message_count or 0}
        for r in rows
    ]


async def get_agent_response_times(
    session: AsyncSession, user_id: str, days: int = 30,
) -> list[dict]:
    """Average response time (seconds) per agent."""
    from sqlalchemy import func, extract

    uid = _to_uuid(user_id)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    duration_expr = extract("epoch", AgentExecution.completed_at - AgentExecution.started_at)

    stmt = (
        select(
            AgentExecution.agent_name,
            func.avg(duration_expr).label("avg_seconds"),
            func.min(duration_expr).label("min_seconds"),
            func.max(duration_expr).label("max_seconds"),
            func.count(AgentExecution.execution_id).label("count"),
        )
        .join(Conversation, Conversation.conversation_id == AgentExecution.conversation_id)
        .where(
            Conversation.user_id == uid,
            AgentExecution.started_at >= cutoff,
            AgentExecution.completed_at.isnot(None),
        )
        .group_by(AgentExecution.agent_name)
        .order_by(func.avg(duration_expr).asc())
    )
    rows = await session.execute(stmt)
    return [
        {"agent_name": r.agent_name,
         "avg_seconds": round(float(r.avg_seconds), 2) if r.avg_seconds else 0,
         "min_seconds": round(float(r.min_seconds), 2) if r.min_seconds else 0,
         "max_seconds": round(float(r.max_seconds), 2) if r.max_seconds else 0,
         "count": r.count}
        for r in rows
    ]


async def get_query_patterns(
    session: AsyncSession, user_id: str, days: int = 30,
) -> dict:
    """Overall query statistics: totals, success rate, conversation/message counts."""
    from sqlalchemy import func

    uid = _to_uuid(user_id)
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    # Agent execution stats
    exec_stmt = (
        select(
            func.count(AgentExecution.execution_id).label("total"),
            func.count(AgentExecution.execution_id).filter(
                AgentExecution.status == "success"
            ).label("success"),
            func.count(AgentExecution.execution_id).filter(
                AgentExecution.status == "failed"
            ).label("failed"),
        )
        .join(Conversation, Conversation.conversation_id == AgentExecution.conversation_id)
        .where(Conversation.user_id == uid, AgentExecution.started_at >= cutoff)
    )
    row = (await session.execute(exec_stmt)).one()
    total = row.total or 0
    success = row.success or 0

    # Conversation count
    conv_stmt = (
        select(func.count(Conversation.conversation_id))
        .where(Conversation.user_id == uid, Conversation.created_at >= cutoff)
    )
    conv_count = (await session.execute(conv_stmt)).scalar() or 0

    # Message count
    msg_stmt = (
        select(func.count(Message.message_id))
        .join(Conversation, Conversation.conversation_id == Message.conversation_id)
        .where(Conversation.user_id == uid, Message.created_at >= cutoff)
    )
    msg_count = (await session.execute(msg_stmt)).scalar() or 0

    return {
        "total_executions": total,
        "success": success,
        "failed": row.failed or 0,
        "success_rate": round(success / total * 100, 1) if total else 0,
        "total_conversations": conv_count,
        "total_messages": msg_count,
    }


async def get_token_breakdown_by_agent(
    session: AsyncSession, user_id: str, days: int = 30,
) -> list[dict]:
    """Aggregate token_details JSONB to get per-agent token totals."""
    from sqlalchemy import text

    uid = str(_to_uuid(user_id))
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    raw_sql = text("""
        SELECT agent_key, SUM(CAST(agent_val->>'total_tokens' AS int)) as total_tokens
        FROM messages m
        JOIN conversations c ON c.conversation_id = m.conversation_id,
        LATERAL jsonb_each(m.token_details) AS kv(agent_key, agent_val)
        WHERE c.user_id = :uid
          AND m.role = 'assistant'
          AND m.created_at >= :cutoff
          AND m.token_details IS NOT NULL
          AND jsonb_typeof(m.token_details) = 'object'
          AND m.token_details != '{}'
        GROUP BY agent_key
        ORDER BY total_tokens DESC
    """)
    result = await session.execute(raw_sql, {"uid": uid, "cutoff": cutoff})
    return [
        {"agent_name": r.agent_key, "total_tokens": r.total_tokens or 0}
        for r in result
    ]


async def get_web_scrape_stats(
    session: AsyncSession, user_id: str,
) -> list[dict]:
    """Return web scrape collection stats for the analytics dashboard."""
    uid = _to_uuid(user_id)
    result = await session.execute(
        select(WebScrapeCollection)
        .where(WebScrapeCollection.user_id == uid)
        .order_by(WebScrapeCollection.created_at.desc())
    )
    collections = result.scalars().all()
    stats = []
    for c in collections:
        stats.append({
            "collection_id": str(c.collection_id),
            "name": c.name,
            "status": c.status,
            "is_active": c.is_active,
            "total_pages": c.total_pages or 0,
            "total_chunks": c.total_chunks or 0,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        })
    return stats