"""
Database helper functions — ensure parent records exist and persist data.

"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import (
    AgentExecution,
    Conversation,
    ConversationSummary,
    Document,
    Message,
    Session,
    User,
)

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
) -> str:
    """Return the latest ``conversation_id`` for a session, creating one if needed."""
    uid = _to_uuid(user_id)
    sid = _to_uuid(session_id)

    result = await session.execute(
        select(Conversation)
        .where(Conversation.session_id == sid, Conversation.user_id == uid)
        .order_by(Conversation.created_at.desc())
        .limit(1)
    )
    existing = result.scalar_one_or_none()
    if existing:
        return str(existing.conversation_id)

    cid = uuid.uuid4()
    session.add(
        Conversation(
            conversation_id=cid,
            session_id=sid,
            user_id=uid,
            title=title or "New conversation",
        )
    )
    await session.flush()
    return str(cid)


async def save_messages(
    session: AsyncSession,
    conversation_id: str,
    user_query: str,
    assistant_answer: str,
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Insert a *user* message and an *assistant* message."""
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
            created_at=now,
        )
    )
    await session.flush()


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
        )
    )
    await session.flush()


async def load_conversation_messages(
    session: AsyncSession,
    conversation_id: str,
    limit: int = 50,
) -> list[dict[str, str]]:
    """
    Load the most recent messages for a conversation from PostgreSQL.

    Returns pairs of user/assistant messages ordered chronologically,
    capped at *limit* rows to bound token usage.
    """
    cid = _to_uuid(conversation_id)
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == cid)
        .order_by(Message.created_at.asc())
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
