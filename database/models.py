"""
SQLAlchemy ORM models mirroring database/schema.sql.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    user_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String(255), unique=True, nullable=False)
    display_name = Column(String(128))
    password_hash = Column(String(255), nullable=False, default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")
    documents = relationship("Document", back_populates="user", cascade="all, delete-orphan")
    connections = relationship("UserConnection", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    title = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("Session", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    agent_executions = relationship("AgentExecution", back_populates="conversation", cascade="all, delete-orphan")
    summaries = relationship("ConversationSummary", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="messages")


class AgentExecution(Base):
    __tablename__ = "agent_executions"

    execution_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    agent_name = Column(String(64), nullable=False)
    task_description = Column(Text)
    status = Column(String(16), default="pending")
    input_payload = Column(JSONB)
    output_payload = Column(JSONB)
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime(timezone=True), nullable=True)

    conversation = relationship("Conversation", back_populates="agent_executions")


class Document(Base):
    __tablename__ = "documents"

    doc_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(512), nullable=False)
    doc_type = Column(String(32))
    description = Column(Text)
    total_chunks = Column(Integer)
    qdrant_collection = Column(String(128))
    uploaded_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="documents")


class ConversationSummary(Base):
    __tablename__ = "conversation_summaries"

    summary_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    summary_text = Column(Text, nullable=False)
    turns_covered = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    conversation = relationship("Conversation", back_populates="summaries")


class HitlRequest(Base):
    __tablename__ = "hitl_requests"

    request_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(
        UUID(as_uuid=True),
        ForeignKey("conversations.conversation_id", ondelete="CASCADE"),
        nullable=False,
    )
    agent_id = Column(Text, nullable=False)
    agent_name = Column(Text, nullable=False)
    tool_names = Column(ARRAY(Text), nullable=False, default=list)
    task_description = Column(Text, nullable=False, default="")
    status = Column(String(16), nullable=False, default="pending")
    user_instructions = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    responded_at = Column(DateTime(timezone=True), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)

    conversation = relationship("Conversation")


class UserLongTermMemory(Base):
    __tablename__ = "user_long_term_memory"

    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), primary_key=True)
    critical_facts = Column(JSONB, nullable=False, default=dict)
    preferences = Column(JSONB, nullable=False, default=dict)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))


class UserConnection(Base):
    __tablename__ = "user_connections"

    connection_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    provider = Column(String(32), nullable=False)
    account_label = Column(String(128))
    account_id = Column(String(256))
    access_token = Column(Text, nullable=False)
    refresh_token = Column(Text)
    token_type = Column(String(32), default="Bearer")
    expires_at = Column(DateTime(timezone=True))
    scopes = Column(ARRAY(Text), default=list)
    provider_meta = Column(JSONB, default=dict)
    status = Column(String(16), nullable=False, default="active")
    connected_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_refreshed = Column(DateTime(timezone=True))
    last_used_at = Column(DateTime(timezone=True))
    error_message = Column(Text)

    user = relationship("User")
