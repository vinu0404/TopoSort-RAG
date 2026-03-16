"""
SQLAlchemy ORM models mirroring database/schema.sql.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    BigInteger,
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
    connections = relationship("UserConnection", back_populates="user", cascade="all, delete-orphan")
    personas = relationship("Persona", back_populates="user", cascade="all, delete-orphan")
    web_scrape_collections = relationship("WebScrapeCollection", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    __tablename__ = "sessions"

    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    started_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)

    user = relationship("User", back_populates="sessions")
    conversations = relationship("Conversation", back_populates="session", cascade="all, delete-orphan")


class Persona(Base):
    __tablename__ = "personas"

    persona_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    name = Column(String(128), nullable=False)
    description = Column(Text, nullable=False, default="")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="personas")
    conversations = relationship("Conversation", back_populates="persona")


class Conversation(Base):
    __tablename__ = "conversations"

    conversation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("sessions.session_id", ondelete="CASCADE"), nullable=False)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    persona_id = Column(UUID(as_uuid=True), ForeignKey("personas.persona_id", ondelete="SET NULL"), nullable=True)
    title = Column(Text)
    share_token = Column(UUID(as_uuid=True), unique=True, nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("Session", back_populates="conversations")
    persona = relationship("Persona", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    agent_executions = relationship("AgentExecution", back_populates="conversation", cascade="all, delete-orphan")
    summaries = relationship("ConversationSummary", back_populates="conversation", cascade="all, delete-orphan")


class Message(Base):
    __tablename__ = "messages"

    message_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    role = Column(String(16), nullable=False)
    content = Column(Text, nullable=False)
    model_used = Column(String(128), nullable=True)
    total_tokens = Column(Integer, default=0)
    token_details = Column(JSONB, default=dict)
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
    processing_status = Column(String(16), nullable=False, default="pending")
    error_message = Column(Text)
    storage_key = Column(String(1024))
    storage_bucket = Column(String(128))
    file_size_bytes = Column(BigInteger)
    content_type = Column(String(128))
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

    user = relationship("User", back_populates="connections")


class WebScrapeCollection(Base):
    __tablename__ = "web_scrape_collections"

    collection_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    name = Column(String(256), nullable=False)
    is_active = Column(Boolean, nullable=False, default=False)
    status = Column(String(16), nullable=False, default="pending")
    total_pages = Column(Integer, default=0)
    total_chunks = Column(Integer, default=0)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    user = relationship("User", back_populates="web_scrape_collections")
    urls = relationship("WebScrapeUrl", back_populates="collection", cascade="all, delete-orphan")


class WebScrapeUrl(Base):
    __tablename__ = "web_scrape_urls"

    url_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    collection_id = Column(UUID(as_uuid=True), ForeignKey("web_scrape_collections.collection_id", ondelete="CASCADE"), nullable=False)
    url = Column(Text, nullable=False)
    depth = Column(Integer, nullable=False, default=1)
    status = Column(String(16), nullable=False, default="pending")
    pages_scraped = Column(Integer, default=0)
    chunks_created = Column(Integer, default=0)
    error_message = Column(Text)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    collection = relationship("WebScrapeCollection", back_populates="urls")


# ── Scheduled Jobs ──────────────────────────────────────────────────────────


class ScheduledJob(Base):
    __tablename__ = "scheduled_jobs"

    job_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.user_id", ondelete="CASCADE"), nullable=False)
    name = Column(String(256), nullable=False)
    description = Column(Text, nullable=False, default="")
    cron_expression = Column(String(128), nullable=False)
    timezone = Column(String(64), nullable=False, default="UTC")
    status = Column(String(16), nullable=False, default="active")
    notification_mode = Column(String(32), nullable=False, default="in_app")
    notification_target = Column(Text, nullable=True)
    max_retries = Column(Integer, nullable=False, default=2)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    last_run_at = Column(DateTime(timezone=True), nullable=True)
    next_run_at = Column(DateTime(timezone=True), nullable=True)

    user = relationship("User")
    steps = relationship(
        "ScheduledJobStep", back_populates="job",
        cascade="all, delete-orphan", order_by="ScheduledJobStep.step_order",
    )
    runs = relationship("ScheduledJobRun", back_populates="job", cascade="all, delete-orphan")


class ScheduledJobStep(Base):
    __tablename__ = "scheduled_job_steps"

    step_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("scheduled_jobs.job_id", ondelete="CASCADE"), nullable=False)
    step_order = Column(Integer, nullable=False)
    agent_name = Column(String(64), nullable=False)
    task = Column(Text, nullable=False)
    entities = Column(JSONB, default=dict)
    tools = Column(ARRAY(Text), default=list)
    depends_on_steps = Column(ARRAY(Integer), default=list)
    timeout = Column(Integer, nullable=False, default=60)
    max_retries = Column(Integer, nullable=False, default=2)
    priority = Column(String(16), nullable=False, default="critical")
    config = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    job = relationship("ScheduledJob", back_populates="steps")


class ScheduledJobRun(Base):
    __tablename__ = "scheduled_job_runs"

    run_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    job_id = Column(UUID(as_uuid=True), ForeignKey("scheduled_jobs.job_id", ondelete="CASCADE"), nullable=False)
    status = Column(String(16), nullable=False, default="pending")
    trigger_type = Column(String(16), nullable=False, default="scheduled")
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    error_summary = Column(Text, nullable=True)
    total_steps = Column(Integer, nullable=False, default=0)
    completed_steps = Column(Integer, nullable=False, default=0)
    failed_steps = Column(Integer, nullable=False, default=0)
    notification_sent = Column(Boolean, nullable=False, default=False)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at_ = Column("created_at", DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    job = relationship("ScheduledJob", back_populates="runs")
    step_results = relationship("ScheduledJobStepResult", back_populates="run", cascade="all, delete-orphan")


class ScheduledJobStepResult(Base):
    __tablename__ = "scheduled_job_step_results"

    result_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    run_id = Column(UUID(as_uuid=True), ForeignKey("scheduled_job_runs.run_id", ondelete="CASCADE"), nullable=False)
    step_id = Column(UUID(as_uuid=True), ForeignKey("scheduled_job_steps.step_id", ondelete="CASCADE"), nullable=False)
    step_order = Column(Integer, nullable=False)
    agent_name = Column(String(64), nullable=False)
    status = Column(String(16), nullable=False, default="pending")
    agent_output = Column(JSONB, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    resource_usage = Column(JSONB, default=dict)

    run = relationship("ScheduledJobRun", back_populates="step_results")


class Artifact(Base):
    __tablename__ = "artifacts"

    artifact_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    conversation_id = Column(UUID(as_uuid=True), ForeignKey("conversations.conversation_id", ondelete="CASCADE"), nullable=False)
    message_id = Column(UUID(as_uuid=True), ForeignKey("messages.message_id", ondelete="SET NULL"), nullable=True)
    agent_id = Column(Text, nullable=False)
    agent_name = Column(String(64), nullable=False)
    filename = Column(String(512), nullable=False)
    artifact_type = Column(String(32), nullable=False)
    content_type = Column(String(128), nullable=False)
    file_size_bytes = Column(BigInteger, nullable=False, default=0)
    storage_key = Column(String(1024), nullable=False)
    storage_bucket = Column(String(128), nullable=True)
    preview_data = Column(JSONB, default=dict)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
