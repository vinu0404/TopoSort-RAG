
-- Users & sessions ──────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS users (
    user_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email         VARCHAR(255) UNIQUE NOT NULL,
    display_name  VARCHAR(128),
    password_hash VARCHAR(255) NOT NULL DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id       UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    ended_at      TIMESTAMPTZ,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE
);

/* BUG-4 FIX — separate CREATE INDEX for PostgreSQL */
CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

-- Conversations ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id          UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    title            TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_user    ON conversations(user_id);

-- Messages ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS messages (
    message_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    role             VARCHAR(16) NOT NULL,       -- 'user' | 'assistant' | 'system'
    content          TEXT NOT NULL,
    metadata         JSONB DEFAULT '{}',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_role         ON messages(role);

-- Agent executions (audit log) ───────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS agent_executions (
    execution_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    agent_name       VARCHAR(64) NOT NULL,
    task_description TEXT,
    status           VARCHAR(16) NOT NULL DEFAULT 'pending', -- pending|running|success|failed
    input_payload    JSONB,
    output_payload   JSONB,
    error_message    TEXT,
    started_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at     TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_agent_exec_conversation ON agent_executions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_agent_exec_agent        ON agent_executions(agent_name);

-- Uploaded documents ─────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS documents (
    doc_id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    filename        VARCHAR(512) NOT NULL,
    doc_type        VARCHAR(32),
    description     TEXT,
    total_chunks    INT,
    qdrant_collection VARCHAR(128),
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_user ON documents(user_id);

-- Conversation summaries (for memory) ────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversation_summaries (
    summary_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    summary_text     TEXT NOT NULL,
    turns_covered    INT NOT NULL DEFAULT 0,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_summaries_conversation ON conversation_summaries(conversation_id);

-- HITL (Human-in-the-Loop) approval requests ─────────────────────────────────

CREATE TABLE IF NOT EXISTS hitl_requests (
    request_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id  UUID NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
    agent_id         TEXT NOT NULL,
    agent_name       TEXT NOT NULL,
    tool_names       TEXT[] NOT NULL DEFAULT '{}',
    task_description TEXT NOT NULL DEFAULT '',
    status           VARCHAR(16) NOT NULL DEFAULT 'pending',
                     -- pending | approved | denied | timed_out | expired
    user_instructions TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    responded_at     TIMESTAMPTZ,
    expires_at       TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_hitl_conversation ON hitl_requests(conversation_id);
CREATE INDEX IF NOT EXISTS idx_hitl_status       ON hitl_requests(status);

-- User long-term memory (critical facts + preferences) ───────────────────────

CREATE TABLE IF NOT EXISTS user_long_term_memory (
    user_id          UUID PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
    critical_facts   JSONB NOT NULL DEFAULT '{}',
    preferences      JSONB NOT NULL DEFAULT '{}',
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
