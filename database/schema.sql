
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

CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);

-- Personas ───────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS personas (
    persona_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name         VARCHAR(128) NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_personas_user ON personas(user_id);

-- Conversations ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS conversations (
    conversation_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id       UUID NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    user_id          UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    persona_id       UUID REFERENCES personas(persona_id) ON DELETE SET NULL,
    title            TEXT,
    share_token      UUID UNIQUE,                -- non-null = publicly shared (read-only)
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
    model_used       VARCHAR(128),               -- which LLM model produced this message
    total_tokens     INTEGER DEFAULT 0,           -- total tokens consumed for this response turn
    token_details    JSONB DEFAULT '{}',          -- per-component breakdown {master_agent: {prompt_tokens, completion_tokens, total_tokens}, ...}
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
    processing_status VARCHAR(16) NOT NULL DEFAULT 'pending',
                      -- pending | processing | ready | failed
    error_message   TEXT,
    storage_key     VARCHAR(1024),           -- S3 object key: uploads/{user_id}/{doc_id}/{filename}
    storage_bucket  VARCHAR(128),            -- bucket name
    file_size_bytes BIGINT,                  -- original file size
    content_type    VARCHAR(128),            -- MIME type (application/pdf, ...)
    uploaded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_user   ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(user_id, processing_status);

-- Web scrape collections ────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS web_scrape_collections (
    collection_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name              VARCHAR(256) NOT NULL,
    is_active         BOOLEAN NOT NULL DEFAULT FALSE,
    status            VARCHAR(16) NOT NULL DEFAULT 'pending',
                      -- pending | scraping | ready | partial | failed
    total_pages       INT DEFAULT 0,
    total_chunks      INT DEFAULT 0,
    error_message     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wsc_user   ON web_scrape_collections(user_id);
CREATE INDEX IF NOT EXISTS idx_wsc_active ON web_scrape_collections(user_id, is_active);

CREATE TABLE IF NOT EXISTS web_scrape_urls (
    url_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collection_id     UUID NOT NULL REFERENCES web_scrape_collections(collection_id) ON DELETE CASCADE,
    url               TEXT NOT NULL,
    depth             INT NOT NULL DEFAULT 1,
    status            VARCHAR(16) NOT NULL DEFAULT 'pending',
                      -- pending | scraping | ready | failed
    pages_scraped     INT DEFAULT 0,
    chunks_created    INT DEFAULT 0,
    error_message     TEXT,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wsu_collection ON web_scrape_urls(collection_id);

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

-- OAuth connections (Gmail, Slack, Notion, GitHub, …) ────────────────────────

CREATE TABLE IF NOT EXISTS user_connections (
    connection_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id          UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    provider         VARCHAR(32) NOT NULL,           -- 'gmail', 'slack', 'notion', 'github'
    account_label    VARCHAR(128),                    -- "john@gmail.com", "Acme Workspace"
    account_id       VARCHAR(256),                    -- provider-side unique ID
    access_token     TEXT NOT NULL,
    refresh_token    TEXT,
    token_type       VARCHAR(32) DEFAULT 'Bearer',
    expires_at       TIMESTAMPTZ,
    scopes           TEXT[] DEFAULT '{}',
    provider_meta    JSONB DEFAULT '{}',
    status           VARCHAR(16) NOT NULL DEFAULT 'active',   -- active | expired | revoked | error
    connected_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_refreshed   TIMESTAMPTZ,
    last_used_at     TIMESTAMPTZ,
    error_message    TEXT,
    UNIQUE(user_id, provider, account_id)
);

CREATE INDEX IF NOT EXISTS idx_conn_user     ON user_connections(user_id);
CREATE INDEX IF NOT EXISTS idx_conn_provider ON user_connections(user_id, provider);
CREATE INDEX IF NOT EXISTS idx_conn_status   ON user_connections(status);

-- Scheduled Jobs ──────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS scheduled_jobs (
    job_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id           UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name              VARCHAR(256) NOT NULL,
    description       TEXT NOT NULL DEFAULT '',
    cron_expression   VARCHAR(128) NOT NULL,
    timezone          VARCHAR(64) NOT NULL DEFAULT 'UTC',
    status            VARCHAR(16) NOT NULL DEFAULT 'active',
                      -- active | paused | deleted
    notification_mode VARCHAR(32) NOT NULL DEFAULT 'in_app',
                      -- email | in_app | none
    notification_target TEXT,
    max_retries       INT NOT NULL DEFAULT 2,
    metadata          JSONB DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_run_at       TIMESTAMPTZ,
    next_run_at       TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_sj_user   ON scheduled_jobs(user_id);
CREATE INDEX IF NOT EXISTS idx_sj_status ON scheduled_jobs(status);

CREATE TABLE IF NOT EXISTS scheduled_job_steps (
    step_id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id            UUID NOT NULL REFERENCES scheduled_jobs(job_id) ON DELETE CASCADE,
    step_order        INT NOT NULL,
    agent_name        VARCHAR(64) NOT NULL,
    task              TEXT NOT NULL,
    entities          JSONB DEFAULT '{}',
    tools             TEXT[] DEFAULT '{}',
    depends_on_steps  INT[] DEFAULT '{}',
    timeout           INT NOT NULL DEFAULT 60,
    max_retries       INT NOT NULL DEFAULT 2,
    priority          VARCHAR(16) NOT NULL DEFAULT 'critical',
    config            JSONB DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(job_id, step_order)
);

CREATE INDEX IF NOT EXISTS idx_sjs_job ON scheduled_job_steps(job_id);

CREATE TABLE IF NOT EXISTS scheduled_job_runs (
    run_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    job_id            UUID NOT NULL REFERENCES scheduled_jobs(job_id) ON DELETE CASCADE,
    status            VARCHAR(16) NOT NULL DEFAULT 'pending',
                      -- pending | running | success | partial_failure | failed
    trigger_type      VARCHAR(16) NOT NULL DEFAULT 'scheduled',
                      -- scheduled | manual
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    error_summary     TEXT,
    total_steps       INT NOT NULL DEFAULT 0,
    completed_steps   INT NOT NULL DEFAULT 0,
    failed_steps      INT NOT NULL DEFAULT 0,
    notification_sent BOOLEAN NOT NULL DEFAULT FALSE,
    metadata          JSONB DEFAULT '{}',
    created_at        TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_sjr_job ON scheduled_job_runs(job_id);

CREATE TABLE IF NOT EXISTS scheduled_job_step_results (
    result_id         UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id            UUID NOT NULL REFERENCES scheduled_job_runs(run_id) ON DELETE CASCADE,
    step_id           UUID NOT NULL REFERENCES scheduled_job_steps(step_id) ON DELETE CASCADE,
    step_order        INT NOT NULL,
    agent_name        VARCHAR(64) NOT NULL,
    status            VARCHAR(16) NOT NULL DEFAULT 'pending',
                      -- pending | running | success | failed | skipped
    agent_output      JSONB,
    error_message     TEXT,
    started_at        TIMESTAMPTZ,
    completed_at      TIMESTAMPTZ,
    resource_usage    JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sjsr_run ON scheduled_job_step_results(run_id);
