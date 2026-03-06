-- Migration 002: Add personas table and persona_id FK on conversations
-- Run: echo "$(cat database/migrations/002_add_personas.sql)" | docker exec -i mrag-postgres-1 psql -U postgres -d mrag

CREATE TABLE IF NOT EXISTS personas (
    persona_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id      UUID NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    name         VARCHAR(128) NOT NULL,
    description  TEXT NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_personas_user ON personas(user_id);

ALTER TABLE conversations
    ADD COLUMN IF NOT EXISTS persona_id UUID REFERENCES personas(persona_id) ON DELETE SET NULL;
