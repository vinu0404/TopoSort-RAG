"""
Application settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


# ── Supported models per provider ─────────────────────────────────────────
# Frontend reads this via GET /api/v1/models to populate the selector.
# label = display name, ctx = context window (informational).
MODEL_CATALOG = {
    "openai": [
        {"id": "gpt-4o",          "label": "GPT-4o",          "ctx": 128_000},
        {"id": "gpt-4o-mini",     "label": "GPT-4o Mini",     "ctx": 128_000},
        {"id": "gpt-4.1",         "label": "GPT-4.1",         "ctx": 1_000_000},
        {"id": "gpt-4.1-mini",    "label": "GPT-4.1 Mini",    "ctx": 1_000_000},
        {"id": "gpt-4.1-nano",    "label": "GPT-4.1 Nano",    "ctx": 1_000_000},
        {"id": "o3-mini",         "label": "o3 Mini",         "ctx": 200_000},
    ],
    "anthropic": [
        {"id": "claude-sonnet-4-20250514",      "label": "Claude Sonnet 4",  "ctx": 200_000},
        {"id": "claude-3-5-sonnet-20241022", "label": "Claude 3.5 Sonnet", "ctx": 200_000},
        {"id": "claude-3-5-haiku-20241022",  "label": "Claude 3.5 Haiku",  "ctx": 200_000},
    ],
    "google": [
        {"id": "gemini-2.0-flash",    "label": "Gemini 2.0 Flash",    "ctx": 1_000_000},
        {"id": "gemini-2.5-pro-preview-05-06","label": "Gemini 2.5 Pro", "ctx": 1_000_000},
    ],
}

# Flat set for quick validation
_VALID_MODELS: set[str] = {m["id"] for models in MODEL_CATALOG.values() for m in models}

def model_provider_for(model_id: str) -> str | None:
    """Return the provider name for a model id, or None if unknown."""
    for provider, models in MODEL_CATALOG.items():
        if any(m["id"] == model_id for m in models):
            return provider
    return None


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    master_model_provider: str = "openai"
    master_model: str = "gpt-4o"
    master_temperature: float = 0.3
    composer_model_provider: str = "openai"
    composer_model: str = "gpt-4o"
    composer_temperature: float = 0.7
    rag_model_provider: str = "openai"
    rag_model: str = "gpt-4o-mini"
    rag_temperature: float = 0.1
    rag_use_llm_reranking: bool = True
    rag_description_top_k: int = 5
    rag_description_score_threshold: float = 0.4
    code_model_provider: str = "openai"
    code_model: str = "gpt-4o"
    code_temperature: float = 0.2
    mail_model_provider: str = "openai"
    mail_model: str = "gpt-4o"
    mail_temperature: float = 0.5
    web_model_provider: str = "openai"
    web_model: str = "gpt-4o-mini"
    web_temperature: float = 0.2
    github_model_provider: str = "openai"
    github_model: str = "gpt-4o-mini"
    github_temperature: float = 0.2
    tavily_api_key: str = ""

    # ── Gmail ────────────────────────────────────────────────────────────
    gmail_sender_email: str = ""  

    # ── Security Secrets ──────────────────────────────────────────────────
    jwt_secret: str = "change-me-jwt-secret-key"      
    jwt_expiry_seconds: int = 604800                   
    oauth_state_secret: str = "change-me-oauth-state" 
    token_encryption_key: str = "encryptmeplease"                       # Fernet key for encrypting OAuth tokens at rest

    # ── OAuth Connectors ─────────────────────────────────────────────────
    google_client_id: str = ""          # Google OAuth Web App client ID
    google_client_secret: str = ""      # Google OAuth Web App client secret
    github_client_id: str = ""          # GitHub App OAuth client ID
    github_client_secret: str = ""      # GitHub App OAuth client secret
    oauth_redirect_base: str = "http://localhost:8000"  # base URL for OAuth callbacks

    # ── Embedding Model ─────────────────────────────────────────────────
    embedding_model_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = "postgresql+asyncpg://user:password@localhost:5432/agentic_rag"
    qdrant_url: str = "http://localhost:6333"

    # ── Server ───────────────────────────────────────────────────────────
    port: int = 8000
    host: str = "0.0.0.0"
    debug: bool = False
    cors_origins: list = ["*"]

    default_agent_timeout: int = 30
    default_max_retries: int = 2
    default_backoff_multiplier: float = 2.0
    
    conversation_summary_interval: int = 5
    max_conversation_history: int = 10

    # ── HITL (Human-in-the-Loop) ─────────────────────────────────────────────────
    hitl_timeout_seconds: int = 120   # how long to wait for user approval
    hitl_poll_interval: float = 6   # seconds between DB polls
    hitl_classifier_provider: str = "openai"   # model for classify enhance vs override
    hitl_classifier_model: str = "gpt-4o-mini"

    # ── Redis / Celery ─────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # ── Voice (STT + TTS) ───────────────────────────────────────────────────────
    stt_provider: str = "assemblyai"
    assemblyai_api_key: str = ""
    tts_provider: str = "aws_polly"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "us-east-1"
    aws_polly_voice_id: str = "Matthew"
    voice_max_audio_size_mb: int = 25
    voice_max_duration_sec: int = 60

    # ── S3-compatible Cloud Storage (Backblaze B2 / AWS / MinIO) ─────────────────
    s3_endpoint: str = ""                       # e.g. https://s3.us-east-005.backblazeb2.com
    s3_region: str = "us-east-005"
    s3_access_key_id: str = ""
    s3_secret_access_key: str = ""
    s3_bucket: str = ""                         # e.g. linkdrop
    s3_presign_expiry: int = 300                # pre-signed URL TTL in seconds (5 min)

    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
    }

    def get_agent_model_config(self, agent_name: str) -> dict:
        """
        Return (provider, model, temperature) for a given agent type.

        """
        mapping = {
            "rag_agent": (self.rag_model_provider, self.rag_model, self.rag_temperature),
            "code_agent": (self.code_model_provider, self.code_model, self.code_temperature),
            "mail_agent": (self.mail_model_provider, self.mail_model, self.mail_temperature),
            "web_search_agent": (self.web_model_provider, self.web_model, self.web_temperature),
            "github_agent": (self.github_model_provider, self.github_model, self.github_temperature),
            "master": (self.master_model_provider, self.master_model, self.master_temperature),
            "composer": (self.composer_model_provider, self.composer_model, self.composer_temperature),
        }
        provider, model, temperature = mapping.get(
            agent_name,
            (self.openai_api_key and "openai" or "openai", "gpt-4o-mini", 0.3),
        )
        return {"provider": provider, "model": model, "temperature": temperature}


config = Settings()


# ── Default personas seeded for every new user ──────────────────────────────
DEFAULT_PERSONAS = [
    {
        "name": "Friend",
        "description": (
            "You are a close, supportive friend. Speak casually and warmly, "
            "use humor when appropriate, be encouraging, and keep the tone "
            "relaxed and genuine — like chatting with a best friend."
        ),
    },
    {
        "name": "Teacher",
        "description": (
            "You are a patient, knowledgeable teacher. Explain concepts clearly "
            "with examples, break down complex topics step by step, encourage "
            "curiosity, and check understanding along the way."
        ),
    },
    {
        "name": "Lover",
        "description": (
            "You are a caring, romantic partner. Speak affectionately and tenderly, "
            "be emotionally supportive and attentive, use a warm intimate tone, "
            "and make the user feel valued and cherished."
        ),
    },
]
