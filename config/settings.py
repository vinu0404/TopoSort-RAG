"""
Application settings loaded from environment variables.
"""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: str = ""
    anthropic_api_key: Optional[str] = None
    master_model_provider: str = "openai"
    master_model: str = "gpt-4o"
    master_temperature: float = 0.3
    composer_model_provider: str = "openai"
    composer_model: str = "gpt-4o"
    composer_temperature: float = 0.7
    rag_model_provider: str = "openai"
    rag_model: str = "gpt-4o-mini"
    rag_temperature: float = 0.1
    code_model_provider: str = "openai"
    code_model: str = "gpt-4o"
    code_temperature: float = 0.2
    mail_model_provider: str = "openai"
    mail_model: str = "gpt-4o"
    mail_temperature: float = 0.5
    web_model_provider: str = "openai"
    web_model: str = "gpt-4o-mini"
    web_temperature: float = 0.2
    tavily_api_key: str = ""

    # ── Gmail OAuth2 ────────────────────────────────────────────────────
    gmail_credentials_file: str = ""   # path to credentials.json (desktop flow, legacy)
    gmail_token_file: str = ""          # path to token.json (auto-created, legacy)
    gmail_sender_email: str = ""        # default sender address

    # ── Security Secrets ──────────────────────────────────────────────────
    jwt_secret: str = "change-me-jwt-secret-key"       # HMAC secret for auth tokens
    jwt_expiry_seconds: int = 604800                    # 7 days
    oauth_state_secret: str = "change-me-oauth-state"   # HMAC secret for OAuth CSRF state
    token_encryption_key: str = ""                       # Fernet key for encrypting OAuth tokens at rest

    # ── OAuth Connectors ─────────────────────────────────────────────────
    google_client_id: str = ""          # Google OAuth Web App client ID
    google_client_secret: str = ""      # Google OAuth Web App client secret
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
    
    conversation_summary_interval: int = 3
    max_conversation_history: int = 8

    # ── HITL (Human-in-the-Loop) ─────────────────────────────────────────────────
    hitl_timeout_seconds: int = 120   # how long to wait for user approval
    hitl_poll_interval: float = 5   # seconds between DB polls
    hitl_classifier_provider: str = "openai"   # cheap model for classify enhance vs override
    hitl_classifier_model: str = "gpt-4o-mini"

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
            "master": (self.master_model_provider, self.master_model, self.master_temperature),
            "composer": (self.composer_model_provider, self.composer_model, self.composer_temperature),
        }
        provider, model, temperature = mapping.get(
            agent_name,
            (self.openai_api_key and "openai" or "openai", "gpt-4o-mini", 0.3),
        )
        return {"provider": provider, "model": model, "temperature": temperature}


config = Settings()
