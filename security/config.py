"""
Security configuration: CORS settings, secret validation, security defaults.
"""

from __future__ import annotations

import logging
import os
from typing import List

logger = logging.getLogger("security.config")


def get_cors_origins() -> List[str]:
    """
    Read allowed CORS origins from the CORS_ORIGINS env var (comma-separated).

    Falls back to common localhost origins when DEBUG=true.
    Returns an empty list in production when not configured — callers
    can decide whether to allow or block all origins.
    """
    origins_str = os.getenv("CORS_ORIGINS", "")

    if origins_str and origins_str.strip() != "*":
        return [o.strip() for o in origins_str.split(",") if o.strip()]

    # In debug mode, allow common local dev origins
    if os.getenv("DEBUG", "").lower() in ("true", "1"):
        return [
            "http://localhost:3000",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8000",
        ]

    # Fallback: keep existing behaviour (allow all) so we don't break
    # deployments that haven't set CORS_ORIGINS yet.
    return ["*"]


def get_cors_config() -> dict:
    """Return kwargs suitable for FastAPI ``CORSMiddleware``."""
    origins = get_cors_origins()
    allow_all = origins == ["*"]

    return {
        "allow_origins": origins,
        "allow_credentials": not allow_all,  # credentials + "*" is invalid
        "allow_methods": ["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"],
        "allow_headers": [
            "Authorization",
            "Content-Type",
            "X-Requested-With",
            "Accept",
            "Origin",
        ],
        "expose_headers": [
            "X-RateLimit-Remaining",
            "X-Process-Time",
            "Retry-After",
        ],
    }


def validate_secrets() -> List[str]:
    """
    Check for insecure default secrets at startup.

    Returns a list of human-readable warning strings.
    """
    from config.settings import config

    warnings: List[str] = []

    if config.jwt_secret == "change-me-jwt-secret-key":
        warnings.append(
            "JWT_SECRET is using the default value — set a secure random string (32+ chars)"
        )
    elif len(config.jwt_secret) < 32:
        warnings.append("JWT_SECRET should be at least 32 characters")

    if config.oauth_state_secret == "change-me-oauth-state":
        warnings.append(
            "OAUTH_STATE_SECRET is using the default value — set a secure random string"
        )

    if config.token_encryption_key == "encryptmeplease":
        warnings.append(
            "TOKEN_ENCRYPTION_KEY is using the default value — generate a Fernet key"
        )

    return warnings
