"""
JWT-style token creation and verification.

Tokens are base64-encoded JSON payloads signed with HMAC-SHA256.
Secret key is loaded from ``config.jwt_secret`` (env var: ``JWT_SECRET``).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from base64 import b64decode, b64encode

from fastapi import HTTPException, status

from config.settings import config

_TOKEN_SECRET = config.jwt_secret
_TOKEN_EXPIRY_SECONDS = config.jwt_expiry_seconds


def create_token(user_id: str) -> str:
    """Create a signed token containing ``user_id`` and expiry."""
    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + _TOKEN_EXPIRY_SECONDS,
    }
    raw = json.dumps(payload).encode()
    sig = hmac.new(_TOKEN_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return b64encode(raw).decode() + "." + sig


def verify_token(token: str) -> str:
    """
    Verify token and return ``user_id``.

    Raises ``HTTPException(401)`` on invalid or expired tokens.
    """
    try:
        parts = token.split(".", 1)
        if len(parts) != 2:
            raise ValueError("bad format")
        raw = b64decode(parts[0])
        expected_sig = hmac.new(
            _TOKEN_SECRET.encode(), raw, hashlib.sha256
        ).hexdigest()
        if not hmac.compare_digest(parts[1], expected_sig):
            raise ValueError("bad signature")
        payload = json.loads(raw)
        if payload.get("exp", 0) < time.time():
            raise ValueError("token expired")
        return payload["user_id"]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        )
