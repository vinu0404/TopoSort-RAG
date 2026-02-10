"""
Authentication — register, login, and token verification.

Uses simple SHA-256 password hashing 
Token is a base64-encoded JSON payload with user_id + expiry.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from base64 import b64decode, b64encode
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.dependencies import db_session
from database.models import User

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])
_TOKEN_SECRET = "mrag-secret-key"
_TOKEN_EXPIRY_SECONDS = 86400 * 7 


class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=2, max_length=64)
    email: str = Field(..., min_length=5, max_length=255)
    password: str = Field(..., min_length=4, max_length=128)


class LoginRequest(BaseModel):
    email: str
    password: str


class AuthResponse(BaseModel):
    user_id: str
    display_name: str
    email: str
    token: str



def _hash_password(password: str) -> str:
    """SHA-256 hash with salt."""
    salt = "mrag-salt"
    return hashlib.sha256(f"{salt}:{password}".encode()).hexdigest()


def _verify_password(password: str, password_hash: str) -> bool:
    return hmac.compare_digest(_hash_password(password), password_hash)



def _create_token(user_id: str) -> str:
    payload = {
        "user_id": user_id,
        "exp": int(time.time()) + _TOKEN_EXPIRY_SECONDS,
    }
    raw = json.dumps(payload).encode()
    sig = hmac.new(_TOKEN_SECRET.encode(), raw, hashlib.sha256).hexdigest()
    return b64encode(raw).decode() + "." + sig


def verify_token(token: str) -> str:
    """Verify token and return user_id. Raises HTTPException on failure."""
    try:
        parts = token.split(".", 1)
        if len(parts) != 2:
            raise ValueError("bad format")
        raw = b64decode(parts[0])
        expected_sig = hmac.new(_TOKEN_SECRET.encode(), raw, hashlib.sha256).hexdigest()
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



@router.post("/register", response_model=AuthResponse)
async def register(
    req: RegisterRequest,
    session: AsyncSession = Depends(db_session),
) -> Dict[str, Any]:
    """Register a new user."""
    result = await session.execute(
        select(User).where(User.email == req.email)
    )
    if result.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email already registered",
        )

    user = User(
        user_id=uuid.uuid4(),
        email=req.email,
        display_name=req.username,
        password_hash=_hash_password(req.password),
    )
    session.add(user)
    await session.flush()

    token = _create_token(str(user.user_id))
    logger.info("Registered user %s (%s)", req.username, user.user_id)

    return {
        "user_id": str(user.user_id),
        "display_name": user.display_name,
        "email": user.email,
        "token": token,
    }


@router.post("/login", response_model=AuthResponse)
async def login(
    req: LoginRequest,
    session: AsyncSession = Depends(db_session),
) -> Dict[str, Any]:
    """Login with email + password."""
    result = await session.execute(
        select(User).where(User.email == req.email)
    )
    user = result.scalar_one_or_none()

    if user is None or not _verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = _create_token(str(user.user_id))
    logger.info("Login: %s (%s)", user.display_name, user.user_id)

    return {
        "user_id": str(user.user_id),
        "display_name": user.display_name,
        "email": user.email,
        "token": token,
    }
