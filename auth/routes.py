"""
Auth API routes — register, login.

Route prefix: /api/v1/auth
"""

from __future__ import annotations

import logging
import uuid
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import db_session
from auth.jwt import create_token
from auth.models import User
from auth.password import hash_password, verify_password

logger = logging.getLogger(__name__)

router = APIRouter(tags=["auth"])


# ── Request / response schemas ─────────────────────────────────────────


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


# ── Endpoints ──────────────────────────────────────────────────────────


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
        password_hash=hash_password(req.password),
    )
    session.add(user)
    await session.flush()

    token = create_token(str(user.user_id))
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

    if user is None or not verify_password(req.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    token = create_token(str(user.user_id))
    logger.info("Login: %s (%s)", user.display_name, user.user_id)

    return {
        "user_id": str(user.user_id),
        "display_name": user.display_name,
        "email": user.email,
        "token": token,
    }
