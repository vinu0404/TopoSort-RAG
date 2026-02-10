"""
FastAPI dependencies (shared across routes).
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db_session


async def db_session(session: AsyncSession = Depends(get_db_session)) -> AsyncGenerator[AsyncSession, None]:
    """Re-export so routes import from a single place."""
    yield session


async def get_current_user_id(
    authorization: str = Header(..., alias="Authorization"),
) -> str:
    """
    Extract and verify the Bearer token from the Authorization header.
    Returns the authenticated user_id (UUID string).
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Bearer token",
        )
    token = authorization[7:]
    from api.auth import verify_token
    return verify_token(token)
