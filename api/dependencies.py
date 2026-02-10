"""
FastAPI dependencies (shared across routes).
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db_session

_bearer_scheme = HTTPBearer()


async def db_session(session: AsyncSession = Depends(get_db_session)) -> AsyncGenerator[AsyncSession, None]:
    """Re-export so routes import from a single place."""
    yield session


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """
    Extract and verify the Bearer token from the Authorization header.
    Returns the authenticated user_id (UUID string).
    """
    from api.auth import verify_token
    return verify_token(credentials.credentials)
