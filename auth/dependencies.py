"""
FastAPI dependencies for authentication.

Provides ``db_session`` and ``get_current_user_id`` dependencies that
are used across all protected routes.
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from database.session import get_db_session

_bearer_scheme = HTTPBearer()


async def db_session(
    session: AsyncSession = Depends(get_db_session),
) -> AsyncGenerator[AsyncSession, None]:
    """Yield a DB session for route handlers."""
    yield session


async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(_bearer_scheme),
) -> str:
    """
    Extract and verify the Bearer token, returning the authenticated
    ``user_id`` (UUID string).
    """
    from auth.jwt import verify_token

    return verify_token(credentials.credentials)
