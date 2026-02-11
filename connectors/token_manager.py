"""
Token manager — get / refresh / store per-user OAuth tokens.

This is the single interface that tools use to get an active token
for a given user + provider combination.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from connectors.registry import ConnectorRegistry
from connectors.models import UserConnection
from connectors.encryption import encrypt_token, decrypt_token
from database.session import async_session_factory

logger = logging.getLogger(__name__)


async def get_active_token(
    user_id: str,
    provider: str,
    *,
    db_session: Optional[AsyncSession] = None,
) -> Optional[str]:
    """
    Get a valid access token for the user + provider.

    1. Look up the connection in DB.
    2. If token is expired, refresh it automatically.
    3. Update ``last_used_at``.
    4. Return the access_token string, or None if not connected.
    """
    own_session = db_session is None
    session = db_session or async_session_factory()
    try:
        result = await session.execute(
            select(UserConnection).where(
                UserConnection.user_id == user_id,
                UserConnection.provider == provider,
                UserConnection.status == "active",
            )
        )
        conn = result.scalar_one_or_none()
        if not conn:
            return None

        # Check if token is expired (with 120s buffer)
        if conn.expires_at and conn.expires_at < datetime.now(timezone.utc) + timedelta(seconds=120):
            if not conn.refresh_token:
                conn.status = "expired"
                conn.error_message = "Token expired and no refresh token available"
                if own_session:
                    await session.commit()
                return None

            # Attempt refresh
            try:
                registry = ConnectorRegistry()
                connector = registry.get(provider)
                if not connector:
                    logger.error("No connector for provider %s", provider)
                    return None

                refreshed = await connector.refresh_access_token(
                    decrypt_token(conn.refresh_token)
                )
                conn.access_token = encrypt_token(refreshed["access_token"])
                conn.expires_at = datetime.now(timezone.utc) + timedelta(
                    seconds=refreshed.get("expires_in", 3600)
                )
                conn.last_refreshed = datetime.now(timezone.utc)
                # Some providers rotate refresh tokens
                if refreshed.get("refresh_token"):
                    conn.refresh_token = encrypt_token(refreshed["refresh_token"])
                conn.status = "active"
                conn.error_message = None
                logger.info("Refreshed %s token for user %s", provider, user_id)
            except Exception as exc:
                conn.status = "error"
                conn.error_message = f"Refresh failed: {exc}"
                logger.warning("Token refresh failed for %s/%s: %s", provider, user_id, exc)
                if own_session:
                    await session.commit()
                return None

        # Update last_used_at
        conn.last_used_at = datetime.now(timezone.utc)
        if own_session:
            await session.commit()

        return decrypt_token(conn.access_token)

    except Exception as exc:
        logger.error("get_active_token error: %s", exc)
        return None
    finally:
        if own_session:
            await session.close()


async def store_connection(
    user_id: str,
    provider: str,
    token_data: dict,
    *,
    db_session: Optional[AsyncSession] = None,
) -> str:
    """
    Store a new OAuth connection (or update existing).

    Parameters
    ----------
    token_data : dict
        Output from connector.handle_callback(): access_token, refresh_token,
        expires_in, scopes, account_id, account_label, provider_meta

    Returns
    -------
    connection_id as string
    """
    own_session = db_session is None
    session = db_session or async_session_factory()
    try:
        result = await session.execute(
            select(UserConnection).where(
                UserConnection.user_id == user_id,
                UserConnection.provider == provider,
                UserConnection.account_id == token_data.get("account_id", ""),
            )
        )
        existing = result.scalar_one_or_none()

        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=token_data.get("expires_in", 3600)
        )

        if existing:
            # Update existing connection
            existing.access_token = encrypt_token(token_data["access_token"])
            existing.refresh_token = encrypt_token(token_data.get("refresh_token") or decrypt_token(existing.refresh_token or ""))
            existing.expires_at = expires_at
            existing.scopes = token_data.get("scopes", [])
            existing.provider_meta = token_data.get("provider_meta", {})
            existing.account_label = token_data.get("account_label") or existing.account_label
            existing.status = "active"
            existing.error_message = None
            existing.connected_at = datetime.now(timezone.utc)
            conn_id = str(existing.connection_id)
            logger.info("Updated %s connection for user %s", provider, user_id)
        else:
            # Create new connection
            import uuid
            conn = UserConnection(
                connection_id=uuid.uuid4(),
                user_id=user_id,
                provider=provider,
                account_label=token_data.get("account_label", ""),
                account_id=token_data.get("account_id", ""),
                access_token=encrypt_token(token_data["access_token"]),
                refresh_token=encrypt_token(token_data.get("refresh_token") or ""),
                token_type="Bearer",
                expires_at=expires_at,
                scopes=token_data.get("scopes", []),
                provider_meta=token_data.get("provider_meta", {}),
                status="active",
            )
            session.add(conn)
            conn_id = str(conn.connection_id)
            logger.info("Created %s connection for user %s", provider, user_id)

        if own_session:
            await session.commit()
        else:
            await session.flush()

        return conn_id

    except Exception as exc:
        logger.error("store_connection error: %s", exc)
        if own_session:
            await session.rollback()
        raise
    finally:
        if own_session:
            await session.close()


async def get_user_connections(
    user_id: str,
    *,
    db_session: Optional[AsyncSession] = None,
) -> list[dict]:
    """Return all connections for a user (no tokens exposed)."""
    own_session = db_session is None
    session = db_session or async_session_factory()
    try:
        result = await session.execute(
            select(UserConnection).where(UserConnection.user_id == user_id)
        )
        rows = result.scalars().all()
        return [
            {
                "connection_id": str(c.connection_id),
                "provider": c.provider,
                "account_label": c.account_label,
                "account_id": c.account_id,
                "status": c.status,
                "scopes": c.scopes or [],
                "connected_at": c.connected_at.isoformat() if c.connected_at else None,
                "last_used_at": c.last_used_at.isoformat() if c.last_used_at else None,
                "error_message": c.error_message,
                "provider_meta": c.provider_meta or {},
            }
            for c in rows
        ]
    finally:
        if own_session:
            await session.close()


async def disconnect(
    user_id: str,
    connection_id: str,
    *,
    db_session: Optional[AsyncSession] = None,
) -> bool:
    """
    Revoke and delete a connection.
    Returns True if deleted, False if not found.
    """
    own_session = db_session is None
    session = db_session or async_session_factory()
    try:
        result = await session.execute(
            select(UserConnection).where(
                UserConnection.connection_id == connection_id,
                UserConnection.user_id == user_id,
            )
        )
        conn = result.scalar_one_or_none()
        if not conn:
            return False
        registry = ConnectorRegistry()
        connector = registry.get(conn.provider)
        if connector:
            try:
                await connector.revoke_token(decrypt_token(conn.access_token))
            except Exception:
                pass  # Best-effort revocation

        await session.delete(conn)
        if own_session:
            await session.commit()
        else:
            await session.flush()

        logger.info("Disconnected %s for user %s", conn.provider, user_id)
        return True

    except Exception as exc:
        logger.error("disconnect error: %s", exc)
        if own_session:
            await session.rollback()
        return False
    finally:
        if own_session:
            await session.close()
