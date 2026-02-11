"""
Connector API routes — OAuth connect/callback, list connections, disconnect.

Route prefix: /api/v1/connectors
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from base64 import b64decode, b64encode
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import db_session, get_current_user_id
from config.settings import config
from connectors.registry import ConnectorRegistry
from connectors.token_manager import disconnect, get_user_connections, store_connection

logger = logging.getLogger(__name__)

router = APIRouter(tags=["connectors"])

# ── State token helpers (CSRF protection) ──────────────────────────────

_STATE_SECRET = config.oauth_state_secret
_STATE_TTL = 600  # seconds


def _create_state(user_id: str) -> str:
    """Create an opaque state string encoding user_id + expiry."""
    payload = json.dumps({"user_id": user_id, "exp": int(time.time()) + _STATE_TTL})
    raw = payload.encode()
    sig = hmac.new(_STATE_SECRET.encode(), raw, hashlib.sha256).hexdigest()[:16]
    return b64encode(raw).decode() + "." + sig


def _verify_state(state: str) -> str:
    """Verify state token, return user_id. Raises on failure."""
    try:
        parts = state.split(".", 1)
        if len(parts) != 2:
            raise ValueError("bad format")
        raw = b64decode(parts[0])
        expected_sig = hmac.new(_STATE_SECRET.encode(), raw, hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(parts[1], expected_sig):
            raise ValueError("bad signature")
        payload = json.loads(raw)
        if payload.get("exp", 0) < time.time():
            raise ValueError("state expired")
        return payload["user_id"]
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid or expired OAuth state: {exc}",
        )


# ── Routes ─────────────────────────────────────────────────────────────


@router.get("/providers")
async def list_providers() -> list[dict]:
    """
    List all available connector providers and their configuration status.
    No auth required — used by frontend to show available connectors.
    """
    registry = ConnectorRegistry()
    return registry.list_providers()


@router.get("/connections")
async def list_connections(
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(db_session),
) -> list[dict]:
    """List all OAuth connections for the authenticated user."""
    return await get_user_connections(user_id, db_session=session)


@router.get("/{provider}/auth-url")
async def get_auth_url(
    provider: str,
    user_id: str = Depends(get_current_user_id),
) -> Dict[str, str]:
    """
    Get the OAuth authorization URL for a provider.

    Frontend should open this URL in a popup window.
    """
    registry = ConnectorRegistry()
    connector = registry.get(provider)
    if not connector:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Provider '{provider}' not found or not configured",
        )

    state = _create_state(user_id)
    auth_url = connector.get_auth_url(state)

    return {"auth_url": auth_url, "provider": provider}


@router.get("/{provider}/callback")
async def oauth_callback(
    provider: str,
    code: str = Query(...),
    state: str = Query(...),
    session: AsyncSession = Depends(db_session),
) -> HTMLResponse:
    """
    OAuth callback — Google (or other provider) redirects here after consent.

    Exchanges the auth code for tokens, stores them, and returns a small
    HTML page that notifies the opener window and auto-closes.
    """
    # 1. Verify state → get user_id
    user_id = _verify_state(state)

    # 2. Get connector
    registry = ConnectorRegistry()
    connector = registry.get(provider)
    if not connector:
        raise HTTPException(404, f"Provider '{provider}' not available")

    # 3. Exchange code for tokens
    try:
        token_data = await connector.handle_callback(code)
    except Exception as exc:
        logger.error("OAuth callback failed for %s: %s", provider, exc)
        return HTMLResponse(
            content=_callback_html(
                success=False,
                message=f"Connection failed: {exc}",
                provider=provider,
            ),
            status_code=200,
        )

    # 4. Store connection
    await store_connection(user_id, provider, token_data, db_session=session)
    await session.commit()

    account_label = token_data.get("account_label", provider)
    logger.info("OAuth connected: user=%s provider=%s account=%s", user_id, provider, account_label)
    return HTMLResponse(
        content=_callback_html(
            success=True,
            message=f"Connected {connector.display_name} as {account_label}",
            provider=provider,
        ),
        status_code=200,
    )


@router.delete("/connections/{connection_id}")
async def delete_connection(
    connection_id: str,
    user_id: str = Depends(get_current_user_id),
    session: AsyncSession = Depends(db_session),
) -> Dict[str, Any]:
    """Disconnect and revoke an OAuth connection."""
    deleted = await disconnect(user_id, connection_id, db_session=session)
    if not deleted:
        raise HTTPException(404, "Connection not found")
    await session.commit()
    return {"status": "disconnected", "connection_id": connection_id}


# ── Callback HTML template ─────────────────────────────────────────────


def _callback_html(success: bool, message: str, provider: str) -> str:
    """
    Small HTML page shown in the OAuth popup after redirect.
    Sends a postMessage to the opener and auto-closes.
    """
    status_emoji = "✅" if success else "❌"
    status_text = "Connected!" if success else "Failed"
    color = "#00d992" if success else "#ef4444"

    return f"""<!DOCTYPE html>
<html>
<head>
    <title>MRAG — {provider} {status_text}</title>
    <style>
        body {{
            font-family: 'Inter', system-ui, sans-serif;
            background: #0b0d11; color: #e4e7ee;
            display: flex; align-items: center; justify-content: center;
            height: 100vh; margin: 0;
        }}
        .card {{
            text-align: center; padding: 40px;
            background: #12151b; border: 1px solid #1f2330;
            border-radius: 12px; max-width: 400px;
        }}
        .emoji {{ font-size: 3rem; }}
        h2 {{ color: {color}; margin: 16px 0 8px; }}
        p {{ color: #a0a6b8; font-size: 0.85rem; }}
        .close-note {{ color: #636a80; font-size: 0.7rem; margin-top: 20px; }}
    </style>
</head>
<body>
    <div class="card">
        <div class="emoji">{status_emoji}</div>
        <h2>{status_text}</h2>
        <p>{message}</p>
        <p class="close-note">This window will close automatically…</p>
    </div>
    <script>
        // Notify the opener window
        if (window.opener) {{
            window.opener.postMessage({{
                type: 'oauth-callback',
                provider: '{provider}',
                success: {'true' if success else 'false'},
                message: '{message}',
            }}, '*');
        }}
        // Auto-close after 2 seconds
        setTimeout(() => window.close(), 2000);
    </script>
</body>
</html>"""
