"""
GmailConnector — OAuth2 web flow for Gmail.

Uses Google's OAuth2 to get per-user Gmail access without the user
sharing any credentials with the application.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from urllib.parse import urlencode

import httpx

from config.settings import config
from connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# Google OAuth2 endpoints
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
_GOOGLE_USERINFO_URL = "https://www.googleapis.com/oauth2/v2/userinfo"


class GmailConnector(BaseConnector):
    """OAuth2 connector for Gmail."""

    @property
    def provider_name(self) -> str:
        return "gmail"

    @property
    def display_name(self) -> str:
        return "Gmail"

    @property
    def scopes(self) -> List[str]:
        return [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.send",
            "https://www.googleapis.com/auth/gmail.compose",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/userinfo.email",
        ]

    @property
    def icon(self) -> str:
        return "📧"

    def is_configured(self) -> bool:
        return bool(config.google_client_id and config.google_client_secret)

    def _redirect_uri(self) -> str:
        return f"{config.oauth_redirect_base}/api/v1/connectors/gmail/callback"

    def get_auth_url(self, state: str) -> str:
        params = {
            "client_id": config.google_client_id,
            "redirect_uri": self._redirect_uri(),
            "response_type": "code",
            "scope": " ".join(self.scopes),
            "access_type": "offline",       # gets refresh_token
            "prompt": "consent",            # force consent to always get refresh_token
            "state": state,
        }
        return f"{_GOOGLE_AUTH_URL}?{urlencode(params)}"

    async def handle_callback(self, code: str) -> Dict[str, Any]:
        """Exchange auth code for tokens."""
        async with httpx.AsyncClient() as client:
            # 1. Exchange code for tokens
            token_resp = await client.post(
                _GOOGLE_TOKEN_URL,
                data={
                    "code": code,
                    "client_id": config.google_client_id,
                    "client_secret": config.google_client_secret,
                    "redirect_uri": self._redirect_uri(),
                    "grant_type": "authorization_code",
                },
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            # 2. Fetch user info to get email (account_label)
            headers = {"Authorization": f"Bearer {token_data['access_token']}"}
            user_resp = await client.get(_GOOGLE_USERINFO_URL, headers=headers)
            user_resp.raise_for_status()
            user_info = user_resp.json()

        return {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_in": token_data.get("expires_in", 3600),
            "scopes": token_data.get("scope", "").split(),
            "account_id": user_info.get("id", user_info.get("email", "")),
            "account_label": user_info.get("email", ""),
            "provider_meta": {
                "email": user_info.get("email"),
                "name": user_info.get("name"),
                "picture": user_info.get("picture"),
            },
        }

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """Use refresh token to get a new access token."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _GOOGLE_TOKEN_URL,
                data={
                    "client_id": config.google_client_id,
                    "client_secret": config.google_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            data = resp.json()

        return {
            "access_token": data["access_token"],
            "expires_in": data.get("expires_in", 3600),
            "refresh_token": data.get("refresh_token"), 
        }

    async def revoke_token(self, access_token: str) -> bool:
        """Revoke the token at Google."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    _GOOGLE_REVOKE_URL,
                    params={"token": access_token},
                )
                return resp.status_code == 200
        except Exception:
            return False
