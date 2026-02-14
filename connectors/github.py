"""
GitHubConnector — OAuth2 for GitHub API access.

Uses GitHub App OAuth flow to get per-user tokens with auto-refresh.
Supports repo management, issues, PRs via the REST API.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List
from urllib.parse import urlencode

import httpx

from config.settings import config
from connectors.base import BaseConnector

logger = logging.getLogger(__name__)

# GitHub OAuth2 endpoints
_GH_AUTH_URL = "https://github.com/login/oauth/authorize"
_GH_TOKEN_URL = "https://github.com/login/oauth/access_token"
_GH_API = "https://api.github.com"


class GitHubConnector(BaseConnector):
    """OAuth2 connector for GitHub."""

    @property
    def provider_name(self) -> str:
        return "github"

    @property
    def display_name(self) -> str:
        return "GitHub"

    @property
    def scopes(self) -> List[str]:
        return ["repo", "read:user", "user:email"]

    @property
    def icon(self) -> str:
        return "🐙"

    def is_configured(self) -> bool:
        return bool(config.github_client_id and config.github_client_secret)

    def _redirect_uri(self) -> str:
        return f"{config.oauth_redirect_base}/api/v1/connectors/github/callback"

    def get_auth_url(self, state: str) -> str:
        params = {
            "client_id": config.github_client_id,
            "redirect_uri": self._redirect_uri(),
            "scope": " ".join(self.scopes),
            "state": state,
        }
        return f"{_GH_AUTH_URL}?{urlencode(params)}"

    async def handle_callback(self, code: str) -> Dict[str, Any]:
        """Exchange auth code for tokens and fetch user profile."""
        async with httpx.AsyncClient() as client:
            # 1. Exchange code for token
            token_resp = await client.post(
                _GH_TOKEN_URL,
                data={
                    "client_id": config.github_client_id,
                    "client_secret": config.github_client_secret,
                    "code": code,
                    "redirect_uri": self._redirect_uri(),
                },
                headers={"Accept": "application/json"},
            )
            token_resp.raise_for_status()
            token_data = token_resp.json()

            if "error" in token_data:
                raise ValueError(
                    f"GitHub OAuth error: {token_data.get('error_description', token_data['error'])}"
                )

            # 2. Fetch user profile
            user_resp = await client.get(
                f"{_GH_API}/user",
                headers={
                    "Authorization": f"Bearer {token_data['access_token']}",
                    "Accept": "application/vnd.github+json",
                },
            )
            user_resp.raise_for_status()
            user = user_resp.json()

        return {
            "access_token": token_data["access_token"],
            "refresh_token": token_data.get("refresh_token"),
            "expires_in": token_data.get("expires_in", 28800),  # 8h for GitHub Apps
            "scopes": token_data.get("scope", "").split(","),
            "account_id": str(user.get("id", "")),
            "account_label": user.get("login", ""),
            "provider_meta": {
                "login": user.get("login"),
                "name": user.get("name"),
                "avatar_url": user.get("avatar_url"),
                "email": user.get("email"),
            },
        }

    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh the access token using a GitHub App refresh token.

        Note: Only GitHub Apps with "Expire user authorization tokens"
        enabled provide refresh tokens. Classic OAuth tokens don't expire.
        """
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                _GH_TOKEN_URL,
                data={
                    "client_id": config.github_client_id,
                    "client_secret": config.github_client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()

            if "error" in data:
                raise ValueError(
                    f"GitHub token refresh error: {data.get('error_description', data['error'])}"
                )

        return {
            "access_token": data["access_token"],
            "refresh_token": data.get("refresh_token"),
            "expires_in": data.get("expires_in", 28800),
        }

    async def revoke_token(self, access_token: str) -> bool:
        """Revoke the token via GitHub's OAuth application API."""
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.delete(
                    f"{_GH_API}/applications/{config.github_client_id}/token",
                    auth=(config.github_client_id, config.github_client_secret),
                    json={"access_token": access_token},
                )
                return resp.status_code == 204
        except Exception:
            logger.warning("GitHub token revocation failed", exc_info=True)
            return False
