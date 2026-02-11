"""
BaseConnector — abstract interface for all OAuth2 connectors.

Every provider (Gmail, Slack, Notion, …) subclasses this and implements
the four core methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseConnector(ABC):
    """Abstract base for all OAuth2 connectors."""

    # ── Identity ────────────────────────────────────────────────────────
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Unique slug: 'gmail', 'slack', 'notion', 'github'."""
        ...

    @property
    @abstractmethod
    def display_name(self) -> str:
        """Human-readable: 'Gmail', 'Slack', 'Notion', 'GitHub'."""
        ...

    @property
    @abstractmethod
    def scopes(self) -> List[str]:
        """OAuth scopes required by this connector."""
        ...

    @property
    def icon(self) -> str:
        """Optional emoji / icon for UI."""
        return "🔗"

    # ── OAuth flow ──────────────────────────────────────────────────────

    @abstractmethod
    def get_auth_url(self, state: str) -> str:
        """
        Build the provider's OAuth2 authorization URL.

        Parameters
        ----------
        state : str
            Opaque state string (encodes user_id + CSRF token).

        Returns
        -------
        The full URL to redirect the user to.
        """
        ...

    @abstractmethod
    async def handle_callback(self, code: str) -> Dict[str, Any]:
        """
        Exchange the authorization code for tokens.

        Parameters
        ----------
        code : str
            Authorization code from the OAuth redirect.

        Returns
        -------
        dict with keys:
            access_token, refresh_token, expires_in, scopes,
            account_id, account_label, provider_meta
        """
        ...

    @abstractmethod
    async def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an expired access token.

        Returns
        -------
        dict with keys: access_token, expires_in, (optional) refresh_token
        """
        ...

    async def revoke_token(self, access_token: str) -> bool:
        """
        Revoke the token at the provider (optional).
        Returns True on success, False if provider doesn't support revocation.
        """
        return False

    # ── Helpers ─────────────────────────────────────────────────────────

    def is_configured(self) -> bool:
        """
        Return True if this connector has all required config
        (API keys, client IDs, etc.).
        """
        return True
