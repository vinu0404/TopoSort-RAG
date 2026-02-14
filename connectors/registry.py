"""
ConnectorRegistry — discovers and provides access to all connectors.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from connectors.base import BaseConnector
from connectors.github import GitHubConnector
from connectors.gmail import GmailConnector

logger = logging.getLogger(__name__)

# ── All known connectors — add new ones here ─────────────────────────────

_ALL_CONNECTORS: List[BaseConnector] = [
    GmailConnector(),
    GitHubConnector(),
    # SlackConnector(),    # future
    # NotionConnector(),   # future
]


class ConnectorRegistry:
    """Singleton registry for all OAuth connectors."""

    _instance: Optional["ConnectorRegistry"] = None

    def __new__(cls) -> "ConnectorRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._connectors = {}
            cls._instance._discovered = False
        return cls._instance

    def discover(self) -> None:
        """Register all configured connectors."""
        if self._discovered:
            return
        for conn in _ALL_CONNECTORS:
            if conn.is_configured():
                self._connectors[conn.provider_name] = conn
                logger.info(
                    "Connector registered: %s (%s)",
                    conn.display_name,
                    conn.provider_name,
                )
            else:
                logger.warning(
                    "Connector %s skipped — not configured (missing client_id/secret)",
                    conn.provider_name,
                )
        self._discovered = True

    def get(self, provider: str) -> Optional[BaseConnector]:
        """Get a connector by provider name."""
        return self._connectors.get(provider)

    def list_providers(self) -> List[Dict[str, str]]:
        """Return info about all available connectors."""
        return [
            {
                "provider": c.provider_name,
                "display_name": c.display_name,
                "icon": c.icon,
                "configured": c.is_configured(),
            }
            for c in _ALL_CONNECTORS
        ]

    def list_configured(self) -> List[str]:
        """Return names of configured connectors."""
        return list(self._connectors.keys())
