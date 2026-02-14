"""
connectors — OAuth integration module for external services.

Provides a generic connector framework that handles:
  • OAuth2 auth-URL generation
  • Callback handling (code → token exchange)
  • Per-user token storage & auto-refresh
  • Fernet encryption of tokens at rest
  • Revocation / disconnect

Each provider (Gmail, Slack, …) is a subclass of BaseConnector.
"""
