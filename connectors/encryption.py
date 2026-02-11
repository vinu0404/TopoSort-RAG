"""
Token encryption — encrypt / decrypt OAuth tokens at rest.

Uses Fernet (AES-128-CBC + HMAC-SHA256) from the ``cryptography`` library.
The encryption key is loaded from ``config.token_encryption_key``
(env var: ``TOKEN_ENCRYPTION_KEY``).

If no key is configured, encryption is **disabled** and tokens are stored
as plaintext (with a startup warning).  Generate a key with::

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

import logging

from config.settings import config

logger = logging.getLogger(__name__)
from cryptography.fernet import Fernet

_fernet = None
_enabled = False


def _init_fernet() -> None:
    """Lazy-initialise the Fernet cipher once."""
    global _fernet, _enabled

    key = config.token_encryption_key
    if not key:
        logger.warning(
            "TOKEN_ENCRYPTION_KEY not set — OAuth tokens will be stored as plaintext. "
            "Generate a key: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        )
        _enabled = False
        return

    try:

        _fernet = Fernet(key.encode() if isinstance(key, str) else key)
        _enabled = True
        logger.info("Token encryption enabled (Fernet/AES-128-CBC)")
    except Exception as exc:
        logger.error("Failed to initialise Fernet with provided key: %s", exc)
        _enabled = False


def encrypt_token(plaintext: str) -> str:
    """
    Encrypt a token string for database storage.

    Returns the Fernet ciphertext (URL-safe base64).
    If encryption is disabled, returns the plaintext unchanged.
    """
    if _fernet is None:
        _init_fernet()

    if not _enabled or _fernet is None:
        return plaintext

    return _fernet.encrypt(plaintext.encode()).decode()


def decrypt_token(ciphertext: str) -> str:
    """
    Decrypt a token string read from the database.

    If encryption is disabled, returns the input unchanged.
    Handles graceful fallback for tokens stored before encryption
    was enabled (they won't be valid Fernet tokens and are returned as-is).
    """
    if _fernet is None:
        _init_fernet()

    if not _enabled or _fernet is None:
        return ciphertext

    try:
        return _fernet.decrypt(ciphertext.encode()).decode()
    except Exception:
        return ciphertext


def is_encryption_enabled() -> bool:
    """Check whether token encryption is active."""
    if _fernet is None:
        _init_fernet()
    return _enabled
