"""
Password hashing and verification.

Uses bcrypt for password hashing with automatic
salting and configurable work factor.
"""

from __future__ import annotations

import bcrypt


def hash_password(password: str) -> str:
    """Hash a password with bcrypt (auto-salted, work factor 12)."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt(rounds=12)).decode()


def verify_password(password: str, password_hash: str) -> bool:
    """Constant-time comparison against a bcrypt hash."""
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except (ValueError, TypeError):
        return False
