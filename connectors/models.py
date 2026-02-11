"""
This module re-exports the UserConnection model from the database package for use in connector-related code.
"""

from database.models import UserConnection  # noqa: F401

__all__ = ["UserConnection"]
