"""This module re-exports the User model from the database package for use in authentication-related code.
"""

from database.models import User  # noqa: F401

__all__ = ["User"]
