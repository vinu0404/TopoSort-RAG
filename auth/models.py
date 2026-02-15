"""This module re-exports the User model from the database package for use in authentication-related code.
"""

from database.models import User

__all__ = ["User"]
