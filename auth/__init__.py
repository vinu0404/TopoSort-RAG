"""
auth — User authentication module.

Provides:
  • JWT token creation & verification
  • Password hashing (SHA-256 with salt)
  • Register / Login API routes
  • ``get_current_user_id`` FastAPI dependency
"""
