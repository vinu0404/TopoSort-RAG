"""
Global middleware — request timer + security layers.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request

from security.middleware import register_security_middleware

logger = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> None:
    """Attach all app-level middleware."""

    # Security middleware (headers, error sanitization)
    register_security_middleware(app)

    @app.middleware("http")
    async def request_timer(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        logger.debug("%s %s — %.3fs", request.method, request.url.path, elapsed)
        return response
