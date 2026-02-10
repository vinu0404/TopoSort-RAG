"""
Global middleware.
"""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request

logger = logging.getLogger(__name__)


def register_middleware(app: FastAPI) -> None:
    """Attach any app-level middleware."""

    @app.middleware("http")
    async def request_timer(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        logger.debug("%s %s — %.3fs", request.method, request.url.path, elapsed)
        return response
