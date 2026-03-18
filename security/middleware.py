"""
Security middleware for FastAPI.

* **SecurityHeadersMiddleware** — adds CSP, X-Frame-Options, etc.
* **ErrorSanitizationMiddleware** — catches unhandled exceptions and
  returns a generic error body (no stack traces to the client).
"""

from __future__ import annotations

import logging
import traceback
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

logger = logging.getLogger("security.middleware")


# ── Security Headers ──────────────────────────────────────────────────────

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Inject standard security headers into every response."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        response = await call_next(request)

        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = (
            "geolocation=(), camera=()"
        )

        # Content-Security-Policy — tune per your frontend needs
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "media-src 'self' blob:; "
            "connect-src 'self' https: wss:; "
            "frame-ancestors 'none';"
        )

        # Rate-limit remaining (set by the rate_limiter dependency)
        remaining = getattr(request.state, "rate_limit_remaining", None)
        if remaining is not None:
            response.headers["X-RateLimit-Remaining"] = str(remaining)

        return response


# ── Error Sanitization ────────────────────────────────────────────────────

# Error types whose detail we can forward to the client as-is
_SAFE_ERROR_TYPES = {
    "validation_error",
    "not_found",
    "unauthorized",
    "forbidden",
    "rate_limit_exceeded",
    "bad_request",
}


class ErrorSanitizationMiddleware(BaseHTTPMiddleware):
    """
    Catch unhandled exceptions and return a generic JSON error.

    * Logs the full traceback internally (keyed by a short ``request_id``).
    * Returns ``{"error": "internal_error", "request_id": "…"}`` to the
      client, with no stack trace or path information.
    * Passes through known safe HTTP errors (4xx with expected detail).
    """

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        request_id = uuid.uuid4().hex[:8]
        request.state.request_id = request_id

        try:
            return await call_next(request)
        except Exception as exc:
            # Log everything server-side
            logger.error(
                "Unhandled exception [request_id=%s]: %s\n%s",
                request_id,
                exc,
                traceback.format_exc(),
            )

            # If 'exc' is an HTTPException with a safe detail dict, forward it.
            detail = getattr(exc, "detail", None)
            if isinstance(detail, dict) and detail.get("error") in _SAFE_ERROR_TYPES:
                return JSONResponse(
                    status_code=getattr(exc, "status_code", 400),
                    content=detail,
                )

            return JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "message": "An internal error occurred. Please try again.",
                    "request_id": request_id,
                },
            )


# ── Registration helper ──────────────────────────────────────────────────

def register_security_middleware(app: FastAPI) -> None:
    """
    Register all security middleware on *app*.

    Call this **after** CORS middleware so the security headers are added
    to responses coming back through the CORS layer.

    Ordering (outermost → innermost):
        ErrorSanitization → SecurityHeaders → …app routes…
    """
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(ErrorSanitizationMiddleware)
