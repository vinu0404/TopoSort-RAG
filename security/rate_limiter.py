"""
Redis-backed sliding-window rate limiter.

Uses ``INCR`` + ``EXPIRE`` on a Redis key per (user, category) pair
for atomic, distributed rate counting.

Usage as a FastAPI dependency::

    from security.rate_limiter import check_rate_limit

    @router.post("/stream")
    async def stream(
        request: Request,
        _rl: None = Depends(lambda r: check_rate_limit(r, "streaming")),
    ):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import redis.asyncio as aioredis
from fastapi import HTTPException, Request

from config.settings import config

logger = logging.getLogger("security.rate_limiter")


@dataclass(frozen=True)
class RateLimitConfig:
    """Rule: *requests* allowed within *window_seconds*."""

    requests: int
    window_seconds: int


# ── Default limits per category ───────────────────────────────────────────
DEFAULT_LIMITS: dict[str, RateLimitConfig] = {
    "standard":       RateLimitConfig(requests=100, window_seconds=60),
    "streaming":      RateLimitConfig(requests=20,  window_seconds=60),
    "code_execution": RateLimitConfig(requests=10,  window_seconds=60),
    "auth":           RateLimitConfig(requests=10,  window_seconds=60),
}

# ── Lazy Redis connection ─────────────────────────────────────────────────
_redis: aioredis.Redis | None = None


async def _get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.from_url(
            config.redis_url,
            decode_responses=True,
            socket_connect_timeout=2,
        )
    return _redis


# ── Core check ────────────────────────────────────────────────────────────

async def _check(
    user_id: str,
    category: str,
    cfg: RateLimitConfig,
) -> tuple[bool, int, int]:
    """
    Returns ``(allowed, remaining, retry_after_seconds)``.

    Uses a single Redis key ``rate:{user_id}:{category}`` that auto-expires
    after the time window.
    """
    r = await _get_redis()
    key = f"rate:{user_id}:{category}"

    try:
        current = await r.incr(key)
        if current == 1:
            # First request in window — set the TTL
            await r.expire(key, cfg.window_seconds)

        if current > cfg.requests:
            ttl = await r.ttl(key)
            retry_after = max(ttl, 1)
            return False, 0, retry_after

        remaining = cfg.requests - current
        return True, remaining, 0

    except aioredis.RedisError:
        # If Redis is down, fail open so the app stays available.
        logger.exception("Redis rate-limit check failed — allowing request")
        return True, cfg.requests, 0


# ── FastAPI dependency ────────────────────────────────────────────────────

async def check_rate_limit(
    request: Request,
    category: str = "standard",
) -> None:
    """
    FastAPI-style dependency — call via ``Depends()``.

    Resolves the user id from ``request.state.user_id`` (set by the auth
    middleware).  Falls back to the client IP for unauthenticated endpoints.
    """
    if not getattr(config, "rate_limit_enabled", True):
        return

    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        user_id = request.client.host if request.client else "unknown"

    cfg = DEFAULT_LIMITS.get(category, DEFAULT_LIMITS["standard"])
    allowed, remaining, retry_after = await _check(str(user_id), category, cfg)

    # Stash for the header middleware
    request.state.rate_limit_remaining = remaining

    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": f"Too many requests. Retry after {retry_after}s.",
                "retry_after": retry_after,
                "category": category,
            },
            headers={
                "Retry-After": str(retry_after),
                "X-RateLimit-Remaining": "0",
            },
        )
