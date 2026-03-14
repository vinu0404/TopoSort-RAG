"""
Celery task for web scraping collections.

Runs in a separate worker process.  Uses ``asyncio.run()`` to bridge
the sync Celery world with the async scraper pipeline.
Publishes status updates via Redis Pub/Sub so the SSE endpoint
can push them to clients in real-time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List

import redis
from celery import Task
from celery.exceptions import SoftTimeLimitExceeded

from celery_app import celery
from config.settings import config

logger = logging.getLogger(__name__)

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(config.redis_url, decode_responses=True)
    return _redis_client


def _publish_status(user_id: str, collection_id: str, **extra: Any) -> None:
    """Publish a status event to the user's web scrape Redis channel."""
    channel = f"web_scrape_status:{user_id}"
    payload = {"collection_id": collection_id, **extra}
    _get_redis().publish(channel, json.dumps(payload))


@celery.task(
    bind=True,
    name="tasks.scrape_web_collection",
    autoretry_for=(Exception,),
    dont_autoretry_for=(SoftTimeLimitExceeded, ValueError),
    max_retries=2,
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
    acks_late=True,
    rate_limit="10/m",
)
def scrape_web_collection_task(
    self: Task,
    user_id: str,
    collection_id: str,
    urls_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Celery task entry point.  Bridges sync -> async via asyncio.run().

    ``urls_data`` is a list of ``{"url_id": str, "url": str, "depth": int}``.
    """
    return asyncio.run(
        _scrape_async(self, user_id, collection_id, urls_data)
    )


async def _scrape_async(
    task: Task,
    user_id: str,
    collection_id: str,
    urls_data: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Async implementation: scrape each URL, chunk, embed, store."""
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from database.helpers import (
        update_web_collection_status,
        update_web_url_status,
    )
    from document_pipeline.web_scraper import scrape_url, process_scraped_pages

    _engine = create_async_engine(
        config.database_url, echo=False, pool_size=2, max_overflow=2,
        pool_recycle=60,
    )
    _session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False,
    )

    total_pages = 0
    total_chunks = 0
    url_results: Dict[str, str] = {}  # url_id -> "ready" | "failed"

    async with _session_factory() as session:
        try:
            # Mark collection as scraping
            await update_web_collection_status(session, collection_id, "scraping")
            await session.commit()
            _publish_status(user_id, collection_id, status="scraping")

            for entry in urls_data:
                url_id = entry["url_id"]
                url = entry["url"]
                depth = entry.get("depth", 1)

                try:
                    # Mark URL as scraping
                    await update_web_url_status(session, url_id, "scraping")
                    await session.commit()
                    _publish_status(
                        user_id, collection_id,
                        url_id=url_id, url_status="scraping",
                    )

                    # Scrape
                    pages = await scrape_url(url, depth=depth)

                    if not pages:
                        await update_web_url_status(
                            session, url_id, "failed",
                            error_message="No content could be scraped",
                        )
                        await session.commit()
                        _publish_status(
                            user_id, collection_id,
                            url_id=url_id, url_status="failed",
                            error="No content could be scraped",
                        )
                        url_results[url_id] = "failed"
                        continue

                    # Process: chunk + embed + store in Qdrant
                    result = await process_scraped_pages(
                        user_id=user_id,
                        web_collection_id=collection_id,
                        pages=pages,
                    )

                    pages_scraped = result["pages_processed"]
                    chunks_created = result["chunks_created"]
                    total_pages += pages_scraped
                    total_chunks += chunks_created

                    await update_web_url_status(
                        session, url_id, "ready",
                        pages_scraped=pages_scraped,
                        chunks_created=chunks_created,
                    )
                    await session.commit()
                    _publish_status(
                        user_id, collection_id,
                        url_id=url_id, url_status="ready",
                        pages_scraped=pages_scraped,
                        chunks_created=chunks_created,
                    )
                    url_results[url_id] = "ready"

                    logger.info(
                        "URL %s scraped: %d pages, %d chunks",
                        url, pages_scraped, chunks_created,
                    )

                except Exception as e:
                    logger.exception("Failed to scrape URL %s", url)
                    try:
                        await session.rollback()
                        await update_web_url_status(
                            session, url_id, "failed",
                            error_message=str(e)[:500],
                        )
                        await session.commit()
                    except Exception:
                        logger.exception("Failed to persist error for url %s", url_id)
                    _publish_status(
                        user_id, collection_id,
                        url_id=url_id, url_status="failed",
                        error=str(e)[:200],
                    )
                    url_results[url_id] = "failed"

            # Determine final collection status
            statuses = set(url_results.values())
            if statuses == {"ready"}:
                final_status = "ready"
            elif statuses == {"failed"}:
                final_status = "failed"
            else:
                final_status = "partial"

            await update_web_collection_status(
                session, collection_id, final_status,
                total_pages=total_pages,
                total_chunks=total_chunks,
            )
            await session.commit()
            _publish_status(
                user_id, collection_id,
                status=final_status,
                total_pages=total_pages,
                total_chunks=total_chunks,
            )

            logger.info(
                "Web scrape collection %s done: status=%s pages=%d chunks=%d",
                collection_id, final_status, total_pages, total_chunks,
            )
            return {
                "collection_id": collection_id,
                "status": final_status,
                "total_pages": total_pages,
                "total_chunks": total_chunks,
            }

        except SoftTimeLimitExceeded:
            logger.warning("Soft time limit for web scrape collection %s", collection_id)
            await session.rollback()
            await update_web_collection_status(
                session, collection_id, "failed",
                error_message="Scraping timed out",
            )
            await session.commit()
            _publish_status(
                user_id, collection_id,
                status="failed", error="Scraping timed out",
            )
            raise

        except Exception as exc:
            logger.exception("Web scrape collection %s failed", collection_id)
            if task.request.retries >= task.max_retries:
                try:
                    await session.rollback()
                    await update_web_collection_status(
                        session, collection_id, "failed",
                        error_message=str(exc)[:500],
                    )
                    await session.commit()
                    _publish_status(
                        user_id, collection_id,
                        status="failed", error=str(exc)[:200],
                    )
                except Exception:
                    logger.exception("Failed to persist error for collection %s", collection_id)
            raise

        finally:
            await _engine.dispose()
