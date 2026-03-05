"""
Celery task for document processing.

Runs in a separate worker process.  Uses ``asyncio.run()`` to bridge
the sync Celery world with the async document pipeline.
Publishes status updates via Redis Pub/Sub so the SSE endpoint
can push them to clients in real-time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict

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


def _publish_status(user_id: str, doc_id: str, status: str, **extra: Any) -> None:
    """Publish a status event to the user's Redis Pub/Sub channel."""
    channel = f"doc_status:{user_id}"
    payload = {"doc_id": doc_id, "processing_status": status, **extra}
    _get_redis().publish(channel, json.dumps(payload))


@celery.task(
    bind=True,
    name="tasks.process_document",
    autoretry_for=(Exception,),
    dont_autoretry_for=(SoftTimeLimitExceeded, ValueError),
    max_retries=3,
    retry_backoff=True,         
    retry_backoff_max=120,      
    retry_jitter=True,           
    acks_late=True,
    rate_limit="20/m",         
)
def process_document_task(
    self: Task,
    user_id: str,
    doc_id: str,
    filename: str,
    file_bytes_hex: str,        
) -> Dict[str, Any]:
    """
    Celery task entry point.  Bridges sync → async via asyncio.run().
    """
    return asyncio.run(
        _process_document_async(self, user_id, doc_id, filename, file_bytes_hex)
    )


async def _process_document_async(
    task: Task,
    user_id: str,
    doc_id: str,
    filename: str,
    file_bytes_hex: str,
) -> Dict[str, Any]:
    """Async implementation of the document processing pipeline."""
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from database.helpers import update_document_status
    from document_pipeline.document_processor import process_document

    file_bytes = bytes.fromhex(file_bytes_hex)

    # Create a fresh engine per task — each asyncio.run() creates a new
    # event loop so we can't re-use the module-level engine/pool.
    _engine = create_async_engine(
        config.database_url, echo=False, pool_size=2, max_overflow=2,
        pool_recycle=60,
    )
    _session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False,
    )

    async with _session_factory() as session:
        try:
            await update_document_status(session, doc_id, "processing")
            await session.commit()
            _publish_status(user_id, doc_id, "processing", filename=filename)

            result = await process_document(
                user_id=user_id,
                file_path=filename,
                file_bytes=file_bytes,
            )
            await update_document_status(
                session,
                doc_id=doc_id,
                status="ready",
                description=result.get("description"),
                total_chunks=result.get("total_chunks"),
            )
            await session.commit()
            _publish_status(
                user_id, doc_id, "ready",
                filename=filename,
                total_chunks=result.get("total_chunks"),
                description=result.get("description"),
            )
            logger.info(
                "Document %s (%s) processed — %d chunks",
                doc_id, filename, result.get("total_chunks", 0),
            )
            return {
                "doc_id": doc_id,
                "filename": filename,
                "status": "ready",
                "total_chunks": result.get("total_chunks"),
            }

        except SoftTimeLimitExceeded:
            logger.warning("Soft time limit hit for doc %s", doc_id)
            await session.rollback()
            await update_document_status(
                session, doc_id, "failed",
                error_message="Processing timed out",
            )
            await session.commit()
            _publish_status(
                user_id, doc_id, "failed",
                filename=filename, error="Processing timed out",
            )
            raise

        except Exception as exc:
            logger.exception("Document processing failed for %s (attempt %d/%d)",
                             doc_id, task.request.retries + 1, task.max_retries + 1)

            if task.request.retries >= task.max_retries:
                try:
                    await session.rollback()
                    await update_document_status(
                        session, doc_id, "failed",
                        error_message=str(exc)[:500],
                    )
                    await session.commit()
                    _publish_status(
                        user_id, doc_id, "failed",
                        filename=filename, error=str(exc)[:200],
                    )
                except Exception:
                    logger.exception("Failed to persist error for doc %s", doc_id)
            else:
                try:
                    await session.rollback()
                    await update_document_status(session, doc_id, "pending")
                    await session.commit()
                    _publish_status(
                        user_id, doc_id, "pending",
                        filename=filename,
                        retry=task.request.retries + 1,
                    )
                except Exception:
                    logger.exception("Failed to persist retry for doc %s", doc_id)

            raise 

        finally:
            await _engine.dispose()
