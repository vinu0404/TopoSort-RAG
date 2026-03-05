"""
Celery application factory.

Start a worker with:
    celery -A celery_app worker --loglevel=info --pool=solo

Monitor with Flower:
    celery -A celery_app flower --port=5555
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so "tasks", "config", etc. are importable
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from celery import Celery

from config.settings import config

celery = Celery(
    "mrag",
    broker=config.celery_broker_url,
    backend=config.celery_result_backend,
)

celery.conf.update(
    # ── Serialisation ────────────────────────────────────────────────────
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",

    # ── Reliability ──────────────────────────────────────────────────────
    task_acks_late=True,                  # ack only after task completes
    worker_prefetch_multiplier=1,         # one task at a time per worker
    task_reject_on_worker_lost=True,      # re-queue if worker crashes

    # ── Result expiry ────────────────────────────────────────────────────
    result_expires=3600,                  # discard results after 1 hour

    # ── Time limits ──────────────────────────────────────────────────────
    task_soft_time_limit=300,             # 5 min soft limit
    task_time_limit=360,                  # 6 min hard kill

    # ── Retry defaults ───────────────────────────────────────────────────
    task_default_retry_delay=10,
    task_max_retries=3,

    # ── Worker ───────────────────────────────────────────────────────────
    worker_concurrency=4,                 # parallel tasks per worker
)

# Explicitly include task modules instead of autodiscover
celery.conf.update(
    include=["tasks.document_tasks"],
)
