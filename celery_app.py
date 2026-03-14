"""
Celery application factory.

Start a worker with:
    celery -A celery_app worker --loglevel=info --pool=solo

OR 

python -m celery -A celery_app worker --pool=threads --concurrency=4


Monitor with Flower:
    celery -A celery_app flower --port=5555
"""

from __future__ import annotations

import sys
from pathlib import Path
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
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    task_acks_late=True,                 
    worker_prefetch_multiplier=1,        
    task_reject_on_worker_lost=True,     
    result_expires=3600,                 
    task_soft_time_limit=300,           
    task_time_limit=360,                  
    task_default_retry_delay=10,
    task_max_retries=3,
    worker_concurrency=4,                 
)

celery.conf.update(
    include=["tasks.document_tasks", "tasks.web_scrape_tasks"],
)
