#!/usr/bin/env bash
set -e

# Start Celery worker in background (document processing + scheduled job execution)
python -m celery -A celery_app worker --loglevel=info --pool=threads --concurrency=4 &

# Start Celery Beat in background (cron triggers for scheduled jobs)
python -m celery -A celery_app beat --scheduler=redbeat.RedBeatScheduler --loglevel=info &

# Run FastAPI in foreground (Render expects one foreground process)
python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
