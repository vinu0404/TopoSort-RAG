#!/usr/bin/env bash
set -e

# Start Celery in background for demo mode
celery -A celery_app worker --loglevel=info --pool=threads --concurrency=2 &

# Run FastAPI in foreground (Render expects one foreground process)
uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
