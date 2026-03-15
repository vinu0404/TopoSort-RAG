"""
FastAPI router for Scheduled Jobs CRUD, trigger, and run history.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import db_session, get_current_user_id
from database.helpers import (
    create_scheduled_job,
    delete_scheduled_job,
    get_scheduled_job,
    get_scheduled_job_run_detail,
    get_scheduled_job_runs,
    get_scheduled_jobs_for_user,
    update_scheduled_job,
)
from tasks.schedule_sync import remove_job_from_beat, sync_job_to_beat
from utils.schemas import (
    CRON_PRESETS,
    ScheduledJobCreate,
    ScheduledJobNLCreate,
    ScheduledJobUpdate,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/scheduled-jobs", tags=["scheduled-jobs"])


def _cron_to_human(expr: str) -> str:
    """Best-effort human-readable description of a cron expression."""
    try:
        from croniter import croniter
        # Validate
        croniter(expr)
    except Exception:
        return expr

    # Check presets first
    for name, cron in CRON_PRESETS.items():
        if cron == expr:
            return name.replace("_", " ").title()

    parts = expr.split()
    if len(parts) != 5:
        return expr

    minute, hour, dom, month, dow = parts
    desc = []

    dow_names = {
        "0": "Sun", "1": "Mon", "2": "Tue", "3": "Wed",
        "4": "Thu", "5": "Fri", "6": "Sat", "7": "Sun",
    }

    if dow != "*":
        if "-" in dow:
            start, end = dow.split("-")
            desc.append(f"{dow_names.get(start, start)}-{dow_names.get(end, end)}")
        else:
            desc.append(dow_names.get(dow, f"day {dow}"))

    if dom != "*":
        desc.append(f"day {dom} of month")

    if month != "*":
        desc.append(f"month {month}")

    if hour != "*" and minute != "*":
        time_parts = []
        for h in hour.split(","):
            time_parts.append(f"{h}:{minute.zfill(2)}")
        desc.append(f"at {', '.join(time_parts)}")
    elif hour == "*":
        desc.append("every hour")

    return " ".join(desc) if desc else expr


# ── CRUD ─────────────────────────────────────────────────────────────


@router.post("")
async def create_job(
    request: ScheduledJobCreate,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Create a new scheduled job with explicit steps."""
    # Validate cron expression
    try:
        from croniter import croniter
        croniter(request.cron_expression)
    except Exception:
        raise HTTPException(400, f"Invalid cron expression: {request.cron_expression}")

    # Validate step dependencies (no cycles, valid indices)
    for i, step in enumerate(request.steps):
        for dep in step.depends_on_steps:
            if dep < 0 or dep >= len(request.steps) or dep >= i:
                raise HTTPException(400, f"Step {i} has invalid dependency index: {dep}")

    steps_data = [s.model_dump() for s in request.steps]
    job = await create_scheduled_job(
        session,
        user_id=user_id,
        name=request.name,
        description=request.description,
        cron_expression=request.cron_expression,
        timezone_str=request.timezone,
        steps_data=steps_data,
        notification_mode=request.notification_mode,
        notification_target=request.notification_target,
    )
    await session.commit()

    # Sync to Celery Beat
    sync_job_to_beat(job["job_id"], request.cron_expression, request.timezone)

    job["cron_human"] = _cron_to_human(request.cron_expression)
    return job


@router.post("/from-prompt")
async def create_job_from_prompt(
    request: ScheduledJobNLCreate,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Parse natural language into a scheduled job plan (preview)."""
    from config.model_list import AgentRegistry
    from config.settings import config as app_config
    from core.master_agent import MasterAgent
    from utils.llm_providers import get_llm_provider

    provider = get_llm_provider(app_config.master_model_provider)
    registry = AgentRegistry()
    master = MasterAgent(agent_registry=registry, llm_provider=provider)

    parsed = await master.plan_scheduled_job(request.prompt, user_id)
    if parsed is None:
        raise HTTPException(422, "Could not parse scheduled job from prompt")

    parsed["cron_human"] = _cron_to_human(parsed.get("cron_expression", ""))
    parsed["timezone"] = request.timezone
    parsed["notification_mode"] = request.notification_mode
    return {"preview": parsed}


@router.get("")
async def list_jobs(
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """List all scheduled jobs for the authenticated user."""
    jobs = await get_scheduled_jobs_for_user(session, user_id)
    for j in jobs:
        j["cron_human"] = _cron_to_human(j.get("cron_expression", ""))
    return jobs


@router.get("/presets")
async def get_presets():
    """Return available cron presets for the UI."""
    return {
        name: {"cron": cron, "label": name.replace("_", " ").title()}
        for name, cron in CRON_PRESETS.items()
    }


@router.get("/status/stream")
async def scheduled_job_status_stream(
    request: Request,
    user_id: str = Depends(get_current_user_id),
):
    """SSE endpoint for real-time scheduled job status updates."""
    import redis.asyncio as aioredis
    from config.settings import config as app_config

    channel_name = f"scheduled_job:{user_id}"

    async def _event_stream():
        r = aioredis.from_url(app_config.redis_url, decode_responses=True)
        pubsub = r.pubsub()
        await pubsub.subscribe(channel_name)

        try:
            while True:
                if await request.is_disconnected():
                    break
                msg = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0,
                )
                if msg and msg["type"] == "message":
                    yield f"event: scheduled_job_status\ndata: {msg['data']}\n\n"
                else:
                    yield ": heartbeat\n\n"
                    await asyncio.sleep(2)
        finally:
            await pubsub.unsubscribe(channel_name)
            await pubsub.close()
            await r.close()

    return StreamingResponse(
        _event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{job_id}")
async def get_job(
    job_id: str,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Get a scheduled job with its steps."""
    job = await get_scheduled_job(session, job_id, user_id)
    if job is None:
        raise HTTPException(404, "Scheduled job not found")
    job["cron_human"] = _cron_to_human(job.get("cron_expression", ""))
    return job


@router.put("/{job_id}")
async def update_job(
    job_id: str,
    request: ScheduledJobUpdate,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Update a scheduled job."""
    kwargs = request.model_dump(exclude_none=True)

    if "cron_expression" in kwargs:
        try:
            from croniter import croniter
            croniter(kwargs["cron_expression"])
        except Exception:
            raise HTTPException(400, f"Invalid cron expression: {kwargs['cron_expression']}")

    job = await update_scheduled_job(session, job_id, user_id, **kwargs)
    if job is None:
        raise HTTPException(404, "Scheduled job not found")
    await session.commit()

    # Re-sync beat if schedule or status changed
    if "cron_expression" in kwargs or "status" in kwargs:
        if job["status"] == "active":
            sync_job_to_beat(job_id, job["cron_expression"], job.get("timezone", "UTC"))
        else:
            remove_job_from_beat(job_id)

    job["cron_human"] = _cron_to_human(job.get("cron_expression", ""))
    return job


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Soft-delete a scheduled job."""
    ok = await delete_scheduled_job(session, job_id, user_id)
    if not ok:
        raise HTTPException(404, "Scheduled job not found")
    await session.commit()
    remove_job_from_beat(job_id)
    return {"status": "deleted"}


@router.post("/{job_id}/pause")
async def pause_job(
    job_id: str,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Pause a scheduled job."""
    job = await update_scheduled_job(session, job_id, user_id, status="paused")
    if job is None:
        raise HTTPException(404, "Scheduled job not found")
    await session.commit()
    remove_job_from_beat(job_id)
    return {"status": "paused"}


@router.post("/{job_id}/resume")
async def resume_job(
    job_id: str,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Resume a paused scheduled job."""
    job = await update_scheduled_job(session, job_id, user_id, status="active")
    if job is None:
        raise HTTPException(404, "Scheduled job not found")
    await session.commit()
    sync_job_to_beat(job_id, job["cron_expression"], job.get("timezone", "UTC"))
    return {"status": "active"}


@router.post("/{job_id}/trigger")
async def trigger_job(
    job_id: str,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Manually trigger a scheduled job (run now)."""
    job = await get_scheduled_job(session, job_id, user_id)
    if job is None:
        raise HTTPException(404, "Scheduled job not found")

    from tasks.scheduled_job_tasks import execute_scheduled_job
    result = execute_scheduled_job.delay(job_id, trigger_type="manual")

    return {"status": "queued", "task_id": result.id, "job_id": job_id}


# ── Run History ──────────────────────────────────────────────────────


@router.get("/{job_id}/runs")
async def list_runs(
    job_id: str,
    limit: int = 20,
    offset: int = 0,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """List run history for a job (paginated, newest first)."""
    runs = await get_scheduled_job_runs(session, job_id, user_id, limit=limit, offset=offset)
    return runs


@router.get("/{job_id}/runs/{run_id}")
async def get_run_detail(
    job_id: str,
    run_id: str,
    session: AsyncSession = Depends(db_session),
    user_id: str = Depends(get_current_user_id),
):
    """Get a single run with step results."""
    run = await get_scheduled_job_run_detail(session, job_id, run_id, user_id)
    if run is None:
        raise HTTPException(404, "Run not found")
    return run
