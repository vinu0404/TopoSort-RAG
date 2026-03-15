"""
RedBeat schedule sync helpers.

Creates / removes dynamic Celery Beat periodic-task entries
backed by Redis (via celery-redbeat).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _convert_cron_to_utc(cron_expr: str, tz_name: str) -> str:
    """Convert a cron expression from a user's timezone to UTC.

    Handles the common case of fixed hour/minute. For wildcard or complex
    expressions in hour/minute fields, returns the cron unchanged (treated as UTC).
    """
    if tz_name == "UTC":
        return cron_expr
    try:
        from datetime import datetime, timedelta
        from zoneinfo import ZoneInfo

        parts = cron_expr.split()
        if len(parts) != 5:
            return cron_expr

        minute, hour, dom, month, dow = parts

        # Only convert when both hour and minute are simple integers
        if not hour.isdigit() or not minute.isdigit():
            return cron_expr

        local_tz = ZoneInfo(tz_name)
        utc_tz = ZoneInfo("UTC")

        # Build a reference datetime in the user's timezone for today
        now = datetime.now(local_tz)
        ref = now.replace(hour=int(hour), minute=int(minute), second=0, microsecond=0)
        utc_ref = ref.astimezone(utc_tz)

        utc_parts = [str(utc_ref.minute), str(utc_ref.hour), dom, month, dow]
        utc_cron = " ".join(utc_parts)
        logger.info("Converted cron %s (%s) → %s (UTC)", cron_expr, tz_name, utc_cron)
        return utc_cron
    except Exception:
        logger.warning("Could not convert cron %s from %s to UTC — using as-is",
                        cron_expr, tz_name)
        return cron_expr


def sync_job_to_beat(job_id: str, cron_expr: str, timezone: str = "UTC") -> None:
    """Create or update a RedBeat periodic task entry for a scheduled job."""
    try:
        from celery.schedules import crontab
        from redbeat import RedBeatSchedulerEntry

        from celery_app import celery

        utc_cron = _convert_cron_to_utc(cron_expr, timezone)
        parts = utc_cron.split()
        if len(parts) != 5:
            logger.error("Invalid cron expression for job %s: %s", job_id, cron_expr)
            return

        entry = RedBeatSchedulerEntry(
            name=f"scheduled_job:{job_id}",
            task="tasks.execute_scheduled_job",
            schedule=crontab(
                minute=parts[0],
                hour=parts[1],
                day_of_month=parts[2],
                month_of_year=parts[3],
                day_of_week=parts[4],
            ),
            args=[job_id],
            kwargs={"trigger_type": "scheduled"},
            app=celery,
        )
        entry.save()
        logger.info("Synced job %s to RedBeat: %s (user: %s %s)",
                     job_id, utc_cron, cron_expr, timezone)
    except ImportError:
        logger.warning(
            "celery-redbeat not installed — cannot sync job %s to beat scheduler. "
            "Install with: pip install celery-redbeat",
            job_id,
        )
    except Exception:
        logger.exception("Failed to sync job %s to RedBeat", job_id)


def remove_job_from_beat(job_id: str) -> None:
    """Remove the periodic task from RedBeat."""
    try:
        from redbeat import RedBeatSchedulerEntry

        from celery_app import celery

        prefix = celery.conf.get("redbeat_key_prefix", "redbeat:")
        entry = RedBeatSchedulerEntry.from_key(
            f"{prefix}scheduled_job:{job_id}", app=celery
        )
        entry.delete()
        logger.info("Removed job %s from RedBeat", job_id)
    except ImportError:
        logger.warning("celery-redbeat not installed — cannot remove job %s", job_id)
    except Exception:
        # Entry may not exist — that's fine
        logger.debug("Could not remove job %s from RedBeat (may not exist)", job_id)
