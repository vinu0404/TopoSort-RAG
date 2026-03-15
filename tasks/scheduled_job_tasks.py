"""
Celery task for scheduled job execution.

Runs in a separate worker process.  Uses ``asyncio.run()`` to bridge
the sync Celery world with the async orchestrator pipeline.
Publishes status updates via Redis Pub/Sub so the frontend SSE endpoint
can push them to clients in real-time.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict

import redis
from celery import Task

from celery_app import celery
from config.settings import config

logger = logging.getLogger(__name__)

_redis_client: redis.Redis | None = None


def _get_redis() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(config.redis_url, decode_responses=True)
    return _redis_client


def _publish_job_status(user_id: str, job_id: str, **extra: Any) -> None:
    """Publish a status event to the user's scheduled-job Redis channel."""
    channel = f"scheduled_job:{user_id}"
    payload = {"job_id": job_id, **extra}
    _get_redis().publish(channel, json.dumps(payload))


@celery.task(
    bind=True,
    name="tasks.execute_scheduled_job",
    acks_late=True,
    max_retries=0,
    soft_time_limit=600,
    time_limit=660,
)
def execute_scheduled_job(
    self: Task,
    job_id: str,
    trigger_type: str = "scheduled",
) -> Dict[str, Any]:
    """Celery entry point — bridge to async."""
    return asyncio.run(
        _execute_scheduled_job_async(self, job_id, trigger_type)
    )


async def _execute_scheduled_job_async(
    task: Task,
    job_id: str,
    trigger_type: str = "scheduled",
) -> Dict[str, Any]:
    """Async implementation: load job → build plan → orchestrate → record results."""
    from sqlalchemy.ext.asyncio import (
        AsyncSession,
        async_sessionmaker,
        create_async_engine,
    )

    from core.agent_factory import build_agent_instances
    from core.orchestrator import Orchestrator
    from database.helpers import (
        create_scheduled_job_run,
        load_scheduled_job_with_steps,
    )
    from database.models import (
        ScheduledJob,
        ScheduledJobRun,
        ScheduledJobStepResult,
    )
    from tools.registry import ToolRegistry
    from utils.schemas import (
        HitlResolvedDecision,
        ResolvedAgentTask,
        ResolvedExecutionPlan,
    )

    _engine = create_async_engine(
        config.database_url, echo=False, pool_size=2, max_overflow=2,
        pool_recycle=60,
    )
    _session_factory = async_sessionmaker(
        _engine, class_=AsyncSession, expire_on_commit=False,
    )

    async with _session_factory() as session:
        try:
            # 1. Load job + steps
            job, steps = await load_scheduled_job_with_steps(session, job_id)
            if job is None:
                logger.warning("Scheduled job %s not found", job_id)
                return {"status": "not_found", "job_id": job_id}

            if job.status != "active":
                logger.info("Scheduled job %s is %s — skipping", job_id, job.status)
                return {"status": "skipped", "job_id": job_id}

            user_id = str(job.user_id)

            # 2. Create run record
            run = await create_scheduled_job_run(
                session, job_id,
                trigger_type=trigger_type,
                total_steps=len(steps),
            )
            await session.commit()
            run_id = str(run.run_id)

            _publish_job_status(user_id, job_id,
                                event="run_started", run_id=run_id,
                                trigger_type=trigger_type)

            # 3. Create step result placeholders
            step_result_map = {}
            for step in steps:
                sr = ScheduledJobStepResult(
                    run_id=run.run_id,
                    step_id=step.step_id,
                    step_order=step.step_order,
                    agent_name=step.agent_name,
                    status="pending",
                )
                session.add(sr)
                await session.flush()
                step_result_map[step.step_order] = sr
            await session.commit()

            # 4. Convert steps → ResolvedExecutionPlan
            step_id_map = {}
            for step in steps:
                agent_id = f"{step.agent_name}_{str(step.step_id).replace('-', '')[:8]}"
                step_id_map[step.step_order] = agent_id

            resolved_tasks = []
            for step in steps:
                resolved_tasks.append(ResolvedAgentTask(
                    agent_id=step_id_map[step.step_order],
                    agent_name=step.agent_name,
                    task=step.task,
                    entities=step.entities or {},
                    tools=step.tools or [],
                    depends_on=[
                        step_id_map[dep_idx]
                        for dep_idx in (step.depends_on_steps or [])
                        if dep_idx in step_id_map
                    ],
                    timeout=step.timeout,
                    max_retries=step.max_retries,
                    priority=step.priority,
                ))

            plan = ResolvedExecutionPlan(agents=resolved_tasks)

            # 5. Build agent instances + orchestrator
            registry = ToolRegistry()
            if not registry.list_tools():
                registry.auto_discover_tools()
            agent_instances = build_agent_instances(registry)
            orchestrator = Orchestrator(
                agent_instances=agent_instances,
                tool_registry=registry,
            )

            # 6. Set user context for OAuth tools
            try:
                from tools.mail_tools import current_user_id
                current_user_id.set(user_id)
            except ImportError:
                pass

            # 7. Build context
            context = {
                "user_id": user_id,
                "query_id": f"scheduled_{job_id}_{run_id}",
                "session_id": None,
                "conversation_history": [],
                "long_term_memory": {},
                "active_web_collection_ids": [],
                "selected_doc_ids": [],
            }

            # 8. Auto-approve HITL callback
            async def auto_approve_hitl(agent_cfg, hitl_tools):
                return HitlResolvedDecision(
                    approved=True,
                    instructions=None,
                    tool_names=hitl_tools,
                )

            # 9. Execute plan
            logger.info("Executing scheduled job %s (run %s) with %d steps",
                        job_id, run_id, len(steps))

            shared_state = await orchestrator.execute_plan(
                plan, context, on_hitl_needed=auto_approve_hitl,
            )

            # 10. Record step results
            completed = 0
            failed = 0
            for step in steps:
                agent_id = step_id_map[step.step_order]
                output = shared_state.get(agent_id)
                sr = step_result_map[step.step_order]
                sr.completed_at = datetime.now(timezone.utc)

                if output is None:
                    sr.status = "skipped"
                    sr.error_message = "No output from orchestrator"
                elif hasattr(output, "task_done") and output.task_done:
                    sr.status = "success"
                    sr.agent_output = output.model_dump() if hasattr(output, "model_dump") else {"raw": str(output)}
                    sr.resource_usage = getattr(output, "resource_usage", {})
                    completed += 1
                else:
                    sr.status = "failed"
                    sr.agent_output = output.model_dump() if hasattr(output, "model_dump") else {"raw": str(output)}
                    sr.error_message = getattr(output, "error", str(output))[:500]
                    failed += 1

                _publish_job_status(user_id, job_id,
                                    event="step_complete", run_id=run_id,
                                    step_order=step.step_order,
                                    agent_name=step.agent_name,
                                    status=sr.status)

            # 10b. Run Composer to produce a human-readable answer
            composed_answer = ""
            try:
                from config.settings import config as app_config
                from core.composer_agent import ComposerAgent
                from utils.llm_providers import get_llm_provider
                from utils.schemas import (
                    ComposerInput,
                    LongTermMemory,
                    Source,
                )

                agent_outputs = [
                    v for v in shared_state.values()
                    if hasattr(v, "agent_id")
                ]
                all_sources = []
                for ao in agent_outputs:
                    for s in (ao.metadata or {}).get("sources", []):
                        if isinstance(s, dict):
                            all_sources.append(Source(
                                agent=ao.agent_id,
                                **{k: v for k, v in s.items() if k in Source.model_fields},
                            ))
                        elif isinstance(s, Source):
                            all_sources.append(s)

                original_query = " | ".join(s.task for s in steps)
                composer_llm = get_llm_provider(
                    app_config.composer_model_provider,
                    default_model=app_config.composer_model,
                )
                composer = ComposerAgent(llm_provider=composer_llm)
                composer_input = ComposerInput(
                    query_id=f"scheduled_{job_id}_{run_id}",
                    original_query=original_query,
                    user_id=user_id,
                    agent_results=agent_outputs,
                    all_sources=all_sources,
                    long_term_memory=LongTermMemory(user_id=user_id),
                    conversation_history=[],
                )
                composer_output = await composer.compose(composer_input)
                composed_answer = composer_output.answer or ""
                logger.info("Composed answer for job %s run %s (%d chars)",
                            job_id, run_id, len(composed_answer))
            except Exception:
                logger.exception("Composer failed for scheduled job %s — using raw output", job_id)

            # Store composed answer in the first successful step result
            if composed_answer:
                for sr in step_result_map.values():
                    if sr.status == "success" and sr.agent_output:
                        sr.agent_output["composed_answer"] = composed_answer
                        break

            # 11. Update run record
            run.completed_at = datetime.now(timezone.utc)
            run.completed_steps = completed
            run.failed_steps = failed
            if failed == 0:
                run.status = "success"
            elif completed > 0:
                run.status = "partial_failure"
            else:
                run.status = "failed"
                run.error_summary = "All steps failed"

            # 12. Update job timestamps
            job.last_run_at = datetime.now(timezone.utc)
            try:
                from croniter import croniter
                from zoneinfo import ZoneInfo
                tz_name = job.timezone if job.timezone and job.timezone != "UTC" else None
                if tz_name:
                    local_tz = ZoneInfo(tz_name)
                    now_local = datetime.now(local_tz)
                    cron = croniter(job.cron_expression, now_local)
                    next_local = cron.get_next(datetime)
                    job.next_run_at = next_local.astimezone(timezone.utc)
                else:
                    cron = croniter(job.cron_expression, datetime.now(timezone.utc))
                    job.next_run_at = cron.get_next(datetime)
            except Exception:
                pass

            job.updated_at = datetime.now(timezone.utc)
            await session.commit()

            result_text = composed_answer[:2000] if composed_answer else ""

            _publish_job_status(user_id, job_id,
                                event="run_complete", run_id=run_id,
                                status=run.status,
                                completed_steps=completed,
                                failed_steps=failed,
                                total_steps=len(steps),
                                result_text=result_text)

            # 13. Send notification
            result_texts = [composed_answer] if composed_answer else []
            await _send_notification(session, job, run, step_result_map, result_texts, user_id)

            logger.info("Scheduled job %s run %s completed: %s (%d/%d steps)",
                        job_id, run_id, run.status, completed, len(steps))

            return {
                "job_id": job_id,
                "run_id": run_id,
                "status": run.status,
                "completed_steps": completed,
                "failed_steps": failed,
            }

        except Exception as exc:
            logger.exception("Scheduled job %s failed", job_id)
            # Try to update run status
            try:
                from sqlalchemy import select, update
                await session.rollback()
                # Find the latest run for this job
                from database.models import ScheduledJobRun as SJR
                result = await session.execute(
                    select(SJR)
                    .where(SJR.job_id == job.job_id if job else False)
                    .order_by(SJR.created_at_.desc())
                    .limit(1)
                )
                run_row = result.scalar_one_or_none()
                if run_row and run_row.status == "running":
                    run_row.status = "failed"
                    run_row.error_summary = str(exc)[:500]
                    run_row.completed_at = datetime.now(timezone.utc)
                    await session.commit()
            except Exception:
                logger.exception("Failed to persist error for scheduled job %s", job_id)
            raise

        finally:
            await _engine.dispose()


async def _send_notification(session, job, run, step_result_map, result_texts, user_id: str):
    """Send notification based on job.notification_mode."""
    if job.notification_mode == "none":
        return

    # Build summary
    lines = [f"Scheduled job \"{job.name}\" completed with status: {run.status}"]
    for order, sr in sorted(step_result_map.items()):
        status_icon = "OK" if sr.status == "success" else "FAIL" if sr.status == "failed" else sr.status
        lines.append(f"  Step {order + 1} ({sr.agent_name}): {status_icon}")
    if result_texts:
        lines.append("")
        lines.append(("\n\n".join(result_texts))[:2000])
    summary = "\n".join(lines)

    if job.notification_mode == "in_app":
        _publish_job_status(user_id, str(job.job_id),
                            event="notification",
                            title=f"Job \"{job.name}\" — {run.status}",
                            summary=summary,
                            run_id=str(run.run_id))

    elif job.notification_mode == "email":
        try:
            from tools.mail_tools import current_user_id, prepare_gmail_service
            current_user_id.set(user_id)
            service = await prepare_gmail_service()
            if service:
                from email.mime.text import MIMEText
                import base64
                target = job.notification_target or ""
                if not target:
                    # Try getting user's email
                    from database.models import User
                    from sqlalchemy import select
                    result = await session.execute(
                        select(User.email).where(User.user_id == job.user_id)
                    )
                    row = result.scalar_one_or_none()
                    target = row if row else ""
                if target:
                    msg = MIMEText(summary)
                    msg["to"] = target
                    msg["subject"] = f"[MRAG] {job.name} — {run.status}"
                    raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
                    service.users().messages().send(
                        userId="me", body={"raw": raw}
                    ).execute()
                    run.notification_sent = True
                    await session.commit()
        except Exception:
            logger.exception("Failed to send email notification for job %s", job.job_id)
