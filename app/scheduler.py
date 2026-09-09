"""
Optional in-process scheduler that deploys/undeploys the Vertex AI Vector
Search index on a cron schedule (e.g. business hours only), so the
hourly-billed endpoint doesn't sit deployed around the clock.

IMPORTANT CAVEAT: this only runs while this FastAPI process itself is
alive. If you stop `uvicorn` (e.g. between dev sessions), nothing here
fires — the index stays in whatever state it was last left in, which
could mean it's deployed (and billing) all night with nothing to
undeploy it. This is an in-app convenience for a continuously-running
deployment (e.g. once this is actually deployed to a server), not a
substitute for an external scheduler (Cloud Scheduler + a Cloud Function
calling vector_search.deploy_index()/undeploy_index()) if you need the
schedule to hold regardless of whether this app happens to be running.

Disabled by default — set VECTOR_SEARCH_SCHEDULE_ENABLED=true to opt in.
"""
import logging
from typing import Optional

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app import vector_search, usage_meter
from app.config import settings

logger = logging.getLogger("vector_index_scheduler")

_scheduler: Optional[AsyncIOScheduler] = None


def _safe_deploy():
    try:
        logger.info("Scheduled deploy: bringing Vector Search index online")
        did = vector_search.deploy_index()
        if did:
            usage_meter.record_event("deploy")
        logger.info("Deploy %s", "started" if did else "skipped (already deployed)")
    except Exception:
        logger.exception("Scheduled deploy failed")


def _safe_undeploy():
    try:
        logger.info("Scheduled undeploy: taking Vector Search index offline")
        did = vector_search.undeploy_index()
        if did:
            usage_meter.record_event("undeploy")
        logger.info("Undeploy %s", "started" if did else "skipped (already undeployed)")
    except Exception:
        logger.exception("Scheduled undeploy failed")


def start() -> None:
    """Start the schedule if enabled. Safe to call more than once (no-ops after the first)."""
    global _scheduler
    if not settings.vector_search_schedule_enabled:
        logger.info("Vector Search schedule disabled (VECTOR_SEARCH_SCHEDULE_ENABLED=false)")
        return
    if _scheduler is not None:
        return

    _scheduler = AsyncIOScheduler(timezone=settings.vector_search_schedule_timezone)
    _scheduler.add_job(
        _safe_deploy,
        CronTrigger.from_crontab(settings.vector_search_deploy_cron, timezone=settings.vector_search_schedule_timezone),
        id="vector_index_deploy",
    )
    _scheduler.add_job(
        _safe_undeploy,
        CronTrigger.from_crontab(settings.vector_search_undeploy_cron, timezone=settings.vector_search_schedule_timezone),
        id="vector_index_undeploy",
    )
    _scheduler.start()
    logger.info(
        "Vector Search schedule active: deploy='%s' undeploy='%s' tz=%s",
        settings.vector_search_deploy_cron,
        settings.vector_search_undeploy_cron,
        settings.vector_search_schedule_timezone,
    )


def stop() -> None:
    global _scheduler
    if _scheduler is not None:
        _scheduler.shutdown(wait=False)
        _scheduler = None
