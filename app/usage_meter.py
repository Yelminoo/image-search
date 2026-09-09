"""
Self-tracked usage meter for the Vector Search endpoint.

This is NOT a real GCP billing lookup — there's no live connection into
your actual bill here. It logs every deploy/undeploy this app performs
(via the /admin endpoints or the schedule) to Firestore, sums up how many
hours the index has actually been deployed, and multiplies by an
estimated hourly rate (settings.vector_search_hourly_rate_estimate_usd)
to give a running ballpark. Treat the dollar figure as an estimate, not a
receipt — the GCP Billing Reports console (see README) has the real
number.

Known gap: this only sees deploys/undeploys that happened THROUGH this
app. The very first deploy — via setup_vector_index.py, run directly, not
through this app — was never logged. ensure_baseline() seeds a synthetic
"deploy" event as of app-startup time so the meter has *some* starting
point, but hours before that point are simply not counted.
"""
import datetime
from typing import List, Optional

from google.cloud import firestore

from app.config import settings

_client = None
COLLECTION = "vector_index_events"


def _get_client() -> firestore.Client:
    global _client
    if _client is None:
        _client = firestore.Client(project=settings.project_id)
    return _client


def record_event(action: str) -> None:
    """action: "deploy" or "undeploy". Only call this for an action that actually happened (not a no-op) — an extra event breaks the pairing math below."""
    _get_client().collection(COLLECTION).add({
        "action": action,
        "timestamp": firestore.SERVER_TIMESTAMP,
    })


def ensure_baseline(currently_deployed: bool) -> None:
    """Seed a synthetic "deploy" event as of now if there's no history yet and the index is already deployed — see module docstring for the caveat."""
    existing = list(_get_client().collection(COLLECTION).limit(1).stream())
    if existing:
        return
    if currently_deployed:
        record_event("deploy")


def _iter_events() -> List[dict]:
    docs = _get_client().collection(COLLECTION).order_by("timestamp").stream()
    return [d.to_dict() for d in docs]


def compute_usage() -> dict:
    """
    Pair up deploy/undeploy events chronologically and sum the deployed
    duration. If the last event is an unmatched "deploy", time up to now
    counts as still-accruing.
    """
    total_seconds = 0.0
    open_deploy_at: Optional[datetime.datetime] = None

    for e in _iter_events():
        ts = e.get("timestamp")
        if ts is None:
            continue
        if e.get("action") == "deploy":
            open_deploy_at = ts
        elif e.get("action") == "undeploy" and open_deploy_at is not None:
            total_seconds += (ts - open_deploy_at).total_seconds()
            open_deploy_at = None

    currently_deployed = open_deploy_at is not None
    if currently_deployed:
        now = datetime.datetime.now(datetime.timezone.utc)
        total_seconds += (now - open_deploy_at).total_seconds()

    total_hours = total_seconds / 3600
    rate = settings.vector_search_hourly_rate_estimate_usd
    return {
        "currently_deployed": currently_deployed,
        "total_deployed_hours": round(total_hours, 3),
        "estimated_cost_usd": round(total_hours * rate, 4),
        "hourly_rate_used_usd": rate,
    }
