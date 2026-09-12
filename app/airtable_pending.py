"""
Human-in-the-loop queue for Airtable webhook notifications.

A webhook ping (see POST /webhooks/airtable in main.py) does NOT trigger
ingestion by itself — it only records that something changed, for a
person to review and explicitly confirm (or dismiss) via the web UI or
POST /airtable/pending/{record_id}/confirm. This is deliberate: silently
auto-ingesting on every Airtable edit means OCR/Vertex API calls (real
cost, and real writes back into Airtable) firing on every change with no
human check — the whole point of this queue is to interpose a review
step before that happens.
"""
from typing import List, Optional

from google.cloud import firestore

from app.config import settings

_client = None
COLLECTION = "airtable_pending"


def _get_client() -> firestore.Client:
    global _client
    if _client is None:
        _client = firestore.Client(project=settings.project_id)
    return _client


def add_pending(record_id: str, base_id: str, table: str, raw_payload: dict) -> None:
    """
    Record a notification. Keyed by record_id — a repeat webhook ping for
    the same record (e.g. edited again before being reviewed) refreshes
    the existing pending entry rather than creating a duplicate.
    """
    _get_client().collection(COLLECTION).document(record_id).set({
        "record_id": record_id,
        "base_id": base_id,
        "table": table,
        "status": "pending",
        "received_at": firestore.SERVER_TIMESTAMP,
        "raw_payload": raw_payload,
    })


def list_pending() -> List[dict]:
    docs = _get_client().collection(COLLECTION).where("status", "==", "pending").stream()
    return [d.to_dict() for d in docs]


def get_pending(record_id: str) -> Optional[dict]:
    snap = _get_client().collection(COLLECTION).document(record_id).get()
    return snap.to_dict() if snap.exists else None


def mark_status(record_id: str, status: str, extra: Optional[dict] = None) -> None:
    """status: "ingested" | "dismissed"."""
    data = {"status": status}
    if extra:
        data.update(extra)
    _get_client().collection(COLLECTION).document(record_id).update(data)
