"""
Airtable source connector.

Pulls image attachments — and every other field on the record, as
metadata — straight from an Airtable base, so they can be fed through the
same ingest pipeline used for manual uploads.

Doesn't need to be told which field holds the image(s) in advance: any
field whose value looks like a list of Airtable attachment objects (dicts
with "url" + "filename") is auto-detected as an attachment field, and
every image-type attachment inside it gets ingested. Every other field on
the record — title, tags, description, whatever your base has — is kept
as-is and passed through as metadata alongside the image.

Airtable's own returned attachment "url" is a short-lived signed link
(expires a couple hours after the API response that contained it) — it's
only used to download the bytes once, immediately, during ingest. It is
never stored; the durable copy is the GCS upload the ingest pipeline
already does.
"""
import time
from typing import Dict, Iterator, List, Optional, Tuple

import requests

AIRTABLE_API_BASE = "https://api.airtable.com/v0"


def _is_attachment_list(value) -> bool:
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(v, dict) and "url" in v and "filename" in v for v in value)
    )


def iter_records(
    base_id: str,
    table_name: str,
    api_key: str,
    view: Optional[str] = None,
    page_size: int = 100,
) -> Iterator[dict]:
    """Yield every record in an Airtable table, handling pagination."""
    url = f"{AIRTABLE_API_BASE}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    offset = None

    while True:
        params = {"pageSize": page_size}
        if view:
            params["view"] = view
        if offset:
            params["offset"] = offset

        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        for record in data.get("records", []):
            yield record

        offset = data.get("offset")
        if not offset:
            break
        time.sleep(0.21)  # stay under Airtable's ~5 requests/sec per-base limit


def extract_image_attachments(fields: dict) -> List[Tuple[str, dict]]:
    """
    Return every (field_name, attachment) pair in a record's fields where
    the attachment is an image. A record can have multiple attachment
    fields, and each field can hold multiple images — all are returned.
    """
    results = []
    for field_name, value in fields.items():
        if not _is_attachment_list(value):
            continue
        for att in value:
            if att.get("type", "").startswith("image/"):
                results.append((field_name, att))
    return results


def non_attachment_fields(fields: dict) -> Dict:
    """Every field on the record that ISN'T an attachment list — kept as metadata."""
    return {k: v for k, v in fields.items() if not _is_attachment_list(v)}


def fetch_attachment_bytes(url: str) -> bytes:
    """Download the actual image bytes from an Airtable attachment URL."""
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    return resp.content


def get_record(base_id: str, table_name: str, api_key: str, record_id: str) -> dict:
    """
    Fetch a single record fresh. Used when confirming a pending webhook
    notification, so the attachment URL is current — Airtable's returned
    attachment URLs expire, so the one from an earlier webhook ping may
    no longer be valid by the time a person actually confirms it.
    """
    url = f"{AIRTABLE_API_BASE}/{base_id}/{table_name}/{record_id}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def update_record(base_id: str, table_name: str, api_key: str, record_id: str, fields: dict) -> dict:
    """Write fields back onto an existing Airtable record (e.g. OCR text, indexed status, our doc id)."""
    url = f"{AIRTABLE_API_BASE}/{base_id}/{table_name}/{record_id}"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    resp = requests.patch(url, headers=headers, json={"fields": fields}, timeout=30)
    resp.raise_for_status()
    return resp.json()
