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


def extract_image_attachments(fields: dict, only_field: Optional[str] = None) -> List[Tuple[str, dict]]:
    """
    Return every (field_name, attachment) pair in a record's fields where
    the attachment is an image. A record can have multiple attachment
    fields, and each field can hold multiple images — all are returned,
    unless only_field is given, restricting this to just that one field
    (used when the user explicitly picked which field holds the image to
    ingest, rather than auto-detecting every attachment field on the table).
    """
    results = []
    for field_name, value in fields.items():
        if only_field and field_name != only_field:
            continue
        if not _is_attachment_list(value):
            continue
        for att in value:
            if att.get("type", "").startswith("image/"):
                results.append((field_name, att))
    return results


def non_attachment_fields(fields: dict) -> Dict:
    """Every field on the record that ISN'T an attachment list — kept as metadata."""
    return {k: v for k, v in fields.items() if not _is_attachment_list(v)}


def get_sample_record(base_id: str, table_name: str, api_key: str) -> Optional[dict]:
    """
    Fetch one record from a table, for a live preview during setup (shows
    an actual thumbnail + real field values, not just field names, so you
    can visually confirm the right field before running a real ingest).
    Returns None if the table has no records.
    """
    url = f"{AIRTABLE_API_BASE}/{base_id}/{table_name}"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, params={"maxRecords": 1}, timeout=30)
    resp.raise_for_status()
    records = resp.json().get("records", [])
    return records[0] if records else None


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


def list_bases(api_key: str) -> List[dict]:
    """
    List every base this token can access — [{id, name, permissionLevel}, ...].
    Requires the token to have schema.bases:read scope (in addition to
    data.records:read for actual ingestion). Used to populate the "pick a
    base" UI instead of hardcoding one in .env.
    """
    url = f"{AIRTABLE_API_BASE}/meta/bases"
    headers = {"Authorization": f"Bearer {api_key}"}
    results = []
    offset = None
    while True:
        params = {"offset": offset} if offset else {}
        resp = requests.get(url, headers=headers, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        results.extend(data.get("bases", []))
        offset = data.get("offset")
        if not offset:
            break
    return results


def list_tables(base_id: str, api_key: str) -> List[dict]:
    """
    List every table in a base, with each field's name and type —
    [{id, name, fields: [{id, name, type}, ...]}, ...]. Used to populate
    the "pick a table, pick which fields to index" UI. Attachment-type
    fields are flagged so the UI can distinguish "this holds the image(s)"
    from "this is text worth indexing" without guessing.
    """
    url = f"{AIRTABLE_API_BASE}/meta/bases/{base_id}/tables"
    headers = {"Authorization": f"Bearer {api_key}"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    tables = resp.json().get("tables", [])
    for table in tables:
        for field in table.get("fields", []):
            field["isAttachment"] = field.get("type") == "multipleAttachments"
    return tables


def create_field(base_id: str, table_id: str, api_key: str, name: str, field_spec: dict) -> dict:
    """
    Create a new field on a table via Airtable's metadata API. Requires
    the token to have schema.bases:write scope in addition to the read
    scopes needed elsewhere — a 403 here usually means that scope is
    missing, not that anything else is wrong.

    field_spec: the field's type (and, for some types, required options),
    e.g. {"type": "singleLineText"} or
    {"type": "checkbox", "options": {"icon": "check", "color": "greenBright"}}.
    """
    url = f"{AIRTABLE_API_BASE}/meta/bases/{base_id}/tables/{table_id}/fields"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    body = {"name": name, **field_spec}
    resp = requests.post(url, headers=headers, json=body, timeout=30)
    resp.raise_for_status()
    return resp.json()
