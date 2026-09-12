"""
Metadata store using local SQLite.

Replaced the original Firestore-backed version (kept for reference in
app/metadata_store_firestore.py) — at this project's actual scale and
access pattern (write one doc per ingest, read everything once at startup
to rebuild the FAISS indexes, look up a handful by id per search, no
complex queries), Firestore was a managed cloud service buying nothing
that a local file doesn't already provide for free. Same reasoning that
already justified swapping Vertex AI Vector Search for in-process FAISS —
see CHANGELOG.md.

Trade-off, so it isn't lost later: this ties the data to whatever machine
runs the app (a local file, not cloud-backed). Fine while this runs
locally; worth reconsidering (mounted volume, periodic backup to GCS, or
back to a managed store) if this ever gets deployed to ephemeral cloud
infrastructure.

Embeddings and the airtable_fields dict are stored as JSON-encoded TEXT —
SQLite has no native array/object type, and at this scale (rebuilt into
memory once at startup) there's no reason for anything more compact.

Every function here keeps the exact same signature as the Firestore
version it replaced, so nothing else in the codebase changed to make this
swap — see main.py, backfill_text_embeddings.py, migrate_firestore_to_sqlite.py.
"""
import datetime
import json
import os
import sqlite3
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "local_data.db")


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db() -> None:
    with _get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                gcs_uri TEXT,
                image_url TEXT,
                ocr_text TEXT,
                indexed_text TEXT,
                source TEXT,
                airtable_base_id TEXT,
                airtable_table TEXT,
                airtable_record_id TEXT,
                airtable_field TEXT,
                airtable_fields TEXT,
                image_embedding TEXT,
                text_embedding TEXT,
                created_at TEXT
            )
        """)


_init_db()


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    if d.get("airtable_fields"):
        d["airtable_fields"] = json.loads(d["airtable_fields"])
    if d.get("image_embedding"):
        d["image_embedding"] = json.loads(d["image_embedding"])
    if d.get("text_embedding"):
        d["text_embedding"] = json.loads(d["text_embedding"])
    return d


def save_metadata(
    doc_id: str, gcs_uri: str, image_url: str, ocr_text: str,
    extra: Optional[dict] = None, created_at: Optional[str] = None,
) -> None:
    """
    Upsert a document. created_at is refreshed to "now" on every call
    (including re-ingesting an existing doc_id) unless explicitly
    overridden — matches the original Firestore version's behavior
    (`.set()` with `firestore.SERVER_TIMESTAMP` every time), which the
    ingest-summary and job-monitoring endpoints rely on to show "most
    recently touched." The override param exists only for
    migrate_firestore_to_sqlite.py, to preserve original timestamps.
    """
    extra = extra or {}
    row = {
        "doc_id": doc_id,
        "gcs_uri": gcs_uri,
        "image_url": image_url,
        "ocr_text": ocr_text,
        "indexed_text": extra.get("indexed_text"),
        "source": extra.get("source"),
        "airtable_base_id": extra.get("airtable_base_id"),
        "airtable_table": extra.get("airtable_table"),
        "airtable_record_id": extra.get("airtable_record_id"),
        "airtable_field": extra.get("airtable_field"),
        "airtable_fields": json.dumps(extra["airtable_fields"]) if extra.get("airtable_fields") is not None else None,
        "image_embedding": json.dumps(extra["image_embedding"]) if extra.get("image_embedding") is not None else None,
        "text_embedding": json.dumps(extra["text_embedding"]) if extra.get("text_embedding") is not None else None,
        "created_at": created_at or datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with _get_connection() as conn:
        conn.execute("""
            INSERT INTO documents (doc_id, gcs_uri, image_url, ocr_text, indexed_text, source,
                airtable_base_id, airtable_table, airtable_record_id, airtable_field, airtable_fields,
                image_embedding, text_embedding, created_at)
            VALUES (:doc_id, :gcs_uri, :image_url, :ocr_text, :indexed_text, :source,
                :airtable_base_id, :airtable_table, :airtable_record_id, :airtable_field, :airtable_fields,
                :image_embedding, :text_embedding, :created_at)
            ON CONFLICT(doc_id) DO UPDATE SET
                gcs_uri=excluded.gcs_uri, image_url=excluded.image_url, ocr_text=excluded.ocr_text,
                indexed_text=excluded.indexed_text, source=excluded.source,
                airtable_base_id=excluded.airtable_base_id, airtable_table=excluded.airtable_table,
                airtable_record_id=excluded.airtable_record_id, airtable_field=excluded.airtable_field,
                airtable_fields=excluded.airtable_fields, image_embedding=excluded.image_embedding,
                text_embedding=excluded.text_embedding, created_at=excluded.created_at
        """, row)


def get_metadata_batch(doc_ids: list) -> dict:
    """Fetch metadata for a list of doc_ids, returned as {doc_id: data}."""
    if not doc_ids:
        return {}
    placeholders = ",".join("?" * len(doc_ids))
    with _get_connection() as conn:
        rows = conn.execute(f"SELECT * FROM documents WHERE doc_id IN ({placeholders})", doc_ids).fetchall()
    return {r["doc_id"]: _row_to_dict(r) for r in rows}


def get_all_with_image_embeddings() -> list:
    """
    Fetch every document that has an image_embedding field, as full
    metadata dicts. LEGACY — was used by the pure-Python brute-force image
    search in app/similarity.py, retired in favor of FAISS. Kept for
    reference; no longer called from main.py.
    """
    with _get_connection() as conn:
        rows = conn.execute("SELECT * FROM documents WHERE image_embedding IS NOT NULL").fetchall()
    return [_row_to_dict(r) for r in rows]


def get_all_image_embedding_pairs() -> list:
    """
    Fetch every (doc_id, image_embedding) pair — used once at app startup
    to rebuild the in-memory FAISS image index (see app/faiss_index.py).
    """
    with _get_connection() as conn:
        rows = conn.execute("SELECT doc_id, image_embedding FROM documents WHERE image_embedding IS NOT NULL").fetchall()
    return [(r["doc_id"], json.loads(r["image_embedding"])) for r in rows]


def get_all_with_text_embeddings() -> list:
    """
    Fetch every (doc_id, text_embedding) pair — used once at app startup
    to rebuild the in-memory FAISS text index (see app/faiss_index.py).
    """
    with _get_connection() as conn:
        rows = conn.execute("SELECT doc_id, text_embedding FROM documents WHERE text_embedding IS NOT NULL").fetchall()
    return [(r["doc_id"], json.loads(r["text_embedding"])) for r in rows]


def get_all_missing_text_embeddings() -> list:
    """
    Fetch every (doc_id, ocr_text) pair for documents that have OCR'd text
    but no text_embedding yet — used by backfill_text_embeddings.py.
    """
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT doc_id, ocr_text FROM documents WHERE ocr_text IS NOT NULL AND ocr_text != '' AND text_embedding IS NULL"
        ).fetchall()
    return [(r["doc_id"], r["ocr_text"]) for r in rows]


def update_text_embedding(doc_id: str, embedding: list) -> None:
    """Write just the text_embedding field onto an existing document (used by the backfill script)."""
    with _get_connection() as conn:
        conn.execute("UPDATE documents SET text_embedding = ? WHERE doc_id = ?", (json.dumps(embedding), doc_id))


_JSON_FIELDS = {"airtable_fields", "image_embedding", "text_embedding"}


def update_fields(doc_id: str, **fields) -> None:
    """
    Partially update specific columns on an existing document — unlike
    save_metadata (which replaces the whole row, matching the original
    Firestore .set() semantics), this only touches the columns you pass.
    For lightweight metadata refreshes that shouldn't wipe fields they
    don't know about — e.g. updating airtable_fields/indexed_text/
    text_embedding after a source field value changes, without
    re-downloading the image or re-running OCR/image-embedding.

    Values for JSON-stored columns (airtable_fields, image_embedding,
    text_embedding) are encoded automatically — pass plain Python
    dicts/lists, not pre-serialized strings. No-ops if fields is empty.
    """
    if not fields:
        return
    for k in list(fields.keys()):
        if k in _JSON_FIELDS and fields[k] is not None:
            fields[k] = json.dumps(fields[k])
    set_clause = ", ".join(f"{k} = :{k}" for k in fields)
    params = dict(fields)
    params["doc_id"] = doc_id
    with _get_connection() as conn:
        conn.execute(f"UPDATE documents SET {set_clause} WHERE doc_id = :doc_id", params)


def get_doc_ids_for_airtable_record(record_id: str) -> list:
    """
    All doc_ids ingested from a given Airtable record — a record with
    multiple images in its attachment field produces multiple docs, all
    sharing this record_id. Used to refresh metadata on every image tied
    to a record that changed, without knowing their doc_ids in advance.
    """
    with _get_connection() as conn:
        rows = conn.execute("SELECT doc_id FROM documents WHERE airtable_record_id = ?", (record_id,)).fetchall()
    return [r["doc_id"] for r in rows]


def get_all_source_summaries() -> list:
    """
    Fetch just enough from every document to build the ingest-by-source
    breakdown (GET /admin/ingest-summary in main.py).
    """
    with _get_connection() as conn:
        rows = conn.execute(
            "SELECT source, airtable_base_id, airtable_table, text_embedding, created_at FROM documents"
        ).fetchall()
    results = []
    for r in rows:
        created_at = datetime.datetime.fromisoformat(r["created_at"]) if r["created_at"] else None
        results.append({
            "source": r["source"] or "upload",
            "airtable_base_id": r["airtable_base_id"],
            "airtable_table": r["airtable_table"],
            "text_indexed": bool(r["text_embedding"]),
            "created_at": created_at,
        })
    return results
