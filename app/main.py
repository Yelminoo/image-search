import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

from app import (
    vision_ocr, embeddings, storage, vector_search, metadata_store, similarity,
    airtable_source, airtable_pending, scheduler, usage_meter,
)
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    usage_meter.ensure_baseline(vector_search.is_deployed())
    scheduler.start()  # no-op unless VECTOR_SEARCH_SCHEDULE_ENABLED=true
    yield
    scheduler.stop()


app = FastAPI(title="OCR + Vector Search (Google Stack)", lifespan=lifespan)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index():
    """Serve the web UI."""
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


class SearchResult(BaseModel):
    doc_id: str
    distance: float
    image_url: str
    ocr_text: str


class SearchResponse(BaseModel):
    results: List[SearchResult]


class ImageSearchResult(BaseModel):
    doc_id: str
    similarity: float
    image_url: str
    ocr_text: str


class ImageSearchResponse(BaseModel):
    results: List[ImageSearchResult]


class BatchIngestItem(BaseModel):
    filename: str
    status: str  # "ok" | "error"
    doc_id: Optional[str] = None
    image_url: Optional[str] = None
    ocr_text: Optional[str] = None
    text_indexed: Optional[bool] = None
    detail: Optional[str] = None


class BatchIngestResponse(BaseModel):
    results: List[BatchIngestItem]


class AirtableIngestItem(BaseModel):
    airtable_record_id: str
    field: str
    filename: str
    status: str  # "ok" | "error" | "dry_run"
    doc_id: Optional[str] = None
    image_url: Optional[str] = None
    ocr_text: Optional[str] = None
    text_indexed: Optional[bool] = None
    detail: Optional[str] = None


class AirtableIngestResponse(BaseModel):
    records_scanned: int
    images_found: int
    results: List[AirtableIngestItem]


class AirtablePendingItem(BaseModel):
    record_id: str
    table: str
    status: str
    preview_fields: Optional[Dict[str, Any]] = None
    preview_image_url: Optional[str] = None
    fetch_error: Optional[str] = None


class AirtablePendingList(BaseModel):
    items: List[AirtablePendingItem]


def _signed_url_for(meta: dict) -> str:
    """Mint a fresh signed URL from a stored gcs_uri, or "" if there isn't one."""
    gcs_uri = meta.get("gcs_uri")
    return storage.generate_signed_url(gcs_uri) if gcs_uri else ""


async def _ingest_one(
    file_bytes: bytes,
    filename: str,
    content_type: str,
    doc_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> dict:
    """
    Core per-image ingest pipeline, shared by /ingest, /ingest/batch, and
    /ingest/airtable:
    1. Upload image to GCS
    2. Run Cloud Vision OCR on it
    3. Embed the image itself (multimodalembedding@001) for similarity search
    4. If text was found: embed it with Vertex AI and upsert into Vector Search
    5. Save the text/URL/image-embedding metadata into Firestore, keyed by doc_id

    Images with no detectable text are still ingested and become searchable
    via /search/image — they just won't show up in /search (text) results,
    since there's no text to embed. `text_indexed` in the response tells you
    which happened.

    doc_id: pass a stable, deterministic id (e.g. derived from an Airtable
    attachment id) to make re-ingesting the same source idempotent —
    save_metadata()/upsert_vector() both overwrite-in-place on a repeat
    doc_id rather than creating a duplicate. Omit it (default) to generate
    a fresh random id, as manual uploads do.

    extra_metadata: arbitrary additional fields merged into the Firestore
    document alongside image_embedding — e.g. where a record came from and
    its other source fields. Never overrides image_embedding/gcs_uri/etc.
    since those are set separately, after this merge.
    """
    if not file_bytes:
        raise ValueError("Empty file")

    doc_id = doc_id or str(uuid.uuid4())

    gcs_uri = storage.upload_image(file_bytes, filename, content_type)

    ocr_result = vision_ocr.run_ocr_from_gcs(gcs_uri)
    ocr_text = ocr_result["full_text"].strip()

    image_embedding = embeddings.embed_image(file_bytes)

    text_indexed = bool(ocr_text)
    if text_indexed:
        text_embedding = embeddings.embed_document(ocr_text)
        vector_search.upsert_vector(doc_id, text_embedding)

    # image_url is NOT stored — it would go stale, since signed URLs expire.
    # gcs_uri (permanent) is stored instead; signed URLs are minted fresh
    # from it wherever a response needs one (see _signed_url_for below).
    extra = {"source": "upload", **(extra_metadata or {}), "image_embedding": image_embedding}
    metadata_store.save_metadata(doc_id, gcs_uri, "", ocr_text, extra=extra)

    return {
        "doc_id": doc_id,
        "image_url": storage.generate_signed_url(gcs_uri),
        "ocr_text": ocr_text,
        "text_indexed": text_indexed,
    }


def _write_back_to_airtable(table_name: str, record_id: str, doc_id: str, ocr_text: str, text_indexed: bool) -> Optional[str]:
    """
    Best-effort write of ingest results back onto the source Airtable
    record. Returns an error string on failure, or None on success —
    never raises, since a write-back failure (e.g. the configured field
    name doesn't exist in your base yet) shouldn't fail the ingest itself,
    which already succeeded by the time this runs.
    """
    try:
        airtable_source.update_record(
            settings.airtable_base_id, table_name, settings.airtable_api_key, record_id,
            fields={
                settings.airtable_write_doc_id_field: doc_id,
                settings.airtable_write_indexed_field: text_indexed,
                settings.airtable_write_ocr_text_field: ocr_text,
            },
        )
        return None
    except Exception as e:
        return str(e)


async def _ingest_airtable_record(record: dict, table_name: str) -> List[dict]:
    """
    Ingest every image attachment on one already-fetched Airtable record,
    writing results back onto the record afterward. Shared by the bulk
    /ingest/airtable sync and the single-record, human-confirmed
    /airtable/pending/{record_id}/confirm. Never raises — failures are
    captured per-image in the returned list so one bad attachment doesn't
    abort the rest.
    """
    record_id = record["id"]
    fields = record.get("fields", {})
    attachments = airtable_source.extract_image_attachments(fields)
    other_fields = airtable_source.non_attachment_fields(fields)

    results = []
    for field_name, att in attachments:
        filename = att.get("filename", "image")
        doc_id = f"airtable_{att['id']}"
        try:
            file_bytes = await asyncio.to_thread(airtable_source.fetch_attachment_bytes, att["url"])
            data = await _ingest_one(
                file_bytes, filename, att.get("type", "image/jpeg"),
                doc_id=doc_id,
                extra_metadata={
                    "source": "airtable",
                    "airtable_record_id": record_id,
                    "airtable_field": field_name,
                    "airtable_fields": other_fields,
                },
            )
            write_back_error = await asyncio.to_thread(
                _write_back_to_airtable, table_name, record_id, data["doc_id"], data["ocr_text"], data["text_indexed"],
            )
            results.append({"field": field_name, "filename": filename, "status": "ok", "write_back_error": write_back_error, **data})
        except Exception as e:
            results.append({"field": field_name, "filename": filename, "status": "error", "detail": str(e)})
    return results


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Ingest a single image. See _ingest_one for the pipeline."""
    file_bytes = await file.read()
    try:
        return await _ingest_one(file_bytes, file.filename, file.content_type or "image/jpeg")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/ingest/batch", response_model=BatchIngestResponse)
async def ingest_batch(files: List[UploadFile] = File(...)):
    """
    Ingest multiple images in one request. Each file is processed
    independently — one bad/unreadable image doesn't abort the rest of the
    batch, it just gets status="error" with a detail message while the
    others still succeed.
    """
    results = []
    for f in files:
        file_bytes = await f.read()
        try:
            data = await _ingest_one(file_bytes, f.filename, f.content_type or "image/jpeg")
            results.append(BatchIngestItem(filename=f.filename, status="ok", **data))
        except Exception as e:
            results.append(BatchIngestItem(filename=f.filename, status="error", detail=str(e)))
    return BatchIngestResponse(results=results)


@app.post("/ingest/airtable", response_model=AirtableIngestResponse)
async def ingest_from_airtable(
    table: Optional[str] = None,
    view: Optional[str] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
):
    """
    Pull every image attachment from an Airtable base — plus every other
    field on each record, kept as metadata — and ingest them through the
    same pipeline as manual uploads. No need to know which field holds the
    image(s) in advance; see app/airtable_source.py for how that's detected.

    Idempotent: each ingested doc is keyed by the Airtable attachment's own
    id (doc_id="airtable_<attachment_id>"), so re-running this later
    updates existing docs in place rather than duplicating them — safe to
    use as a periodic sync.

    dry_run=true walks the base and reports what WOULD be ingested,
    without calling Vision/Vertex/GCS — use it to sanity-check attachment
    detection against your actual base before running for real.

    table/view/limit override the .env defaults for one-off runs without
    changing configuration.
    """
    if not settings.airtable_api_key or not settings.airtable_base_id:
        raise HTTPException(
            status_code=400,
            detail="AIRTABLE_API_KEY / AIRTABLE_BASE_ID not configured in .env",
        )

    table_name = table or settings.airtable_table_name
    if not table_name:
        raise HTTPException(
            status_code=400,
            detail="No table specified — pass ?table=... or set AIRTABLE_TABLE_NAME in .env",
        )

    results: List[AirtableIngestItem] = []
    records_scanned = 0
    images_found = 0

    for record in airtable_source.iter_records(
        settings.airtable_base_id, table_name, settings.airtable_api_key, view=view,
    ):
        if limit and records_scanned >= limit:
            break
        records_scanned += 1

        record_id = record["id"]
        attachments = airtable_source.extract_image_attachments(record.get("fields", {}))
        images_found += len(attachments)

        if dry_run:
            for field_name, att in attachments:
                results.append(AirtableIngestItem(
                    airtable_record_id=record_id, field=field_name,
                    filename=att.get("filename", ""), status="dry_run",
                    doc_id=f"airtable_{att['id']}",
                ))
            continue

        for r in await _ingest_airtable_record(record, table_name):
            detail = r.get("detail")
            if r.get("write_back_error"):
                detail = f"ingested OK, but write-back to Airtable failed: {r['write_back_error']}"
            results.append(AirtableIngestItem(
                airtable_record_id=record_id, field=r["field"], filename=r["filename"],
                status=r["status"], doc_id=r.get("doc_id"), image_url=r.get("image_url"),
                ocr_text=r.get("ocr_text"), text_indexed=r.get("text_indexed"), detail=detail,
            ))

    return AirtableIngestResponse(
        records_scanned=records_scanned, images_found=images_found, results=results,
    )


# --- Airtable webhook: notify-only, human confirms before anything ingests ---
# See README for the exact Airtable Automation setup ("Send webhook" action).

@app.post("/webhooks/airtable")
async def airtable_webhook(
    payload: Dict[str, Any],
    secret: Optional[str] = None,
    x_webhook_secret: Optional[str] = Header(None),
):
    """
    Receives a notification from an Airtable Automation's "Send webhook"
    action. Does NOT ingest anything by itself — it only queues the record
    for a person to review and explicitly confirm, via the web UI's
    "Pending from Airtable" panel or POST /airtable/pending/{id}/confirm.
    This is deliberate: auto-ingesting on every Airtable edit means real
    OCR/Vertex API calls (and writes back into Airtable) firing on every
    change with no human check.

    Expected body: {"record_id": "recXXXXXXXXXXXXXX", "table": "TableName"}
    ("table" is optional — falls back to AIRTABLE_TABLE_NAME).

    If AIRTABLE_WEBHOOK_SECRET is set, requires it as either ?secret=...
    or an X-Webhook-Secret header — this endpoint is meant to be reachable
    from the internet for Airtable to call it, so unlike the rest of this
    unauthenticated API, it's worth actually locking down once deployed
    anywhere public.
    """
    if settings.airtable_webhook_secret:
        if (secret or x_webhook_secret) != settings.airtable_webhook_secret:
            raise HTTPException(status_code=401, detail="Invalid or missing webhook secret")

    record_id = payload.get("record_id")
    if not record_id:
        raise HTTPException(status_code=400, detail="Missing record_id in webhook payload")
    table_name = payload.get("table") or settings.airtable_table_name

    await asyncio.to_thread(airtable_pending.add_pending, record_id, table_name, payload)
    return {"queued": True, "record_id": record_id}


@app.get("/airtable/pending", response_model=AirtablePendingList)
async def list_airtable_pending():
    """
    Notifications queued by the webhook, awaiting human confirmation.
    Fetches a live preview (fields + first image) from Airtable for each —
    best-effort, since the webhook ping itself carries no content, only a
    record id.
    """
    pending = await asyncio.to_thread(airtable_pending.list_pending)
    items = []
    for p in pending:
        try:
            record = await asyncio.to_thread(
                airtable_source.get_record,
                settings.airtable_base_id, p["table"], settings.airtable_api_key, p["record_id"],
            )
            fields = record.get("fields", {})
            attachments = airtable_source.extract_image_attachments(fields)
            items.append(AirtablePendingItem(
                record_id=p["record_id"], table=p["table"], status=p["status"],
                preview_fields=airtable_source.non_attachment_fields(fields),
                preview_image_url=attachments[0][1]["url"] if attachments else None,
            ))
        except Exception as e:
            items.append(AirtablePendingItem(
                record_id=p["record_id"], table=p["table"], status=p["status"], fetch_error=str(e),
            ))
    return AirtablePendingList(items=items)


@app.post("/airtable/pending/{record_id}/confirm")
async def confirm_airtable_pending(record_id: str):
    """
    Actually ingest a pending record now. Re-fetches it fresh (so the
    attachment URL is current, not the possibly-stale one from when the
    webhook fired), ingests every image attachment on it, writes results
    back onto the Airtable record, and marks the pending entry resolved.
    """
    pending = await asyncio.to_thread(airtable_pending.get_pending, record_id)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending notification for that record_id")

    table_name = pending["table"]
    record = await asyncio.to_thread(
        airtable_source.get_record,
        settings.airtable_base_id, table_name, settings.airtable_api_key, record_id,
    )
    results = await _ingest_airtable_record(record, table_name)

    await asyncio.to_thread(
        airtable_pending.mark_status, record_id, "ingested",
        {"ingested_doc_ids": [r["doc_id"] for r in results if r.get("doc_id")]},
    )
    return {"record_id": record_id, "results": results}


@app.post("/airtable/pending/{record_id}/dismiss")
async def dismiss_airtable_pending(record_id: str):
    """Reject a pending notification without ingesting anything."""
    pending = await asyncio.to_thread(airtable_pending.get_pending, record_id)
    if not pending:
        raise HTTPException(status_code=404, detail="No pending notification for that record_id")
    await asyncio.to_thread(airtable_pending.mark_status, record_id, "dismissed")
    return {"record_id": record_id, "dismissed": True}


@app.get("/search", response_model=SearchResponse)
async def search(q: str, top_k: int = 10):
    """
    1. Embed the query text (query task_type, not document task_type)
    2. Find nearest neighbors in Vertex AI Vector Search
    3. Look up the matched doc_ids' text/URL in Firestore
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = embeddings.embed_query(q)
    neighbors = vector_search.search(query_embedding, top_k=top_k)

    doc_ids = [n.id for n in neighbors]
    metadata = metadata_store.get_metadata_batch(doc_ids)

    results = []
    for n in neighbors:
        meta = metadata.get(n.id)
        if not meta:
            continue
        results.append(SearchResult(
            doc_id=n.id,
            distance=n.distance,
            image_url=_signed_url_for(meta),
            ocr_text=meta.get("ocr_text", ""),
        ))

    return SearchResponse(results=results)


@app.post("/search/image", response_model=ImageSearchResponse)
async def search_by_image(file: UploadFile = File(...), top_k: int = 10):
    """
    Visual similarity search — upload an image, get back the most visually
    similar ones already ingested, regardless of what text (if any) they
    contain. Uses brute-force cosine similarity over embeddings stored in
    Firestore (see app/similarity.py) rather than a second Vector Search
    index — fine at small scale, revisit if the collection grows large.
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    query_embedding = embeddings.embed_image(file_bytes)
    candidates = metadata_store.get_all_with_image_embeddings()

    top = similarity.search_similar_images(query_embedding, candidates, top_k=top_k)

    results = [
        ImageSearchResult(
            doc_id=c["doc_id"],
            similarity=c["similarity"],
            image_url=_signed_url_for(c),
            ocr_text=c.get("ocr_text", ""),
        )
        for c in top
    ]
    return ImageSearchResponse(results=results)


@app.get("/health")
async def health():
    return {"status": "ok"}


# --- Vector Search index cost control -------------------------------------
# Manual override for the scheduled deploy/undeploy in app/scheduler.py.
# Useful outside the configured schedule (e.g. an unscheduled late-night
# test session) or when VECTOR_SEARCH_SCHEDULE_ENABLED is off entirely.
#
# WARNING: unlike every other endpoint in this API, these directly control
# real hourly billing (deploy_index) and search availability (undeploy_index
# breaks /search until redeployed). This API has no auth on ANY endpoint
# (a known, documented gap) — that's a bigger deal here than on /ingest or
# /search, since anyone who can reach these can run up your bill or take
# text search offline. Put these behind auth before exposing this API
# beyond localhost.

@app.get("/admin/vector-index/status")
async def vector_index_status():
    deployed = await asyncio.to_thread(vector_search.is_deployed)
    return {"deployed": deployed}


@app.post("/admin/vector-index/deploy")
async def vector_index_deploy():
    # deploy_index()/undeploy_index() are synchronous SDK calls that can
    # block for minutes (redeploying isn't instant even though the index
    # itself already exists) — offloaded to a thread so they don't freeze
    # the whole app (every other request, including /search) meanwhile.
    started = await asyncio.to_thread(vector_search.deploy_index)
    if started:
        await asyncio.to_thread(usage_meter.record_event, "deploy")
    return {"deployed": True, "action_taken": started}


@app.post("/admin/vector-index/undeploy")
async def vector_index_undeploy():
    stopped = await asyncio.to_thread(vector_search.undeploy_index)
    if stopped:
        await asyncio.to_thread(usage_meter.record_event, "undeploy")
    return {"deployed": False, "action_taken": stopped}


@app.get("/admin/vector-index/usage")
async def vector_index_usage():
    """
    Self-tracked usage meter — NOT a live GCP billing lookup, just this
    app's own log of deploy/undeploy events turned into an hours + rough
    cost estimate. See app/usage_meter.py for exactly what it does and
    doesn't capture.
    """
    return await asyncio.to_thread(usage_meter.compute_usage)
