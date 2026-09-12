import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union

from app import (
    vision_ocr, embeddings, storage, metadata_store, faiss_index,
    airtable_source, airtable_pending, jobs, auth,
)
from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Rebuild both in-memory FAISS indexes from SQLite (the durable source
    # of truth for embeddings — see app/metadata_store.py) at startup. Cheap
    # at this project's scale (low milliseconds per the CHANGELOG benchmark).
    faiss_index.text_index.rebuild_from(await asyncio.to_thread(metadata_store.get_all_with_text_embeddings))
    faiss_index.image_index.rebuild_from(await asyncio.to_thread(metadata_store.get_all_image_embedding_pairs))
    yield


app = FastAPI(title="OCR + Vector Search (Google Stack)", lifespan=lifespan)

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# --- Login gate --------------------------------------------------------
# See app/auth.py. Everything except the paths below requires a valid
# session cookie. This is a real, if simple, security boundary — the
# project's "no auth on any endpoint" gap is now closed, not just noted.
_PUBLIC_PATHS = {"/login", "/auth/login", "/health", "/favicon.ico"}
_PUBLIC_PREFIXES = ("/static/", "/docs", "/redoc", "/openapi.json")


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.url.path
        if path in _PUBLIC_PATHS or path.startswith(_PUBLIC_PREFIXES):
            return await call_next(request)

        if not auth.is_valid_session(request.cookies.get("session")):
            wants_html = "text/html" in request.headers.get("accept", "")
            if request.method == "GET" and wants_html:
                return RedirectResponse(url=f"/login?next={path}")
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        return await call_next(request)


app.add_middleware(AuthMiddleware)


@app.get("/login", include_in_schema=False)
async def login_page():
    return FileResponse(os.path.join(STATIC_DIR, "login.html"))


@app.post("/auth/login", include_in_schema=False)
async def do_login(email: str = Form(...), password: str = Form(...)):
    if not auth.is_configured():
        raise HTTPException(status_code=500, detail="ADMIN_EMAIL/ADMIN_PASSWORD not configured in .env")
    if not auth.verify_credentials(email, password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = auth.create_session()
    response = JSONResponse({"ok": True})
    response.set_cookie(
        "session", token, httponly=True, samesite="lax", max_age=auth.SESSION_TTL_SECONDS,
    )
    return response


@app.post("/auth/logout", include_in_schema=False)
async def do_logout(request: Request):
    auth.destroy_session(request.cookies.get("session"))
    response = JSONResponse({"ok": True})
    response.delete_cookie("session")
    return response


@app.get("/", include_in_schema=False)
async def index():
    """Default landing page once logged in."""
    return RedirectResponse(url="/ingest")


@app.get("/ingest", include_in_schema=False)
async def ingest_page():
    """Ingest UI — manual upload, Airtable pull, pending-review queue."""
    return FileResponse(os.path.join(STATIC_DIR, "ingest.html"))


@app.get("/search", include_in_schema=False)
async def search_page():
    """Search UI — by text and by image similarity."""
    return FileResponse(os.path.join(STATIC_DIR, "search.html"))


class SearchResult(BaseModel):
    doc_id: str
    distance: float
    image_url: str
    ocr_text: str
    fields: Dict[str, Any] = {}  # Airtable record fields, if this doc came from Airtable — see airtable_fields in SQLite


class SearchResponse(BaseModel):
    results: List[SearchResult]


class ImageSearchResult(BaseModel):
    doc_id: str
    similarity: float
    image_url: str
    ocr_text: str
    fields: Dict[str, Any] = {}


class ImageSearchResponse(BaseModel):
    results: List[ImageSearchResult]


class AirtableFieldInfo(BaseModel):
    id: str
    name: str
    type: str
    isAttachment: bool


class AirtableTableInfo(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    fields: List[AirtableFieldInfo]


class AirtableTablesResponse(BaseModel):
    tables: List[AirtableTableInfo]


class AirtableBaseInfo(BaseModel):
    id: str
    name: str
    permissionLevel: Optional[str] = None


class AirtableBasesResponse(BaseModel):
    bases: List[AirtableBaseInfo]


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


class AirtableIngestJobStarted(BaseModel):
    job_id: str
    status: str = "started"


class AirtableIngestJobStatus(BaseModel):
    job_id: str
    status: str  # "running" | "done" | "error"
    current: Optional[str] = None
    records_scanned: int
    images_found: int
    images_done: int
    results: List[AirtableIngestItem] = []
    error: Optional[str] = None


class AirtableSampleResponse(BaseModel):
    record_id: Optional[str] = None
    image_previews: Dict[str, List[str]] = {}  # field_name -> thumbnail URLs, straight from Airtable
    fields: Dict[str, Any] = {}


class AirtableRecordRow(BaseModel):
    record_id: str
    thumbnail_url: Optional[str] = None
    preview: Dict[str, Any] = {}
    already_ingested: bool = False


class AirtableRecordsResponse(BaseModel):
    records: List[AirtableRecordRow]
    truncated: bool = False  # true if the table has more records than max_records and got cut off


class RefreshMetadataResult(BaseModel):
    record_id: str
    doc_ids_updated: List[str]
    text_reembedded: bool
    write_back_error: Optional[str] = None


class CreateAirtableFieldRequest(BaseModel):
    name: str
    kind: str  # "text" | "long_text" | "checkbox" -- see AIRTABLE_FIELD_KIND_SPECS


class CreateAirtableFieldResponse(BaseModel):
    id: str
    name: str
    type: str


# Friendly write-back "kind" names -> the actual Airtable field type (+ any
# required options) needed to create one via the metadata API. Matches what
# each write-back item actually is: doc_id/text -> plain text, indexed -> checkbox.
AIRTABLE_FIELD_KIND_SPECS = {
    "text": {"type": "singleLineText"},
    "long_text": {"type": "multilineText"},
    "checkbox": {"type": "checkbox", "options": {"icon": "check", "color": "greenBright"}},
}


class AirtablePendingItem(BaseModel):
    record_id: str
    table: str
    status: str
    preview_fields: Optional[Dict[str, Any]] = None
    preview_image_url: Optional[str] = None
    fetch_error: Optional[str] = None


class AirtablePendingList(BaseModel):
    items: List[AirtablePendingItem]


class IngestSourceSummary(BaseModel):
    source: str  # "upload" | "airtable"
    base_id: Optional[str] = None
    base_name: Optional[str] = None  # resolved live from Airtable when possible
    table: Optional[str] = None
    count: int
    text_indexed_count: int
    last_ingested_at: Optional[str] = None  # ISO timestamp


class IngestSummaryResponse(BaseModel):
    total_documents: int
    sources: List[IngestSourceSummary]


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
    extra_search_text: Optional[str] = None,
) -> dict:
    """
    Core per-image ingest pipeline, shared by /ingest, /ingest/batch, and
    /ingest/airtable:
    1. Upload image to GCS
    2. Run Cloud Vision OCR on it
    3. Embed the image itself (multimodalembedding@001), store it in
       SQLite, and upsert it into the in-memory FAISS image index
    4. If text was found: embed it with Vertex AI, store it in SQLite,
       and upsert it into the in-memory FAISS text index
       (both indexes: see app/faiss_index.py)
    5. Save the text/URL/embedding metadata into SQLite, keyed by doc_id

    Images with no detectable text are still ingested and become searchable
    via /search/image — they just won't show up in /search (text) results,
    since there's no text to embed. `text_indexed` in the response tells you
    which happened.

    doc_id: pass a stable, deterministic id (e.g. derived from an Airtable
    attachment id) to make re-ingesting the same source idempotent —
    save_metadata() and both faiss_index upsert() calls overwrite-in-place
    on a repeat doc_id rather than creating a duplicate. Omit it (default)
    to generate a fresh random id, as manual uploads do.

    extra_metadata: arbitrary additional fields merged into the SQLite
    document alongside image_embedding — e.g. where a record came from and
    its other source fields. Never overrides image_embedding/gcs_uri/etc.
    since those are set separately, after this merge.

    extra_search_text: additional text combined with the OCR'd text before
    embedding for /search — e.g. selected Airtable field values (SKU,
    Category, Item Name...), which for product photos with no visible text
    are usually far more useful to search on than OCR ever would be. The
    combined text is stored as `indexed_text` in SQLite (separate from
    `ocr_text`, which stays purely what Vision actually read off the image).
    """
    if not file_bytes:
        raise ValueError("Empty file")

    doc_id = doc_id or str(uuid.uuid4())

    gcs_uri = storage.upload_image(file_bytes, filename, content_type)

    ocr_result = vision_ocr.run_ocr_from_gcs(gcs_uri)
    ocr_text = ocr_result["full_text"].strip()

    image_embedding = embeddings.embed_image(file_bytes)
    faiss_index.image_index.upsert(doc_id, image_embedding)  # keep the in-memory index in sync immediately

    search_text = "\n".join(t.strip() for t in [ocr_text, extra_search_text or ""] if t.strip())
    text_indexed = bool(search_text)
    text_embedding = None
    if text_indexed:
        text_embedding = embeddings.embed_document(search_text)
        faiss_index.text_index.upsert(doc_id, text_embedding)

    # image_url is NOT stored — it would go stale, since signed URLs expire.
    # gcs_uri (permanent) is stored instead; signed URLs are minted fresh
    # from it wherever a response needs one (see _signed_url_for below).
    extra = {"source": "upload", **(extra_metadata or {}), "image_embedding": image_embedding}
    if text_embedding is not None:
        extra["text_embedding"] = text_embedding
    if search_text:
        extra["indexed_text"] = search_text
    metadata_store.save_metadata(doc_id, gcs_uri, "", ocr_text, extra=extra)

    return {
        "doc_id": doc_id,
        "image_url": storage.generate_signed_url(gcs_uri),
        "ocr_text": ocr_text,
        "text_indexed": text_indexed,
    }


def _resolve_write_back_mapping(
    write_doc_id_field: Optional[str], write_indexed_field: Optional[str], write_text_field: Optional[str],
) -> Dict[str, str]:
    """
    Build a {"doc_id"|"indexed"|"ocr_text": "<Airtable field name>"} mapping
    for write-back. Each of the three inputs is None (not specified by the
    caller — fall back to the .env default, preserving old behavior for
    API callers that don't know about this) or a string — "" explicitly
    disables that item, any other string names the field to write it to.
    An entry is present in the returned dict only if that item is enabled.
    """
    def resolve(explicit: Optional[str], default: str) -> Optional[str]:
        chosen = default if explicit is None else explicit
        return chosen or None

    mapping = {}
    doc_id_field = resolve(write_doc_id_field, settings.airtable_write_doc_id_field)
    indexed_field = resolve(write_indexed_field, settings.airtable_write_indexed_field)
    text_field = resolve(write_text_field, settings.airtable_write_ocr_text_field)
    if doc_id_field:
        mapping["doc_id"] = doc_id_field
    if indexed_field:
        mapping["indexed"] = indexed_field
    if text_field:
        mapping["ocr_text"] = text_field
    return mapping


def _write_back_to_airtable(
    base_id: str, table_name: str, record_id: str, mapping: Dict[str, str],
    doc_id: str, ocr_text: str, text_indexed: bool,
) -> Optional[str]:
    """
    Best-effort write of ingest results back onto the source Airtable
    record, writing only the items present in `mapping` (see
    _resolve_write_back_mapping) to whichever field names it specifies.
    Returns an error string on failure, or None on success/nothing-to-write
    — never raises, since a write-back failure (e.g. the configured field
    name doesn't exist in your base) shouldn't fail the ingest itself,
    which already succeeded by the time this runs.
    """
    if not mapping:
        return None
    fields = {}
    if "doc_id" in mapping:
        fields[mapping["doc_id"]] = doc_id
    if "indexed" in mapping:
        fields[mapping["indexed"]] = text_indexed
    if "ocr_text" in mapping:
        fields[mapping["ocr_text"]] = ocr_text
    if not fields:
        return None
    try:
        airtable_source.update_record(base_id, table_name, settings.airtable_api_key, record_id, fields=fields)
        return None
    except Exception as e:
        return str(e)


def _build_search_text(other_fields: dict, text_fields: Optional[List[str]]) -> Optional[str]:
    """Concatenate the selected Airtable field values into one string to embed for text search."""
    if not text_fields:
        return None
    parts = []
    for f in text_fields:
        v = other_fields.get(f)
        if v not in (None, "", []):
            parts.append(f"{f}: {v}")
    return " | ".join(parts) if parts else None


async def _ingest_airtable_record(
    record: dict, base_id: str, table_name: str,
    text_fields: Optional[List[str]] = None, image_field: Optional[str] = None,
    write_back_mapping: Optional[Dict[str, str]] = None,
) -> List[dict]:
    """
    Ingest every image attachment on one already-fetched Airtable record,
    writing results back onto the record afterward. Shared by the bulk
    /ingest/airtable sync and the single-record, human-confirmed
    /airtable/pending/{record_id}/confirm. Never raises — failures are
    captured per-image in the returned list so one bad attachment doesn't
    abort the rest.

    text_fields: names of Airtable fields (e.g. ["Item Name", "Category",
    "SKU"]) whose values get combined and embedded for /search, alongside
    whatever OCR finds on the image itself. For product photos with no
    visible text, this is usually the only thing worth searching on.

    image_field: restrict ingestion to attachments in just this one field,
    for tables with more than one attachment field. Omit to auto-detect
    and ingest from every attachment field on the table (the original
    behavior).

    write_back_mapping: see _resolve_write_back_mapping — which of
    doc_id/indexed/ocr_text to write back, and to which field names. Omit
    to fall back to the .env defaults (see that function).
    """
    record_id = record["id"]
    fields = record.get("fields", {})
    attachments = airtable_source.extract_image_attachments(fields, only_field=image_field)
    other_fields = airtable_source.non_attachment_fields(fields)
    search_text = _build_search_text(other_fields, text_fields)
    if write_back_mapping is None:
        write_back_mapping = _resolve_write_back_mapping(None, None, None)

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
                    "airtable_base_id": base_id,
                    "airtable_table": table_name,
                    "airtable_record_id": record_id,
                    "airtable_field": field_name,
                    "airtable_fields": other_fields,
                },
                extra_search_text=search_text,
            )
            write_back_error = await asyncio.to_thread(
                _write_back_to_airtable, base_id, table_name, record_id, write_back_mapping,
                data["doc_id"], data["ocr_text"], data["text_indexed"],
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


def _make_airtable_ingest_item(record_id: str, r: dict) -> AirtableIngestItem:
    detail = r.get("detail")
    if r.get("write_back_error"):
        detail = f"ingested OK, but write-back to Airtable failed: {r['write_back_error']}"
    return AirtableIngestItem(
        airtable_record_id=record_id, field=r["field"], filename=r["filename"],
        status=r["status"], doc_id=r.get("doc_id"), image_url=r.get("image_url"),
        ocr_text=r.get("ocr_text"), text_indexed=r.get("text_indexed"), detail=detail,
    )


async def _run_airtable_ingest_job(
    job_id: str, base_id: str, table_name: str, view: Optional[str], limit: Optional[int],
    text_fields: Optional[List[str]], image_field: Optional[str], write_back_mapping: Dict[str, str],
    record_ids: Optional[List[str]] = None,
) -> None:
    """
    The actual bulk-ingest work, run as a background task (see jobs.fire_and_forget
    in the endpoint below) — NOT tied to the HTTP request that triggered it.
    Updates the job's progress in app/jobs.py's in-memory store as it goes,
    for GET /ingest/airtable/jobs/{job_id} to report back to a polling UI.
    A client disconnecting, or nobody polling at all, has no effect on this
    loop — it runs to completion (or failure) regardless.

    record_ids: if given, ingest exactly these records (fetched directly by
    id) instead of scanning the whole table — this is what the "mark which
    records to ingest" UI drives. `view`/`limit` are ignored when set.
    """
    try:
        records_scanned = 0
        images_found = 0

        if record_ids:
            records = await asyncio.to_thread(lambda: [
                airtable_source.get_record(base_id, table_name, settings.airtable_api_key, rid)
                for rid in record_ids
            ])
            records = [r for r in records if r]
        else:
            records = await asyncio.to_thread(
                lambda: list(airtable_source.iter_records(base_id, table_name, settings.airtable_api_key, view=view))
            )
            if limit:
                records = records[:limit]

        for record in records:
            records_scanned += 1
            record_id = record["id"]

            attachments = airtable_source.extract_image_attachments(record.get("fields", {}), only_field=image_field)
            images_found += len(attachments)
            jobs.update_job(
                job_id, records_scanned=records_scanned, images_found=images_found,
                current=f"Scanning record {records_scanned} ({record_id}) — {len(attachments)} image(s) found",
            )

            for r in await _ingest_airtable_record(
                record, base_id, table_name, text_fields=text_fields, image_field=image_field,
                write_back_mapping=write_back_mapping,
            ):
                jobs.update_job(job_id, current=f"{r['status'].upper()}: {r['filename']} ({record_id})")
                jobs.append_result(job_id, _make_airtable_ingest_item(record_id, r).model_dump())

        jobs.finish_job(job_id, status="done")
    except Exception as e:
        jobs.finish_job(job_id, status="error", error=str(e))


@app.post("/ingest/airtable", response_model=Union[AirtableIngestResponse, AirtableIngestJobStarted])
async def ingest_from_airtable(
    base: Optional[str] = None,
    table: Optional[str] = None,
    view: Optional[str] = None,
    limit: Optional[int] = None,
    dry_run: bool = False,
    text_fields: Optional[List[str]] = Query(None),
    image_field: Optional[str] = None,
    write_doc_id_field: Optional[str] = None,
    write_indexed_field: Optional[str] = None,
    write_text_field: Optional[str] = None,
    record_ids: Optional[List[str]] = Query(None),
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
    detection against your actual base before running for real. Runs
    synchronously (it's fast — no Vision/Vertex/GCS calls) and returns the
    full AirtableIngestResponse directly.

    dry_run=false (the real ingest) does NOT run synchronously — it can
    take a long time for a large table, and blocking the HTTP response on
    the whole thing means no progress visibility until it's completely
    done. Instead this kicks off the work as an independent background
    task and returns immediately with {job_id, status: "started"}. Poll
    GET /ingest/airtable/jobs/{job_id} for live progress — see the web
    UI's Airtable panel for the progress bar built on top of that.

    base/table/view/limit override the .env defaults for one-off runs
    without changing configuration.

    text_fields: repeatable query param (?text_fields=Item+Name&text_fields=Category)
    naming which Airtable fields to combine and embed for /search — see
    app/main.py's _build_search_text. Omit to index OCR text only, which
    for product photos with no visible text means /search won't find them
    even though /search/image will.

    image_field: restrict ingestion to attachments in just this one field
    (for tables with more than one attachment field). Omit to auto-detect
    and ingest from every attachment field on the table.

    write_doc_id_field/write_indexed_field/write_text_field: which
    Airtable field to write each write-back item to (see
    _resolve_write_back_mapping). Omit any of these to fall back to its
    .env default; pass an empty string to explicitly disable that one
    write-back item for this run.

    record_ids: repeatable query param (?record_ids=rec123&record_ids=rec456)
    naming exactly which records to ingest — from the "mark which records
    to ingest" UI (see GET .../records, whose already_ingested flag drives
    which records get pre-checked/unchecked there). When given, view/limit
    are ignored — only these specific records are processed, fetched
    directly rather than scanning the whole table.
    """
    if not settings.airtable_api_key:
        raise HTTPException(status_code=400, detail="AIRTABLE_API_KEY not configured in .env")

    base_id = base or settings.airtable_base_id
    if not base_id:
        raise HTTPException(
            status_code=400,
            detail="No base specified — pass ?base=... or set AIRTABLE_BASE_ID in .env. GET /airtable/bases lists what's available.",
        )

    table_name = table or settings.airtable_table_name
    if not table_name:
        raise HTTPException(
            status_code=400,
            detail="No table specified — pass ?table=... or set AIRTABLE_TABLE_NAME in .env. GET /airtable/bases/{base_id}/tables lists what's available.",
        )

    if dry_run:
        results: List[AirtableIngestItem] = []
        records_scanned = 0
        images_found = 0

        if record_ids:
            source_records = [
                r for r in (
                    airtable_source.get_record(base_id, table_name, settings.airtable_api_key, rid)
                    for rid in record_ids
                ) if r
            ]
        else:
            source_records = airtable_source.iter_records(base_id, table_name, settings.airtable_api_key, view=view)

        for record in source_records:
            if not record_ids and limit and records_scanned >= limit:
                break
            records_scanned += 1
            record_id = record["id"]
            attachments = airtable_source.extract_image_attachments(record.get("fields", {}), only_field=image_field)
            images_found += len(attachments)
            for field_name, att in attachments:
                results.append(AirtableIngestItem(
                    airtable_record_id=record_id, field=field_name,
                    filename=att.get("filename", ""), status="dry_run",
                    doc_id=f"airtable_{att['id']}",
                ))

        return AirtableIngestResponse(
            records_scanned=records_scanned, images_found=images_found, results=results,
        )

    write_back_mapping = _resolve_write_back_mapping(write_doc_id_field, write_indexed_field, write_text_field)

    job_id = jobs.create_job("ingest_airtable")
    jobs.fire_and_forget(_run_airtable_ingest_job(
        job_id, base_id, table_name, view, limit, text_fields, image_field, write_back_mapping,
        record_ids=record_ids,
    ))
    return AirtableIngestJobStarted(job_id=job_id)


@app.get("/ingest/airtable/jobs/{job_id}", response_model=AirtableIngestJobStatus)
async def get_airtable_ingest_job(job_id: str):
    job = jobs.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="No such job (may have been from a previous server process — jobs are in-memory, not persisted)")
    return AirtableIngestJobStatus(**job)


@app.get("/admin/jobs")
async def list_jobs():
    """
    Every known background job in this process (in-memory, not persisted —
    a server restart clears this), most recently started first. Useful to
    find a running ingest's job_id without already knowing it, e.g. to
    check "is anything running right now" — full per-image results are
    left out here to stay lightweight; hit GET /ingest/airtable/jobs/{job_id}
    for the complete picture on one job.
    """
    return {
        "jobs": [
            {
                "job_id": j["job_id"],
                "kind": j["kind"],
                "status": j["status"],
                "current": j["current"],
                "records_scanned": j["records_scanned"],
                "images_found": j["images_found"],
                "images_done": j["images_done"],
                "started_at": j["started_at"],
                "finished_at": j["finished_at"],
                "error": j["error"],
            }
            for j in jobs.list_jobs()
        ]
    }


# --- Airtable base/table/field discovery -----------------------------------
# Backs the web UI's "Ingest from Airtable" panel: pick a base, pick a
# table, pick which fields to index for search — instead of hardcoding one
# base/table in .env. Requires AIRTABLE_API_KEY to have schema.bases:read
# scope, in addition to data.records:read for actual ingestion.

@app.get("/airtable/bases", response_model=AirtableBasesResponse)
async def list_airtable_bases():
    if not settings.airtable_api_key:
        raise HTTPException(status_code=400, detail="AIRTABLE_API_KEY not configured in .env")
    bases = await asyncio.to_thread(airtable_source.list_bases, settings.airtable_api_key)
    return AirtableBasesResponse(bases=[AirtableBaseInfo(**b) for b in bases])


@app.get("/airtable/bases/{base_id}/tables", response_model=AirtableTablesResponse)
async def list_airtable_tables(base_id: str):
    if not settings.airtable_api_key:
        raise HTTPException(status_code=400, detail="AIRTABLE_API_KEY not configured in .env")
    tables = await asyncio.to_thread(airtable_source.list_tables, base_id, settings.airtable_api_key)
    return AirtableTablesResponse(tables=[AirtableTableInfo(**t) for t in tables])


@app.get("/airtable/bases/{base_id}/tables/{table_name}/sample", response_model=AirtableSampleResponse)
async def airtable_table_sample(base_id: str, table_name: str):
    """
    One real record from the table — actual thumbnail(s) and actual field
    values, not just field names — so the "Ingest from Airtable" panel can
    show a genuine preview before you commit to a real ingest. Thumbnail
    URLs are Airtable's own (short-lived, same as attachment URLs
    elsewhere) — fine for an immediate preview, not stored anywhere.
    """
    if not settings.airtable_api_key:
        raise HTTPException(status_code=400, detail="AIRTABLE_API_KEY not configured in .env")

    record = await asyncio.to_thread(
        airtable_source.get_sample_record, base_id, table_name, settings.airtable_api_key,
    )
    if not record:
        return AirtableSampleResponse()

    fields = record.get("fields", {})
    image_previews: Dict[str, List[str]] = {}
    for field_name, att in airtable_source.extract_image_attachments(fields):
        thumb = (att.get("thumbnails") or {}).get("small", {}).get("url") or att.get("url")
        image_previews.setdefault(field_name, []).append(thumb)

    return AirtableSampleResponse(
        record_id=record.get("id"),
        image_previews=image_previews,
        fields=airtable_source.non_attachment_fields(fields),
    )


@app.get("/airtable/bases/{base_id}/tables/{table_name}/records", response_model=AirtableRecordsResponse)
async def list_airtable_records(
    base_id: str, table_name: str,
    image_field: Optional[str] = None,
    doc_id_field: Optional[str] = None,
    max_records: int = 500,
):
    """
    Lists records in a table for the "mark which records to ingest" UI —
    a thumbnail + a few field values per record, and whether it looks
    already ingested.

    "Already ingested" is only knowable via doc_id_field — pass the field
    name your Doc ID write-back is configured to use (see the ③
    Write-back section), and any record with a non-empty value there is
    flagged. Without doc_id_field, there's no signal to go on, so every
    record comes back unflagged — that's expected, not a bug.

    Reads the whole table in one call (not paginated) — capped at
    max_records as a safety net; `truncated: true` means the table has
    more records than that and some were left out.
    """
    if not settings.airtable_api_key:
        raise HTTPException(status_code=400, detail="AIRTABLE_API_KEY not configured in .env")

    all_records = await asyncio.to_thread(
        lambda: list(airtable_source.iter_records(base_id, table_name, settings.airtable_api_key))
    )

    rows = []
    for record in all_records[:max_records]:
        fields = record.get("fields", {})
        attachments = airtable_source.extract_image_attachments(fields, only_field=image_field)
        thumb = None
        if attachments:
            att = attachments[0][1]
            thumb = (att.get("thumbnails") or {}).get("small", {}).get("url") or att.get("url")

        already_ingested = bool(doc_id_field and fields.get(doc_id_field))

        preview = dict(list(airtable_source.non_attachment_fields(fields).items())[:4])

        rows.append(AirtableRecordRow(
            record_id=record["id"], thumbnail_url=thumb, preview=preview, already_ingested=already_ingested,
        ))

    return AirtableRecordsResponse(records=rows, truncated=len(all_records) > max_records)


@app.post("/airtable/bases/{base_id}/tables/{table_id}/fields", response_model=CreateAirtableFieldResponse)
async def create_airtable_field(base_id: str, table_id: str, body: CreateAirtableFieldRequest):
    """
    Create a new field on a table — backs the write-back UI's "+ Create
    new field" option, for when you want to write ingest results back but
    the table doesn't have a suitable field yet. Requires the configured
    AIRTABLE_API_KEY to have schema.bases:write scope in addition to the
    read scopes needed elsewhere.
    """
    if not settings.airtable_api_key:
        raise HTTPException(status_code=400, detail="AIRTABLE_API_KEY not configured in .env")

    field_spec = AIRTABLE_FIELD_KIND_SPECS.get(body.kind)
    if not field_spec:
        raise HTTPException(status_code=400, detail=f"Unknown kind '{body.kind}', expected one of {list(AIRTABLE_FIELD_KIND_SPECS)}")

    try:
        field = await asyncio.to_thread(
            airtable_source.create_field, base_id, table_id, settings.airtable_api_key, body.name, field_spec,
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to create field (check your token has schema.bases:write scope): {e}")

    return CreateAirtableFieldResponse(id=field["id"], name=field["name"], type=field["type"])


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

    Expected body: {"record_id": "recXXXXXXXXXXXXXX", "table": "TableName", "base_id": "appXXXXXXXXXXXXXX"}
    ("table" and "base_id" are optional — fall back to AIRTABLE_TABLE_NAME / AIRTABLE_BASE_ID).

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
    base_id = payload.get("base_id") or settings.airtable_base_id
    if not base_id:
        raise HTTPException(status_code=400, detail="No base_id in payload and AIRTABLE_BASE_ID not set in .env")

    await asyncio.to_thread(airtable_pending.add_pending, record_id, base_id, table_name, payload)
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
                p.get("base_id") or settings.airtable_base_id, p["table"], settings.airtable_api_key, p["record_id"],
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
    base_id = pending.get("base_id") or settings.airtable_base_id
    record = await asyncio.to_thread(
        airtable_source.get_record,
        base_id, table_name, settings.airtable_api_key, record_id,
    )
    results = await _ingest_airtable_record(record, base_id, table_name)

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


@app.post("/airtable/records/{record_id}/refresh-metadata", response_model=RefreshMetadataResult)
async def refresh_airtable_metadata(
    record_id: str,
    base: Optional[str] = None,
    table: Optional[str] = None,
    text_fields: Optional[List[str]] = Query(None),
    write_doc_id_field: Optional[str] = None,
    write_indexed_field: Optional[str] = None,
    write_text_field: Optional[str] = None,
):
    """
    Refresh metadata for every already-ingested document tied to this
    Airtable record — WITHOUT re-downloading the image, re-running OCR,
    or re-embedding the image. Use this when a record's non-image field
    values changed (e.g. SKU renamed, category updated) but the photo
    itself didn't, so a full re-ingest (and its OCR/Vertex/GCS cost)
    isn't needed just to pick up the new field values.

    Always updates airtable_fields to the record's current values on
    every doc tied to it (a record with multiple images in its attachment
    field produces multiple docs — all get refreshed).

    text_fields (repeatable query param, same as /ingest/airtable): if
    given, also recombines each doc's already-stored ocr_text with the
    record's current field values and re-embeds JUST that combined text
    (cheap — one Vertex call per doc, no image involved) so /search stays
    accurate to the new field values. image_embedding is never touched.

    write_doc_id_field/write_indexed_field/write_text_field: same as
    /ingest/airtable — write the refreshed state back onto the record.
    """
    if not settings.airtable_api_key:
        raise HTTPException(status_code=400, detail="AIRTABLE_API_KEY not configured in .env")

    base_id = base or settings.airtable_base_id
    table_name = table or settings.airtable_table_name
    if not base_id or not table_name:
        raise HTTPException(status_code=400, detail="base/table required — pass ?base=...&table=..., or set AIRTABLE_BASE_ID/AIRTABLE_TABLE_NAME in .env")

    doc_ids = await asyncio.to_thread(metadata_store.get_doc_ids_for_airtable_record, record_id)
    if not doc_ids:
        raise HTTPException(status_code=404, detail=f"No ingested documents found for Airtable record {record_id}")

    record = await asyncio.to_thread(airtable_source.get_record, base_id, table_name, settings.airtable_api_key, record_id)
    fields = record.get("fields", {})
    other_fields = airtable_source.non_attachment_fields(fields)
    field_text = _build_search_text(other_fields, text_fields)

    write_back_mapping = _resolve_write_back_mapping(write_doc_id_field, write_indexed_field, write_text_field)
    write_back_error = None
    reembedded_any = False

    existing_batch = await asyncio.to_thread(metadata_store.get_metadata_batch, doc_ids)

    for doc_id in doc_ids:
        update = {"airtable_fields": other_fields}
        existing = existing_batch.get(doc_id, {})
        ocr_text = existing.get("ocr_text", "") or ""
        text_indexed = bool(existing.get("text_embedding"))

        if text_fields is not None:
            combined = "\n".join(t.strip() for t in [ocr_text, field_text or ""] if t.strip())
            if combined:
                new_embedding = await asyncio.to_thread(embeddings.embed_document, combined)
                update["indexed_text"] = combined
                update["text_embedding"] = new_embedding
                faiss_index.text_index.upsert(doc_id, new_embedding)
                reembedded_any = True
                text_indexed = True
            else:
                update["indexed_text"] = None
                update["text_embedding"] = None
                text_indexed = False

        await asyncio.to_thread(metadata_store.update_fields, doc_id, **update)

        wb_err = await asyncio.to_thread(
            _write_back_to_airtable, base_id, table_name, record_id, write_back_mapping,
            doc_id, ocr_text, text_indexed,
        )
        if wb_err:
            write_back_error = wb_err

    return RefreshMetadataResult(
        record_id=record_id, doc_ids_updated=doc_ids, text_reembedded=reembedded_any, write_back_error=write_back_error,
    )


@app.get("/api/search", response_model=SearchResponse)
async def search(q: str, top_k: int = 10):
    """
    1. Embed the query text (query task_type, not document task_type)
    2. Find nearest neighbors via the in-memory FAISS index (app/faiss_index.py)
       — exact cosine-similarity search, not a deployed/billed service
    3. Look up the matched doc_ids' text/URL in SQLite

    `distance` in the response is kept as the field name for API/UI
    compatibility with the earlier Vertex-backed version, computed as
    `1 - cosine_similarity` (so lower still means "closer," same as before).

    Path is /api/search, not /search — GET /search is the search page
    (see the page route near the top of this file); this JSON endpoint
    needed a distinct path to avoid colliding with it.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    query_embedding = embeddings.embed_query(q)
    neighbors = faiss_index.text_index.search(query_embedding, top_k=top_k)  # [(doc_id, similarity), ...]

    doc_ids = [doc_id for doc_id, _ in neighbors]
    metadata = metadata_store.get_metadata_batch(doc_ids)

    results = []
    for doc_id, sim in neighbors:
        meta = metadata.get(doc_id)
        if not meta:
            continue
        results.append(SearchResult(
            doc_id=doc_id,
            distance=1.0 - sim,
            image_url=_signed_url_for(meta),
            ocr_text=meta.get("ocr_text", ""),
            fields=meta.get("airtable_fields") or {},
        ))

    return SearchResponse(results=results)


@app.post("/search/image", response_model=ImageSearchResponse)
async def search_by_image(file: UploadFile = File(...), top_k: int = 10):
    """
    Visual similarity search — upload an image, get back the most visually
    similar ones already ingested, regardless of what text (if any) they
    contain. FAISS-backed (app/faiss_index.py), same exact-search approach
    as text search — previously a pure-Python brute-force scan
    (app/similarity.py, retired, kept in the repo for reference).
    """
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Empty file")

    query_embedding = embeddings.embed_image(file_bytes)
    neighbors = faiss_index.image_index.search(query_embedding, top_k=top_k)  # [(doc_id, similarity), ...]

    doc_ids = [doc_id for doc_id, _ in neighbors]
    metadata = metadata_store.get_metadata_batch(doc_ids)

    results = []
    for doc_id, sim in neighbors:
        meta = metadata.get(doc_id)
        if not meta:
            continue
        results.append(ImageSearchResult(
            doc_id=doc_id,
            similarity=sim,
            image_url=_signed_url_for(meta),
            ocr_text=meta.get("ocr_text", ""),
            fields=meta.get("airtable_fields") or {},
        ))
    return ImageSearchResponse(results=results)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/admin/faiss-index/status")
async def faiss_index_status():
    """
    Sanity-check endpoint for the in-memory FAISS indexes — how many
    documents each currently holds. (The old /admin/vector-index/*
    endpoints for the retired Vertex AI Vector Search endpoint are gone
    along with it — see app/vector_search.py, app/scheduler.py,
    app/usage_meter.py, kept in the repo but no longer wired in, and
    CHANGELOG.md for why.)
    """
    return {
        "text_documents_indexed": faiss_index.text_index.size(),
        "image_documents_indexed": faiss_index.image_index.size(),
    }


@app.get("/admin/ingest-summary", response_model=IngestSummaryResponse)
async def ingest_summary():
    """
    Breakdown of every ingested document by source — manual upload vs.
    which specific Airtable base+table — with a count, how many of those
    are text-searchable, and when the most recent one in that group landed.

    Reads the whole SQLite table once (same brute-force-at-this-
    scale tradeoff as the rest of this app — see CHANGELOG.md); fine up to
    a few thousand documents.

    Note: docs ingested before airtable_base_id/airtable_table started
    being recorded show up with base_id/table = null, grouped together —
    there's no way to retroactively attribute those without re-ingesting
    them (harmless to do — ingestion is idempotent, keyed by Airtable
    attachment id).
    """
    rows = await asyncio.to_thread(metadata_store.get_all_source_summaries)

    groups: Dict[tuple, Dict[str, Any]] = {}
    for r in rows:
        key = (r["source"], r.get("airtable_base_id"), r.get("airtable_table"))
        g = groups.setdefault(key, {"count": 0, "text_indexed_count": 0, "last_ingested_at": None})
        g["count"] += 1
        if r["text_indexed"]:
            g["text_indexed_count"] += 1
        ts = r.get("created_at")
        if ts is not None and (g["last_ingested_at"] is None or ts > g["last_ingested_at"]):
            g["last_ingested_at"] = ts

    base_names: Dict[str, str] = {}
    if settings.airtable_api_key:
        try:
            bases = await asyncio.to_thread(airtable_source.list_bases, settings.airtable_api_key)
            base_names = {b["id"]: b["name"] for b in bases}
        except Exception:
            pass  # best-effort only -- summary still works without base names

    sources = [
        IngestSourceSummary(
            source=source, base_id=base_id, base_name=base_names.get(base_id) if base_id else None,
            table=table, count=g["count"], text_indexed_count=g["text_indexed_count"],
            last_ingested_at=g["last_ingested_at"].isoformat() if g["last_ingested_at"] is not None else None,
        )
        for (source, base_id, table), g in groups.items()
    ]
    sources.sort(key=lambda s: -s.count)

    return IngestSummaryResponse(total_documents=len(rows), sources=sources)
