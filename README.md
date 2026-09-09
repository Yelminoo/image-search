# OCR + Semantic Search — Full Google Stack

Cloud Vision (OCR) → Vertex AI (embeddings) → Vertex AI Vector Search (nearest-neighbor search), with Firestore holding the text/metadata side-table and GCS holding the raw images.

## Why this shape

Vertex AI Vector Search only stores vector IDs and the vectors themselves — no text, no URLs. So every vector needs a matching row in Firestore, keyed by the same `doc_id`, to get the human-readable content back after a search hit.

## One-time GCP setup

1. Create a GCP project, enable billing.
2. Enable these APIs: Cloud Vision API, Vertex AI API, Cloud Storage, Firestore.
3. Create a GCS bucket for images.
4. Create a Firestore database (Native mode) in the same project.
5. Create a service account with roles: `Cloud Vision AI User`, `Vertex AI User`, `Storage Object Admin`, `Cloud Datastore User`. Download its JSON key.
6. Copy `.env.example` to `.env` and fill in your project ID, bucket name, and the path to the service account key.

## Deploy the vector index (one-time, ~20-60 min)

```bash
pip install -r requirements.txt
python setup_vector_index.py
```

Copy the printed `VERTEX_INDEX_ID` and `VERTEX_INDEX_ENDPOINT_ID` into your `.env`.

**Note:** Vertex AI Vector Search bills for the deployed endpoint by the hour, whether or not you're querying it. Undeploy it when not in active use if cost matters — see "Cost control" below for a scheduled or manual way to do that.

## Cost control — deploy/undeploy on a schedule

The Vector Search endpoint is the one real ongoing cost in this stack (everything else is free-tier or fractions of a cent at normal usage) — it bills hourly for simply being deployed, independent of query volume. Two ways to control that:

**Manual switch, from the web UI** — the bar at the top of `http://localhost:8000/` shows live status (online/offline dot), the usage meter, and Start/Stop buttons. No curl needed for day-to-day use.

**Manual, via API:**
```bash
curl -X POST http://localhost:8000/admin/vector-index/undeploy   # stop billing
curl -X POST http://localhost:8000/admin/vector-index/deploy     # bring it back (fast — the index itself already exists, no rebuild)
curl http://localhost:8000/admin/vector-index/status             # check current state
curl http://localhost:8000/admin/vector-index/usage              # hours deployed + estimated cost (see meter caveat below)
```

**Usage meter:** `/admin/vector-index/usage` (and the web UI bar) shows cumulative deployed-hours and an estimated dollar cost. This is **self-tracked, not a real GCP billing lookup** — it works by logging every deploy/undeploy *this app* performs to Firestore and summing the deployed duration between them, then multiplying by `VECTOR_SEARCH_HOURLY_RATE_ESTIMATE_USD`. It also can't see the very first deploy from `setup_vector_index.py` (run outside this app) — it seeds a synthetic starting point at whenever this app first started tracking, so hours before that aren't counted. Treat the number as a ballpark for "is this thing costing me money right now," not an invoice.

**On the rate itself:** the default was originally an unconfirmed third-party guess of $0.077/hr for a small `e2-standard-2` machine — turned out to be way off. Updated 2026-09-07 to **$0.87/hr**, derived from a real observed charge (200 THB for 7 hours deployed). `setup_vector_index.py` never specifies a `machine_type` or replica count, so Vertex deployed it with "automatic resources" — evidently sized considerably larger/more redundant than that original guess assumed. $0.87/hr is still just one data point, not a guaranteed rate — if you get another real bill, compare it and update `VECTOR_SEARCH_HOURLY_RATE_ESTIMATE_USD` again if it's drifted. At $0.87/hr, leaving the endpoint deployed continuously runs **~$21/day, ~$625/month** — a much stronger case for keeping it undeployed by default and only starting it while actively testing (see the Start/Stop controls above).

**Automatic, in-process schedule** (e.g. business hours only): set in `.env`:
```env
VECTOR_SEARCH_SCHEDULE_ENABLED=true
VECTOR_SEARCH_DEPLOY_CRON=0 9 * * 1-5     # 9am Mon-Fri
VECTOR_SEARCH_UNDEPLOY_CRON=0 18 * * 1-5  # 6pm Mon-Fri
VECTOR_SEARCH_SCHEDULE_TIMEZONE=Asia/Singapore
```
**This only runs while the `uvicorn` process itself is alive** — stopping the app between sessions means nothing undeploys/redeploys on your behalf, and the index sits in whatever state it was last left in. It's a convenience for a continuously-running deployment, not a substitute for an external scheduler (Cloud Scheduler + a small Cloud Function calling the same deploy/undeploy) if you need the schedule to hold regardless of whether this app happens to be running.

**Security note:** the `/admin/vector-index/*` endpoints have no auth, same as every other endpoint in this API (see "Known gaps" below) — but unlike `/ingest` or `/search`, these directly control real billing and can take text search offline. Put them behind auth before exposing this API beyond localhost.

## Run the API

```bash
uvicorn app.main:app --reload --port 8000
```

A web UI is served at `http://localhost:8000/` — upload, text search, and image similarity search, all from the browser. `/docs` still gives you FastAPI's interactive Swagger UI if you'd rather hit the routes directly.

## Use it

**Ingest an image:**
```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@/path/to/certificate.jpg"
```
Returns `text_indexed: true/false` — images with no detectable text still get ingested (and become searchable via `/search/image`), they just won't appear in `/search` (text) results.

**Ingest multiple images at once:**
```bash
curl -X POST "http://localhost:8000/ingest/batch" \
  -F "files=@/path/to/one.jpg" \
  -F "files=@/path/to/two.jpg" \
  -F "files=@/path/to/three.jpg"
```
Each file succeeds or fails independently — one bad image doesn't abort the rest. Response is a list of per-file results (`status: "ok"` or `"error"`). The web UI's Ingest panel supports multi-select/drag for this too.

**Search by text:**
```bash
curl "http://localhost:8000/search?q=gold%20purity%2018k&top_k=5"
```

**Search by visual similarity** — upload an image, get back images that *look* similar, regardless of their text content:
```bash
curl -X POST "http://localhost:8000/search/image?top_k=5" \
  -F "file=@/path/to/query.jpg"
```

**Pull directly from Airtable** — if your source images already live in an Airtable base, skip the manual download-then-upload step:
```bash
curl -X POST "http://localhost:8000/ingest/airtable"
```
Requires `AIRTABLE_API_KEY` / `AIRTABLE_BASE_ID` / `AIRTABLE_TABLE_NAME` set in `.env` first. Optional query params: `?table=...` / `?view=...` / `?limit=5` to override the `.env` defaults for one-off runs, and `?dry_run=true` to preview what would be ingested (record/field/attachment detection) without actually calling Vision/Vertex/GCS. Safe to re-run any time — it's idempotent, keyed by Airtable's own attachment ID, so a repeat run updates existing docs instead of duplicating them. Every other field on each Airtable record (title, tags, description, whatever your base has) is auto-detected and stored alongside the image as metadata — no need to tell it which fields matter.

**Every Airtable-sourced ingest writes results back onto the source record** — `AIRTABLE_WRITE_OCR_TEXT_FIELD` / `AIRTABLE_WRITE_INDEXED_FIELD` / `AIRTABLE_WRITE_DOC_ID_FIELD` in `.env` (default field names: "OCR Text", "Indexed", "Doc ID") must already exist in your base with a compatible type (long text / checkbox / single line text respectively) — this app doesn't create fields for you. Write-back is best-effort: if a field is missing or the wrong type, that one write fails with a message in the response's `detail`, but the ingest itself (GCS/Firestore/Vertex) still succeeded — it's not rolled back.

## Airtable webhook — notify-only, you confirm before anything ingests

Rather than auto-ingesting the moment something changes in Airtable, this app takes a **notify-then-confirm** approach: an Airtable Automation pings this app when a record changes, the app queues it, and a person reviews and explicitly confirms (or dismisses) each one — nothing gets OCR'd, embedded, indexed, or written back to Airtable until that confirm step.

**Set up the Airtable side** (once per base):
1. In your Airtable base: **Automations → Create automation**.
2. Trigger: **"When a record is created"** (or **"updated"**, or both as separate automations) on your table.
3. Action: **"Send webhook"**.
4. URL: `http://YOUR_HOST:8000/webhooks/airtable` (if you set `AIRTABLE_WEBHOOK_SECRET`, append `?secret=YOUR_SECRET`).
5. Body (JSON): `{"record_id": "{{Record ID}}", "table": "YourTableName"}` — insert the trigger record's ID token via Airtable's field-picker in the webhook body editor.
6. Turn the automation on.

**Review and confirm** — either from the web UI's "Pending from Airtable" panel (shows a live preview: the image + every other field on the record, with **Confirm & ingest** / **Dismiss** buttons), or via API:
```bash
curl http://localhost:8000/airtable/pending                              # list what's waiting for review
curl -X POST http://localhost:8000/airtable/pending/recXXXXXXXXXXXXXX/confirm   # ingest it now + write results back
curl -X POST http://localhost:8000/airtable/pending/recXXXXXXXXXXXXXX/dismiss   # reject without ingesting
```

Confirming re-fetches the record fresh from Airtable (rather than trusting anything cached from when the webhook fired) — Airtable's attachment URLs expire, so this avoids acting on a stale link if you don't get to reviewing something right away.

## Notes on quality

- `DOCUMENT_TEXT_DETECTION` (used here) works better than `TEXT_DETECTION` for both dense text (scans) and sparse text (labels) — no reason to switch.
- `text-embedding-004` uses different `task_type` hints for storage vs. query (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) — this is already wired in and meaningfully improves recall over using the same type for both.
- Image URLs in every response are **signed URLs** (`storage.generate_signed_url()`), not plain public links — Public Access Prevention is enforced on this project, which blocks making the bucket/objects public at all. Signed URLs are minted fresh at response time (they expire, 60 min by default) from the permanent `gcs_uri` stored in Firestore — never store a signed URL itself, it'll go stale. Requires `GOOGLE_APPLICATION_CREDENTIALS` to point at an actual service account JSON key (not ambient/impersonated credentials), since signing happens locally with that key's private key.

## Airtable as source, GCS as working copy

If your images originate in Airtable, `/ingest/airtable` still uploads a full copy of each one into GCS rather than referencing the Airtable attachment URL directly — this is deliberate, not an oversight. Airtable's own attachment URLs expire a couple hours after being returned by the API, so they can't be stored long-term as a stable pointer the way `gcs_uri` is; anything durable (OCR, embeddings, signed-URL previews weeks later) needs a copy that isn't going to 404. The GCS copy is genuinely a second copy of the bytes, not just metadata — worth knowing if storage cost ever becomes a concern. Also worth knowing: `doc_id` is derived from the Airtable *attachment's* own id, not the record id. Replacing an attachment in Airtable (delete + re-upload) gets a new attachment id, so re-running `/ingest/airtable` correctly ingests the new image as a new doc — but the old doc (GCS copy, Firestore row, and its entry in the Vertex text index if it had one) is never cleaned up. There's no delete/orphan-detection path here yet; add one if attachments get replaced often enough for that to matter.

## Image similarity search (`/search/image`)

Text search only ever matches on OCR'd text — it has no idea what an image *looks* like. `/search/image` covers that: every ingested image also gets embedded with Vertex AI's `multimodalembedding@001` (1408-dim, a different vector space than the text embeddings, so it's never compared against them), stored as a field on that document's Firestore row.

Deliberately **not** backed by a second Vertex AI Vector Search index — that would mean a second deployed endpoint billing hourly on top of the text one. Instead, `/search/image` does a brute-force cosine-similarity scan over every stored image embedding in Firestore (see `app/similarity.py`). Fine up to a few thousand images; degrades linearly past that. If you outgrow it, stand up a second index the same way `setup_vector_index.py` does for text, and swap `app/similarity.py`'s linear scan for a `vector_search.search()`-style call against it — the embedding step (`embeddings.embed_image`) doesn't need to change.

## Known gaps to fill in before production

- No auth on the FastAPI endpoints — add an API key or IAM-based auth in front.
- No retry/backoff around the Vision/Vertex calls — add for production traffic.
- `distance` from Vector Search is cosine distance, not a 0-1 similarity score — invert/normalize if you want to show a "match %" to users.
