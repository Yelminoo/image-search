# OCR + Semantic Search — Google Stack + FAISS

Cloud Vision (OCR) → Vertex AI (text + image embeddings) → in-process FAISS search, with a local SQLite database holding metadata + embeddings (the durable source of truth) and GCS holding the raw images.

**Both text search and image similarity search run as separate in-memory FAISS indexes inside this app** — there is no separately deployed/billed search infrastructure. See "Why no deployed Vector Search" below for how this evolved and why.

## Why this shape

SQLite (`local_data.db`, gitignored) is the only durable store here — for text/metadata *and* for every embedding (text + image). FAISS's index lives entirely in this app's memory and is rebuilt from it once at startup (see `app/faiss_index.py`), so there's no separate index file to manage and no risk of the search index and the metadata drifting out of sync — SQLite is always the truth, FAISS is just a fast, disposable, rebuildable view over it.

This used to be Firestore (see `app/metadata_store_firestore.py`, kept for reference, and CHANGELOG.md for the full reasoning and a real migration verified against production data) — same logic that already justified swapping Vertex AI Vector Search for in-process FAISS: at this project's scale and access pattern, a managed cloud service bought nothing a local file doesn't provide for free. **The trade-off:** this ties the data to whatever machine runs the app — fine locally, worth reconsidering (mounted volume / periodic GCS backup / back to a managed store) if this ever deploys to ephemeral cloud infrastructure.

## One-time GCP setup

1. Create a GCP project, enable billing.
2. Enable these APIs: Cloud Vision API, Vertex AI API, Cloud Storage. (Firestore is no longer used — see "Why this shape" above.)
3. Create a GCS bucket for images.
4. Create a service account with roles: `Vertex AI User`, `Storage Object Admin` (or `objectCreator` for least-privilege). Download its JSON key. (No dedicated IAM role is needed for Cloud Vision API itself — access is gated by the API being enabled + valid credentials from the project, not a resource-level role.)
5. Copy `.env.example` to `.env` and fill in your project ID, bucket name, and the path to the service account key.

There is **no infrastructure deployment step** beyond this — no vector index to build, no endpoint to deploy, nothing that bills by the hour. Install deps and run:
```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The whole app is gated behind a login (`ADMIN_EMAIL`/`ADMIN_PASSWORD` in `.env` — set these before first run, see `.env.example`). Visit `http://localhost:8000/` — you'll land on `/login` first, then `/ingest` (uploads, Airtable pull, pending-review queue) and `/search` (text + image similarity search) as separate pages once signed in. `/docs` still gives you FastAPI's interactive Swagger UI, also behind the same login.

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

**Every Airtable-sourced ingest writes results back onto the source record** — `AIRTABLE_WRITE_OCR_TEXT_FIELD` / `AIRTABLE_WRITE_INDEXED_FIELD` / `AIRTABLE_WRITE_DOC_ID_FIELD` in `.env` (default field names: "OCR Text", "Indexed", "Doc ID") must already exist in your base with a compatible type (long text / checkbox / single line text respectively) — this app doesn't create fields for you. Write-back is best-effort: if a field is missing or the wrong type, that one write fails with a message in the response's `detail`, but the ingest itself (GCS/local database/search-index) still succeeded — it's not rolled back.

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

## Refreshing metadata without a full re-ingest

If a record's field values change (SKU renamed, category updated) but the photo itself didn't, a full re-ingest wastes a real OCR/Vertex/GCS call just to pick up new field values. This updates metadata only:
```bash
curl -X POST "http://localhost:8000/airtable/records/recXXXXXXXXXXXXXX/refresh-metadata?base=...&table=...&text_fields=Item+Name"
```
Updates `airtable_fields` to the record's current values on every doc tied to that record (a multi-image record produces multiple docs, all refreshed). If `text_fields` is given, also recombines each doc's already-stored `ocr_text` with the new field values and re-embeds just that text — one cheap Vertex call per doc. **The image itself is never re-downloaded, OCR never re-runs, `image_embedding` is never touched** — verified by SHA1 checksum during development (see `CHANGELOG.md`) that it stays byte-for-byte identical across a refresh.

## Why no deployed Vector Search (and the FAISS switch)

This project originally used **Vertex AI Vector Search** (a deployed, hourly-billed ANN index) for text search. It was replaced after a real charge — **200 THB for 7 hours deployed, ≈$0.87/hour** — turned out to be ~11x higher than an initial estimate, and would run **~$150/month even limited to business hours, ~$625/month if left running continuously**. Full reasoning and the benchmark that informed the replacement are in `CHANGELOG.md`.

**What replaced it:** every embedding (text *and* image) is stored as a plain field on its SQLite row. At app startup, `app/faiss_index.py` builds two in-memory FAISS `IndexFlatIP` indexes — one 768-dim (`text_index`, from `text_embedding`) and one 1408-dim (`image_index`, from `image_embedding`), separate vector spaces, never compared against each other. This is **exact** search (mathematically identical to brute-force cosine similarity, just computed via optimized BLAS/SIMD instead of a Python loop), not an approximation. New ingests call `upsert()` on the relevant index(es) to stay current without a restart.

**Cost:** $0 for search infrastructure, on both text and image search. What's left is only the same fractions-of-a-cent per-call OCR/embedding cost that was already negligible.

**The trade-off:** both FAISS indexes are in-memory and rebuilt from SQLite on every process restart — cheap at this project's scale (see the benchmark below), but means a restart briefly has empty indexes until the rebuild completes (milliseconds in practice).

**Benchmark that informed the decision** (768-dim vectors matching `text-embedding-004`, top-10 query, run on real hardware — see `CHANGELOG.md` for the full write-up):

| Docs | Pure Python loop | Numpy (vectorized) | FAISS (`IndexFlatIP`, exact) |
|---|---|---|---|
| 100 | 8.3 ms | 0.01 ms | 0.04 ms |
| 1,000 | 92 ms | 0.12 ms | 1.2 ms |
| 10,000 | 937 ms | 1.0 ms | 20 ms |
| 100,000 | 10.0 s | 11.7 ms | 133 ms |

All three produce identical results at every scale — the only differentiator is speed, and FAISS/numpy both comfortably outperform a naive Python loop by orders of magnitude.

**Retired, not deleted:** `app/vector_search.py`, `app/scheduler.py`, `app/usage_meter.py`, and `setup_vector_index.py` are still in the repo (in case you want to look back at how the deployed-endpoint approach worked, or reintroduce it at a much larger scale) but are no longer imported or called from `main.py`. The `/admin/vector-index/*` endpoints and the cost-control web UI panel they backed are gone along with them — replaced by a single `GET /admin/faiss-index/status` (returns how many documents are currently indexed) since there's no deployed resource left to manage or meter.

**Existing data / backfill:** text embeddings used to go straight to Vertex Vector Search and were never stored in SQLite, so documents ingested before this switch have `ocr_text` but no `text_embedding` — invisible to the new FAISS-backed `/search` until backfilled. Run once, as needed:
```bash
python backfill_text_embeddings.py
```
Re-embeds each such document's already-stored `ocr_text` (no re-upload, no re-OCR) and writes `text_embedding` onto the existing SQLite row. Restart the app afterward (or wait for the next restart) to pick the new embeddings up into the FAISS index.

## Notes on quality

- `DOCUMENT_TEXT_DETECTION` (used here) works better than `TEXT_DETECTION` for both dense text (scans) and sparse text (labels) — no reason to switch.
- `text-embedding-004` uses different `task_type` hints for storage vs. query (`RETRIEVAL_DOCUMENT` vs `RETRIEVAL_QUERY`) — this is already wired in and meaningfully improves recall over using the same type for both.
- Image URLs in every response are **signed URLs** (`storage.generate_signed_url()`), not plain public links — Public Access Prevention is enforced on this project, which blocks making the bucket/objects public at all. Signed URLs are minted fresh at response time (they expire, 60 min by default) from the permanent `gcs_uri` stored in SQLite — never store a signed URL itself, it'll go stale. Requires `GOOGLE_APPLICATION_CREDENTIALS` to point at an actual service account JSON key (not ambient/impersonated credentials), since signing happens locally with that key's private key.
- `/search`'s `distance` field is kept (rather than renamed to `similarity`, like `/search/image` uses) for API/UI compatibility with the earlier Vertex-backed version — it's computed as `1 - cosine_similarity`, so lower still means "closer," same meaning as before.

## Airtable as source, GCS as working copy

If your images originate in Airtable, `/ingest/airtable` still uploads a full copy of each one into GCS rather than referencing the Airtable attachment URL directly — this is deliberate, not an oversight. Airtable's own attachment URLs expire a couple hours after being returned by the API, so they can't be stored long-term as a stable pointer the way `gcs_uri` is; anything durable (OCR, embeddings, signed-URL previews weeks later) needs a copy that isn't going to 404. The GCS copy is genuinely a second copy of the bytes, not just metadata — worth knowing if storage cost ever becomes a concern. Also worth knowing: `doc_id` is derived from the Airtable *attachment's* own id, not the record id. Replacing an attachment in Airtable (delete + re-upload) gets a new attachment id, so re-running `/ingest/airtable` correctly ingests the new image as a new doc — but the old doc (GCS copy, SQLite row, and its entries in the FAISS indexes) is never cleaned up. There's no delete/orphan-detection path here yet; add one if attachments get replaced often enough for that to matter.

## Image similarity search (`/search/image`)

Text search only ever matches on OCR'd text — it has no idea what an image *looks* like. `/search/image` covers that: every ingested image also gets embedded with Vertex AI's `multimodalembedding@001` (1408-dim, a different vector space than the text embeddings, so it's compared only against `faiss_index.image_index`, never `text_index`).

Also FAISS-backed, same as text search (`app/faiss_index.py`'s `image_index`) — verified to return byte-identical similarity scores to the pure-Python brute-force scan (`app/similarity.py`) it replaced, tested live against real ingested images before the switch. `app/similarity.py` is retired, kept in the repo for reference, no longer called.

## Known gaps to fill in before production

- ~~No auth on the FastAPI endpoints~~ — closed: every route (pages and API alike) requires a valid session cookie now (`app/auth.py`, see CHANGELOG.md). What's still missing: this is a single hardcoded account with a session cookie, not designed for programmatic/API-key access — fine for one human operator in a browser, not the shape you'd want for a second caller (a script, another service) to authenticate.
- No retry/backoff around the Vision/Vertex calls — add for production traffic.
- Both FAISS indexes are in-memory only, rebuilt from SQLite on every restart — fine at this project's scale (see the benchmark above), but means a brief empty-index window right after a restart, and no persistence across process crashes beyond what the SQLite file already holds.
- No delete endpoint for either GCS objects or SQLite rows — see "Airtable as source" above for where this bites (orphaned docs from replaced Airtable attachments).
