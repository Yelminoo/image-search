# Changelog

## 2026-09-10 (later same day) — Moved image similarity search onto FAISS too

**Status:** implemented, staged for testing (not yet committed/pushed).

**Why:** the text-search FAISS switch below only covered `/search`. `/search/image` still used the original pure-Python brute-force scan (`app/similarity.py`) — asked to move that over too ("all replace by FAISS").

**What changed:**
1. `app/faiss_index.py` refactored from two standalone module-level functions into a reusable `FaissIndex` class, instantiated twice: `text_index` (768-dim) and `image_index` (1408-dim) — separate vector spaces, never compared against each other. All the same exact-search/upsert/rebuild behavior as before, just parameterized instead of duplicated.
2. `app/metadata_store.py` — added `get_all_image_embedding_pairs()` (mirrors `get_all_with_text_embeddings()`) for the image index's startup rebuild. `get_all_with_image_embeddings()` (the old full-dict fetcher used by `app/similarity.py`) is now legacy, kept but unused.
3. `_ingest_one()` — now upserts into `image_index` for every ingest (every image gets an image embedding, text or not), same as it already did for `text_index` when text was found.
4. `/search/image` — now calls `faiss_index.image_index.search()` instead of `similarity.search_similar_images()` + `metadata_store.get_all_with_image_embeddings()`. Response shape unchanged (`doc_id`, `similarity`, `image_url`, `ocr_text`) — no client/UI changes needed.
5. `main.py`'s lifespan now rebuilds both indexes at startup.
6. `/admin/faiss-index/status` now reports both `text_documents_indexed` and `image_documents_indexed`.
7. Web UI (`app/static/index.html`) — the top status bar still referenced the now-removed `/admin/vector-index/*` endpoints from the earlier text-search switch (would have shown "Status unavailable" errors on load). Replaced with a simple live document-count display backed by the new `/admin/faiss-index/status`. No changes needed to the actual ingest/search panels — same API contract throughout.
8. `app/similarity.py` retired (unwired, kept in the repo) — same pattern as `vector_search.py`/`scheduler.py`/`usage_meter.py`.

**Verified live**, not just compiled — ran the actual app against real ingested data (7 jewelry photos already in GCS/Firestore from earlier testing):
- Before this change: queried `/search/image` with a real ring photo (downloaded from GCS) via the old brute-force path. Top match: the duplicate copy of the same photo, similarity ≈1.0. Second: a genuinely different but visually-similar bangle bracelet, similarity 0.577. Rest: unrelated images, ~0.47–0.48.
- After this change (server restarted to load the new code): re-ran the identical query against the new FAISS-backed path. **Byte-identical results** — same ranking, same similarity scores to 6 decimal places — confirming FAISS's exact search is truly equivalent to what it replaced.
- Confirmed live upsert works without a restart: ingested the same photo as a new document, watched `image_documents_indexed` go from 7 → 8 via the status endpoint, then cleaned up that test document (Firestore doc + GCS object) since it wasn't real data, and restarted once more to confirm a clean rebuild back to 7.

---

## 2026-09-10 — Replaced Vertex AI Vector Search with in-process FAISS for text search

**Status:** implemented, staged for testing (not yet committed/pushed).

**Why:** the deployed Vertex AI Vector Search endpoint (used only by `/search`, text search) was found to cost **~$0.87/hour** in practice (derived from a real charge: 200 THB for 7 hours deployed) — roughly 11x an earlier, unconfirmed estimate of $0.077/hour. Left running continuously that's ~$626/month; even business-hours-only (~173 hrs/month) is ~$150/month. `/search/image` (image similarity search) was never affected — it already used a brute-force approach from when it was first built, specifically to avoid standing up a second billed endpoint.

**Options considered:**
1. Reduce deployed hours (manual start/stop, or the schedule already built in `app/scheduler.py`) — cuts cost but doesn't eliminate it, and still requires remembering to manage a billed resource.
2. Numpy-vectorized brute force — same architecture `/search/image` already uses (embeddings as a Firestore field, scored at query time), upgraded from a pure-Python loop to numpy. $0 infrastructure cost, zero new dependencies. Initially the chosen direction (see benchmark below), superseded by option 3 on request.
3. **FAISS, run in-process (chosen and implemented)** — same $0 infrastructure cost as option 2, same exact (non-approximate) results at this project's scale, better long-term scaling headroom if the collection grows well past what a linear numpy scan stays comfortable with. `app/faiss_index.py` uses `IndexFlatIP` (exact search) with vectors L2-normalized so inner product = cosine similarity, rebuilt from Firestore at app startup and kept in sync incrementally via `upsert()` on ingest.

**Benchmark that informed the decision** (run on real hardware, `text-embedding-004`'s 768 dimensions, top-10 query, results compared for exact match across all three methods):

| Docs | Pure Python (old `app/similarity.py`-style loop) | Numpy (vectorized) | FAISS (`IndexFlatIP`, exact) | Same results? |
|---|---|---|---|---|
| 100 | 8.3 ms | 0.01 ms | 0.04 ms | ✅ |
| 1,000 | 92 ms | 0.12 ms | 1.2 ms | ✅ |
| 10,000 | 937 ms | 1.0 ms | 20 ms | ✅ |
| 100,000 | 10.0 s | 11.7 ms | 133 ms | ✅ |

All three produce **identical top-k results at every scale tested** — FAISS's exact/Flat index is mathematically brute force too, just implemented in optimized C++/BLAS rather than a Python loop, so there's no accuracy trade-off between any of these at this project's scale.

**What actually changed:**
1. `app/faiss_index.py` (new) — in-memory FAISS `IndexIDMap(IndexFlatIP(768))`. Doc ids are mapped to FAISS's required int64 ids via a deterministic SHA1-derived hash (no separate id-allocation table to persist). `rebuild_from()` (called once at startup from `main.py`'s lifespan), `upsert()` (called from `_ingest_one()` on every ingest with detected text), `search()`.
2. `app/metadata_store.py` — added `get_all_with_text_embeddings()` (startup rebuild), `get_all_missing_text_embeddings()` + `update_text_embedding()` (backfill script).
3. `_ingest_one()` in `main.py` — stopped calling `vector_search.upsert_vector()`; now stores `text_embedding` as a Firestore field (same pattern as the existing `image_embedding`) and calls `faiss_index.upsert()`.
4. `/search` in `main.py` — stopped calling `vector_search.search()`; now calls `faiss_index.search()`. Response keeps the existing `distance` field name (computed as `1 - similarity`) so no API/UI consumers needed to change.
5. `backfill_text_embeddings.py` (new, project root) — one-time script for docs ingested before this switch (they have `ocr_text` but never got `text_embedding` written, since it used to go straight to Vertex). Re-embeds from the already-stored text; no re-upload/re-OCR. Checked against the live project at implementation time: **0 documents needed it** (the 7 existing docs all have empty `ocr_text` — jewelry photos with no detected text) — script is ready for whenever it's actually needed.
6. Retired (unwired from `main.py`, files kept in place): `app/vector_search.py`, `app/scheduler.py`, `app/usage_meter.py`, `setup_vector_index.py`. The `/admin/vector-index/*` endpoints and the cost-control web UI panel are gone, replaced by a single `GET /admin/faiss-index/status`.
7. `requirements.txt` — added `faiss-cpu==1.15.0`, `numpy==2.5.2` (now direct dependencies, not just transitive).

**Verified before staging:**
- `python -m py_compile` on all touched files — clean.
- Full app import + route listing — clean, all expected routes present, old `/admin/vector-index/*` routes gone.
- `faiss_index.py` core logic tested standalone (synthetic 768-dim vectors): `rebuild_from()`, self-match search (similarity ≈ 1.0 for a document queried with its own vector), `upsert()` of a new doc, `upsert()` overwriting an existing doc_id without duplicating — all passed.
- Real app lifespan exercised end-to-end (live, read-only Firestore call) — starts cleanly, index rebuilds from Firestore (0 docs, as expected — see point 5 above).

**Not yet done, deliberately:**
- Not committed/pushed to git — staged locally for testing first.
- Existing Vertex index + endpoint resources in GCP still exist (undeployed, $0 cost) — not deleted. Worth doing once confident nothing needs them, to remove any future accidental-redeploy risk.
- Image similarity search (`/search/image`) still uses the pure-Python brute-force scan in `app/similarity.py`, not FAISS — nothing has required it to be faster yet. Straightforward to move over later (second `IndexFlatIP`, 1408-dim, same startup-rebuild pattern).

---

## 2026-09-05 — Initial build

- OCR + semantic text search + image similarity search pipeline: Cloud Vision (OCR) → Vertex AI (text + multimodal embeddings) → Vertex AI Vector Search (text) / brute-force Firestore scan (image similarity), with Firestore holding metadata and GCS holding raw images.
- Web UI (`app/static/index.html`): ingest (single + multi-file), text search, image similarity search.
- Airtable integration: bulk pull (`/ingest/airtable`), write-back of OCR results onto source records, notify-then-confirm webhook queue (`/webhooks/airtable` → `/airtable/pending`) rather than auto-ingesting on every change.
- Vector Search cost controls: manual Start/Stop (web UI + `/admin/vector-index/*`), optional business-hours schedule (`app/scheduler.py`), self-tracked usage meter (`app/usage_meter.py`).
