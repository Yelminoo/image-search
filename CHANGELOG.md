# Changelog

## Unreleased

### Planned: replace Vertex AI Vector Search with numpy brute-force for text search

**Status:** decided, not yet implemented.

**Why:** the deployed Vertex AI Vector Search endpoint (used only by `/search`, text search) was found to cost **~$0.87/hour** in practice (derived from a real charge: 200 THB for 7 hours deployed) — roughly 11x an earlier, unconfirmed estimate of $0.077/hour. Left running continuously that's ~$626/month; even business-hours-only (~173 hrs/month) is ~$150/month. `/search/image` (image similarity search) was never affected — it already used a brute-force approach from when it was first built, specifically to avoid standing up a second billed endpoint.

**Options considered:**
1. Reduce deployed hours (manual start/stop, or the schedule already built in `app/scheduler.py`) — cuts cost but doesn't eliminate it, and still requires remembering to manage a billed resource.
2. **FAISS**, run in-process — evaluated and benchmarked (see below). Rejected: not because it's worse, but because a plain numpy rewrite gets the same practical result (exact results, sub-20ms at 100K+ docs) with zero new dependencies and no index-persistence logic to build.
3. **Numpy-vectorized brute force** (chosen) — same architecture `/search/image` already uses (embeddings stored as a Firestore field, scored at query time), upgraded from the existing pure-Python loop to numpy for speed. $0 infrastructure cost, matches the pattern already proven in this codebase.

**Benchmark that informed the decision** (run on real hardware, `text-embedding-004`'s 768 dimensions, top-10 query, results compared for exact match across all three methods):

| Docs | Pure Python (current `app/similarity.py`) | Numpy (vectorized) | FAISS (`IndexFlatIP`, exact) | Same results? |
|---|---|---|---|---|
| 100 | 8.3 ms | 0.01 ms | 0.04 ms | ✅ |
| 1,000 | 92 ms | 0.12 ms | 1.2 ms | ✅ |
| 10,000 | 937 ms | 1.0 ms | 20 ms | ✅ |
| 100,000 | 10.0 s | 11.7 ms | 133 ms | ✅ |

All three produce **identical top-k results at every scale tested** — FAISS's exact/Flat index is mathematically brute force too, just implemented in optimized C++/BLAS rather than a Python loop, so there's no accuracy trade-off between any of these three at this project's scale. The only real differentiator is speed, and numpy alone already closes nearly all of that gap.

**Planned implementation** (see conversation — not yet applied to the codebase):
1. `_ingest_one()`: stop calling `vector_search.upsert_vector()`; store `text_embedding` as a Firestore field instead (same pattern as the existing `image_embedding`).
2. `/search`: stop calling `vector_search.search()`; fetch every doc with a `text_embedding`, score with numpy, return top-k. Response keeps the existing `distance` field name (computed as `1 - similarity`) so no API/UI consumers need to change.
3. One-time backfill script for already-ingested docs — they have `ocr_text` stored but never got `text_embedding` written (it used to go straight to Vertex); re-embed from the stored text, no re-upload/re-OCR needed.
4. Retire (stop wiring into `main.py`, keep the files): `app/scheduler.py`, `app/usage_meter.py`, `/admin/vector-index/*` endpoints, `setup_vector_index.py`, `app/vector_search.py`.
5. Separately, once confident: delete the actual Vertex index + endpoint resources in GCP (not just leave them undeployed) to remove any future accidental-redeploy risk. Real deletion, done deliberately, not automatically.

---

## 2026-09-05 — Initial build

- OCR + semantic text search + image similarity search pipeline: Cloud Vision (OCR) → Vertex AI (text + multimodal embeddings) → Vertex AI Vector Search (text) / brute-force Firestore scan (image similarity), with Firestore holding metadata and GCS holding raw images.
- Web UI (`app/static/index.html`): ingest (single + multi-file), text search, image similarity search.
- Airtable integration: bulk pull (`/ingest/airtable`), write-back of OCR results onto source records, notify-then-confirm webhook queue (`/webhooks/airtable` → `/airtable/pending`) rather than auto-ingesting on every change.
- Vector Search cost controls: manual Start/Stop (web UI + `/admin/vector-index/*`), optional business-hours schedule (`app/scheduler.py`), self-tracked usage meter (`app/usage_meter.py`).
