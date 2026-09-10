"""
One-time backfill: writes a text_embedding field onto every existing
Firestore document that has OCR'd text but no text_embedding yet.

Why this is needed: before the switch to FAISS, text embeddings went
straight to Vertex AI Vector Search and were never stored in Firestore.
The FAISS index (app/faiss_index.py) rebuilds itself from Firestore's
text_embedding field at app startup — so without this backfill, every
document ingested before the switch would be invisible to text search
(still fully present, just not embedded for search) until re-ingested.

This does NOT re-upload images or re-run OCR — ocr_text is already
stored, so this only re-embeds that existing text (a cheap Vertex AI
call) and writes the result back onto the same document.

Usage:
    python backfill_text_embeddings.py
"""
from app import embeddings, metadata_store

pending = metadata_store.get_all_missing_text_embeddings()

if not pending:
    print("Nothing to backfill — every document with OCR text already has a text_embedding.")
else:
    print(f"Backfilling {len(pending)} document(s)...")
    for i, (doc_id, ocr_text) in enumerate(pending, 1):
        try:
            embedding = embeddings.embed_document(ocr_text)
            metadata_store.update_text_embedding(doc_id, embedding)
            print(f"  [{i}/{len(pending)}] {doc_id}: OK")
        except Exception as e:
            print(f"  [{i}/{len(pending)}] {doc_id}: FAILED - {e}")

print("\nDone. Restart the app (or call the rebuild) to load the new embeddings into the FAISS index.")
