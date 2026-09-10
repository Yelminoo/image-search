"""
Metadata store using Firestore.

Vertex AI Vector Search only stores vector IDs + the vectors themselves —
it does not store your original text, image URL, or any other fields.
So every indexed vector needs a matching "datapoint_id" that you can look
up afterward to get the actual content back. Firestore is the natural
Google-native choice for that side-table (fast key lookups, no schema
migrations, same GCP project/billing).
"""
from google.cloud import firestore
from app.config import settings

_client = None
COLLECTION = "ocr_documents"


def _get_client() -> firestore.Client:
    global _client
    if _client is None:
        _client = firestore.Client(project=settings.project_id)
    return _client


def save_metadata(doc_id: str, gcs_uri: str, image_url: str, ocr_text: str, extra: dict = None) -> None:
    client = _get_client()
    data = {
        "gcs_uri": gcs_uri,
        "image_url": image_url,
        "ocr_text": ocr_text,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    if extra:
        data.update(extra)
    client.collection(COLLECTION).document(doc_id).set(data)


def get_metadata_batch(doc_ids: list) -> dict:
    """Fetch metadata for a list of doc_ids, returned as {doc_id: data}."""
    client = _get_client()
    results = {}
    for doc_id in doc_ids:
        snap = client.collection(COLLECTION).document(doc_id).get()
        if snap.exists:
            results[doc_id] = snap.to_dict()
    return results


def get_all_with_image_embeddings() -> list:
    """
    Fetch every document that has an image_embedding field, as full
    metadata dicts. LEGACY — was used by the pure-Python brute-force image
    search in app/similarity.py, retired in favor of FAISS (see
    get_all_image_embedding_pairs below). Kept for reference; no longer
    called from main.py.
    """
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("image_embedding"):
            data["doc_id"] = doc.id
            results.append(data)
    return results


def get_all_image_embedding_pairs() -> list:
    """
    Fetch every (doc_id, image_embedding) pair — used once at app startup
    to rebuild the in-memory FAISS image index (see app/faiss_index.py).
    Mirrors get_all_with_text_embeddings() below for the text index.
    """
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("image_embedding"):
            results.append((doc.id, data["image_embedding"]))
    return results


def get_all_with_text_embeddings() -> list:
    """
    Fetch every (doc_id, text_embedding) pair — used once at app startup to
    rebuild the in-memory FAISS text index (see app/faiss_index.py).
    Firestore is the durable source of truth for embeddings; the FAISS
    index itself is rebuilt from here on every process start rather than
    persisted to its own file.
    """
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("text_embedding"):
            results.append((doc.id, data["text_embedding"]))
    return results


def get_all_missing_text_embeddings() -> list:
    """
    Fetch every (doc_id, ocr_text) pair for documents that have OCR'd text
    but no text_embedding yet — used by backfill_text_embeddings.py for
    documents ingested before the switch to FAISS (back when text
    embeddings went straight to Vertex Vector Search instead of being
    stored here).
    """
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("ocr_text") and not data.get("text_embedding"):
            results.append((doc.id, data["ocr_text"]))
    return results


def update_text_embedding(doc_id: str, embedding: list) -> None:
    """Write just the text_embedding field onto an existing document (used by the backfill script)."""
    _get_client().collection(COLLECTION).document(doc_id).update({"text_embedding": embedding})
