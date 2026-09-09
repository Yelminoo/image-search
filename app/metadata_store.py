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
    Fetch every document that has an image_embedding field, for brute-force
    image similarity search (see app/similarity.py).

    This reads the whole collection on every call — deliberate for now,
    since there's no separate Vector Search index for images. Fine up to a
    few thousand documents; if the collection grows past that, this is the
    place to swap in a real ANN index instead (see vector_search.py for the
    text-search equivalent).
    """
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("image_embedding"):
            data["doc_id"] = doc.id
            results.append(data)
    return results
