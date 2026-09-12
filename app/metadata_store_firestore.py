"""
RETIRED — the original Firestore-backed metadata store. Replaced by the
SQLite-backed app/metadata_store.py (same module name, same function
signatures — a drop-in swap, nothing else in the codebase needed to
change). Kept here for reference, and as the source side of the one-time
migration in migrate_firestore_to_sqlite.py. Not imported from main.py.

Why it was replaced: at this project's scale (a few hundred documents,
one local server), Firestore's actual usage here was just a durable
key-value store — write one doc per ingest, read everything once at
startup to rebuild the FAISS indexes, look up a handful by id per search.
No complex queries. Same reasoning that already justified swapping Vertex
AI Vector Search for in-process FAISS (see CHANGELOG.md): a managed cloud
service bought nothing here that a local file didn't already provide for
free, with less latency and zero billing surface.

The trade-off, so it isn't lost when reading this later: SQLite ties this
data to whatever machine runs the app. Firestore survives that machine
dying and is naturally shared across processes/servers. Fine while this
runs locally; worth reconsidering if this ever gets deployed to ephemeral
cloud infrastructure (mount a persistent volume, or sync to GCS, or move
back to a managed store at that point).
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
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("image_embedding"):
            data["doc_id"] = doc.id
            results.append(data)
    return results


def get_all_image_embedding_pairs() -> list:
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("image_embedding"):
            results.append((doc.id, data["image_embedding"]))
    return results


def get_all_with_text_embeddings() -> list:
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("text_embedding"):
            results.append((doc.id, data["text_embedding"]))
    return results


def get_all_missing_text_embeddings() -> list:
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        if data.get("ocr_text") and not data.get("text_embedding"):
            results.append((doc.id, data["ocr_text"]))
    return results


def update_text_embedding(doc_id: str, embedding: list) -> None:
    _get_client().collection(COLLECTION).document(doc_id).update({"text_embedding": embedding})


def get_all_source_summaries() -> list:
    client = _get_client()
    results = []
    for doc in client.collection(COLLECTION).stream():
        data = doc.to_dict()
        results.append({
            "source": data.get("source", "upload"),
            "airtable_base_id": data.get("airtable_base_id"),
            "airtable_table": data.get("airtable_table"),
            "text_indexed": bool(data.get("text_embedding")),
            "created_at": data.get("created_at"),
        })
    return results


def get_all_raw() -> list:
    """Every document's full raw data + id — used only by the one-time migration script."""
    client = _get_client()
    return [(doc.id, doc.to_dict()) for doc in client.collection(COLLECTION).stream()]
