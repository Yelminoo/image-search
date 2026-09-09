"""
Brute-force image similarity search.

For text search, vector_search.py talks to a real Vertex AI Vector Search
(ANN) index — necessary at scale, but it means a second deployed endpoint
billing hourly if we stood one up for images too. At small scale, a plain
linear scan over embeddings already sitting in Firestore is simpler, adds
no extra deployed infrastructure, and is fast enough (a few thousand
documents is well under a second in pure Python).

If the collection grows past that, replace this module's approach with a
second Vertex AI Vector Search index (see setup_vector_index.py for the
pattern) — the rest of the ingest/search code barely changes, since the
embedding step (embeddings.embed_image) is already separate from how the
resulting vector gets searched.
"""
import math
from typing import Dict, List


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def search_similar_images(
    query_embedding: List[float],
    candidates: List[Dict],
    top_k: int = 10,
) -> List[Dict]:
    """
    candidates: dicts each containing at least "doc_id" and "image_embedding".
    Returns the top_k candidates sorted by similarity descending (1.0 = identical
    direction, -1.0 = opposite), each with a "similarity" field added.
    """
    scored = [
        {**c, "similarity": _cosine_similarity(query_embedding, c["image_embedding"])}
        for c in candidates
    ]
    scored.sort(key=lambda c: c["similarity"], reverse=True)
    return scored[:top_k]
