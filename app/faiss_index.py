"""
In-process FAISS indexes — used for BOTH text search (768-dim) and image
similarity search (1408-dim). Replaces the deployed Vertex AI Vector
Search endpoint (text search's original backend, retired — see
CHANGELOG.md for why: it cost ~$0.87/hour just for existing) and the
pure-Python brute-force scan in app/similarity.py (image search's
original backend, also retired — same approach, just never needed to be
faster until now).

FAISS's IndexFlatIP does EXACT (not approximate) search — mathematically
identical results to a brute-force cosine-similarity scan, just computed
via optimized BLAS/SIMD instead of a Python loop. No accuracy trade-off
versus brute force at this project's scale (see the benchmark in
CHANGELOG.md — verified up to 100K documents).

Persistence: Firestore holds the durable embeddings (`text_embedding` /
`image_embedding` fields on each document — see metadata_store.py). Both
indexes below are rebuilt from Firestore once at app startup (see
main.py's lifespan) and kept in sync incrementally via upsert() as
documents are ingested. There's no separate FAISS index file to manage.

Text and image embeddings come from different models and live in
different vector spaces (768-dim vs. 1408-dim) — they get separate
FaissIndex instances (text_index / image_index below), never compared
against each other.

Vectors are L2-normalized before adding/querying, since IndexFlatIP
computes a plain inner product — on normalized vectors that's equivalent
to cosine similarity. embeddings.py returns raw (non-normalized) vectors,
so normalization happens here, not upstream.
"""
import hashlib
from typing import Dict, List, Tuple

import faiss
import numpy as np

from app.config import settings


def _doc_id_to_faiss_id(doc_id: str) -> int:
    """
    Deterministic, stable positive int64 derived from a string doc_id —
    avoids needing a separate persisted id-allocation table. Collision
    probability at 60 bits is negligible at any scale this project will
    realistically reach (birthday-bound ~2^30 documents for a 50% chance
    of one collision).
    """
    digest = hashlib.sha1(doc_id.encode("utf-8")).hexdigest()
    return int(digest[:15], 16)  # 60 bits — safely positive, fits int64


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


class FaissIndex:
    """One exact-search (IndexFlatIP) FAISS index over a fixed vector dimension."""

    def __init__(self, dim: int):
        self.dim = dim
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
        self._id_to_doc_id: Dict[int, str] = {}

    def rebuild_from(self, documents: List[Tuple[str, List[float]]]) -> None:
        """Replace the entire index from a fresh set of (doc_id, embedding) pairs — called once at app startup."""
        self._index = faiss.IndexIDMap(faiss.IndexFlatIP(self.dim))
        self._id_to_doc_id = {}

        if not documents:
            return

        ids = np.array([_doc_id_to_faiss_id(doc_id) for doc_id, _ in documents], dtype="int64")
        vecs = np.stack([_normalize(np.array(emb, dtype="float32")) for _, emb in documents]).astype("float32")
        self._index.add_with_ids(vecs, ids)
        for doc_id, _ in documents:
            self._id_to_doc_id[_doc_id_to_faiss_id(doc_id)] = doc_id

    def upsert(self, doc_id: str, embedding: List[float]) -> None:
        """Add or update a single vector. IndexFlatIP has no in-place update, so an existing entry for this doc_id is removed first."""
        faiss_id = _doc_id_to_faiss_id(doc_id)

        if faiss_id in self._id_to_doc_id:
            self._index.remove_ids(np.array([faiss_id], dtype="int64"))

        vec = _normalize(np.array(embedding, dtype="float32")).reshape(1, -1)
        self._index.add_with_ids(vec, np.array([faiss_id], dtype="int64"))
        self._id_to_doc_id[faiss_id] = doc_id

    def search(self, query_embedding: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        """Returns [(doc_id, similarity), ...] sorted by similarity descending. similarity is cosine similarity in [-1, 1] (1 = identical direction)."""
        if self._index.ntotal == 0:
            return []

        q = _normalize(np.array(query_embedding, dtype="float32")).reshape(1, -1)
        k = min(top_k, self._index.ntotal)
        similarities, ids = self._index.search(q, k)

        results = []
        for sim, faiss_id in zip(similarities[0], ids[0]):
            if faiss_id == -1:  # FAISS pads with -1 when fewer than k results exist
                continue
            doc_id = self._id_to_doc_id.get(int(faiss_id))
            if doc_id:
                results.append((doc_id, float(sim)))
        return results

    def size(self) -> int:
        return self._index.ntotal


# Two singleton indexes, one per embedding model/vector-space.
text_index = FaissIndex(dim=settings.embedding_dim)
image_index = FaissIndex(dim=settings.image_embedding_dim)
