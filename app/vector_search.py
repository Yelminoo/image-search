"""
Vertex AI Vector Search (formerly Matching Engine) client.

Assumes an index + a deployed index endpoint already exist — see
setup_vector_index.py for the one-time creation script. This module only
handles the ongoing upsert/query traffic against that already-deployed
infrastructure.
"""
from typing import List
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine import (
    MatchingEngineIndex,
    MatchingEngineIndexEndpoint,
)
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import (
    MatchNeighbor,
)
from app.config import settings

_index = None
_endpoint = None


def _init():
    aiplatform.init(project=settings.project_id, location=settings.location)


def _get_index() -> MatchingEngineIndex:
    global _index
    if _index is None:
        _init()
        _index = MatchingEngineIndex(index_name=settings.vertex_index_id)
    return _index


def _get_endpoint() -> MatchingEngineIndexEndpoint:
    global _endpoint
    if _endpoint is None:
        _init()
        _endpoint = MatchingEngineIndexEndpoint(
            index_endpoint_name=settings.vertex_index_endpoint_id
        )
    return _endpoint


def upsert_vector(doc_id: str, embedding: List[float]) -> None:
    """Add or update a single vector in the live index (streaming update)."""
    index = _get_index()
    index.upsert_datapoints(
        datapoints=[
            {
                "datapoint_id": doc_id,
                "feature_vector": embedding,
            }
        ]
    )


def upsert_vectors_batch(doc_ids: List[str], embeddings: List[List[float]]) -> None:
    index = _get_index()
    datapoints = [
        {"datapoint_id": doc_id, "feature_vector": emb}
        for doc_id, emb in zip(doc_ids, embeddings)
    ]
    index.upsert_datapoints(datapoints=datapoints)


def search(embedding: List[float], top_k: int = 10) -> List[MatchNeighbor]:
    """Find nearest neighbors to the query embedding. Returns [(id, distance), ...]."""
    endpoint = _get_endpoint()
    response = endpoint.find_neighbors(
        deployed_index_id=settings.vertex_deployed_index_id,
        queries=[embedding],
        num_neighbors=top_k,
    )
    # response is a list (one per query) of lists of MatchNeighbor
    return response[0] if response else []


def delete_vector(doc_id: str) -> None:
    index = _get_index()
    index.remove_datapoints(datapoint_ids=[doc_id])


def is_deployed() -> bool:
    """Whether our configured deployed_index_id is currently live on the endpoint."""
    endpoint = _get_endpoint()
    return any(d.id == settings.vertex_deployed_index_id for d in endpoint.deployed_indexes)


def deploy_index() -> bool:
    """
    (Re)deploy the index onto its endpoint if not already deployed. This is
    what resumes hourly billing on the endpoint — see undeploy_index().
    No-op (returns False) if already deployed, so it's safe to call
    unconditionally from a schedule. Blocks until the deploy completes
    (this SDK call is synchronous), which can take a few minutes.
    """
    if is_deployed():
        return False
    _get_endpoint().deploy_index(
        index=_get_index(),
        deployed_index_id=settings.vertex_deployed_index_id,
    )
    return True


def undeploy_index() -> bool:
    """
    Undeploy the index from its endpoint if currently deployed. This is
    what STOPS hourly billing on the endpoint. The index resource itself
    (and everything in it) is untouched — only the deployment (the thing
    that costs money by the hour) goes away; redeploying later is fast,
    since the index doesn't need to be rebuilt.
    No-op (returns False) if already undeployed.
    """
    if not is_deployed():
        return False
    _get_endpoint().undeploy_index(deployed_index_id=settings.vertex_deployed_index_id)
    return True
