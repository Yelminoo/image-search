"""
One-time setup: create a Vertex AI Vector Search index, an index endpoint,
and deploy the index to the endpoint.

Run this ONCE before starting the FastAPI app. It takes 20-60 minutes for
the index to build and deploy — this is normal for Vertex AI Vector Search,
not a hang. Print the resulting IDs into your .env file when it finishes.

Usage:
    python setup_vector_index.py
"""
from google.cloud import aiplatform
from app.config import settings

aiplatform.init(project=settings.project_id, location=settings.location)

print("Creating index (this can take a while)...")
index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="ocr-text-index",
    dimensions=settings.embedding_dim,
    approximate_neighbors_count=150,
    distance_measure_type="COSINE_DISTANCE",
    index_update_method="STREAM_UPDATE",  # required for upsert_datapoints() at runtime
)
print(f"Index created: {index.resource_name}")

print("Creating index endpoint...")
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="ocr-text-index-endpoint",
    public_endpoint_enabled=True,
)
print(f"Endpoint created: {endpoint.resource_name}")

print("Deploying index to endpoint (this is the slow part)...")
endpoint.deploy_index(
    index=index,
    deployed_index_id=settings.vertex_deployed_index_id,
)
print("Deployed.")

print("\n--- Add these to your .env file ---")
print(f"VERTEX_INDEX_ID={index.resource_name}")
print(f"VERTEX_INDEX_ENDPOINT_ID={endpoint.resource_name}")
