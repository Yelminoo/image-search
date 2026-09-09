"""
Embedding layer using Vertex AI's text embedding model.

text-embedding-004 outputs 768-dim vectors by default and is tuned for
retrieval tasks (it accepts a `task_type` hint — RETRIEVAL_DOCUMENT when
storing, RETRIEVAL_QUERY when searching — which measurably improves
recall over using the same task type for both).
"""
from typing import List
import vertexai
from vertexai.language_models import TextEmbeddingModel, TextEmbeddingInput
from vertexai.vision_models import Image as VertexImage, MultiModalEmbeddingModel
from app.config import settings

_model = None
_multimodal_model = None


def _get_model() -> TextEmbeddingModel:
    global _model
    if _model is None:
        vertexai.init(project=settings.project_id, location=settings.location)
        _model = TextEmbeddingModel.from_pretrained(settings.embedding_model)
    return _model


def _get_multimodal_model() -> MultiModalEmbeddingModel:
    global _multimodal_model
    if _multimodal_model is None:
        vertexai.init(project=settings.project_id, location=settings.location)
        _multimodal_model = MultiModalEmbeddingModel.from_pretrained(settings.multimodal_embedding_model)
    return _multimodal_model


def embed_document(text: str) -> List[float]:
    """Embed text that will be stored/indexed."""
    model = _get_model()
    inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_DOCUMENT")]
    result = model.get_embeddings(inputs)
    return result[0].values


def embed_query(text: str) -> List[float]:
    """Embed a search query — uses a different task_type hint than documents."""
    model = _get_model()
    inputs = [TextEmbeddingInput(text=text, task_type="RETRIEVAL_QUERY")]
    result = model.get_embeddings(inputs)
    return result[0].values


def embed_documents_batch(texts: List[str]) -> List[List[float]]:
    """Batch embed multiple documents in one call (more efficient for bulk ingest)."""
    model = _get_model()
    inputs = [TextEmbeddingInput(text=t, task_type="RETRIEVAL_DOCUMENT") for t in texts]
    results = model.get_embeddings(inputs)
    return [r.values for r in results]


def embed_image(image_bytes: bytes) -> List[float]:
    """
    Embed raw image bytes for visual similarity search, using
    multimodalembedding@001. This is a *different* vector space than
    embed_document/embed_query above — never compare an image embedding
    against a text embedding, only against other image embeddings.
    """
    model = _get_multimodal_model()
    image = VertexImage(image_bytes=image_bytes)
    result = model.get_embeddings(image=image, dimension=settings.image_embedding_dim)
    return result.image_embedding
