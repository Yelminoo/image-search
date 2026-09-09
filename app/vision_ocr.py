"""
OCR layer using Google Cloud Vision API.

Uses DOCUMENT_TEXT_DETECTION, which is the variant tuned for dense text
(scans, documents, certificates) rather than TEXT_DETECTION, which is
tuned for sparse text in photos (signs, labels). DOCUMENT_TEXT_DETECTION
generally gives better results for both cases, so we default to it.
"""
from google.cloud import vision
from app.config import settings

_client = None


def _get_client() -> vision.ImageAnnotatorClient:
    global _client
    if _client is None:
        _client = vision.ImageAnnotatorClient()
    return _client


def run_ocr_from_gcs(gcs_uri: str) -> dict:
    """
    Run OCR on an image already sitting in Cloud Storage.
    gcs_uri format: gs://bucket-name/path/to/image.jpg
    """
    client = _get_client()
    image = vision.Image()
    image.source.image_uri = gcs_uri

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    full_text = response.full_text_annotation.text if response.full_text_annotation else ""

    # Pull out per-block confidence + bounding info in case you want it later
    blocks = []
    if response.full_text_annotation:
        for page in response.full_text_annotation.pages:
            for block in page.blocks:
                block_text = ""
                for paragraph in block.paragraphs:
                    for word in paragraph.words:
                        block_text += "".join(s.text for s in word.symbols) + " "
                blocks.append({
                    "text": block_text.strip(),
                    "confidence": block.confidence,
                })

    return {
        "full_text": full_text,
        "blocks": blocks,
    }


def run_ocr_from_bytes(image_bytes: bytes) -> dict:
    """Run OCR directly on raw image bytes (no GCS round-trip needed)."""
    client = _get_client()
    image = vision.Image(content=image_bytes)

    response = client.document_text_detection(image=image)

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    full_text = response.full_text_annotation.text if response.full_text_annotation else ""
    return {"full_text": full_text}
