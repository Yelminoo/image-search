import datetime
import uuid
from google.cloud import storage
from app.config import settings

_client = None


def _get_client() -> storage.Client:
    global _client
    if _client is None:
        _client = storage.Client(project=settings.project_id)
    return _client


def upload_image(file_bytes: bytes, filename: str, content_type: str = "image/jpeg") -> str:
    """
    Upload raw image bytes to GCS, return the gs:// URI.
    Filenames are prefixed with a UUID to avoid collisions.
    """
    client = _get_client()
    bucket = client.bucket(settings.bucket)
    blob_name = f"images/{uuid.uuid4()}_{filename}"
    blob = bucket.blob(blob_name)
    blob.upload_from_string(file_bytes, content_type=content_type)
    return f"gs://{settings.bucket}/{blob_name}"


def gcs_uri_to_public_url(gcs_uri: str) -> str:
    """
    Convert gs://bucket/path to a plain public URL. Only works if the
    bucket/object is publicly readable — NOT the case here, since Public
    Access Prevention is enforced on this project and blocks making
    objects public at all. Kept for reference / projects without that
    policy; this app uses generate_signed_url() below instead.
    """
    path = gcs_uri.replace("gs://", "")
    return f"https://storage.googleapis.com/{path}"


def generate_signed_url(gcs_uri: str, expiration_minutes: int = 60) -> str:
    """
    Generate a short-lived signed URL for a private GCS object — works
    regardless of Public Access Prevention, since it's a time-limited
    bearer link rather than a public grant. Requires the app to be
    running with a service account JSON key (GOOGLE_APPLICATION_CREDENTIALS
    pointing at a key file, not ambient/impersonated credentials), since
    signing happens locally with that key's private key.

    Call this at read-time (building an API response), not at ingest
    time — the URL expires, so storing one permanently would go stale.
    """
    path = gcs_uri.replace("gs://", "")
    bucket_name, blob_name = path.split("/", 1)
    blob = _get_client().bucket(bucket_name).blob(blob_name)
    return blob.generate_signed_url(
        version="v4",
        expiration=datetime.timedelta(minutes=expiration_minutes),
        method="GET",
    )
