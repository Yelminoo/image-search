import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    project_id: str = os.getenv("GCP_PROJECT_ID", "")
    location: str = os.getenv("GCP_LOCATION", "us-central1")
    bucket: str = os.getenv("GCS_BUCKET", "")

    vertex_index_id: str = os.getenv("VERTEX_INDEX_ID", "")
    vertex_index_endpoint_id: str = os.getenv("VERTEX_INDEX_ENDPOINT_ID", "")
    vertex_deployed_index_id: str = os.getenv("VERTEX_DEPLOYED_INDEX_ID", "deployed_ocr_index")

    embedding_model: str = os.getenv("VERTEX_TEXT_EMBEDDING_MODEL", "text-embedding-004")
    embedding_dim: int = int(os.getenv("VERTEX_EMBEDDING_DIM", "768"))

    # Image similarity search — separate model/dimension from the text embeddings above.
    # multimodalembedding@001 outputs a different vector space (1408-dim by default),
    # so it can't share an index with the text embeddings.
    multimodal_embedding_model: str = os.getenv("VERTEX_MULTIMODAL_EMBEDDING_MODEL", "multimodalembedding@001")
    image_embedding_dim: int = int(os.getenv("VERTEX_IMAGE_EMBEDDING_DIM", "1408"))

    # Airtable source connector — pulls images + record metadata directly
    # from a base instead of manual upload. See app/airtable_source.py.
    airtable_api_key: str = os.getenv("AIRTABLE_API_KEY", "")
    airtable_base_id: str = os.getenv("AIRTABLE_BASE_ID", "")
    airtable_table_name: str = os.getenv("AIRTABLE_TABLE_NAME", "")

    # Field names to write ingest results back onto the source Airtable
    # record (see main.py's _write_back_to_airtable). These fields must
    # already exist in your base with a compatible type (OCR Text: long
    # text, Indexed: checkbox, Doc ID: single line text) — write-back is
    # best-effort and won't fail the ingest itself if a field is missing.
    airtable_write_ocr_text_field: str = os.getenv("AIRTABLE_WRITE_OCR_TEXT_FIELD", "OCR Text")
    airtable_write_indexed_field: str = os.getenv("AIRTABLE_WRITE_INDEXED_FIELD", "Indexed")
    airtable_write_doc_id_field: str = os.getenv("AIRTABLE_WRITE_DOC_ID_FIELD", "Doc ID")

    # Shared secret for POST /webhooks/airtable. That endpoint is meant to
    # be reachable from the internet (Airtable has to be able to call it),
    # so unlike the rest of this unauthenticated API, it's worth actually
    # locking down once deployed anywhere public. Empty = no check.
    airtable_webhook_secret: str = os.getenv("AIRTABLE_WEBHOOK_SECRET", "")

    # In-process schedule that deploys/undeploys the Vector Search index
    # (e.g. business hours only) so the hourly-billed endpoint isn't
    # running around the clock. Disabled by default — opt in explicitly.
    # See app/scheduler.py for the "only runs while this app is up" caveat.
    vector_search_schedule_enabled: bool = os.getenv("VECTOR_SEARCH_SCHEDULE_ENABLED", "false").lower() == "true"
    vector_search_deploy_cron: str = os.getenv("VECTOR_SEARCH_DEPLOY_CRON", "0 9 * * 1-5")
    vector_search_undeploy_cron: str = os.getenv("VECTOR_SEARCH_UNDEPLOY_CRON", "0 18 * * 1-5")
    vector_search_schedule_timezone: str = os.getenv("VECTOR_SEARCH_SCHEDULE_TIMEZONE", "Asia/Singapore")

    # Estimated $/hour for the deployed Vector Search endpoint, used ONLY by
    # the self-tracked usage meter (app/usage_meter.py) to turn logged
    # deploy/undeploy events into a rough cost estimate. This is a guess,
    # not your real billing rate — override it once you've confirmed the
    # actual rate on the GCP Billing Reports page.
    # Was 0.077 (an unconfirmed third-party guess for e2-standard-2). Updated
    # 2026-09-07 to 0.87, derived from a real observed charge: 200 THB for
    # 7 hours deployed (≈$6.08 at the THB/USD rate that day, ≈$0.87/hr) —
    # actual deployed resources are evidently larger than that guess
    # assumed. Still an estimate, not a live billing lookup — override if
    # your own confirmed rate differs.
    vector_search_hourly_rate_estimate_usd: float = float(os.getenv("VECTOR_SEARCH_HOURLY_RATE_ESTIMATE_USD", "0.87"))

    # Single-user login gate (see app/auth.py) — closes the "no auth on any
    # endpoint" gap flagged since the start of this project. Deliberately
    # simple (one hardcoded account, not a user table) since this app has
    # exactly one operator. Set these in .env, never commit real values —
    # .env.example only ever has placeholders.
    admin_email: str = os.getenv("ADMIN_EMAIL", "")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "")


settings = Settings()
