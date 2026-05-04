"""Supabase Storage integration for video uploads. Client is lazily
initialised so the service can start without Supabase credentials (local dev).
"""

import logging
import os
import time

from src.config import get_settings

logger = logging.getLogger(__name__)

_supabase = None


def is_configured() -> bool:
    """Return True when the required Supabase env vars are present."""
    s = get_settings()
    return bool(s.supabase_url) and bool(s.supabase_service_key)


def _get_client():
    """Return the Supabase client, creating it on first call."""
    global _supabase
    if _supabase is None:
        s = get_settings()
        if not s.supabase_url or not s.supabase_service_key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set"
            )
        from supabase import create_client

        _supabase = create_client(s.supabase_url, s.supabase_service_key)
    return _supabase


def upload_video(file_path: str, object_key: str) -> str:
    """Upload an MP4 file to Supabase Storage and return a signed URL."""
    s = get_settings()
    client = _get_client()
    bucket = s.supabase_bucket

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    logger.info(
        "Uploading %s (%.1f MB) to supabase://%s/%s",
        file_path, size_mb, bucket, object_key,
    )

    upload_start = time.time()
    with open(file_path, "rb") as f:
        client.storage.from_(bucket).upload(
            path=object_key,
            file=f,
            file_options={"content-type": "video/mp4"},
        )
    logger.info(
        "Upload completed in %.1fs (%.1f MB/s)",
        time.time() - upload_start,
        size_mb / max(time.time() - upload_start, 1e-6),
    )

    expiry = s.supabase_url_expiry_seconds
    res = client.storage.from_(bucket).create_signed_url(
        path=object_key,
        expires_in=expiry,
    )

    # Handle different SDK response formats
    if isinstance(res, str):
        url = res
    elif isinstance(res, dict):
        url = res.get("signedURL") or res.get("signedUrl", "")
    else:
        url = str(res)

    logger.info("Generated signed URL (expires in %ds)", expiry)
    return url
