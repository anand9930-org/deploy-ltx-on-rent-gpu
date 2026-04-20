"""Supabase Storage integration for video uploads.

The client is lazily initialised on the first call to :func:`upload_video`
so that the rest of the application can start without Supabase credentials
(useful for local development with BentoML).
"""

import logging
import os
import time

logger = logging.getLogger(__name__)

_supabase = None


def is_configured() -> bool:
    """Return True when the required Supabase env vars are present."""
    return bool(os.getenv("SUPABASE_URL")) and bool(os.getenv("SUPABASE_SERVICE_KEY"))


def _get_client():
    """Return the Supabase client, creating it on first call."""
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY environment variables must be set"
            )
        from supabase import create_client

        _supabase = create_client(url, key)
    return _supabase


def upload_video(file_path: str, object_key: str) -> str:
    """Upload an MP4 file to Supabase Storage and return a signed URL."""
    client = _get_client()
    bucket = os.getenv("SUPABASE_BUCKET", "ltx-videos")

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

    expiry = int(os.getenv("SUPABASE_URL_EXPIRY_SECONDS", "604800"))
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
