"""Materialize caller-supplied reference videos (URL or base64) to a tempfile
path for ``ICLoraPipeline``'s ``video_conditioning``. Lighter validation
than ``image_input`` — we don't crack the container; upstream's
``decode_video_by_frame`` surfaces format errors.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import tempfile

import httpx

logger = logging.getLogger(__name__)

MAX_DOWNLOAD_BYTES = 200 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60.0

_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "video/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_DATA_URI_RE = re.compile(r"^data:video/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def _fetch_url(url: str) -> tuple[bytes, str]:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"reference_video_url must be http(s); got scheme of {url!r}")
    try:
        resp = httpx.get(
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=_FETCH_HEADERS,
        )
    except httpx.HTTPError as e:
        raise ValueError(f"failed to fetch reference_video_url: {e}") from e
    if resp.status_code != 200:
        host = httpx.URL(url).host
        raise ValueError(
            f"reference_video_url returned HTTP {resp.status_code} {resp.reason_phrase} "
            f"from {host!r}"
        )
    ctype = resp.headers.get("content-type", "").lower()
    if not (ctype.startswith("video/") or ctype.startswith("application/octet-stream")):
        raise ValueError(
            f"reference_video_url Content-Type must be video/*; got {ctype!r}"
        )
    body = resp.content
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"reference_video_url body {len(body)} bytes exceeds {MAX_DOWNLOAD_BYTES} cap"
        )
    return body, ctype


def _decode_b64(s: str) -> bytes:
    s = "".join(s.split())
    s = _DATA_URI_RE.sub("", s)
    # Cap by character count BEFORE decode so a malicious payload can't force
    # ~225 MB of bytes to materialise and only then be rejected. Base64 is a
    # 4:3 inflation, so N decoded bytes ≤ N*4/3 chars.
    max_b64_chars = (MAX_DOWNLOAD_BYTES * 4 + 2) // 3
    if len(s) > max_b64_chars:
        raise ValueError(
            f"reference_video_b64 has {len(s)} chars; cap is "
            f"{max_b64_chars} chars (~{MAX_DOWNLOAD_BYTES} decoded bytes)"
        )
    try:
        body = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"reference_video_b64 is not valid base64: {e}") from e
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"reference_video_b64 decoded to {len(body)} bytes, exceeds {MAX_DOWNLOAD_BYTES} cap"
        )
    return body


def _suffix_from_ctype(ctype: str | None) -> str:
    if not ctype:
        return ".mp4"
    ctype = ctype.split(";", 1)[0].strip().lower()
    if ctype == "video/webm":
        return ".webm"
    if ctype == "video/quicktime":
        return ".mov"
    return ".mp4"


def materialize_video(
    reference_video_url: str | None,
    reference_video_b64: str | None,
) -> str:
    """Return a tempfile path holding the video bytes. Exactly one of
    ``reference_video_url`` / ``reference_video_b64`` must be set. Caller
    must ``os.unlink`` the path."""
    if reference_video_url is not None and reference_video_b64 is not None:
        raise ValueError(
            "supply at most one of reference_video_url / reference_video_b64, not both"
        )
    if reference_video_url is None and reference_video_b64 is None:
        raise ValueError(
            "either reference_video_url or reference_video_b64 is required for V2V"
        )

    if reference_video_url is not None:
        blob, ctype = _fetch_url(reference_video_url)
    else:
        blob = _decode_b64(reference_video_b64)
        ctype = None
    suffix = _suffix_from_ctype(ctype)

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(blob)
        tmp.flush()
    finally:
        tmp.close()
    logger.info(
        "V2V reference video materialized: ctype=%s, %d bytes -> %s",
        ctype, len(blob), tmp.name,
    )
    return tmp.name
