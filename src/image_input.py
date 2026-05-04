"""Materialize caller-supplied images to a temp-file path for I2V.

The upstream ``ICLoraPipeline`` accepts ``images: list[ImageConditioningInput]``,
where each entry is ``(path, frame_idx, strength, crf)``. We accept the image
either as a public URL or as base64-encoded bytes embedded in the JSON body
and produce a path that upstream's ``decode_image`` can read directly.

Validation is strict: the only way a request reaches the GPU is if the bytes
parse as a real PIL image in a supported format. Caller mistakes (oversize
body, wrong content type, truncated bytes, both inputs set, neither set) raise
``ValueError`` synchronously so the task fails fast at the BentoML layer
rather than burning GPU time and surfacing as an obscure pipeline crash.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
import tempfile
from io import BytesIO

import httpx
from PIL import Image, UnidentifiedImageError

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = ("JPEG", "PNG", "WEBP")
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 30.0

# Many CDNs/WAFs (Wikimedia, signed-URL providers, etc.) 403 on the default
# `python-httpx/<ver>` UA. Send a real-browser-shaped UA so common public
# image hosts don't reject us. Accept-* hint that we want an image so the
# origin can negotiate a sensible representation.
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# data:image/png;base64,iVBORw0... — strip the URI prefix browsers send.
_DATA_URI_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)

_GRID = 64
_MAX_SIDE = 1920
_MIN_SIDE = 256


def _round_to_grid(value: int) -> int:
    return (value // _GRID) * _GRID


def _validate_image_bytes(blob: bytes) -> str:
    """Return the PIL ``.format`` string after a verify+reopen round-trip.

    ``Image.verify()`` consumes the file object and invalidates the Image
    instance, so we open twice: first to verify, second to read ``.format``.
    """
    try:
        with Image.open(BytesIO(blob)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, ValueError) as e:
        raise ValueError(f"image bytes are not a decodable image: {e}") from e

    with Image.open(BytesIO(blob)) as probe:
        fmt = probe.format
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(
            f"unsupported image format {fmt!r}; supported: {SUPPORTED_FORMATS}"
        )
    return fmt


def _fetch_url(url: str) -> bytes:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"image_url must be http(s); got scheme of {url!r}")
    try:
        # `stream=True` would let us abort mid-download on size, but httpx's
        # streaming API doesn't expose Content-Length cleanly across
        # transports. Cheaper to do a normal GET and check len(content) — a
        # 50 MB cap on a single request is fine to materialize in memory.
        resp = httpx.get(
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=_FETCH_HEADERS,
        )
    except httpx.HTTPError as e:
        raise ValueError(f"failed to fetch image_url: {e}") from e
    if resp.status_code != 200:
        # Surface the host so a 403/404 from a specific CDN is debuggable
        # from pod logs without guessing which URL the caller passed.
        host = httpx.URL(url).host
        raise ValueError(
            f"image_url returned HTTP {resp.status_code} {resp.reason_phrase} "
            f"from {host!r}"
        )
    ctype = resp.headers.get("content-type", "").lower()
    if not ctype.startswith("image/"):
        raise ValueError(
            f"image_url Content-Type must be image/*; got {ctype!r}"
        )
    body = resp.content
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"image_url body {len(body)} bytes exceeds {MAX_DOWNLOAD_BYTES} cap"
        )
    return body


def _decode_b64(s: str) -> bytes:
    # Long base64 payloads frequently arrive with embedded whitespace
    # (curl line-wraps, manual copy-paste, MIME-style 76-col chunks).
    # Strip ALL whitespace before validating — keeping the strict
    # `validate=True` for character-set + padding correctness.
    s = "".join(s.split())
    s = _DATA_URI_RE.sub("", s)
    try:
        body = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"image_b64 is not valid base64: {e}") from e
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"image_b64 decoded to {len(body)} bytes, exceeds {MAX_DOWNLOAD_BYTES} cap"
        )
    return body


def materialize_image(
    image_url: str | None,
    image_b64: str | None,
) -> str:
    """Resolve caller input to a tempfile path with verified image bytes.

    Exactly one of ``image_url`` or ``image_b64`` must be set. The returned
    path has a suffix matching the detected format (.jpg/.png/.webp) so
    upstream's ``decode_image`` (which keys off the extension in some paths)
    sees the right thing. The caller is responsible for ``os.unlink``-ing the
    path in a ``finally:`` block once the pipeline has consumed it.
    """
    if image_url is not None and image_b64 is not None:
        raise ValueError("supply at most one of image_url / image_b64, not both")
    if image_url is None and image_b64 is None:
        raise ValueError("either image_url or image_b64 is required for I2V")

    blob = _fetch_url(image_url) if image_url is not None else _decode_b64(image_b64)
    fmt = _validate_image_bytes(blob)
    suffix = ".jpg" if fmt == "JPEG" else f".{fmt.lower()}"

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(blob)
        tmp.flush()
    finally:
        tmp.close()
    logger.info(
        "I2V input materialized: format=%s, %d bytes -> %s",
        fmt, len(blob), tmp.name,
    )
    return tmp.name


def derive_dims_from_image(image_path: str) -> tuple[int, int]:
    """Return (width, height) for the auto-AR case.

    Scales DOWN so the longest input dimension lands at most MAX_SIDE
    (1920); never upscales — upscaling a small input only forces the VAE
    to interpolate the same information into more pixels, producing a
    blurry first frame and wasting VRAM. Both dims are then floor-rounded
    to the 64-grid that the LTX latent stride requires (matching
    ``_round_user_inputs`` so the dim is stable through downstream
    rounding), and the shorter side is clamped to MIN_SIDE (256).
    """
    with Image.open(image_path) as im:
        iw, ih = im.size
    if iw <= 0 or ih <= 0:
        raise ValueError(f"image has invalid dimensions: {iw}x{ih}")

    longest = max(iw, ih)
    scale = min(1.0, _MAX_SIDE / longest)
    w = int(round(iw * scale))
    h = int(round(ih * scale))

    w = max(_MIN_SIDE, _round_to_grid(w))
    h = max(_MIN_SIDE, _round_to_grid(h))
    logger.info(
        "I2V auto-AR: input %dx%d -> output %dx%d (scale=%.3f, /64-grid)",
        iw, ih, w, h, scale,
    )
    return w, h
