"""Materialize caller-supplied audio (URL or base64) to a tempfile path for
``A2VidPipelineTwoStage``'s ``audio_path``. Lighter validation than image
input — upstream's ``decode_audio_from_file`` surfaces format errors.
Normalises every payload to 2-channel PCM WAV via ffmpeg because LTX-2.3's
audio VAE ``conv_in`` was trained on stereo (weight=[128, 2, 3, 3]) and a
mono input crashes with a channel-mismatch ``RuntimeError`` on the first
denoising step.
"""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
import subprocess
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
    "Accept": "audio/*,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

_DATA_URI_RE = re.compile(r"^data:audio/[a-zA-Z0-9.+-]+;base64,", re.IGNORECASE)


def _fetch_url(url: str) -> tuple[bytes, str]:
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"audio_url must be http(s); got scheme of {url!r}")
    try:
        resp = httpx.get(
            url,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
            follow_redirects=True,
            headers=_FETCH_HEADERS,
        )
    except httpx.HTTPError as e:
        raise ValueError(f"failed to fetch audio_url: {e}") from e
    if resp.status_code != 200:
        host = httpx.URL(url).host
        raise ValueError(
            f"audio_url returned HTTP {resp.status_code} {resp.reason_phrase} "
            f"from {host!r}"
        )
    ctype = resp.headers.get("content-type", "").lower()
    if not (ctype.startswith("audio/") or ctype.startswith("application/octet-stream")):
        raise ValueError(
            f"audio_url Content-Type must be audio/*; got {ctype!r}"
        )
    body = resp.content
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"audio_url body {len(body)} bytes exceeds {MAX_DOWNLOAD_BYTES} cap"
        )
    return body, ctype


def _decode_b64(s: str) -> bytes:
    s = "".join(s.split())
    s = _DATA_URI_RE.sub("", s)
    max_b64_chars = (MAX_DOWNLOAD_BYTES * 4 + 2) // 3
    if len(s) > max_b64_chars:
        raise ValueError(
            f"audio_b64 has {len(s)} chars; cap is "
            f"{max_b64_chars} chars (~{MAX_DOWNLOAD_BYTES} decoded bytes)"
        )
    try:
        body = base64.b64decode(s, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError(f"audio_b64 is not valid base64: {e}") from e
    if len(body) > MAX_DOWNLOAD_BYTES:
        raise ValueError(
            f"audio_b64 decoded to {len(body)} bytes, exceeds {MAX_DOWNLOAD_BYTES} cap"
        )
    return body


def _normalize_to_stereo(input_path: str) -> str:
    """Convert any audio file to 2-channel PCM WAV via ffmpeg. Returns the
    new path and deletes the source. ``-ac 2`` upmixes mono by duplicating
    the single channel and downmixes ≥3-ch sources via ffmpeg's standard
    L/R recipe; for already-stereo input it's a near-noop reencode.
    Required because LTX-2.3's audio VAE ``conv_in`` rejects 1-channel
    input — see the module docstring."""
    out_path = input_path + ".stereo.wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error", "-i", input_path,
                "-ac", "2", "-c:a", "pcm_s16le", out_path,
            ],
            check=True, capture_output=True, timeout=60,
        )
    except subprocess.CalledProcessError as e:
        try:
            os.unlink(input_path)
        except OSError:
            pass
        stderr = e.stderr.decode("utf-8", errors="replace")[:500] if e.stderr else ""
        raise ValueError(
            f"ffmpeg failed to normalise audio to stereo PCM WAV: {stderr}"
        ) from e
    except subprocess.TimeoutExpired as e:
        try:
            os.unlink(input_path)
        except OSError:
            pass
        raise ValueError("ffmpeg stereo-normalisation timed out (>60 s)") from e
    try:
        os.unlink(input_path)
    except OSError:
        pass
    return out_path


def _suffix_from_ctype(ctype: str | None) -> str:
    if not ctype:
        return ".wav"
    ctype = ctype.split(";", 1)[0].strip().lower()
    if ctype == "audio/mpeg":
        return ".mp3"
    if ctype == "audio/ogg":
        return ".ogg"
    if ctype == "audio/flac":
        return ".flac"
    if ctype == "audio/mp4":
        return ".m4a"
    return ".wav"


def materialize_audio(
    audio_url: str | None,
    audio_b64: str | None,
) -> str:
    """Return a tempfile path holding the audio bytes. Exactly one of
    ``audio_url`` / ``audio_b64`` must be set. Caller must ``os.unlink``
    the path."""
    if audio_url is not None and audio_b64 is not None:
        raise ValueError(
            "supply at most one of audio_url / audio_b64, not both"
        )
    if audio_url is None and audio_b64 is None:
        raise ValueError(
            "either audio_url or audio_b64 is required for A2V"
        )

    if audio_url is not None:
        blob, ctype = _fetch_url(audio_url)
    else:
        blob = _decode_b64(audio_b64)
        ctype = None
    suffix = _suffix_from_ctype(ctype)

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(blob)
        tmp.flush()
    finally:
        tmp.close()
    final_path = _normalize_to_stereo(tmp.name)
    logger.info(
        "A2V audio materialized: ctype=%s, %d bytes -> %s (normalised to stereo PCM WAV)",
        ctype, len(blob), final_path,
    )
    return final_path
