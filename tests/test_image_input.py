"""Tests for src/image_input.py — URL fetch, base64 decode, validation, auto-AR."""

import base64
import os
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.image_input import (
    MAX_DOWNLOAD_BYTES,
    derive_dims_from_image,
    materialize_image,
)
from tests.conftest import minimal_png_bytes


def _mock_response(content: bytes, status_code: int = 200, content_type: str = "image/png"):
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.reason_phrase = "OK" if status_code == 200 else "ERR"
    resp.headers = {"content-type": content_type}
    resp.content = content
    return resp


class TestMaterializeImage:
    def test_url_fetch_writes_tempfile(self):
        png = minimal_png_bytes(64, 32)
        with patch("src.image_input.httpx.get", return_value=_mock_response(png)):
            path = materialize_image(image_url="https://example.com/x.png", image_b64=None)
        try:
            assert os.path.exists(path)
            assert path.endswith(".png")
            with open(path, "rb") as f:
                assert f.read() == png
        finally:
            os.unlink(path)

    def test_url_fetch_sends_browser_user_agent(self):
        # Wikimedia and several other CDN/WAFs 403 the default `python-httpx`
        # UA. We send a real-browser-shaped UA on the fetch — pin that here
        # so a future refactor that drops the headers re-breaks loudly in
        # tests instead of silently in production.
        png = minimal_png_bytes(8, 8)
        with patch("src.image_input.httpx.get", return_value=_mock_response(png)) as mock_get:
            path = materialize_image(image_url="https://example.com/x.png", image_b64=None)
        try:
            headers = mock_get.call_args.kwargs["headers"]
            ua = headers["User-Agent"]
            assert "Mozilla/" in ua and "python-httpx" not in ua.lower()
        finally:
            os.unlink(path)

    def test_url_403_message_includes_host(self):
        # Surface the host in the error so a CDN block is debuggable from
        # pod logs without guessing which URL the caller passed.
        with patch(
            "src.image_input.httpx.get",
            return_value=_mock_response(b"", status_code=403),
        ):
            with pytest.raises(ValueError, match="403.*example.com"):
                materialize_image(image_url="https://example.com/x.png", image_b64=None)

    def test_b64_decode_writes_tempfile(self):
        png = minimal_png_bytes(32, 32)
        b64 = base64.b64encode(png).decode("ascii")
        path = materialize_image(image_url=None, image_b64=b64)
        try:
            assert os.path.exists(path)
            with open(path, "rb") as f:
                assert f.read() == png
        finally:
            os.unlink(path)

    def test_b64_strips_data_uri_prefix(self):
        png = minimal_png_bytes(8, 8)
        data_uri = "data:image/png;base64," + base64.b64encode(png).decode("ascii")
        path = materialize_image(image_url=None, image_b64=data_uri)
        try:
            with open(path, "rb") as f:
                assert f.read() == png
        finally:
            os.unlink(path)

    def test_b64_tolerates_embedded_whitespace(self):
        # curl, copy-paste, and MIME-style 76-col wrapping all emit base64
        # with embedded \n. We strip whitespace before decode so these
        # work without the caller having to pre-process.
        png = minimal_png_bytes(16, 16)
        b64 = base64.b64encode(png).decode("ascii")
        wrapped = "\n".join(b64[i : i + 76] for i in range(0, len(b64), 76)) + "\n"
        path = materialize_image(image_url=None, image_b64=wrapped)
        try:
            with open(path, "rb") as f:
                assert f.read() == png
        finally:
            os.unlink(path)

    def test_rejects_both_inputs(self):
        with pytest.raises(ValueError, match="at most one"):
            materialize_image(image_url="https://x", image_b64="abc")

    def test_rejects_neither_input(self):
        with pytest.raises(ValueError, match="required"):
            materialize_image(image_url=None, image_b64=None)

    def test_rejects_non_http_scheme(self):
        with pytest.raises(ValueError, match="http"):
            materialize_image(image_url="ftp://x/y.png", image_b64=None)

    def test_rejects_non_image_content_type(self):
        with patch(
            "src.image_input.httpx.get",
            return_value=_mock_response(b"<html></html>", content_type="text/html"),
        ):
            with pytest.raises(ValueError, match="Content-Type"):
                materialize_image(image_url="https://x/y", image_b64=None)

    def test_rejects_oversize_url(self):
        big = b"x" * (MAX_DOWNLOAD_BYTES + 1)
        with patch("src.image_input.httpx.get", return_value=_mock_response(big)):
            with pytest.raises(ValueError, match="exceeds"):
                materialize_image(image_url="https://x/y.png", image_b64=None)

    def test_rejects_http_error(self):
        with patch(
            "src.image_input.httpx.get",
            return_value=_mock_response(b"", status_code=404),
        ):
            with pytest.raises(ValueError, match="404"):
                materialize_image(image_url="https://x/missing", image_b64=None)

    def test_rejects_invalid_base64(self):
        with pytest.raises(ValueError, match="base64"):
            materialize_image(image_url=None, image_b64="this is not !!base64!!")

    def test_rejects_undecodable_image(self):
        bogus = base64.b64encode(b"not actually an image").decode("ascii")
        with pytest.raises(ValueError, match="not a decodable image"):
            materialize_image(image_url=None, image_b64=bogus)


class TestDeriveDimsFromImage:
    def _write(self, tmp_path, w: int, h: int) -> str:
        path = str(tmp_path / "i.png")
        with open(path, "wb") as f:
            f.write(minimal_png_bytes(w, h))
        return path

    def test_landscape_1920x1080(self, tmp_path):
        # longest = MAX_SIDE; no scale; 1080 floored to 64-grid -> 1024
        # (matches the documented LTX-2.3 landscape behavior — see memory)
        w, h = derive_dims_from_image(self._write(tmp_path, 1920, 1080))
        assert (w, h) == (1920, 1024)

    def test_portrait_1080x1920(self, tmp_path):
        w, h = derive_dims_from_image(self._write(tmp_path, 1080, 1920))
        assert (w, h) == (1024, 1920)

    def test_small_input_not_upscaled(self, tmp_path):
        # No upscaling: 1024x1024 -> 1024x1024 (already on grid).
        # Without this guard a small image would be ballooned to 1920x1920,
        # forcing the VAE to interpolate and the diffusion to render extra
        # pixels of nothing-new — blurry first frame, wasted VRAM.
        w, h = derive_dims_from_image(self._write(tmp_path, 1024, 1024))
        assert (w, h) == (1024, 1024)

    def test_tiny_input_clamped_to_min(self, tmp_path):
        # 100x80 -> below MIN_SIDE on both axes; floor to 64 = (64, 64),
        # clamp to MIN_SIDE = (256, 256). Aspect-ratio is sacrificed only
        # in the extreme-tiny case; documented behavior.
        w, h = derive_dims_from_image(self._write(tmp_path, 100, 80))
        assert w == 256 and h == 256

    def test_oversized_input_capped_to_1920(self, tmp_path):
        # 4000x3000 -> scale 0.48 -> 1920x1440 -> floor /64 -> 1920x1408
        w, h = derive_dims_from_image(self._write(tmp_path, 4000, 3000))
        assert (w, h) == (1920, 1408)

    def test_extreme_aspect_clamps_to_min(self, tmp_path):
        # 4000x100 -> short side after scale = 48 -> floored to 0 -> clamped to 256
        w, h = derive_dims_from_image(self._write(tmp_path, 4000, 100))
        assert w == 1920
        assert h == 256

    def test_dims_are_64_divisible(self, tmp_path):
        for size in [(1500, 750), (777, 1234), (1920, 800), (300, 400)]:
            w, h = derive_dims_from_image(self._write(tmp_path, *size))
            assert w % 64 == 0
            assert h % 64 == 0
            assert w >= 256 and h >= 256
            assert w <= 1920 and h <= 1920
