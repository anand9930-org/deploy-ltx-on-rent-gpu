"""Shared test fixtures — mock GPU pipeline for local testing."""

import os
import tempfile
import uuid
from io import BytesIO

import pytest


class MockGenerator:
    """Drop-in replacement for LTXVideoGenerator that runs without GPU.

    Returns a tiny valid MP4 file so the full serving flow
    (BentoML endpoint → generate → encode → response) can be tested locally.
    Records the last call's kwargs on ``last_call`` so tests can assert that
    ``image_url`` / ``image_b64`` / dims were passed through correctly.
    """

    def __init__(self) -> None:
        self.last_call: dict | None = None

    def generate(self, **kwargs):
        self.last_call = dict(kwargs)
        job_id = uuid.uuid4().hex[:12]
        output_filename = f"ltx_{job_id}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        _write_dummy_mp4(output_path)

        has_image = (
            kwargs.get("image_url") is not None
            or kwargs.get("image_b64") is not None
        )
        mode = "triple_comfyui_i2v" if has_image else "triple_comfyui_t2v"
        parameters = {
            "mode": mode,
            "width": kwargs.get("width", 896),
            "height": kwargs.get("height", 1280),
            "num_frames": kwargs.get("num_frames", 241),
            "seed": kwargs.get("seed", 42),
            "frame_rate": kwargs.get("frame_rate", 24.0),
            "image_frame_idx": (
                kwargs.get("image_frame_idx", 0) if has_image else None
            ),
            "enhance_prompt": kwargs.get("enhance_prompt", False),
        }
        return {
            "output_path": output_path,
            "output_filename": output_filename,
            "generation_time_seconds": 0.01,
            "parameters": parameters,
        }


def minimal_png_bytes(width: int = 16, height: int = 16, color=(255, 0, 0)) -> bytes:
    """Return a real PNG byte blob — used by I2V tests so we never depend on
    a fixture file. PIL is a hard dep of this repo (see pyproject.toml)."""
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _write_dummy_mp4(path: str) -> None:
    """Write a minimal valid MP4 file (~24 bytes)."""
    ftyp = b"\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42"
    with open(path, "wb") as f:
        f.write(ftyp)


@pytest.fixture
def mock_generator():
    return MockGenerator()
