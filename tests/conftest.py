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
    ``image_url`` / ``image_b64`` / ``reference_video_url`` / dims were
    passed through correctly. Echoes the same ``parameters`` block shape the
    real generator emits, including the ``mode`` discriminator. IC-LoRA
    fields (``reference_downscale_factor``, ``reference_video_strength``,
    ``conditioning_attention_strength``) appear only on the I2V/V2V branch
    — A2V is vanilla and does not emit them.
    """

    def __init__(self) -> None:
        self.last_call: dict | None = None

    def generate(self, **kwargs):
        self.last_call = dict(kwargs)
        job_id = uuid.uuid4().hex[:12]
        output_filename = f"ltx_{job_id}.mp4"
        output_path = os.path.join(tempfile.gettempdir(), output_filename)

        _write_dummy_mp4(output_path)

        has_audio = (
            kwargs.get("audio_url") is not None
            or kwargs.get("audio_b64") is not None
        )
        has_image = (
            kwargs.get("image_url") is not None
            or kwargs.get("image_b64") is not None
        )
        has_ref_video = (
            kwargs.get("reference_video_url") is not None
            or kwargs.get("reference_video_b64") is not None
        )
        if has_audio:
            mode = "a2v"
        elif has_ref_video:
            mode = "v2v"
        elif has_image:
            mode = "i2v"
        else:
            mode = "t2v"

        if mode == "a2v":
            parameters = {
                "mode": mode,
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1536),
                "num_frames": kwargs.get("num_frames", 121),
                "num_inference_steps": kwargs.get("num_inference_steps", 30),
                "seed": kwargs.get("seed", 42),
                "frame_rate": kwargs.get("frame_rate", 24.0),
                "cfg_scale": kwargs.get("cfg_scale", 3.0),
                "stg_scale": kwargs.get("stg_scale", 1.0),
                "rescale_scale": kwargs.get("rescale_scale", 0.7),
                "enhance_prompt": kwargs.get("enhance_prompt", False),
            }
        elif mode == "t2v":
            parameters = {
                "mode": mode,
                "width": kwargs.get("width", 1024),
                "height": kwargs.get("height", 1536),
                "num_frames": kwargs.get("num_frames", 121),
                "num_inference_steps": kwargs.get("num_inference_steps", 30),
                "seed": kwargs.get("seed", 42),
                "frame_rate": kwargs.get("frame_rate", 24.0),
                "cfg_scale": kwargs.get("cfg_scale", 3.0),
                "stg_scale": kwargs.get("stg_scale", 1.0),
                "rescale_scale": kwargs.get("rescale_scale", 0.7),
            }
        else:
            parameters = {
                "mode": mode,
                "width": kwargs.get("width", 512),
                "height": kwargs.get("height", 768),
                "num_frames": kwargs.get("num_frames", 25),
                "seed": kwargs.get("seed", 42),
                "frame_rate": kwargs.get("frame_rate", 24.0),
                "reference_downscale_factor": 2,
                "reference_video_strength": (
                    kwargs.get("reference_video_strength", 1.0)
                    if has_ref_video else None
                ),
                "conditioning_attention_strength": (
                    kwargs.get("conditioning_attention_strength", 1.0)
                    if (has_image or has_ref_video) else None
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
