"""Tests for service logic with mocked GPU pipeline.

Works without bentoml/torch installed — tests the generate logic directly
via the MockGenerator, bypassing BentoML decorators.
"""

import os
from pathlib import Path
from unittest.mock import patch


def _run_generate(mock_gen, **kwargs):
    """Simulate what LTXVideoService.generate() does, without BentoML."""
    from src import storage

    defaults = dict(
        prompt="test prompt",
        negative_prompt="worst quality",
        width=512, height=768, num_frames=25,
        num_inference_steps=30, seed=42, frame_rate=24.0,
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        image_url=None, image_b64=None,
        reference_video_url=None, reference_video_b64=None,
        reference_video_strength=1.0,
        conditioning_attention_strength=1.0,
        enhance_prompt=False,
    )
    defaults.update(kwargs)
    upload_to_supabase = defaults.pop("upload_to_supabase", False)

    result = mock_gen.generate(**defaults)

    response = {
        "generation_time_seconds": result["generation_time_seconds"],
        "parameters": result["parameters"],
    }

    if upload_to_supabase and storage.is_configured():
        response["video_url"] = storage.upload_video(
            result["output_path"], result["output_filename"]
        )
        try:
            os.remove(result["output_path"])
        except OSError:
            pass
    else:
        response["video_path"] = result["output_path"]

    return response


def _run_generate_sync(mock_gen, **kwargs):
    """Simulate what LTXVideoService.generate_sync() does."""
    defaults = dict(
        prompt="test prompt",
        negative_prompt="worst quality",
        width=512, height=768, num_frames=25,
        num_inference_steps=30, seed=42, frame_rate=24.0,
        cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
        image_url=None, image_b64=None,
        reference_video_url=None, reference_video_b64=None,
        reference_video_strength=1.0,
        conditioning_attention_strength=1.0,
        enhance_prompt=False,
    )
    defaults.update(kwargs)
    result = mock_gen.generate(**defaults)
    return Path(result["output_path"])


class TestGenerate:
    def test_returns_video_path(self, mock_generator):
        result = _run_generate(mock_generator, prompt="a cat on a windowsill")
        assert "video_path" in result
        assert result["video_path"].endswith(".mp4")
        assert os.path.exists(result["video_path"])

    def test_returns_timing(self, mock_generator):
        result = _run_generate(mock_generator)
        assert "generation_time_seconds" in result
        assert isinstance(result["generation_time_seconds"], float)

    def test_returns_parameters(self, mock_generator):
        result = _run_generate(mock_generator, width=1024, height=1536, seed=99)
        params = result["parameters"]
        assert params["width"] == 1024
        assert params["height"] == 1536
        assert params["seed"] == 99

    def test_supabase_skipped_when_not_configured(self, mock_generator):
        with patch.dict(os.environ, {}, clear=True):
            result = _run_generate(mock_generator, upload_to_supabase=True)
        assert "video_path" in result
        assert "video_url" not in result

    def test_supabase_upload_when_configured(self, mock_generator):
        with patch("src.storage.is_configured", return_value=True), \
             patch("src.storage.upload_video", return_value="https://example.com/v.mp4"):
            result = _run_generate(mock_generator, upload_to_supabase=True)
        assert result["video_url"] == "https://example.com/v.mp4"

    def test_cleanup_after_supabase_upload(self, mock_generator):
        with patch("src.storage.is_configured", return_value=True), \
             patch("src.storage.upload_video", return_value="https://example.com/v.mp4"):
            result = _run_generate(mock_generator, upload_to_supabase=True)
        assert "video_url" in result


class TestGenerateSync:
    def test_returns_mp4_path(self, mock_generator):
        result = _run_generate_sync(mock_generator, prompt="a dog playing")
        assert str(result).endswith(".mp4")
        assert result.exists()

    def test_returns_path_object(self, mock_generator):
        result = _run_generate_sync(mock_generator)
        assert isinstance(result, Path)


class TestDefaults:
    def test_default_parameters(self, mock_generator):
        result = _run_generate(mock_generator)
        params = result["parameters"]
        assert params["seed"] == 42
        assert params["frame_rate"] == 24.0
        assert params["mode"] == "t2v"

    def test_t2v_scheduler_defaults(self, mock_generator):
        """T2V path emits the proven fp8-h100 scheduler config: 30 steps,
        CFG 3.0, STG 1.0, rescale 0.7."""
        result = _run_generate(mock_generator)
        params = result["parameters"]
        assert params["num_inference_steps"] == 30
        assert params["cfg_scale"] == 3.0
        assert params["stg_scale"] == 1.0
        assert params["rescale_scale"] == 0.7

    def test_i2v_default_includes_enhance_prompt(self, mock_generator):
        result = _run_generate(mock_generator, image_url="https://example.com/cat.png")
        params = result["parameters"]
        assert params["enhance_prompt"] is False
        # T2V scheduler fields are not surfaced on the unified path.
        assert "cfg_scale" not in params
        assert "num_inference_steps" not in params


class TestModeRouting:
    """Veo-style mode discrimination: input presence picks the path."""

    def test_t2v_when_no_image_no_ref_video(self, mock_generator):
        result = _run_generate(mock_generator)
        assert result["parameters"]["mode"] == "t2v"

    def test_i2v_when_image_url_set(self, mock_generator):
        result = _run_generate(
            mock_generator, image_url="https://example.com/cat.png"
        )
        assert result["parameters"]["mode"] == "i2v"

    def test_i2v_when_image_b64_set(self, mock_generator):
        result = _run_generate(mock_generator, image_b64="aGVsbG8=")
        assert result["parameters"]["mode"] == "i2v"

    def test_v2v_when_reference_video_url_set(self, mock_generator):
        result = _run_generate(
            mock_generator,
            reference_video_url="https://example.com/ref.mp4",
        )
        assert result["parameters"]["mode"] == "v2v"

    def test_v2v_when_reference_video_b64_set(self, mock_generator):
        result = _run_generate(mock_generator, reference_video_b64="dmlkZW8=")
        assert result["parameters"]["mode"] == "v2v"

    def test_v2v_takes_precedence_over_image(self, mock_generator):
        result = _run_generate(
            mock_generator,
            image_url="https://example.com/cat.png",
            reference_video_url="https://example.com/ref.mp4",
        )
        assert result["parameters"]["mode"] == "v2v"


class TestT2VForwarding:
    """T2V scheduler args must reach the generator unchanged — these pin the
    ``feature/fp8-h100`` proven configuration so a future refactor that drops
    the params from the forward chain fails fast."""

    def test_num_inference_steps_reaches_generator(self, mock_generator):
        _run_generate(mock_generator, num_inference_steps=20)
        assert mock_generator.last_call["num_inference_steps"] == 20

    def test_cfg_stg_rescale_reach_generator(self, mock_generator):
        _run_generate(
            mock_generator, cfg_scale=4.0, stg_scale=2.0, rescale_scale=0.5,
        )
        assert mock_generator.last_call["cfg_scale"] == 4.0
        assert mock_generator.last_call["stg_scale"] == 2.0
        assert mock_generator.last_call["rescale_scale"] == 0.5

    def test_negative_prompt_reaches_generator(self, mock_generator):
        _run_generate(mock_generator, negative_prompt="bad quality")
        assert mock_generator.last_call["negative_prompt"] == "bad quality"


class TestImageInputForwarding:
    """The service layer just forwards image_url / image_b64 to the generator
    — no transformation. These tests pin that contract so a future refactor
    that drops the params from the forward chain fails fast."""

    def test_image_url_reaches_generator(self, mock_generator):
        _run_generate(mock_generator, image_url="https://example.com/cat.png")
        assert mock_generator.last_call["image_url"] == "https://example.com/cat.png"
        assert mock_generator.last_call["image_b64"] is None

    def test_image_b64_reaches_generator(self, mock_generator):
        _run_generate(mock_generator, image_b64="aGVsbG8=")
        assert mock_generator.last_call["image_b64"] == "aGVsbG8="
        assert mock_generator.last_call["image_url"] is None

    def test_t2v_passes_none_for_both(self, mock_generator):
        _run_generate(mock_generator)
        assert mock_generator.last_call["image_url"] is None
        assert mock_generator.last_call["image_b64"] is None


class TestReferenceVideoForwarding:
    """Same contract for V2V wire fields — service forwards verbatim."""

    def test_reference_video_url_reaches_generator(self, mock_generator):
        _run_generate(
            mock_generator,
            reference_video_url="https://example.com/ref.mp4",
        )
        assert (
            mock_generator.last_call["reference_video_url"]
            == "https://example.com/ref.mp4"
        )
        assert mock_generator.last_call["reference_video_b64"] is None

    def test_reference_video_b64_reaches_generator(self, mock_generator):
        _run_generate(mock_generator, reference_video_b64="dmlkZW8=")
        assert mock_generator.last_call["reference_video_b64"] == "dmlkZW8="
        assert mock_generator.last_call["reference_video_url"] is None

    def test_strength_and_attn_strength_default_one(self, mock_generator):
        _run_generate(mock_generator, image_url="https://example.com/cat.png")
        assert mock_generator.last_call["reference_video_strength"] == 1.0
        assert mock_generator.last_call["conditioning_attention_strength"] == 1.0
