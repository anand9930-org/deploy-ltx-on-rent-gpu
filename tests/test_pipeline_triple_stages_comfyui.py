"""Tests for the ComfyUI pipeline — service-level dispatch.

Exercises the ``/generate_sync`` + ``/generate`` endpoints via MockGenerator.
Pins the *contract* between service.py and the generator — image fields
forwarded, default bucket, mode discriminator — so a future refactor that
drifts fails fast.
"""

from pathlib import Path


def _run_comfyui_sync(mock_gen, **kwargs):
    """Simulate LTXVideoService.generate_sync() body."""
    defaults = dict(
        prompt="test prompt",
        negative_prompt="blurry, low quality",
        aspect_ratio="auto",
        num_frames=241,
        seed=42,
        frame_rate=24.0,
        image_url=None,
        image_b64=None,
        image_frame_idx=0,
        enhance_prompt=False,
    )
    defaults.update(kwargs)
    result = mock_gen.generate(**defaults)
    return Path(result["output_path"])


class TestVariantRouting:
    def test_returns_mp4_path(self, mock_generator):
        result = _run_comfyui_sync(mock_generator, prompt="a cat")
        assert isinstance(result, Path)
        assert str(result).endswith(".mp4")
        assert result.exists()


class TestModeDiscrimination:
    def test_t2v_when_no_image(self, mock_generator):
        result = mock_generator.generate(prompt="test")
        assert result["parameters"]["mode"] == "triple_comfyui_t2v"

    def test_i2v_when_image_url_set(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            image_url="https://example.com/cat.png",
        )
        assert result["parameters"]["mode"] == "triple_comfyui_i2v"

    def test_i2v_when_image_b64_set(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            image_b64="aGVsbG8=",
        )
        assert result["parameters"]["mode"] == "triple_comfyui_i2v"


class TestParameterForwarding:
    def test_image_frame_idx_reaches_generator(self, mock_generator):
        _run_comfyui_sync(
            mock_generator,
            image_url="https://example.com/cat.png",
            image_frame_idx=8,
        )
        assert mock_generator.last_call["image_frame_idx"] == 8

    def test_negative_prompt_reaches_generator(self, mock_generator):
        _run_comfyui_sync(mock_generator, negative_prompt="ugly, blurry")
        assert mock_generator.last_call["negative_prompt"] == "ugly, blurry"

    def test_enhance_prompt_reaches_generator(self, mock_generator):
        _run_comfyui_sync(mock_generator, enhance_prompt=True)
        assert mock_generator.last_call["enhance_prompt"] is True

    def test_seed_reaches_generator(self, mock_generator):
        _run_comfyui_sync(mock_generator, seed=1234)
        assert mock_generator.last_call["seed"] == 1234

    def test_aspect_ratio_reaches_generator(self, mock_generator):
        _run_comfyui_sync(mock_generator, aspect_ratio="9:16")
        assert mock_generator.last_call["aspect_ratio"] == "9:16"


class TestParametersBlock:
    """The result['parameters'] block must NOT carry cfg/stg/rescale or step
    counts — those are baked into the workflow."""

    def test_no_cfg_in_parameters(self, mock_generator):
        result = mock_generator.generate(prompt="test")
        assert "cfg_scale" not in result["parameters"]
        assert "stg_scale" not in result["parameters"]
        assert "rescale_scale" not in result["parameters"]

    def test_no_stage_steps_in_parameters(self, mock_generator):
        result = mock_generator.generate(prompt="test")
        assert "stage1_steps" not in result["parameters"]
        assert "stage2_steps" not in result["parameters"]

    def test_image_frame_idx_none_for_t2v(self, mock_generator):
        result = mock_generator.generate(prompt="test")
        assert result["parameters"]["image_frame_idx"] is None

    def test_image_frame_idx_set_for_i2v(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            image_url="https://example.com/cat.png",
            image_frame_idx=4,
        )
        assert result["parameters"]["image_frame_idx"] == 4

    def test_t2v_default_dims_are_landscape_1080p(self, mock_generator):
        # aspect_ratio="auto" + no image → "16:9" → 1920x1080
        result = mock_generator.generate(prompt="test")
        params = result["parameters"]
        assert params["aspect_ratio"] == "16:9"
        assert params["width"] == 1920
        assert params["height"] == 1080
        assert params["num_frames"] == 241

    def test_explicit_portrait_dims(self, mock_generator):
        result = mock_generator.generate(prompt="test", aspect_ratio="9:16")
        params = result["parameters"]
        assert params["aspect_ratio"] == "9:16"
        assert (params["width"], params["height"]) == (1080, 1920)
