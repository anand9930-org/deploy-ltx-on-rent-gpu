"""Tests for the triple-stages-ComfyUI pipeline variant — service-level dispatch.

Mirrors test_pipeline_triple_stages.py but exercises the new
``/generate_triple_stages_comfyui_sync`` + ``/generate_triple_stages_comfyui``
endpoints via MockGenerator. The real ``TripleStagesComfyUIMixin`` cannot be
imported here (it pulls in ``src.upstream`` which requires ltx_core /
ltx_pipelines installed); these tests pin the *contract* between service.py
and the generator — ``pipeline_variant``, the absent cfg/stg/rescale + step
kwargs, image fields forwarded — so a future refactor that drifts fails fast.
"""

from pathlib import Path

import pytest


def _run_comfyui_sync(mock_gen, **kwargs):
    """Simulate LTXVideoService.generate_triple_stages_comfyui_sync() body."""
    defaults = dict(
        prompt="test prompt",
        negative_prompt="blurry, low quality",
        width=896, height=1280, num_frames=241,
        seed=42, frame_rate=24.0,
        image_url=None, image_b64=None,
        image_frame_idx=0,
        enhance_prompt=False,
    )
    defaults.update(kwargs)
    result = mock_gen.generate(
        pipeline_variant="triple_stages_comfyui",
        **defaults,
    )
    return Path(result["output_path"])


class TestVariantRouting:
    def test_variant_kwarg_reaches_generator(self, mock_generator):
        _run_comfyui_sync(mock_generator)
        assert mock_generator.last_call["pipeline_variant"] == "triple_stages_comfyui"

    def test_returns_mp4_path(self, mock_generator):
        result = _run_comfyui_sync(mock_generator, prompt="a cat")
        assert isinstance(result, Path)
        assert str(result).endswith(".mp4")
        assert result.exists()


class TestModeDiscrimination:
    def test_t2v_when_no_image(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages_comfyui",
        )
        assert result["parameters"]["mode"] == "triple_comfyui_t2v"

    def test_i2v_when_image_url_set(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages_comfyui",
            image_url="https://example.com/cat.png",
        )
        assert result["parameters"]["mode"] == "triple_comfyui_i2v"

    def test_i2v_when_image_b64_set(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages_comfyui",
            image_b64="aGVsbG8=",
        )
        assert result["parameters"]["mode"] == "triple_comfyui_i2v"


class TestRefVideoRejection:
    """ComfyUI triple-stages doesn't support video conditioning — both
    reference_video fields must raise ValueError when set."""

    def test_reject_reference_video_url(self, mock_generator):
        with pytest.raises(ValueError, match="reference_video"):
            mock_generator.generate(
                prompt="test",
                pipeline_variant="triple_stages_comfyui",
                reference_video_url="https://example.com/ref.mp4",
            )

    def test_reject_reference_video_b64(self, mock_generator):
        with pytest.raises(ValueError, match="reference_video"):
            mock_generator.generate(
                prompt="test",
                pipeline_variant="triple_stages_comfyui",
                reference_video_b64="dmlkZW8=",
            )


class TestParameterForwarding:
    """The endpoint exposes a deliberately narrow set of kwargs. Pin the
    contract."""

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


class TestParametersBlock:
    """The result['parameters'] block must NOT carry cfg/stg/rescale or step
    counts — those are baked into the vendored class. It must surface the
    new mode discriminator and the dim/frame defaults the workflow uses."""

    def test_no_cfg_in_parameters(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages_comfyui",
        )
        assert "cfg_scale" not in result["parameters"]
        assert "stg_scale" not in result["parameters"]
        assert "rescale_scale" not in result["parameters"]

    def test_no_stage_steps_in_parameters(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages_comfyui",
        )
        assert "stage1_steps" not in result["parameters"]
        assert "stage2_steps" not in result["parameters"]

    def test_image_frame_idx_none_for_t2v(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages_comfyui",
        )
        assert result["parameters"]["image_frame_idx"] is None

    def test_image_frame_idx_set_for_i2v(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages_comfyui",
            image_url="https://example.com/cat.png",
            image_frame_idx=4,
        )
        assert result["parameters"]["image_frame_idx"] == 4

    def test_default_dims_match_workflow(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages_comfyui",
        )
        assert result["parameters"]["width"] == 896
        assert result["parameters"]["height"] == 1280
        assert result["parameters"]["num_frames"] == 241
