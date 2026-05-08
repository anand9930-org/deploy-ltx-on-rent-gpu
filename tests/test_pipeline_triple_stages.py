"""Tests for the triple-stages pipeline variant — service-level dispatch.

Mirrors the structure of test_service.py but exercises the new
``/generate_triple_stages_sync`` endpoint behaviour via MockGenerator. The
real ``TripleStagesMixin`` cannot be imported here (it pulls in
``src.upstream`` which requires ltx_core/ltx_pipelines installed); these
tests pin the *contract* between service.py and the generator —
``pipeline_variant``, ``stage1_steps`` / ``stage2_steps``, image fields
forwarded — so a future refactor that drops them fails fast.
"""

from pathlib import Path

import pytest


def _run_triple_stages_sync(mock_gen, **kwargs):
    """Simulate LTXVideoService.generate_triple_stages_sync() body."""
    defaults = dict(
        prompt="test prompt",
        negative_prompt="worst quality",
        width=1536, height=1024, num_frames=121,
        seed=42, frame_rate=24.0,
        cfg_scale=1.0, stg_scale=0.0, rescale_scale=0.0,
        image_url=None, image_b64=None,
        image_strength=1.0, image_frame_idx=0,
        stage1_steps=16, stage2_steps=8,
        enhance_prompt=False,
    )
    defaults.update(kwargs)
    result = mock_gen.generate(
        pipeline_variant="triple_stages",
        **defaults,
    )
    return Path(result["output_path"])


class TestVariantRouting:
    def test_t2v_route_when_no_image(self, mock_generator):
        _run_triple_stages_sync(mock_generator)
        assert mock_generator.last_call["pipeline_variant"] == "triple_stages"

    def test_returns_mp4_path(self, mock_generator):
        result = _run_triple_stages_sync(mock_generator, prompt="a cat")
        assert isinstance(result, Path)
        assert str(result).endswith(".mp4")
        assert result.exists()

    def test_default_variant_unaffected(self, mock_generator):
        """Calling generate() without pipeline_variant must keep existing
        behaviour — t2v mode for prompt-only input."""
        result = mock_generator.generate(
            prompt="default path",
            negative_prompt="bad",
            width=1024, height=1536, num_frames=121,
            num_inference_steps=30, seed=42, frame_rate=24.0,
            cfg_scale=3.0, stg_scale=1.0, rescale_scale=0.7,
            image_url=None, image_b64=None,
            reference_video_url=None, reference_video_b64=None,
            reference_video_strength=1.0,
            conditioning_attention_strength=1.0,
            enhance_prompt=False,
        )
        assert result["parameters"]["mode"] == "t2v"


class TestModeDiscrimination:
    def test_triple_t2v_when_no_image(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages",
            stage1_steps=16, stage2_steps=8,
        )
        assert result["parameters"]["mode"] == "triple_t2v"

    def test_triple_i2v_when_image_url_set(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages",
            image_url="https://example.com/cat.png",
            stage1_steps=16, stage2_steps=8,
        )
        assert result["parameters"]["mode"] == "triple_i2v"

    def test_triple_i2v_when_image_b64_set(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages",
            image_b64="aGVsbG8=",
            stage1_steps=16, stage2_steps=8,
        )
        assert result["parameters"]["mode"] == "triple_i2v"


class TestRefVideoRejection:
    """Triple-stages doesn't support video conditioning — both reference_video
    fields must raise ValueError when set."""

    def test_reject_reference_video_url(self, mock_generator):
        with pytest.raises(ValueError, match="reference_video"):
            mock_generator.generate(
                prompt="test",
                pipeline_variant="triple_stages",
                reference_video_url="https://example.com/ref.mp4",
            )

    def test_reject_reference_video_b64(self, mock_generator):
        with pytest.raises(ValueError, match="reference_video"):
            mock_generator.generate(
                prompt="test",
                pipeline_variant="triple_stages",
                reference_video_b64="dmlkZW8=",
            )


class TestParameterForwarding:
    """The new endpoint adds stage1_steps / stage2_steps / image_strength /
    image_frame_idx to the forward chain. Pin the contract."""

    def test_stage_steps_reach_generator(self, mock_generator):
        _run_triple_stages_sync(mock_generator, stage1_steps=24, stage2_steps=4)
        assert mock_generator.last_call["stage1_steps"] == 24
        assert mock_generator.last_call["stage2_steps"] == 4

    def test_image_strength_reaches_generator(self, mock_generator):
        _run_triple_stages_sync(
            mock_generator,
            image_url="https://example.com/cat.png",
            image_strength=0.7,
        )
        assert mock_generator.last_call["image_strength"] == 0.7

    def test_image_frame_idx_reaches_generator(self, mock_generator):
        _run_triple_stages_sync(
            mock_generator,
            image_url="https://example.com/cat.png",
            image_frame_idx=8,
        )
        assert mock_generator.last_call["image_frame_idx"] == 8

    def test_cfg_stg_rescale_reach_generator(self, mock_generator):
        _run_triple_stages_sync(
            mock_generator, cfg_scale=2.0, stg_scale=0.5, rescale_scale=0.3,
        )
        assert mock_generator.last_call["cfg_scale"] == 2.0
        assert mock_generator.last_call["stg_scale"] == 0.5
        assert mock_generator.last_call["rescale_scale"] == 0.3

    def test_negative_prompt_reaches_generator(self, mock_generator):
        _run_triple_stages_sync(mock_generator, negative_prompt="ugly, blurry")
        assert mock_generator.last_call["negative_prompt"] == "ugly, blurry"

    def test_enhance_prompt_reaches_generator(self, mock_generator):
        _run_triple_stages_sync(mock_generator, enhance_prompt=True)
        assert mock_generator.last_call["enhance_prompt"] is True


class TestParametersBlock:
    """The result['parameters'] block must surface the stage step counts so
    the caller can verify the schedule that ran."""

    def test_stage_steps_in_parameters(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages",
            stage1_steps=20, stage2_steps=6,
        )
        assert result["parameters"]["stage1_steps"] == 20
        assert result["parameters"]["stage2_steps"] == 6

    def test_image_fields_none_for_t2v(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages",
            stage1_steps=16, stage2_steps=8,
        )
        assert result["parameters"]["image_strength"] is None
        assert result["parameters"]["image_frame_idx"] is None

    def test_image_fields_set_for_i2v(self, mock_generator):
        result = mock_generator.generate(
            prompt="test",
            pipeline_variant="triple_stages",
            image_url="https://example.com/cat.png",
            image_strength=0.5, image_frame_idx=4,
            stage1_steps=16, stage2_steps=8,
        )
        assert result["parameters"]["image_strength"] == 0.5
        assert result["parameters"]["image_frame_idx"] == 4
