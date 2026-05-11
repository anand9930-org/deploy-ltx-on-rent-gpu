"""Headless tests for the ComfyUI-graph triple-stages pipeline module.

``src/pipeline/triple_stages_comfyui_graph.py`` runs real ComfyUI nodes at
runtime, so the cascade itself can't be exercised without a GPU + the ComfyUI
checkout (that's the pod step). But the module must be *importable* without
ComfyUI (``service.py`` imports ``DEFAULT_NEGATIVE_PROMPT`` from it at module
level), and its pure bits — the per-stage seed derivation and the
workflow-literal constants — must stay correct. A stray edit to a sigma string
or sampler name would silently change the output, so it's pinned here.
"""

from src.pipeline.triple_stages_comfyui_graph import (
    COMFY_CFG,
    COMFY_DECODE_TILING,
    COMFY_DEFAULT_FRAME_RATE,
    COMFY_DEFAULT_NUM_FRAMES,
    COMFY_DISTILLED_LORA_STRENGTH,
    COMFY_IMG_COND_STRENGTH,
    COMFY_LATENT_DOWNSCALE,
    COMFY_STAGE_1_IMAGE_CRF,
    COMFY_STAGE_1_SAMPLER,
    COMFY_STAGE_1_SIGMAS,
    COMFY_STAGE_23_SAMPLER,
    COMFY_STAGE_23_SIGMAS,
    DEFAULT_NEGATIVE_PROMPT,
    _derive_stage_seeds,
)


class TestDeriveStageSeeds:
    def test_three_distinct_seeds(self):
        s = _derive_stage_seeds(42)
        assert len(s) == 3
        assert len(set(s)) == 3, "per-stage seeds must be distinct"

    def test_stage_one_is_the_request_seed(self):
        # salt[0] == 0, so stage 1 uses the (masked) request seed verbatim —
        # nice for reproducibility (seed=42 → stage1 noise_seed=42).
        assert _derive_stage_seeds(42)[0] == 42

    def test_seeds_fit_a_positive_int64(self):
        for s in _derive_stage_seeds(2**63 - 1):
            assert 0 <= s < 2**63

    def test_deterministic(self):
        assert _derive_stage_seeds(1234) == _derive_stage_seeds(1234)


class TestWorkflowLiterals:
    """These must equal the 3mljpp-api.json node inputs verbatim."""

    def test_sigma_schedules(self):
        assert COMFY_STAGE_1_SIGMAS == "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
        assert COMFY_STAGE_23_SIGMAS == "0.85, 0.7250, 0.4219, 0.0"

    def test_sampler_names(self):
        assert COMFY_STAGE_1_SAMPLER == "euler_ancestral_cfg_pp"
        assert COMFY_STAGE_23_SAMPLER == "euler_cfg_pp"

    def test_scalar_literals(self):
        assert COMFY_STAGE_1_IMAGE_CRF == 18
        assert COMFY_DISTILLED_LORA_STRENGTH == 0.5
        assert COMFY_IMG_COND_STRENGTH == 1.0
        assert COMFY_CFG == 1.0 and isinstance(COMFY_CFG, float)
        assert COMFY_DEFAULT_NUM_FRAMES == 241
        assert COMFY_DEFAULT_FRAME_RATE == 24.0
        assert COMFY_LATENT_DOWNSCALE == 4

    def test_decode_tiling(self):
        assert COMFY_DECODE_TILING == {
            "tile_size": 512, "overlap": 64, "temporal_size": 512, "temporal_overlap": 4,
        }

    def test_negative_prompt_is_the_workflow_literal(self):
        assert DEFAULT_NEGATIVE_PROMPT.startswith("camera zooming out, low resolution, blurry,")
        assert DEFAULT_NEGATIVE_PROMPT.endswith("warping, extra body parts")
        assert "scene cut, scene transition" in DEFAULT_NEGATIVE_PROMPT


def test_module_imports_without_comfyui():
    """The module (and the constant service.py pulls from it) must import on a
    box without ComfyUI installed — i.e. no top-level ``import nodes`` etc.

    The import at the top of this file already proves it; this is just an
    explicit anchor for the invariant.
    """
    import src.pipeline.triple_stages_comfyui_graph as m

    assert hasattr(m, "TripleStagesComfyUIGraphPipeline")
    assert hasattr(m, "DEFAULT_NEGATIVE_PROMPT")
