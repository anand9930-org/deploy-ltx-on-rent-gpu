"""LTX-2.3 ComfyUI-only pipeline wrapper.

Holds ``LTXVideoGenerator`` (init, ``_ensure_mode``, ``generate``). The class
delegates all build + generate work to ``TripleStagesComfyUIMixin``.
"""

import gc
import logging
import os
import time
from typing import Literal

import torch

from src.pipeline.triple_stages_comfyui import TripleStagesComfyUIMixin
from src.pipeline.triple_stages_comfyui_graph import DEFAULT_NEGATIVE_PROMPT

logger = logging.getLogger(__name__)

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


_MODE_TRIPLE_STAGES_COMFYUI = "triple_stages_comfyui"


class LTXVideoGenerator(TripleStagesComfyUIMixin):
    """ComfyUI-graph pipeline wrapper. Runs the 3mljpp 3-stage AV workflow
    via real ComfyUI core nodes. Single resident pipeline."""

    def __init__(self, model_dir: str = "/models") -> None:
        self._model_dir = model_dir
        self._pipeline = None
        self._active_mode: str | None = None

        logger.info("Preloading pipeline: %s", _MODE_TRIPLE_STAGES_COMFYUI)
        self._ensure_mode(_MODE_TRIPLE_STAGES_COMFYUI)
        logger.info("Pipeline ready.")

    def _ensure_mode(self, mode: str) -> None:
        if self._active_mode == mode and self._pipeline is not None:
            return
        if self._pipeline is not None:
            logger.info("Tearing down pipeline for rebuild")
            t_teardown = time.perf_counter()
            self._pipeline = None
            from src import comfyui_runtime

            comfyui_runtime.unload_models()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Teardown took %.2fs", time.perf_counter() - t_teardown)
        if mode != _MODE_TRIPLE_STAGES_COMFYUI:
            raise ValueError(f"Unknown pipeline mode: {mode!r}")
        self._build_triple_stages_comfyui()
        self._active_mode = mode

    # @torch.no_grad() (not @torch.inference_mode()) — torch 2.11+ rejects
    # wrapping inference tensors in nn.Parameter, and ComfyUI's lazy
    # first-request GPU model upload (model_management.partially_load -> .to())
    # goes through `torch.nn.Parameter(p, requires_grad=False)` on every moved
    # FP8 weight. inference_mode would mark those tensors as inference tensors,
    # then Parameter.__new__ would fail with "Cannot set version_counter for
    # inference tensor". no_grad has the same gradient-tracking-off semantics
    # without the inference-tensor marking.
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        aspect_ratio: Literal["16:9", "9:16", "auto"] = "auto",
        num_frames: int = 241,
        seed: int = 42,
        frame_rate: float = 24.0,
        image_url: str | None = None,
        image_b64: str | None = None,
        image_frame_idx: int = 0,
        enhance_prompt: bool = False,
    ) -> dict:
        self._ensure_mode(_MODE_TRIPLE_STAGES_COMFYUI)
        return self._triple_stages_comfyui_generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=aspect_ratio,
            num_frames=num_frames,
            seed=seed,
            frame_rate=frame_rate,
            image_url=image_url,
            image_b64=image_b64,
            image_frame_idx=image_frame_idx,
            enhance_prompt=enhance_prompt,
        )
