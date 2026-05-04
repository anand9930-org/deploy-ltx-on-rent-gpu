"""TeaCache (training-free diffusion-step caching) for LTX-2.3.

At each step, measures the rescaled relative-L1 distance between the
current transformer input and the previous one; if below threshold,
skips the forward and reuses the previous output. 1.6–2.1× lossless
end-to-end speedup on LTX-Video (ali-vilab/TeaCache).

Patched at ``DiffusionStage._transformer_ctx`` so streaming / batch-split
downstream is untouched and the same patch works for guided and simple
denoisers. Configured via :class:`src.config.Settings`
(``ENABLE_TEACACHE`` / ``TEACACHE_THRESHOLD`` / ``TEACACHE_STAGES``);
default caches only ``stage_1`` — ``stage_2``'s 3-step distilled
schedule is too short to amortise.
"""

from __future__ import annotations

import functools
import logging
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable

import torch

from src.config import get_settings

logger = logging.getLogger(__name__)


# Polynomial coefficients fitted on LTX-Video (ali-vilab/TeaCache),
# numpy.polyval ordering (highest-degree first). May need re-fitting
# for LTX-2.3's longer seq_len + two-stage schedule.
_POLY_COEFFS: tuple[float, ...] = (
    2.14700694e1,
    -1.28016453e1,
    2.31279151e0,
    7.92487521e-1,
    9.69274326e-3,
)


def _polyval(coeffs: tuple[float, ...], x: float) -> float:
    """Horner polynomial eval; numpy-free to keep numpy off the hot path."""
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


def _rescale(rel_l1: float) -> float:
    """TeaCache's learned rescaling on a raw relative-L1 distance.
    Input clamped to [0, 1] (the fit's validity region)."""
    x = max(0.0, min(1.0, rel_l1))
    return _polyval(_POLY_COEFFS, x)


@dataclass
class _State:
    """Per-stage cache state, fresh on each ``_transformer_ctx`` entry."""

    rel_l1_thresh: float
    prev_input: torch.Tensor | None = None
    prev_vx: torch.Tensor | None = None
    prev_ax: torch.Tensor | None = None
    accumulated_dist: float = 0.0
    step: int = 0
    computes: int = 0
    skips: int = 0

    def update_from_miss(
        self,
        ref_input: torch.Tensor | None,
        vx: torch.Tensor | None,
        ax: torch.Tensor | None,
    ) -> None:
        self.prev_input = ref_input.detach() if ref_input is not None else None
        self.prev_vx = vx.detach() if vx is not None else None
        self.prev_ax = ax.detach() if ax is not None else None
        self.accumulated_dist = 0.0
        self.computes += 1

    @property
    def skip_rate(self) -> float:
        total = self.computes + self.skips
        return 0.0 if total == 0 else self.skips / total


def _extract_reference_input(video: Any, audio: Any) -> torch.Tensor | None:
    """Similarity-key tensor for this call. Video preferred (bigger,
    dominates compute), audio fallback; None only if both are absent."""
    if video is not None and getattr(video, "latent", None) is not None:
        return video.latent
    if audio is not None and getattr(audio, "latent", None) is not None:
        return audio.latent
    return None


def _make_wrapped_forward(
    original_forward: Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]],
    state: _State,
) -> Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Build a ``X0Model.forward`` replacement that short-circuits on
    cache hits. Captures the bound ``original_forward`` + ``state`` in
    closure; returned callable is assigned onto the transformer."""

    def forward(
        video: Any = None,
        audio: Any = None,
        perturbations: Any = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        ref_input = _extract_reference_input(video, audio)
        state.step += 1

        # First step of this stage: nothing cached, always compute.
        if state.prev_input is None or ref_input is None:
            if state.step == 1 and ref_input is not None:
                logger.info(
                    "TeaCache: first step — reference latent shape=%s dtype=%s",
                    tuple(ref_input.shape), ref_input.dtype,
                )
            vx, ax = original_forward(video, audio, perturbations)
            state.update_from_miss(ref_input, vx, ax)
            return vx, ax

        # Shape or dtype mismatch across consecutive steps means the
        # cache isn't meaningful; recompute and reset.
        if (
            ref_input.shape != state.prev_input.shape
            or ref_input.dtype != state.prev_input.dtype
        ):
            vx, ax = original_forward(video, audio, perturbations)
            state.update_from_miss(ref_input, vx, ax)
            return vx, ax

        # Relative L1 distance on the batched latent. This handles
        # CFG batching implicitly: both steps carry the same
        # (uncond, cond[, stg, modality]) stack so the comparison is
        # across-step rather than across-pass.
        denom = state.prev_input.abs().mean().clamp_min(1e-8)
        rel_l1 = ((ref_input - state.prev_input).abs().mean() / denom).item()
        state.accumulated_dist += _rescale(rel_l1)

        if state.accumulated_dist < state.rel_l1_thresh:
            # Cache hit — reuse last computed outputs.
            state.skips += 1
            return state.prev_vx, state.prev_ax

        # Cache miss — compute and refresh.
        vx, ax = original_forward(video, audio, perturbations)
        state.update_from_miss(ref_input, vx, ax)
        return vx, ax

    return forward


class _PatchedContextManager:
    """Patches ``X0Model.forward`` with the TeaCache wrapper on entry;
    restores the original and logs per-stage stats on exit."""

    def __init__(
        self,
        inner: AbstractContextManager,
        threshold: float,
        stage_name: str,
    ) -> None:
        self._inner = inner
        self._threshold = threshold
        self._stage_name = stage_name
        self._transformer: Any = None
        self._original_forward: Callable | None = None
        self._state: _State | None = None

    def __enter__(self) -> Any:
        self._transformer = self._inner.__enter__()
        self._state = _State(rel_l1_thresh=self._threshold)
        self._original_forward = self._transformer.forward
        self._transformer.forward = _make_wrapped_forward(
            self._original_forward, self._state
        )
        logger.info(
            "TeaCache [%s]: patched X0Model.forward (threshold=%.3f)",
            self._stage_name,
            self._threshold,
        )
        return self._transformer

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore before unwinding the inner context, so the transformer
        # goes into teardown in its original shape.
        if self._transformer is not None and self._original_forward is not None:
            try:
                self._transformer.forward = self._original_forward
            except Exception:
                logger.debug(
                    "TeaCache [%s]: forward restoration failed",
                    self._stage_name,
                    exc_info=True,
                )
        if self._state is not None:
            logger.info(
                "TeaCache [%s]: computes=%d skips=%d skip_rate=%.1f%%",
                self._stage_name,
                self._state.computes,
                self._state.skips,
                100.0 * self._state.skip_rate,
            )
            # Explicitly drop tensor refs — without this, our cached
            # (prev_vx, prev_ax) hold GPU memory across the stage
            # teardown, contributing to the Stage 1 → Stage 2 pinned-
            # memory pressure that crashes pin_memory() on 24 GB cards.
            self._state.prev_input = None
            self._state.prev_vx = None
            self._state.prev_ax = None
        return self._inner.__exit__(exc_type, exc_val, exc_tb)


def _patch_stage(stage: Any, threshold: float, stage_name: str) -> None:
    """Wrap one ``DiffusionStage._transformer_ctx``. Idempotent via the
    ``_teacache_patched`` marker."""
    if getattr(stage, "_teacache_patched", False):
        logger.warning(
            "TeaCache [%s]: already patched, skipping", stage_name
        )
        return

    original_ctx = stage._transformer_ctx

    @functools.wraps(original_ctx)
    def wrapped_ctx(*args: Any, **kwargs: Any) -> AbstractContextManager:
        inner = original_ctx(*args, **kwargs)
        return _PatchedContextManager(inner, threshold, stage_name)

    stage._transformer_ctx = wrapped_ctx
    stage._teacache_patched = True


def enable_teacache(
    pipeline: Any,
    threshold: float = 0.03,
    stages: tuple[str, ...] = ("stage_1",),
) -> int:
    """Patch named stages of a two-stage LTX pipeline; returns count patched.

    ``threshold`` is the rel_l1 cache-miss cutoff (0.03 ≈ lossless, 0.05 ≈
    aggressive ≈ 2× speedup with small quality hit). ``stage_2``'s 3-step
    distilled schedule is too short to amortise caching.
    """
    patched = 0
    for stage_attr in stages:
        stage = getattr(pipeline, stage_attr, None)
        if stage is None:
            logger.warning("TeaCache: no stage named %r on pipeline", stage_attr)
            continue
        if not hasattr(stage, "_transformer_ctx"):
            logger.warning(
                "TeaCache: stage %r has no _transformer_ctx, skipping", stage_attr
            )
            continue
        _patch_stage(stage, threshold, stage_name=stage_attr)
        patched += 1
    logger.info("TeaCache: %d stage(s) patched", patched)
    return patched


def teacache_config_from_env() -> dict[str, Any] | None:
    """Build kwargs for ``enable_teacache(**cfg)`` from
    :class:`src.config.Settings`; ``None`` when disabled."""
    s = get_settings()
    if not s.enable_teacache:
        return None
    return {"threshold": s.teacache_threshold, "stages": s.teacache_stages}
