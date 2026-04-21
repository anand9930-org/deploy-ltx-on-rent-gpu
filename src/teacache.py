"""TeaCache integration for LTX-2.3.

Training-free diffusion-step caching. At each inference step, measures
how much the transformer input changed vs the previous step; if the
rescaled relative L1 distance falls below a threshold, skips the
transformer forward and reuses the previous output. Validated on
LTX-Video with 1.6–2.1× lossless end-to-end speedup (ali-vilab/TeaCache).

We patch at the `DiffusionStage._transformer_ctx` layer rather than
wrapping the denoising loop — this way the streaming + batch-split
machinery downstream is untouched, and the same patch works for
guided and simple denoisers alike.

Toggled via env var `ENABLE_TEACACHE=1`. By default caches only
`stage_1` (the expensive 30-step stage); `stage_2` has a fixed 3-step
distilled schedule so caching has no room to pay for itself.

Environment variables:
    ENABLE_TEACACHE        "1" / "true" / "yes" to enable (default off)
    TEACACHE_THRESHOLD     float in (0, 1], typically 0.03 lossless /
                           0.05 aggressive (default 0.03)
    TEACACHE_STAGES        comma-separated stage attrs to patch, e.g.
                           "stage_1" (default) or "stage_1,stage_2"
"""

from __future__ import annotations

import functools
import logging
import os
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


# Polynomial coefficients fitted on LTX-Video (ali-vilab/TeaCache).
# Kept in the same ordering used by numpy.polyval: highest-degree first.
# These may need re-fitting for LTX-2.3's longer seq_len + two-stage
# schedule; empirical testing comes first, re-fitting if needed.
_POLY_COEFFS: tuple[float, ...] = (
    2.14700694e1,
    -1.28016453e1,
    2.31279151e0,
    7.92487521e-1,
    9.69274326e-3,
)


def _polyval(coeffs: tuple[float, ...], x: float) -> float:
    """Horner-method polynomial evaluation, numpy-free so we don't
    drag numpy onto the hot path."""
    acc = 0.0
    for c in coeffs:
        acc = acc * x + c
    return acc


def _rescale(rel_l1: float) -> float:
    """Apply TeaCache's learned rescaling to a raw relative-L1 distance.
    Clamps the input to [0, 1] to stay inside the fit's validity region."""
    x = max(0.0, min(1.0, rel_l1))
    return _polyval(_POLY_COEFFS, x)


@dataclass
class _State:
    """Per-stage cache state, recreated fresh on each ``_transformer_ctx``
    entry (i.e. once per pipeline stage per generation)."""

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
    """Pick a single tensor to use as the similarity key for this call.
    Prefer video (bigger, dominates compute). Fall back to audio.
    Returns None iff both modalities are absent (shouldn't happen in
    LTX-2.3's two-stage pipeline but guarded for safety)."""
    if video is not None and getattr(video, "latent", None) is not None:
        return video.latent
    if audio is not None and getattr(audio, "latent", None) is not None:
        return audio.latent
    return None


def _make_wrapped_forward(
    original_forward: Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]],
    state: _State,
) -> Callable[..., tuple[torch.Tensor | None, torch.Tensor | None]]:
    """Build a replacement for ``X0Model.forward`` that short-circuits
    on cache hits. Captures ``original_forward`` (a bound method) and
    ``state`` in closure; the returned callable is assigned onto the
    transformer instance."""

    def forward(
        video: Any = None,
        audio: Any = None,
        perturbations: Any = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        ref_input = _extract_reference_input(video, audio)
        state.step += 1

        # First step of this stage: nothing cached, always compute.
        if state.prev_input is None or ref_input is None:
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
    """Wraps ``DiffusionStage._transformer_ctx``'s returned context
    manager so that, on entry, we patch ``X0Model.forward`` with the
    TeaCache wrapper. Restores the original forward on exit and logs
    per-stage stats (computes / skips / skip-rate)."""

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
            total = self._state.computes + self._state.skips
            # Effective DiT-call speedup: total steps / actual computes.
            # 0 skips → 1.00x (baseline). Higher means more steps elided.
            effective_speedup = (
                (total / self._state.computes) if self._state.computes > 0 else 0.0
            )
            logger.info(
                "TeaCache [%s] STATS: threshold=%.3f  total_steps=%d  "
                "computed=%d  skipped=%d  skip_rate=%.1f%%  "
                "effective_DiT_speedup=%.2fx",
                self._stage_name,
                self._threshold,
                total,
                self._state.computes,
                self._state.skips,
                100.0 * self._state.skip_rate,
                effective_speedup,
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
    """Install TeaCache on one ``DiffusionStage`` by wrapping its
    ``_transformer_ctx`` method. Idempotent-ish: marks the stage with
    ``_teacache_patched`` so a second call is a no-op."""
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
    """Install TeaCache on the named stages of a two-stage LTX pipeline.

    Args:
        pipeline: a ``TI2VidTwoStagesPipeline`` instance (or any object
            exposing the named stages as attributes with a
            ``_transformer_ctx`` method).
        threshold: rel_l1 cache-miss cutoff. 0.03 ≈ lossless; 0.05 ≈
            aggressive (≈ 2× speedup with small quality hit).
        stages: which stage attributes to patch. ``stage_2`` has a
            fixed 3-step distilled schedule, so caching there is
            rarely worth it; default patches ``stage_1`` only.

    Returns:
        Number of stages successfully patched.
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
    """Read ENABLE_TEACACHE / TEACACHE_THRESHOLD / TEACACHE_STAGES and
    return a config dict suitable for ``enable_teacache(**cfg)``, or
    ``None`` if disabled.

    Defaults:
        ENABLE_TEACACHE=0
        TEACACHE_THRESHOLD=0.03
        TEACACHE_STAGES=stage_1
    """
    if os.getenv("ENABLE_TEACACHE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None

    try:
        threshold = float(os.getenv("TEACACHE_THRESHOLD", "0.03"))
    except ValueError:
        logger.warning(
            "TeaCache: TEACACHE_THRESHOLD is not a float, falling back to 0.03"
        )
        threshold = 0.03

    if not (0.0 < threshold <= 1.0):
        logger.warning(
            "TeaCache: TEACACHE_THRESHOLD=%r out of (0, 1], falling back to 0.03",
            threshold,
        )
        threshold = 0.03

    stages_env = os.getenv("TEACACHE_STAGES", "stage_1")
    stages = tuple(s.strip() for s in stages_env.split(",") if s.strip())
    if not stages:
        stages = ("stage_1",)

    return {"threshold": threshold, "stages": stages}
