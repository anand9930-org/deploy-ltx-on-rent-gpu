"""Install SageAttention as the default LTX attention backend.

LTX-core ships with `AttentionFunction.DEFAULT` resolving to XFormers (or
PyTorch SDPA as a last resort). On Hopper H100 that leaves us on an
FA2-class mem-eff kernel even though Sage exposes kernels tuned for INT8
Q/K with various PV-accumulation strategies.

We install the precision-safe SageAttention2++ variant
(`sageattn_qk_int8_pv_fp8_cuda` with `pv_accum_dtype="fp32+fp16"`).
The plain `sm_90` FP8-PV kernel saturates on LTX-2.3 22B and produces
black / noisy output; Sage2++ keeps the INT8-QK speed win while moving
PV accumulation into mixed fp32+fp16 accumulators.

Every `Attention(...)` site inside the LTX transformer uses
`attention_function=AttentionFunction.DEFAULT`, so redirecting what
DEFAULT resolves to flips Sage on everywhere with no call-site changes.
The Gemma text-encoder connector passes padding masks, which sageattn
cannot consume — those calls route to `PytorchAttention` (torch SDPA).

Import this module once at process start — before any LTX pipeline
import — and it'll do the swap if the `sageattention` wheel is
importable. If Sage is unavailable the original XFormers/PyTorch
resolver stays in place.

All logs use the `[SAGE_PATCH]` prefix for greppability in BentoML pod
stdout.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

_SAGE_CUDA_HEAD_DIMS = {64, 96, 128}


def _install() -> str:
    try:
        import sageattention  # noqa: F401 — verify importability before picking a kernel
    except ImportError:
        logger.info("[SAGE_PATCH] sageattention not importable; leaving default attention backend")
        return "unchanged"

    # Bug #1 fix: pick the precision-safe Sage2++ kernel instead of the
    # auto-dispatched sm_90 FP8-PV kernel. Fallback chain keeps us working
    # even if the Comfy-Org wheel changes what it exports.
    try:
        from sageattention import sageattn_qk_int8_pv_fp8_cuda as _sage_fn
        _sage_kwargs = {"pv_accum_dtype": "fp32+fp16"}
        _kernel_name = "sageattn_qk_int8_pv_fp8_cuda (Sage2++ / pv_accum=fp32+fp16)"
    except ImportError:
        try:
            from sageattention import sageattn_qk_int8_pv_fp16_triton as _sage_fn
            _sage_kwargs: dict = {}
            _kernel_name = "sageattn_qk_int8_pv_fp16_triton (FP16-PV fallback)"
        except ImportError:
            from sageattention import sageattn as _sage_fn
            _sage_kwargs = {}
            _kernel_name = "sageattn auto-dispatch (LAST RESORT — may be lossy on sm_90)"

    device_capability = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else None
    logger.info(
        "[SAGE_PATCH] kernel=%s device_capability=%s sageattention_version=%s",
        _kernel_name,
        device_capability,
        getattr(sageattention, "__version__", "unknown"),
    )

    from ltx_core.model.transformer import attention as _attn

    _fallback_mask_attention = _attn.PytorchAttention()

    # One-shot diagnostic state so first-call logs run exactly once per process.
    _diag = {"first_call_logged": False, "finite_checked": False}

    class SageAttention(_attn.AttentionCallable):
        """Sage2++ for mask-free DiT calls, torch SDPA fallback when masked.

        The Gemma text-encoder connector passes a padding mask which sageattn
        cannot consume; those calls route through `PytorchAttention`. Every
        DiT block (and the spatial upscaler's DiT) flows through the Sage
        path.
        """

        def __call__(
            self,
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            heads: int,
            mask: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if mask is not None:
                return _fallback_mask_attention(q, k, v, heads, mask)

            b, seq_len, inner_dim = q.shape
            dim_head = inner_dim // heads

            # Bug #2 fix: emit one-shot observability on the first call so we
            # can verify heads / head_dim / dtypes from BentoML logs without
            # instrumenting every call.
            if not _diag["first_call_logged"]:
                _diag["first_call_logged"] = True
                logger.info(
                    "[SAGE_PATCH] first call: heads=%d head_dim=%d seq_len=%d "
                    "q.dtype=%s k.dtype=%s v.dtype=%s kernel=%s",
                    heads, dim_head, seq_len, q.dtype, k.dtype, v.dtype, _kernel_name,
                )
                if dim_head not in _SAGE_CUDA_HEAD_DIMS:
                    logger.warning(
                        "[SAGE_PATCH] head_dim=%d outside Sage CUDA-supported set %s; "
                        "kernel may fall back to Triton or miscompute",
                        dim_head, sorted(_SAGE_CUDA_HEAD_DIMS),
                    )

            q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
            out = _sage_fn(
                q.to(v.dtype), k.to(v.dtype), v,
                tensor_layout="HND", is_causal=False,
                **_sage_kwargs,
            )

            # Bug #3 fix: sanity-check the first attention output for
            # NaN/Inf saturation. One reduction per generation — negligible
            # cost, but pinpoints a broken kernel the moment it fires.
            if not _diag["finite_checked"]:
                _diag["finite_checked"] = True
                if not torch.isfinite(out).all():
                    logger.error(
                        "[SAGE_PATCH] non-finite values in first attention output — "
                        "kernel %s is saturating; DiT output will be noise",
                        _kernel_name,
                    )
                else:
                    abs_max = out.abs().max().item()
                    logger.info(
                        "[SAGE_PATCH] first output finite: min=%.3g max=%.3g abs_max=%.3g",
                        out.min().item(), out.max().item(), abs_max,
                    )

            return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

    _attn.SageAttention = SageAttention
    _attn.sageattn = _sage_fn

    _original_to_callable = _attn.AttentionFunction.to_callable

    def to_callable(self):
        if self is _attn.AttentionFunction.DEFAULT:
            return SageAttention()
        return _original_to_callable(self)

    _attn.AttentionFunction.to_callable = to_callable
    logger.info("[SAGE_PATCH] installed SageAttention as DEFAULT attention backend")
    return "installed"


status = _install()
