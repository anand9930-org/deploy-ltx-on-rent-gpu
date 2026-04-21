"""Install SageAttention as the default LTX attention backend.

LTX-core ships with `AttentionFunction.DEFAULT` resolving to XFormers (or
PyTorch SDPA as a last resort). On Hopper H100 that leaves us on an
FA2-class mem-eff kernel even though `sageattn` exposes a Hopper-specific
INT8 QK + FP8 PV kernel with better speed *and* accuracy.

Every `Attention(...)` site inside the LTX transformer uses
`attention_function=AttentionFunction.DEFAULT`, so redirecting what DEFAULT
resolves to flips Sage on everywhere with no call-site changes.

Import this module once at process start — before any LTX pipeline import —
and it'll do the swap if the `sageattention` wheel is importable. If Sage
is unavailable the original XFormers/PyTorch resolver stays in place.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)


def _install() -> str:
    try:
        from sageattention import sageattn
    except ImportError:
        logger.info("sage_patch: sageattention not importable; leaving default attention backend")
        return "unchanged"

    from ltx_core.model.transformer import attention as _attn

    class SageAttention(_attn.AttentionCallable):
        """sageattn auto-dispatches to the optimal per-arch kernel.

        sm_90 (Hopper) → sageattn_qk_int8_pv_fp8_cuda_sm90 (INT8 QK + FP8 PV).
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
                raise NotImplementedError("Mask is not supported for SageAttention")
            b, _, dim_head = q.shape
            dim_head //= heads
            q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
            out = sageattn(q.to(v.dtype), k.to(v.dtype), v, tensor_layout="HND", is_causal=False)
            return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

    _attn.SageAttention = SageAttention
    _attn.sageattn = sageattn

    _original_to_callable = _attn.AttentionFunction.to_callable

    def to_callable(self):
        if self is _attn.AttentionFunction.DEFAULT:
            return SageAttention()
        return _original_to_callable(self)

    _attn.AttentionFunction.to_callable = to_callable
    logger.info("sage_patch: installed SageAttention as DEFAULT attention backend")
    return "installed"


status = _install()
