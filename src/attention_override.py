"""FlashAttention 3 enablement for LTX-2.3.

LTX-2's ``AttentionFunction.DEFAULT`` resolves to XFormersAttention (if
xformers is installed) or PytorchAttention — never FA3. To force FA3 we
monkey-patch the model configurator at process boot to inject
``attention_type=flash_attention_3`` into the transformer config dict before
the LTXModel is built. Every ``Attention`` module constructed thereafter
captures ``FlashAttention3()`` as its callable.

A defensive SDPA fallback is installed on ``FlashAttention3.__call__`` for
any non-None mask. Static trace of the 22B AV pipeline confirmed every
attention call site receives ``mask=None`` for normal text-to-video inputs
(Gemma's attention_mask is produced but immediately discarded in
``TI2VidTwoStagesPipeline.__call__``, and ``modality_from_latent_state``
hardcodes ``context_mask=None``). The fallback is insurance against future
pipeline variants — if the first-call log line fires, FA3's fast path is not
being taken on every layer and we need to investigate.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_applied = False


def enable_flash_attention_3() -> None:
    """Install FA3 configurator patch and mask fallback. Idempotent."""
    global _applied
    if _applied:
        return

    import flash_attn_interface  # noqa: F401 — fail loud if wheel is missing
    fa3_version = getattr(flash_attn_interface, "__version__", "unknown")

    from ltx_core.model.transformer.attention import FlashAttention3
    from ltx_core.model.transformer import model_configurator as _mc

    _install_mask_fallback(FlashAttention3)
    _install_configurator_patch(_mc)

    _applied = True
    logger.info(
        "FA3 enabled: configurator patched (flash_attn_interface %s), mask-fallback installed",
        fa3_version,
    )


def _install_mask_fallback(FlashAttention3_cls) -> None:
    _original_call = FlashAttention3_cls.__call__

    def _patched_call(self, q, k, v, heads, mask=None):
        if mask is None:
            return _original_call(self, q, k, v, heads, mask=None)
        if not getattr(_patched_call, "_warned", False):
            logger.info("FA3 mask-fallback engaged on first call — running SDPA for masked attention")
            _patched_call._warned = True
        b, _, dim_head = q.shape
        dim_head //= heads
        q_, k_, v_ = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        out = torch.nn.functional.scaled_dot_product_attention(
            q_, k_, v_, attn_mask=mask, dropout_p=0.0, is_causal=False
        )
        return out.transpose(1, 2).reshape(b, -1, heads * dim_head)

    FlashAttention3_cls.__call__ = _patched_call


def _install_configurator_patch(mc_module) -> None:
    for cls_name in ("LTXModelConfigurator", "LTXVideoOnlyModelConfigurator"):
        cls = getattr(mc_module, cls_name, None)
        if cls is None:
            continue
        _original_func = cls.__dict__["from_config"].__func__

        def _make_patched(original):
            def _patched(cls_arg, config):
                transformer = dict(config.get("transformer", {}))
                transformer["attention_type"] = "flash_attention_3"
                return original(cls_arg, {**config, "transformer": transformer})
            return classmethod(_patched)

        cls.from_config = _make_patched(_original_func)
