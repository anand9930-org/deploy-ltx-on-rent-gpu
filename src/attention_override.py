"""FA3 enablement + AttentionFunction singleton-cache.

Two patches on ``ltx_core.model.transformer.attention``:

  * ``enable_flash_attention_3()`` — routes every ``Attention`` module
    through the FA3 wrapper; falls back to SDPA if a non-None mask appears.
    Static trace shows masks are always None on T2V/I2V/V2V, so the
    fallback is insurance — if its log fires, FA3's fast path is silently
    being skipped and needs investigation.
  * ``enable_attention_callable_singleton()`` — memoises
    ``AttentionFunction.to_callable()`` so per-block ``attention_function``
    is id-stable across transformer rebuilds. Without this, every rebuild
    builds fresh callables, the Dynamo ``___check_obj_id`` guard fails on
    every block (47 × 2 × N rebuilds), the default
    ``accumulated_recompile_limit=256`` blows after ~3 jobs and Dynamo
    falls back to eager — the 2026-05-01 P3/P4 latency regression.

Both callables are stateless, so a single shared instance is equivalent
to per-module instantiation.
"""

import logging

import torch

logger = logging.getLogger(__name__)

_applied = False
_singleton_applied = False

# Exposed for the compile shim to print a per-rebuild diagnostic summary.
# Updated in-place by the patched ``to_callable``.
singleton_stats: dict = {
    "installed": False,
    "hits": 0,
    "misses": 0,
    "ids": {},  # enum_name -> id(callable)
}


def enable_flash_attention_3() -> None:
    """Install FA3 configurator patch and mask fallback. Idempotent."""
    global _applied
    if _applied:
        return

    import flash_attn_interface  # noqa: F401 — fail loud if wheel is missing
    fa3_version = getattr(flash_attn_interface, "__version__", "unknown")

    from src.upstream import FlashAttention3, ltx_model_configurator as _mc

    _install_mask_fallback(FlashAttention3)
    _install_configurator_patch(_mc)

    _applied = True
    logger.info(
        "FA3 enabled: configurator patched (flash_attn_interface %s), mask-fallback installed",
        fa3_version,
    )


def enable_attention_callable_singleton() -> None:
    """Memoise ``AttentionFunction.to_callable`` per enum value. Idempotent.

    Must run BEFORE the first transformer build. Dominant lever against
    torch.compile recompile thrash — id-stable ``attention_function``
    keeps the Dynamo obj_id guard hitting across per-job rebuilds.
    """
    global _singleton_applied
    if _singleton_applied:
        return

    from src.upstream import AttentionFunction

    _original_to_callable = AttentionFunction.to_callable
    _cache: dict = {}

    def _patched_to_callable(self):
        cached = _cache.get(self)
        if cached is not None:
            singleton_stats["hits"] += 1
            return cached
        callable_obj = _original_to_callable(self)
        _cache[self] = callable_obj
        singleton_stats["misses"] += 1
        singleton_stats["ids"][self.name] = id(callable_obj)
        logger.info(
            "AttentionFunction.to_callable: cached %s -> %s (id=%d). "
            "Future rebuilds will return this same instance — Dynamo "
            "obj_id guard on attn{1,2}.attention_function should now hit.",
            self.name, type(callable_obj).__name__, id(callable_obj),
        )
        return callable_obj

    AttentionFunction.to_callable = _patched_to_callable
    singleton_stats["installed"] = True
    _singleton_applied = True
    logger.info(
        "AttentionFunction singleton-cache installed on %s.to_callable",
        AttentionFunction.__module__,
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
