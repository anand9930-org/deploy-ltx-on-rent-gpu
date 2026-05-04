"""torch.compile config-flag compatibility shim for LTX-2 on NGC 25.06.

Upstream `ltx_core/model/transformer/compiling.py` wraps each compiled
forward in a `with` block that patches four config flags. One of them,
`torch._inductor.config.unsafe_skip_cache_dynamic_shape_guards`, was added
in PyTorch PR #150670 (merged 2025-04-15) — a few hours AFTER the SHA that
NGC 25.06 pins. So on this image, attempting to enter that context manager
raises `AttributeError` on the first compiled forward, and the entire
context rolls back without ever calling the model.

We monkey-patch `compile_transformer` to use an `ExitStack` that gates each
flag on `hasattr` before patching. The dropped flag is a cache-lookup
speed knob (skips sympy guard evaluation in `FxGraphCache._lookup_graph`),
not a correctness or kernel-runtime knob — so the big perf wins
(tracing inlining, recompile budget, int unspecialization) all still
apply on NGC 25.06.

Activated via `enable_compile_config_shim()` from `src/pipeline.py`. Must
run BEFORE `TI2VidTwoStagesPipeline` lazily builds any transformer (i.e.
before the first `__call__`).
"""

import contextlib
import logging

import torch

logger = logging.getLogger(__name__)

_applied = False


def enable_compile_config_shim() -> None:
    """Replace upstream `compile_transformer` with a hasattr-gated version.
    Idempotent."""
    global _applied
    if _applied:
        return

    from ltx_core.model.transformer import compiling as _c
    from ltx_core.model.transformer.model import LTXModel

    # `accumulated_recompile_limit` defaults to 256 in PyTorch. On this stack
    # the upstream pipeline rebuilds the entire transformer per job and the
    # default ceiling gets hit after ~3 jobs (47 blocks × 2 attns × ~3 rebuilds
    # ≈ 280 entries) — Dynamo then prints
    #   "torch._dynamo hit config.accumulated_recompile_limit (256)"
    # and falls back to eager, which is the failure mode observed in the
    # 2026-05-01 P3/P4 logs. Bumping to 8192 is harmless (it just caps the
    # cache memory budget) and gives plenty of headroom even if the
    # singleton patch in src/attention_override.py misses an edge case.
    desired = [
        (torch._inductor.config, "unsafe_skip_cache_dynamic_shape_guards", True),
        (torch._dynamo.config, "inline_inbuilt_nn_modules", True),
        (torch._dynamo.config, "cache_size_limit", 256),
        (torch._dynamo.config, "accumulated_recompile_limit", 8192),
        (torch._dynamo.config, "allow_unspec_int_on_nn_module", True),
    ]

    applied_names: list[str] = []
    skipped_names: list[str] = []
    for cfg, name, _value in desired:
        if hasattr(cfg, name):
            applied_names.append(f"{cfg.__name__}.{name}")
        else:
            skipped_names.append(f"{cfg.__name__}.{name}")
    logger.info(
        "torch.compile config shim — will apply: %s; will skip (missing on this PyTorch): %s",
        applied_names, skipped_names or "none",
    )

    _logged_first_forward = {"done": False}
    _rebuild_counter = {"n": 0}

    def patched_compile_transformer(model: LTXModel) -> LTXModel:
        _rebuild_counter["n"] += 1
        rebuild_n = _rebuild_counter["n"]

        # Snapshot attention-function singleton state at the moment the
        # transformer is built. After the first build, every subsequent
        # build should be all hits (no new ids), confirming the Dynamo
        # obj_id guard on attention_function will pass across rebuilds.
        try:
            from src.attention_override import singleton_stats as _attn_stats
            attn_snapshot = (
                f"installed={_attn_stats['installed']} "
                f"hits={_attn_stats['hits']} misses={_attn_stats['misses']} "
                f"ids={_attn_stats['ids']}"
            )
        except Exception:  # noqa: BLE001 — diagnostic, never fail the build
            attn_snapshot = "unavailable"

        n_blocks = len(model.transformer_blocks)
        model.transformer_blocks = torch.nn.ModuleList(
            torch.compile(m) for m in model.transformer_blocks
        )
        logger.info(
            "torch.compile rebuild #%d: wrapped %d transformer blocks. "
            "AttentionFunction singleton stats — %s",
            rebuild_n, n_blocks, attn_snapshot,
        )

        def patched_dynamo_forward(*args, **kwargs):
            stack = contextlib.ExitStack()
            with stack:
                for cfg, name, value in desired:
                    if hasattr(cfg, name):
                        stack.enter_context(cfg.patch(**{name: value}))
                if not _logged_first_forward["done"]:
                    logger.info(
                        "torch.compile first compiled forward entered — config patches active: %s",
                        applied_names,
                    )
                    _logged_first_forward["done"] = True
                return model.forward_without_compilation(*args, **kwargs)

        model.forward_without_compilation = model.forward
        model.forward = patched_dynamo_forward
        return model

    # `ModuleOps` is a NamedTuple (immutable). We cannot assign to .mutator;
    # instead, replace the module-level COMPILE_TRANSFORMER with a fresh
    # NamedTuple whose mutator points to our patched function. Anything that
    # already imported the old instance keeps a stale reference, so we ALSO
    # have to update every importer. In this codebase the only consumer is
    # `ltx_pipelines/utils/blocks.py` which does
    # `from ltx_core.model.transformer.compiling import COMPILE_TRANSFORMER`
    # at module-import time and then references it as a local name in
    # `DiffusionStage._build_transformer`. Patching both module bindings
    # covers it. (`compile_transformer` also gets reassigned in case other
    # code paths call it directly.)
    from ltx_core.loader.module_ops import ModuleOps

    new_op = ModuleOps(
        name=_c.COMPILE_TRANSFORMER.name,
        matcher=_c.COMPILE_TRANSFORMER.matcher,
        mutator=patched_compile_transformer,
    )
    _c.compile_transformer = patched_compile_transformer
    _c.COMPILE_TRANSFORMER = new_op

    try:
        from ltx_pipelines.utils import blocks as _blocks
        if hasattr(_blocks, "COMPILE_TRANSFORMER"):
            _blocks.COMPILE_TRANSFORMER = new_op
            logger.info(
                "torch.compile shim: ltx_pipelines.utils.blocks.COMPILE_TRANSFORMER rebound"
            )
    except ImportError:
        logger.warning(
            "torch.compile shim: ltx_pipelines.utils.blocks not importable; "
            "compile shim may not take effect if blocks.py held a stale binding"
        )

    _applied = True
    logger.info(
        "torch.compile shim installed: compile_transformer rewritten + COMPILE_TRANSFORMER rebound"
    )
