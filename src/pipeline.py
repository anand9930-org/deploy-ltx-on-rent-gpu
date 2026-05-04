"""LTX-2.3 Unified Pipeline wrapper.

Dispatches across two upstream pipelines based on request shape:
  - prompt only             → ``TI2VidTwoStagesPipeline`` (T2V two-stage, 30 steps,
                              dev-fp8 stage 1 + distilled-fp8 stage 2). Identical to
                              the proven ``feature/fp8-h100`` configuration.
  - prompt + image          → ``ICLoraPipeline`` (I2V, identity-strict via IC-LoRA
                              Union-Control fused on stage 1).
  - prompt + reference_video → ``ICLoraPipeline`` (V2V, style transfer / edit).

H100 80 GB cannot hold both upstream pipelines resident simultaneously; the
wrapper keeps at most one alive and lazy-swaps on cross-mode requests
(~30-60 s build penalty per swap). ``LTX_DEFAULT_MODE`` (``i2v`` default) picks
which side preloads at boot.
"""

import functools
import gc
import logging
import os
import re
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)


# FP8 + expandable_segments — required by the upstream LTX FP8 README:
# the scaled_mm path's CUDA graph capture is incompatible with the default
# allocator's segment splitting. Set before any tensor is allocated, but
# only when the FP8 path is requested so BF16 deployments stay unchanged.
if os.environ.get("LTX_FP8_MODE", "").strip().lower() in ("scaled_mm", "cast"):
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, oversaturated"
)


def _round_to(value: int, divisor: int) -> int:
    return (value // divisor) * divisor


def _round_frames(n: int) -> int:
    return ((n - 1) // 8) * 8 + 1


def _round_user_inputs(width: int, height: int, num_frames: int) -> tuple[int, int, int]:
    """Round user-supplied dims to pipeline grid and log when we change anything.

    LTX-2.3 requires width/height divisible by 64 (latent stride × patch size)
    and frame counts of the form 8k+1. Silently clamping surprises callers when
    a request that "worked" locally comes back at a different resolution; the
    log line makes the adjustment discoverable from pod output.
    """
    w, h, f = _round_to(width, 64), _round_to(height, 64), _round_frames(num_frames)
    if (w, h, f) != (width, height, num_frames):
        logger.info(
            "Input rounded to pipeline grid: %dx%d×%d → %dx%d×%d",
            width, height, num_frames, w, h, f,
        )
    return w, h, f


def _install_stage2_cleanup_hook(pipeline) -> None:
    """Flush device + host allocators at the Stage 1 → Stage 2 boundary.

    LTX-2's weight-offloading path (``OffloadMode.CPU`` / ``DISK`` via
    ``block_streaming``) pins transformer blocks in host memory during
    inference. Upstream removed the best-effort host-side empty_cache
    that used to run in the streaming teardown, so pinned pages are no
    longer returned to the OS between stages. On VRAM-constrained pods
    that intermittently exhausts the pinned arena and the next
    ``tensor.data.pin_memory()`` call (inside the Stage 2 pool builder)
    raises ``torch.AcceleratorError: CUDA error: invalid argument``.

    We force a synchronous cleanup cycle (Python GC → device empty_cache
    → CUDA sync → host empty_cache) right before Stage 2's transformer
    context manager enters, so the pinned arena is in a known-drained
    state. Idempotent; safe to call multiple times.
    """
    stage = getattr(pipeline, "stage_2", None)
    if stage is None:
        logger.warning("Stage 2 cleanup hook: pipeline has no stage_2")
        return
    if getattr(stage, "_stage2_cleanup_hooked", False):
        return

    original_ctx = stage._transformer_ctx

    @functools.wraps(original_ctx)
    def hooked_ctx(*args, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            alloc_before = torch.cuda.memory_allocated(0) / 1e9
            res_before = torch.cuda.memory_reserved(0) / 1e9
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch._C, "_host_emptyCache"):
                try:
                    torch._C._host_emptyCache()
                except Exception:  # pragma: no cover — best effort
                    logger.debug("Stage 2 cleanup: _host_emptyCache raised", exc_info=True)
            alloc_after = torch.cuda.memory_allocated(0) / 1e9
            res_after = torch.cuda.memory_reserved(0) / 1e9
            alloc_delta = alloc_before - alloc_after
            res_delta = res_before - res_after
            alloc_note = (
                "" if alloc_delta > 0.01
                else " (unchanged; stage 1 already released via gpu_model.__exit__)"
            )
            logger.info(
                "Stage 2 boundary cleanup: alloc %.2f → %.2f GB%s, "
                "reserved %.2f → %.2f GB (%.2f GB returned to pool)",
                alloc_before, alloc_after, alloc_note,
                res_before, res_after, res_delta,
            )
        else:
            logger.info("Stage 2 boundary cleanup done (no CUDA)")
        return original_ctx(*args, **kwargs)

    stage._transformer_ctx = hooked_ctx
    stage._stage2_cleanup_hooked = True


# ---- FP8 helpers (identical to the parent feature/fp8-h100 branch) ---------
# These adapt to Lightricks' specific FP8 packaging: a `model.diffusion_model.`
# comfy prefix on disk, BF16 leftovers in transformer block 1 that upstream's
# baseline `EXCLUDED_LAYER_SUBSTRINGS` doesn't cover, and `_orig_mod.` segments
# injected by `torch.compile` that break upstream's substring matcher. Without
# them the FP8 load fails at first inference with FP8Linear/Linear shape
# mismatches.

# Lightricks FP8 safetensors store DiT weights with this prefix on disk;
# `LTXV_MODEL_COMFY_RENAMING_MAP` strips it when the loader reads them. Our
# probe reads RAW keys so we strip it too — otherwise the exclusion names we
# emit won't match the in-memory module paths.
_COMFY_PREFIX = "model.diffusion_model."

# torch.compile wraps each transformer block in an OptimizedModule whose inner
# attributes are reached through `_orig_mod`. After compile,
# `model.named_modules()` yields paths like
# `transformer_blocks.1._orig_mod.attn1.to_gate_logits`. Strip the injected
# segment before substring matching so exclusions can use the plain
# (pre-compile) form regardless of whether compile ran.
_ORIG_MOD_SEGMENT_RE = re.compile(r"\._orig_mod(?=\.|$)")


def _strip_comfy_prefix(name: str) -> str:
    return name[len(_COMFY_PREFIX):] if name.startswith(_COMFY_PREFIX) else name


def _normalize_for_match(name: str) -> str:
    return _ORIG_MOD_SEGMENT_RE.sub("", _strip_comfy_prefix(name))


def _probe_fp8_exclusions(paths: list[str]) -> tuple[str, ...]:
    """Return module paths that must stay as nn.Linear (BF16).

    Opens each safetensors header (no tensor data loaded) and collects every
    ``.weight`` key whose dtype is not ``float8_e4m3fn``. Each name is
    canonicalized by stripping ``model.diffusion_model.`` so it matches the
    post-rename module path the LTX loader produces.

    Why we probe: the Lightricks ``*-fp8.safetensors`` checkpoints leave a
    subset of transformer block 1 in BF16 (AV cross modules, a few MLPs) that
    upstream's hard-coded ``EXCLUDED_LAYER_SUBSTRINGS`` does not cover.
    Without these extras, ``_apply_fp8_prepare_to_model`` swaps those modules
    to FP8Linear and ``load_state_dict`` fails with size mismatches at first
    inference.
    """
    from safetensors import safe_open  # lazy — downloads pull safetensors

    bare: set[str] = set()
    for path in paths:
        dtype_counts: dict[str, int] = {}
        bf16_samples: list[str] = []
        file_bare_adds = 0
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if not key.endswith(".weight"):
                    continue
                dt = f.get_slice(key).get_dtype()
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                if dt != "F8_E4M3":
                    normalized = _strip_comfy_prefix(key[: -len(".weight")])
                    bare.add(normalized)
                    file_bare_adds += 1
                    if len(bf16_samples) < 3:
                        bf16_samples.append(f"{key} [{dt}]")
        logger.info(
            "FP8 probe: %s — dtype histogram=%s, non-FP8 weights=%d",
            os.path.basename(path),
            sorted(dtype_counts.items()),
            file_bare_adds,
        )
        if bf16_samples:
            logger.info(
                "FP8 probe: %s — raw BF16 samples: %s",
                os.path.basename(path), bf16_samples,
            )

    block_entries = sorted(n for n in bare if n.startswith("transformer_blocks."))
    if not block_entries:
        logger.warning(
            "FP8 probe: NO transformer_blocks.* BF16 weights found across %d "
            "checkpoint(s). Either the Lightricks FP8 format changed or the "
            "prefix/dtype probe logic is stale. Block-1 load is expected to "
            "fail with size mismatches.",
            len(paths),
        )
    else:
        logger.info(
            "FP8 probe: %d transformer_blocks.* BF16 entries (first 10): %s",
            len(block_entries),
            block_entries[:10],
        )

    return tuple(sorted(bare))


def _build_scaled_mm_policy(extras: tuple[str, ...]):
    """Wrapper-only equivalent of ``QuantizationPolicy.fp8_scaled_mm()`` that
    accepts a per-checkpoint exclusion list AND normalizes names before
    matching.

    The upstream factory has two problems for the Lightricks
    ``*-fp8.safetensors`` format:

    1. Its ``EXCLUDED_LAYER_SUBSTRINGS`` doesn't cover the BF16 modules
       Lightricks left in transformer block 1 — append ``extras`` produced by
       ``_probe_fp8_exclusions``.
    2. ``_should_skip_layer`` upstream is a raw substring check. With regional
       ``torch.compile`` active, module names pick up an ``._orig_mod.``
       segment that breaks substring matching against bare exclusions. We
       replace ``_apply_fp8_prepare_to_model`` and
       ``_create_transpose_kv_operation`` with local versions that normalize
       ``_orig_mod`` out before the check.
    """
    from src.upstream import (
        FP8_PREPARE_MODULE_OPS,
        FP8_TRANSPOSE_SD_OPS,
        FP8Linear,
        KeyValueOperationResult,
        LTXModel,
        ModuleOps,
        QuantizationPolicy,
        SDOps,
        _linear_to_fp8linear,
    )

    try:
        import tensorrt_llm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "tensorrt_llm not installed — fp8-trtllm extra missing"
        ) from e

    if not extras:
        # No checkpoint-driven extras — but we still want `_orig_mod`
        # normalization. Upstream's baseline exclusions like
        # `"transformer_blocks.0."` happen to tolerate compile (they stop at
        # the dot, so substring matches regardless), so the upstream
        # primitives are safe here.
        return QuantizationPolicy(
            sd_ops=FP8_TRANSPOSE_SD_OPS,
            module_ops=(FP8_PREPARE_MODULE_OPS,),
        )

    # Exact match against the probe result, NOT substring match against
    # upstream's baseline. Baseline includes block-level catch-alls like
    # `"transformer_blocks.0."` that substring-match every module in those
    # blocks; Lightricks' FP8 checkpoint keeps some modules there as FP8 —
    # the substring rule over-excluded them, leaving FP8 weights loaded into
    # plain nn.Linear modules and crashing F.linear on first forward.
    exclusions = frozenset(extras)

    def _should_skip(name: str) -> bool:
        return _normalize_for_match(name) in exclusions

    from torch import nn

    def _prepare(model):
        from collections import defaultdict

        replacements: list[tuple[nn.Module, str, nn.Linear]] = []
        block_totals: dict[int, int] = defaultdict(int)
        block_swapped: dict[int, int] = defaultdict(int)
        block_skipped: dict[int, int] = defaultdict(int)
        block_swap_samples: dict[int, list[str]] = defaultdict(list)
        block_skip_samples: dict[int, list[str]] = defaultdict(list)

        _BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")

        def _block_idx(n: str) -> int | None:
            m = _BLOCK_RE.match(n)
            return int(m.group(1)) if m else None

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
                continue
            normalized = _normalize_for_match(name)
            bi = _block_idx(normalized)
            if bi is not None:
                block_totals[bi] += 1
            if _should_skip(name):
                if bi is not None:
                    block_skipped[bi] += 1
                    if len(block_skip_samples[bi]) < 3:
                        block_skip_samples[bi].append(normalized)
                continue
            if bi is not None:
                block_swapped[bi] += 1
                if len(block_swap_samples[bi]) < 3:
                    block_swap_samples[bi].append(normalized)
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            replacements.append((parent, attr_name, module))

        for parent, attr_name, linear in replacements:
            setattr(parent, attr_name, _linear_to_fp8linear(linear))

        total_in_blocks = sum(block_totals.values())
        total_swapped = sum(block_swapped.values())
        total_skipped = sum(block_skipped.values())
        pct_swapped = (100.0 * total_swapped / total_in_blocks) if total_in_blocks else 0.0
        logger.info(
            "FP8 prepare: aggregate over %d transformer blocks — "
            "linears=%d, swapped=%d (%.1f%%), skipped=%d",
            len(block_totals), total_in_blocks, total_swapped, pct_swapped, total_skipped,
        )

        all_bf16_blocks = []
        for bi in sorted(block_totals.keys()):
            tag = ""
            if block_totals[bi] > 0 and block_swapped[bi] == 0:
                tag = " [ALL-BF16]"
                all_bf16_blocks.append(bi)
            logger.info(
                "FP8 prepare: block-%02d count=%d, swapped=%d, skipped=%d%s",
                bi, block_totals[bi], block_swapped[bi], block_skipped[bi], tag,
            )
        if all_bf16_blocks:
            logger.info(
                "FP8 prepare: blocks fully BF16 (swapped=0): %s",
                all_bf16_blocks,
            )

        seen = sorted(block_totals.keys())
        spot_blocks = []
        if seen:
            spot_blocks = [seen[0]]
            mid = seen[len(seen) // 2]
            if mid not in spot_blocks:
                spot_blocks.append(mid)
            if seen[-1] not in spot_blocks:
                spot_blocks.append(seen[-1])
        for bi in spot_blocks:
            if block_swap_samples[bi]:
                logger.info(
                    "FP8 prepare: block-%02d swap samples: %s",
                    bi, block_swap_samples[bi],
                )
            if block_skip_samples[bi]:
                logger.info(
                    "FP8 prepare: block-%02d skip samples: %s",
                    bi, block_skip_samples[bi],
                )
        return model

    def _transpose_op(key: str, value: torch.Tensor):
        if not key.endswith(".weight"):
            return [KeyValueOperationResult(key, value)]
        if value.dim() != 2 or value.dtype != torch.float8_e4m3fn:
            return [KeyValueOperationResult(key, value)]
        layer_name = key.rsplit(".weight", 1)[0]
        if _should_skip(layer_name):
            return [KeyValueOperationResult(key, value)]
        return [KeyValueOperationResult(key, value.t())]

    sd_ops = SDOps("fp8_transpose_weights_normalized").with_kv_operation(
        _transpose_op,
        key_prefix="transformer_blocks.",
        key_suffix=".weight",
    )
    module_ops = ModuleOps(
        name="fp8_prepare_for_loading_normalized",
        matcher=lambda m: isinstance(m, LTXModel),
        mutator=_prepare,
    )
    return QuantizationPolicy(sd_ops=sd_ops, module_ops=(module_ops,))


def _select_fp8_mode() -> str | None:
    """Pick the FP8 compute path. Returns None when FP8 is not requested.

    Order of precedence:
      1. ``LTX_FP8_MODE=scaled_mm|cast`` env override.
      2. None when LTX_FP8_MODE is unset/empty (BF16 default).
    """
    override = os.getenv("LTX_FP8_MODE", "").strip().lower()
    if override in ("scaled_mm", "cast"):
        return override
    return None


def _install_build_transformer_audit(stage, label: str) -> None:
    """Wrap ``DiffusionStage._build_transformer`` with timing + dtype audit.

    Logs wall time for the transformer build (load → SDOps transpose →
    ``_prepare`` swap) and a post-load dtype histogram over
    ``transformer_blocks.*`` parameters. The histogram is the GROUND TRUTH for
    "did the FP8 swap actually take?"; the per-block prepare-step log only
    reflects what the matcher decided to swap, which can diverge from the
    actual loaded buffers if the matcher picked the wrong module names.

    Idempotent; safe to call once per stage.
    """
    if getattr(stage, "_build_transformer_audited", False):
        return
    if not hasattr(stage, "_build_transformer"):
        logger.warning(
            "Audit hook [%s]: stage has no _build_transformer; skipping",
            label,
        )
        return

    original_build = stage._build_transformer

    @functools.wraps(original_build)
    def audited_build(*args, **kwargs):
        logger.info("[%s] _build_transformer start", label)
        if torch.cuda.is_available():
            alloc_before = torch.cuda.memory_allocated(0) / 1e9
            res_before = torch.cuda.memory_reserved(0) / 1e9
            torch.cuda.reset_peak_memory_stats(0)
        else:
            alloc_before = res_before = 0.0
        t_start = time.perf_counter()
        model = original_build(*args, **kwargs)
        elapsed = time.perf_counter() - t_start
        if torch.cuda.is_available():
            alloc_after = torch.cuda.memory_allocated(0) / 1e9
            res_after = torch.cuda.memory_reserved(0) / 1e9
            this_stage_peak = torch.cuda.max_memory_allocated(0) / 1e9
        else:
            alloc_after = res_after = this_stage_peak = 0.0
        logger.info(
            "[%s] _build_transformer done in %.1fs (load+swap+assign). "
            "Live alloc %.2f → %.2f GB, reserved %.2f → %.2f GB, "
            "this-stage peak %.2f GB.",
            label, elapsed,
            alloc_before, alloc_after, res_before, res_after, this_stage_peak,
        )

        try:
            from collections import Counter
            dtype_counts: Counter = Counter()
            block_dtype_sets: dict[int, set[str]] = {}
            bf16_in_body: list[str] = []
            block_re = re.compile(r"transformer_blocks\.(\d+)\.")
            for pname, p in model.named_parameters():
                m = block_re.search(pname)
                if m is None:
                    continue
                dt_name = str(p.dtype).removeprefix("torch.")
                dtype_counts[dt_name] += 1
                bi = int(m.group(1))
                block_dtype_sets.setdefault(bi, set()).add(dt_name)
                if (
                    dt_name == "bfloat16"
                    and bi >= 2
                    and len(bf16_in_body) < 5
                    and pname.endswith(".weight")
                ):
                    bf16_in_body.append(pname)

            total = sum(dtype_counts.values())
            fp8_n = dtype_counts.get("float8_e4m3fn", 0)
            logger.info(
                "FP8 audit [%s]: %d FP8 weight tensors loaded. "
                "Param-level dtype histogram (transformer_blocks.*, total=%d): %s",
                label, fp8_n, total, dict(dtype_counts),
            )
            if bf16_in_body:
                logger.info(
                    "FP8 audit [%s]: first %d BF16 weight names in blocks ≥ 2: %s",
                    label, len(bf16_in_body), bf16_in_body,
                )
            fully_bf16_blocks = sorted(
                bi for bi, ds in block_dtype_sets.items()
                if "float8_e4m3fn" not in ds
            )
            if fully_bf16_blocks:
                logger.info(
                    "FP8 audit [%s]: blocks with NO FP8 params: %s",
                    label, fully_bf16_blocks,
                )
        except Exception:
            logger.exception("FP8 audit [%s]: dtype histogram failed", label)

        return model

    stage._build_transformer = audited_build
    stage._build_transformer_audited = True


# Modes that the dispatcher recognises. Internal — do not expose to the wire.
_MODE_T2V = "t2v"
_MODE_UNIFIED = "unified"  # serves I2V + V2V via ICLoraPipeline


class LTXVideoGenerator:
    """Unified Pipeline wrapper around two upstream pipelines.

    T2V uses ``TI2VidTwoStagesPipeline`` (dev-fp8 + distilled-fp8, 30 steps —
    the proven feature/fp8-h100 configuration). I2V/V2V use ``ICLoraPipeline``
    (distilled-1.1 BF16 base + IC-LoRA Union-Control on stage 1, distilled-fp8
    on stage 2 in scaled_mm mode). Only one upstream pipeline is resident at a
    time; cross-mode requests trigger a tear-down + rebuild.
    """

    def __init__(self, model_dir: str = "/models") -> None:
        self._model_dir = model_dir

        # Common asset paths.
        self._spatial_upsampler_path = os.path.join(
            model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        self._gemma_root = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")

        # T2V (TI2VidTwoStages) assets.
        self._dev_bf16_path = os.path.join(model_dir, "ltx-2.3-22b-dev.safetensors")
        self._distilled_lora_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        self._dev_fp8_path = os.path.join(model_dir, "ltx-2.3-22b-dev-fp8.safetensors")

        # Unified (ICLora) assets — distilled-1.1 BF16 base + IC-LoRA delta.
        from src.download_models import (
            DISTILLED_CHECKPOINT_FILENAME,
            IC_LORA_FILENAME,
        )
        self._distilled_bf16_path = os.path.join(model_dir, DISTILLED_CHECKPOINT_FILENAME)
        self._ic_lora_path = os.path.join(model_dir, IC_LORA_FILENAME)

        # Shared FP8 stage 2 (and ICLora stage 1 base) — Lightricks-pre-fused.
        self._distilled_fp8_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-fp8.safetensors"
        )

        # Resolve runtime configuration once. Both pipelines share these.
        self._fp8_mode = _select_fp8_mode()
        self._fp8_enabled = self._fp8_mode is not None

        from src.upstream import encode_video, OffloadMode

        self._encode_video = encode_video
        self._OffloadMode = OffloadMode

        gpu_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0
        )
        # FP8 → no offload (fits two ~30 GB DiTs on 80 GB H100 + unlocks compile).
        # BF16 → CPU streaming on <40 GB pods, NONE otherwise.
        if self._fp8_enabled:
            self._offload_mode = OffloadMode.NONE
        else:
            self._offload_mode = (
                OffloadMode.NONE if gpu_vram_gb >= 40 else OffloadMode.CPU
            )
        logger.info(
            "Offload mode: %s (GPU=%.0f GB; precision=%s)",
            self._offload_mode.value, gpu_vram_gb,
            "fp8" if self._fp8_enabled else "bf16",
        )

        # torch.compile only legal when offload_mode == NONE (upstream guard).
        env_compile = os.getenv("ENABLE_TORCH_COMPILE", "1").strip().lower()
        torch_compile_requested = env_compile not in ("0", "false", "no", "off", "")
        if torch_compile_requested and self._offload_mode != OffloadMode.NONE:
            logger.warning(
                "torch.compile requested but offload_mode=%s disallows it "
                "(upstream DiffusionStage guard). Running uncompiled.",
                self._offload_mode.value,
            )
            self._torch_compile_enabled = False
        else:
            self._torch_compile_enabled = torch_compile_requested
            if torch_compile_requested:
                logger.info("torch.compile ENABLED (regional per transformer block)")
            else:
                logger.info(
                    "torch.compile disabled via ENABLE_TORCH_COMPILE=%s",
                    os.getenv("ENABLE_TORCH_COMPILE"),
                )

        # FA3 attention patch — runs ONCE for the process lifetime so it's
        # active for whichever pipeline we build first AND any rebuild.
        self._requested_attn = os.environ.get("LTX_ATTENTION_TYPE", "").lower()
        if self._requested_attn == "flash_attention_3":
            from src.attention_override import enable_flash_attention_3
            enable_flash_attention_3()

        # Compatibility shim: upstream `compile_transformer` patches a
        # PyTorch 2.8+ inductor flag (`unsafe_skip_cache_dynamic_shape_guards`)
        # that does not exist on the NGC 25.06 PyTorch pin. Without the shim
        # the first compiled forward crashes before running. Idempotent —
        # one boot-time install covers `_build_t2v`, `_build_unified`, and
        # every cross-mode rebuild.
        if self._torch_compile_enabled:
            from src.compile_override import enable_compile_config_shim
            enable_compile_config_shim()
            # Stabilise id() of every Attention.attention_function across
            # transformer rebuilds. Without this the Dynamo obj_id guard
            # on attn{1,2}.attention_function fails on every per-job
            # rebuild (47 blocks × 2 attns × N rebuilds), exhausts the
            # accumulated_recompile_limit, and drops back to eager — the
            # smoking gun in the 2026-05-01 P3/P4 latency report and again
            # in the 2026-05-04 logs (28).txt: 191 recompiles across 2
            # jobs, warm-start first-step 190 s vs cold-start 128 s.
            from src.attention_override import enable_attention_callable_singleton
            enable_attention_callable_singleton()

        # Shared TilingConfig helpers (optional).
        from src.upstream import HAS_TILING, TilingConfig, get_video_chunks_number
        self._TilingConfig = TilingConfig if HAS_TILING else None
        self._get_video_chunks_number = get_video_chunks_number if HAS_TILING else None

        # Per-checkpoint FP8 exclusion cache. Avoids re-probing the same
        # safetensors header on every cross-mode rebuild.
        self._fp8_extras_cache: dict[str, tuple[str, ...]] = {}

        # Pipeline cache — at most one resident.
        self._pipeline = None
        self._active_mode: str | None = None
        # Per-pipeline metadata (e.g. ICLora's reference_downscale_factor).
        self._unified_meta: dict = {}
        self._teacache_enabled = False  # set by builders

        # Preload the requested mode at boot. Default `i2v` (most pods are
        # I2V-heavy; T2V pods set LTX_DEFAULT_MODE=t2v explicitly).
        default_mode_env = os.getenv("LTX_DEFAULT_MODE", "i2v").strip().lower()
        if default_mode_env == "t2v":
            initial_mode = _MODE_T2V
        elif default_mode_env in ("i2v", "v2v", "unified", ""):
            initial_mode = _MODE_UNIFIED
        else:
            logger.warning(
                "Unrecognised LTX_DEFAULT_MODE=%r; falling back to i2v",
                default_mode_env,
            )
            initial_mode = _MODE_UNIFIED
        logger.info("Preloading default mode: %s", initial_mode)
        self._ensure_mode(initial_mode)
        logger.info("Pipeline ready.")

    # ------------------------------------------------------------------
    # Mode swap + builders
    # ------------------------------------------------------------------

    def _ensure_mode(self, mode: str) -> None:
        if self._active_mode == mode and self._pipeline is not None:
            return
        if self._pipeline is not None:
            logger.info(
                "Mode swap: tearing down %s pipeline → building %s",
                self._active_mode, mode,
            )
            t_teardown = time.perf_counter()
            self._pipeline = None
            self._unified_meta = {}
            self._teacache_enabled = False
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "Mode swap: teardown took %.2fs", time.perf_counter() - t_teardown,
            )
        if mode == _MODE_T2V:
            self._build_t2v()
        elif mode == _MODE_UNIFIED:
            self._build_unified()
        else:
            raise ValueError(f"Unknown pipeline mode: {mode!r}")
        self._active_mode = mode

    def _extras_for(self, *paths: str) -> tuple[str, ...]:
        """Return cached union of FP8 BF16-exclusion entries across given paths."""
        merged: set[str] = set()
        for p in paths:
            if p not in self._fp8_extras_cache:
                self._fp8_extras_cache[p] = _probe_fp8_exclusions([p])
            merged.update(self._fp8_extras_cache[p])
        return tuple(sorted(merged))

    def _build_t2v(self) -> None:
        """Construct the parent feature/fp8-h100 ``TI2VidTwoStagesPipeline``.

        Stage 1: dev-fp8 (scaled_mm) or dev BF16 + distilled-LoRA (cast/bf16).
        Stage 2: distilled-fp8 (scaled_mm) or dev BF16 + distilled-LoRA.
        """
        OffloadMode = self._OffloadMode
        offload_mode = self._offload_mode
        fp8_mode = self._fp8_mode
        torch_compile_enabled = self._torch_compile_enabled

        from src.upstream import TI2VidTwoStagesPipeline

        for label, path in (
            ("dev BF16", self._dev_bf16_path),
            ("spatial upsampler", self._spatial_upsampler_path),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"T2V required {label} checkpoint missing at {path}. "
                    "Run download_models.py before pipeline init."
                )
        if fp8_mode == "scaled_mm":
            for label, path in (
                ("dev FP8 DiT", self._dev_fp8_path),
                ("distilled FP8 DiT", self._distilled_fp8_path),
            ):
                if not os.path.exists(path):
                    raise RuntimeError(
                        f"T2V scaled_mm requires {label} at {path}. "
                        "Run download_models.py with LTX_FP8_MODE=scaled_mm."
                    )

        from src.upstream import QuantizationPolicy
        if fp8_mode == "scaled_mm":
            extras_dev = self._extras_for(self._dev_fp8_path)
            extras_distilled = self._extras_for(self._distilled_fp8_path)
            logger.info(
                "FP8 checkpoint probe: dev=%d, distilled=%d non-FP8 weight modules",
                len(extras_dev), len(extras_distilled),
            )
            for label, extras in (("dev", extras_dev), ("distilled", extras_distilled)):
                if extras:
                    preview = ", ".join(extras[:5])
                    more = f" (+{len(extras) - 5} more)" if len(extras) > 5 else ""
                    logger.info("FP8 probe extras (%s, first 5): %s%s", label, preview, more)
            quantization_dev = _build_scaled_mm_policy(extras_dev)
            quantization_distilled = _build_scaled_mm_policy(extras_distilled)
            quantization = quantization_dev  # placeholder; rebuilt per-stage below
            logger.info(
                "T2V FP8 mode: scaled_mm (W8A8, TRT-LLM cublas_scaled_mm, H100-optimised)"
            )
        elif fp8_mode == "cast":
            quantization = QuantizationPolicy.fp8_cast()
            quantization_dev = None
            quantization_distilled = None
            logger.info("T2V FP8 mode: cast (W8A16, weights FP8 / activations BF16)")
        else:
            quantization = None
            quantization_dev = None
            quantization_distilled = None

        # Drop FP8 + compile when offload != NONE (upstream DiffusionStage guard).
        if offload_mode != OffloadMode.NONE and quantization is not None:
            logger.warning(
                "Offload mode %s requires non-quantized BF16 — dropping FP8 + compile.",
                offload_mode.value,
            )
            quantization = None
            quantization_dev = None
            quantization_distilled = None

        from src.upstream import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
            StateDictRegistry,
        )
        registry = None
        try:
            registry = StateDictRegistry()
            logger.info("T2V using StateDictRegistry (CPU weight caching)")
        except Exception:
            logger.warning("StateDictRegistry not available", exc_info=True)

        # Distilled LoRA — cast/bf16 only. scaled_mm uses pre-fused distilled-fp8.
        if fp8_mode != "scaled_mm":
            if not os.path.exists(self._distilled_lora_path):
                raise RuntimeError(
                    f"T2V (cast/bf16) requires distilled LoRA at "
                    f"{self._distilled_lora_path}. Run download_models.py."
                )
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    path=self._distilled_lora_path,
                    strength=0.8,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]
        else:
            distilled_lora = []

        pipeline_kwargs = dict(
            checkpoint_path=self._dev_bf16_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=self._spatial_upsampler_path,
            gemma_root=self._gemma_root,
            loras=[],
            offload_mode=offload_mode,
        )
        if quantization is not None:
            pipeline_kwargs["quantization"] = quantization
        if registry is not None:
            pipeline_kwargs["registry"] = registry
        if torch_compile_enabled and offload_mode == OffloadMode.NONE and quantization is not None:
            pipeline_kwargs["torch_compile"] = True

        _t0 = time.perf_counter()
        self._pipeline = TI2VidTwoStagesPipeline(**pipeline_kwargs)
        logger.info(
            "T2V init: TI2VidTwoStagesPipeline construction took %.2fs",
            time.perf_counter() - _t0,
        )
        self._log_vram("T2V after pipeline init")

        # scaled_mm: rebuild stages to point at FP8 DiT files.
        if fp8_mode == "scaled_mm":
            from src.upstream import DiffusionStage
            _t1 = time.perf_counter()
            self._pipeline.stage_1 = DiffusionStage(
                checkpoint_path=self._dev_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization_dev,
                registry=None,
                torch_compile=pipeline_kwargs.get("torch_compile", False),
                offload_mode=offload_mode,
            )
            logger.info(
                "T2V init: DiffusionStage(stage_1) constructor took %.3fs",
                time.perf_counter() - _t1,
            )
            _t2 = time.perf_counter()
            self._pipeline.stage_2 = DiffusionStage(
                checkpoint_path=self._distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization_distilled,
                registry=None,
                torch_compile=pipeline_kwargs.get("torch_compile", False),
                offload_mode=offload_mode,
            )
            logger.info(
                "T2V init: DiffusionStage(stage_2) constructor took %.3fs",
                time.perf_counter() - _t2,
            )
            logger.info(
                "T2V scaled_mm: stage_1 → %s, stage_2 → %s (distilled pre-fused)",
                os.path.basename(self._dev_fp8_path),
                os.path.basename(self._distilled_fp8_path),
            )
            _install_build_transformer_audit(self._pipeline.stage_1, "t2v_stage_1")
            _install_build_transformer_audit(self._pipeline.stage_2, "t2v_stage_2")

        if offload_mode != OffloadMode.NONE:
            _install_stage2_cleanup_hook(self._pipeline)

        # Optional MultiModalGuiderParams (T2V uses CFG + STG).
        from src.upstream import HAS_GUIDERS, MultiModalGuiderParams
        self._MultiModalGuiderParams = MultiModalGuiderParams if HAS_GUIDERS else None

        # TeaCache opt-in.
        from src.teacache import enable_teacache, teacache_config_from_env
        teacache_cfg = teacache_config_from_env()
        self._teacache_enabled = teacache_cfg is not None
        if self._teacache_enabled:
            enable_teacache(self._pipeline, **teacache_cfg)

        self._log_attention_fingerprint()

    def _build_unified(self) -> None:
        """Construct ``ICLoraPipeline`` (unified I2V + V2V).

        Distilled-1.1 BF16 base, IC-LoRA Union-Control fused on stage 1,
        distilled-fp8 stage 2 (scaled_mm path swaps both stages to FP8).
        """
        OffloadMode = self._OffloadMode
        offload_mode = self._offload_mode
        fp8_mode = self._fp8_mode
        fp8_enabled = self._fp8_enabled
        torch_compile_enabled = self._torch_compile_enabled

        for label, path in (
            ("distilled-1.1 BF16", self._distilled_bf16_path),
            ("spatial upsampler", self._spatial_upsampler_path),
            ("IC-LoRA Union-Control", self._ic_lora_path),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Unified pipeline required {label} checkpoint missing at "
                    f"{path}. Run download_models.py before pipeline init."
                )
        if fp8_enabled and not os.path.exists(self._distilled_fp8_path):
            raise RuntimeError(
                f"Unified pipeline FP8 mode requires distilled FP8 DiT at "
                f"{self._distilled_fp8_path}. Run download_models.py with "
                f"LTX_FP8_MODE={fp8_mode}."
            )

        from src.upstream import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
            StateDictRegistry,
        )
        from src.upstream import ICLoraPipeline

        # Quantization policy.
        if fp8_enabled:
            from src.upstream import QuantizationPolicy
            if fp8_mode == "scaled_mm":
                extras = self._extras_for(self._distilled_fp8_path)
                logger.info(
                    "Unified FP8 checkpoint probe: %d non-FP8 weight modules",
                    len(extras),
                )
                if extras:
                    preview = ", ".join(extras[:5])
                    more = f" (+{len(extras) - 5} more)" if len(extras) > 5 else ""
                    logger.info("Unified FP8 probe extras (first 5): %s%s", preview, more)
                quantization = _build_scaled_mm_policy(extras)
                logger.info(
                    "Unified FP8 mode: scaled_mm (W8A8, TRT-LLM cublas_scaled_mm)"
                )
            else:
                quantization = QuantizationPolicy.fp8_cast()
                logger.info("Unified FP8 mode: cast (W8A16, weights FP8 / activations BF16)")
        else:
            quantization = None

        registry = StateDictRegistry()
        logger.info("Unified pipeline using StateDictRegistry (CPU weight caching)")

        stage_1_loras = [
            LoraPathStrengthAndSDOps(
                path=self._ic_lora_path,
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]

        _t0 = time.perf_counter()
        self._pipeline = ICLoraPipeline(
            distilled_checkpoint_path=self._distilled_bf16_path,
            spatial_upsampler_path=self._spatial_upsampler_path,
            gemma_root=self._gemma_root,
            loras=stage_1_loras,
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile_enabled,
            offload_mode=offload_mode,
        )
        logger.info(
            "Unified init: ICLoraPipeline construction took %.2fs "
            "(reference_downscale_factor=%d)",
            time.perf_counter() - _t0,
            self._pipeline.reference_downscale_factor,
        )
        self._unified_meta["reference_downscale_factor"] = (
            self._pipeline.reference_downscale_factor
        )
        self._log_vram("Unified after pipeline init")

        # FP8: rebuild both DiffusionStages onto distilled-fp8.
        if fp8_enabled:
            from src.upstream import DiffusionStage
            _t1 = time.perf_counter()
            self._pipeline.stage_1 = DiffusionStage(
                checkpoint_path=self._distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=tuple(stage_1_loras),
                quantization=quantization,
                registry=None,
                torch_compile=torch_compile_enabled,
                offload_mode=offload_mode,
            )
            self._pipeline.stage_2 = DiffusionStage(
                checkpoint_path=self._distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization,
                registry=None,
                torch_compile=torch_compile_enabled,
                offload_mode=offload_mode,
            )
            logger.info(
                "Unified init: FP8 DiffusionStage rebuild took %.3fs",
                time.perf_counter() - _t1,
            )
            _install_build_transformer_audit(self._pipeline.stage_1, "unified_stage_1")
            _install_build_transformer_audit(self._pipeline.stage_2, "unified_stage_2")

        if offload_mode != OffloadMode.NONE:
            _install_stage2_cleanup_hook(self._pipeline)

        logger.info(
            "Unified pipeline configured: precision=%s mode=%s offload=%s torch_compile=%s",
            "fp8" if fp8_enabled else "bf16",
            fp8_mode or "n/a", offload_mode.value, torch_compile_enabled,
        )

        # TeaCache stays OFF for unified by default — MEMORY.md notes I2V
        # quality regression; ENABLE_TEACACHE=1 honoured for experiments.
        from src.teacache import enable_teacache, teacache_config_from_env
        teacache_cfg = teacache_config_from_env()
        self._teacache_enabled = teacache_cfg is not None
        if self._teacache_enabled:
            enable_teacache(self._pipeline, **teacache_cfg)

        self._log_attention_fingerprint()

    def _log_attention_fingerprint(self) -> None:
        try:
            from src.upstream import AttentionFunction, ltx_attention as _ltx_attn
            _has_fa3_live = _ltx_attn.flash_attn_interface is not None
            _default_resolved = type(AttentionFunction.DEFAULT.to_callable()).__name__
            if self._requested_attn == "flash_attention_3":
                _effective_enum = AttentionFunction.FLASH_ATTENTION_3
            else:
                _effective_enum = AttentionFunction.DEFAULT
            _effective_resolved = type(_effective_enum.to_callable()).__name__
        except Exception as _e:
            _has_fa3_live = None
            _default_resolved = f"unknown ({type(_e).__name__})"
            _effective_resolved = _default_resolved
        logger.info(
            "Attention fingerprint — ltx_core.fa3=%s, requested=%s, "
            "effective enum resolves to: %s (DEFAULT enum would resolve to: %s)",
            _has_fa3_live, self._requested_attn or "default",
            _effective_resolved, _default_resolved,
        )

    def _log_vram(self, label: str) -> None:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1e9
            res = torch.cuda.memory_reserved(0) / 1e9
            logger.info("VRAM %s: %.2f GB allocated, %.2f GB reserved", label, alloc, res)

    # ------------------------------------------------------------------
    # Generate dispatcher
    # ------------------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: int | None = None,
        height: int | None = None,
        num_frames: int = 121,
        num_inference_steps: int = 30,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
        image_url: str | None = None,
        image_b64: str | None = None,
        reference_video_url: str | None = None,
        reference_video_b64: str | None = None,
        reference_video_strength: float = 1.0,
        conditioning_attention_strength: float = 1.0,
        enhance_prompt: bool = False,
    ) -> dict:
        """Run inference, encode MP4, return result dict.

        Mode is decided by inputs:
          - prompt only → T2V (TI2VidTwoStagesPipeline, 30 steps)
          - prompt + image → I2V (ICLoraPipeline, identity-strict)
          - prompt + reference_video (± image) → V2V (ICLoraPipeline, edit)

        T2V scheduler args (``num_inference_steps``, ``cfg_scale``,
        ``stg_scale``, ``rescale_scale``, ``negative_prompt``) are used only
        on the T2V path; the unified path uses ICLoraPipeline's
        SimpleDenoiser (no CFG/STG/negative).

        First request after a mode change pays a ~30-60 s pipeline rebuild;
        same-mode subsequent requests have zero penalty.
        """
        has_image = image_url is not None or image_b64 is not None
        has_ref_video = (
            reference_video_url is not None or reference_video_b64 is not None
        )
        target_mode = _MODE_UNIFIED if (has_image or has_ref_video) else _MODE_T2V
        self._ensure_mode(target_mode)

        if target_mode == _MODE_T2V:
            if width is None:
                width = 1024
            if height is None:
                height = 1536
            return self._t2v_generate(
                prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, num_frames=num_frames,
                num_inference_steps=num_inference_steps, seed=seed,
                frame_rate=frame_rate, cfg_scale=cfg_scale,
                stg_scale=stg_scale, rescale_scale=rescale_scale,
            )
        return self._unified_generate(
            prompt=prompt, width=width, height=height,
            num_frames=num_frames, seed=seed, frame_rate=frame_rate,
            image_url=image_url, image_b64=image_b64,
            reference_video_url=reference_video_url,
            reference_video_b64=reference_video_b64,
            reference_video_strength=reference_video_strength,
            conditioning_attention_strength=conditioning_attention_strength,
            enhance_prompt=enhance_prompt,
            has_image=has_image, has_ref_video=has_ref_video,
        )

    # ------------------------------------------------------------------
    # T2V (TI2VidTwoStagesPipeline) generate body
    # ------------------------------------------------------------------

    def _t2v_generate(
        self, *, prompt: str, negative_prompt: str,
        width: int, height: int, num_frames: int,
        num_inference_steps: int, seed: int,
        frame_rate: float, cfg_scale: float,
        stg_scale: float, rescale_scale: float,
    ) -> dict:
        width, height, num_frames = _round_user_inputs(width, height, num_frames)
        job_id = uuid.uuid4().hex[:12]

        logger.info(
            "Job %s: T2V prompt=%r, %dx%d, %d frames, %d steps, seed=%d",
            job_id, prompt[:80], width, height, num_frames, num_inference_steps, seed,
        )

        try:
            video_guider_params = None
            audio_guider_params = None
            if self._MultiModalGuiderParams is not None:
                video_guider_params = self._MultiModalGuiderParams(
                    cfg_scale=cfg_scale, stg_scale=stg_scale,
                    rescale_scale=rescale_scale, modality_scale=3.0, stg_blocks=[28],
                )
                audio_guider_params = self._MultiModalGuiderParams(
                    cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                    modality_scale=3.0, stg_blocks=[28],
                )

            tiling_config = None
            video_chunks_number = None
            if self._TilingConfig and self._get_video_chunks_number:
                tiling_config = self._TilingConfig.default()
                video_chunks_number = self._get_video_chunks_number(num_frames, tiling_config)

            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            max_batch_size = 1
            logger.info(
                "Job %s: offload_mode=%s, max_batch_size=%d, teacache=%s",
                job_id, self._offload_mode.value, max_batch_size, self._teacache_enabled,
            )

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate, num_inference_steps=num_inference_steps,
                images=[],
                max_batch_size=max_batch_size,
            )
            if video_guider_params is not None:
                call_kwargs["video_guider_params"] = video_guider_params
            if audio_guider_params is not None:
                call_kwargs["audio_guider_params"] = audio_guider_params
            if tiling_config is not None:
                call_kwargs["tiling_config"] = tiling_config

            result = self._pipeline(**call_kwargs)
            video, audio = result if isinstance(result, tuple) else (result, None)
            generation_time = time.time() - start_time
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                logger.info(
                    "Job %s: T2V generation took %.1fs (peak VRAM %.2f GB, audio=%s)",
                    job_id, generation_time, peak, audio is not None,
                )
            else:
                logger.info("Job %s: T2V generation took %.1fs", job_id, generation_time)

            output_filename = f"ltx_{job_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            encode_kwargs = dict(video=video, fps=int(frame_rate), output_path=output_path)
            if audio is not None:
                encode_kwargs["audio"] = audio
            if video_chunks_number is not None:
                encode_kwargs["video_chunks_number"] = video_chunks_number
            encode_start = time.time()
            self._encode_video(**encode_kwargs)
            logger.info(
                "Job %s: mp4 encode %.1fs → %s",
                job_id, time.time() - encode_start, output_path,
            )

            return {
                "output_path": output_path,
                "output_filename": output_filename,
                "generation_time_seconds": round(generation_time, 2),
                "parameters": {
                    "mode": "t2v",
                    "width": width, "height": height, "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps, "seed": seed,
                    "frame_rate": frame_rate, "cfg_scale": cfg_scale,
                    "stg_scale": stg_scale, "rescale_scale": rescale_scale,
                },
            }

        except Exception:
            logger.exception("Job %s failed", job_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise

    # ------------------------------------------------------------------
    # Unified (ICLoraPipeline) generate body
    # ------------------------------------------------------------------

    def _unified_generate(
        self, *, prompt: str,
        width: int | None, height: int | None, num_frames: int,
        seed: int, frame_rate: float,
        image_url: str | None, image_b64: str | None,
        reference_video_url: str | None, reference_video_b64: str | None,
        reference_video_strength: float,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        has_image: bool, has_ref_video: bool,
    ) -> dict:
        job_id = uuid.uuid4().hex[:12]
        image_path: str | None = None
        ref_video_path: str | None = None
        try:
            if has_image:
                from src.image_input import (
                    derive_dims_from_image,
                    materialize_image,
                )
                image_path = materialize_image(image_url, image_b64)
                if width is None and height is None:
                    width, height = derive_dims_from_image(image_path)

            if has_ref_video:
                from src.video_input import materialize_video
                ref_video_path = materialize_video(
                    reference_video_url, reference_video_b64
                )

            if width is None:
                width = 1024
            if height is None:
                height = 1536
            width, height, num_frames = _round_user_inputs(width, height, num_frames)

            mode = "v2v" if has_ref_video else "i2v"

            src_label = (
                "image_url" if image_url is not None
                else "image_b64" if image_b64 is not None
                else "none"
            )
            ref_label = (
                "reference_video_url" if reference_video_url is not None
                else "reference_video_b64" if reference_video_b64 is not None
                else "none"
            )
            logger.info(
                "Job %s: %s (image_src=%s, ref_video_src=%s) "
                "prompt=%r, %dx%d, %d frames, seed=%d, "
                "ref_strength=%.2f, attn_strength=%.2f, enhance_prompt=%s",
                job_id, mode.upper(), src_label, ref_label, prompt[:80],
                width, height, num_frames, seed,
                reference_video_strength, conditioning_attention_strength,
                enhance_prompt,
            )

            tiling_config = None
            video_chunks_number = None
            if self._TilingConfig and self._get_video_chunks_number:
                tiling_config = self._TilingConfig.default()
                video_chunks_number = self._get_video_chunks_number(num_frames, tiling_config)

            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            ref_downscale = self._unified_meta.get("reference_downscale_factor")
            logger.info(
                "Job %s: offload_mode=%s, teacache=%s, ref_downscale=%s",
                job_id, self._offload_mode.value, self._teacache_enabled,
                ref_downscale,
            )

            if has_ref_video:
                result = self._run_v2v(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    image_path=image_path,
                    ref_video_path=ref_video_path,  # type: ignore[arg-type]
                    reference_video_strength=reference_video_strength,
                    conditioning_attention_strength=conditioning_attention_strength,
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                )
            else:
                result = self._run_i2v(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    image_path=image_path,  # type: ignore[arg-type]
                    conditioning_attention_strength=conditioning_attention_strength,
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                )
            video, audio = result if isinstance(result, tuple) else (result, None)
            generation_time = time.time() - start_time

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                logger.info(
                    "Job %s: %s generation took %.1fs (peak VRAM %.2f GB, audio=%s)",
                    job_id, mode.upper(), generation_time, peak, audio is not None,
                )
            else:
                logger.info(
                    "Job %s: %s generation took %.1fs",
                    job_id, mode.upper(), generation_time,
                )

            output_filename = f"ltx_{job_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            encode_kwargs = dict(video=video, fps=int(frame_rate), output_path=output_path)
            if audio is not None:
                encode_kwargs["audio"] = audio
            if video_chunks_number is not None:
                encode_kwargs["video_chunks_number"] = video_chunks_number
            encode_start = time.time()
            self._encode_video(**encode_kwargs)
            logger.info(
                "Job %s: mp4 encode %.1fs → %s",
                job_id, time.time() - encode_start, output_path,
            )

            return {
                "output_path": output_path,
                "output_filename": output_filename,
                "generation_time_seconds": round(generation_time, 2),
                "parameters": {
                    "mode": mode,
                    "width": width, "height": height, "num_frames": num_frames,
                    "seed": seed, "frame_rate": frame_rate,
                    "reference_downscale_factor": ref_downscale,
                    "reference_video_strength": (
                        reference_video_strength if has_ref_video else None
                    ),
                    "conditioning_attention_strength": (
                        conditioning_attention_strength
                        if (has_image or has_ref_video) else None
                    ),
                    "enhance_prompt": enhance_prompt,
                },
            }

        except Exception:
            logger.exception("Job %s failed", job_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            if image_path is not None:
                try:
                    os.unlink(image_path)
                except OSError:
                    logger.debug("I2V tempfile cleanup failed for %s", image_path, exc_info=True)
            if ref_video_path is not None:
                try:
                    os.unlink(ref_video_path)
                except OSError:
                    logger.debug("V2V tempfile cleanup failed for %s", ref_video_path, exc_info=True)

    def _run_i2v(
        self, *,
        prompt: str, seed: int, height: int, width: int,
        num_frames: int, frame_rate: float,
        image_path: str,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        tiling_config,
    ):
        from src.upstream import ImageConditioningInput
        kwargs = dict(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate,
            images=[ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)],
            video_conditioning=[],
            conditioning_attention_strength=conditioning_attention_strength,
            enhance_prompt=enhance_prompt,
        )
        if tiling_config is not None:
            kwargs["tiling_config"] = tiling_config
        return self._pipeline(**kwargs)

    def _run_v2v(
        self, *,
        prompt: str, seed: int, height: int, width: int,
        num_frames: int, frame_rate: float,
        image_path: str | None,
        ref_video_path: str,
        reference_video_strength: float,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        tiling_config,
    ):
        from src.upstream import ImageConditioningInput
        images = (
            [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]
            if image_path is not None else []
        )
        kwargs = dict(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate,
            images=images,
            video_conditioning=[(ref_video_path, reference_video_strength)],
            conditioning_attention_strength=conditioning_attention_strength,
            enhance_prompt=enhance_prompt,
        )
        if tiling_config is not None:
            kwargs["tiling_config"] = tiling_config
        return self._pipeline(**kwargs)
