"""LTX-2.3 Unified Pipeline wrapper — shared core.

Holds the module-level FP8/expandable_segments handoff, the shared FP8
helpers (probe, scaled_mm policy, build-transformer audit), the Stage 1→2
cleanup hook, and the ``LTXVideoGenerator`` class skeleton (``__init__``,
``_ensure_mode``, ``_extras_for``, ``_log_attention_fingerprint``,
``_log_vram``, ``generate`` dispatcher). Per-scenario method bodies live in
``t2v.py``, ``i2v.py``, ``v2v.py`` and are mixed in via multiple inheritance.
"""

import functools
import gc
import logging
import os
import re
import time

import torch

from src.config import get_settings
from src.pipeline.i2v import I2VMixin
from src.pipeline.t2v import T2VMixin
from src.pipeline.triple_stages import TripleStagesMixin
from src.pipeline.triple_stages_comfyui import TripleStagesComfyUIMixin
from src.pipeline.v2v import V2VMixin

logger = logging.getLogger(__name__)


# FP8 + expandable_segments — required by the upstream LTX FP8 README:
# the scaled_mm path's CUDA graph capture is incompatible with the default
# allocator's segment splitting. Set before any tensor is allocated, but
# only when the FP8 path is requested so BF16 deployments stay unchanged.
# Reads via Settings (not os.environ directly) so that load_dotenv()
# called in the entrypoint before this import wins for local-dev .env files.
if get_settings().ltx_fp8_mode.strip().lower() in ("scaled_mm", "cast"):
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
    """Round to pipeline grid (W/H divisible by 64, frames = 8k+1) and log
    when we change anything — silent clamping surprises callers."""
    w, h, f = _round_to(width, 64), _round_to(height, 64), _round_frames(num_frames)
    if (w, h, f) != (width, height, num_frames):
        logger.info(
            "Input rounded to pipeline grid: %dx%d×%d → %dx%d×%d",
            width, height, num_frames, w, h, f,
        )
    return w, h, f


def _install_stage2_cleanup_hook(pipeline) -> None:
    """Flush device + host allocators at the Stage 1 → Stage 2 boundary.

    Upstream removed the host-side ``empty_cache`` from offload teardown, so
    pinned pages aren't returned between stages and Stage 2's
    ``pin_memory()`` raises ``CUDA error: invalid argument`` on VRAM-tight
    pods. Force a sync cleanup (GC → empty_cache → sync → host empty_cache)
    before Stage 2 enters. Idempotent.
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
    """Return module paths that must stay nn.Linear (BF16).

    Lightricks ``*-fp8.safetensors`` leave a subset of transformer block 1 in
    BF16 (AV cross modules, a few MLPs) that upstream's hard-coded
    ``EXCLUDED_LAYER_SUBSTRINGS`` doesn't cover. Reads each safetensors header
    (no tensor data) and returns every non-FP8 ``.weight`` key, canonicalized
    by stripping the ``model.diffusion_model.`` prefix.
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
    """``QuantizationPolicy.fp8_scaled_mm`` equivalent with two fixes for
    Lightricks' ``*-fp8.safetensors``: (1) appends per-checkpoint ``extras``
    that upstream's ``EXCLUDED_LAYER_SUBSTRINGS`` misses; (2) normalises
    ``._orig_mod.`` (injected by regional ``torch.compile``) out of module
    names so exact-match exclusions work post-compile.
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
    """Return ``"scaled_mm"`` / ``"cast"`` from ``LTX_FP8_MODE``, or ``None``
    for BF16 default."""
    override = get_settings().ltx_fp8_mode.strip().lower()
    if override in ("scaled_mm", "cast"):
        return override
    return None


def _install_build_transformer_audit(stage, label: str) -> None:
    """Wrap ``_build_transformer`` with timing + post-load dtype histogram.
    The histogram is GROUND TRUTH for whether the FP8 swap actually took —
    the per-block prepare log only reflects matcher intent. Idempotent.
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
_MODE_TRIPLE_STAGES = "triple_stages"  # vendored TI2VidTripleStagesPipeline (T2V + I2V)
_MODE_TRIPLE_STAGES_COMFYUI = "triple_stages_comfyui"  # vendored ComfyUI workflow port

# Pipeline variant — wire-level opt-in. Default routes via input-driven mode
# detection (T2V/Unified). "triple_stages" forces the vendored class;
# "triple_stages_comfyui" forces the ComfyUI workflow port.
_VARIANT_DEFAULT = "default"
_VARIANT_TRIPLE_STAGES = "triple_stages"
_VARIANT_TRIPLE_STAGES_COMFYUI = "triple_stages_comfyui"


class LTXVideoGenerator(
    T2VMixin, I2VMixin, V2VMixin, TripleStagesMixin, TripleStagesComfyUIMixin,
):
    """Wrapper around T2V (``TI2VidTwoStagesPipeline``), unified I2V/V2V
    (``ICLoraPipeline``), and the vendored triple-stages pipeline. Only one
    upstream pipeline is resident at a time; cross-mode requests tear down +
    rebuild.

    Per-scenario builders + generate bodies are mixed in: ``T2VMixin`` owns
    ``_build_t2v`` / ``_t2v_generate``; ``I2VMixin`` owns the unified
    lifecycle (``_build_unified``, ``_unified_generate``) plus ``_run_i2v``;
    ``V2VMixin`` owns ``_run_v2v``; ``TripleStagesMixin`` owns
    ``_build_triple_stages`` / ``_triple_stages_generate``. V2V piggybacks on
    the unified pipeline that I2V already builds — the only V2V-specific code
    is the per-call pipeline kwargs in ``_run_v2v``.
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
        _settings = get_settings()
        torch_compile_requested = _settings.enable_torch_compile
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
                logger.info("torch.compile disabled via ENABLE_TORCH_COMPILE=0")

        # FA3 attention patch — runs ONCE for the process lifetime so it's
        # active for whichever pipeline we build first AND any rebuild.
        self._requested_attn = _settings.ltx_attention_type.lower()
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

        # Vendored triple-stages skew filter — wraps PromptEncoder and
        # DiffusionStage __call__ to swallow newer-API kwargs (e.g.
        # streaming_prefetch_count) the eisneim fork passes that our pinned
        # upstream signature doesn't accept. No-op on T2V/I2V/V2V (those
        # paths never pass the offending kwargs); load-bearing for
        # triple-stages. Idempotent + sig-aware.
        from src.prompt_encoder_override import enable_triple_stages_kwarg_filter
        enable_triple_stages_kwarg_filter()

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
        # triple_stages / triple_stages_comfyui are accepted for pods that
        # serve those endpoints exclusively — avoids the ~30-60s rebuild on
        # the first request.
        default_mode_env = _settings.ltx_default_mode.strip().lower()
        if default_mode_env == "t2v":
            initial_mode = _MODE_T2V
        elif default_mode_env in ("i2v", "v2v", "unified", ""):
            initial_mode = _MODE_UNIFIED
        elif default_mode_env == "triple_stages":
            initial_mode = _MODE_TRIPLE_STAGES
        elif default_mode_env == "triple_stages_comfyui":
            initial_mode = _MODE_TRIPLE_STAGES_COMFYUI
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
            # If the triple-stages-comfyui-graph pipeline was active, free
            # ComfyUI's resident model weights so the ltx_pipelines path gets the
            # GPU back. No-op if ComfyUI was never bootstrapped (legacy/other mode).
            from src import comfyui_runtime
            comfyui_runtime.unload_models()
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
        elif mode == _MODE_TRIPLE_STAGES:
            self._build_triple_stages()
        elif mode == _MODE_TRIPLE_STAGES_COMFYUI:
            self._build_triple_stages_comfyui()
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
        pipeline_variant: str = _VARIANT_DEFAULT,
        stage1_steps: int = 16,
        stage2_steps: int = 8,
        image_strength: float = 1.0,
        image_frame_idx: int = 0,
    ) -> dict:
        """Run inference, encode MP4, return result dict.

        Default variant: mode is decided by inputs (prompt-only → T2V;
        +image → I2V; +reference_video → V2V). T2V scheduler args
        (``cfg_scale``, ``stg_scale``, ``rescale_scale``,
        ``negative_prompt``, ``num_inference_steps``) are ignored on the
        unified path (SimpleDenoiser, no CFG/STG).

        ``pipeline_variant="triple_stages"``: route to the vendored
        ``TI2VidTripleStagesPipeline`` (T2V if no image, I2V if image
        supplied — ``reference_video_*`` is rejected, the variant doesn't
        support video conditioning). Uses ``stage1_steps`` /
        ``stage2_steps`` instead of ``num_inference_steps``;
        ``image_strength`` and ``image_frame_idx`` control I2V conditioning.

        ``pipeline_variant="triple_stages_comfyui"``: route to the vendored
        ``TI2VidTripleStagesComfyUIPipeline`` (workflow port of
        ``scripts/workflow_3mljpp.py``). Sigmas + cfg are baked in, so
        ``cfg_scale``/``stg_scale``/``rescale_scale``/``stage1_steps``/
        ``stage2_steps``/``num_inference_steps``/``image_strength`` are
        ignored. Resolution must be /128.

        Cross-mode requests pay a ~30-60 s rebuild.
        """
        has_image = image_url is not None or image_b64 is not None
        has_ref_video = (
            reference_video_url is not None or reference_video_b64 is not None
        )

        if pipeline_variant == _VARIANT_TRIPLE_STAGES:
            if has_ref_video:
                raise ValueError(
                    "pipeline_variant='triple_stages' does not support "
                    "reference_video_* inputs (the vendored class has no "
                    "video conditioning path)."
                )
            self._ensure_mode(_MODE_TRIPLE_STAGES)
            return self._triple_stages_generate(
                prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, num_frames=num_frames,
                seed=seed, frame_rate=frame_rate,
                cfg_scale=cfg_scale, stg_scale=stg_scale,
                rescale_scale=rescale_scale,
                image_url=image_url, image_b64=image_b64,
                image_strength=image_strength,
                image_frame_idx=image_frame_idx,
                stage1_steps=stage1_steps, stage2_steps=stage2_steps,
                enhance_prompt=enhance_prompt,
            )
        if pipeline_variant == _VARIANT_TRIPLE_STAGES_COMFYUI:
            if has_ref_video:
                raise ValueError(
                    "pipeline_variant='triple_stages_comfyui' does not support "
                    "reference_video_* inputs (the vendored class has no "
                    "video conditioning path)."
                )
            self._ensure_mode(_MODE_TRIPLE_STAGES_COMFYUI)
            return self._triple_stages_comfyui_generate(
                prompt=prompt, negative_prompt=negative_prompt,
                width=width, height=height, num_frames=num_frames,
                seed=seed, frame_rate=frame_rate,
                image_url=image_url, image_b64=image_b64,
                image_frame_idx=image_frame_idx,
                enhance_prompt=enhance_prompt,
            )
        if pipeline_variant != _VARIANT_DEFAULT:
            raise ValueError(
                f"Unknown pipeline_variant={pipeline_variant!r}. Valid: "
                f"{_VARIANT_DEFAULT!r}, {_VARIANT_TRIPLE_STAGES!r}, "
                f"{_VARIANT_TRIPLE_STAGES_COMFYUI!r}."
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
