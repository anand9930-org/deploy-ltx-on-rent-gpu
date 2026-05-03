"""Shared LTX-2.3 pipeline wrapper.

Single ICLoraPipeline serves all three modes (T2V, I2V, V2V); the active
mode is determined by which inputs the caller supplies. IC-LoRA is loaded
unconditionally so I2V identity preservation works the moment an image is
attached, with no per-request reload cost.
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


# ---- FP8 helpers (ported from feature/fp8-h100 reference branch) -----------
# These adapt to Lightricks' specific FP8 packaging: a `model.diffusion_model.`
# comfy prefix on disk, BF16 leftovers in transformer block 1 that upstream's
# baseline `EXCLUDED_LAYER_SUBSTRINGS` doesn't cover, and `_orig_mod.` segments
# injected by `torch.compile` that break upstream's substring matcher. Without
# them the FP8 load fails at first inference with FP8Linear/Linear shape
# mismatches. Source: /tmp/fp8ref/src_pipeline.py.

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
    `.weight` key whose dtype is not `float8_e4m3fn`. Each name is canonicalized
    by stripping `model.diffusion_model.` so it matches the post-rename module
    path the LTX loader produces.

    Why we probe: the Lightricks `*-fp8.safetensors` checkpoints leave a subset
    of transformer block 1 in BF16 (AV cross modules, a few MLPs) that
    upstream's hard-coded `EXCLUDED_LAYER_SUBSTRINGS` does not cover. Without
    these extras, `_apply_fp8_prepare_to_model` swaps those modules to
    FP8Linear and `load_state_dict` fails with size mismatches at first
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
    """Wrapper-only equivalent of `QuantizationPolicy.fp8_scaled_mm()` that
    accepts a per-checkpoint exclusion list AND normalizes names before
    matching.

    The upstream factory has two problems for the Lightricks
    `*-fp8.safetensors` format:

    1. Its `EXCLUDED_LAYER_SUBSTRINGS` doesn't cover the BF16 modules
       Lightricks left in transformer block 1 — append `extras` produced by
       `_probe_fp8_exclusions`.
    2. `_should_skip_layer` upstream is a raw substring check. With regional
       `torch.compile` active, module names pick up an `._orig_mod.` segment
       that breaks substring matching against bare exclusions. We replace
       `_apply_fp8_prepare_to_model` and `_create_transpose_kv_operation`
       with local versions that normalize `_orig_mod` out before the check.
    """
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
    from ltx_core.model.transformer import LTXModel
    from ltx_core.quantization import QuantizationPolicy
    from ltx_core.quantization.fp8_scaled_mm import (
        FP8_PREPARE_MODULE_OPS,
        FP8_TRANSPOSE_SD_OPS,
        FP8Linear,
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
      2. GPU capability auto-detect when LTX_FP8_MODE is empty: H100/H200
         (SM 9.0) → ``scaled_mm``, otherwise ``cast``. (Requires the env to
         be set to a recognized non-empty value, otherwise we return None
         to keep BF16 default behaviour.)
      3. None when LTX_FP8_MODE is unset/empty.
    """
    override = os.getenv("LTX_FP8_MODE", "").strip().lower()
    if override in ("scaled_mm", "cast"):
        return override
    return None


def _install_build_transformer_audit(stage, label: str) -> None:
    """Wrap ``DiffusionStage._build_transformer`` with timing + dtype audit.

    Logs wall time for the transformer build (load → SDOps transpose →
    `_prepare` swap) and a post-load dtype histogram over
    `transformer_blocks.*` parameters. The histogram is the GROUND TRUTH for
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


class LTXVideoGenerator:
    """Initialises a single ICLoraPipeline serving T2V / I2V / V2V."""

    def __init__(self, model_dir: str = "/models") -> None:
        # Lightricks-official distilled-1.1 BF16 checkpoint. Holds DiT, VAE,
        # audio decoder, vocoder, image encoder, and embeddings processor in
        # one safetensors — required by every non-DiT block via
        # *_COMFY_KEYS_FILTER. IC-LoRA Union-Control was trained against
        # this exact distilled baseline, so applying it at runtime as the
        # single LoRA reproduces the trained configuration.
        from src.download_models import (
            DISTILLED_CHECKPOINT_FILENAME,
            DISTILLED_FP8_FILENAME,
            IC_LORA_FILENAME,
        )
        checkpoint_path = os.path.join(model_dir, DISTILLED_CHECKPOINT_FILENAME)
        distilled_fp8_path = os.path.join(model_dir, DISTILLED_FP8_FILENAME)
        spatial_upsampler_path = os.path.join(
            model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        ic_lora_path = os.path.join(model_dir, IC_LORA_FILENAME)
        gemma_root = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")

        # FP8 path is opt-in via LTX_FP8_MODE; default stays BF16 so production
        # behaviour doesn't change silently. Decide here so the existence check
        # below can demand the FP8 file too when requested.
        fp8_mode = _select_fp8_mode()
        fp8_enabled = fp8_mode is not None

        required = [
            ("distilled-1.1 BF16", checkpoint_path),
            ("spatial upsampler", spatial_upsampler_path),
            ("IC-LoRA Union-Control", ic_lora_path),
        ]
        if fp8_enabled:
            required.append(("distilled FP8 DiT", distilled_fp8_path))
        for label, path in required:
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Required {label} checkpoint missing at {path}. "
                    "Run download_models.py before pipeline init."
                )

        logger.info("Initializing LTX-2.3 ICLoraPipeline ...")
        self._log_vram("before pipeline init")

        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
            StateDictRegistry,
        )
        from ltx_pipelines.ic_lora import ICLoraPipeline
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.types import OffloadMode

        self._encode_video = encode_video
        self._OffloadMode = OffloadMode

        # Offload mode follows precision: FP8 fits two ~30 GB DiTs on H100
        # 80 GB without offload, BF16 doesn't (two ~44 GB DiTs overflow). The
        # NONE setting also unlocks `torch.compile` (upstream's
        # DiffusionStage rejects compile when offload_mode != NONE).
        gpu_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0
        )
        offload_mode = OffloadMode.NONE if fp8_enabled else OffloadMode.CPU
        self._offload_mode = offload_mode
        logger.info(
            "Offload mode: %s (GPU=%.0f GB; precision=%s)",
            offload_mode.value, gpu_vram_gb, "fp8" if fp8_enabled else "bf16",
        )

        # Quantization policy. scaled_mm = H100 W8A8 through TRT-LLM
        # cublas_scaled_mm; cast = W8A16 runtime downcast/upcast (any FP8 GPU).
        if fp8_enabled:
            from ltx_core.quantization import QuantizationPolicy
            if fp8_mode == "scaled_mm":
                extras = _probe_fp8_exclusions([distilled_fp8_path])
                logger.info(
                    "FP8 checkpoint probe: %d non-FP8 weight modules", len(extras),
                )
                if extras:
                    preview = ", ".join(extras[:5])
                    more = f" (+{len(extras) - 5} more)" if len(extras) > 5 else ""
                    logger.info("FP8 probe extras (first 5): %s%s", preview, more)
                quantization = _build_scaled_mm_policy(extras)
                logger.info(
                    "FP8 mode: scaled_mm (W8A8, TRT-LLM cublas_scaled_mm, H100-optimised)"
                )
            else:
                quantization = QuantizationPolicy.fp8_cast()
                logger.info("FP8 mode: cast (W8A16, weights FP8 / activations BF16)")
        else:
            quantization = None

        # CPU weight cache — the distilled-1.1 BF16 file is read by stage_1,
        # by upstream's stage_2, AND by PromptEncoder/ImageConditioner/
        # VideoDecoder/VideoUpsampler/AudioDecoder (each consumes its own
        # *_COMFY_KEYS_FILTER subset). Caching parsed bytes once on CPU
        # avoids re-reading the 46 GB file multiple times during init.
        registry = StateDictRegistry()
        logger.info("Using StateDictRegistry (CPU weight caching)")

        # Compile is allowed only when offload_mode == NONE (upstream guard
        # in DiffusionStage). FP8 path satisfies that; BF16 with CPU offload
        # does not — kept the warn so a future re-enable surfaces.
        torch_compile_enabled = os.getenv(
            "ENABLE_TORCH_COMPILE", "1"
        ).strip().lower() not in ("0", "false", "no", "off", "")
        if torch_compile_enabled and offload_mode != OffloadMode.NONE:
            logger.warning(
                "torch.compile requested but offload_mode=%s disallows it "
                "(upstream DiffusionStage guard). Running uncompiled. "
                "Set LTX_FP8_MODE=scaled_mm to unblock compile on H100.",
                offload_mode.value,
            )
            torch_compile_enabled = False
        elif torch_compile_enabled:
            logger.info("torch.compile ENABLED (regional per transformer block)")

        # FA3 — patch the configurator BEFORE the pipeline builds any
        # transformer (lazy on first __call__).
        _requested_attn = os.environ.get("LTX_ATTENTION_TYPE", "").lower()
        if _requested_attn == "flash_attention_3":
            from src.attention_override import enable_flash_attention_3
            enable_flash_attention_3()

        # Stage 1 receives the IC-LoRA Union-Control as the single LoRA
        # fusion. The base checkpoint is already distilled-1.1, so no
        # dev → distilled converter LoRA is required; this matches the
        # exact configuration IC-LoRA Union-Control was trained against.
        # Upstream's stage_2 ships with loras=() — correct, since the base
        # is already distilled and IC-LoRA cross-attention deltas are not
        # used in stage_2 (which uses combined_image_conditionings).
        stage_1_loras = [
            LoraPathStrengthAndSDOps(
                path=ic_lora_path,
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]

        _t0 = time.perf_counter()
        self._pipeline = ICLoraPipeline(
            distilled_checkpoint_path=checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=stage_1_loras,
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile_enabled,
            offload_mode=offload_mode,
        )
        logger.info(
            "Init timing: ICLoraPipeline construction took %.2fs "
            "(reference_downscale_factor=%d)",
            time.perf_counter() - _t0,
            self._pipeline.reference_downscale_factor,
        )
        self._reference_downscale_factor = self._pipeline.reference_downscale_factor
        self._log_vram("after pipeline init")

        # FP8 path: rebuild both DiffusionStages to point at the FP8 DiT file.
        # ICLoraPipeline ships every block with the BF16 distilled-1.1
        # checkpoint, which is required for the non-DiT blocks (PromptEncoder
        # / ImageConditioner / VideoUpsampler / VideoDecoder / AudioDecoder
        # — each pulls its slice via *_COMFY_KEYS_FILTER). The FP8 DiT file is
        # DiT-only, so we keep the BF16 file at construction time and only
        # swap the two DiffusionStages here. DiffusionStage defers weight
        # load to its first __call__, so this swap is cheap.
        if fp8_enabled:
            from ltx_pipelines.utils.blocks import DiffusionStage
            _t1 = time.perf_counter()
            self._pipeline.stage_1 = DiffusionStage(
                checkpoint_path=distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=tuple(stage_1_loras),
                quantization=quantization,
                registry=None,
                torch_compile=torch_compile_enabled,
                offload_mode=offload_mode,
            )
            self._pipeline.stage_2 = DiffusionStage(
                checkpoint_path=distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization,
                registry=None,
                torch_compile=torch_compile_enabled,
                offload_mode=offload_mode,
            )
            logger.info(
                "Init timing: FP8 DiffusionStage rebuild took %.3fs "
                "(weight load deferred to first __call__)",
                time.perf_counter() - _t1,
            )
            _install_build_transformer_audit(self._pipeline.stage_1, "stage_1")
            _install_build_transformer_audit(self._pipeline.stage_2, "stage_2")

        # Stage 1 → Stage 2 boundary cleanup is only meaningful when the
        # streaming arena pins host pages — i.e. CPU offload. On the FP8
        # NONE-offload path the streaming arena is inactive and the hook
        # is a no-op, so skip the install.
        if offload_mode != OffloadMode.NONE:
            _t2 = time.perf_counter()
            _install_stage2_cleanup_hook(self._pipeline)
            logger.info(
                "Init timing: _install_stage2_cleanup_hook took %.3fs",
                time.perf_counter() - _t2,
            )

        logger.info(
            "Pipeline configured: precision=%s mode=%s offload=%s torch_compile=%s",
            "fp8" if fp8_enabled else "bf16",
            fp8_mode or "n/a", offload_mode.value, torch_compile_enabled,
        )

        # TeaCache — opt-in via ENABLE_TEACACHE=1. Default OFF:
        # MEMORY.md notes TeaCache hurts I2V quality; with IC-LoRA always
        # loaded and the reference-token cross-attention path active on
        # any I2V/V2V request, skipping forwards is even more harmful.
        from src.teacache import enable_teacache, teacache_config_from_env
        teacache_cfg = teacache_config_from_env()
        self._teacache_enabled = teacache_cfg is not None
        if self._teacache_enabled:
            _t3 = time.perf_counter()
            enable_teacache(self._pipeline, **teacache_cfg)
            logger.info(
                "Init timing: enable_teacache took %.3fs",
                time.perf_counter() - _t3,
            )

        # Boot-time attention-backend fingerprint.
        try:
            from ltx_core.model.transformer import attention as _ltx_attn
            from ltx_core.model.transformer.attention import AttentionFunction
            _has_fa3_live = _ltx_attn.flash_attn_interface is not None
            _default_resolved = type(AttentionFunction.DEFAULT.to_callable()).__name__
            if _requested_attn == "flash_attention_3":
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
            _has_fa3_live, _requested_attn or "default",
            _effective_resolved, _default_resolved,
        )
        if _requested_attn == "flash_attention_3":
            if _has_fa3_live is False:
                logger.warning(
                    "FA3 requested but ltx_core.attention.flash_attn_interface is None "
                    "— first attention call will fail."
                )
            elif _effective_resolved != "FlashAttention3":
                logger.warning(
                    "FA3 requested and flash_attn_interface is live, but "
                    "AttentionFunction.FLASH_ATTENTION_3.to_callable() returned %s "
                    "instead of FlashAttention3 — investigate.",
                    _effective_resolved,
                )

        logger.info(
            "torch debug env — TORCH_LOGS=%r, TORCHDYNAMO_VERBOSE=%r",
            os.environ.get("TORCH_LOGS"),
            os.environ.get("TORCHDYNAMO_VERBOSE"),
        )

        # Tiling helpers (unchanged from the previous pipeline).
        self._TilingConfig = None
        self._get_video_chunks_number = None
        try:
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
            self._TilingConfig = TilingConfig
            self._get_video_chunks_number = get_video_chunks_number
        except ImportError:
            pass

        logger.info("Pipeline ready.")

    def _log_vram(self, label: str) -> None:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1e9
            res = torch.cuda.memory_reserved(0) / 1e9
            logger.info("VRAM %s: %.2f GB allocated, %.2f GB reserved", label, alloc, res)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: int | None = None,
        height: int | None = None,
        num_frames: int = 121,
        seed: int = 42,
        frame_rate: float = 24.0,
        image_url: str | None = None,
        image_b64: str | None = None,
        reference_video_url: str | None = None,
        reference_video_b64: str | None = None,
        reference_video_strength: float = 1.0,
        conditioning_attention_strength: float = 1.0,
        enhance_prompt: bool = False,
    ) -> dict:
        """Run inference and encode MP4. Routes to T2V / I2V / V2V by input.

        - prompt only → T2V
        - prompt + image → I2V (identity-strict via IC-LoRA reference token)
        - prompt + reference_video (± image) → V2V (style transfer / edit)

        ``negative_prompt`` is accepted for wire compatibility but the
        upstream ICLoraPipeline uses ``SimpleDenoiser`` (no CFG/STG), so
        it is a no-op.
        """
        del negative_prompt  # SimpleDenoiser has no negative-prompt path

        job_id = uuid.uuid4().hex[:12]
        has_image = image_url is not None or image_b64 is not None
        has_ref_video = (
            reference_video_url is not None or reference_video_b64 is not None
        )

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

            if has_ref_video:
                mode = "V2V"
            elif has_image:
                mode = "I2V"
            else:
                mode = "T2V"

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
                job_id, mode, src_label, ref_label, prompt[:80],
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

            logger.info(
                "Job %s: offload_mode=%s, teacache=%s, ref_downscale=%d",
                job_id, self._offload_mode.value,
                getattr(self, "_teacache_enabled", False),
                self._reference_downscale_factor,
            )

            if has_ref_video:
                result = self._run_v2v(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    image_path=image_path, ref_video_path=ref_video_path,  # type: ignore[arg-type]
                    reference_video_strength=reference_video_strength,
                    conditioning_attention_strength=conditioning_attention_strength,
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                )
            elif has_image:
                result = self._run_i2v(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    image_path=image_path,  # type: ignore[arg-type]
                    conditioning_attention_strength=conditioning_attention_strength,
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                )
            else:
                result = self._run_t2v(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                )
            video, audio = result if isinstance(result, tuple) else (result, None)
            generation_time = time.time() - start_time

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                logger.info(
                    "Job %s: generation took %.1fs (peak VRAM %.2f GB, audio=%s)",
                    job_id, generation_time, peak, audio is not None,
                )
            else:
                logger.info("Job %s: generation took %.1fs", job_id, generation_time)

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
                    "mode": mode.lower(),
                    "width": width, "height": height, "num_frames": num_frames,
                    "seed": seed, "frame_rate": frame_rate,
                    "reference_downscale_factor": self._reference_downscale_factor,
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

    def _run_t2v(
        self,
        prompt: str, seed: int, height: int, width: int,
        num_frames: int, frame_rate: float,
        enhance_prompt: bool,
        tiling_config,
    ):
        kwargs = dict(
            prompt=prompt, seed=seed, height=height, width=width,
            num_frames=num_frames, frame_rate=frame_rate,
            images=[], video_conditioning=[],
            enhance_prompt=enhance_prompt,
        )
        if tiling_config is not None:
            kwargs["tiling_config"] = tiling_config
        return self._pipeline(**kwargs)

    def _run_i2v(
        self,
        prompt: str, seed: int, height: int, width: int,
        num_frames: int, frame_rate: float,
        image_path: str,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        tiling_config,
    ):
        from ltx_pipelines.utils.args import ImageConditioningInput
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
        self,
        prompt: str, seed: int, height: int, width: int,
        num_frames: int, frame_rate: float,
        image_path: str | None,
        ref_video_path: str,
        reference_video_strength: float,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        tiling_config,
    ):
        from ltx_pipelines.utils.args import ImageConditioningInput
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
