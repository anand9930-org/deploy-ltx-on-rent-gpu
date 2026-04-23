"""Shared LTX-2.3 pipeline wrapper.

All VRAM optimisation techniques live here.
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

    LTX-2's layer-streaming path pins the full transformer's weights to
    host memory at each stage entry. Between Stage 1 teardown and
    Stage 2 setup LTX calls ``torch._C._host_emptyCache()`` best-effort,
    but on 24 GB GPUs that call intermittently fails to release enough
    pinned pages and the first ``tensor.data.pin_memory()`` call inside
    Stage 2's ``_LayerStore.__init__`` raises

        torch.AcceleratorError: CUDA error: invalid argument

    We force a synchronous cleanup cycle (Python GC → device
    empty_cache → CUDA sync → host empty_cache) right before Stage 2's
    transformer context manager enters, so the pinned arena is in a
    known-drained state. Idempotent; safe to call multiple times.
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
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch._C, "_host_emptyCache"):
                try:
                    torch._C._host_emptyCache()
                except Exception:  # pragma: no cover — best effort
                    logger.debug("Stage 2 cleanup: _host_emptyCache raised", exc_info=True)
            alloc_after = torch.cuda.memory_allocated(0) / 1e9
            logger.info(
                "Stage 2 boundary cleanup done: VRAM %.2f → %.2f GB allocated",
                alloc_before, alloc_after,
            )
        else:
            logger.info("Stage 2 boundary cleanup done (no CUDA)")
        return original_ctx(*args, **kwargs)

    stage._transformer_ctx = hooked_ctx
    stage._stage2_cleanup_hooked = True


# The Lightricks FP8 safetensors store DiT weights with this prefix on
# disk; `LTXV_MODEL_COMFY_RENAMING_MAP` strips it when the loader reads
# them. Our probe reads RAW keys so we strip it too — otherwise the
# exclusion names we emit won't match the in-memory module paths.
_COMFY_PREFIX = "model.diffusion_model."

# torch.compile wraps each transformer block in an OptimizedModule whose
# inner attributes are reached through `_orig_mod`. After compile,
# `model.named_modules()` yields paths like
# `transformer_blocks.1._orig_mod.attn1.to_gate_logits`. We strip the
# injected segment before the substring check so exclusions can use
# the plain (pre-compile) form regardless of whether compile ran.
_ORIG_MOD_SEGMENT_RE = re.compile(r"\._orig_mod(?=\.|$)")


def _strip_comfy_prefix(name: str) -> str:
    return name[len(_COMFY_PREFIX):] if name.startswith(_COMFY_PREFIX) else name


def _normalize_for_match(name: str) -> str:
    """Canonicalize a module/key path for exclusion matching.

    Removes the `_orig_mod.` segment injected by torch.compile AND the
    `model.diffusion_model.` prefix that may persist on raw safetensors
    keys. The result is the canonical in-memory module path used by
    `EXCLUDED_LAYER_SUBSTRINGS` and by our probe output.
    """
    return _ORIG_MOD_SEGMENT_RE.sub("", _strip_comfy_prefix(name))


def _probe_fp8_exclusions(paths: list[str]) -> tuple[str, ...]:
    """Return module paths that must stay as nn.Linear (BF16).

    Opens each `.safetensors` header (no tensor data loaded) and
    collects every `.weight` key whose dtype is not `float8_e4m3fn`.
    Each name is canonicalized by stripping `model.diffusion_model.`
    so it matches the post-rename module path the LTX loader produces.

    Why we probe: the Lightricks `*-fp8.safetensors` checkpoints leave
    a subset of transformer block 1 in BF16 (AV cross modules, a few
    MLPs) that upstream's hard-coded `EXCLUDED_LAYER_SUBSTRINGS` does
    not cover. Without these extras, `_apply_fp8_prepare_to_model`
    swaps those modules to FP8Linear and `load_state_dict` fails with
    size mismatches at first inference.

    Logs a dtype histogram per file, a sample of transformer-block
    extras, and a warning if zero transformer-block BF16 entries were
    found (which would mean the crash that motivated this probe isn't
    actually being prevented).
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
            logger.info("FP8 probe: %s — raw BF16 samples: %s",
                        os.path.basename(path), bf16_samples)

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
    """Wrapper-only equivalent of `QuantizationPolicy.fp8_scaled_mm()`
    that accepts a per-call exclusion list AND normalizes names before
    substring matching.

    The upstream factory has two problems for the Lightricks
    `*-fp8.safetensors` format:

    1. Its `EXCLUDED_LAYER_SUBSTRINGS` doesn't cover the BF16 modules
       Lightricks left in transformer block 1. So we append `extras`
       produced by `_probe_fp8_exclusions`.

    2. `_should_skip_layer` in upstream is a raw substring check. With
       regional `torch.compile` active, module names pick up a
       `._orig_mod.` segment that breaks substring matching against
       bare exclusions. We replace `_apply_fp8_prepare_to_model` and
       `_create_transpose_kv_operation` with local versions that
       normalize `_orig_mod` out before the check.

    We also can't edit upstream directly: the Dockerfile (line 73)
    clones `Lightricks/LTX-2` fresh at build, so any change in
    `LTX-2-ref/` is local-only and never reaches the pod.
    """
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
    from ltx_core.model.transformer import LTXModel
    from ltx_core.quantization import QuantizationPolicy
    from ltx_core.quantization.fp8_scaled_mm import (
        EXCLUDED_LAYER_SUBSTRINGS,
        FP8_PREPARE_MODULE_OPS,
        FP8Linear,
        FP8_TRANSPOSE_SD_OPS,
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
        # normalization, because upstream's baseline exclusions like
        # `"transformer_blocks.0."` happen to tolerate compile (they
        # stop at the dot, so substring matches regardless). Keeping
        # the upstream primitives here is safe; no crash has been
        # observed without extras.
        return QuantizationPolicy(
            sd_ops=FP8_TRANSPOSE_SD_OPS,
            module_ops=(FP8_PREPARE_MODULE_OPS,),
        )

    merged = EXCLUDED_LAYER_SUBSTRINGS + tuple(extras)

    def _should_skip(name: str) -> bool:
        canon = _normalize_for_match(name)
        return any(sub in canon for sub in merged)

    import torch
    from torch import nn

    def _prepare(model):
        """Drop-in replacement for `_apply_fp8_prepare_to_model`.

        Walks named_modules; for each nn.Linear (not already FP8Linear)
        whose canonicalized name does NOT match an exclusion, swaps it
        to FP8Linear. Logs the first swap/skip decisions under block 1
        so we can verify the matcher at boot.
        """
        replacements: list[tuple[nn.Module, str, nn.Linear]] = []
        swap_samples: list[str] = []
        skip_samples: list[str] = []
        block1_linear_total = 0

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
                continue
            if name.startswith("transformer_blocks.1."):
                block1_linear_total += 1
            if _should_skip(name):
                if len(skip_samples) < 5 and name.startswith("transformer_blocks.1."):
                    skip_samples.append(name)
                continue
            if len(swap_samples) < 5 and name.startswith("transformer_blocks.1."):
                swap_samples.append(name)
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            replacements.append((parent, attr_name, module))

        for parent, attr_name, linear in replacements:
            setattr(parent, attr_name, _linear_to_fp8linear(linear))

        logger.info(
            "FP8 prepare: block-1 nn.Linear count=%d, swapped=%d, skipped=%d",
            block1_linear_total,
            len(swap_samples),
            len(skip_samples),
        )
        if swap_samples:
            logger.info("FP8 prepare: block-1 swap samples: %s", swap_samples)
        if skip_samples:
            logger.info("FP8 prepare: block-1 skip samples: %s", skip_samples)
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


def _select_fp8_mode() -> str:
    """Pick the FP8 compute path.

    ``scaled_mm`` is the H100-optimised W8A8 path (real FP8 GEMM via
    TRT-LLM ``cublas_scaled_mm``), ``cast`` is the Ada/Blackwell W8A16
    path (weights FP8, activations upcast to BF16 per forward).

    Order of precedence:
      1. ``LTX_FP8_MODE=scaled_mm|cast`` env override (matches
         download_models.py so downloads and runtime agree).
      2. GPU capability auto-detect: H100/H200 (SM 9.0) → ``scaled_mm``,
         otherwise → ``cast``.
      3. ``cast`` fallback when CUDA is unavailable.
    """
    override = os.getenv("LTX_FP8_MODE", "").strip().lower()
    if override in ("scaled_mm", "cast"):
        return override
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap == (9, 0):
            return "scaled_mm"
    return "cast"


class LTXVideoGenerator:
    """Initialises the LTX-2.3 two-stage pipeline and runs inference."""

    def __init__(self, model_dir: str = "/models") -> None:
        # BF16 DiT checkpoint. Also holds VAE / audio decoder / vocoder /
        # image encoder / embeddings processor weights — loaded by the
        # non-DiT blocks through *_COMFY_KEYS_FILTER, so this file is
        # required on every path (including scaled_mm).
        checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-dev.safetensors")
        spatial_upsampler_path = os.path.join(
            model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        distilled_lora_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        dev_fp8_path = os.path.join(model_dir, "ltx-2.3-22b-dev-fp8.safetensors")
        distilled_fp8_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-fp8.safetensors"
        )
        gemma_root = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")

        logger.info("Initializing LTX-2.3 pipeline ...")
        self._log_vram("before pipeline init")

        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.types import OffloadMode

        self._encode_video = encode_video

        # Weight placement. Upstream PR #201 (2026-04-23) replaced the
        # per-call `streaming_prefetch_count` kwarg with a constructor
        # `offload_mode` enum on `TI2VidTwoStagesPipeline` and
        # `DiffusionStage`. OffloadMode.NONE keeps all weights on GPU
        # (fastest); CPU streams layers from pinned host RAM for <40 GB
        # pods; DISK re-reads from disk each pass for tiny cards.
        gpu_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0
        )
        offload_mode = OffloadMode.NONE if gpu_vram_gb >= 40 else OffloadMode.CPU
        self._offload_mode = offload_mode
        logger.info(
            "Offload mode: %s (GPU=%.0f GB)", offload_mode.value, gpu_vram_gb,
        )

        # Pick FP8 path. scaled_mm = H100 W8A8 through TRT-LLM
        # cublas_scaled_mm; cast = W8A16 runtime downcast/upcast.
        fp8_mode = _select_fp8_mode()
        self._fp8_mode = fp8_mode

        from ltx_core.quantization import QuantizationPolicy
        if fp8_mode == "scaled_mm":
            if not (os.path.exists(dev_fp8_path) and os.path.exists(distilled_fp8_path)):
                raise RuntimeError(
                    f"LTX_FP8_MODE=scaled_mm requires pre-quantized FP8 checkpoints at "
                    f"{dev_fp8_path} and {distilled_fp8_path}. Run download_models.py "
                    f"with LTX_FP8_MODE=scaled_mm."
                )
            # Checkpoint-driven exclusion list. The Lightricks FP8 DiT
            # keeps a subset of block-1's modules (audio-video cross
            # modules, some MLPs) in BF16 that upstream's
            # EXCLUDED_LAYER_SUBSTRINGS does not cover. Without this
            # probe, _apply_fp8_prepare_to_model swaps those to
            # FP8Linear and load_state_dict fails with size/dtype
            # mismatches at first inference. See docs/FP8_H100_Spec.md
            # Verification #7.
            extras = _probe_fp8_exclusions([dev_fp8_path, distilled_fp8_path])
            logger.info(
                "FP8 checkpoint probe: %d non-FP8 weight modules added to exclusion list",
                len(extras),
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

        # CPU weight caching — only one model on GPU at a time
        registry = None
        try:
            from ltx_core.loader import StateDictRegistry
            registry = StateDictRegistry()
            logger.info("Using StateDictRegistry (CPU weight caching)")
        except ImportError:
            logger.warning("StateDictRegistry not available")

        # Distilled LoRA — only meaningful on the cast path. scaled_mm
        # cannot fuse a BF16 LoRA into pre-quantized FP8 weights at load
        # time; we instead swap Stage 2 to the pre-fused distilled-fp8
        # checkpoint further down.
        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        if fp8_mode == "cast":
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    path=distilled_lora_path,
                    strength=0.8,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]
        else:
            distilled_lora = []

        # Build pipeline. On scaled_mm we pass the BF16 checkpoint so
        # PromptEncoder / ImageConditioner / VideoDecoder / AudioDecoder /
        # VideoUpsampler see their VAE + encoder keys; Stage 1 and Stage 2
        # DiffusionStages are rebuilt below to point at the FP8 DiT files.
        pipeline_kwargs = dict(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            offload_mode=offload_mode,
        )
        if quantization is not None:
            pipeline_kwargs["quantization"] = quantization
        if registry is not None:
            pipeline_kwargs["registry"] = registry

        # torch.compile — LTX-2's regional compile (per transformer block,
        # not whole model) via COMPILE_TRANSFORMER SDOps in
        # ltx_core/model/transformer/compiling.py. Each transformer block
        # gets wrapped with torch.compile(m); small blocks compile fast
        # and are cached, so the effective cost is paid once per pod boot.
        # Expected ~15-30% Stage 1 speedup on Ada + Hopper; transparent to
        # numerics. Default ON; set ENABLE_TORCH_COMPILE=0 to disable.
        torch_compile_enabled = os.getenv("ENABLE_TORCH_COMPILE", "1").strip().lower() not in (
            "0", "false", "no", "off", "",
        )
        if torch_compile_enabled:
            pipeline_kwargs["torch_compile"] = True
            logger.info("torch.compile ENABLED (regional per transformer block)")
        else:
            logger.info("torch.compile disabled via ENABLE_TORCH_COMPILE=%s", os.getenv("ENABLE_TORCH_COMPILE"))

        # Optional FA3 enablement. When `LTX_ATTENTION_TYPE=flash_attention_3`
        # is set, patch LTX's model configurator so every Attention module
        # constructed by the pipeline routes through the FA3 wrapper. Must
        # run before the TI2VidTwoStagesPipeline's Builder constructs any
        # transformer, which happens lazily on the first __call__.
        _requested_attn = os.environ.get("LTX_ATTENTION_TYPE", "").lower()
        if _requested_attn == "flash_attention_3":
            from src.attention_override import enable_flash_attention_3
            enable_flash_attention_3()

        self._pipeline = TI2VidTwoStagesPipeline(**pipeline_kwargs)
        self._log_vram("after pipeline init")

        # scaled_mm: swap Stage 1 and Stage 2 DiffusionStages to point at
        # the pre-quantized FP8 DiT checkpoints. The upstream
        # TI2VidTwoStagesPipeline exposes a single checkpoint_path that it
        # feeds to every block, which is fine for the cast path but wrong
        # here — we need BF16 for the VAE/encoder blocks and FP8 for the
        # two DiT stages. Rebuilding after construction is safe because
        # DiffusionStage defers weight load to its first __call__.
        if fp8_mode == "scaled_mm":
            from ltx_pipelines.utils.blocks import DiffusionStage
            self._pipeline.stage_1 = DiffusionStage(
                checkpoint_path=dev_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization,
                registry=registry,
                torch_compile=pipeline_kwargs.get("torch_compile", False),
                offload_mode=offload_mode,
            )
            # Stage 2 uses the distilled-fp8 checkpoint (distilled weights
            # already fused in) instead of base + distilled LoRA.
            self._pipeline.stage_2 = DiffusionStage(
                checkpoint_path=distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization,
                registry=registry,
                torch_compile=pipeline_kwargs.get("torch_compile", False),
                offload_mode=offload_mode,
            )
            logger.info(
                "scaled_mm: rebuilt stage_1 → %s, stage_2 → %s (distilled pre-fused)",
                os.path.basename(dev_fp8_path),
                os.path.basename(distilled_fp8_path),
            )

        # Always-on: aggressive allocator flush at the Stage 1 → Stage 2
        # boundary. Fixes intermittent
        #   torch.AcceleratorError: CUDA error: invalid argument
        # coming out of layer_streaming.py:63 on 24 GB cards. See
        # _install_stage2_cleanup_hook above.
        _install_stage2_cleanup_hook(self._pipeline)

        # TeaCache — opt-in via ENABLE_TEACACHE=1. Skips the transformer
        # forward on diffusion steps where the input hasn't changed
        # enough (rescaled relative-L1 below TEACACHE_THRESHOLD). Works
        # across any GPU + attention backend — it never touches the
        # attention kernel, just caches the module's output. Validated
        # on LTX-Video with ~1.6–2.1× lossless speedup.
        from src.teacache import enable_teacache, teacache_config_from_env
        teacache_cfg = teacache_config_from_env()
        self._teacache_enabled = teacache_cfg is not None
        if self._teacache_enabled:
            enable_teacache(self._pipeline, **teacache_cfg)

        # Boot-time attention-backend fingerprint. LTX-2's
        # `AttentionFunction.DEFAULT` resolves to `PytorchAttention` (torch
        # SDPA dispatcher) in this image. We probe ltx_core's actual
        # module-level binding of `flash_attn_interface` (not our env) so
        # the log reflects what the pipeline will use.
        try:
            from ltx_core.model.transformer import attention as _ltx_attn
            from ltx_core.model.transformer.attention import AttentionFunction
            _has_fa3_live = _ltx_attn.flash_attn_interface is not None
            _attn_resolved = type(AttentionFunction.DEFAULT.to_callable()).__name__
        except Exception as _e:
            _has_fa3_live = None
            _attn_resolved = f"unknown ({type(_e).__name__})"
        logger.info(
            "Attention fingerprint — ltx_core.fa3=%s, "
            "LTX default resolves to: %s, requested: %s",
            _has_fa3_live, _attn_resolved, _requested_attn or "default",
        )
        if _requested_attn == "flash_attention_3" and _has_fa3_live is False:
            logger.warning(
                "FA3 requested but ltx_core.attention.flash_attn_interface is None "
                "— first attention call will fail."
            )

        # Optional components (guiders, tiling)
        self._MultiModalGuiderParams = None
        try:
            from ltx_core.components.guiders import MultiModalGuiderParams
            self._MultiModalGuiderParams = MultiModalGuiderParams
        except ImportError:
            pass

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
        width: int = 1024,
        height: int = 1536,
        num_frames: int = 121,
        num_inference_steps: int = 30,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
    ) -> dict:
        """Run inference and encode MP4. Returns dict with output_path, output_filename, etc."""
        width, height, num_frames = _round_user_inputs(width, height, num_frames)
        job_id = uuid.uuid4().hex[:12]

        logger.info(
            "Job %s: prompt=%r, %dx%d, %d frames, %d steps, seed=%d",
            job_id, prompt[:80], width, height, num_frames, num_inference_steps, seed,
        )

        try:
            # Guidance params
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

            # Tiling config
            tiling_config = None
            video_chunks_number = None
            if self._TilingConfig and self._get_video_chunks_number:
                tiling_config = self._TilingConfig.default()
                video_chunks_number = self._get_video_chunks_number(num_frames, tiling_config)

            # Run pipeline
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            # Weight placement was decided at __init__ time via
            # `offload_mode` (PR #201 upstream refactor — `streaming_prefetch_count`
            # no longer exists on `__call__`). `max_batch_size=1` is the
            # upstream default and is correct on big GPUs; raising it
            # above 1 has previously caused issues on ≥40 GB pods.
            max_batch_size = 1
            logger.info(
                "Job %s: offload_mode=%s, max_batch_size=%d, teacache=%s",
                job_id, self._offload_mode.value, max_batch_size,
                getattr(self, "_teacache_enabled", False),
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
                    "Job %s: generation took %.1fs (peak VRAM %.2f GB, audio=%s)",
                    job_id, generation_time, peak, audio is not None,
                )
            else:
                logger.info("Job %s: generation took %.1fs", job_id, generation_time)

            # Encode video
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
                    "width": width, "height": height, "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps, "seed": seed,
                    "frame_rate": frame_rate, "cfg_scale": cfg_scale,
                    "stg_scale": stg_scale, "rescale_scale": rescale_scale,
                },
            }

        except Exception:
            logger.exception("Job %s failed", job_id)
            torch.cuda.empty_cache()
            raise
