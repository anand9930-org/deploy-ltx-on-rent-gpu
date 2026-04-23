"""Shared LTX-2.3 pipeline wrapper.

All VRAM optimisation techniques live here.
"""

import functools
import gc
import logging
import os
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


def _probe_fp8_exclusions(paths: list[str]) -> tuple[str, ...]:
    """Return the set of module names that must stay BF16.

    Reads the safetensors headers of each ``paths`` entry (no tensor data
    loaded) and collects every ``.weight`` key whose dtype is not
    ``float8_e4m3fn``. Returns the module path with ``.weight`` stripped
    (e.g. ``transformer_blocks.1.audio_attn1.to_q``) so it flows straight
    into ``_should_skip_layer``'s substring check — which compares
    against module names from ``named_modules()`` that also have no
    ``.weight`` suffix.

    Motivation: the Lightricks FP8 DiT keeps some modules in block 1 as
    BF16 (audio-video cross modules and a few MLPs) that upstream's
    ``EXCLUDED_LAYER_SUBSTRINGS`` does not cover. Without this probe
    ``_apply_fp8_prepare_to_model`` swaps those to ``FP8Linear`` and
    ``load_state_dict`` fails with size/dtype mismatches at first
    inference.
    """
    from safetensors import safe_open  # lazy — downloads pull safetensors

    names: set[str] = set()
    for path in paths:
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if not key.endswith(".weight"):
                    continue
                slice_ = f.get_slice(key)
                if slice_.get_dtype() != "F8_E4M3":
                    names.add(key[: -len(".weight")])
    return tuple(sorted(names))


def _build_scaled_mm_policy(extras: tuple[str, ...]):
    """Wrapper-only equivalent of ``QuantizationPolicy.fp8_scaled_mm()``
    that accepts a per-call exclusion list.

    We can't extend the upstream factory because the Dockerfile clones
    ``Lightricks/LTX-2`` fresh at build time (see Dockerfile line 73) —
    any edit to the vendored ``LTX-2-ref/`` tree is git-ignored and
    never reaches the pod. So we rebuild the policy here using the
    same upstream primitives the baseline factory uses; the only
    difference is that ``EXCLUDED_LAYER_SUBSTRINGS`` is augmented with
    ``extras`` so modules the Lightricks FP8 checkpoint left in BF16
    aren't swapped to ``FP8Linear``.

    When ``extras`` is empty we return a policy byte-equivalent to the
    upstream factory output (same ``sd_ops`` + ``module_ops`` objects),
    so this helper is a drop-in replacement.
    """
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.loader.sd_ops import SDOps
    from ltx_core.model.transformer import LTXModel
    from ltx_core.quantization import QuantizationPolicy
    from ltx_core.quantization.fp8_scaled_mm import (
        EXCLUDED_LAYER_SUBSTRINGS,
        FP8_PREPARE_MODULE_OPS,
        FP8_TRANSPOSE_SD_OPS,
        _apply_fp8_prepare_to_model,
        _create_transpose_kv_operation,
    )

    try:
        import tensorrt_llm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "tensorrt_llm not installed — fp8-trtllm extra missing"
        ) from e

    if not extras:
        return QuantizationPolicy(
            sd_ops=FP8_TRANSPOSE_SD_OPS,
            module_ops=(FP8_PREPARE_MODULE_OPS,),
        )

    merged = EXCLUDED_LAYER_SUBSTRINGS + tuple(extras)
    sd_ops = SDOps("fp8_transpose_weights").with_kv_operation(
        _create_transpose_kv_operation(merged),
        key_prefix="transformer_blocks.",
        key_suffix=".weight",
    )
    module_ops = ModuleOps(
        name="fp8_prepare_for_loading",
        matcher=lambda m: isinstance(m, LTXModel),
        mutator=lambda m: _apply_fp8_prepare_to_model(m, merged),
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

        self._encode_video = encode_video

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

            # Streaming: builds models on CPU, streams layers to GPU on demand.
            # Required for <48GB GPUs — without it, LoRA fusion OOMs because
            # the 22B transformer + LoRA deltas exceed GPU memory.
            # With streaming, build+fuse happens on CPU (plenty of RAM),
            # then only 2-3 layers live on GPU at any time during inference.
            # max_batch_size=4 batches guidance passes to reduce PCIe round-trips.
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
            streaming = 2 if gpu_vram_gb < 40 else None
            max_batch_size = 4 if streaming else 1

            logger.info(
                "Job %s: GPU=%.0fGB, streaming=%s, max_batch_size=%d, teacache=%s",
                job_id, gpu_vram_gb, streaming, max_batch_size,
                getattr(self, "_teacache_enabled", False),
            )

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate, num_inference_steps=num_inference_steps,
                images=[],
                streaming_prefetch_count=streaming,
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
