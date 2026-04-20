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


class LTXVideoGenerator:
    """Initialises the LTX-2.3 two-stage pipeline and runs inference."""

    def __init__(self, model_dir: str = "/models") -> None:
        checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-dev.safetensors")
        spatial_upsampler_path = os.path.join(
            model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        distilled_lora_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        gemma_root = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")

        logger.info("Initializing LTX-2.3 pipeline ...")
        self._log_vram("before pipeline init")

        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.media_io import encode_video

        self._encode_video = encode_video

        # FP8 quantization — downcasts BF16 weights to FP8 on the fly,
        # upcasts back to BF16 during forward. ~40% VRAM reduction.
        try:
            from ltx_core.quantization import QuantizationPolicy
            quantization = QuantizationPolicy.fp8_cast()
            logger.info("Using FP8 quantization (fp8_cast)")
        except ImportError:
            quantization = None
            logger.warning("QuantizationPolicy not available")

        # CPU weight caching — only one model on GPU at a time
        registry = None
        try:
            from ltx_core.loader import StateDictRegistry
            registry = StateDictRegistry()
            logger.info("Using StateDictRegistry (CPU weight caching)")
        except ImportError:
            logger.warning("StateDictRegistry not available")

        # Distilled LoRA
        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        # Build pipeline — pass registry and quantization at construction time
        # so VRAM is managed correctly from the start
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
