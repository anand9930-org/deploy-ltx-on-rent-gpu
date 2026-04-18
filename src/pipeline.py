"""Shared LTX-2.3 pipeline wrapper.

All VRAM optimisation techniques live here.
"""

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

        self._pipeline = TI2VidTwoStagesPipeline(**pipeline_kwargs)
        self._log_vram("after pipeline init")

        # TeaCache — opt-in via ENABLE_TEACACHE=1. Skips the transformer
        # forward on diffusion steps where the input hasn't changed
        # enough (rescaled relative-L1 below TEACACHE_THRESHOLD). Works
        # across any GPU + attention backend — it never touches the
        # attention kernel, just caches the module's output. Validated
        # on LTX-Video with ~1.6–2.1× lossless speedup.
        from src.teacache import enable_teacache, teacache_config_from_env
        teacache_cfg = teacache_config_from_env()
        if teacache_cfg is not None:
            enable_teacache(self._pipeline, **teacache_cfg)

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
        width = _round_to(width, 64)
        height = _round_to(height, 64)
        num_frames = _round_frames(num_frames)
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

            # Streaming: builds models on CPU, streams layers to GPU on demand.
            # Required for <48GB GPUs — without it, LoRA fusion OOMs because
            # the 22B transformer + LoRA deltas exceed GPU memory.
            # With streaming, build+fuse happens on CPU (plenty of RAM),
            # then only 2-3 layers live on GPU at any time during inference.
            # max_batch_size=4 batches guidance passes to reduce PCIe round-trips.
            gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
            streaming = 2 if gpu_vram_gb < 40 else None
            if streaming:
                logger.info("Job %s: streaming enabled (GPU VRAM: %.0f GB < 40 GB)", job_id, gpu_vram_gb)

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate, num_inference_steps=num_inference_steps,
                images=[],
                streaming_prefetch_count=streaming,
                max_batch_size=4 if streaming else 1,
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
            logger.info("Job %s: generation took %.1fs", job_id, generation_time)

            # Encode video
            output_filename = f"ltx_{job_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            encode_kwargs = dict(video=video, fps=int(frame_rate), output_path=output_path)
            if audio is not None:
                encode_kwargs["audio"] = audio
            if video_chunks_number is not None:
                encode_kwargs["video_chunks_number"] = video_chunks_number
            self._encode_video(**encode_kwargs)

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
