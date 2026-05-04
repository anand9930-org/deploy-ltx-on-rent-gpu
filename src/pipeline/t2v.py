"""T2V scenario: build + run for ``TI2VidTwoStagesPipeline``.

Mixin holds ``_build_t2v`` (boot/rebuild) and ``_t2v_generate`` (per-call
denoise → encode). Methods reach into the shared instance state set up by
``LTXVideoGenerator.__init__`` (``_dev_bf16_path``, ``_offload_mode``,
``_TilingConfig``, …).
"""

import logging
import os
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)


class T2VMixin:
    """Methods used when ``LTX_DEFAULT_MODE=t2v`` or the request is
    prompt-only. Mixed into ``LTXVideoGenerator`` via multiple inheritance —
    do not instantiate directly."""

    def _build_t2v(self) -> None:
        """Build ``TI2VidTwoStagesPipeline``: stage 1 dev-fp8 (scaled_mm) or
        dev BF16 + distilled-LoRA (cast/bf16); stage 2 distilled-fp8 or
        dev BF16 + distilled-LoRA."""
        from src.pipeline.core import (
            _build_scaled_mm_policy,
            _install_build_transformer_audit,
            _install_stage2_cleanup_hook,
        )

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

    def _t2v_generate(
        self, *, prompt: str, negative_prompt: str,
        width: int, height: int, num_frames: int,
        num_inference_steps: int, seed: int,
        frame_rate: float, cfg_scale: float,
        stg_scale: float, rescale_scale: float,
    ) -> dict:
        from src.pipeline.core import _round_user_inputs

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
