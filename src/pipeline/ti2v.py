"""TI2V scenario: full-DiT TI2VidTwoStagesPipeline with IC-LoRA on stage 1.

Why this exists separately from ``T2VMixin``: the upstream
``TI2VidTwoStagesPipeline`` drifts off the input image after the first
frame on prompt+image inputs (CFG/STG amplify the prompt; the clean frame-0
token loses its anchoring grip after a few denoise steps). Loading the
distilled IC-LoRA-Union-Control on stage 1 — empirically observed in the
``ICLoraPipeline`` path — restores attention to the conditioning frame.
We mirror that pattern on the full DiT here:

* stage 1 = full DiT + IC-LoRA Union-Control (anchors frame 0)
* stage 2 = LoRA-free in scaled_mm (uses pre-fused distilled-fp8); distilled-LoRA-only
  in cast/bf16 (the upstream production-quality recipe, minus IC-LoRA spillover)

The IC-LoRA was trained against the full LTX-2.3-22b base, so loading it
on stage 1 is base-correct (not a cross-base hack). LoRA fusion is pure
key-match → matrix add (``ltx_core/loader/fuse_loras.py``); no
architecture/version checks fight us.

Mode opt-in: ``LTX_DEFAULT_MODE=ti2v``. Builder + per-call methods are
mixed into ``LTXVideoGenerator`` — do not instantiate ``Ti2vMixin``
directly.
"""

import logging
import os
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)


class Ti2vMixin:
    """Methods used when ``LTX_DEFAULT_MODE=ti2v``. Builds
    ``TI2VidTwoStagesPipeline`` and attaches IC-LoRA Union-Control to
    stage 1 only via post-construction stage rebuild."""

    def _build_ti2v(self) -> None:
        """Build ``TI2VidTwoStagesPipeline``: full-DiT both stages, IC-LoRA
        on stage 1, stage 2 LoRA-free (scaled_mm) or distilled-LoRA-only
        (cast/bf16). Mirrors the manual stage-rebuild pattern from
        ``_build_t2v`` and ``_build_unified``."""
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
            ("IC-LoRA Union-Control", self._ic_lora_path),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"TI2V required {label} checkpoint missing at {path}. "
                    "Run download_models.py before pipeline init."
                )
        if fp8_mode == "scaled_mm":
            for label, path in (
                ("dev FP8 DiT", self._dev_fp8_path),
                ("distilled FP8 DiT", self._distilled_fp8_path),
            ):
                if not os.path.exists(path):
                    raise RuntimeError(
                        f"TI2V scaled_mm requires {label} at {path}. "
                        "Run download_models.py with LTX_FP8_MODE=scaled_mm."
                    )

        from src.upstream import QuantizationPolicy
        if fp8_mode == "scaled_mm":
            extras_dev = self._extras_for(self._dev_fp8_path)
            extras_distilled = self._extras_for(self._distilled_fp8_path)
            logger.info(
                "TI2V FP8 checkpoint probe: dev=%d, distilled=%d non-FP8 weight modules",
                len(extras_dev), len(extras_distilled),
            )
            quantization_dev = _build_scaled_mm_policy(extras_dev)
            quantization_distilled = _build_scaled_mm_policy(extras_distilled)
            quantization = quantization_dev  # placeholder for ctor; rebuilt per-stage below
            logger.info(
                "TI2V FP8 mode: scaled_mm (W8A8, TRT-LLM cublas_scaled_mm, H100-optimised)"
            )
        elif fp8_mode == "cast":
            quantization = QuantizationPolicy.fp8_cast()
            quantization_dev = None
            quantization_distilled = None
            logger.info("TI2V FP8 mode: cast (W8A16, weights FP8 / activations BF16)")
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
            logger.info("TI2V using StateDictRegistry (CPU weight caching)")
        except Exception:
            logger.warning("StateDictRegistry not available", exc_info=True)

        # IC-LoRA Union-Control — applied to stage 1 only (the anchor that fixes
        # T2V's frame-0 drift). Manual rebuild below keeps it off stage 2.
        ic_lora_stage_1 = LoraPathStrengthAndSDOps(
            path=self._ic_lora_path,
            strength=1.0,
            sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
        )

        # Distilled LoRA — cast/bf16 stage 2 only. scaled_mm uses pre-fused distilled-fp8.
        if fp8_mode != "scaled_mm":
            if not os.path.exists(self._distilled_lora_path):
                raise RuntimeError(
                    f"TI2V (cast/bf16) requires distilled LoRA at "
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
            "TI2V init: TI2VidTwoStagesPipeline construction took %.2fs",
            time.perf_counter() - _t0,
        )
        self._log_vram("TI2V after pipeline init")

        # Manual stage rebuild — IC-LoRA on stage 1, stage 2 LoRA-free (scaled_mm)
        # or distilled-LoRA-only (cast/bf16, left alone).
        from src.upstream import DiffusionStage
        if fp8_mode == "scaled_mm":
            _t1 = time.perf_counter()
            self._pipeline.stage_1 = DiffusionStage(
                checkpoint_path=self._dev_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(ic_lora_stage_1,),
                quantization=quantization_dev,
                registry=None,
                torch_compile=pipeline_kwargs.get("torch_compile", False),
                offload_mode=offload_mode,
            )
            logger.info(
                "TI2V init: DiffusionStage(stage_1, dev_fp8 + IC-LoRA) took %.3fs",
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
                "TI2V init: DiffusionStage(stage_2, distilled_fp8 LoRA-free) took %.3fs",
                time.perf_counter() - _t2,
            )
            logger.info(
                "TI2V scaled_mm: stage_1 → %s + IC-LoRA, stage_2 → %s (distilled pre-fused, LoRA-free)",
                os.path.basename(self._dev_fp8_path),
                os.path.basename(self._distilled_fp8_path),
            )
            _install_build_transformer_audit(self._pipeline.stage_1, "ti2v_stage_1")
            _install_build_transformer_audit(self._pipeline.stage_2, "ti2v_stage_2")
        else:
            # cast/bf16: rebuild stage_1 to attach IC-LoRA. Stage 2 keeps the
            # distilled LoRA from the pipeline ctor (TI2VidTwoStagesPipeline
            # builds stage_2 with loras=(*loras, *distilled_lora) — since we
            # passed loras=[], it ended up with just the distilled LoRA, which
            # is exactly what we want).
            _t1 = time.perf_counter()
            self._pipeline.stage_1 = DiffusionStage(
                checkpoint_path=self._dev_bf16_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(ic_lora_stage_1,),
                quantization=quantization,
                registry=None,
                torch_compile=pipeline_kwargs.get("torch_compile", False),
                offload_mode=offload_mode,
            )
            logger.info(
                "TI2V init: DiffusionStage(stage_1, dev_bf16 + IC-LoRA) took %.3fs",
                time.perf_counter() - _t1,
            )
            logger.info(
                "TI2V %s: stage_1 → dev_bf16 + IC-LoRA, stage_2 → dev_bf16 + distilled-LoRA (pipeline default)",
                fp8_mode or "bf16",
            )

        if offload_mode != OffloadMode.NONE:
            _install_stage2_cleanup_hook(self._pipeline)

        # MultiModalGuiderParams (TI2V uses CFG + STG, same as T2V).
        from src.upstream import HAS_GUIDERS, MultiModalGuiderParams
        self._MultiModalGuiderParams = MultiModalGuiderParams if HAS_GUIDERS else None

        # TeaCache stays OFF for TI2V (mirrors I2V — frame-anchored conditioning
        # conflicts with TeaCache's residual reuse; see MEMORY.md teacache_i2v_quality).
        self._teacache_enabled = False

        self._log_attention_fingerprint()

    def _ti2v_generate(
        self, *, prompt: str, negative_prompt: str,
        width: int | None, height: int | None, num_frames: int,
        num_inference_steps: int, seed: int,
        frame_rate: float, cfg_scale: float,
        stg_scale: float, rescale_scale: float,
        image_url: str | None, image_b64: str | None,
        enhance_prompt: bool,
    ) -> dict:
        from src.pipeline.core import _round_user_inputs

        job_id = uuid.uuid4().hex[:12]
        image_path: str | None = None
        has_image = image_url is not None or image_b64 is not None

        try:
            if has_image:
                from src.pipeline.inputs import (
                    derive_dims_from_image,
                    materialize_image,
                )
                image_path = materialize_image(image_url, image_b64)
                if width is None and height is None:
                    width, height = derive_dims_from_image(image_path)

            if width is None:
                width = 1024
            if height is None:
                height = 1536
            width, height, num_frames = _round_user_inputs(width, height, num_frames)

            src_label = (
                "image_url" if image_url is not None
                else "image_b64" if image_b64 is not None
                else "none"
            )
            logger.info(
                "Job %s: TI2V (image_src=%s) prompt=%r, %dx%d, %d frames, %d steps, "
                "seed=%d, enhance_prompt=%s",
                job_id, src_label, prompt[:80], width, height, num_frames,
                num_inference_steps, seed, enhance_prompt,
            )

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

            from src.upstream import ImageConditioningInput
            images_arg = (
                [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]
                if has_image else []
            )

            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            max_batch_size = 1
            logger.info(
                "Job %s: offload_mode=%s, max_batch_size=%d, teacache=%s, ic_lora=stage_1",
                job_id, self._offload_mode.value, max_batch_size, self._teacache_enabled,
            )

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate, num_inference_steps=num_inference_steps,
                images=images_arg,
                enhance_prompt=enhance_prompt,
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
                    "Job %s: TI2V generation took %.1fs (peak VRAM %.2f GB, audio=%s)",
                    job_id, generation_time, peak, audio is not None,
                )
            else:
                logger.info("Job %s: TI2V generation took %.1fs", job_id, generation_time)

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
                    "mode": "ti2v",
                    "image_src": src_label,
                    "width": width, "height": height, "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps, "seed": seed,
                    "frame_rate": frame_rate, "cfg_scale": cfg_scale,
                    "stg_scale": stg_scale, "rescale_scale": rescale_scale,
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
                    logger.debug("TI2V tempfile cleanup failed for %s", image_path, exc_info=True)
