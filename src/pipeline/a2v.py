"""A2V scenario: build + run for vanilla ``A2VidPipelineTwoStage``.

Frozen external audio conditioning (consistent voice via TTS, natural
lipsync) + frame-0 image pin via ``VideoConditionByLatentIndex``. Identity
propagates to subsequent frames through transformer self-attention from
the pinned frame-0 latent — same mechanism as I2V.

We previously layered IC-LoRA's ``VideoConditionByReferenceLatent`` on
top of A2V to strengthen identity. An empirical sweep on
``ref_strength × attn_strength ∈ {(0.5,0.5),(0.3,0.3)}`` confirmed the
combination kills audio-driven motion at every tested setting (face
frozen after the first second, PSNR > 42 dB between consecutive seconds);
LTX-2.3 was not trained on dual conditioning and the reference tokens
dominate self-attention regardless of how aggressively we tune the dials.
For stronger identity we will layer a face-restoration post-process on
top of vanilla A2V instead.

Mixin holds ``_build_a2v`` (boot/rebuild) and ``_a2v_generate`` (per-call
materialise → denoise → encode). Methods reach into the shared instance
state set up by ``LTXVideoGenerator.__init__``.
"""

import logging
import os
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)


class A2VMixin:
    """Methods used when the request supplies audio input. Mixed into
    ``LTXVideoGenerator`` via multiple inheritance."""

    def _build_a2v(self) -> None:
        """Build vanilla ``A2VidPipelineTwoStage``: dev checkpoint (audio
        conditioning calibrated) + distilled LoRA on stage 2. FP8
        quantisation follows the T2V/I2V pattern."""
        from src.pipeline.core import (
            _build_scaled_mm_policy,
            _install_build_transformer_audit,
            _install_stage2_cleanup_hook,
        )

        OffloadMode = self._OffloadMode
        offload_mode = self._offload_mode
        fp8_mode = self._fp8_mode
        torch_compile_enabled = self._torch_compile_enabled

        from src.upstream import A2VidPipelineTwoStage

        for label, path in (
            ("dev BF16", self._dev_bf16_path),
            ("spatial upsampler", self._spatial_upsampler_path),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"A2V required {label} checkpoint missing at {path}. "
                    "Run download_models.py before pipeline init."
                )
        if fp8_mode == "scaled_mm":
            for label, path in (
                ("dev FP8 DiT", self._dev_fp8_path),
                ("distilled FP8 DiT", self._distilled_fp8_path),
            ):
                if not os.path.exists(path):
                    raise RuntimeError(
                        f"A2V scaled_mm requires {label} at {path}. "
                        "Run download_models.py with LTX_FP8_MODE=scaled_mm."
                    )

        from src.upstream import QuantizationPolicy
        if fp8_mode == "scaled_mm":
            extras_dev = self._extras_for(self._dev_fp8_path)
            extras_distilled = self._extras_for(self._distilled_fp8_path)
            logger.info(
                "A2V FP8 checkpoint probe: dev=%d, distilled=%d non-FP8 weight modules",
                len(extras_dev), len(extras_distilled),
            )
            quantization_dev = _build_scaled_mm_policy(extras_dev)
            quantization_distilled = _build_scaled_mm_policy(extras_distilled)
            quantization = quantization_dev
            logger.info("A2V FP8 mode: scaled_mm (W8A8, H100-optimised)")
        elif fp8_mode == "cast":
            quantization = QuantizationPolicy.fp8_cast()
            quantization_dev = None
            quantization_distilled = None
            logger.info("A2V FP8 mode: cast (W8A16)")
        else:
            quantization = None
            quantization_dev = None
            quantization_distilled = None

        if offload_mode != OffloadMode.NONE and quantization is not None:
            logger.warning(
                "A2V offload mode %s requires non-quantized BF16 — dropping FP8 + compile.",
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
            logger.info("A2V using StateDictRegistry (CPU weight caching)")
        except Exception:
            logger.warning("StateDictRegistry not available", exc_info=True)

        if fp8_mode != "scaled_mm":
            if not os.path.exists(self._distilled_lora_path):
                raise RuntimeError(
                    f"A2V (cast/bf16) requires distilled LoRA at "
                    f"{self._distilled_lora_path}. Run download_models.py."
                )
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    path=self._distilled_lora_path,
                    strength=0.8,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                ),
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
        self._pipeline = A2VidPipelineTwoStage(**pipeline_kwargs)
        logger.info(
            "A2V init: A2VidPipelineTwoStage construction took %.2fs",
            time.perf_counter() - _t0,
        )
        self._log_vram("A2V after pipeline init")

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
                "A2V init: DiffusionStage(stage_1) constructor took %.3fs",
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
                "A2V init: DiffusionStage(stage_2) constructor took %.3fs",
                time.perf_counter() - _t2,
            )
            logger.info(
                "A2V scaled_mm: stage_1 → %s, stage_2 → %s",
                os.path.basename(self._dev_fp8_path),
                os.path.basename(self._distilled_fp8_path),
            )
            _install_build_transformer_audit(self._pipeline.stage_1, "a2v_stage_1")
            _install_build_transformer_audit(self._pipeline.stage_2, "a2v_stage_2")

        if offload_mode != OffloadMode.NONE:
            _install_stage2_cleanup_hook(self._pipeline)

        from src.upstream import HAS_GUIDERS, MultiModalGuiderParams
        self._MultiModalGuiderParams = MultiModalGuiderParams if HAS_GUIDERS else None

        self._log_attention_fingerprint()

    # ------------------------------------------------------------------
    # Per-call run
    # ------------------------------------------------------------------

    def _run_a2v(
        self, *,
        prompt: str, negative_prompt: str,
        seed: int, height: int, width: int,
        num_frames: int, frame_rate: float,
        num_inference_steps: int,
        cfg_scale: float, stg_scale: float, rescale_scale: float,
        audio_path: str,
        image_path: str,
        enhance_prompt: bool,
        tiling_config,
    ):
        """Call the A2Vid pipeline with frozen audio + frame-0 image pin.

        Identity is anchored at frame 0 via ``VideoConditionByLatentIndex``
        (``combined_image_conditionings`` inside upstream). It propagates
        to subsequent frames through transformer self-attention — same
        pattern I2V uses. Audio drives motion via the
        ``MultiModalGuider`` cross-modal stream.
        """
        from src.upstream import A2VidPipelineTwoStage

        pipeline: A2VidPipelineTwoStage = self._pipeline

        video_guider_params = None
        if self._MultiModalGuiderParams is not None:
            video_guider_params = self._MultiModalGuiderParams(
                cfg_scale=cfg_scale, stg_scale=stg_scale,
                rescale_scale=rescale_scale, modality_scale=3.0,
                stg_blocks=[28],
            )

        if video_guider_params is None:
            raise RuntimeError(
                "A2V requires MultiModalGuiderParams (HAS_GUIDERS). "
                "Check upstream SHA compatibility."
            )

        from src.upstream import ImageConditioningInput
        images = [ImageConditioningInput(path=image_path, frame_idx=0, strength=1.0)]

        call_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            num_inference_steps=num_inference_steps,
            video_guider_params=video_guider_params,
            images=images,
            audio_path=audio_path,
            enhance_prompt=enhance_prompt,
            max_batch_size=1,
        )
        if tiling_config is not None:
            call_kwargs["tiling_config"] = tiling_config
        return pipeline(**call_kwargs)

    # ------------------------------------------------------------------
    # Full generate flow (materialise → denoise → encode → result dict)
    # ------------------------------------------------------------------

    def _a2v_generate(
        self, *,
        prompt: str, negative_prompt: str,
        width: int | None, height: int | None, num_frames: int,
        num_inference_steps: int, seed: int,
        frame_rate: float,
        cfg_scale: float, stg_scale: float, rescale_scale: float,
        audio_url: str | None, audio_b64: str | None,
        image_url: str | None, image_b64: str | None,
        enhance_prompt: bool,
    ) -> dict:
        from src.pipeline.core import _round_user_inputs

        job_id = uuid.uuid4().hex[:12]
        audio_path: str | None = None
        image_path: str | None = None

        try:
            from src.pipeline.inputs import materialize_audio, materialize_image, derive_dims_from_image

            has_image = image_url is not None or image_b64 is not None
            if not has_image:
                raise ValueError(
                    "A2V mode requires an image input (image_url or image_b64) "
                    "alongside the audio input."
                )

            audio_path = materialize_audio(audio_url, audio_b64)
            image_path = materialize_image(image_url, image_b64)

            if width is None and height is None:
                width, height = derive_dims_from_image(image_path)
            if width is None:
                width = 1024
            if height is None:
                height = 1536
            width, height, num_frames = _round_user_inputs(width, height, num_frames)

            audio_src = "audio_url" if audio_url is not None else "audio_b64"
            image_src = "image_url" if image_url is not None else "image_b64"
            logger.info(
                "Job %s: A2V (audio_src=%s, image_src=%s) "
                "prompt=%r, %dx%d, %d frames, %d steps, seed=%d",
                job_id, audio_src, image_src, prompt[:80],
                width, height, num_frames, num_inference_steps, seed,
            )

            tiling_config = None
            video_chunks_number = None
            if self._TilingConfig and self._get_video_chunks_number:
                tiling_config = self._TilingConfig.default()
                video_chunks_number = self._get_video_chunks_number(
                    num_frames, tiling_config,
                )

            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            logger.info(
                "Job %s: offload_mode=%s, teacache=%s",
                job_id, self._offload_mode.value,
                getattr(self, "_teacache_enabled", False),
            )

            result = self._run_a2v(
                prompt=prompt, negative_prompt=negative_prompt,
                seed=seed, height=height, width=width,
                num_frames=num_frames, frame_rate=frame_rate,
                num_inference_steps=num_inference_steps,
                cfg_scale=cfg_scale, stg_scale=stg_scale,
                rescale_scale=rescale_scale,
                audio_path=audio_path,
                image_path=image_path,
                enhance_prompt=enhance_prompt,
                tiling_config=tiling_config,
            )
            video, audio = result if isinstance(result, tuple) else (result, None)
            generation_time = time.time() - start_time

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                logger.info(
                    "Job %s: A2V generation took %.1fs (peak VRAM %.2f GB, audio=%s)",
                    job_id, generation_time, peak, audio is not None,
                )
            else:
                logger.info("Job %s: A2V generation took %.1fs", job_id, generation_time)

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
                    "mode": "a2v",
                    "width": width, "height": height, "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps, "seed": seed,
                    "frame_rate": frame_rate,
                    "cfg_scale": cfg_scale, "stg_scale": stg_scale,
                    "rescale_scale": rescale_scale,
                    "enhance_prompt": enhance_prompt,
                },
            }

        except Exception:
            logger.exception("Job %s failed", job_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
        finally:
            if audio_path is not None:
                try:
                    os.unlink(audio_path)
                except OSError:
                    logger.debug("A2V tempfile cleanup failed for %s", audio_path, exc_info=True)
            if image_path is not None:
                try:
                    os.unlink(image_path)
                except OSError:
                    logger.debug("A2V tempfile cleanup failed for %s", image_path, exc_info=True)
