"""A2V scenario: build + run for ``A2VidPipelineTwoStage`` with IC-LoRA.

Combines frozen external audio conditioning (consistent voice via TTS,
natural lipsync) with IC-LoRA ``VideoConditionByReferenceLatent`` (strong
image fidelity). The two conditioning mechanisms operate on separate
transformer streams and do not compete for attention budget.

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

from safetensors import safe_open

logger = logging.getLogger(__name__)


def _read_lora_reference_downscale_factor(lora_path: str) -> int:
    try:
        with safe_open(lora_path, framework="pt") as f:
            metadata = f.metadata() or {}
            return int(metadata.get("reference_downscale_factor", 1))
    except Exception as e:
        logger.warning("Failed to read metadata from LoRA file %r: %s", lora_path, e)
        return 1


class A2VMixin:
    """Methods used when the request supplies audio input. Mixed into
    ``LTXVideoGenerator`` via multiple inheritance."""

    def _build_a2v(self) -> None:
        """Build ``A2VidPipelineTwoStage`` with IC-LoRA: dev checkpoint
        (audio conditioning calibrated) + IC-LoRA weights (strong image
        reference). FP8 quantisation follows the T2V/I2V pattern."""
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
            ("IC-LoRA", self._ic_lora_path),
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

        ic_lora = [
            LoraPathStrengthAndSDOps(
                path=self._ic_lora_path,
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]

        reference_downscale_factor = _read_lora_reference_downscale_factor(
            self._ic_lora_path
        )
        logger.info(
            "A2V IC-LoRA: %s (reference_downscale_factor=%d)",
            os.path.basename(self._ic_lora_path), reference_downscale_factor,
        )

        pipeline_kwargs = dict(
            checkpoint_path=self._dev_bf16_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=self._spatial_upsampler_path,
            gemma_root=self._gemma_root,
            loras=ic_lora,
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
                loras=tuple(ic_lora),
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
                "A2V scaled_mm: stage_1 → %s (+IC-LoRA), stage_2 → %s",
                os.path.basename(self._dev_fp8_path),
                os.path.basename(self._distilled_fp8_path),
            )
            _install_build_transformer_audit(self._pipeline.stage_1, "a2v_stage_1")
            _install_build_transformer_audit(self._pipeline.stage_2, "a2v_stage_2")

        if offload_mode != OffloadMode.NONE:
            _install_stage2_cleanup_hook(self._pipeline)

        self._unified_meta["reference_downscale_factor"] = reference_downscale_factor

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
        reference_video_strength: float,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        tiling_config,
    ):
        """Call the A2Vid pipeline with frozen audio + IC-LoRA image ref.

        Uses the same image for both the standard A2Vid frame-0 pin
        (``VideoConditionByLatentIndex``) and the IC-LoRA reference
        (``VideoConditionByReferenceLatent``). The IC-LoRA conditioning
        is injected into Stage 1 only, matching ``ICLoraPipeline``'s
        pattern.
        """
        from src.upstream import (
            A2VidPipelineTwoStage,
            VideoConditionByReferenceLatent,
            ConditioningItemAttentionStrengthWrapper,
            HAS_GUIDERS,
        )

        pipeline: A2VidPipelineTwoStage = self._pipeline
        ref_downscale = self._unified_meta.get("reference_downscale_factor", 1)

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

        images = [(image_path, 0, 1.0)]

        # --- IC-LoRA injection via image_conditioner wrapper ---
        #
        # A2VidPipelineTwoStage.__call__ calls self.image_conditioner(fn)
        # twice: once for Stage 1, once for Stage 2. We wrap the
        # image_conditioner to intercept the first call (Stage 1) and
        # append VideoConditionByReferenceLatent conditionings. Stage 2
        # gets standard conditionings only (matching ICLoraPipeline).
        original_ic = pipeline.image_conditioner

        class _ICLoRAConditionerWrapper:
            """Wraps ImageConditioner to add IC-LoRA reference tokens on
            the first call (Stage 1) only."""

            def __init__(self, original, img_path, dsf, strength, attn_strength):
                self._original = original
                self._img_path = img_path
                self._dsf = dsf
                self._strength = strength
                self._attn_strength = attn_strength
                self._call_count = 0

            def __call__(self, fn):
                self._call_count += 1
                if self._call_count == 1:
                    img_path = self._img_path
                    dsf = self._dsf
                    strength = self._strength
                    attn_strength = self._attn_strength

                    def enhanced_fn(enc):
                        conditionings = fn(enc)
                        from src.upstream import decode_video_by_frame, video_preprocess
                        frame_gen = decode_video_by_frame(
                            path=img_path,
                            frame_cap=num_frames,
                            device=pipeline.device,
                        )
                        ref_h = height // (2 * dsf) if dsf != 1 else height // 2
                        ref_w = width // (2 * dsf) if dsf != 1 else width // 2
                        video_tensor = video_preprocess(
                            frame_gen, ref_h, ref_w,
                            pipeline.dtype, pipeline.device,
                        )
                        encoded_ref = enc(video_tensor)
                        cond = VideoConditionByReferenceLatent(
                            latent=encoded_ref,
                            downscale_factor=dsf,
                            strength=strength,
                        )
                        if attn_strength < 1.0:
                            cond = ConditioningItemAttentionStrengthWrapper(
                                cond, attention_mask=attn_strength,
                            )
                        conditionings.append(cond)
                        return conditionings

                    return self._original(enhanced_fn)
                return self._original(fn)

        pipeline.image_conditioner = _ICLoRAConditionerWrapper(
            original_ic, image_path, ref_downscale,
            reference_video_strength, conditioning_attention_strength,
        )

        try:
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
        finally:
            pipeline.image_conditioner = original_ic

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
        reference_video_strength: float,
        conditioning_attention_strength: float,
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
            ref_downscale = self._unified_meta.get("reference_downscale_factor")
            logger.info(
                "Job %s: A2V (audio_src=%s, image_src=%s) "
                "prompt=%r, %dx%d, %d frames, %d steps, seed=%d, "
                "ref_strength=%.2f, attn_strength=%.2f, ref_downscale=%s",
                job_id, audio_src, image_src, prompt[:80],
                width, height, num_frames, num_inference_steps, seed,
                reference_video_strength, conditioning_attention_strength,
                ref_downscale,
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
                reference_video_strength=reference_video_strength,
                conditioning_attention_strength=conditioning_attention_strength,
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
                    "reference_downscale_factor": ref_downscale,
                    "reference_video_strength": reference_video_strength,
                    "conditioning_attention_strength": conditioning_attention_strength,
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
