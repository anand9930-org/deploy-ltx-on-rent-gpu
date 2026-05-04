"""I2V scenario: build + run for the unified ``ICLoraPipeline``.

The unified pipeline serves both I2V (image conditioning) and V2V (video
conditioning) — V2V piggybacks on the build done here. This mixin owns
``_build_unified`` (boot/rebuild), ``_unified_generate`` (per-call
materialise → dispatch → encode), and ``_run_i2v`` (per-call upstream call
when only an image is supplied). The V2V-specific ``_run_v2v`` lives in
``v2v.py``.
"""

import logging
import os
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)


class I2VMixin:
    """Methods used when the request supplies an image and/or reference
    video (the unified ICLoraPipeline path). Mixed into
    ``LTXVideoGenerator`` via multiple inheritance — do not instantiate
    directly."""

    def _build_unified(self) -> None:
        """Build ``ICLoraPipeline`` (I2V + V2V): distilled-1.1 BF16 base,
        IC-LoRA Union-Control on stage 1, distilled-fp8 stage 2 (scaled_mm
        swaps both stages to FP8)."""
        from src.pipeline.core import (
            _build_scaled_mm_policy,
            _install_build_transformer_audit,
            _install_stage2_cleanup_hook,
        )

        OffloadMode = self._OffloadMode
        offload_mode = self._offload_mode
        fp8_mode = self._fp8_mode
        fp8_enabled = self._fp8_enabled
        torch_compile_enabled = self._torch_compile_enabled

        for label, path in (
            ("distilled-1.1 BF16", self._distilled_bf16_path),
            ("spatial upsampler", self._spatial_upsampler_path),
            ("IC-LoRA Union-Control", self._ic_lora_path),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Unified pipeline required {label} checkpoint missing at "
                    f"{path}. Run download_models.py before pipeline init."
                )
        if fp8_enabled and not os.path.exists(self._distilled_fp8_path):
            raise RuntimeError(
                f"Unified pipeline FP8 mode requires distilled FP8 DiT at "
                f"{self._distilled_fp8_path}. Run download_models.py with "
                f"LTX_FP8_MODE={fp8_mode}."
            )

        from src.upstream import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
            StateDictRegistry,
        )
        from src.upstream import ICLoraPipeline

        # Quantization policy.
        if fp8_enabled:
            from src.upstream import QuantizationPolicy
            if fp8_mode == "scaled_mm":
                extras = self._extras_for(self._distilled_fp8_path)
                logger.info(
                    "Unified FP8 checkpoint probe: %d non-FP8 weight modules",
                    len(extras),
                )
                if extras:
                    preview = ", ".join(extras[:5])
                    more = f" (+{len(extras) - 5} more)" if len(extras) > 5 else ""
                    logger.info("Unified FP8 probe extras (first 5): %s%s", preview, more)
                quantization = _build_scaled_mm_policy(extras)
                logger.info(
                    "Unified FP8 mode: scaled_mm (W8A8, TRT-LLM cublas_scaled_mm)"
                )
            else:
                quantization = QuantizationPolicy.fp8_cast()
                logger.info("Unified FP8 mode: cast (W8A16, weights FP8 / activations BF16)")
        else:
            quantization = None

        registry = StateDictRegistry()
        logger.info("Unified pipeline using StateDictRegistry (CPU weight caching)")

        stage_1_loras = [
            LoraPathStrengthAndSDOps(
                path=self._ic_lora_path,
                strength=1.0,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            ),
        ]

        _t0 = time.perf_counter()
        self._pipeline = ICLoraPipeline(
            distilled_checkpoint_path=self._distilled_bf16_path,
            spatial_upsampler_path=self._spatial_upsampler_path,
            gemma_root=self._gemma_root,
            loras=stage_1_loras,
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile_enabled,
            offload_mode=offload_mode,
        )
        logger.info(
            "Unified init: ICLoraPipeline construction took %.2fs "
            "(reference_downscale_factor=%d)",
            time.perf_counter() - _t0,
            self._pipeline.reference_downscale_factor,
        )
        self._unified_meta["reference_downscale_factor"] = (
            self._pipeline.reference_downscale_factor
        )
        self._log_vram("Unified after pipeline init")

        # FP8: rebuild both DiffusionStages onto distilled-fp8.
        if fp8_enabled:
            from src.upstream import DiffusionStage
            _t1 = time.perf_counter()
            self._pipeline.stage_1 = DiffusionStage(
                checkpoint_path=self._distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=tuple(stage_1_loras),
                quantization=quantization,
                registry=None,
                torch_compile=torch_compile_enabled,
                offload_mode=offload_mode,
            )
            self._pipeline.stage_2 = DiffusionStage(
                checkpoint_path=self._distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization,
                registry=None,
                torch_compile=torch_compile_enabled,
                offload_mode=offload_mode,
            )
            logger.info(
                "Unified init: FP8 DiffusionStage rebuild took %.3fs",
                time.perf_counter() - _t1,
            )
            _install_build_transformer_audit(self._pipeline.stage_1, "unified_stage_1")
            _install_build_transformer_audit(self._pipeline.stage_2, "unified_stage_2")

        if offload_mode != OffloadMode.NONE:
            _install_stage2_cleanup_hook(self._pipeline)

        logger.info(
            "Unified pipeline configured: precision=%s mode=%s offload=%s torch_compile=%s",
            "fp8" if fp8_enabled else "bf16",
            fp8_mode or "n/a", offload_mode.value, torch_compile_enabled,
        )

        # TeaCache stays OFF for unified by default — MEMORY.md notes I2V
        # quality regression; ENABLE_TEACACHE=1 honoured for experiments.
        from src.teacache import enable_teacache, teacache_config_from_env
        teacache_cfg = teacache_config_from_env()
        self._teacache_enabled = teacache_cfg is not None
        if self._teacache_enabled:
            enable_teacache(self._pipeline, **teacache_cfg)

        self._log_attention_fingerprint()

    def _unified_generate(
        self, *, prompt: str,
        width: int | None, height: int | None, num_frames: int,
        seed: int, frame_rate: float,
        image_url: str | None, image_b64: str | None,
        reference_video_url: str | None, reference_video_b64: str | None,
        reference_video_strength: float,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        has_image: bool, has_ref_video: bool,
    ) -> dict:
        from src.pipeline.core import _round_user_inputs

        job_id = uuid.uuid4().hex[:12]
        image_path: str | None = None
        ref_video_path: str | None = None
        try:
            if has_image:
                from src.pipeline.inputs import (
                    derive_dims_from_image,
                    materialize_image,
                )
                image_path = materialize_image(image_url, image_b64)
                if width is None and height is None:
                    width, height = derive_dims_from_image(image_path)

            if has_ref_video:
                from src.pipeline.inputs import materialize_video
                ref_video_path = materialize_video(
                    reference_video_url, reference_video_b64
                )

            if width is None:
                width = 1024
            if height is None:
                height = 1536
            width, height, num_frames = _round_user_inputs(width, height, num_frames)

            mode = "v2v" if has_ref_video else "i2v"

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
                job_id, mode.upper(), src_label, ref_label, prompt[:80],
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

            ref_downscale = self._unified_meta.get("reference_downscale_factor")
            logger.info(
                "Job %s: offload_mode=%s, teacache=%s, ref_downscale=%s",
                job_id, self._offload_mode.value, self._teacache_enabled,
                ref_downscale,
            )

            if has_ref_video:
                result = self._run_v2v(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    image_path=image_path,
                    ref_video_path=ref_video_path,  # type: ignore[arg-type]
                    reference_video_strength=reference_video_strength,
                    conditioning_attention_strength=conditioning_attention_strength,
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                )
            else:
                result = self._run_i2v(
                    prompt=prompt, seed=seed, height=height, width=width,
                    num_frames=num_frames, frame_rate=frame_rate,
                    image_path=image_path,  # type: ignore[arg-type]
                    conditioning_attention_strength=conditioning_attention_strength,
                    enhance_prompt=enhance_prompt,
                    tiling_config=tiling_config,
                )
            video, audio = result if isinstance(result, tuple) else (result, None)
            generation_time = time.time() - start_time

            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                logger.info(
                    "Job %s: %s generation took %.1fs (peak VRAM %.2f GB, audio=%s)",
                    job_id, mode.upper(), generation_time, peak, audio is not None,
                )
            else:
                logger.info(
                    "Job %s: %s generation took %.1fs",
                    job_id, mode.upper(), generation_time,
                )

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
                    "mode": mode,
                    "width": width, "height": height, "num_frames": num_frames,
                    "seed": seed, "frame_rate": frame_rate,
                    "reference_downscale_factor": ref_downscale,
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
            if torch.cuda.is_available():
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

    def _run_i2v(
        self, *,
        prompt: str, seed: int, height: int, width: int,
        num_frames: int, frame_rate: float,
        image_path: str,
        conditioning_attention_strength: float,
        enhance_prompt: bool,
        tiling_config,
    ):
        from src.upstream import ImageConditioningInput
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
