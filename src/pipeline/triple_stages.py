"""Triple-stages scenario: build + run for the vendored
``TI2VidTripleStagesPipeline``.

Wraps ``src/vendor/ti2vid_triple_stages.py`` (a community fork of the
upstream two-stage pipeline that runs Stage 1 at half-resolution then two
full-res refinement passes — typically yields better I2V quality).
Mixin owns ``_build_triple_stages`` (boot/rebuild) and
``_triple_stages_generate`` (per-call denoise → encode). The upstream class
takes both T2V (empty ``images=[]``) and I2V (``ImageConditioningInput``
list) through the same ``__call__``, so a single mixin covers both.

Key differences from ``T2VMixin``:
  * Three ``DiffusionStage`` instances, not two — scaled_mm path rebuilds
    stage_1+stage_2 onto dev-fp8 and stage_3 onto distilled-fp8 (distilled
    LoRA is pre-fused into the distilled-fp8 checkpoint).
  * Vendored class doesn't accept ``offload_mode`` — CPU offload is not
    supported. Triple-stages requires ``OffloadMode.NONE``.
  * Per-call params are ``stage1_steps`` + ``stage2_steps`` (Karras sigma
    schedule), not the unified ``num_inference_steps``.
"""

import logging
import os
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)


class TripleStagesMixin:
    """Methods used when ``pipeline_variant="triple_stages"``. Mixed into
    ``LTXVideoGenerator`` via multiple inheritance — do not instantiate
    directly."""

    def _build_triple_stages(self) -> None:
        """Build ``TI2VidTripleStagesPipeline``: stage 1+2 on dev (or dev-fp8
        in scaled_mm), stage 3 on distilled (or distilled-fp8). Distilled
        LoRA goes on stage 3 only — matches the vendored class's
        ``__init__`` (see ``src/vendor/ti2vid_triple_stages.py``)."""
        from src.pipeline.core import (
            _build_scaled_mm_policy,
            _install_build_transformer_audit,
        )

        OffloadMode = self._OffloadMode
        offload_mode = self._offload_mode
        fp8_mode = self._fp8_mode
        torch_compile_enabled = self._torch_compile_enabled

        from src.upstream import HAS_TRIPLE_STAGES, TI2VidTripleStagesPipeline
        if not HAS_TRIPLE_STAGES:
            raise RuntimeError(
                "Triple-stages pipeline unavailable: vendored module failed to "
                "import (likely missing ltx_core.components.guiders on the "
                "pinned upstream SHA). Check src/upstream.py HAS_TRIPLE_STAGES."
            )

        if offload_mode != OffloadMode.NONE:
            raise RuntimeError(
                "Triple-stages requires OffloadMode.NONE (vendored class "
                f"doesn't accept offload_mode); current mode is {offload_mode.value}. "
                "Either run on a GPU with ≥40 GB VRAM or stick to the "
                "two-stage pipeline."
            )

        for label, path in (
            ("dev BF16", self._dev_bf16_path),
            ("spatial upsampler", self._spatial_upsampler_path),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Triple-stages required {label} checkpoint missing at {path}. "
                    "Run download_models.py before pipeline init."
                )
        if fp8_mode == "scaled_mm":
            for label, path in (
                ("dev FP8 DiT", self._dev_fp8_path),
                ("distilled FP8 DiT", self._distilled_fp8_path),
            ):
                if not os.path.exists(path):
                    raise RuntimeError(
                        f"Triple-stages scaled_mm requires {label} at {path}. "
                        "Run download_models.py with LTX_FP8_MODE=scaled_mm."
                    )

        from src.upstream import QuantizationPolicy
        if fp8_mode == "scaled_mm":
            extras_dev = self._extras_for(self._dev_fp8_path)
            extras_distilled = self._extras_for(self._distilled_fp8_path)
            logger.info(
                "FP8 checkpoint probe (triple-stages): dev=%d, distilled=%d "
                "non-FP8 weight modules",
                len(extras_dev), len(extras_distilled),
            )
            quantization_dev = _build_scaled_mm_policy(extras_dev)
            quantization_distilled = _build_scaled_mm_policy(extras_distilled)
            quantization = quantization_dev  # placeholder; rebuilt per-stage below
            logger.info(
                "Triple-stages FP8 mode: scaled_mm (W8A8, TRT-LLM cublas_scaled_mm)"
            )
        elif fp8_mode == "cast":
            quantization = QuantizationPolicy.fp8_cast()
            quantization_dev = None
            quantization_distilled = None
            logger.info(
                "Triple-stages FP8 mode: cast (W8A16, weights FP8 / activations BF16)"
            )
        else:
            quantization = None
            quantization_dev = None
            quantization_distilled = None

        from src.upstream import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
            StateDictRegistry,
        )
        registry: StateDictRegistry | None = None
        try:
            registry = StateDictRegistry()
            logger.info("Triple-stages using StateDictRegistry (CPU weight caching)")
        except Exception:
            logger.warning("StateDictRegistry not available", exc_info=True)

        # Distilled LoRA — cast/bf16 only. scaled_mm uses pre-fused distilled-fp8
        # for stage 3. The vendored class applies distilled_lora to stage 3 only.
        if fp8_mode != "scaled_mm":
            if not os.path.exists(self._distilled_lora_path):
                raise RuntimeError(
                    f"Triple-stages (cast/bf16) requires distilled LoRA at "
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

        # Vendored TI2VidTripleStagesPipeline.__init__ does NOT accept
        # offload_mode (we asserted NONE above). Build kwargs accordingly.
        pipeline_kwargs = dict(
            checkpoint_path=self._dev_bf16_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=self._spatial_upsampler_path,
            gemma_root=self._gemma_root,
            loras=[],
        )
        if quantization is not None:
            pipeline_kwargs["quantization"] = quantization
        if registry is not None:
            pipeline_kwargs["registry"] = registry
        if torch_compile_enabled and quantization is not None:
            pipeline_kwargs["torch_compile"] = True

        _t0 = time.perf_counter()
        self._pipeline = TI2VidTripleStagesPipeline(**pipeline_kwargs)
        logger.info(
            "Triple-stages init: TI2VidTripleStagesPipeline construction took %.2fs",
            time.perf_counter() - _t0,
        )
        self._log_vram("Triple-stages after pipeline init")

        # scaled_mm: rebuild stages onto FP8 DiT files. Stages 1+2 use dev-fp8
        # (base model); stage 3 uses distilled-fp8 (LoRA pre-fused).
        if fp8_mode == "scaled_mm":
            from src.upstream import DiffusionStage
            for stage_attr, ckpt_path, qpolicy in (
                ("stage_1", self._dev_fp8_path, quantization_dev),
                ("stage_2", self._dev_fp8_path, quantization_dev),
                ("stage_3", self._distilled_fp8_path, quantization_distilled),
            ):
                _t1 = time.perf_counter()
                setattr(
                    self._pipeline, stage_attr,
                    DiffusionStage(
                        checkpoint_path=ckpt_path,
                        dtype=self._pipeline.dtype,
                        device=self._pipeline.device,
                        loras=(),
                        quantization=qpolicy,
                        registry=None,
                        torch_compile=pipeline_kwargs.get("torch_compile", False),
                    ),
                )
                logger.info(
                    "Triple-stages init: DiffusionStage(%s) constructor took %.3fs (%s)",
                    stage_attr, time.perf_counter() - _t1, os.path.basename(ckpt_path),
                )
                _install_build_transformer_audit(
                    getattr(self._pipeline, stage_attr), f"triple_{stage_attr}"
                )

        # Optional MultiModalGuiderParams (vendored class requires it).
        from src.upstream import HAS_GUIDERS, MultiModalGuiderParams
        if not HAS_GUIDERS:
            raise RuntimeError(
                "Triple-stages requires MultiModalGuiderParams; HAS_GUIDERS is False. "
                "Check upstream pin in Dockerfile."
            )
        self._MultiModalGuiderParams = MultiModalGuiderParams

        # TeaCache stays OFF for triple-stages — same I2V quality concern as
        # unified (memory: teacache_i2v_quality.md). Leave hook absent.
        self._teacache_enabled = False

        self._log_attention_fingerprint()

    def _triple_stages_generate(
        self, *, prompt: str, negative_prompt: str,
        width: int | None, height: int | None, num_frames: int,
        seed: int, frame_rate: float,
        cfg_scale: float, stg_scale: float, rescale_scale: float,
        image_url: str | None, image_b64: str | None,
        image_strength: float,
        image_frame_idx: int,
        stage1_steps: int, stage2_steps: int,
        enhance_prompt: bool,
    ) -> dict:
        from src.pipeline.core import _round_user_inputs

        job_id = uuid.uuid4().hex[:12]
        image_path: str | None = None
        try:
            has_image = image_url is not None or image_b64 is not None
            if has_image:
                from src.pipeline.inputs import (
                    derive_dims_from_image,
                    materialize_image,
                )
                image_path = materialize_image(image_url, image_b64)
                if width is None and height is None:
                    width, height = derive_dims_from_image(image_path)

            if width is None:
                width = 1536
            if height is None:
                height = 1024
            width, height, num_frames = _round_user_inputs(width, height, num_frames)

            mode = "triple_i2v" if has_image else "triple_t2v"
            src_label = (
                "image_url" if image_url is not None
                else "image_b64" if image_b64 is not None
                else "none"
            )
            logger.info(
                "Job %s: %s (image_src=%s) prompt=%r, %dx%d, %d frames, "
                "stage1=%d steps, stage2=%d steps (×2), seed=%d",
                job_id, mode.upper(), src_label, prompt[:80],
                width, height, num_frames, stage1_steps, stage2_steps, seed,
            )

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
            images = (
                [ImageConditioningInput(
                    path=image_path, frame_idx=image_frame_idx, strength=image_strength,
                )]
                if has_image else []
            )

            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate,
                video_guider_params=video_guider_params,
                audio_guider_params=audio_guider_params,
                images=images,
                stage1_steps=stage1_steps,
                stage2_steps=stage2_steps,
                enhance_prompt=enhance_prompt,
                max_batch_size=1,
            )
            if tiling_config is not None:
                call_kwargs["tiling_config"] = tiling_config

            result = self._pipeline(**call_kwargs)
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
                    "stage1_steps": stage1_steps, "stage2_steps": stage2_steps,
                    "seed": seed, "frame_rate": frame_rate,
                    "cfg_scale": cfg_scale, "stg_scale": stg_scale,
                    "rescale_scale": rescale_scale,
                    "image_strength": image_strength if has_image else None,
                    "image_frame_idx": image_frame_idx if has_image else None,
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
                    logger.debug(
                        "Triple-stages tempfile cleanup failed for %s",
                        image_path, exc_info=True,
                    )
