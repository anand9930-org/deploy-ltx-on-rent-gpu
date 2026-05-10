"""Triple-stages ComfyUI scenario: build + run for the vendored
``TI2VidTripleStagesComfyUIPipeline``.

Wraps ``src/vendor/ti2vid_triple_stages_comfyui.py`` — the byte-for-byte
behavioural port of ``scripts/workflow_3mljpp.py`` (ComfyUI 3-stage AV
workflow). Mixin owns ``_build_triple_stages_comfyui`` (boot/rebuild) and
``_triple_stages_comfyui_generate`` (per-call denoise → encode). Single
mixin covers both T2V (empty ``images=[]``) and I2V.

Key differences from ``TripleStagesMixin`` (the standard fork):

* Distilled LoRA at strength **0.5** (vs 0.8) is stacked on **all three**
  stages by the vendored class — the workflow's single
  ``LoraLoaderModelOnly`` feeds every CFGGuider.
* ``scaled_mm`` rebuilds **all three** stages onto distilled-fp8 (LoRA
  pre-fused at upstream's chosen strength) — the standard fork uses dev-fp8
  for stages 1+2 and distilled-fp8 only for stage 3, but the workflow has
  the LoRA on every stage so the ComfyUI variant must too. Operational
  caveat: the LoRA strength baked into distilled-fp8 may not equal 0.5;
  this is the accepted reality of pre-fused FP8 checkpoints.
* No ``MultiModalGuiderParams`` — the vendored ``__call__`` doesn't accept
  ``video_guider_params``/``audio_guider_params``. Every stage runs cfg=1
  → ``SimpleDenoiser``.
* Sigma schedules + tiling are **baked in** at the vendor module
  (``COMFY_STAGE_*_SIGMAS``, ``COMFY_TILING_CONFIG``); per-call params
  reduce to prompt/dims/seed/image. No ``stage1_steps``/``stage2_steps`` —
  step counts are fixed by the workflow's ManualSigmas literals (9/4/4).
* Resolution must be divisible by **128** (vendored ``_assert_quad_resolution``
  enforces this for the 4× downscale chain).
"""

import logging
import os
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)


def _round_to_128(value: int) -> int:
    return (value // 128) * 128


def _round_frames_8k1(n: int) -> int:
    return ((n - 1) // 8) * 8 + 1


def _round_user_inputs_comfyui(
    width: int, height: int, num_frames: int
) -> tuple[int, int, int]:
    """Round to ComfyUI variant's 4×-chain grid (W/H divisible by 128, frames
    = 8k+1) and log when we change anything."""
    w = _round_to_128(width)
    h = _round_to_128(height)
    f = _round_frames_8k1(num_frames)
    if (w, h, f) != (width, height, num_frames):
        logger.info(
            "ComfyUI input rounded to 4×-chain grid: %dx%d×%d → %dx%d×%d",
            width, height, num_frames, w, h, f,
        )
    return w, h, f


class TripleStagesComfyUIMixin:
    """Methods used when ``pipeline_variant="triple_stages_comfyui"``. Mixed
    into ``LTXVideoGenerator`` via multiple inheritance — do not instantiate
    directly."""

    def _build_triple_stages_comfyui(self) -> None:
        """Build ``TI2VidTripleStagesComfyUIPipeline``. Distilled LoRA goes on
        all three stages (cast/bf16) at strength 0.5; scaled_mm rebuilds
        all three stages onto distilled-fp8 (LoRA pre-fused)."""
        from src.pipeline.core import (
            _build_scaled_mm_policy,
            _install_build_transformer_audit,
        )

        OffloadMode = self._OffloadMode
        offload_mode = self._offload_mode
        fp8_mode = self._fp8_mode
        torch_compile_enabled = self._torch_compile_enabled

        from src.upstream import (
            HAS_TRIPLE_STAGES_COMFYUI,
            TI2VidTripleStagesComfyUIPipeline,
        )
        if not HAS_TRIPLE_STAGES_COMFYUI:
            raise RuntimeError(
                "Triple-stages-ComfyUI pipeline unavailable: vendored module "
                "failed to import. Check src/upstream.py "
                "HAS_TRIPLE_STAGES_COMFYUI."
            )

        if offload_mode != OffloadMode.NONE:
            raise RuntimeError(
                "Triple-stages-ComfyUI requires OffloadMode.NONE (vendored "
                f"class doesn't accept offload_mode); current mode is "
                f"{offload_mode.value}. Either run on a GPU with ≥40 GB VRAM "
                "or stick to the two-stage pipeline."
            )

        for label, path in (
            ("dev BF16", self._dev_bf16_path),
            ("spatial upsampler", self._spatial_upsampler_path),
        ):
            if not os.path.exists(path):
                raise RuntimeError(
                    f"Triple-stages-ComfyUI required {label} checkpoint missing "
                    f"at {path}. Run download_models.py before pipeline init."
                )
        if fp8_mode == "scaled_mm":
            if not os.path.exists(self._distilled_fp8_path):
                raise RuntimeError(
                    f"Triple-stages-ComfyUI scaled_mm requires distilled FP8 "
                    f"DiT at {self._distilled_fp8_path}. Run "
                    "download_models.py with LTX_FP8_MODE=scaled_mm."
                )

        from src.upstream import QuantizationPolicy
        if fp8_mode == "scaled_mm":
            extras_distilled = self._extras_for(self._distilled_fp8_path)
            logger.info(
                "FP8 checkpoint probe (triple-stages-comfyui): distilled=%d "
                "non-FP8 weight modules",
                len(extras_distilled),
            )
            quantization_distilled = _build_scaled_mm_policy(extras_distilled)
            quantization = quantization_distilled  # placeholder; rebuilt per-stage below
            logger.info(
                "Triple-stages-ComfyUI FP8 mode: scaled_mm (W8A8, TRT-LLM "
                "cublas_scaled_mm) — distilled-fp8 on all three stages"
            )
        elif fp8_mode == "cast":
            quantization = QuantizationPolicy.fp8_cast()
            quantization_distilled = None
            logger.info(
                "Triple-stages-ComfyUI FP8 mode: cast (W8A16, weights FP8 / "
                "activations BF16)"
            )
        else:
            quantization = None
            quantization_distilled = None

        from src.upstream import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
            StateDictRegistry,
        )
        from src.vendor.ti2vid_triple_stages_comfyui import (
            COMFY_DISTILLED_LORA_STRENGTH,
        )
        registry: StateDictRegistry | None = None
        try:
            registry = StateDictRegistry()
            logger.info(
                "Triple-stages-ComfyUI using StateDictRegistry (CPU weight caching)"
            )
        except Exception:
            logger.warning("StateDictRegistry not available", exc_info=True)

        # Distilled LoRA — cast/bf16 only at workflow strength 0.5. The
        # vendored class stacks `distilled_lora` on every stage. scaled_mm
        # uses pre-fused distilled-fp8 instead and passes [] here.
        if fp8_mode != "scaled_mm":
            if not os.path.exists(self._distilled_lora_path):
                raise RuntimeError(
                    f"Triple-stages-ComfyUI (cast/bf16) requires distilled LoRA "
                    f"at {self._distilled_lora_path}. Run download_models.py."
                )
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    path=self._distilled_lora_path,
                    strength=COMFY_DISTILLED_LORA_STRENGTH,
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
        )
        if quantization is not None:
            pipeline_kwargs["quantization"] = quantization
        if registry is not None:
            pipeline_kwargs["registry"] = registry
        if torch_compile_enabled and quantization is not None:
            pipeline_kwargs["torch_compile"] = True

        _t0 = time.perf_counter()
        self._pipeline = TI2VidTripleStagesComfyUIPipeline(**pipeline_kwargs)
        logger.info(
            "Triple-stages-ComfyUI init: TI2VidTripleStagesComfyUIPipeline "
            "construction took %.2fs",
            time.perf_counter() - _t0,
        )
        self._log_vram("Triple-stages-ComfyUI after pipeline init")

        # scaled_mm: rebuild all three stages onto distilled-fp8 (LoRA pre-fused
        # at upstream's chosen strength). Diverges from the standard fork,
        # which keeps stages 1+2 on dev-fp8.
        if fp8_mode == "scaled_mm":
            from src.upstream import DiffusionStage
            for stage_attr in ("stage_1", "stage_2", "stage_3"):
                _t1 = time.perf_counter()
                setattr(
                    self._pipeline, stage_attr,
                    DiffusionStage(
                        checkpoint_path=self._distilled_fp8_path,
                        dtype=self._pipeline.dtype,
                        device=self._pipeline.device,
                        loras=(),
                        quantization=quantization_distilled,
                        registry=None,
                        torch_compile=pipeline_kwargs.get("torch_compile", False),
                    ),
                )
                logger.info(
                    "Triple-stages-ComfyUI init: DiffusionStage(%s) constructor "
                    "took %.3fs (%s)",
                    stage_attr, time.perf_counter() - _t1,
                    os.path.basename(self._distilled_fp8_path),
                )
                _install_build_transformer_audit(
                    getattr(self._pipeline, stage_attr),
                    f"triple_comfyui_{stage_attr}",
                )

        # TeaCache stays OFF for triple-stages-comfyui — same I2V quality
        # concern as unified (memory: teacache_i2v_quality.md).
        self._teacache_enabled = False

        self._log_attention_fingerprint()

    def _triple_stages_comfyui_generate(
        self, *, prompt: str, negative_prompt: str,
        width: int | None, height: int | None, num_frames: int,
        seed: int, frame_rate: float,
        image_url: str | None, image_b64: str | None,
        image_frame_idx: int,
        enhance_prompt: bool,
    ) -> dict:
        from src.vendor.ti2vid_triple_stages_comfyui import COMFY_TILING_CONFIG

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

            # Workflow's effective output is 896×1280 (via the hardcoded
            # EmptyLTXVLatentVideo(width=224, height=320) × 4× chain). Use
            # that as the fallback when caller omits dims.
            if width is None:
                width = 896
            if height is None:
                height = 1280
            width, height, num_frames = _round_user_inputs_comfyui(
                width, height, num_frames
            )

            mode = "triple_comfyui_i2v" if has_image else "triple_comfyui_t2v"
            src_label = (
                "image_url" if image_url is not None
                else "image_b64" if image_b64 is not None
                else "none"
            )
            logger.info(
                "Job %s: %s (image_src=%s) prompt=%r, %dx%d, %d frames, "
                "fixed schedule (stage1=9/stage2=4/stage3=4 sigmas), seed=%d",
                job_id, mode.upper(), src_label, prompt[:80],
                width, height, num_frames, seed,
            )

            tiling_config = COMFY_TILING_CONFIG
            video_chunks_number = None
            if self._get_video_chunks_number is not None:
                video_chunks_number = self._get_video_chunks_number(
                    num_frames, tiling_config
                )

            from src.upstream import ImageConditioningInput
            images = (
                [ImageConditioningInput(
                    path=image_path, frame_idx=image_frame_idx, strength=1.0,
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
                images=images,
                tiling_config=tiling_config,
                enhance_prompt=enhance_prompt,
                max_batch_size=1,
            )

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
                    "seed": seed, "frame_rate": frame_rate,
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
                        "Triple-stages-ComfyUI tempfile cleanup failed for %s",
                        image_path, exc_info=True,
                    )
