"""Triple-stages ComfyUI scenario: build + run for the ComfyUI graph pipeline.

Mixin owns ``_build_triple_stages_comfyui`` (boot/rebuild) and
``_triple_stages_comfyui_generate`` (per-call denoise → encode). Single
mixin covers both T2V (empty ``images=[]``) and I2V.

Output dimensions are resolved from ``aspect_ratio`` into two canonical
buckets: landscape generates 1920x1152 and post-crops to 1920x1080, portrait
generates 1152x1920 and post-crops to 1080x1920. The /128 generation grid is
required by the 3-stage cascade (Stage 1 runs at final/4 and must be VAE-/32
aligned); the post-decode center-crop is what turns that into "true 1080p".
"""

import logging
import os
import tempfile
import time
import uuid
from typing import Literal

import torch

logger = logging.getLogger(__name__)


# (gen_w, gen_h, out_w, out_h) — generation dims must be /128 (3-stage cascade);
# output dims are what the caller receives after the post-decode center-crop.
LANDSCAPE_BUCKET = (1920, 1152, 1920, 1080)
PORTRAIT_BUCKET = (1152, 1920, 1080, 1920)


def _round_frames_8k1(n: int) -> int:
    """Floor ``n`` to ``8k+1`` — LTX-2 VAE temporal factor."""
    return ((n - 1) // 8) * 8 + 1


def _resolve_aspect_ratio(
    aspect_ratio: Literal["16:9", "9:16", "auto"],
    image_path: str | None,
) -> tuple[int, int, int, int]:
    """Resolve the request's ``aspect_ratio`` into ``(gen_w, gen_h, out_w, out_h)``.

    ``"auto"`` derives orientation from the input image (I2V) or falls back to
    landscape (T2V). Explicit ``"16:9"`` / ``"9:16"`` always win, even if the
    input image's orientation disagrees — that's an intentional caller override.
    """
    if aspect_ratio == "auto":
        if image_path is not None:
            from src.pipeline.inputs import derive_orientation

            aspect_ratio = derive_orientation(image_path)
        else:
            aspect_ratio = "16:9"
    return LANDSCAPE_BUCKET if aspect_ratio == "16:9" else PORTRAIT_BUCKET


class TripleStagesComfyUIMixin:
    """ComfyUI graph pipeline mixin — mixed into ``LTXVideoGenerator``."""

    def _build_triple_stages_comfyui(self) -> None:
        """Build ``TripleStagesComfyUIGraphPipeline`` — runs the real ComfyUI
        **core** node graph (ComfyUI's own model loading). The ComfyUI runtime
        is bootstrapped lazily on first use and stays resident for the process."""
        from src import comfyui_runtime
        from src.config import get_settings
        from src.pipeline.triple_stages_comfyui_graph import (
            TripleStagesComfyUIGraphPipeline,
        )

        settings = get_settings()
        comfyui_runtime.reset_to_clean_gpu()
        _t0 = time.perf_counter()
        self._pipeline = TripleStagesComfyUIGraphPipeline(
            model_dir=settings.model_dir,
            comfyui_path=settings.comfyui_path,
        )
        logger.info(
            "Pipeline ready in %.1fs (ComfyUI=%s)",
            time.perf_counter() - _t0,
            settings.comfyui_path,
        )

    def _triple_stages_comfyui_generate(
        self,
        *,
        prompt: str,
        negative_prompt: str,
        aspect_ratio: Literal["16:9", "9:16", "auto"],
        num_frames: int,
        seed: int,
        frame_rate: float,
        image_url: str | None,
        image_b64: str | None,
        image_frame_idx: int,
        enhance_prompt: bool,
    ) -> dict:
        job_id = uuid.uuid4().hex[:12]
        image_path: str | None = None
        try:
            has_image = image_url is not None or image_b64 is not None
            if has_image:
                from src.pipeline.inputs import materialize_image

                image_path = materialize_image(image_url, image_b64)

            gen_w, gen_h, out_w, out_h = _resolve_aspect_ratio(
                aspect_ratio, image_path,
            )
            resolved_ratio: Literal["16:9", "9:16"] = (
                "16:9" if (gen_w, gen_h) == LANDSCAPE_BUCKET[:2] else "9:16"
            )
            rounded_frames = _round_frames_8k1(num_frames)
            if rounded_frames != num_frames:
                logger.info(
                    "num_frames rounded to 8k+1: %d → %d",
                    num_frames, rounded_frames,
                )
            num_frames = rounded_frames

            mode = "triple_comfyui_i2v" if has_image else "triple_comfyui_t2v"
            src_label = (
                "image_url"
                if image_url is not None
                else "image_b64" if image_b64 is not None else "none"
            )
            logger.info(
                "Job %s: %s (image_src=%s) prompt=%r, aspect=%s, gen=%dx%d, "
                "out=%dx%d, %d frames, fixed schedule "
                "(stage1=9/stage2=4/stage3=4 sigmas), seed=%d",
                job_id,
                mode.upper(),
                src_label,
                prompt[:80],
                resolved_ratio,
                gen_w,
                gen_h,
                out_w,
                out_h,
                num_frames,
                seed,
            )

            from src.pipeline.triple_stages_comfyui_graph import ImageInput

            images = (
                [
                    ImageInput(
                        path=image_path,
                        frame_idx=image_frame_idx,
                        strength=1.0,
                    ),
                ]
                if has_image
                else []
            )

            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            call_kwargs = dict(
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed,
                height=gen_h,
                width=gen_w,
                num_frames=num_frames,
                frame_rate=frame_rate,
                images=images,
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
                    job_id,
                    mode.upper(),
                    generation_time,
                    peak,
                    audio is not None,
                )
            else:
                logger.info(
                    "Job %s: %s generation took %.1fs",
                    job_id,
                    mode.upper(),
                    generation_time,
                )

            output_filename = f"ltx_{job_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            encode_start = time.time()
            self._pipeline.encode_to_mp4(
                frames=video,
                audio=audio,
                output_path=output_path,
                fps=int(frame_rate),
                out_w=out_w,
                out_h=out_h,
            )
            logger.info(
                "Job %s: mp4 encode %.1fs → %s",
                job_id,
                time.time() - encode_start,
                output_path,
            )

            return {
                "output_path": output_path,
                "output_filename": output_filename,
                "generation_time_seconds": round(generation_time, 2),
                "parameters": {
                    "mode": mode,
                    "aspect_ratio": resolved_ratio,
                    "width": out_w,
                    "height": out_h,
                    "num_frames": num_frames,
                    "seed": seed,
                    "frame_rate": frame_rate,
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
                        "Tempfile cleanup failed for %s",
                        image_path,
                        exc_info=True,
                    )
