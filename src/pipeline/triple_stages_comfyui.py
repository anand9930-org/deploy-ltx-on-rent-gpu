"""Triple-stages ComfyUI scenario: build + run for the ComfyUI graph pipeline.

Mixin owns ``_build_triple_stages_comfyui`` (boot/rebuild) and
``_triple_stages_comfyui_generate`` (per-call denoise → encode). Single
mixin covers both T2V (empty ``images=[]``) and I2V.

Resolution must be divisible by **128** (the 4× downscale chain enforces this
for the 4× downscale chain). Sigma schedules and step counts are fixed by the
workflow's ManualSigmas literals (8/3/3 steps at sigmas 1.0 and 0.85).
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
    width: int, height: int, num_frames: int,
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
        width: int | None,
        height: int | None,
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
                from src.pipeline.inputs import (
                    derive_dims_from_image,
                    materialize_image,
                )

                image_path = materialize_image(image_url, image_b64)
                if width is None and height is None:
                    width, height = derive_dims_from_image(image_path)

            if width is None:
                width = 896
            if height is None:
                height = 1280
            width, height, num_frames = _round_user_inputs_comfyui(
                width, height, num_frames,
            )

            mode = "triple_comfyui_i2v" if has_image else "triple_comfyui_t2v"
            src_label = (
                "image_url"
                if image_url is not None
                else "image_b64" if image_b64 is not None else "none"
            )
            logger.info(
                "Job %s: %s (image_src=%s) prompt=%r, %dx%d, %d frames, "
                "fixed schedule (stage1=9/stage2=4/stage3=4 sigmas), seed=%d",
                job_id,
                mode.upper(),
                src_label,
                prompt[:80],
                width,
                height,
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
                height=height,
                width=width,
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
                    "width": width,
                    "height": height,
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
