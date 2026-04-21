"""BentoML service for LTX-2.3 text-to-video generation.

Endpoints:
    generate      — async task (POST /generate/submit, GET /status, /get)
    generate_sync — synchronous, returns MP4 directly

Built-in: /readyz, /healthz, /metrics, /docs
"""

import logging
import os
from pathlib import Path
from typing import Annotated

import bentoml
from pydantic import Field

from src import storage
from src.pipeline import DEFAULT_NEGATIVE_PROMPT, LTXVideoGenerator

logger = logging.getLogger(__name__)


@bentoml.service(
    name="ltx-video-generator",
    resources={"gpu": 1},
    # max_concurrency=1: the pipeline holds non-reentrant GPU state (HQQ-cached
    # Gemma, FP8 DiT weights, torch.compile graph captures). Allowing parallel
    # entry into generate() races over VRAM and crashes all in-flight jobs (see
    # pod logs 2026-04-21T05:23:22 — 3 simultaneous submissions OOM'd within
    # 5 s, one process held 44.39 GiB and the failed-pipeline cleanup path
    # didn't release it, so even a solo retry afterwards OOM'd). Bump only
    # after adding an in-process semaphore around generate().
    traffic={"timeout": 300, "max_concurrency": 1},
    workers=1,
)
class LTXVideoService:

    def __init__(self) -> None:
        model_dir = os.getenv("MODEL_DIR", "/models")
        self.generator = LTXVideoGenerator(model_dir=model_dir)

    @bentoml.task
    def generate(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: Annotated[int, Field(ge=256, le=1920)] = 1024,
        height: Annotated[int, Field(ge=256, le=1920)] = 1536,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 121,
        num_inference_steps: Annotated[int, Field(ge=1, le=100)] = 30,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
        upload_to_supabase: bool = True,
    ) -> dict:
        result = self.generator.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            num_inference_steps=num_inference_steps, seed=seed,
            frame_rate=frame_rate, cfg_scale=cfg_scale,
            stg_scale=stg_scale, rescale_scale=rescale_scale,
        )

        response = {
            "generation_time_seconds": result["generation_time_seconds"],
            "parameters": result["parameters"],
        }

        if upload_to_supabase and storage.is_configured():
            response["video_url"] = storage.upload_video(
                result["output_path"], result["output_filename"]
            )
            try:
                os.remove(result["output_path"])
            except OSError:
                pass
        else:
            response["video_path"] = result["output_path"]

        return response

    @bentoml.api
    def generate_sync(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: Annotated[int, Field(ge=256, le=1920)] = 1024,
        height: Annotated[int, Field(ge=256, le=1920)] = 1536,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 121,
        num_inference_steps: Annotated[int, Field(ge=1, le=100)] = 30,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
    ) -> Annotated[Path, bentoml.validators.ContentType("video/*")]:
        result = self.generator.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            num_inference_steps=num_inference_steps, seed=seed,
            frame_rate=frame_rate, cfg_scale=cfg_scale,
            stg_scale=stg_scale, rescale_scale=rescale_scale,
        )
        return Path(result["output_path"])
