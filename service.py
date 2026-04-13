"""BentoML service for LTX-2.3 text-to-video generation.

Single synchronous endpoint:
    generate — runs the full pipeline in one HTTP request and returns
               a JSON response containing the Supabase video URL.

Built-in BentoML routes: /readyz, /healthz, /metrics, /docs

The previous @bentoml.task async pattern was removed because BentoML's
task state is stored per-worker in memory, which breaks when RunPod's
load balancer routes /generate/status to a different worker than the one
that handled /generate/submit. A single long-lived HTTP request stays
pinned to one worker and also avoids the per-poll billing overhead on
RunPod's pay-per-second model.

Note: RunPod serverless /ping health check runs on a separate lightweight
HTTP server on port 8001 (see src/health_server.py, launched by start.sh).
BentoML's @bentoml.api decorator only creates POST endpoints, which is
incompatible with RunPod's GET /ping requirement.
"""

import logging
import os
from typing import Annotated

import bentoml
from pydantic import Field

from src import storage
from src.pipeline import DEFAULT_NEGATIVE_PROMPT, LTXVideoGenerator

logger = logging.getLogger(__name__)


@bentoml.service(
    name="ltx-video-generator",
    resources={"gpu": 1},
    traffic={"timeout": 300, "max_concurrency": 1},
    workers=1,
)
class LTXVideoService:

    def __init__(self) -> None:
        model_dir = os.getenv("MODEL_DIR", "/models")
        self.generator = LTXVideoGenerator(model_dir=model_dir)

    @bentoml.api
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

        if storage.is_configured():
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
