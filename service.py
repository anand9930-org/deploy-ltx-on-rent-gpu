"""BentoML service for LTX-2.3 unified video generation (T2V / I2V / V2V).

Mode is selected by which inputs the caller supplies (Veo-style):
    prompt only                                    → T2V
    prompt + image_url|image_b64                   → I2V (identity-strict via IC-LoRA)
    prompt + reference_video_url|reference_video_b64 (± image)
                                                   → V2V (style transfer / edit)

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

# BentoML leaves the root logger at WARNING by default, which suppresses the
# INFO lines our pipeline emits for feature activation, VRAM, timing, and
# TeaCache stats. Configure once at import time so pod logs actually show
# them. Respect LOG_LEVEL so operators can dial it up/down without a rebuild.
_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("src").setLevel(_level)

logger = logging.getLogger(__name__)


@bentoml.service(
    name="ltx-video-generator",
    resources={"gpu": 1},
    traffic={"timeout": 300, "max_concurrency": 3},
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
        width: Annotated[int | None, Field(ge=256, le=1920)] = None,
        height: Annotated[int | None, Field(ge=256, le=1920)] = None,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 121,
        seed: int = 42,
        frame_rate: float = 24.0,
        image_url: Annotated[str | None, Field(max_length=2048)] = None,
        image_b64: Annotated[str | None, Field(max_length=70_000_000)] = None,
        reference_video_url: Annotated[str | None, Field(max_length=2048)] = None,
        reference_video_b64: Annotated[str | None, Field(max_length=300_000_000)] = None,
        reference_video_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0,
        conditioning_attention_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0,
        enhance_prompt: bool = False,
        upload_to_supabase: bool = True,
    ) -> dict:
        result = self.generator.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            seed=seed, frame_rate=frame_rate,
            image_url=image_url, image_b64=image_b64,
            reference_video_url=reference_video_url,
            reference_video_b64=reference_video_b64,
            reference_video_strength=reference_video_strength,
            conditioning_attention_strength=conditioning_attention_strength,
            enhance_prompt=enhance_prompt,
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
        width: Annotated[int | None, Field(ge=256, le=1920)] = None,
        height: Annotated[int | None, Field(ge=256, le=1920)] = None,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 121,
        seed: int = 42,
        frame_rate: float = 24.0,
        image_url: Annotated[str | None, Field(max_length=2048)] = None,
        image_b64: Annotated[str | None, Field(max_length=70_000_000)] = None,
        reference_video_url: Annotated[str | None, Field(max_length=2048)] = None,
        reference_video_b64: Annotated[str | None, Field(max_length=300_000_000)] = None,
        reference_video_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0,
        conditioning_attention_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0,
        enhance_prompt: bool = False,
    ) -> Annotated[Path, bentoml.validators.ContentType("video/*")]:
        result = self.generator.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            seed=seed, frame_rate=frame_rate,
            image_url=image_url, image_b64=image_b64,
            reference_video_url=reference_video_url,
            reference_video_b64=reference_video_b64,
            reference_video_strength=reference_video_strength,
            conditioning_attention_strength=conditioning_attention_strength,
            enhance_prompt=enhance_prompt,
        )
        return Path(result["output_path"])
