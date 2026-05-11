"""BentoML service for LTX-2.3 video generation (ComfyUI graph pipeline).

Endpoints: ``generate`` (async task), ``generate_sync`` (returns MP4).
Both run the 3-stage ComfyUI cascade (T2V or I2V depending on whether
an image is supplied).
"""

import logging
import os
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

load_dotenv()

import bentoml
from pydantic import Field

from src import storage
from src.config import get_settings
from src.pipeline import DEFAULT_NEGATIVE_PROMPT, LTXVideoGenerator

_settings = get_settings()

logging.basicConfig(
    level=_settings.log_level.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("src").setLevel(_settings.log_level.upper())

logger = logging.getLogger(__name__)


@bentoml.service(
    name="ltx-video-generator",
    resources={"gpu": 1},
    traffic={"timeout": 300, "max_concurrency": 3},
    workers=1,
)
class LTXVideoService:

    def __init__(self) -> None:
        self.generator = LTXVideoGenerator(model_dir=get_settings().model_dir)

    @bentoml.api
    def generate_sync(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: Annotated[int | None, Field(ge=256, le=1920)] = None,
        height: Annotated[int | None, Field(ge=256, le=1920)] = None,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 241,
        seed: int = 42,
        frame_rate: float = 24.0,
        image_url: Annotated[str | None, Field(max_length=2048)] = None,
        image_b64: Annotated[str | None, Field(max_length=70_000_000)] = None,
        image_frame_idx: Annotated[int, Field(ge=0)] = 0,
        enhance_prompt: bool = False,
    ) -> Annotated[Path, bentoml.validators.ContentType("video/*")]:
        """Run the 3-stage ComfyUI pipeline. T2V if no image, I2V if
        ``image_url``/``image_b64`` supplied.

        Resolution must be divisible by 128 (the 4× downscale chain enforces
        this); callers that supply mismatched dims will see them rounded down.

        Defaults mirror the workflow: ``num_frames=241``, ``frame_rate=24.0``,
        and width/height derived from the image (or 896×1280 fallback for T2V).
        """
        result = self.generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            seed=seed,
            frame_rate=frame_rate,
            image_url=image_url,
            image_b64=image_b64,
            image_frame_idx=image_frame_idx,
            enhance_prompt=enhance_prompt,
        )
        return Path(result["output_path"])

    @bentoml.task
    def generate(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: Annotated[int | None, Field(ge=256, le=1920)] = None,
        height: Annotated[int | None, Field(ge=256, le=1920)] = None,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 241,
        seed: int = 42,
        frame_rate: float = 24.0,
        image_url: Annotated[str | None, Field(max_length=2048)] = None,
        image_b64: Annotated[str | None, Field(max_length=70_000_000)] = None,
        image_frame_idx: Annotated[int, Field(ge=0)] = 0,
        enhance_prompt: bool = False,
        upload_to_supabase: bool = True,
    ) -> dict:
        """Async task variant of ``generate_sync`` — runs the 3-stage ComfyUI
        pipeline and (when configured) uploads the MP4 to Supabase, returning
        a signed URL. Poll ``/generate/status`` and fetch via ``/generate/get``
        once ``status=success``.
        """
        result = self.generator.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            num_frames=num_frames,
            seed=seed,
            frame_rate=frame_rate,
            image_url=image_url,
            image_b64=image_b64,
            image_frame_idx=image_frame_idx,
            enhance_prompt=enhance_prompt,
        )

        response = {
            "generation_time_seconds": result["generation_time_seconds"],
            "parameters": result["parameters"],
        }

        if upload_to_supabase and storage.is_configured():
            response["video_url"] = storage.upload_video(
                result["output_path"], result["output_filename"],
            )
            try:
                os.remove(result["output_path"])
            except OSError:
                pass
        else:
            response["video_path"] = result["output_path"]

        return response
