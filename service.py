"""BentoML service for LTX-2.3 unified video generation (T2V / I2V / V2V).

Mode is selected Veo-style by which inputs the caller supplies (prompt-only
→ T2V; +image → I2V; +reference_video → V2V). Cross-mode requests pay a
~30-60 s pipeline rebuild — H100 80 GB can't hold both upstream pipelines
resident; ``LTX_DEFAULT_MODE`` picks the boot-preloaded side.

Endpoints: ``generate`` (async task), ``generate_sync`` (returns MP4),
``generate_triple_stages_sync`` (vendored 3-stage pipeline; T2V or I2V).
"""

import logging
import os
from pathlib import Path
from typing import Annotated

# python-dotenv populates os.environ from .env (if present) BEFORE any
# src.* import — load-bearing because src/pipeline.py runs a module-level
# read of LTX_FP8_MODE at import time to decide whether to set
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments. In production .env does
# not exist; load_dotenv() is a silent no-op and env vars come from the
# RunPod pod template / GitHub Actions secrets.
from dotenv import load_dotenv

load_dotenv()

import bentoml
from pydantic import Field

from src import storage
from src.config import get_settings
from src.pipeline import DEFAULT_NEGATIVE_PROMPT, LTXVideoGenerator
from src.vendor.ti2vid_triple_stages_comfyui import COMFY_DEFAULT_NEGATIVE_PROMPT

_settings = get_settings()

# BentoML leaves the root logger at WARNING by default, which suppresses the
# INFO lines our pipeline emits for feature activation, VRAM, timing, and
# TeaCache stats. Configure once at import time so pod logs actually show
# them. Respect LOG_LEVEL so operators can dial it up/down without a rebuild.
_level = _settings.log_level.upper()
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
        self.generator = LTXVideoGenerator(model_dir=get_settings().model_dir)

    @bentoml.task
    def generate(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: Annotated[int | None, Field(ge=256, le=1920)] = None,
        height: Annotated[int | None, Field(ge=256, le=1920)] = None,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 121,
        num_inference_steps: Annotated[int, Field(ge=1, le=100)] = 30,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: Annotated[float, Field(ge=0.0, le=20.0)] = 3.0,
        stg_scale: Annotated[float, Field(ge=0.0, le=10.0)] = 1.0,
        rescale_scale: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7,
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
            num_inference_steps=num_inference_steps,
            seed=seed, frame_rate=frame_rate,
            cfg_scale=cfg_scale, stg_scale=stg_scale,
            rescale_scale=rescale_scale,
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
        num_inference_steps: Annotated[int, Field(ge=1, le=100)] = 30,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: Annotated[float, Field(ge=0.0, le=20.0)] = 3.0,
        stg_scale: Annotated[float, Field(ge=0.0, le=10.0)] = 1.0,
        rescale_scale: Annotated[float, Field(ge=0.0, le=1.0)] = 0.7,
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
            num_inference_steps=num_inference_steps,
            seed=seed, frame_rate=frame_rate,
            cfg_scale=cfg_scale, stg_scale=stg_scale,
            rescale_scale=rescale_scale,
            image_url=image_url, image_b64=image_b64,
            reference_video_url=reference_video_url,
            reference_video_b64=reference_video_b64,
            reference_video_strength=reference_video_strength,
            conditioning_attention_strength=conditioning_attention_strength,
            enhance_prompt=enhance_prompt,
        )
        return Path(result["output_path"])

    @bentoml.api
    def generate_triple_stages_sync(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: Annotated[int | None, Field(ge=256, le=1920)] = None,
        height: Annotated[int | None, Field(ge=256, le=1920)] = None,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 121,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: Annotated[float, Field(ge=0.0, le=20.0)] = 1.0,
        stg_scale: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0,
        rescale_scale: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
        image_url: Annotated[str | None, Field(max_length=2048)] = None,
        image_b64: Annotated[str | None, Field(max_length=70_000_000)] = None,
        image_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0,
        image_frame_idx: Annotated[int, Field(ge=0)] = 0,
        stage1_steps: Annotated[int, Field(ge=1, le=64)] = 16,
        stage2_steps: Annotated[int, Field(ge=1, le=32)] = 8,
        enhance_prompt: bool = False,
    ) -> Annotated[Path, bentoml.validators.ContentType("video/*")]:
        """Run the vendored ``TI2VidTripleStagesPipeline`` (T2V if no image,
        I2V if ``image_url``/``image_b64`` supplied). Pays a pipeline rebuild
        on first call if the boot-preloaded mode is different.

        Defaults match the vendored class: stage1=16 / stage2=8 sigma steps,
        cfg=1.0 / stg=0.0 / rescale=0.0 (image conditioning drives output;
        the file's docstring notes "cfg=1, minimal CFG"). For T2V the caller
        usually wants higher cfg (e.g. 3.0); image_url is required for I2V.
        """
        result = self.generator.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            seed=seed, frame_rate=frame_rate,
            cfg_scale=cfg_scale, stg_scale=stg_scale, rescale_scale=rescale_scale,
            image_url=image_url, image_b64=image_b64,
            enhance_prompt=enhance_prompt,
            pipeline_variant="triple_stages",
            stage1_steps=stage1_steps, stage2_steps=stage2_steps,
            image_strength=image_strength, image_frame_idx=image_frame_idx,
        )
        return Path(result["output_path"])

    @bentoml.task
    def generate_triple_stages(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: Annotated[int | None, Field(ge=256, le=1920)] = None,
        height: Annotated[int | None, Field(ge=256, le=1920)] = None,
        num_frames: Annotated[int, Field(ge=9, le=257)] = 121,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: Annotated[float, Field(ge=0.0, le=20.0)] = 1.0,
        stg_scale: Annotated[float, Field(ge=0.0, le=10.0)] = 0.0,
        rescale_scale: Annotated[float, Field(ge=0.0, le=1.0)] = 0.0,
        image_url: Annotated[str | None, Field(max_length=2048)] = None,
        image_b64: Annotated[str | None, Field(max_length=70_000_000)] = None,
        image_strength: Annotated[float, Field(ge=0.0, le=1.0)] = 1.0,
        image_frame_idx: Annotated[int, Field(ge=0)] = 0,
        stage1_steps: Annotated[int, Field(ge=1, le=64)] = 16,
        stage2_steps: Annotated[int, Field(ge=1, le=32)] = 8,
        enhance_prompt: bool = False,
        upload_to_supabase: bool = True,
    ) -> dict:
        """Async task variant of ``generate_triple_stages_sync`` — runs the
        vendored ``TI2VidTripleStagesPipeline`` and (when configured) uploads
        the MP4 to Supabase, returning a signed URL. Mirrors the existing
        ``generate(...)`` task contract: poll ``/generate_triple_stages/status``
        and fetch via ``/generate_triple_stages/get`` once ``status=success``.
        """
        result = self.generator.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            seed=seed, frame_rate=frame_rate,
            cfg_scale=cfg_scale, stg_scale=stg_scale, rescale_scale=rescale_scale,
            image_url=image_url, image_b64=image_b64,
            enhance_prompt=enhance_prompt,
            pipeline_variant="triple_stages",
            stage1_steps=stage1_steps, stage2_steps=stage2_steps,
            image_strength=image_strength, image_frame_idx=image_frame_idx,
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
    def generate_triple_stages_comfyui_sync(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = COMFY_DEFAULT_NEGATIVE_PROMPT,
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
        """Run the vendored ``TI2VidTripleStagesComfyUIPipeline`` (workflow
        port of ``scripts/workflow_3mljpp.py``). T2V if no image, I2V if
        ``image_url``/``image_b64`` supplied. Pays a pipeline rebuild on
        first call if the boot-preloaded mode is different.

        Sigmas + cfg are baked into the vendored class (cfg=1 →
        SimpleDenoiser; manual sigma schedules at 9/4/4 steps), so this
        endpoint deliberately does not expose ``cfg_scale``/``stg_scale``/
        ``rescale_scale``/``stage1_steps``/``stage2_steps``. Resolution
        must be divisible by 128 (the 4× downscale chain enforces this);
        callers that supply mismatched dims will see them rounded down at
        the dispatcher.

        Defaults mirror the workflow: ``num_frames=241`` (matches the
        workflow's ``LTXVAddGuide(num_frames=241)``), ``frame_rate=24.0``,
        and width/height derived from the image (or 896×1280 fallback for
        T2V — the workflow's hardcoded ``EmptyLTXVLatentVideo(224×320)``
        × 4× chain).
        """
        result = self.generator.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            seed=seed, frame_rate=frame_rate,
            image_url=image_url, image_b64=image_b64,
            enhance_prompt=enhance_prompt,
            pipeline_variant="triple_stages_comfyui",
            image_frame_idx=image_frame_idx,
        )
        return Path(result["output_path"])

    @bentoml.task
    def generate_triple_stages_comfyui(
        self,
        prompt: Annotated[str, Field(max_length=2000)],
        negative_prompt: str = COMFY_DEFAULT_NEGATIVE_PROMPT,
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
        """Async task variant of ``generate_triple_stages_comfyui_sync`` —
        runs the vendored ``TI2VidTripleStagesComfyUIPipeline`` and (when
        configured) uploads the MP4 to Supabase, returning a signed URL.
        Mirrors the existing ``generate(...)`` task contract: poll
        ``/generate_triple_stages_comfyui/status`` and fetch via
        ``/generate_triple_stages_comfyui/get`` once ``status=success``.
        """
        result = self.generator.generate(
            prompt=prompt, negative_prompt=negative_prompt,
            width=width, height=height, num_frames=num_frames,
            seed=seed, frame_rate=frame_rate,
            image_url=image_url, image_b64=image_b64,
            enhance_prompt=enhance_prompt,
            pipeline_variant="triple_stages_comfyui",
            image_frame_idx=image_frame_idx,
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
