"""RunPod queue-based serverless handler for LTX-2.3 video generation.

Replaces the BentoML HTTP server for production deployment to queue-based
RunPod endpoints. The container's main process becomes
`python runpod_handler.py`, which registers with RunPod's queue and
executes jobs as they arrive.

Local dev can still use `bentoml serve service:LTXVideoService` against
service.py — both entrypoints share the same underlying
`LTXVideoGenerator`.
"""

import logging
import os

import runpod

from src import storage
from src.pipeline import DEFAULT_NEGATIVE_PROMPT, LTXVideoGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "/models")

logger.info("Initializing LTX pipeline (one-time cold start)...")
_GENERATOR = LTXVideoGenerator(model_dir=MODEL_DIR)
logger.info("Pipeline ready; worker is now accepting jobs from the queue.")


def handler(job: dict) -> dict:
    """RunPod job handler. `job['input']` carries the request params."""
    job_input = job.get("input") or {}

    prompt = job_input.get("prompt")
    if not prompt:
        return {"error": "missing required 'prompt' field in input"}

    try:
        result = _GENERATOR.generate(
            prompt=prompt,
            negative_prompt=job_input.get("negative_prompt", DEFAULT_NEGATIVE_PROMPT),
            width=int(job_input.get("width", 1024)),
            height=int(job_input.get("height", 1536)),
            num_frames=int(job_input.get("num_frames", 121)),
            num_inference_steps=int(job_input.get("num_inference_steps", 30)),
            seed=int(job_input.get("seed", 42)),
            frame_rate=float(job_input.get("frame_rate", 24.0)),
            cfg_scale=float(job_input.get("cfg_scale", 3.0)),
            stg_scale=float(job_input.get("stg_scale", 1.0)),
            rescale_scale=float(job_input.get("rescale_scale", 0.7)),
        )
    except Exception as exc:
        logger.exception("Job failed")
        return {"error": str(exc)}

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


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
