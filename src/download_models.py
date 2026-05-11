import logging
import os

from huggingface_hub import hf_hub_download

from src.config import get_settings

logger = logging.getLogger(__name__)

# Dev FP8 checkpoint (~30 GB). ComfyUI handles FP8 natively.
DEV_FP8_FILENAME = "ltx-2.3-22b-dev-fp8.safetensors"
DEV_FP8_REPO = "Lightricks/LTX-2.3-fp8"

# Distilled LoRA — fused at strength 0.5 on all three stages by the workflow.
DISTILLED_LORA_FILENAME = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
DISTILLED_LORA_REPO = "Lightricks/LTX-2.3"


def _hf_get(repo_id: str, filename: str, model_dir: str, hf_token: str, label: str) -> None:
    path = os.path.join(model_dir, filename)
    if os.path.exists(path):
        logger.info("%s already cached.", label)
        return
    logger.info("Downloading %s ...", label)
    hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=model_dir,
        token=hf_token,
    )


def ensure_models_downloaded(model_dir: str) -> None:
    """Download every checkpoint the ComfyUI graph pipeline needs."""
    os.makedirs(model_dir, exist_ok=True)
    settings = get_settings()
    hf_token = settings.hf_token
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    # 1. Dev FP8 checkpoint (~30 GB).
    _hf_get(
        DEV_FP8_REPO,
        DEV_FP8_FILENAME,
        model_dir,
        hf_token,
        "LTX-2.3 dev FP8 checkpoint (~30 GB)",
    )

    # 2. Distilled LoRA (~7.6 GB).
    _hf_get(
        DISTILLED_LORA_REPO,
        DISTILLED_LORA_FILENAME,
        model_dir,
        hf_token,
        "Distilled LoRA (~7.6 GB)",
    )

    # 3. Spatial upscaler 2x (~1 GB).
    _hf_get(
        "Lightricks/LTX-2.3",
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        model_dir,
        hf_token,
        "Spatial upscaler (~1 GB)",
    )

    # 4. Gemma single-file text encoder for ComfyUI (~24 GB).
    _hf_get(
        "Comfy-Org/ltx-2",
        "split_files/text_encoders/gemma_3_12B_it.safetensors",
        model_dir,
        hf_token,
        "Gemma 3 12B text encoder for ComfyUI (~24 GB)",
    )

    logger.info("All models verified / downloaded to %s", model_dir)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_models_downloaded(get_settings().model_dir)
