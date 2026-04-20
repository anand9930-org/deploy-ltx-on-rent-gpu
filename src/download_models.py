import os
import logging

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


# Gemma source is always the Google QAT-dequantized BF16 checkpoint. Any
# runtime weight reduction happens post-load (see src/pipeline.py post-load
# quant walker) so the on-disk format stays compatible with LTX-2's custom
# safetensors loader.
_GEMMA_REPO_ID = "google/gemma-3-12b-it-qat-q4_0-unquantized"
_GEMMA_DIR_NAME = "gemma-3-12b-it-qat-q4_0-unquantized"


def ensure_models_downloaded(model_dir: str) -> None:
    """Download all required LTX-2.3 models to *model_dir* if not already present.

    Uses the official FP8 checkpoint (~29 GB) for lower VRAM usage.
    Downloads are idempotent -- existing files are skipped.
    """
    os.makedirs(model_dir, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            f"https://huggingface.co/{_GEMMA_REPO_ID}"
        )

    # 1. LTX-2.3 BF16 checkpoint (~46 GB, runtime fp8_cast downcasts on the fly)
    checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-dev.safetensors")
    if not os.path.exists(checkpoint_path):
        logger.info("Downloading LTX-2.3 BF16 checkpoint (~46 GB) ...")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-22b-dev.safetensors",
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("LTX-2.3 checkpoint already cached.")

    # 2. Spatial upscaler 2x (~1 GB)
    upscaler_path = os.path.join(
        model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
    )
    if not os.path.exists(upscaler_path):
        logger.info("Downloading spatial upscaler (~1 GB) ...")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("Spatial upscaler already cached.")

    # 3. Distilled LoRA (~7.6 GB, compatible with FP8 checkpoint)
    lora_path = os.path.join(
        model_dir, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    )
    if not os.path.exists(lora_path):
        logger.info("Downloading distilled LoRA (~7.6 GB) ...")
        hf_hub_download(
            repo_id="Lightricks/LTX-2.3",
            filename="ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("Distilled LoRA already cached.")

    # 4. Gemma 3 12B text encoder (~26 GB BF16, Google QAT-dequantized).
    gemma_dir = os.path.join(model_dir, _GEMMA_DIR_NAME)
    gemma_has_weights = os.path.isdir(gemma_dir) and any(
        f.endswith(".safetensors")
        for f in os.listdir(gemma_dir)
        if os.path.isfile(os.path.join(gemma_dir, f))
    )
    if not gemma_has_weights:
        logger.info("Downloading Gemma 3 12B text encoder (~26 GB) from %s ...", _GEMMA_REPO_ID)
        try:
            snapshot_download(
                repo_id=_GEMMA_REPO_ID,
                local_dir=gemma_dir,
                token=hf_token,
            )
        except Exception as e:
            logger.error(
                "Failed to download Gemma 3 from %s: %s. "
                "Accept the license at https://huggingface.co/%s and wait for approval.",
                _GEMMA_REPO_ID, e, _GEMMA_REPO_ID,
            )
            raise
    else:
        logger.info("Gemma 3 text encoder already cached at %s", gemma_dir)

    # Safety net for the rare case the snapshot lacks tokenizer.model (some
    # upstream mirrors have dropped the SentencePiece binary). LTX-2's loader
    # requires it (ltx_core/text_encoders/gemma/encoders/base_encoder.py:178).
    tokenizer_model_path = os.path.join(gemma_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_model_path):
        logger.info("tokenizer.model missing in %s — pulling it explicitly", gemma_dir)
        hf_hub_download(
            repo_id=_GEMMA_REPO_ID,
            filename="tokenizer.model",
            local_dir=gemma_dir,
            token=hf_token,
        )

    logger.info("All models verified / downloaded to %s", model_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_models_downloaded(os.getenv("MODEL_DIR", "/models"))
