import os
import logging

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


# Gemma checkpoint choice is driven by GEMMA_QUANT (see src/pipeline.py).
# w4a16 is a community GPTQ-quantized checkpoint with published ~98% parity;
# bf16 is the official Google QAT-dequantized copy.
_GEMMA_REPOS: dict[str, tuple[str, str, str]] = {
    # quant : (repo_id, local_dir_name, approx_size_label)
    "bf16":  ("google/gemma-3-12b-it-qat-q4_0-unquantized",
              "gemma-3-12b-it-qat-q4_0-unquantized", "~26 GB"),
    "w4a16": ("RedHatAI/gemma-3-12b-it-quantized.w4a16",
              "gemma-3-12b-it-w4a16", "~7 GB"),
}


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
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    gemma_quant = os.getenv("GEMMA_QUANT", "w4a16").strip().lower()
    if gemma_quant not in _GEMMA_REPOS:
        logger.warning("Unknown GEMMA_QUANT=%r; falling back to w4a16", gemma_quant)
        gemma_quant = "w4a16"
    gemma_repo_id, gemma_dir_name, gemma_size = _GEMMA_REPOS[gemma_quant]

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

    # 4. Gemma 3 12B text encoder. Checkpoint selected by GEMMA_QUANT:
    #    w4a16 (default) -> RedHatAI GPTQ, ~7 GB, weight-only (BF16 activations)
    #    bf16            -> Google QAT-dequantized, ~26 GB, reference baseline
    gemma_dir = os.path.join(model_dir, gemma_dir_name)
    gemma_has_weights = os.path.isdir(gemma_dir) and any(
        f.endswith(".safetensors")
        for f in os.listdir(gemma_dir)
        if os.path.isfile(os.path.join(gemma_dir, f))
    )
    if not gemma_has_weights:
        logger.info(
            "Downloading Gemma 3 12B text encoder [%s] (%s) from %s ...",
            gemma_quant, gemma_size, gemma_repo_id,
        )
        try:
            snapshot_download(
                repo_id=gemma_repo_id,
                local_dir=gemma_dir,
                token=hf_token,
            )
        except Exception as e:
            logger.error(
                "Failed to download Gemma 3 [%s] from %s: %s. "
                "For the Google repo you may need to accept the license at "
                "https://huggingface.co/%s and wait for approval.",
                gemma_quant, gemma_repo_id, e, gemma_repo_id,
            )
            raise
    else:
        logger.info("Gemma 3 text encoder [%s] already cached at %s", gemma_quant, gemma_dir)

    # LTX-2's loader hardcodes a SentencePiece `tokenizer.model` lookup
    # (ltx_core/text_encoders/gemma/encoders/base_encoder.py:178).
    # Community-quantized Gemma repos (RedHatAI W4A16 etc.) ship only the
    # fast `tokenizer.json`, so fetch the SentencePiece binary from Google's
    # canonical repo and drop it in. Same vocab, same license gate.
    tokenizer_model_path = os.path.join(gemma_dir, "tokenizer.model")
    if not os.path.exists(tokenizer_model_path):
        logger.info(
            "tokenizer.model missing in %s — pulling from google/gemma-3-12b-it-qat-q4_0-unquantized",
            gemma_dir,
        )
        try:
            hf_hub_download(
                repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                filename="tokenizer.model",
                local_dir=gemma_dir,
                token=hf_token,
            )
        except Exception as e:
            logger.error(
                "Failed to fetch tokenizer.model from Google repo: %s. "
                "LTX-2 requires this SentencePiece file; accept the license at "
                "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized",
                e,
            )
            raise

    logger.info("All models verified / downloaded to %s", model_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_models_downloaded(os.getenv("MODEL_DIR", "/models"))
