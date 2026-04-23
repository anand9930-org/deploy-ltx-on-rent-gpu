import os
import logging

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


def _fp8_mode() -> str:
    """Return the active FP8 mode. ``scaled_mm`` is the H100-optimised W8A8
    path; ``cast`` is the default W8A16 path used on Ada/Blackwell.

    The pipeline auto-detects H100 at runtime, but downloads happen at pod
    boot before the GPU is queried, so we key off ``LTX_FP8_MODE`` instead.
    """
    return os.getenv("LTX_FP8_MODE", "cast").strip().lower()


def ensure_models_downloaded(model_dir: str) -> None:
    """Download all required LTX-2.3 models to *model_dir* if not already present.

    On the default ``cast`` path we download the BF16 DiT checkpoint (~46 GB)
    and the distilled LoRA (~7.6 GB); runtime ``fp8_cast`` downcasts weights
    to FP8 at load time.

    On the ``scaled_mm`` path (H100) we additionally pull the pre-quantized
    FP8 DiT checkpoints (~58 GB) from ``Lightricks/LTX-2.3-fp8``. The BF16
    file is still required because VAE, audio decoder, image encoder and
    embeddings-processor weights live inside it and are loaded by other
    pipeline blocks via key filters.
    """
    os.makedirs(model_dir, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    fp8_mode = _fp8_mode()
    logger.info("LTX_FP8_MODE=%s", fp8_mode)

    # 1. LTX-2.3 BF16 checkpoint (~46 GB). Always downloaded: non-DiT
    # subcomponents (VAE encoder/decoder, audio decoder, vocoder, image
    # encoder, embeddings processor) are packaged inside this file and
    # loaded by PromptEncoder / ImageConditioner / VideoDecoder /
    # AudioDecoder / VideoUpsampler via *_COMFY_KEYS_FILTER state-dict ops.
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
        logger.info("LTX-2.3 BF16 checkpoint already cached.")

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

    # 3. Distilled LoRA (~7.6 GB). Required on the cast path (Stage 2
    # applies the LoRA on top of the BF16 DiT). On the scaled_mm path the
    # distilled weights ship pre-fused inside ltx-2.3-22b-distilled-fp8,
    # so we skip this download unless the user explicitly keeps it.
    lora_path = os.path.join(
        model_dir, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
    )
    if fp8_mode != "scaled_mm":
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
    else:
        logger.info("Skipping distilled LoRA (pre-fused inside distilled-fp8 checkpoint).")

    # 3b. Pre-quantized FP8 DiT checkpoints for the H100 scaled_mm path.
    # Weights-only repo (no config/tokenizer); per-tensor amax scales are
    # embedded as *_scale parameters inside each safetensors file.
    if fp8_mode == "scaled_mm":
        dev_fp8_path = os.path.join(model_dir, "ltx-2.3-22b-dev-fp8.safetensors")
        if not os.path.exists(dev_fp8_path):
            logger.info("Downloading LTX-2.3 dev FP8 DiT (~29 GB) ...")
            hf_hub_download(
                repo_id="Lightricks/LTX-2.3-fp8",
                filename="ltx-2.3-22b-dev-fp8.safetensors",
                local_dir=model_dir,
                token=hf_token,
            )
        else:
            logger.info("LTX-2.3 dev FP8 DiT already cached.")

        distilled_fp8_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-fp8.safetensors"
        )
        if not os.path.exists(distilled_fp8_path):
            logger.info("Downloading LTX-2.3 distilled FP8 DiT (~29 GB) ...")
            hf_hub_download(
                repo_id="Lightricks/LTX-2.3-fp8",
                filename="ltx-2.3-22b-distilled-fp8.safetensors",
                local_dir=model_dir,
                token=hf_token,
            )
        else:
            logger.info("LTX-2.3 distilled FP8 DiT already cached.")

    # 4. Gemma 3 12B text encoder (~26 GB, full snapshot)
    gemma_dir = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")
    gemma_has_weights = os.path.isdir(gemma_dir) and any(
        f.endswith(".safetensors")
        for f in os.listdir(gemma_dir)
        if os.path.isfile(os.path.join(gemma_dir, f))
    )
    if not gemma_has_weights:
        logger.info("Downloading Gemma 3 12B text encoder (~26 GB) ...")
        try:
            snapshot_download(
                repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
                local_dir=gemma_dir,
                token=hf_token,
            )
        except Exception as e:
            logger.error(
                "Failed to download Gemma 3: %s. "
                "You may need to accept the license at "
                "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized "
                "and wait for approval.",
                e,
            )
            raise
    else:
        logger.info("Gemma 3 text encoder already cached.")

    logger.info("All models verified / downloaded to %s", model_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    ensure_models_downloaded(os.getenv("MODEL_DIR", "/models"))
