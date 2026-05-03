import os
import logging

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


# Lightricks-official distilled-1.1 BF16 baseline. IC-LoRA Union-Control was
# trained against this exact distilled checkpoint, so applying the IC-LoRA at
# runtime on top of distilled-1.1 reproduces the trained configuration. This
# replaces the prior `dev BF16 + distilled-LoRA` two-LoRA stack and the
# third-party pre-fused checkpoint variant.
DISTILLED_CHECKPOINT_FILENAME = "ltx-2.3-22b-distilled-1.1.safetensors"
DISTILLED_CHECKPOINT_REPO = "Lightricks/LTX-2.3"

IC_LORA_FILENAME = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
IC_LORA_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"

# Pre-quantized distilled FP8 DiT (~30 GB). DiT-only — VAE / Gemma /
# image encoder / decoder / upsampler still come from the BF16 file via
# *_COMFY_KEYS_FILTER, so distilled-1.1 stays mandatory regardless.
DISTILLED_FP8_FILENAME = "ltx-2.3-22b-distilled-fp8.safetensors"
DISTILLED_FP8_REPO = "Lightricks/LTX-2.3-fp8"


def ensure_models_downloaded(model_dir: str) -> None:
    """Download all required LTX-2.3 models to *model_dir* if not already present.

    The pipeline is BF16-only on this branch:
      - Distilled-1.1 BF16 checkpoint (~46 GB): officially-released distilled
        baseline. Packs DiT + VAE encoder/decoder, audio decoder, vocoder,
        image encoder, and the embeddings processor; every non-DiT block reads
        its slice via *_COMFY_KEYS_FILTER.
      - Spatial upsampler (~1 GB): used by VideoUpsampler between stages.
      - IC-LoRA Union-Control (~few GB): provides the
        ``VideoConditionByReferenceLatent`` cross-attention deltas that keep
        I2V identity / V2V structure aligned to the reference. Applied at
        runtime as the single LoRA fusion on stage 1.
      - Gemma 3 12B text encoder (~26 GB).
    """
    os.makedirs(model_dir, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    # 1. LTX-2.3 distilled-1.1 BF16 checkpoint (~46 GB). Holds DiT + every
    # non-DiT block (VAE, audio decoder, vocoder, image encoder, embeddings).
    checkpoint_path = os.path.join(model_dir, DISTILLED_CHECKPOINT_FILENAME)
    if not os.path.exists(checkpoint_path):
        logger.info("Downloading LTX-2.3 distilled-1.1 BF16 checkpoint (~46 GB) ...")
        hf_hub_download(
            repo_id=DISTILLED_CHECKPOINT_REPO,
            filename=DISTILLED_CHECKPOINT_FILENAME,
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("LTX-2.3 distilled-1.1 BF16 checkpoint already cached.")

    # 2. Spatial upscaler 2x (~1 GB).
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

    # 3. IC-LoRA Union-Control. Provides positionally-aligned reference-token
    # cross-attention; consumed by ICLoraPipeline stage 1 to keep identity /
    # structure aligned to the supplied image (I2V) or reference video (V2V).
    # Applied at runtime as the single LoRA on top of the distilled-1.1 base.
    ic_lora_path = os.path.join(model_dir, IC_LORA_FILENAME)
    if not os.path.exists(ic_lora_path):
        logger.info("Downloading IC-LoRA Union-Control ...")
        hf_hub_download(
            repo_id=IC_LORA_REPO,
            filename=IC_LORA_FILENAME,
            local_dir=model_dir,
            token=hf_token,
        )
    else:
        logger.info("IC-LoRA Union-Control already cached.")

    # 4. Pre-quantized distilled FP8 DiT (~30 GB) — only when the runtime is
    # configured to load it. LTX_FP8_MODE=scaled_mm|cast at boot enables the
    # FP8 path in pipeline.py; downloading is gated on the same flag so BF16
    # pods don't pay the bandwidth/disk cost. The FP8 file is DiT-only — every
    # non-DiT block still reads its slice from the BF16 distilled-1.1 file
    # above via *_COMFY_KEYS_FILTER, so this download is purely additive.
    fp8_mode = os.environ.get("LTX_FP8_MODE", "").strip().lower()
    if fp8_mode in ("scaled_mm", "cast"):
        fp8_path = os.path.join(model_dir, DISTILLED_FP8_FILENAME)
        if not os.path.exists(fp8_path):
            logger.info(
                "Downloading LTX-2.3 distilled FP8 DiT (~30 GB) for LTX_FP8_MODE=%s ...",
                fp8_mode,
            )
            hf_hub_download(
                repo_id=DISTILLED_FP8_REPO,
                filename=DISTILLED_FP8_FILENAME,
                local_dir=model_dir,
                token=hf_token,
            )
        else:
            logger.info("LTX-2.3 distilled FP8 DiT already cached.")

    # 5. Gemma 3 12B text encoder (~26 GB, full snapshot)
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
