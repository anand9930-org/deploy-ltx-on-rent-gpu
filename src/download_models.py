import os
import logging

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)


# T2V (TI2VidTwoStagesPipeline) baseline. Required regardless of FP8 mode —
# stage 1 uses dev BF16 directly (cast path) and the BF16 file also packs
# every non-DiT block (VAE, audio decoder, vocoder, image encoder, embeddings
# processor) loaded by other pipeline blocks via *_COMFY_KEYS_FILTER, so
# scaled_mm still needs it for non-DiT subcomponents.
DEV_CHECKPOINT_FILENAME = "ltx-2.3-22b-dev.safetensors"
DEV_CHECKPOINT_REPO = "Lightricks/LTX-2.3"

# Distilled LoRA used on T2V cast/bf16 path (stage 2 fuses it on top of dev
# BF16 to recover distilled quality without the FP8 quantisation step).
# Skipped on the scaled_mm path — the pre-quantised distilled-fp8 file
# already ships the LoRA fused inside.
DISTILLED_LORA_FILENAME = "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
DISTILLED_LORA_REPO = "Lightricks/LTX-2.3"

# Pre-quantized dev FP8 DiT (~30 GB). Stage 1 of the T2V scaled_mm path.
DEV_FP8_FILENAME = "ltx-2.3-22b-dev-fp8.safetensors"
DEV_FP8_REPO = "Lightricks/LTX-2.3-fp8"

# Lightricks-official distilled-1.1 BF16 baseline. IC-LoRA Union-Control was
# trained against this exact distilled checkpoint, so applying the IC-LoRA at
# runtime on top of distilled-1.1 reproduces the trained configuration. Used
# by the unified (I2V/V2V) pipeline only.
DISTILLED_CHECKPOINT_FILENAME = "ltx-2.3-22b-distilled-1.1.safetensors"
DISTILLED_CHECKPOINT_REPO = "Lightricks/LTX-2.3"

IC_LORA_FILENAME = "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
IC_LORA_REPO = "Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control"

# Pre-quantized distilled FP8 DiT (~30 GB). Shared between T2V stage 2 and
# unified stages 1+2 in scaled_mm mode. DiT-only — non-DiT blocks still
# load from the BF16 files via *_COMFY_KEYS_FILTER.
DISTILLED_FP8_FILENAME = "ltx-2.3-22b-distilled-fp8.safetensors"
DISTILLED_FP8_REPO = "Lightricks/LTX-2.3-fp8"


def _hf_get(repo_id: str, filename: str, model_dir: str, hf_token: str, label: str) -> None:
    path = os.path.join(model_dir, filename)
    if os.path.exists(path):
        logger.info("%s already cached.", label)
        return
    logger.info("Downloading %s ...", label)
    hf_hub_download(
        repo_id=repo_id, filename=filename,
        local_dir=model_dir, token=hf_token,
    )


def ensure_models_downloaded(model_dir: str) -> None:
    """Download all required LTX-2.3 models to *model_dir* if not already present.

    The wrapper dispatches across two upstream pipelines (TI2VidTwoStagesPipeline
    for T2V; ICLoraPipeline for I2V/V2V), so we materialise both asset families:

      - dev BF16 (~46 GB): T2V stage 1 (cast) and source for non-DiT blocks
        (VAE, audio decoder, vocoder, image encoder, embeddings processor).
      - distilled-LoRA-384 (~7.6 GB): T2V cast-path stage 2 LoRA. Skipped on
        scaled_mm (LoRA is pre-fused inside distilled-fp8).
      - dev-fp8 (~30 GB, scaled_mm only): T2V stage 1 quantised.
      - distilled-1.1 BF16 (~46 GB): Unified pipeline base. IC-LoRA was
        trained against this exact checkpoint.
      - spatial upsampler (~1 GB): used by VideoUpsampler between stages.
      - IC-LoRA Union-Control: cross-attention deltas keeping I2V identity /
        V2V structure aligned to the reference image / video.
      - distilled-fp8 (~30 GB, scaled_mm/cast only): shared FP8 DiT — T2V
        stage 2 and unified stages 1+2.
      - Gemma 3 12B text encoder (~26 GB).
    """
    os.makedirs(model_dir, exist_ok=True)
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN env var required. Gemma 3 model needs license acceptance at "
            "https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized"
        )

    fp8_mode = os.environ.get("LTX_FP8_MODE", "").strip().lower()

    # 1. dev BF16 — T2V stage 1 base (cast) + non-DiT blocks for both pipelines.
    _hf_get(
        DEV_CHECKPOINT_REPO, DEV_CHECKPOINT_FILENAME, model_dir, hf_token,
        "LTX-2.3 dev BF16 checkpoint (~46 GB)",
    )

    # 2. distilled-LoRA — T2V stage 2 LoRA on cast/bf16. Skip on scaled_mm
    # (LoRA is already fused inside the pre-quantised distilled-fp8 file).
    if fp8_mode != "scaled_mm":
        _hf_get(
            DISTILLED_LORA_REPO, DISTILLED_LORA_FILENAME, model_dir, hf_token,
            "T2V distilled LoRA (~7.6 GB)",
        )

    # 3. dev-fp8 — T2V stage 1 quantised. scaled_mm only.
    if fp8_mode == "scaled_mm":
        _hf_get(
            DEV_FP8_REPO, DEV_FP8_FILENAME, model_dir, hf_token,
            "LTX-2.3 dev FP8 DiT (~30 GB)",
        )

    # 4. distilled-1.1 BF16 — unified pipeline base (IC-LoRA was trained
    # against this exact checkpoint).
    _hf_get(
        DISTILLED_CHECKPOINT_REPO, DISTILLED_CHECKPOINT_FILENAME, model_dir, hf_token,
        "LTX-2.3 distilled-1.1 BF16 checkpoint (~46 GB)",
    )

    # 5. Spatial upscaler 2x (~1 GB).
    _hf_get(
        "Lightricks/LTX-2.3", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        model_dir, hf_token, "Spatial upscaler (~1 GB)",
    )

    # 6. IC-LoRA Union-Control — unified pipeline stage 1 LoRA (I2V/V2V).
    _hf_get(
        IC_LORA_REPO, IC_LORA_FILENAME, model_dir, hf_token,
        "IC-LoRA Union-Control",
    )

    # 7. Distilled FP8 DiT — shared by T2V stage 2 and unified stages on FP8.
    if fp8_mode in ("scaled_mm", "cast"):
        _hf_get(
            DISTILLED_FP8_REPO, DISTILLED_FP8_FILENAME, model_dir, hf_token,
            f"LTX-2.3 distilled FP8 DiT (~30 GB) for LTX_FP8_MODE={fp8_mode}",
        )

    # 8. Gemma 3 12B text encoder (~26 GB, full snapshot).
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
