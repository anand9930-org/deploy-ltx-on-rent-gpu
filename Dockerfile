# ============================================================================
# LTX-2.3 22B — BentoML video generation service
# ============================================================================
# Models are expected at $MODEL_DIR (default /models), typically mounted
# as a volume.  Downloaded on first boot if not already present (~64 GB).
# ============================================================================

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface

# ---- System dependencies + uv ----------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git-lfs gcc \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ---- Clone LTX-2 and install its packages ----------------------------------
# In-place patch: force use_fast=True on the Gemma image processor so the
# Rust-backed tokenizer runs instead of the slow Python fallback. Default
# in transformers >=4.52 anyway; forcing it here eliminates the
# "Using a slow image processor" warning and shaves a second or two off
# text encoding. Targeted sed; fails loudly if Lightricks ever refactors
# the call, which is how we'd want to know.
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && sed -i 's|AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)|AutoImageProcessor.from_pretrained(processor_root, local_files_only=True, use_fast=True)|' \
        /app/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py \
    && grep -q "use_fast=True" /app/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py \
    && uv pip install --system --no-cache \
        -e /app/LTX-2/packages/ltx-core \
        -e /app/LTX-2/packages/ltx-pipelines

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --no-cache /app

# ---- Install xformers for Ada-friendly attention ---------------------------
# LTX-2's AttentionFunction.DEFAULT picks XFormersAttention when xformers
# is importable (faster than the PytorchAttention / SDPA fallback for the
# BF16 non-causal shapes LTX-2 uses). Works on Ada sm_89 (L40S, RTX 6000
# Ada), Ampere, and Hopper. We deliberately do NOT install FA3 here:
# flash_attn_3 is Hopper-only and would shadow xformers even if present.
# SageAttention is a potential follow-up for INT8/FP8 attention on Ada.
RUN pip install --no-cache-dir xformers \
    && python3 -c "import xformers; print(f'xformers {xformers.__version__} loaded OK')"

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app
EXPOSE 8000

CMD ["/app/start.sh"]
