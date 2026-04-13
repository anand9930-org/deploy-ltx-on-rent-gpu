# ============================================================================
# LTX-2.3 22B — BentoML video generation service (models baked in)
# ============================================================================
# Single-stage build: downloads models directly into the runtime image
# to avoid multi-stage COPY disk overhead on constrained CI runners.
#
# Build command:
#   DOCKER_BUILDKIT=1 docker build --build-arg HF_TOKEN=$HF_TOKEN \
#       -t anand9930/ltx-video:latest .
# ============================================================================

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ARG HF_TOKEN

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface \
    MODEL_DIR=/models

# ---- System dependencies + uv ----------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git-lfs gcc \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ---- Install download tool (lightweight, for model baking) ------------------
RUN uv pip install --system --no-cache "huggingface-hub[hf_xet]"

# ---- Baked-in models — one RUN per file so BuildKit can compress,
# push, and cache each layer independently. Largest file first because
# the layer cache is linear: invalidating an earlier layer forces every
# later one to rebuild, so the most-stable (biggest) file goes on top.

# LTX-2.3 22B BF16 dev checkpoint (~46 GB)
RUN hf download Lightricks/LTX-2.3 ltx-2.3-22b-dev.safetensors \
        --local-dir /models \
    && rm -rf /models/.cache

# LTX-2.3 distilled LoRA (~7.6 GB)
RUN hf download Lightricks/LTX-2.3 ltx-2.3-22b-distilled-lora-384.safetensors \
        --local-dir /models \
    && rm -rf /models/.cache

# LTX-2.3 spatial upscaler (~1 GB)
RUN hf download Lightricks/LTX-2.3 ltx-2.3-spatial-upscaler-x2-1.1.safetensors \
        --local-dir /models \
    && rm -rf /models/.cache

# Gemma-3 12B text encoder (~24 GB, license-gated)
RUN hf download google/gemma-3-12b-it-qat-q4_0-unquantized \
        --token "$HF_TOKEN" \
        --local-dir /models/gemma-3-12b-it-qat-q4_0-unquantized \
    && rm -rf /models/gemma-3-12b-it-qat-q4_0-unquantized/.cache

# ---- Clone LTX-2 and install its packages ----------------------------------
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && uv pip install --system --no-cache \
        -e /app/LTX-2/packages/ltx-core \
        -e /app/LTX-2/packages/ltx-pipelines

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --no-cache /app

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY runpod_handler.py /app/runpod_handler.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app
# 8000 = BentoML service, 8001 = lightweight health server for RunPod /ping
EXPOSE 8000 8001

CMD ["/app/start.sh"]
