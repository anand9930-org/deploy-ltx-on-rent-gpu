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
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && uv pip install --system --no-cache \
        -e /app/LTX-2/packages/ltx-core \
        -e /app/LTX-2/packages/ltx-pipelines

# ---- SageAttention (Hopper sm_90 FP8 kernel auto-dispatch) -----------------
# Prebuilt wheel from Comfy-Org/wheels — cu128 / torch2.8 / cp311 / manylinux2_34.
# thu-ml upstream doesn't ship v2.x wheels on PyPI; this is the ABI-matched drop-in.
RUN uv pip install --system --no-cache \
    "https://github.com/Comfy-Org/wheels/releases/download/sageattention-latest/sageattention-2.2.0%2Bcu128torch2.8-cp311-cp311-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --no-cache /app

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app
EXPOSE 8000

CMD ["/app/start.sh"]
