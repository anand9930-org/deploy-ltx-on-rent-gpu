# ============================================================================
# LTX-2.3 22B — BentoML video generation service
# ============================================================================
# Models are expected at $MODEL_DIR (default /models), typically mounted
# as a volume.  Downloaded on first boot if not already present (~64 GB).
# ============================================================================

# Base image matches the one TensorRT-LLM v1.0.0 itself is built against
# (nvidia/TensorRT-LLM@v1.0.0 docker/Dockerfile.multi). Ships Python 3.12,
# torch 2.8.0a0, CUDA 12.9.1 on Ubuntu 24.04 — cp312 is required because
# `tensorrt-llm==1.0.0` only publishes cp310/cp312 wheels (no cp311).
FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface \
    LTX_FP8_MODE=scaled_mm

# ---- System dependencies + uv ----------------------------------------------
# libopenmpi-dev provides the MPI headers/libraries that tensorrt-llm's
# openmpi wheel dlopens at import time. Without them the
# `import tensorrt_llm` probe inside QuantizationPolicy.fp8_scaled_mm()
# fails with a bare `ImportError` and we silently fall back to BF16.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git-lfs gcc libopenmpi-dev \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ---- Clone LTX-2 and install its packages ----------------------------------
# The `[fp8-trtllm]` extra pulls tensorrt-llm==1.0.0 + onnx + openmpi from
# pypi.nvidia.com and registers the `torch.ops.tensorrt_llm.*` +
# `torch.ops.trtllm.*` ops used by FP8Linear.forward on the scaled_mm
# (H100 W8A8) path. This extra is mutex with `[xformers]` in upstream
# pyproject.toml — that's fine here because LTX-2-ref's attention
# dispatcher falls back to torch SDPA and FA3 is installed separately
# below.
#
# In-place patch: force use_fast=True on the Gemma image processor so the
# Rust-backed tokenizer runs instead of the slow Python fallback. Default
# in transformers >=4.52 anyway; forcing it here eliminates the
# "Using a slow image processor" warning and shaves a second or two off
# text encoding. Targeted sed; fails loudly (warning persists in logs)
# if Lightricks ever refactors the call, which is how we'd want to know.
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && sed -i 's|AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)|AutoImageProcessor.from_pretrained(processor_root, local_files_only=True, use_fast=True)|' \
        /app/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py \
    && grep -q "use_fast=True" /app/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py \
    && uv pip install --system --no-cache \
        --extra-index-url https://pypi.nvidia.com \
        -e "/app/LTX-2/packages/ltx-core[fp8-trtllm]" \
        -e /app/LTX-2/packages/ltx-pipelines

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --no-cache /app

# ---- FlashAttention 3 — DEFERRED on this base ------------------------------
# NGC 25.06 ships CUDA 12.9.1 + NVIDIA-patched torch 2.8.0a0. The only
# prebuilt FA3 wheel index we trust (windreamer) publishes
# cu128_torch280 wheels, so the CUDA minor mismatches the container.
# Rather than risk a silent ABI break at runtime, skip FA3 for now and
# let LTX-2-ref's attention dispatcher fall back to torch SDPA (BF16).
# TODO: re-enable FA3 once a cu129_torch280 wheel exists (either from
# windreamer or a self-hosted build) so we recover the Hopper FA3 perf.

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app
EXPOSE 8000

CMD ["/app/start.sh"]
