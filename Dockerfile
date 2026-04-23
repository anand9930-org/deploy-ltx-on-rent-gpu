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

# ---- Install FlashAttention 3 (Hopper-only, community wheel index) ---------
# FA3 has no PyPI wheel. Building from source needs ~80-150 GB RAM and does
# not fit on GH-hosted runners (issue Dao-AILab/flash-attention#1043), so we
# install a prebuilt wheel from the windreamer community index — cp39-abi3
# stable-ABI wheels (usable from 3.11) built against torch 2.8.0 + cu128,
# rebuilt bi-weekly from Dao-AILab/flash-attention main. Apache-2.0,
# unsigned; acceptable for research validation, swap for a SHA-pinned
# self-hosted build before shipping to customer traffic.
RUN pip install --no-cache-dir flash_attn_3 \
        --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch280/ \
    && python3 -c "import flash_attn_interface; print('flash_attn_interface loaded OK')"

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app
EXPOSE 8000

CMD ["/app/start.sh"]
