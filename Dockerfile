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
# Snapshot the base image's torch ABI BEFORE any of our pip installs run, so
# the assertions below can detect a silent downgrade. The base image ships a
# matched torch + torchaudio pair built against the same libc10_cuda; if any
# subsequent install downgrades torch, torchaudio's C++ extension fails to
# load at pod boot with `undefined symbol: _ZN3c104cuda29c10_cuda_check...`.
RUN python3 -c "import torch; print(f'BASE_TORCH={torch.__version__}')" \
        > /tmp/base_torch_version
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --no-cache /app \
    && python3 -c "import torch; \
        base = open('/tmp/base_torch_version').read().split('=',1)[1].strip(); \
        assert torch.__version__ == base, \
            f'torch downgraded by project install: {base!r} -> {torch.__version__!r}'; \
        print(f'torch unchanged after project install: {torch.__version__}')"

# ---- Install xformers for Ada-friendly attention ---------------------------
# LTX-2's AttentionFunction.DEFAULT picks XFormersAttention when xformers
# is importable (faster than the PytorchAttention / SDPA fallback for the
# BF16 non-causal shapes LTX-2 uses). Works on Ada sm_89 (L40S, RTX 6000
# Ada), Ampere, and Hopper. We deliberately do NOT install FA3 here:
# flash_attn_3 is Hopper-only and would shadow xformers even if present.
#
# `--no-deps` is LOAD-BEARING. Without it, pip's resolver reads xformers's
# torch pin and may downgrade torch to satisfy it (xformers releases lag
# torch by 1-2 weeks on PyPI). torchaudio in the base image was built
# against the original torch's libtorch, so an ABI mismatch surfaces as:
#
#   OSError: libtorchaudio.so: undefined symbol:
#   _ZN3c104cuda29c10_cuda_check_implementationEiPKcS2_ib
#
# at module-import time, which crashes the BentoML service before any
# generation logic runs. See commit fb312f2 (feat/fa3-teacache) for the
# original incident and analysis. Same class of bug as torchaudio vs
# SageAttention installs from earlier.
RUN pip install --no-cache-dir --no-deps xformers \
    && python3 -c "import xformers; print(f'xformers {xformers.__version__} loaded OK')" \
    && python3 -c "import torch; \
        base = open('/tmp/base_torch_version').read().split('=',1)[1].strip(); \
        assert torch.__version__ == base, \
            f'torch downgraded by xformers install: {base!r} -> {torch.__version__!r}'; \
        print(f'torch unchanged after xformers install: {torch.__version__}')" \
    && python3 -c "import torch, torchaudio; \
        assert torchaudio.__version__.split('+')[0] == torch.__version__.split('+')[0], \
            f'torch/torchaudio version mismatch: torch={torch.__version__} torchaudio={torchaudio.__version__}'; \
        print(f'torch+torchaudio ABI OK: torch={torch.__version__} torchaudio={torchaudio.__version__}')"

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

WORKDIR /app
EXPOSE 8000

CMD ["/app/start.sh"]
