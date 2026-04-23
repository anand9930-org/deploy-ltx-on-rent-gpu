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

# ---- NGC pip-constraint hygiene --------------------------------------------
# NGC ships /etc/pip/constraint.txt pinning every pre-installed Python
# package to the versions NVIDIA tested. The documented way to override a
# pin (NGC PyTorch 25.06 release notes) is to strip the package's line
# from this file before installing your own version. We strip `anyio`
# because NGC pins it below 4.9 and httpx_ws (pulled by bentoml) needs
# anyio.AsyncContextManagerMixin, added in 4.9.0.
#
# Note: `uv pip install` does NOT read this file — uv only honors
# UV_CONSTRAINT / --constraint. We still strip the line as hygiene for
# any downstream pip invocation (e.g. `pip install` inside a subprocess
# or future migration to `uv --constraint /etc/pip/constraint.txt`).
# The actual uv-side upgrade is performed explicitly further below.
RUN sed -i '/^anyio/d' /etc/pip/constraint.txt

# ---- Clone LTX-2 and install its packages ----------------------------------
# The `[fp8-trtllm]` extra pulls tensorrt-llm==1.0.0 + onnx + openmpi from
# pypi.nvidia.com and registers the `torch.ops.tensorrt_llm.*` +
# `torch.ops.trtllm.*` ops used by FP8Linear.forward on the scaled_mm
# (H100 W8A8) path. This extra is mutex with `[xformers]` in upstream
# pyproject.toml — that's fine here because LTX-2-ref's attention
# dispatcher falls back to torch SDPA and FA3 is installed separately
# below.
#
# In-place patches (both targeted seds with grep-based verification so the
# build fails loudly if upstream refactors):
#
# 1. force use_fast=True on the Gemma image processor so the Rust-backed
#    tokenizer runs instead of the slow Python fallback. Default in
#    transformers >=4.52 anyway; forcing it eliminates the "Using a slow
#    image processor" warning and shaves a second or two off text encoding.
#
# 2. strip the unconstrained `"torchaudio",` dep from ltx-core's pyproject
#    BEFORE uv resolves it. Every PyPI torchaudio release declares a strict
#    companion `torch==<exact>` pin that would force uv to replace NGC's
#    NVIDIA-patched torch 2.8.0a0+...nv25.6 with stock PyPI torch — which
#    breaks NGC torchvision's ABI (`torchvision::nms` op fails to register,
#    bombing `transformers.AutoImageProcessor` at import). Our service is
#    video-only (audio_guider_params is None on every request), so the
#    audio_vae code path that touches torchaudio APIs is unreachable. A
#    pure-Python stub is dropped into site-packages further down satisfies
#    the `import torchaudio` at module load.
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && sed -i 's|AutoImageProcessor.from_pretrained(processor_root, local_files_only=True)|AutoImageProcessor.from_pretrained(processor_root, local_files_only=True, use_fast=True)|' \
        /app/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py \
    && grep -q "use_fast=True" /app/LTX-2/packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/base_encoder.py \
    && sed -i '/"torchaudio",/d' /app/LTX-2/packages/ltx-core/pyproject.toml \
    && ! grep -q '^\s*"torchaudio"' /app/LTX-2/packages/ltx-core/pyproject.toml \
    && uv pip install --system --break-system-packages --no-cache \
        --extra-index-url https://pypi.nvidia.com \
        -e "/app/LTX-2/packages/ltx-core[fp8-trtllm]" \
        -e /app/LTX-2/packages/ltx-pipelines

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --break-system-packages --no-cache /app

# ---- Pure-Python torchaudio stub -------------------------------------------
# ltx-core's audio_vae module does `import torchaudio` at module load time
# (ops.py:2) even though the class instantiation (`MelSpectrogram(...)`) and
# function calls (`torchaudio.functional.resample`) are lazy and only fire
# when an audio generation is requested. Since our service never passes
# `audio_guider_params`, those APIs are never hit — but the top-level import
# still needs to succeed or every pipeline load fails.
#
# We don't install the real torchaudio (see rationale in the two-sed block
# above). Instead, write a minimal pure-Python package that has the right
# import surface. If anything ever reaches the NotImplementedError, we want
# a loud failure rather than a silent wrong-result.
RUN SITE=$(python -c 'import site; print(site.getsitepackages()[0])') \
    && mkdir -p "$SITE/torchaudio" \
    && printf '%s\n' \
        'from . import functional, transforms  # noqa: F401' \
        '__version__ = "0.0.0-stub"' \
        > "$SITE/torchaudio/__init__.py" \
    && printf '%s\n' \
        'class MelSpectrogram:' \
        '    def __init__(self, *a, **kw):' \
        '        raise NotImplementedError(' \
        '            "torchaudio stub: audio path disabled on this deployment"' \
        '        )' \
        > "$SITE/torchaudio/transforms.py" \
    && printf '%s\n' \
        'def resample(*a, **kw):' \
        '    raise NotImplementedError(' \
        '        "torchaudio stub: audio path disabled on this deployment"' \
        '    )' \
        > "$SITE/torchaudio/functional.py"

# ---- Consolidated boot-blocker assertions ----------------------------------
# Upgrade anyio past NGC's pre-installed 4.8.x (httpx_ws needs
# AsyncContextManagerMixin from 4.9.0) AND verify the full stack in one
# shot: NGC torch still in place, torchvision's native ops load, and the
# torchaudio import resolves to our stub. Any failure aborts the build.
RUN uv pip install --system --break-system-packages --no-cache --upgrade 'anyio>=4.9' \
    && python -c "\
import importlib.metadata as m, anyio, torch, torchvision, torchaudio; \
assert hasattr(anyio, 'AsyncContextManagerMixin'), f'anyio too old: {m.version(\"anyio\")}'; \
assert 'nv25.6' in torch.__version__, f'NGC torch was replaced: {torch.__version__}'; \
torchvision.ops.nms; \
assert torchaudio.__version__ == '0.0.0-stub', f'real torchaudio leaked: {torchaudio.__version__}'; \
print('anyio', m.version('anyio'), '/ torch', torch.__version__, '/ torchvision', torchvision.__version__, '/ torchaudio stub OK')"

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
