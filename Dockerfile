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
    LTX_FP8_MODE=scaled_mm \
    TORCH_LOGS=recompiles_verbose

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
#    bombing `transformers.AutoImageProcessor` at import). We DO need real
#    torchaudio for A2V audio encoding (MelSpectrogram + resample), so it's
#    installed separately with `--no-deps` further down to sidestep the pin.
# Pin upstream LTX-2 to a known-good commit. Bumping is a one-line ARG
# change — see docs/upstream-bump.md when that runbook lands. Without
# this pin, every rebuild silently captures whatever upstream pushed to
# main last; the two sed patches below are also keyed off this SHA, so
# unpinned upstream is a regex roulette every build.
#
# `git init + fetch <SHA>` (instead of `clone --depth 1`) keeps the
# shallow-clone speed (~5 MB) while letting us check out an arbitrary
# historical SHA — `clone --depth 1` only ever gives the branch tip.
ARG LTX2_UPSTREAM_SHA=41d924371612b692c0fd1e4d9d94c3dfb3c02cb3
RUN git init /app/LTX-2 \
    && git -C /app/LTX-2 remote add origin https://github.com/Lightricks/LTX-2.git \
    && git -C /app/LTX-2 fetch --depth 1 origin "${LTX2_UPSTREAM_SHA}" \
    && git -C /app/LTX-2 checkout FETCH_HEAD \
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

# ---- FlashAttention 3 (sm_90 only) -----------------------------------------
# Self-built wheel cached as a GitHub Release asset, keyed by
# (FA3 commit SHA, NGC base tag). See docs/fa3-wheel-process.md for the
# ABI rationale and the build/bump runbook. sm_90 only — non-Hopper pods
# must launch with `LTX_ATTENTION_TYPE=` (see start.sh) for SDPA fallback.
ARG FA3_WHEEL_URL=https://github.com/anand9930-org/deploy-ltx-on-rent-gpu/releases/download/fa3-ngc25.06-6c73fb50/flash_attn_3-3.0.0-cp39-abi3-linux_x86_64.whl
ARG FA3_WHEEL_SHA256=1f1598465ea9ea3ba51084050358f90b08d2d5b5435ef205b8265fd3d59aff9c
# Decode `%2B` → `+` so uv reads the on-disk filename per PEP 427.
RUN if [ "${FA3_WHEEL_URL}" = "__SET_BY_BUILD_FA3_WHEEL_SH__" ] \
        || [ "${FA3_WHEEL_SHA256}" = "__SET_BY_BUILD_FA3_WHEEL_SH__" ]; then \
        echo "ERROR: FA3_WHEEL_URL/SHA256 are unset placeholders." >&2; \
        echo "       Run scripts/runpod_build_fa3_wheel.sh, then paste the" >&2; \
        echo "       printed values into the ARG defaults above." >&2; \
        echo "       See docs/fa3-wheel-process.md for the full runbook." >&2; \
        exit 1; \
    fi \
    && FA3_WHEEL_FILE="/tmp/$(basename "${FA3_WHEEL_URL}" | sed 's/%2B/+/g')" \
    && curl -fsSL --retry 3 -o "${FA3_WHEEL_FILE}" "${FA3_WHEEL_URL}" \
    && echo "${FA3_WHEEL_SHA256}  ${FA3_WHEEL_FILE}" | sha256sum -c - \
    && uv pip install --system --break-system-packages --no-cache --no-deps "${FA3_WHEEL_FILE}" \
    && python -c "import flash_attn_interface; \
        v = getattr(flash_attn_interface, '__version__', 'unknown'); \
        assert hasattr(flash_attn_interface, 'flash_attn_func'), 'FA3 wheel missing flash_attn_func API'; \
        print('FA3 wheel installed OK:', v)" \
    && rm -f "${FA3_WHEEL_FILE}"

# ---- Real torchaudio (--no-deps + C++ ext bypass) --------------------------
# ltx-core's audio_vae needs torchaudio.transforms.MelSpectrogram and
# torchaudio.functional.resample for the A2V audio encoding path. Two
# obstacles, both keyed off NGC's torch fork (2.8.0a0+nv25.6):
#
# 1. uv resolution: every PyPI torchaudio release declares a strict
#    `torch==<exact>` pin that would replace NGC's NVIDIA-patched torch
#    (see the two-sed rationale above). `--no-deps` sidesteps this.
#
# 2. C++ extension ABI: PyPI torchaudio's `libtorchaudio.so` links the
#    two-arg `c10::cuda::SetDevice(int8_t, bool)` symbol that NGC's torch
#    lacks (same root cause as FA3 — see docs/fa3-wheel-process.md). At
#    `import torchaudio`, `_extension/__init__.py` calls
#    `_load_lib("libtorchaudio")` and crashes with
#    `OSError: undefined symbol: _ZN3c104cuda9SetDeviceEab`.
#
#    Both APIs we need (MelSpectrogram, resample) are pure-PyTorch on top
#    of `torch.stft` / `torch.nn.functional.conv1d` and don't touch
#    `torch.ops.torchaudio.*`. We bypass the load by overwriting
#    `_extension/__init__.py` with a stub that exports the names other
#    submodules import (`_IS_TORCHAUDIO_EXT_AVAILABLE=False` is
#    load-bearing — `functional/filtering.py` branches on it to pick the
#    pure-Python path; `lazy_import_sox_ext` etc. are imported at module
#    load by `_backend/utils.py`, `sox_effects/sox_effects.py`,
#    `functional/_alignment.py`).
#
#    `torio/_extension/__init__.py` (a sibling package torchaudio depends
#    on for streaming I/O — `libtorio_ffmpeg{N}.so`) hits the same symbol
#    and gets the same stub treatment.
#
#    The next RUN block actually exercises MelSpectrogram + resample at
#    build time so any other ABI surprise aborts the build, not the pod.
RUN uv pip install --system --break-system-packages --no-cache --no-deps 'torchaudio>=2.8,<2.9' \
    && SITE=$(python -c 'import site; print(site.getsitepackages()[0])') \
    && printf '%s\n' \
        '"""Stubbed for NGC torch ABI compatibility — see Dockerfile."""' \
        'import logging' \
        '_LG = logging.getLogger(__name__)' \
        '_IS_TORCHAUDIO_EXT_AVAILABLE = False' \
        '_IS_RIR_AVAILABLE = False' \
        '_IS_ALIGN_AVAILABLE = False' \
        'def _check_cuda_version(): return None' \
        'class _UnavailableExt:' \
        '    def is_available(self): return False' \
        '    def __getattr__(self, name):' \
        '        raise RuntimeError(f"torchaudio C++ ext disabled: {name}")' \
        '_unavailable_singleton = _UnavailableExt()' \
        'def lazy_import_sox_ext(): return _unavailable_singleton' \
        'def lazy_import_ffmpeg_ext(): return _unavailable_singleton' \
        'def fail_if_no_rir(fn):' \
        '    def _stub(*a, **k):' \
        '        raise RuntimeError("torchaudio RIR not built")' \
        '    return _stub' \
        'def fail_if_no_align(fn):' \
        '    def _stub(*a, **k):' \
        '        raise RuntimeError("torchaudio align not built")' \
        '    return _stub' \
        '__all__ = ["_check_cuda_version", "_IS_TORCHAUDIO_EXT_AVAILABLE", "_IS_RIR_AVAILABLE", "lazy_import_sox_ext"]' \
        > "$SITE/torchaudio/_extension/__init__.py" \
    && if [ -d "$SITE/torio/_extension" ]; then \
        printf '%s\n' \
            '"""Stubbed for NGC torch ABI compatibility — see Dockerfile."""' \
            'class _UnavailableExt:' \
            '    def is_available(self): return False' \
            '    def __getattr__(self, name):' \
            '        raise RuntimeError(f"torio C++ ext disabled: {name}")' \
            '_unavailable_singleton = _UnavailableExt()' \
            'def lazy_import_ffmpeg_ext(): return _unavailable_singleton' \
            > "$SITE/torio/_extension/__init__.py"; \
    fi

# ---- Consolidated boot-blocker assertions ----------------------------------
# Upgrade anyio past NGC's pre-installed 4.8.x (httpx_ws needs
# AsyncContextManagerMixin from 4.9.0) AND verify the full stack in one
# shot: NGC torch still in place, torchvision's native ops load, torchaudio
# has the A2V-required APIs (MelSpectrogram, resample), and FA3 is present.
# Pin numpy<2 — NGC torch is built against NumPy 1.x; NumPy 2.x silently
# breaks `torch.Tensor.numpy()` and crashes encode_video.
RUN uv pip install --system --break-system-packages --no-cache --upgrade 'anyio>=4.9' 'numpy<2' \
    && python -c "\
import importlib.metadata as m, anyio, numpy, torch, torchvision, torchaudio, flash_attn_interface; \
assert hasattr(anyio, 'AsyncContextManagerMixin'), f'anyio too old: {m.version(\"anyio\")}'; \
assert numpy.__version__.split('.')[0] == '1', f'numpy must be 1.x for NGC torch ABI; got {numpy.__version__}'; \
assert '.nv' in torch.__version__, f'NGC torch was replaced: {torch.__version__}'; \
torchvision.ops.nms; \
assert torchaudio.__version__ != '0.0.0-stub', 'torchaudio is still the stub — real package required for A2V'; \
assert torchaudio._extension._IS_TORCHAUDIO_EXT_AVAILABLE is False, 'torchaudio C++ ext bypass not in effect'; \
mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, f_min=0.0, f_max=11025.0, n_mels=80, window_fn=torch.hann_window, center=True, pad_mode='reflect', power=1.0, mel_scale='slaney', norm='slaney')(torch.randn(1, 22050)); \
assert mel.shape[-2] == 80, f'MelSpectrogram broken: shape={mel.shape}'; \
rs = torchaudio.functional.resample(torch.randn(1, 22050), 22050, 16000); \
assert rs.shape[-1] == 16000, f'resample broken: shape={rs.shape}'; \
assert hasattr(flash_attn_interface, 'flash_attn_func'), 'FA3 wheel missing flash_attn_func'; \
torch.zeros(2).numpy(); \
print('anyio', m.version('anyio'), '/ numpy', numpy.__version__, '/ torch', torch.__version__, '/ torchvision', torchvision.__version__, '/ torchaudio', torchaudio.__version__, '(C++ ext bypassed) / FA3', getattr(flash_attn_interface, '__version__', 'unknown'))"

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ---- Upstream contract fail-fast -------------------------------------------
# `src/upstream.py` is the single chokepoint for every ltx_core / ltx_pipelines
# symbol the service consumes. Importing it here turns an upstream rename at
# the pinned SHA into a build failure with the exact missing name, instead of
# a cryptic AttributeError 30 minutes into a generation on a deployed pod.
RUN PYTHONPATH=/app python -c "import src.upstream; print('upstream contract OK')"

WORKDIR /app
EXPOSE 8000

CMD ["/app/start.sh"]
