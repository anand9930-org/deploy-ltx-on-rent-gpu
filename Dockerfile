# ============================================================================
# LTX-2.3 22B — BentoML video generation service (ComfyUI graph pipeline)
# ============================================================================
# Targets RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120 — verified
# live; the actual torch.cuda.get_device_capability() returns (12, 0)).
# Models expected at $MODEL_DIR (default /models), typically a mounted volume;
# downloaded on first boot if not already present (~63 GB).
#
# Stack: NVIDIA CUDA 12.8 base image + stable PyTorch **2.8.0 from PyPI's
# cu128 index**. RunPod's entire RTX PRO 6000 Blackwell fleet (Community AND
# Secure Cloud) is currently on driver 570.195.03 (CUDA 12.8 max) as of
# 2026-05-14 — cu130 wheels would crashloop with "driver too old" until the
# fleet rolls forward to driver 580+ (likely once 580 becomes the production
# branch later in 2026).
#
# FP8 weights in LTX-2.3 are dequantized to BF16 at every matmul on this
# cu128 stack. Phase 1.6d tried recovering the FP8 fast path via ComfyUI's
# `--fast fp8_matrix_mult` flag (torch._scaled_mm); rolled back in Phase 1.6g
# after live verification found it regresses I2V image conditioning at frame
# counts >= 241 (per-tensor FP8 activation cast saturates cross-attention).
# See src/comfyui_runtime.py for the full rationale.
#
# Future-work: when RunPod's fleet upgrades to driver 580+, revisit cu130 to
# unlock ComfyUI's `comfy_kitchen` CUDA backend (FP8 path gated by
# `cuda_version >= (13,)` in comfy/quant_ops.py). That path has per-module
# FP8 enable lists that skip cross-attention, avoiding the regression we hit
# with the blanket `--fast fp8_matrix_mult` flag.
# ============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface

# ---- System dependencies + uv ----------------------------------------------
# Ubuntu 24.04 LTS ships Python 3.12 by default. PyTorch 2.8.0 cu128 has cp312
# wheels so no pyenv / deadsnakes needed. uv handles pip with cache discipline.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg git git-lfs gcc curl ca-certificates \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- PyTorch 2.8.0 stack (cu128 wheels) ------------------------------------
# Single index, matched ABI across torch / torchvision / torchaudio. cu128 is
# the highest CUDA minor version supported by RunPod's current driver fleet
# (570.195.03 = CUDA 12.8 max). The cu128 wheels ship sm_120 SASS for the
# RTX PRO 6000 Blackwell card; FP8 perf is recovered via ComfyUI's
# `--fast fp8_matrix_mult` flag (see src/comfyui_runtime.py), which routes
# FP8 weights through torch._scaled_mm (cuBLAS FP8) on sm_120 with cu128.
RUN uv pip install --system --break-system-packages --no-cache \
        --index-url https://download.pytorch.org/whl/cu128 \
        'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0'

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --break-system-packages --no-cache /app

# ---- Boot-blocker assertions -----------------------------------------------
# Verify (1) anyio is current enough for httpx_ws (4.9+), (2) torch is the
# cu128 build (matches the host driver fleet at 570.x = CUDA 12.8 max), and
# (3) the wheel ships kernels (or PTX) for Blackwell sm_120. The arch check
# catches a regression or wrong-index install at build time, in GHA, with no
# GPU. Accepted: sm_120 (binary) or compute_120 (PTX; JITs at first kernel
# launch). sm_122 / compute_122 also accepted defensively in case NVIDIA
# ships a re-tagged variant.
#
# Why _C._cuda_getArchFlags() and not torch.cuda.get_arch_list()?
# get_arch_list() gates on torch.cuda.is_available(), which returns False on
# the GHA ubuntu-latest builder (no GPU), so the wrapper always returns [].
# The private getter reads from a compile-time static NVCC_FLAGS_EXTRA string
# and works without a GPU — it is exactly what get_arch_list() delegates to
# when CUDA is available.
RUN python -c "\
import importlib.metadata as m, anyio, numpy, torch, torchvision; \
assert hasattr(anyio, 'AsyncContextManagerMixin'), f'anyio too old: {m.version(\"anyio\")}'; \
assert torch.__version__.startswith('2.8.'), f'expected torch 2.8.x; got {torch.__version__}'; \
assert torch.version.cuda and torch.version.cuda.startswith('12.'), f'expected CUDA 12.x runtime (cu128 wheel); got {torch.version.cuda}'; \
arch_flags = torch._C._cuda_getArchFlags() or ''; \
arch_list = arch_flags.split(); \
needed = ('sm_122', 'compute_122', 'sm_120', 'compute_120'); \
assert any(a in needed for a in arch_list), \
    f'torch wheel lacks Blackwell workstation arch (sm_120/sm_122); got {arch_list!r}'; \
torchvision.ops.nms; torch.zeros(2).numpy(); \
print('anyio', m.version('anyio'), '/ torch', torch.__version__, '/ cuda', torch.version.cuda, '/ torchvision', torchvision.__version__, '/ numpy', numpy.__version__, '/ arch_list', arch_list)"

# ---- ComfyUI ----------------------------------------------------------------
# The pipeline runs real ComfyUI **core** node classes — no ComfyUI-LTXVideo.
# Clone (not pip-install) so the SHA is pinnable. Strip torch / torchvision /
# torchaudio / numpy / transformers / diffusers / Pillow from ComfyUI's
# requirements so our pinned cu130 wheels survive.
ARG COMFYUI_SHA=64b8457f55cd7fb54ca7a956d9c73b505e903e0c
RUN git init /app/ComfyUI \
    && git -C /app/ComfyUI remote add origin https://github.com/comfyanonymous/ComfyUI.git \
    && git -C /app/ComfyUI fetch --depth 1 origin "${COMFYUI_SHA}" \
    && git -C /app/ComfyUI checkout FETCH_HEAD \
    && sed -i -E '/^(torch|torchvision|torchaudio|numpy|transformers|diffusers|[Pp]illow)\b/d' \
        /app/ComfyUI/requirements.txt \
    && uv pip install --system --break-system-packages --no-cache \
        -r /app/ComfyUI/requirements.txt \
    && python -c "\
import torch, torchvision, torchaudio; \
assert torch.__version__.startswith('2.8.'), f'torch clobbered by ComfyUI install: {torch.__version__}'; \
torchvision.ops.nms; \
print('stack intact after ComfyUI install — torch', torch.__version__, '/ torchaudio', torchaudio.__version__)"
ENV COMFYUI_PATH=/app/ComfyUI

# ---- Copy application code -------------------------------------------------
COPY src/ /app/src/
COPY service.py /app/service.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# ---- ComfyUI node-contract fail-fast ----------------------------------------
# Bootstrap the minimal ComfyUI runtime and import every core node class the
# cascade uses, so a node rename / wrong COMFYUI_SHA fails the build instead
# of a request 5 minutes in.
RUN PYTHONPATH=/app python -c "\
import os; \
from src import comfyui_runtime; \
comfyui_runtime.bootstrap_once(os.environ.get('COMFYUI_PATH', '/app/ComfyUI'), os.environ.get('MODEL_DIR', '/models'), cpu_only=True); \
from nodes import CheckpointLoaderSimple, LoraLoaderModelOnly, LoadImage, VAEDecodeTiled, CLIPTextEncode; \
from comfy_extras.nodes_custom_sampler import KSamplerSelect, ManualSigmas, RandomNoise, CFGGuider, SamplerCustomAdvanced; \
from comfy_extras.nodes_hunyuan import LatentUpscaleModelLoader; \
from comfy_extras.nodes_lt import EmptyLTXVLatentVideo, LTXVConcatAVLatent, LTXVConditioning, LTXVImgToVideoInplace, LTXVPreprocess, LTXVSeparateAVLatent; \
from comfy_extras.nodes_lt_audio import LTXAVTextEncoderLoader, LTXVAudioVAEDecode, LTXVAudioVAELoader, LTXVEmptyLatentAudio; \
from comfy_extras.nodes_lt_upsampler import LTXVLatentUpsampler; \
from comfy_extras.nodes_post_processing import ResizeImageMaskNode; \
from comfy_extras.nodes_video import CreateVideo; \
print('ComfyUI node contract OK')"

WORKDIR /app
EXPOSE 8000

CMD ["/app/start.sh"]
