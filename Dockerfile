# ============================================================================
# LTX-2.3 22B — BentoML video generation service (ComfyUI graph pipeline)
# ============================================================================
# Targets RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120 — verified
# live, despite earlier docs saying sm_122).
# Models expected at $MODEL_DIR (default /models), typically a mounted volume;
# downloaded on first boot if not already present (~63 GB).
#
# Stack: NVIDIA CUDA 12.8 base image + stable PyTorch **2.10.0 from PyPI's
# cu130 index**. The cu130 wheels carry their own runtime CUDA 13 libs
# (libcudart, cuBLAS, cuDNN) and only need a CUDA-13-compatible host driver,
# which RunPod's Blackwell pods ship (driver >= 575). cu130 unlocks ComfyUI's
# `comfy_kitchen` CUDA backend (FP8 scaled matmul on Blackwell) — gated by
# `cuda_version >= (13,)` in ComfyUI's comfy/quant_ops.py. With cu128 the
# LTX-2.3 FP8 weights are silently dequantized to BF16 at every matmul.
# ============================================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface

# ---- System dependencies + uv ----------------------------------------------
# Ubuntu 24.04 LTS ships Python 3.12 by default. PyTorch 2.10.0 cu130 has cp312
# wheels so no pyenv / deadsnakes needed. uv handles pip with cache discipline.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv \
        ffmpeg git git-lfs gcc curl ca-certificates \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python

# ---- PyTorch 2.10.0 stack (cu130 wheels) -----------------------------------
# Single index, matched ABI across torch / torchvision / torchaudio. cu130
# unlocks ComfyUI's `comfy_kitchen` CUDA backend (FP8 scaled matmul on
# Blackwell sm_120) — gated by `torch.version.cuda >= (13,)` in
# comfy/quant_ops.py. Without this, FP8 LTX-2.3 weights are silently
# dequantized to BF16 at every matmul, costing ~30-50% throughput on the
# diffusion hot path (per ComfyUI Blackwell discussion #6643).
RUN uv pip install --system --break-system-packages --no-cache \
        --index-url https://download.pytorch.org/whl/cu130 \
        'torch==2.10.0' 'torchvision==0.25.0' 'torchaudio==2.10.0'

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --break-system-packages --no-cache /app

# ---- Boot-blocker assertions -----------------------------------------------
# Verify (1) anyio is current enough for httpx_ws (4.9+), (2) torch is the
# cu130 build (NOT cu128 — the FP8 backend gate trips at cu < 13), and (3) the
# wheel ships kernels (or PTX) for Blackwell sm_120. The arch check catches a
# regression or wrong-index install at build time, in GHA, with no GPU.
# Accepted: sm_120 (binary) or compute_120 (PTX; JITs at first kernel launch
# on RTX PRO 6000 Blackwell). sm_122 / compute_122 also accepted defensively
# in case NVIDIA ships a re-tagged variant.
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
assert torch.__version__.startswith('2.10.'), f'expected torch 2.10.x; got {torch.__version__}'; \
assert torch.version.cuda and torch.version.cuda.startswith('13.'), f'expected CUDA 13.x runtime (cu130 wheel); got {torch.version.cuda}'; \
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
assert torch.__version__.startswith('2.10.'), f'torch clobbered by ComfyUI install: {torch.__version__}'; \
torchvision.ops.nms; \
print('stack intact after ComfyUI install — torch', torch.__version__, '/ torchaudio', torchaudio.__version__)"
ENV COMFYUI_PATH=/app/ComfyUI

# ---- ComfyUI FP8 backend gate verification ---------------------------------
# Verify cu130 actually unlocked ComfyUI's comfy_kitchen CUDA backend.
# ComfyUI's comfy/quant_ops.py disables the 'cuda' backend (and logs the
# "You need pytorch with cu130 or higher" warning) when torch.version.cuda is
# < (13,). With our cu130 wheel this should pass; if it doesn't, the install
# regressed and we'd silently run on BF16 dequant — same as the cu128 build.
RUN python -c "\
import torch; \
cv = tuple(map(int, str(torch.version.cuda).split('.'))); \
assert cv >= (13, 0), f'torch CUDA runtime is {torch.version.cuda} — must be >= 13.0 for ComfyUI FP8 backend'; \
print('FP8 backend gate cleared — torch.version.cuda =', torch.version.cuda, '(>= 13.0)')"

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
