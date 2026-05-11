# ============================================================================
# LTX-2.3 22B — BentoML video generation service (ComfyUI graph pipeline)
# ============================================================================
# Models are expected at $MODEL_DIR (default /models), typically mounted
# as a volume.  Downloaded on first boot if not already present (~63 GB).
# ============================================================================

FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    HF_HOME=/models/huggingface \
    TORCH_LOGS=recompiles_verbose

# ---- System dependencies + uv ----------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git-lfs gcc \
    && git lfs install \
    && rm -rf /var/lib/apt/lists/*

# ---- NGC pip-constraint hygiene --------------------------------------------
RUN sed -i '/^anyio/d' /etc/pip/constraint.txt

# ---- Install project dependencies ------------------------------------------
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --break-system-packages --no-cache /app

# ---- Boot-blocker assertions -----------------------------------------------
# Upgrade anyio past NGC's pre-installed 4.8.x (httpx_ws needs
# AsyncContextManagerMixin from 4.9.0) AND verify the stack in one shot.
# Pin numpy<2 — NGC torch is built against NumPy 1.x.
RUN uv pip install --system --break-system-packages --no-cache --upgrade 'anyio>=4.9' 'numpy<2' \
    && python -c "\
import importlib.metadata as m, anyio, numpy, torch, torchvision; \
assert hasattr(anyio, 'AsyncContextManagerMixin'), f'anyio too old: {m.version(\"anyio\")}'; \
assert numpy.__version__.split('.')[0] == '1', f'numpy must be 1.x for NGC torch ABI; got {numpy.__version__}'; \
assert '.nv' in torch.__version__, f'NGC torch was replaced: {torch.__version__}'; \
torchvision.ops.nms; \
torch.zeros(2).numpy(); \
print('anyio', m.version('anyio'), '/ numpy', numpy.__version__, '/ torch', torch.__version__, '/ torchvision', torchvision.__version__)"

# ---- ComfyUI ----------------------------------------------------------------
# The pipeline runs real ComfyUI **core** node classes — no ComfyUI-LTXVideo.
# Clone (not pip-install) so the SHA is pinnable. Strip torch / torchvision /
# torchaudio / numpy / transformers / diffusers / Pillow from ComfyUI's
# requirements so the NGC torch ABI pins survive.
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
import torch, numpy, torchvision; \
assert numpy.__version__.split('.')[0] == '1', f'numpy clobbered by ComfyUI install: {numpy.__version__}'; \
assert '.nv' in torch.__version__, f'NGC torch clobbered by ComfyUI install: {torch.__version__}'; \
torchvision.ops.nms; torch.zeros(2).numpy(); \
print('NGC stack intact after ComfyUI install — torch', torch.__version__, '/ numpy', numpy.__version__, '/ torchvision', torchvision.__version__)"
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
