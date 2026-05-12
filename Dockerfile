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

# ---- Real torchaudio (--no-deps + C++ ext bypass) --------------------------
# ComfyUI's audio_vae.py unconditionally imports torchaudio at module level
# (nodes -> comfy.sd -> comfy.ldm.lightricks.vae.audio_vae -> import torchaudio).
# The A2V pipeline needs torchaudio.transforms.MelSpectrogram and
# torchaudio.functional.resample at runtime. Two obstacles with NGC's torch:
#
# 1. PyPI torchaudio pins torch==<exact>, which would replace NGC's torch.
#    --no-deps sidesteps this.
#
# 2. PyPI torchaudio's libtorchaudio.so links c10::cuda::SetDevice(int8_t,bool)
#    which NGC's torch doesn't export. Both APIs we need are pure-PyTorch
#    (torch.stft / torch.nn.functional.conv1d). We overwrite
#    _extension/__init__.py with a stub that sets
#    _IS_TORCHAUDIO_EXT_AVAILABLE=False (filtering.py branches on this for
#    the pure-Python path) and exports names other submodules import at load.
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
    fi \
    && python -c "\
import torch, numpy, torchvision, torchaudio; \
assert '.nv' in torch.__version__, f'NGC torch clobbered by torchaudio: {torch.__version__}'; \
assert numpy.__version__.split('.')[0] == '1', f'numpy clobbered by torchaudio: {numpy.__version__}'; \
torchvision.ops.nms; \
assert torchaudio._extension._IS_TORCHAUDIO_EXT_AVAILABLE is False, 'C++ ext bypass not in effect'; \
mel = torchaudio.transforms.MelSpectrogram(sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, f_min=0.0, f_max=11025.0, n_mels=80, window_fn=torch.hann_window, center=True, pad_mode='reflect', power=1.0, mel_scale='slaney', norm='slaney')(torch.randn(1, 22050)); \
assert mel.shape[-2] == 80, f'MelSpectrogram broken: shape={mel.shape}'; \
rs = torchaudio.functional.resample(torch.randn(1, 22050), 22050, 16000); \
assert rs.shape[-1] == 16000, f'resample broken: shape={rs.shape}'; \
print('torchaudio', torchaudio.__version__, '(C++ ext bypassed) — MelSpectrogram + resample OK')"

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
