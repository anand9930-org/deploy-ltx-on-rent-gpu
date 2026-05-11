#!/bin/bash
set -e

# Default attention backend for this image: FA3 on Hopper. Override at pod
# launch (e.g. `LTX_ATTENTION_TYPE=` or `LTX_ATTENTION_TYPE=pytorch`) to fall
# back to torch SDPA — required on non-Hopper GPUs (sm_89/sm_120) where the
# FA3 wheel is unusable.
export LTX_ATTENTION_TYPE="${LTX_ATTENTION_TYPE:-flash_attention_3}"

# Which upstream pipeline to preload at boot. I2V default (this deployment
# is I2V-heavy); override to `t2v` for T2V-first pods. Cross-mode requests
# at runtime trigger a tear-down + rebuild — only one upstream pipeline is
# resident at a time. `i2v`, `v2v`, and `unified` all preload the same
# ICLoraPipeline (I2V and V2V share weights).
export LTX_DEFAULT_MODE="${LTX_DEFAULT_MODE:-i2v}"

# Cloned ComfyUI checkout — used only by the triple_stages_comfyui pipeline (see
# the Dockerfile + src/comfyui_runtime.py). The Dockerfile already exports this;
# the default here keeps `./start.sh` working in a non-Docker dev shell.
export COMFYUI_PATH="${COMFYUI_PATH:-/app/ComfyUI}"

echo "=== LTX-2.3 Video Generation Service ==="
echo "MODEL_DIR=${MODEL_DIR:-/models}"
echo "LTX_ATTENTION_TYPE=${LTX_ATTENTION_TYPE}"
echo "LTX_DEFAULT_MODE=${LTX_DEFAULT_MODE}"
echo "COMFYUI_PATH=${COMFYUI_PATH}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"

# Download models (idempotent — skips if already present).
# Module form (`-m src.download_models`) is load-bearing: the script-form
# `python3 /app/src/download_models.py` puts /app/src/ on sys.path instead
# of /app, breaking `from src.config import get_settings` (added in the
# pydantic-settings refactor). WORKDIR=/app in the Dockerfile makes the
# module form resolve correctly without an explicit PYTHONPATH.
cd /app
echo "=== Checking/downloading models ==="
python3 -u -m src.download_models

# Start BentoML service
echo "=== Starting BentoML service on port 8000 ==="
exec bentoml serve service:LTXVideoService --host 0.0.0.0 --port 8000
