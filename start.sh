#!/bin/bash
set -e

# Cloned ComfyUI checkout — used by the triple_stages_comfyui pipeline (see
# the Dockerfile + src/comfyui_runtime.py). The Dockerfile already exports this;
# the default here keeps `./start.sh` working in a non-Docker dev shell.
export COMFYUI_PATH="${COMFYUI_PATH:-/app/ComfyUI}"

echo "=== LTX-2.3 Video Generation Service ==="
echo "MODEL_DIR=${MODEL_DIR:-/models}"
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
