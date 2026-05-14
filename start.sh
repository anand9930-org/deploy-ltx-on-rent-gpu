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

# Verify the GPU compute capability before any 63 GB download.
# This branch targets RTX PRO 6000 Blackwell (sm_120). The RTX PRO 6000
# Blackwell Server *and* Workstation Editions report cap == (12, 0); the
# pytorch/pytorch#157549 issue's "SM122" label was misleading (verified live
# against the actual hardware). Non-Blackwell GPUs pass with a warning;
# pre-Hopper hard-fails.
python3 -c "
import torch
cap = torch.cuda.get_device_capability()
name = torch.cuda.get_device_name()
print(f'GPU: {name}, compute capability sm_{cap[0]}{cap[1]}')
if cap == (12, 0):
    print('Blackwell sm_120 detected — RTX PRO 6000 Blackwell or RTX 50 series (expected target on this branch)')
elif cap[0] == 12:
    print(f'Other Blackwell variant sm_{cap[0]}{cap[1]} detected — should still run')
elif cap[0] >= 9:
    print(f'Non-Blackwell GPU sm_{cap[0]}{cap[1]} — this branch is Blackwell-tuned but should still run')
else:
    raise SystemExit(f'Unsupported GPU sm_{cap[0]}{cap[1]} — this branch targets Blackwell sm_120')
"

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
