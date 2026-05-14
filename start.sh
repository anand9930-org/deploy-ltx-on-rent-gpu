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

# Boot-time diagnostic: print image build requirements vs host machine
# specs, then verdict. Uses nvidia-smi (NVML) for host inspection — does NOT
# trigger torch.cuda._lazy_init, so a driver/runtime mismatch surfaces as a
# clear "DRIVER TOO OLD: image needs CUDA X.Y+, host supports CUDA A.B max"
# verdict in the log instead of a Python traceback. Exits 2 on driver
# mismatch, 3 on unsupported compute capability — both cases are caught by
# the script's `set -e` and propagate cleanly to the container exit code.
#
# `cd /app` lands us where `src.diagnostics` resolves (WORKDIR=/app in the
# Dockerfile). Module form is preferred over script-form for the same
# sys.path reason that `python3 -m src.download_models` is below.
cd /app
python3 -u -m src.diagnostics

# Download models (idempotent — skips if already present).
echo "=== Checking/downloading models ==="
python3 -u -m src.download_models

# Start BentoML service
echo "=== Starting BentoML service on port 8000 ==="
exec bentoml serve service:LTXVideoService --host 0.0.0.0 --port 8000
