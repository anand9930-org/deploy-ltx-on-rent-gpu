#!/bin/bash
set -e

echo "=== LTX-2.3 Video Generation — RunPod QB Worker ==="
echo "MODEL_DIR=${MODEL_DIR:-/models}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"

# Download models (idempotent — skips if already present)
echo "=== Checking/downloading models ==="
python3 -u /app/src/download_models.py

# Hand off to the RunPod handler (blocks on the serverless queue).
# The handler constructs the pipeline once at module load, then executes
# jobs as RunPod's queue dispatches them. No BentoML HTTP server runs in
# this container; /readyz, /livez, /metrics, etc. are not available.
echo "=== Starting RunPod QB handler ==="
exec python3 -u /app/runpod_handler.py
