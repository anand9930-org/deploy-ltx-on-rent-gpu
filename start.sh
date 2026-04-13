#!/bin/bash
set -e

echo "=== LTX-2.3 Video Generation Service ==="
echo "MODEL_DIR=${MODEL_DIR:-/models}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"

# Start lightweight health server on port 8001 in the background.
# RunPod's load balancer polls GET /ping on this port to decide whether
# the worker is healthy. Returning 200 OK immediately (before BentoML
# finishes loading the ~80 GB of models) prevents the worker from being
# marked dead during the ~60s pipeline initialization.
echo "=== Starting health server on port ${HEALTH_PORT:-8001} ==="
python3 -u /app/src/health_server.py &
HEALTH_PID=$!
trap "kill $HEALTH_PID 2>/dev/null" EXIT

# Download models (idempotent — skips if already present)
echo "=== Checking/downloading models ==="
python3 -u /app/src/download_models.py

# Start BentoML service
echo "=== Starting BentoML service on port 8000 ==="
exec bentoml serve service:LTXVideoService --host 0.0.0.0 --port 8000
