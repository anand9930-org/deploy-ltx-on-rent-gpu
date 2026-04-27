#!/bin/bash
set -e

# Default attention backend for this image: FA3 on Hopper. Override at pod
# launch (e.g. `LTX_ATTENTION_TYPE=` or `LTX_ATTENTION_TYPE=pytorch`) to fall
# back to torch SDPA — required on non-Hopper GPUs (sm_89/sm_120) where the
# FA3 wheel is unusable.
export LTX_ATTENTION_TYPE="${LTX_ATTENTION_TYPE:-flash_attention_3}"

echo "=== LTX-2.3 Video Generation Service ==="
echo "MODEL_DIR=${MODEL_DIR:-/models}"
echo "LTX_ATTENTION_TYPE=${LTX_ATTENTION_TYPE}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"

# Download models (idempotent — skips if already present)
echo "=== Checking/downloading models ==="
python3 -u /app/src/download_models.py

# Pre-warm Linux page cache for the large weight files. Reads are dropped
# to /dev/null; the kernel keeps the bytes in page cache so the
# safetensors load path inside BentoML hits RAM instead of NVMe.
# One-time boot cost (~60-90s); turns the ~348s first-request cold tax
# (Gemma 24 GB + dev-fp8 29 GB + distilled-fp8 29 GB of disk reads) into
# the warm-pod regime (~13s). RunPod H100 80GB pods ship with 128-256 GB
# host RAM, so the ~80 GB working set fits in page cache.
#
# We deliberately skip ltx-2.3-22b-dev.safetensors (BF16 base, 46 GB) —
# only its VAE / embeddings-processor keys are touched at runtime
# (~5 GB hot bytes), so caching the whole file would evict pages we do
# need for ~40 GB of bytes we don't.
echo "=== Pre-warming page cache (~60-90s) ==="
MODELS="${MODEL_DIR:-/models}"
PREWARM_FILES=()
for p in \
    "$MODELS"/ltx-2.3-22b-dev-fp8.safetensors \
    "$MODELS"/ltx-2.3-22b-distilled-fp8.safetensors ; do
    [ -f "$p" ] && PREWARM_FILES+=("$p")
done
# Gemma shards: download_models.py uses snapshot_download(local_dir=...)
# so files land flat under $MODELS/gemma-3-12b-it-qat-q4_0-unquantized/,
# NOT under the HF hub cache layout. Glob the top level for any future
# variant that shares the gemma-3-12b-it* prefix.
for g in "$MODELS"/gemma-3-12b-it*/model*.safetensors; do
    [ -f "$g" ] && PREWARM_FILES+=("$g")
done
echo "Pre-warming ${#PREWARM_FILES[@]} weight files..."
for f in "${PREWARM_FILES[@]}"; do
    cat "$f" > /dev/null &
done
wait
echo "=== Page cache warm ($(grep ^Cached /proc/meminfo | awk '{printf "%.1f GB", $2/1024/1024}')) ==="

# Start BentoML service
echo "=== Starting BentoML service on port 8000 ==="
exec bentoml serve service:LTXVideoService --host 0.0.0.0 --port 8000
