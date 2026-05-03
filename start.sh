#!/bin/bash
set -e

# Default attention backend for this image: FA3 on Hopper. Override at pod
# launch (e.g. `LTX_ATTENTION_TYPE=` or `LTX_ATTENTION_TYPE=pytorch`) to fall
# back to torch SDPA — required on non-Hopper GPUs (sm_89/sm_120) where the
# FA3 wheel is unusable.
export LTX_ATTENTION_TYPE="${LTX_ATTENTION_TYPE:-flash_attention_3}"

# Which upstream pipeline to preload at boot. T2V default; set to `i2v`
# (or `v2v`/`unified`) on I2V-heavy deployments to skip the first-request
# ~30-60 s mode-swap penalty. Cross-mode requests at runtime trigger a
# tear-down + rebuild — only one upstream pipeline is resident at a time.
export LTX_DEFAULT_MODE="${LTX_DEFAULT_MODE:-t2v}"

echo "=== LTX-2.3 Video Generation Service ==="
echo "MODEL_DIR=${MODEL_DIR:-/models}"
echo "LTX_ATTENTION_TYPE=${LTX_ATTENTION_TYPE}"
echo "LTX_DEFAULT_MODE=${LTX_DEFAULT_MODE}"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not available')"

# Download models (idempotent — skips if already present)
echo "=== Checking/downloading models ==="
python3 -u /app/src/download_models.py

# Pre-warm Linux page cache for the large weight files. Reads are dropped
# to /dev/null; the kernel keeps the bytes in page cache so the
# safetensors load path inside BentoML hits RAM instead of NVMe.
# One-time boot cost (~60-90s); turns the cold-disk tax into a warm-pod
# regime including the lazy mode-swap rebuild path. Working set is the
# union of T2V (dev-fp8 ~30 GB or dev BF16 ~46 GB on cast) and unified
# (distilled-1.1 BF16 ~46 GB + IC-LoRA + distilled-fp8 ~30 GB) plus
# Gemma (~26 GB). RunPod H100 80 GB pods typically ship 200+ GB host RAM.
#
# We deliberately skip ltx-2.3-22b-dev.safetensors (BF16 base, ~46 GB) on
# scaled_mm pods — only its VAE / embeddings keys are touched at runtime
# (~5 GB hot bytes), so caching the whole file would evict pages we do
# need for ~40 GB of bytes we don't.
echo "=== Pre-warming page cache (~60-90s) ==="
MODELS="${MODEL_DIR:-/models}"
PREWARM_FILES=()
for p in \
    "$MODELS"/ltx-2.3-22b-dev-fp8.safetensors \
    "$MODELS"/ltx-2.3-22b-distilled-1.1.safetensors \
    "$MODELS"/ltx-2.3-22b-distilled-fp8.safetensors \
    "$MODELS"/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors ; do
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
