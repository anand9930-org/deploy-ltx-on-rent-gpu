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
# The triple_stages_comfyui-graph path loads the *full* dev BF16 (~46 GB), not
# dev-fp8 — prewarm it only when this pod preloads that mode (other modes touch
# only its VAE/embeddings keys, so caching all 46 GB would evict pages they need).
if [ "${LTX_DEFAULT_MODE}" = "triple_stages_comfyui" ]; then
    [ -f "$MODELS"/ltx-2.3-22b-dev.safetensors ] && PREWARM_FILES+=("$MODELS"/ltx-2.3-22b-dev.safetensors)
fi
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
