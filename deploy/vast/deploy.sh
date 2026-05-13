#!/bin/bash
# Deploy LTX-2.3 video generation service to Vast.ai with persistent volume.
#
# Usage:
#   ./deploy/vast/deploy.sh                    # default: cheapest RTX 4090
#   ./deploy/vast/deploy.sh --bid 0.20         # interruptible (spot) instance
#   ./deploy/vast/deploy.sh --offer 33953886   # specific offer ID
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY
#   .env file with HF_TOKEN, SUPABASE_URL, SUPABASE_SERVICE_KEY, SUPABASE_BUCKET

set -e

# ---- Configuration ----------------------------------------------------------
IMAGE="anand9930/ltx-video:latest"
VOLUME_NAME="ltx-models"
VOLUME_SIZE=100  # GB
DISK=50          # local disk (small — models go on volume)
GPU_QUERY='gpu_name=RTX_4090 gpu_ram>=23 disk_space>=50 inet_down>=200 reliability>0.95'

# ---- Load .env if present ---------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading env vars from .env"
    set -a
    source "$PROJECT_ROOT/.env"
    set +a
fi

# ---- Parse args -------------------------------------------------------------
BID_PRICE=""
OFFER_ID=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --bid) BID_PRICE="--bid_price $2"; shift 2 ;;
        --offer) OFFER_ID="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Validate ---------------------------------------------------------------
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HF_TOKEN not set. Add it to .env or export it."
    exit 1
fi
if ! command -v vastai &> /dev/null; then
    echo "Error: vastai CLI not found. Install with: pip install vastai"
    exit 1
fi

# ---- Build env string -------------------------------------------------------
ENV_STR="-p 8000:8000 -e HF_TOKEN=${HF_TOKEN} -e MODEL_DIR=/models"
[ -n "$SUPABASE_URL" ] && ENV_STR="$ENV_STR -e SUPABASE_URL=${SUPABASE_URL}"
[ -n "$SUPABASE_SERVICE_KEY" ] && ENV_STR="$ENV_STR -e SUPABASE_SERVICE_KEY=${SUPABASE_SERVICE_KEY}"
[ -n "$SUPABASE_BUCKET" ] && ENV_STR="$ENV_STR -e SUPABASE_BUCKET=${SUPABASE_BUCKET}"

# ---- Check for existing volume ----------------------------------------------
echo "Checking for existing volume '$VOLUME_NAME'..."
EXISTING_VOL=$(vastai show volumes --raw 2>/dev/null | python3 -c "
import sys, json
try:
    vols = json.load(sys.stdin)
    for v in vols:
        if v.get('label') == '$VOLUME_NAME':
            print(v['id'])
            break
except: pass
" 2>/dev/null)

if [ -n "$EXISTING_VOL" ]; then
    echo "Found existing volume: $VOLUME_NAME (ID: $EXISTING_VOL)"
    ENV_STR="$ENV_STR -v $VOLUME_NAME:/models"
else
    echo "No existing volume found. Will create one on the target machine."
fi

# ---- Find offer --------------------------------------------------------------
if [ -z "$OFFER_ID" ]; then
    echo "Searching for cheapest RTX 4090..."
    OFFER_ID=$(vastai search offers "$GPU_QUERY" -o 'dph_total' --raw 2>/dev/null \
        | python3 -c "import sys,json; offers=json.load(sys.stdin); print(offers[0]['id'])")
fi

PRICE=$(vastai search offers "$GPU_QUERY" -o 'dph_total' --raw 2>/dev/null \
    | python3 -c "
import sys,json
offers=json.load(sys.stdin)
for o in offers:
    if str(o['id']) == '$OFFER_ID':
        print(f\"\${o['dph_total']:.2f}/hr\")
        break
" 2>/dev/null || echo "unknown")

echo "Using offer: $OFFER_ID ($PRICE)"

# ---- Create volume if needed -------------------------------------------------
if [ -z "$EXISTING_VOL" ]; then
    echo "Creating ${VOLUME_SIZE}GB volume '$VOLUME_NAME' on machine..."
    # Search for volume offer on the same machine
    VOL_RESULT=$(vastai search volumes --raw 2>/dev/null | python3 -c "
import sys, json
vols = json.load(sys.stdin)
# Find a volume offer (any available)
if vols:
    print(vols[0]['id'])
" 2>/dev/null)

    if [ -n "$VOL_RESULT" ]; then
        vastai create volume "$VOL_RESULT" -s "$VOLUME_SIZE" -n "$VOLUME_NAME" 2>&1
        echo "Volume '$VOLUME_NAME' created."
        ENV_STR="$ENV_STR -v $VOLUME_NAME:/models"
    else
        echo "Warning: Could not create volume. Proceeding without persistent storage."
        echo "Models will re-download on each boot (~15 min)."
    fi
fi

# ---- Create instance ---------------------------------------------------------
echo "Creating instance..."
RESULT=$(vastai create instance "$OFFER_ID" \
    --image "$IMAGE" \
    --env "$ENV_STR" \
    --disk "$DISK" \
    --entrypoint /app/start.sh \
    $BID_PRICE \
    --raw 2>&1 \
    --args)

INSTANCE_ID=$(echo "$RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['new_contract'])")
echo "Instance created: $INSTANCE_ID"

# ---- Wait for running --------------------------------------------------------
echo "Waiting for container to start..."
while true; do
    STATUS=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null \
        | python3 -c "import sys,json; print(json.load(sys.stdin).get('actual_status','unknown'))")
    echo "  Status: $STATUS"
    if [ "$STATUS" = "running" ]; then break; fi
    if [ "$STATUS" = "exited" ]; then
        echo "Error: Instance exited. Check logs: vastai logs $INSTANCE_ID"
        exit 1
    fi
    sleep 15
done

# ---- Get endpoint ------------------------------------------------------------
ENDPOINT=$(vastai show instance "$INSTANCE_ID" --raw 2>/dev/null | python3 -c "
import sys,json
d=json.load(sys.stdin)
ip=d['public_ipaddr']
port=d['ports']['8000/tcp'][0]['HostPort']
print(f'http://{ip}:{port}')
")

echo ""
echo "Container running. Endpoint: $ENDPOINT"
if [ -n "$EXISTING_VOL" ]; then
    echo "Volume mounted — models may already be cached (~1 min to ready)."
else
    echo "First boot — downloading models (~15 min to ready)."
fi

# ---- Wait for readiness ------------------------------------------------------
while true; do
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" -m 5 "$ENDPOINT/readyz" 2>/dev/null || echo "000")
    if [ "$HTTP_CODE" = "200" ]; then
        echo ""
        echo "=========================================="
        echo "  Service is READY!"
        echo "  API:     $ENDPOINT"
        echo "  Docs:    $ENDPOINT/docs"
        echo "  Metrics: $ENDPOINT/metrics"
        echo ""
        echo "  Instance: $INSTANCE_ID"
        echo "  Stop:     vastai stop instance $INSTANCE_ID"
        echo "  Destroy:  vastai destroy instance $INSTANCE_ID"
        echo "  Logs:     vastai logs $INSTANCE_ID"
        echo "=========================================="
        break
    fi
    echo "  Not ready (HTTP $HTTP_CODE) — loading..."
    sleep 30
done
