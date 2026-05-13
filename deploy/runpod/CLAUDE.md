# RunPod Deployment — Step-by-Step Commands

## Prerequisites

```bash
# Install CLI
brew install runpod/runpodctl/runpodctl
# or: wget -qO- cli.runpod.net | sudo bash

# Set API key (from https://www.runpod.io/console/user/settings)
export RUNPOD_API_KEY=your_key
```

## 1. Create a Network Volume (one-time)

Network volumes persist across pod restarts. They are region-based (not machine-bound like Vast.ai).

```bash
# List datacenters with RTX 4090 availability
runpodctl datacenter list

# Create 100GB volume (volume + pod MUST be in same datacenter)
runpodctl network-volume create --name ltx-models --size 100 --datacenter US-KS-2

# Verify
runpodctl network-volume list
```

Save the volume ID — you'll use it for every pod creation.

## 2. Create Pod with Network Volume

```bash
# Source env vars
source .env

# Create pod (Secure Cloud — required for network volumes)
runpodctl create pod \
  --name ltx-video \
  --imageName anand9930/ltx-video:latest \
  --gpuType "NVIDIA GeForce RTX 4090" \
  --gpuCount 1 \
  --containerDiskSize 20 \
  --networkVolumeId <VOLUME_ID> \
  --ports "8000/http" \
  --env HF_TOKEN=$HF_TOKEN \
  --env MODEL_DIR=/runpod-volume/models \
  --env SUPABASE_URL=$SUPABASE_URL \
  --env SUPABASE_SERVICE_KEY=$SUPABASE_SERVICE_KEY \
  --env SUPABASE_BUCKET=$SUPABASE_BUCKET
```

**First boot**: Downloads ~80GB of models to `/runpod-volume/models`. Takes ~15 min.
**Subsequent boots**: Models already on volume. Service ready in ~1 min.

**Endpoint**: `https://<POD_ID>-8000.proxy.runpod.net`

## 3. Monitor

```bash
# Pod status
runpodctl pod list

# View logs
runpodctl pod logs <POD_ID>

# SSH into pod
runpodctl ssh <POD_ID>
```

## 4. Test the API

```bash
ENDPOINT="https://<POD_ID>-8000.proxy.runpod.net"

# Check readiness
curl $ENDPOINT/readyz

# Generate video (with Supabase upload — returns video_url)
curl -X POST $ENDPOINT/generate/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A golden retriever running through a sunlit meadow, cinematic, 35mm film",
    "width": 512, "height": 768, "num_frames": 25,
    "num_inference_steps": 8, "seed": 42,
    "upload_to_supabase": true
  }'

# Generate video (sync — returns MP4 file directly)
curl -X POST $ENDPOINT/generate_sync \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A cat sitting on a windowsill watching rain",
    "width": 512, "height": 768, "num_frames": 25,
    "num_inference_steps": 8
  }' --output video.mp4
```

## 5. Stop / Delete / Restart

```bash
# Stop pod (pauses GPU billing, keeps volume, still charges storage)
runpodctl pod stop <POD_ID>

# Resume stopped pod
runpodctl pod start <POD_ID>

# Delete pod (stops all compute charges, network volume persists)
runpodctl pod delete <POD_ID>

# Redeploy later — same volume, models already cached
runpodctl create pod ... --networkVolumeId <SAME_VOLUME_ID> ...
```

## 6. Delete Volume (when done permanently)

```bash
# Delete all pods using the volume first
runpodctl pod delete <POD_ID>

# Then delete the volume
runpodctl network-volume delete <VOLUME_ID>
```

## Cost Summary

| Component | Cost |
|-----------|------|
| RTX 4090 Secure Cloud | $0.59/hr |
| Network Volume (100GB) | $7.00/month |
| 4-hour session | ~$2.36 + storage |

## Key Differences from Vast.ai

| | RunPod | Vast.ai |
|--|--------|---------|
| Volume type | Network (region-based) | Machine-bound |
| Volume portability | Any pod in same datacenter | Same physical machine only |
| Cold start | FlashBoot (~seconds) | Docker pull (~minutes) |
| Endpoint URL | HTTPS proxy auto-assigned | Random IP:port |
| GPU pricing (RTX 4090) | $0.59/hr (Secure) | $0.27-0.34/hr |
| Storage pricing | $0.07/GB/mo | ~$0.07/GB/mo |
