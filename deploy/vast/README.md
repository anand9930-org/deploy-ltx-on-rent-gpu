# Deploy on Vast.ai

## Prerequisites

```bash
pip install vastai
vastai set api-key YOUR_VASTAI_API_KEY
export HF_TOKEN=hf_YOUR_HUGGINGFACE_TOKEN
```

Get your Vast.ai API key from https://cloud.vast.ai/cli/

## Quick Deploy

```bash
./deploy/vast/deploy.sh
```

This will:
1. Find the cheapest RTX 4090 with 100GB+ disk and fast internet
2. Create an instance with the Docker image
3. Wait for the container to start
4. Wait for model download + pipeline load (~10-15 min on first boot)
5. Print the API endpoint when ready

For a cheaper spot instance (can be interrupted):
```bash
./deploy/vast/deploy.sh --bid 0.20
```

## Manual Deploy

```bash
# Search for offers
vastai search offers 'gpu_name=RTX_4090 gpu_ram>=23 disk_space>=100 inet_down>=200 reliability>0.95' -o 'dph_total'

# Create instance (replace OFFER_ID)
vastai create instance OFFER_ID \
  --image anand9930/ltx-video:latest \
  --env '-p 8000:8000 -e HF_TOKEN=hf_xxx -e MODEL_DIR=/models' \
  --disk 100

# Monitor
vastai show instances
vastai logs INSTANCE_ID
```

## Using Volumes (Persistent Model Storage)

Volumes keep downloaded models across instance restarts, eliminating the 10-15 min download on each boot.

```bash
# 1. Find volume offers on a machine with an RTX 4090
vastai search volumes

# 2. Create a 100GB volume
vastai create volume VOLUME_OFFER_ID --size 100

# 3. Create instance with volume mounted at /models
vastai create instance OFFER_ID \
  --image anand9930/ltx-video:latest \
  --env '-p 8000:8000 -e HF_TOKEN=hf_xxx -e MODEL_DIR=/models' \
  --disk 50 \
  --link-volume VOLUME_ID \
  --mount-path /models

# 4. First boot downloads models to /models (on the volume)
# 5. Subsequent boots on same machine skip downloads instantly
```

**Note:** Vast.ai volumes are machine-bound — they only work on the same physical machine.

## Managing Instances

```bash
# View running instances
vastai show instances

# View logs
vastai logs INSTANCE_ID

# Stop (pauses GPU billing, keeps disk)
vastai stop instance INSTANCE_ID

# Resume
vastai start instance INSTANCE_ID

# Destroy (stops all billing)
vastai destroy instance INSTANCE_ID
```

## Cost

| Usage | RTX 4090 | Approx. Cost |
|-------|----------|-------------|
| Single test | 15 min | ~$0.06 |
| Dev session | 4 hours | ~$1.00 |
| Always-on | 1 month | ~$180-250 |
| Spot instance | 1 month | ~$100-150 |
