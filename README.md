# LTX-2.3 Video Generation Service

Text-to-video generation API using LTX-2.3 22B (FP8) served with BentoML. Generates high-quality videos from text prompts on a single 24GB GPU.

## Architecture

```
service.py          BentoML service (async task + sync endpoints)
src/pipeline.py     LTXVideoGenerator — model loading, VRAM optimization, inference
src/storage.py      Optional Supabase upload for generated videos
src/download_models.py  Idempotent model downloader from HuggingFace
```

**Model files (~64 GB total, downloaded on first boot):**

| File | Size | Source |
|------|------|--------|
| `ltx-2.3-22b-dev-fp8.safetensors` | 29 GB | `Lightricks/LTX-2.3-fp8` |
| `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | 7.6 GB | `Lightricks/LTX-2.3` |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | 1 GB | `Lightricks/LTX-2.3` |
| `gemma-3-12b-it-qat-q4_0-unquantized/` | 26 GB | `google/gemma-3-12b-it-qat-q4_0-unquantized` |

**GPU requirement:** RTX 4090 or L4 (24GB VRAM, Ada Lovelace architecture for FP8 support). Bigger GPUs (H100 80GB, L40S 48GB) automatically skip CPU↔GPU layer streaming for pure-GPU inference.

**Attention backend:** SageAttention 2.2.0 is installed and picked up automatically via `AttentionFunction.DEFAULT`. On Hopper (H100) it dispatches to `sageattn_qk_int8_pv_fp8_cuda_sm90` (INT8 QK + FP8 PV) for both a step-time and accuracy win over the XFormers / FA2-class fallback. If the `sageattention` wheel fails to import, we fall back to XFormers → PyTorch SDPA.

## Prerequisites

1. **HuggingFace token** — Accept the [Gemma 3 license](https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized) then get your token from https://huggingface.co/settings/tokens

2. **Docker image** — The GitHub Actions workflow pushes to GHCR on every push to `main`:
   ```
   ghcr.io/<your-username>/deploy-ltx-on-rent-gpu:latest
   ```
   Or build locally:
   ```bash
   docker build -t ltx-video .
   ```

## Deploy

### RunPod (recommended — persistent volumes, HTTPS endpoint)

```bash
brew install runpod/runpodctl/runpodctl
export RUNPOD_API_KEY=your_key
./deploy/runpod/deploy.sh
```

RTX 4090 Secure Cloud: $0.59/hr + $7/mo storage. See [deploy/runpod/README.md](deploy/runpod/README.md).

### Vast.ai (cheapest hourly rate)

```bash
pip install vastai
vastai set api-key YOUR_KEY
./deploy/vast/deploy.sh
```

RTX 4090: ~$0.27/hr + $5.60/mo storage. See [deploy/vast/README.md](deploy/vast/README.md).

### Any Docker host

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_YOUR_TOKEN -e MODEL_DIR=/models \
  -v /path/to/models:/models \
  anand9930/ltx-video:latest
```

### Test the API

Check readiness:

```bash
curl http://<EXTERNAL_IP>:<MAPPED_PORT>/readyz
```

Returns `200` when ready. Then test generation:

```bash
# Async (recommended) — submit a task
curl -X POST http://<EXTERNAL_IP>:<MAPPED_PORT>/generate/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A golden retriever running through a sunlit meadow, cinematic, 35mm film",
    "width": 512,
    "height": 768,
    "num_frames": 25,
    "num_inference_steps": 8,
    "seed": 42,
    "upload_to_supabase": false
  }'

# Response: {"task_id": "abc123..."}

# Check status
curl http://<EXTERNAL_IP>:<MAPPED_PORT>/generate/status?task_id=abc123

# Get result when done
curl http://<EXTERNAL_IP>:<MAPPED_PORT>/generate/get?task_id=abc123
```

Or use the synchronous endpoint (returns MP4 directly, holds connection open):

```bash
curl -X POST http://<EXTERNAL_IP>:<MAPPED_PORT>/generate_sync \
  -H 'Content-Type: application/json' \
  -d '{"prompt": "A cat sitting on a windowsill", "width": 512, "height": 768, "num_frames": 25, "num_inference_steps": 8}' \
  --output video.mp4
```

## API Reference

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/generate/submit` | Submit async video generation task |
| `GET` | `/generate/status?task_id=...` | Check task status |
| `GET` | `/generate/get?task_id=...` | Get task result |
| `POST` | `/generate_sync` | Synchronous generation (returns MP4) |
| `GET` | `/readyz` | Readiness probe |
| `GET` | `/healthz` | Health check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/docs` | Swagger UI |

### Request Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | string | **required** | Text description (max 2000 chars) |
| `negative_prompt` | string | *built-in* | Things to avoid |
| `width` | int | 1024 | 256–1920, rounded to nearest 64 |
| `height` | int | 1536 | 256–1920, rounded to nearest 64 |
| `num_frames` | int | 121 | 9–257, rounded to 8k+1 |
| `num_inference_steps` | int | 30 | 1–100 |
| `seed` | int | 42 | Random seed |
| `frame_rate` | float | 24.0 | Output FPS |
| `cfg_scale` | float | 3.0 | Classifier-free guidance |
| `stg_scale` | float | 1.0 | Spatial-temporal guidance |
| `rescale_scale` | float | 0.7 | Guidance rescaling |
| `upload_to_supabase` | bool | true | Upload to Supabase (async endpoint only) |

### Quick test parameters (lower VRAM, faster)

For initial testing, use smaller resolution and fewer frames:

```json
{
  "prompt": "Your prompt here",
  "width": 512,
  "height": 768,
  "num_frames": 25,
  "num_inference_steps": 8,
  "seed": 42,
  "upload_to_supabase": false
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HF_TOKEN` | Yes | — | HuggingFace token (Gemma license required) |
| `MODEL_DIR` | No | `/models` | Path to model storage |
| `SUPABASE_URL` | No | — | Supabase project URL (enables upload) |
| `SUPABASE_SERVICE_KEY` | No | — | Supabase service role key |
| `SUPABASE_BUCKET` | No | `ltx-videos` | Storage bucket name |
| `SUPABASE_URL_EXPIRY_SECONDS` | No | `604800` | Signed URL expiry (7 days) |

## Cost Estimate (Vast.ai)

| Usage | GPU | Cost |
|-------|-----|------|
| Single test run | RTX 4090 | ~$0.05 (10 min total) |
| Dev session (4 hours) | RTX 4090 | ~$1.20 |
| Always-on (monthly) | RTX 4090 | ~$210–250 |

## Local Development

```bash
# Install dependencies
uv pip install -e .

# Download models (requires GPU machine)
MODEL_DIR=./models HF_TOKEN=hf_xxx python src/download_models.py

# Start service
MODEL_DIR=./models bentoml serve service:LTXVideoService
```
