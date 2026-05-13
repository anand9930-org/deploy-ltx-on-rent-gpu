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

**GPU requirement (`feature/RTX-6000-pro-deployment` branch):** RTX PRO 6000 Blackwell Server Edition (96 GB GDDR7, sm_122). For the legacy H100/Hopper or RTX 4090/Ada paths, see `main`.

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

RTX PRO 6000 Blackwell Server Edition on Secure Cloud. Check live rate with `runpodctl gpu list | grep -i 6000`; storage ~$7/mo. See [deploy/runpod/README.md](deploy/runpod/README.md).

### Vast.ai (cheapest hourly rate)

```bash
pip install vastai
vastai set api-key YOUR_KEY
./deploy/vast/deploy.sh
```

RTX PRO 6000 Blackwell on Vast.ai. Discover the exact `gpu_name` with `vastai search offers 'gpu_ram>=95' -o 'dph_total'`. See [deploy/vast/README.md](deploy/vast/README.md).

### Any Docker host

```bash
docker run --gpus all -p 8000:8000 \
  -e HF_TOKEN=hf_YOUR_TOKEN -e MODEL_DIR=/models \
  -v /path/to/models:/models \
  anand9930/ltx-video-blackwell:latest
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
    "aspect_ratio": "16:9",
    "num_frames": 121,
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
  -d '{"prompt": "A cat sitting on a windowsill", "aspect_ratio": "9:16", "num_frames": 121}' \
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
| `aspect_ratio` | `"16:9" \| "9:16" \| "auto"` | `"auto"` | Output is 1920×1080 (`"16:9"`) or 1080×1920 (`"9:16"`). `"auto"` derives from input image; T2V falls back to landscape. |
| `num_frames` | int | 241 | 9–257, rounded to 8k+1 |
| `seed` | int | 42 | Random seed |
| `frame_rate` | float | 24.0 | Output FPS |
| `image_url` | string \| null | null | I2V conditioning image URL (http(s); up to 50 MB; image/* content type) |
| `image_b64` | string \| null | null | I2V conditioning image base64 (alternative to `image_url`) |
| `image_frame_idx` | int | 0 | Frame index the conditioning image lands on |
| `enhance_prompt` | bool | false | No-op for ComfyUI graph (the workflow has no prompt-enhancer node) |
| `upload_to_supabase` | bool | true | Upload to Supabase (async endpoint only) |

### Quick test parameters (lower VRAM, faster)

For initial testing, use fewer frames (resolution is fixed at 1920×1080 / 1080×1920):

```json
{
  "prompt": "Your prompt here",
  "aspect_ratio": "16:9",
  "num_frames": 25,
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

Live pricing varies. RTX PRO 6000 Blackwell rates on Vast.ai have been observed in
the ~$1.50-3.00/hr range in early 2026; run `vastai search offers 'gpu_ram>=95' -o 'dph_total' | head -10`
for the current rate before committing to a long-running deployment.

| Usage | GPU | Cost (approx) |
|-------|-----|---------------|
| Single test run (10 min) | RTX PRO 6000 Blackwell | check live rate |
| Dev session (4 hours) | RTX PRO 6000 Blackwell | check live rate |
| Always-on (monthly) | RTX PRO 6000 Blackwell | check live rate |

## Local Development

```bash
# Install dependencies
uv pip install -e .

# Download models (requires GPU machine)
MODEL_DIR=./models HF_TOKEN=hf_xxx python src/download_models.py

# Start service
MODEL_DIR=./models bentoml serve service:LTXVideoService
```
