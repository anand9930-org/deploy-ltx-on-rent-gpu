# Vast.ai Deployment — Step-by-Step Commands

## Prerequisites

```bash
pip install vastai
vastai set api-key YOUR_API_KEY
```

## 1. Create a Persistent Volume (one-time)

Volumes are **machine-bound** — they persist across instance restarts but only work on the same physical machine.

```bash
# Search for volume offers (look for Norway host_id 1276, or pick any machine)
vastai search volumes "disk_space>100"

# Create an 100GB volume (pick an offer_id from the search above)
vastai create volume <VOLUME_OFFER_ID> -s 100 -n ltx-models

# Verify it was created
vastai show volumes
```

Save the volume name (`ltx-models`) — you'll use it in every instance creation.

## 2. Find an RTX 4090 on the Same Machine

```bash
# Search for RTX 4090 offers — filter by the same machine as your volume
vastai search offers 'gpu_name=RTX_4090 gpu_ram>=23 disk_space>=50 inet_down>=200 reliability>0.95' -o 'dph_total'
```

Pick an offer on the **same machine_id** where you created the volume.

## 3. Create Instance with Volume

```bash
# Full command with all env vars and volume mounted at /models
vastai create instance <OFFER_ID> \
  --image anand9930/ltx-video:latest \
  --env "-p 8000:8000 \
    -e HF_TOKEN=$HF_TOKEN \
    -e MODEL_DIR=/models \
    -e SUPABASE_URL=$SUPABASE_URL \
    -e SUPABASE_SERVICE_KEY=$SUPABASE_SERVICE_KEY \
    -e SUPABASE_BUCKET=ltx-videos \
    -v ltx-models:/models" \
  --disk 50 \
  --entrypoint /app/start.sh \
  --args
```

**First boot**: Downloads ~80GB of models to `/models` (on the volume). Takes ~15 min.
**Subsequent boots**: Models already on volume. Service ready in ~1 min.

## 4. Monitor Instance

```bash
# Check status
vastai show instances

# View logs (download progress, errors)
vastai logs <INSTANCE_ID>

# Get endpoint URL
vastai show instance <INSTANCE_ID> --raw | python3 -c "
import sys, json
d = json.load(sys.stdin)
ip = d['public_ipaddr']
port = d['ports']['8000/tcp'][0]['HostPort']
print(f'http://{ip}:{port}')
"
```

## 5. Test the API

```bash
ENDPOINT="http://<IP>:<PORT>"

# Check readiness
curl $ENDPOINT/readyz

# Generate video (with Supabase upload — returns video_url)
curl -X POST $ENDPOINT/generate/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A golden retriever puppy running through a sunlit meadow, cinematic, 35mm film",
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

## 6. Stop / Destroy / Restart

```bash
# Stop instance (pauses GPU billing, keeps disk + volume)
vastai stop instance <INSTANCE_ID>

# Resume stopped instance
vastai start instance <INSTANCE_ID>

# Destroy instance (stops GPU billing, volume persists separately)
vastai destroy instance <INSTANCE_ID>

# Create new instance on same machine — volume still has models
# (use same command from step 3)
```

## 7. Delete Volume (when done permanently)

```bash
# Must destroy all instances using the volume first
vastai destroy instance <INSTANCE_ID>

# Then delete the volume
vastai delete volume --template-id <VOLUME_ID>

# Verify
vastai show volumes
```

## Quick Reference

| What | Command |
|------|---------|
| Search GPUs | `vastai search offers 'gpu_name=RTX_4090 ...' -o 'dph_total'` |
| Search volumes | `vastai search volumes "disk_space>100"` |
| Create volume | `vastai create volume <ID> -s 100 -n ltx-models` |
| Create instance | See step 3 above |
| Check status | `vastai show instances` |
| View logs | `vastai logs <ID>` |
| Stop (keep volume) | `vastai stop instance <ID>` |
| Destroy (keep volume) | `vastai destroy instance <ID>` |
| Delete volume | `vastai delete volume --template-id <ID>` |

## Norway Machines (Proven Reliable)

These Norway machines (host_id 1276) have worked reliably in testing:

| Offer Pattern | Machine | VRAM | Speed |
|--------------|---------|------|-------|
| `machine_id=838` | Norway 1x RTX 4090 | 24GB | ~800 Mbps |
| `machine_id=4905` | Norway 1x RTX 4090 | 24GB | ~860 Mbps |
| `machine_id=8328` | Norway 1x RTX 4090 | 24GB | ~875 Mbps |

Filter for these: `vastai search offers 'gpu_name=RTX_4090 reliability>0.95' -o 'dph_total' | grep Norway`
