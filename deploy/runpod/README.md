# Deploy on RunPod

## Prerequisites

```bash
brew install runpod/runpodctl/runpodctl
export RUNPOD_API_KEY=your_key   # from https://www.runpod.io/console/user/settings
```

## Quick Deploy

```bash
./deploy/runpod/deploy.sh
```

This will:
1. Create a 100GB network volume (first time only, $7/month)
2. Deploy an RTX 4090 pod with the volume mounted
3. Download models on first boot (~15 min), cached on subsequent boots (~1 min)
4. Print the HTTPS endpoint when ready

Options:
```bash
./deploy/runpod/deploy.sh --datacenter EU-RO-1      # specific region
./deploy/runpod/deploy.sh --volume-id abc123         # reuse existing volume
```

## Manual Deploy

See [CLAUDE.md](CLAUDE.md) for step-by-step CLI commands.

## Cost

| Usage | Cost |
|-------|------|
| Single 4-hour session | ~$2.36 |
| Storage (monthly) | $7.00 |
| Always-on (monthly) | ~$425 |

RTX 4090 on Secure Cloud: $0.59/hr (required for network volumes).
