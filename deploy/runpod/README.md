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
2. Deploy an RTX PRO 6000 Blackwell Server Edition pod (96 GB, sm_122) with the volume mounted
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

Live pricing varies — run `runpodctl gpu list | grep -i 6000` for the current rate
before committing. Storage is unchanged from RTX 4090 era.

| Usage | Cost |
|-------|------|
| Storage (monthly) | $7.00 |
| GPU rate | check `runpodctl gpu list` |

RTX PRO 6000 Blackwell Server Edition on Secure Cloud (required for network volumes).
