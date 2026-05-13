# FlashAttention 3 wheel — sourcing and bumping

## TL;DR

We build the FA3 wheel ourselves inside the NGC PyTorch base image,
cache it on this repo's GitHub Releases, and curl it from `Dockerfile`
at image-build time. The cache is keyed by `(FA3 commit SHA, NGC base
tag)` — the two inputs that determine ABI compatibility — so a build
only runs the first time you use a new pair.

| Concern | Where |
|---|---|
| Local one-liner (recommended) | `scripts/runpod_build_fa3_wheel.sh` |
| Cache + build core | `scripts/build_fa3_wheel.sh` |
| Wheel storage | GitHub Releases on this repo, tag `fa3-ngc<NGC>-<FA3 short SHA>` |
| Consumption | `Dockerfile` ARGs `FA3_WHEEL_URL` + `FA3_WHEEL_SHA256` |
| Integrity | `Dockerfile` runs `sha256sum -c -` before install |
| Runtime check | Same `Dockerfile` boot-blocker that asserts NGC torch + torchvision |

## Quick path — drive a one-shot RunPod build from your Mac

```bash
./scripts/runpod_build_fa3_wheel.sh
```

That's it. The script:

1. **Cache-checks first.** If `Release fa3-ngc25.06-<sha>` already
   exists on this repo, it prints the URL+sha256 and exits in ~1 s.
   Costs $0. Re-runs are safe.
2. **Cache miss → spawns a 32 vCPU CPU pod** (`cpu3g-32-128`, 128 GB
   RAM) on RunPod, default cloudType COMMUNITY, running
   `nvcr.io/nvidia/pytorch:25.06-py3`, with sshd installed inline via
   `dockerArgs` and your RunPod-managed SSH key authorized on the pod.
   FA3 compilation is 100% CPU work (nvcc → ptxas → sm_90a PTX), so
   no GPU is needed for the build itself — saves ~80% vs an H100 pod.
3. **scps + ssh-runs** `scripts/build_fa3_wheel.sh` on the pod with
   your local `gh auth token` forwarded as `GH_TOKEN`. Build takes
   ~25-30 min.
4. **Uploads** the wheel as a GitHub Release asset on this repo with
   the sha256 in the body. The upload happens *before* the smoke import,
   so a smoke failure (PEP 668, missing import-time deps) doesn't
   cost you the 25-min rebuild — the next runner will hit the cache.
5. **Terminates the pod** via a trap. Even if you Ctrl-C, even if the
   script crashes, even if your laptop reboots mid-run, the EXIT trap
   issues `podTerminate`.
6. **Prints the Dockerfile snippet** to paste:
   ```
   ARG FA3_WHEEL_URL=https://github.com/.../<wheel>.whl
   ARG FA3_WHEEL_SHA256=<hex>
   ```

End-to-end cost on cache miss: **~$0.20-0.40** for a successful CPU
build. Cache hits cost nothing.

### Prereqs (one-time setup on your Mac)

```bash
# 1. RunPod CLI (only used to drop an apikey into ~/.runpod/config.toml)
brew install runpod/runpodctl/runpodctl   # or: curl -fsSL ...
runpodctl config --apiKey <your-runpod-key>
ls ~/.runpod/ssh/RunPod-Key-Go            # auto-created keypair

# 2. GitHub CLI authenticated to this repo
gh auth login            # or already done
gh auth status           # confirms 'repo' scope

# 3. Local CLI deps (most likely already installed)
brew install jq          # script needs jq + curl + ssh + scp + nc
```

The orchestrator reads `apikey` from `~/.runpod/config.toml`, the GH
PAT from `gh auth token`, and the SSH keypair from
`~/.runpod/ssh/RunPod-Key-Go`. Override any of those with env vars
`RUNPOD_API_KEY`, `GH_TOKEN`, `RUNPOD_SSH_KEY_PATH`.

### Flags

| Flag | What |
|---|---|
| `--yes` | Skip the spend-confirmation prompt |
| `--dry-run` | Print the GraphQL spawn payload + skip API calls |
| `--instance <id>` | Override CPU instance id (default `cpu3g-32-128`) |
| `--secure-cloud` | Use `cloudType: SECURE` (~30-50% pricier; bigger inventory) |
| `--keep-pod` | Don't terminate the pod on exit (debugging) |

### After it succeeds

Paste the printed `ARG` lines into `Dockerfile` (under
`# ---- FlashAttention 3 (sm_90 only) ----`), commit, then trigger
the image build:

```bash
gh workflow run docker-build.yml --ref feature/fp8-h100
```

## Manual path — run on an existing pod (skip RunPod orchestration)

If you already have a pod running `nvcr.io/nvidia/pytorch:25.06-py3`
for other work, copy `scripts/build_fa3_wheel.sh` onto it and run
directly. No docker-in-docker required.

```bash
# On the pod (already running nvcr.io/nvidia/pytorch:25.06-py3)
gh auth login            # one-time, needs PAT with 'repo' scope
git clone https://github.com/<owner>/<this-repo>.git
cd <this-repo>
./scripts/build_fa3_wheel.sh
```

## Background — why we build inside NGC

PyPI / community FA3 wheels link `c10::cuda::SetDevice(signed char,
bool)` — the two-arg overload, mangled `_ZN3c104cuda9SetDeviceEab`.
That overload was added to upstream torch *after* NGC 25.06's
snapshot was cut but *before* the v2.8.0 stable tag, so NGC's
`libc10_cuda.so` only exports the single-arg version. Every external
wheel fails the boot-blocker with `undefined symbol`.

CI source builds on `ubuntu-latest` (4-core or 8-core) OOM at the
CUTLASS template-instantiation step — `FLASH_ATTENTION_DISABLE_*`
kernel-trim env vars are silently ignored at the pinned upstream
SHA, forcing the full ~50-kernel build.

Building inside NGC on a RunPod CPU pod sidesteps both: ABI matches
by construction, and 32 vCPU + 128 GB RAM finishes in ~25 min.

## How the cache works

`scripts/build_fa3_wheel.sh` is the cache primitive. It:

1. Reads `FA3_COMMIT` and `NGC_TAG` (env, with defaults).
2. Computes the Release tag `fa3-ngc<NGC>-<FA3 short SHA>` —
   e.g. `fa3-ngc25.06-6c73fb50`.
3. **If a Release with that tag exists** on this repo → prints the
   URL + sha256 and exits. No build.
4. **Else** → builds natively inside the NGC pytorch container,
   `gh release create`s the wheel asset with sha256 in the body,
   prints the same URL + sha256.

The orchestrator (`runpod_build_fa3_wheel.sh`) layers RunPod
spawn/terminate on top, but the cache logic lives entirely in the
build script — so the manual path benefits from caching too.

## Bumping FA3

Change `FA3_COMMIT` to a newer flash-attention SHA:

```bash
FA3_COMMIT=<new-sha> ./scripts/runpod_build_fa3_wheel.sh
```

This recomputes the tag, hits the cache miss path, builds + uploads,
prints the new ARG snippet. The previous wheel stays available for
rollback under its old tag. Update the default `FA3_COMMIT` value at
the top of `scripts/build_fa3_wheel.sh` so future runs default to
the new SHA.

## Switching the NGC base image

```bash
NGC_TAG=25.09-py3 ./scripts/runpod_build_fa3_wheel.sh
```

Different tag, different cache entry. Both wheels stay available. But
note: bumping NGC also shifts torch ABI, CUDA version, and a number
of other image-level invariants (torchvision, anyio, torchaudio stub,
tensorrt-llm), so this is rarely a wheel-side decision in isolation.

## Rollback

- **Pod-level**: set `LTX_ATTENTION_TYPE=` (empty) — service falls
  back to torch SDPA. No image rebuild.
- **Image-level, recent regression**: point `FA3_WHEEL_URL` /
  `FA3_WHEEL_SHA256` at a prior Release tag's asset URL + sha256
  (`gh release list --repo <repo>`), commit, rebuild.
- **Image-level, full revert**: revert the Dockerfile commit; the
  placeholder defaults will fail the build, forcing an explicit choice
  rather than silently shipping a broken image.

## Debugging the orchestrator

| Symptom | What to check |
|---|---|
| Stuck in "waiting for SSH up" | Pod stuck PROVISIONING (capacity pressure) — re-run with `--secure-cloud`, or wait. |
| `nc -z` succeeds but `ssh` "Permission denied" | `~/.runpod/ssh/RunPod-Key-Go.pub` doesn't match the key authorized on the pod. Check the pubkey landed in the `PUBLIC_KEY` env var (web UI → pod → Container env). |
| Build succeeds on pod but orchestrator can't parse URL | `gh release create` failed silently — check the pod's stdout for a 4xx from GitHub. Likely a missing `repo` scope on the GH PAT. |
| Trap doesn't terminate pod | Look for "WARN: podTerminate failed" in the orchestrator's tail output. Manually: `runpodctl stop pod <id>`. |

## API surface — what LTX uses

`src/attention_override.py:34` imports:

```python
from flash_attn_interface import flash_attn_func, flash_attn_qkvpacked_func, ...
```

LTX-2 only ever calls `flash_attn_func(q, k, v)` on BF16 hdim128
forward. No code path uses paged/varlen/fp16/hdim64/hdim192/softcap/
backward, so the full kernel build is overkill but not a correctness
risk — the unused kernels are dead weight only.

