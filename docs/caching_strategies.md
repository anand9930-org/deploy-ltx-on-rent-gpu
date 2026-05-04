# Caching Strategies for LTX-2.3 Serverless Cold-Start Optimisation

**Date:** 2026-05-04
**Target stack:** LTX-2.3 22B · BentoML · NGC PyTorch 25.06 (torch 2.8.0a0, CUDA 12.x) · FP8 + FA3 H100 SXM5 · RunPod Serverless `min_replicas=0`
**Goal:** reduce cold-start latency without compromising output quality, prioritising high-impact, low-risk strategies.

---

## 1. Why this document exists

Live A/B (Maya I2V, 1280×1920, 241 frames, seed 42, 2026-05-04):

| | torch.compile OFF | torch.compile ON |
|---|---|---|
| Cold | 83.0 s | 109.2 s |
| Warm | 69.7 s | 59.3 s |

On `min_replicas=0`, ~70-95% of requests hit a cold pod, so compile-OFF is currently the best per-request choice — at the cost of leaving ~10 s of warm-path savings on the table.

**The persistence layer changes the math.** Caching `torch.compile` artefacts to a volume that survives pod death turns "cold-pod" into "warm-cache cold-pod." Production reports across Replicate, Baseten, vLLM, Modal, and Tensorfuse show 50-70% cold-start reduction with this pattern. For LTX-2.3 it should plausibly bring compile-ON cold from 109 s down to **~65-85 s on cache-hit pods** — below current compile-OFF baseline — while keeping the 59 s warm-path win.

The catch: a small subset of caching strategies introduce a *silent quality regression* (cached autotune choices can yield non-deterministic kernels). This document separates the safe layers from the risky ones and proposes an implementation order that captures the wins while sidestepping the bugs.

---

## 2. Current state — what's already in place

| Layer | Status | Where |
|---|---|---|
| Linux page-cache pre-warm | **In place** — selective, ~60-90 s at boot | `start.sh:40-62` |
| HuggingFace cache (`HF_HOME=/models/huggingface`) | In place; weights flat under `$MODEL_DIR` | `Dockerfile:17`, `src/download_models.py:45-148` |
| `torch.compile` regional per-block | In place; default `ENABLE_TORCH_COMPILE=1` on FP8 ≥40 GB pods | `src/pipeline.py:588-632` |
| Inductor config shims (NGC 2.8 compatibility) | In place | `src/compile_override.py:44-58` |
| Attention-callable singleton (Dynamo `obj_id` stability) | In place — fixed cross-mode recompile thrash | `src/attention_override.py:95-110` |
| Recompile-budget bump (`accumulated_recompile_limit=8192`) | In place | `src/compile_override.py` |
| TeaCache (opt-in, `ENABLE_TEACACHE=1`, OFF for I2V quality) | In place | `src/teacache.py` |
| FP8 extras header probe cache | In place (per-generator instance) | `src/pipeline.py:646` |
| **`TORCHINDUCTOR_CACHE_DIR` / `_FX_GRAPH_CACHE`** | **MISSING** — Inductor uses ephemeral `/tmp/torchinductor_root` | — |
| **`TORCHINDUCTOR_AUTOGRAD_CACHE`** | **MISSING** | — |
| **`TRITON_CACHE_DIR`** | **MISSING** — Triton uses default ephemeral location | — |
| **`CUDA_CACHE_PATH` / `CUDA_CACHE_MAXSIZE`** | **MISSING** — defaults to `~/.nv/ComputeCache`, 256 MiB cap | — |
| Persistent Inductor / Triton mount | **MISSING** | — |
| Persistent volume probe (`/runpod-volume`) in `start.sh` | **MISSING** (deploy-side path-injection only via `deploy/runpod/deploy.sh:38`) | — |
| Mega-cache (`save_cache_artifacts` portable blob) | **MISSING** | — |
| CUDA Graphs / `mode="reduce-overhead"` | **NOT ENABLED** (current is default Inductor mode) | — |
| Cross-deploy cache fingerprint / namespacing | **MISSING** | — |
| Frame-hash regression test on deploy | **MISSING** | — |

**Headline:** all the compile-time work — FX graph lowering, Triton kernel build, autotune sweep, CUDA driver JIT — is rebuilt from scratch on every cold pod. None of it persists.

---

## 3. Mental model — where the cold-start time goes

`torch.compile` produces artefacts at five logically-distinct layers, each with its own cache:

1. **Dynamo bytecode trace** — `TORCHDYNAMO_TRACING_CACHE`
2. **AOTAutograd joint graph** — `AOTAutogradCache` (`TORCHINDUCTOR_AUTOGRAD_CACHE`)
3. **Inductor lowered FX graph + generated Triton/C++ source** — `FXGraphCache` (`TORCHINDUCTOR_FX_GRAPH_CACHE`)
4. **Triton compilation** (TTIR → TTGIR → LLIR → PTX → cubin) — `TRITON_CACHE_DIR` (Inductor sets this to a subpath if unset)
5. **NVIDIA driver PTX → SASS** — `CUDA_CACHE_PATH`

**Mega-cache** (PyTorch ≥2.7, `torch.compiler.save_cache_artifacts` / `load_cache_artifacts`) packages PGO + AOTAutograd + Inductor + Triton + autotune into one transportable blob.

Cold-start cost on a 22B FP8 video model breaks down as:
- Weight load (NVMe → GPU): **~60-90 s** — page-cache pre-warm already mitigates
- Pipeline build (Python construction): **~30-60 s**
- Inductor lowering + Triton compile of every kernel: **~30-80 s** (estimate — observed 26 s of compile-ON tax over compile-OFF on Maya)
- CUDA driver PTX→SASS JIT: **~5-15 s** (mostly library-shipped fat-binaries)
- Autotune sweep (if applicable): **~5-30 s**

**Strategies in §5 address layers 2-5.** Weight-load is a separate problem (see §7).

---

## 4. Quality preservation — the safety claim

PyTorch tutorial: the cache "validates that the cache artifacts are used with the same PyTorch and Triton version, as well as, same GPU when device is set to be cuda" ([docs](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)). vLLM design doc: cache is "safe to use and will not cause unexpected behavior" because the hash includes "all material factors needed for safe reuse" ([vLLM design](https://docs.vllm.ai/en/latest/design/torch_compile/)).

**Cache hits replay the exact binary a fresh compile would have produced for that key.** Output is bit-identical; numerical drift cannot be introduced by caching in the deterministic path.

**Two documented exceptions** where cached graph diverges from freshly-traced:

1. **Triton `@autotune` non-determinism — [triton#9368](https://github.com/triton-lang/triton/issues/9368)** (Feb 2026, unfixed)
   When `TRITON_CACHE_DIR` is set + a kernel uses `@triton.autotune`, repeated runs reusing cached entries can produce non-bitwise-identical outputs. The cached *config* is reused but the result drifts. **This is the single silent quality regression vector to defend against.**
   *Mitigation:* audit the LTX hot path for `@triton.autotune` decorators; FA3 typically uses `@triton.heuristics` (deterministic), not autotune. For any kernel that does use autotune, hard-pin configs.

2. **FX graph cache + autograd second-run crash — [pytorch#144609](https://github.com/pytorch/pytorch/issues/144609), [pytorch#145377](https://github.com/pytorch/pytorch/issues/145377)**
   Hard error, not silent drift. Inference-mode (which LTX uses) is unaffected.

Everything else (FXGraphCache, AOTAutogradCache, BundledAOTAutogradCache, Mega-cache, CUDA driver JIT cache) is documented as deterministic-on-hit with no silent-divergence reports.

**Defence in depth:** add a frame-bytes hash regression test on every deploy (fixed seed + fixed prompt → SHA256 of output frames). Cheap insurance against any unforeseen drift.

---

## 5. Strategy recommendations — ranked

### S1 · Persist Inductor cache to RunPod Network Volume — **DO FIRST**

**Mechanism.** Set:
```
TORCHINDUCTOR_FX_GRAPH_CACHE=1
TORCHINDUCTOR_AUTOGRAD_CACHE=1
TORCHINDUCTOR_CACHE_DIR=/runpod-volume/inductor-cache/${TORCH_VER}-${TRITON_VER}-${CUDA_VER}-h100-${IMAGE_SHA}
TRITON_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR}/triton
```

Inductor sets `TRITON_CACHE_DIR` to a subpath of `TORCHINDUCTOR_CACHE_DIR` when unset, so layers 2-4 are covered by one mount.

**Expected impact.** 30-90 s saved on warm-cache cold start (Replicate Flux: 400 s → 150 s [62%]; Baseten: 30-90 s → 5-20 s; Tensorfuse vLLM: 294 s → 82 s [70%]; Modal: "up to an order of magnitude"; vLLM Qwen-32B dynamo step: 8 s → 0.3 s). For LTX-2.3 22B expect **20-30 s recovered** on a cache-hit cold pod.

**Risk.** Low for inference-mode. The two `pytorch#144609/145377` bugs are autograd-only.

**Quality.** Identical-on-hit by construction.

**Implementation complexity.** Small — env vars in `start.sh` + one verification command.

**Why this first.** Highest impact-to-risk ratio. Single-volume mount, single env block, gracefully degrades to current behaviour if the volume is missing (Inductor falls back to `/tmp`).

---

### S2 · CUDA driver JIT cache on volume — **TRIVIAL ADD-ON TO S1**

**Mechanism.**
```
CUDA_CACHE_PATH=/runpod-volume/.nv/ComputeCache
CUDA_CACHE_MAXSIZE=4294967296   # 4 GiB
```

Default cap is 256 MiB and default path is ephemeral `~/.nv/ComputeCache`.

**Expected impact.** Small but free. Helps cuBLAS, cuDNN, CUTLASS fat-binary kernels that ship PTX without SM-matched cubins. Estimate **~5-15 s** on first-ever cold pod, near-zero on subsequent cold pods.

**Risk.** Very low. Driver auto-invalidates on driver upgrade.

**Quality.** Identical (driver-managed, deterministic JIT).

---

### S3 · Cache fingerprint namespacing — **HARDEN S1/S2**

**Mechanism.** Bake torch+triton+CUDA+driver+image SHA into the cache directory path so a deploy with a newer torch wheel cannot collide with old entries:

```bash
TORCH_VER=$(python -c 'import torch; print(torch.__version__)')
TRITON_VER=$(python -c 'import triton; print(triton.__version__)')
CUDA_VER=$(python -c 'import torch; print(torch.version.cuda)')
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
IMAGE_SHA="${IMAGE_SHA:-unknown}"   # injected at docker build
FP="${TORCH_VER}-${TRITON_VER}-${CUDA_VER}-h100-${DRIVER}-${IMAGE_SHA}"
export TORCHINDUCTOR_CACHE_DIR="/runpod-volume/inductor-cache/${FP}"
```

**Expected impact.** Eliminates a whole class of correctness incidents from cache poisoning across upgrades.

**Risk.** Very low. Cost is ~1 extra GB-month per deployed image (cleaned up via S6 sweep).

**Why this matters.** [pytorch#144859](https://github.com/pytorch/pytorch/issues/144859) — same H100 SKU + same code can still miss because `torch_key()` differs across pip wheels. Fingerprinting lets us *deliberately* invalidate on upgrade and debug "why didn't I hit the cache" by inspecting the path.

---

### S4 · Pre-bake the cache during image build — **STABILISE UNDER BURST**

**Mechanism.** Add a "cache warmup" step to the Docker build (or a one-shot job after the first deploy) that runs a representative `generate()` call inside the container, captures the resulting `TORCHINDUCTOR_CACHE_DIR`, and writes it to the Network Volume read-only mount. Pods at runtime see a fully-warm read-only cache + per-pod r/w overlay.

**Expected impact.** Same cold-start time as S1, but **stable under scale-up storms** when N pods race to populate the same cache directory ([pytorch#119698](https://github.com/pytorch/pytorch/issues/119698) JSONDecodeError on concurrent writes).

**Risk.** Low. Pre-bake job is reproducible; if it fails, runtime falls back to S1 behaviour.

**Implementation complexity.** Medium. Needs a single warmup invocation per supported (W, H, num_frames) bucket — recommend three: 1280×720, 1920×1088, 1280×1920 (matches Maya / standard vlog / standard cinematic).

---

### S5 · Mega-cache portable blob — **PORTABILITY UPGRADE**

**Mechanism.** After warmup, call `torch.compiler.save_cache_artifacts()` and ship the resulting `.bin` blob via the image (or pull from S3 at boot). On startup, `torch.compiler.load_cache_artifacts(blob)`.

**Expected impact.** Removes filesystem-shape coupling — one blob per `(torch, triton, cuda, sm, image)` tuple, transportable across volume types. Thomas J Pfan reports 21.1 s → 8.9 s on a sample model with this approach ([blog](https://www.thomasjpfan.com/2025/04/keep-warm-with-portable-torchcompile-caches/)).

**Status in PyTorch 2.8.** Stable (RFC #143341 merged in 2.7).

**Risk.** Low. Same key semantics as on-disk cache.

**When to consider.** After S1+S3 are running and a multi-region or multi-platform deploy comes up. Skip until then.

---

### S6 · Cache lifecycle — LRU sweep and quota monitor — **OPERATIONAL**

**Mechanism.**
- Cron / boot-time sweep deletes cache fingerprints older than N days OR older than the last 3 image SHAs.
- Volume usage alarm at 80% capacity.
- Per-deploy log: cache path, hit rate (count of `*.so` artefacts loaded vs compiled), volume free space.

**Expected impact.** Prevents disk-quota exhaustion as fingerprints accumulate. Each fingerprint can hold 200 MB - 2 GB of artefacts.

**Risk.** Very low.

**When.** After S1 stabilises. Not blocking initial rollout.

---

### S7 · Audit `@triton.autotune` on the hot path — **QUALITY DEFENCE**

**Mechanism.** Grep the LTX-2 / FA3 source for `@triton.autotune` decorators. For any hits in the diffusion forward path:
- Confirm if `TRITON_CACHE_DIR` is set, kernel selection is reproducible across runs (test: same seed, two runs, frame-bytes hash matches).
- If non-deterministic, hard-pin the config (replace `@autotune([cfgs])` with the chosen `cfg` directly, or set `cache_results=False`).

**Expected impact.** Defuses [triton#9368](https://github.com/triton-lang/triton/issues/9368). LTX hot path likely uses `@triton.heuristics` (deterministic), not autotune — verify before assuming.

**Risk.** Mitigates a silent quality regression vector. Worth doing as a dedicated PR before flipping S1 in production.

---

### S8 · Frame-hash regression test on every deploy — **CONTINUOUS QUALITY GATE**

**Mechanism.** CI/CD step that runs a fixed prompt + fixed seed + fixed dimensions through `/generate_sync`, computes SHA256 of the output mp4 frames, compares to a golden hash. Block traffic shift if mismatch.

**Expected impact.** Catches *any* numerical regression — not just from caching, but from any image/wheel/driver/upstream change.

**Risk.** None.

**Implementation complexity.** Small. One `pytest` test + a stored golden hash per supported (W, H, frames) tuple. Re-bless hash on intentional model/quantisation changes.

---

### Strategies explicitly **NOT** recommended

| Strategy | Why skip |
|---|---|
| **Inductor remote Redis cache** (`TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE`) | Adds a managed-Redis dependency; Network Volume + Mega-cache covers the same need with less ops surface. |
| **CUDA Graphs `mode="reduce-overhead"`** | Capture is rebuilt every cold pod (not cached) → *adds* cold-start cost. Per-shape graph bucket explosion. Captures don't compose with TeaCache. Defer until per-step latency becomes the bottleneck. |
| **Bake weights into the image** | Image bloats to ~80 GB; CI/CD pipeline cost balloons; image pull becomes the new cold-start bottleneck. RunPod Network Volume + page-cache pre-warm is the correct shape. |
| **Service-level request memoisation** (cache identical prompts → outputs) | Doesn't fit the LTX use case (every prompt is unique by design); high storage cost; quality risk if seed drifts. |

---

## 6. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cache poisoning across torch/driver upgrades | Medium | S3 (fingerprint namespacing) |
| Concurrent write contention ([pytorch#119698](https://github.com/pytorch/pytorch/issues/119698)) | High during scale-up storms | S4 (pre-baked read-only cache + per-pod overlay) |
| Silent quality regression from autotune ([triton#9368](https://github.com/triton-lang/triton/issues/9368)) | Low-medium | S7 (audit + pin) + S8 (regression test) |
| FX cache + autograd crash ([pytorch#144609](https://github.com/pytorch/pytorch/issues/144609)) | Low (we run inference-only) | `inference_mode`; kill switch `TORCHINDUCTOR_FX_GRAPH_CACHE=0` in env |
| Disk quota exhaustion | Medium over time | S6 (LRU sweep, alarm) |
| Cross-host hit miss after copy ([pytorch#144859](https://github.com/pytorch/pytorch/issues/144859)) | Medium | Bake cache from inside the same image hash; S5 (Mega-cache) for portable case |
| Volume unavailable at boot | Low | Code path falls back to ephemeral `/tmp` (Inductor default) — degraded performance, not failure |

---

## 7. Out of scope (acknowledged but separate)

- **Weight-load cold-start (60-90 s)** — addressed by existing page-cache pre-warm. Further wins live in zstd-compressed weight bundles + memory-map, or layered baked weights. Separate workstream.
- **Pipeline build cost (30-60 s)** — Python construction time of `LTXVideoGenerator` and stages. Not addressed by Inductor caching. Optimisation possible via deferred builder + lazy stage-2 init.
- **Model download (5-10 min on fresh volume)** — already idempotent; mitigation is pinning the volume across pod rotations.

---

## 8. Verification approach

### 8.1 Local correctness
```bash
# Verify env wiring loads cleanly
python3 -c "
import torch._inductor.config as c
print('fx_graph_cache:', c.fx_graph_cache)
print('fx_graph_remote_cache:', c.fx_graph_remote_cache)
print('bundled_autograd_cache:', getattr(c, 'bundled_autograd_cache', 'n/a'))
"

# Verify cache directory is writable from container
python3 -c "
import os, tempfile
d = os.environ['TORCHINDUCTOR_CACHE_DIR']
os.makedirs(d, exist_ok=True)
with tempfile.NamedTemporaryFile(dir=d, delete=True) as f:
    f.write(b'ok')
print('writable:', d)
"
```

### 8.2 Cold-warm-warm A/B on a deployed pod
1. Deploy fresh pod with strategy S1+S2+S3 wired in. First request: **cold-cache cold pod** — measure `gen_time`.
2. Wait for pod to scale down (or kill it manually). Bring up a second pod against the same Network Volume. First request: **warm-cache cold pod** — measure `gen_time`.
3. Send a second request to the warm-cache cold pod: **warm pod** — measure `gen_time`.

Record:
- `gen_time` for each of the three runs
- Cache directory artefact count (`find $TORCHINDUCTOR_CACHE_DIR -name '*.so' | wc -l`)
- Volume usage delta
- Frame-hash equality across the three runs (S8)

Expected outcomes:

| Run | Expected gen_time |
|---|---|
| Cold-cache cold pod | ~109 s (today's compile-ON cold) |
| Warm-cache cold pod | **~65-85 s** (target) |
| Warm pod | ~59 s (today's compile-ON warm) |

If warm-cache cold ≥ cold-cache cold, the cache isn't being hit — investigate fingerprint mismatch or volume mount.

### 8.3 Quality gate (S8)
Same prompt + seed + dimensions across all three runs. Compute SHA256 of output mp4 frame bytes (decode → raw frames → hash). All three must match. Mismatch = blocker for production.

### 8.4 Long-tail soak
Run a 10-pod scale-up burst with S4 pre-baked cache. Inspect logs for `JSONDecodeError`, `pytorch#119698` markers, or any kernel-compile log lines on warm-cache pods (there should be none).

---

## 9. Proposed implementation sequence

1. **S7 audit** — search for `@triton.autotune` in LTX-2-ref + FA3 source. Estimated 30 min.
2. **S8 regression test** — add `tests/test_frame_hash.py` with one golden hash per (1280×720, 1920×1088, 1280×1920) bucket. Estimated 1 hour.
3. **S1 + S2 + S3** — env wiring in `start.sh` with fingerprint namespacing. One PR. Estimated 1-2 hours.
4. **§8 verification** — three-run A/B on a real pod, capture numbers, update `docs/latency_report.md`. Estimated 30 min of pod time + analysis.
5. **S6 sweep** — only if §8 passes; basic LRU cron. Estimated 1 hour.
6. **S4 pre-bake** — only after S1 is in production for ≥1 week with no incidents and a scale-up scenario surfaces. Estimated 4-6 hours.
7. **S5 Mega-cache** — only if multi-region or multi-platform deploy comes onto the roadmap. Defer indefinitely otherwise.

---

## 10. Open questions

1. **Network Volume availability on the production RunPod Serverless setup** — confirm `/runpod-volume` is mounted with write permission on every worker. The `deploy/runpod/deploy.sh:38` line suggests yes, but verify against current platform contract.
2. **Image SHA injection** — is there an existing `IMAGE_SHA` env or label on the docker image we ship? If not, S3 needs a build-time arg added to the Dockerfile.
3. **Acceptable warm-cache cold-start target** — is **80 s** acceptable for the first user on a cold pod? If the SLO demands <60 s, we need to layer S4+S5 on top, not just S1.
4. **Resolution buckets to support** — confirm the three (W, H) tuples to pre-warm if S4 is greenlit.

---

## 11. Critical files (for implementation)

- `start.sh:8-21` — env block, where the new env vars land.
- `Dockerfile:14-19` — `ENV` block already has `LTX_FP8_MODE` + `TORCH_LOGS`; bake `IMAGE_SHA` ARG here for S3.
- `src/pipeline.py:588-632` — torch.compile install site; no changes needed for S1-S3 (env-driven), but a verification log line would help.
- `src/compile_override.py` — log Inductor cache hit/miss summary post-compile (diagnostic).
- `tests/test_frame_hash.py` — new file for S8.

---

## 12. References

- PyTorch tutorial — Compile Time Caching Configuration: https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html
- PyTorch tutorial — Compile Time Caching: https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html
- vLLM torch.compile blog: https://vllm.ai/blog/torch-compile
- vLLM torch.compile design: https://docs.vllm.ai/en/latest/design/torch_compile/
- Modal Flux example: https://modal.com/docs/examples/flux
- Replicate torch.compile caching: https://replicate.com/blog/torch-compile-caching
- Baseten cold-starts: https://docs.baseten.co/performance/cold-starts
- Baseten torch.compile cache: https://docs.baseten.co/development/model/torch-compile-cache
- Tensorfuse cold-start playbook: https://tensorfuse.io/docs/blogs/reducing_gpu_cold_start
- Spheron PyTorch 2.6 production guide: https://www.spheron.network/blog/torch-compile-cuda-graphs-llm-inference-pytorch-2-6/
- Thomas J Pfan portable caches: https://www.thomasjpfan.com/2025/04/keep-warm-with-portable-torchcompile-caches/
- Red Hat Triton cache deep dive: https://next.redhat.com/2025/05/16/understanding-triton-cache-optimizing-gpu-kernel-compilation/
- NVIDIA CUDA fat binaries / JIT cache: https://developer.nvidia.com/blog/cuda-pro-tip-understand-fat-binaries-jit-caching/
- RunPod model caching: https://docs.runpod.io/serverless/endpoints/model-caching
- Mega-cache RFC (PyTorch #143341): https://github.com/pytorch/pytorch/pull/143341
- Triton autotune determinism (#9368): https://github.com/triton-lang/triton/issues/9368
- PyTorch #119698 (cached_autotune thread safety): https://github.com/pytorch/pytorch/issues/119698
- PyTorch #144609 / #145377 (FX cache + autograd crash): https://github.com/pytorch/pytorch/issues/144609
- PyTorch #144859 (cache miss across H100 hosts): https://github.com/pytorch/pytorch/issues/144859
