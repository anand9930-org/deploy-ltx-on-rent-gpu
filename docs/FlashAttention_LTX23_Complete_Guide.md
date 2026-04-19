# FlashAttention × LTX-2.3 — Complete Deployment Guide

> **Purpose.** This is the *how*. It operationalises the compatibility analysis in `FlashAttention_Compatibility_Analysis.md` into a concrete, step-by-step implementation plan for enabling FlashAttention 2 on Ada (RTX 4090, RTX 6000 Ada, L40S) and FlashAttention 3 on Hopper (H100, H200). No code is written by this document — it is a pre-flight plan that the implementation PR will follow.
>
> **Companion documents.**
> - `docs/FlashAttention_Compatibility_Analysis.md` — the *why*, the hardware matrix, the numerical-stability reasoning, the measured baselines this guide refers to.
>
> **Branch.** Written on `research/flash-attention-ltx23`, forked from `origin/main`. The sister branch `fix/interference-mod-api-optimized-version` (the SageAttention work) stays untouched. When this plan is executed, the implementation PR will open from a new `feat/flash-attention` branch off this research branch.

---

## Table of contents

0. [TL;DR — what to ship](#0)
1. [Prerequisites and assumptions](#1)
2. [Implementation path A: 4090 / Ada — FA2 via xformers](#2)
3. [Implementation path B: H100 / Hopper — FA3 via `flash_attn_interface`](#3)
4. [How to engage LTX-2's built-in FA3 backend](#4)
5. [The cross-attention mask fallback](#5)
6. [Dockerfile changes, per target](#6)
7. [Testing plan](#7)
8. [Deployment checklist](#8)
9. [Rollback plan](#9)
10. [Expected outcomes vs measured baselines](#10)
11. [Cost per clip and GPU selection](#11)
12. [Troubleshooting](#12)
13. [Timeline estimate](#13)

---

<a id="0"></a>

## 0. TL;DR — what to ship

Two independent, parallel production paths.

**Ada path (4090 / L40S / 6000 Ada).**
- Ensure `xformers` is in `requirements.txt`. LTX-2's `AttentionFunction.DEFAULT` auto-routes to it. Done.
- If xformers is already present: verify with `XFORMERS_VERBOSE=1` that it dispatches to FA2 kernels for our BF16 shapes. If it's using its own Cutlass kernel instead, explicitly install `flash-attn==2.8.3` and switch the enum to `XFORMERS` (xformers-wrapping-FA2) or wire up a direct `FlashAttention2` callable (see §2.3).
- Expected wall clock on 4090: 585 s → ~450 s. $/clip 0.055 → 0.043.

**Hopper path (H100 / H200).**
- Build FA3 wheel in a multi-stage Dockerfile (same pattern as we used for SageAttention's builder stage).
- Patch LTX-2's `FlashAttention3.__call__` with a two-line mask fallback (see §5) — this is a modification to an editable install, not a fork.
- Override the checkpoint config so `attention_type="flash_attention_3"` gets fed to `LTXAudioVideoModelConfigurator` (see §4).
- Expected wall clock on H100: 155 s → ~105 s. $/clip 0.108 → 0.073.

**Do not** mix FA3 with SageAttention in the same image. Pick one.

---

<a id="1"></a>

## 1. Prerequisites and assumptions

### 1.1 Base image

`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime` and its `-devel` sibling. These are the images we've been using on the SageAttention branch; they're known to work with CUDA 12.8 drivers on RunPod and to host `torch==2.8.0+cu128` natively.

Verified on the companion doc's empirical runs (2026-04-17):

- `driver: 565.57.01` (H100 pod) and `driver: 550.127.05` (earlier 4090 pods)
- `torch: 2.8.0+cu128`
- `triton: 3.4.0`

### 1.2 LTX-2 source

We run LTX-2 as an **editable install** (`uv pip install --system --no-cache -e /app/LTX-2/packages/ltx-core -e /app/LTX-2/packages/ltx-pipelines`). That means modifying files inside `/app/LTX-2/...` at build time is cheap and persistent in the image. The mask-fallback patch in §5 uses this.

### 1.3 Network-volume model cache

`/workspace/models` (or `/runpod-volume/...` on main branch's layout). First boot downloads LTX-2.3 BF16 checkpoint (46 GB), spatial upscaler (1 GB), distilled LoRA (7.6 GB), Gemma-3 12B text encoder (26 GB). Roughly 80 GB; must be on a persistent volume or we pay this cost every pod launch. Unchanged by FA adoption.

### 1.4 What's *not* assumed

- We do **not** assume xformers is already installed. That's a §2.1 task.
- We do **not** assume the cross-attention path is mask-free. §5 patches in the safety net for the case where it isn't.
- We do **not** assume FA3 builds in under 5 minutes on every CI runner. §6.2 accounts for 3–10 min depending on cores/RAM.

### 1.5 Branch / PR strategy

- **This branch (`research/flash-attention-ltx23`)**: research docs only (this file, the compatibility analysis). No code. Keeps research reusable and reviewable independent of implementation.
- **New branch `feat/flash-attention-ada`**: off this research branch. Implements the Ada path. Small PR, low risk.
- **New branch `feat/flash-attention-hopper`**: off this research branch, after Ada lands. Implements the Hopper path including the FA3 Dockerfile builder stage and the mask fallback.
- **Keep `fix/interference-mod-api-optimized-version` untouched.** It's the SageAttention reference; we don't modify it so we can re-measure against it if needed.

---

<a id="2"></a>

## 2. Implementation path A: 4090 / Ada — FA2 via xformers

### 2.1 Add xformers to requirements

On the `feat/flash-attention-ada` branch:

```diff
# requirements.txt
+ xformers==0.0.28.post3
```

Version 0.0.28.post3 is the current stable that builds against PyTorch 2.8 + CUDA 12.8. Newer versions (0.0.29, 0.0.30) may also work but are not verified on our stack.

That's it for the Ada path, *if* xformers auto-routes to FA2 for our shapes.

### 2.2 Verify xformers is actually using FA2

In one of the deploy pods, run:

```bash
XFORMERS_VERBOSE=1 python3 -c "
import torch
import xformers.ops as xops
q = torch.randn(1, 5500, 32, 128, device='cuda', dtype=torch.bfloat16)
k = torch.randn(1, 5500, 32, 128, device='cuda', dtype=torch.bfloat16)
v = torch.randn(1, 5500, 32, 128, device='cuda', dtype=torch.bfloat16)
out = xops.memory_efficient_attention(q, k, v)
print('shape', out.shape, 'dtype', out.dtype)
"
```

The verbose output will print something like:

```
xformers.ops:operator dispatch: memory_efficient_attention_forward -> fMHA@flshattF
```

`flshattF` means FA2 forward. If instead it says `cutlassF` or `decoderF`, xformers picked a different kernel.

Expected outcome: **flshattF for our 1080p shape**, based on xformers' dispatch rules (BF16 + head_dim 128 + non-causal goes to FA2 on Ada).

### 2.3 Fallback if xformers is not picking FA2

If the verbose output shows Cutlass, we add an explicit FA2 callable:

```python
# Pseudocode — goes into a small module like src/attention_backends.py
from typing import Any
import torch
try:
    import flash_attn
    _HAS_FA2 = True
except ImportError:
    _HAS_FA2 = False


class FlashAttention2Callable:
    """Drop-in AttentionCallable for LTX-2 using flash-attn directly."""
    def __call__(self, q, k, v, heads, mask=None):
        if mask is not None:
            # Fall through to xformers for masked calls.
            from ltx_core.model.transformer.attention import XFormersAttention
            return XFormersAttention()(q, k, v, heads, mask)
        b, _, dim = q.shape
        dim_head = dim // heads
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))
        out = flash_attn.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        return out.reshape(b, -1, heads * dim_head)
```

This is ~20 lines. Would live in a small `src/attention_backends.py`. The pipeline's model-loading code (see §4) then sets `attention_type` to a custom callable via the Path B override we describe there.

### 2.4 Expected outcome — Ada path

| | before FA2 | after FA2 (measured from other projects) |
|---|---|---|
| 4090 wall clock, 1080p/30/121 frames | 585 s | ~450 s (−23 %) |
| $/clip @ $0.34/hr | $0.055 | $0.043 |

**The caveat** (repeated from §8 of the compatibility doc): the 585 s baseline may already have been using FA2 via xformers' auto-dispatch. If so, the "after" number will be only marginally better. We'll find out with the §7.1 profile — it's the first thing to do.

---

<a id="3"></a>

## 3. Implementation path B: H100 / Hopper — FA3 via `flash_attn_interface`

### 3.1 The `flash_attn_interface` package

FA3 is distributed as a separate Python package compiled from `Dao-AILab/flash-attention` under the `hopper/` directory. When installed, it exposes `import flash_attn_interface` with `flash_attn_interface.flash_attn_func(q, k, v)`. This is exactly the interface LTX-2's `FlashAttention3` class already imports (line 17 of `attention.py`).

No PyPI pre-built wheel exists as of 2026-04-18. Installation is source build:

```bash
cd /tmp && git clone --depth 1 --branch main https://github.com/Dao-AILab/flash-attention.git
cd flash-attention/hopper
MAX_JOBS=4 python setup.py install
```

On a 16 GB / 4-core runner, ~4 min. On an 8 GB / 2-core runner, could stretch to 15 min.

### 3.2 Pinning a commit for reproducible builds

Rather than `--branch main`, pin a specific commit. As of this writing, the `hopper/` tree is stable enough at the HEAD of `main`, but for reproducibility:

```bash
ARG FA3_REF=abc123def456...   # pin this in the Dockerfile
cd flash-attention && git checkout ${FA3_REF}
```

This prevents "works for me, fails in CI two weeks later" drift.

### 3.3 Expected outcome — Hopper path

| | before FA3 | after FA3 (projected) |
|---|---|---|
| H100 wall clock, 1080p/30/121 frames | 155 s (SDPA default) | ~105 s (−32 %) |
| $/clip @ $2.50/hr | $0.108 | $0.073 |

Note the Hopper path is bigger in relative terms because on H100 attention is 37 % of per-step time (vs 6 % on 4090); see compatibility doc §5.

---

<a id="4"></a>

## 4. How to engage LTX-2's built-in FA3 backend

The `AttentionFunction` enum is read from the checkpoint's embedded config dict at `LTXAudioVideoModelConfigurator.from_config()` line 53 (and the video-only variant at line 112). Our LTX-2.3 22B checkpoints don't set an explicit `attention_type`, so the default fires.

Three ways to override. Ordered by cleanliness.

### 4.1 Path A — config override at load time (recommended)

Wrap the configurator with a tiny subclass that injects our preferred `attention_type` before delegating:

```python
# pseudocode for src/attention_override.py
from ltx_core.model.transformer.model_configurator import LTXAudioVideoModelConfigurator

def make_fa3_configurator():
    class _FA3Configurator(LTXAudioVideoModelConfigurator):
        @classmethod
        def from_config(cls, config: dict):
            # Shallow-copy so we don't mutate caller state
            cfg = dict(config)
            cfg["transformer"] = dict(cfg.get("transformer", {}))
            cfg["transformer"]["attention_type"] = "flash_attention_3"
            return super().from_config(cfg)
    return _FA3Configurator
```

Then wherever `LTXAudioVideoModelConfigurator` is passed into the builder (e.g. in `ti2vid_two_stages.py` or our wrapper `src/pipeline.py`), substitute the subclass.

~10 lines of code plus one substitution point per pipeline stage. Clean; traceable in git; no runtime monkey-patching.

### 4.2 Path B — post-construction module walk

Build the model as usual, then traverse and swap:

```python
from ltx_core.model.transformer.attention import Attention, FlashAttention3

def force_fa3(model):
    fa = FlashAttention3()
    n = 0
    for m in model.modules():
        if isinstance(m, Attention):
            m.attention_function = fa
            n += 1
    return n

# Call after building:
count = force_fa3(stage_1_transformer)
logger.info("Forced FA3 on %d Attention modules in stage 1", count)
```

This is the pattern the SageAttention integration used on the sister branch. It works; it's slightly less clean than Path A (runtime mutation) but has the advantage of **per-layer control** — you could swap only `attn1` and leave `attn2` on xformers, for instance.

### 4.3 Path C — environment variable, routed through a configurator factory

Build a single "attention selector" that reads an env var and produces the right configurator or module walker:

```python
def get_attention_backend():
    backend = os.getenv("LTX_ATTENTION_BACKEND", "default")
    return {
        "flash_attention_3": AttentionFunction.FLASH_ATTENTION_3,
        "xformers": AttentionFunction.XFORMERS,
        "pytorch": AttentionFunction.PYTORCH,
        "default": AttentionFunction.DEFAULT,
    }[backend]
```

Wrap Path A or B with this selector, and the deploy pod's env var controls which attention backend runs. Same ergonomics as today's `ENABLE_SAGE_ATTENTION=1` pattern; consistent with how this project has been doing feature toggles.

### 4.4 Recommendation

**Path A + C together.** Config-override is the cleanest way to actually force the attention, env-var is the cleanest way to select which config to force. Under 50 lines combined.

---

<a id="5"></a>

## 5. The cross-attention mask fallback

LTX-2's built-in `FlashAttention3.__call__` at `attention.py:111-112` raises `NotImplementedError` if a mask is passed. Self-attention (`attn1`, `audio_attn1`) in the current code path doesn't pass masks, but cross-attention (`attn2`, `audio_attn2`, cross-modal) *may* — the Gemma-3 text encoder emits padding-aware masks, and the question is whether LTX-2 threads that mask through or ignores it. This must be confirmed in the §7 test phase; until then, ship the fallback as a safety net.

### 5.1 The patch

Modify `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py:94-116` **in the editable install** (i.e. in the Docker image's `/app/LTX-2/packages/ltx-core/...`). The patch is two lines:

```diff
 class FlashAttention3(AttentionCallable):
     def __call__(
         self,
         q: torch.Tensor,
         k: torch.Tensor,
         v: torch.Tensor,
         heads: int,
         mask: torch.Tensor | None = None,
     ) -> torch.Tensor:
         if flash_attn_interface is None:
             raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

         b, _, dim_head = q.shape
         dim_head //= heads

         q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

         if mask is not None:
-            raise NotImplementedError("Mask is not supported for FlashAttention3")
+            # Fall back to xformers/SDPA for masked calls — keeps FA3 active
+            # for the fast, mask-free self-attention path while staying
+            # correct on cross-attention where Gemma's text mask may flow in.
+            return (XFormersAttention() if memory_efficient_attention else PytorchAttention())(
+                q.reshape(b, -1, heads * dim_head),
+                k.reshape(b, -1, heads * dim_head),
+                v.reshape(b, -1, heads * dim_head),
+                heads, mask,
+            )

         out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
         out = out.reshape(b, -1, heads * dim_head)
         return out
```

Net change: +5 lines, -1 line.

### 5.2 Where to apply it

Three options:

**Option 1 — Dockerfile `sed` patch after LTX-2 install.** Small, reversible, contained:

```dockerfile
RUN sed -i 's|raise NotImplementedError("Mask is not supported for FlashAttention3")|return (XFormersAttention() if memory_efficient_attention else PytorchAttention())(q.reshape(b, -1, heads * dim_head), k.reshape(b, -1, heads * dim_head), v.reshape(b, -1, heads * dim_head), heads, mask)|' \
    /app/LTX-2/packages/ltx-core/src/ltx_core/model/transformer/attention.py
```

Ugly, but single Docker layer and no LTX-2 fork required. Good for a first pass.

**Option 2 — Python-level monkey-patch at import time.** Do it in our own package's `__init__.py`:

```python
# src/__init__.py (or wherever pipeline loads)
from ltx_core.model.transformer import attention as _attn

_orig = _attn.FlashAttention3.__call__
def _patched(self, q, k, v, heads, mask=None):
    if mask is not None:
        backend = _attn.XFormersAttention() if _attn.memory_efficient_attention else _attn.PytorchAttention()
        return backend(q, k, v, heads, mask)
    return _orig(self, q, k, v, heads, None)
_attn.FlashAttention3.__call__ = _patched
```

Cleaner than sed; lives in our repo instead of as an in-image diff. Best long-term.

**Option 3 — Upstream PR.** Submit the fallback to Lightricks/LTX-2. Right thing to do for the community but doesn't help us in the short term; we still need Option 1 or 2 until the PR merges.

**Recommendation: Option 2** for the `feat/flash-attention-hopper` PR; open an upstream PR (Option 3) in parallel.

---

<a id="6"></a>

## 6. Dockerfile changes, per target

### 6.1 Ada image

Minimal — just add xformers (or `flash-attn` if §2.3 fallback is needed). Base image stays; no multi-stage build needed.

```diff
  # Dockerfile.ada  (derived from current main-branch Dockerfile)
  FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04
  ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ...
  COPY requirements.txt /tmp/requirements.txt
+ # xformers triggers FA2 for our BF16 shapes; no extra config needed.
  RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt
```

Where `requirements.txt` has gained a `xformers==0.0.28.post3` line.

Build time delta: +30 s to pip-install the pre-built xformers wheel. Image size delta: +~200 MB.

### 6.2 Hopper image

Multi-stage build. Stage 1 compiles FA3 in the `-devel` image; stage 2 copies the wheel into the `-runtime` image.

```dockerfile
# Dockerfile.hopper (skeleton)

# ---- Stage 1: Build FA3 wheel (requires CUDA compiler, only Hopper arch) ----
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS fa3-builder

ENV TORCH_CUDA_ARCH_LIST="9.0"    # Hopper-only
ENV MAX_JOBS=4                    # tune to CI runner RAM

# Pin FA3 commit for reproducibility; bump deliberately, not drift
ARG FA3_REF=<fill-in-SHA>
RUN git clone --depth 1 https://github.com/Dao-AILab/flash-attention.git /tmp/fa3 \
    && cd /tmp/fa3 && git checkout $FA3_REF \
    && cd /tmp/fa3/hopper && python setup.py bdist_wheel -d /wheels \
    && rm -rf /tmp/fa3

# ---- Stage 2: Runtime ----
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/workspace/models/huggingface

# Install the FA3 wheel produced in stage 1
COPY --from=fa3-builder /wheels /tmp/wheels
RUN pip install --no-cache-dir --no-deps /tmp/wheels/*.whl \
    && python3 -c "import flash_attn_interface; print('FA3 OK')" \
    && rm -rf /tmp/wheels

# ... rest same as current main-branch Dockerfile ...
```

Build time delta: +3–5 min for the FA3 wheel build (reusable via buildx cache). Image size delta: +~50 MB.

### 6.3 Shared: the runtime verification

Add to `start.sh` (both paths):

```bash
echo "=== Attention backend fingerprint ==="
python3 -c "
import os
import torch
backend = os.getenv('LTX_ATTENTION_BACKEND', 'default')
print(f'LTX_ATTENTION_BACKEND={backend}')
print(f'torch={torch.__version__}')

try:
    import xformers
    print(f'xformers={xformers.__version__}')
except ImportError:
    print('xformers: not installed')

try:
    import flash_attn
    print(f'flash-attn(FA2)={flash_attn.__version__}')
except ImportError:
    print('flash-attn(FA2): not installed')

try:
    import flash_attn_interface
    print(f'flash_attn_interface(FA3): installed')
except ImportError:
    print('flash_attn_interface(FA3): not installed')
"
```

Prints at container start so each run's logs tell us what backend is physically loaded.

---

<a id="7"></a>

## 7. Testing plan

Three phases — each phase's pass criterion gates the next.

### 7.1 Phase 1 — xformers dispatch audit (Ada only)

Purpose: confirm whether the baseline 585 s on 4090 is already FA2-backed via xformers.

Steps:
1. Deploy the current image on a 4090 pod.
2. In a shell, run the `XFORMERS_VERBOSE=1` probe from §2.2.
3. Capture the kernel name printed (`flshattF`, `cutlassF`, `decoderF`, `triton_flashF`, etc.).
4. Document in the test report.

Pass criterion: **xformers dispatches to `flshattF` for BF16 (1, 5500, 32, 128) inputs.** If so, Ada path is effectively already live and we'd only pursue explicit FA2 if we observe measurable speedup in Phase 3.

### 7.2 Phase 2 — mask audit (before shipping FA3)

Purpose: determine whether any real inference request flows a mask into `attention_function` for the four cross-attention modules.

Steps:
1. Apply **only** the mask-fallback patch from §5 (Option 2: Python-level monkey-patch) but **also** add a `logger.warning("mask passed to FA3 fallback for %s", module_name)` line inside the patched callable.
2. Build a test-only image, deploy on any sm_90 pod, submit one 1080p/30 request.
3. Inspect the pod log for the warning line.

Pass criterion: **either no warnings fire (cross-attn is mask-free, FA3 runs on everything) or the warnings identify exactly which modules need the fallback.** Either outcome is safe to ship; the information shapes whether we bother optimising further.

### 7.3 Phase 3 — end-to-end latency + output correctness

Purpose: validate the projected wall-clock improvements and confirm numerical correctness.

Steps (for each arch × backend combo):
1. Run the same 1080p / 121 frames / 30 steps / seed 300 request.
2. Measure `generation_time_seconds`.
3. Download the MP4. Extract frame 0 as PNG. Compute mean pixel L2 distance vs the baseline frame (`ltx_361b4675e4fc.mp4` for 4090, `ltx_954d135fb6b7.mp4` for H100) stored on the reference Supabase bucket.
4. Compute H.264 mean bytes/frame; cross-check against baseline (a large increase means noise, like the SageAttention failure).

Pass criteria:
- Wall clock matches §10 projection within ±15 %.
- Frame-0 PNG L2 distance relative to baseline: < 5 %.
- Mean bytes/frame ratio vs baseline: within 1.3× (the SAGE=1 noise run was 4.7×, so 1.3× is a conservative corruption detector).
- Manual eyeball check of 3 extracted frames (t=0, t=2, t=4 s): no brown-noise failure, recognizable panda.

### 7.4 Phase 4 — regression tests against the sister branch

Once Ada + Hopper FA paths are live, run one sanity job on both `fix/interference-mod-api-optimized-version` (SageAttention off, xformers default) and `feat/flash-attention-ada` / `feat/flash-attention-hopper`. Compare outputs. The FA branches should differ from xformers-default outputs by < 5 % pixel L2 (we accept they're numerically different but require they're visually equivalent).

---

<a id="8"></a>

## 8. Deployment checklist

### 8.1 Before first Ada deploy

- [ ] `requirements.txt` has `xformers==0.0.28.post3` pinned.
- [ ] `start.sh` prints the backend fingerprint block (§6.3).
- [ ] Phase 1 (xformers dispatch audit) completed and documented.
- [ ] Rollback tag created: `git tag pre-fa-ada v<...>` before merging PR.

### 8.2 Before first Hopper deploy

- [ ] FA3 git SHA pinned in Dockerfile (`ARG FA3_REF=...`).
- [ ] `Dockerfile.hopper` builds cleanly in CI with `MAX_JOBS=4`.
- [ ] FA3 import verification (`python3 -c "import flash_attn_interface"`) passes at the end of the Docker build, not at runtime (catches the bad wheel early).
- [ ] Mask-fallback patch (§5, Option 2) applied and unit-tested locally.
- [ ] `LTX_ATTENTION_BACKEND=flash_attention_3` env var wired into the configurator-selection code.
- [ ] Phase 2 (mask audit) completed on a 1080p/30 request and documented.
- [ ] Phase 3 (end-to-end latency + correctness) passed.
- [ ] Rollback tag created: `git tag pre-fa-hopper v<...>` before merging.

### 8.3 Shared

- [ ] Neither Dockerfile installs SageAttention alongside FA. One backend at a time in the same image.
- [ ] `FlashAttention_Compatibility_Analysis.md` and this doc are up-to-date in the PR that lands the feature (they reference measured numbers; if they change, update).
- [ ] README has a section pointing at these docs for the attention-backend toggle.

---

<a id="9"></a>

## 9. Rollback plan

FA adoption is fully reversible. For either path:

1. Revert the PR (or set `LTX_ATTENTION_BACKEND=default` in the RunPod env).
2. The image still has xformers / FA3 installed, but the configurator falls back to xformers-default on Ada and SDPA-default on Hopper.
3. Pod latency returns to the measured baselines (585 s Ada, 155 s Hopper).

**Zero-downtime rollback** because both FA2 (via xformers) and FA3 are additive — they add a new backend; they don't remove the old one. The enum falls back via `DEFAULT` if the chosen backend isn't available or misbehaves.

### 9.1 Partial rollback (mask-fallback only)

If the mask-fallback is triggered frequently (i.e. cross-attn does pass masks and we're bouncing out of FA3 on the majority of attention calls), the FA3 speedup is lost. Detection: the warning log from §7.2 fires often. Action: remove FA3 from Hopper and fall back to SDPA; keep the investigation going on whether we can thread masks natively through FA3 via `flash_attn_varlen_func`.

---

<a id="10"></a>

## 10. Expected outcomes vs measured baselines

### 10.1 Absolute wall clock

| config | wall clock (measured/projected) | source |
|---|---|---|
| 4090 SAGE=0 today | **585 s** (measured 2026-04-17, `ltx_361b4675e4fc.mp4`) | baseline |
| 4090 + FA2 (xformers already active) | 585 s (if xformers-default was already FA2) | Phase 1 outcome |
| 4090 + FA2 (if currently Cutlass) | ~490 s | projected |
| 4090 + FA2 explicit | ~450 s | projected §5.1 of compat doc |
| H100 SAGE=0 today | **155 s** (measured 2026-04-17, `ltx_954d135fb6b7.mp4`) | baseline |
| H100 + FA3 | ~105 s | projected §5.2 of compat doc |
| H100 + FA3 optimised (if cross-attn is mask-free) | ~95 s | best case |

### 10.2 $/clip (at RunPod community pricing, 2026-04-18)

| config | $/hr | s/clip | $/clip | vs 4090 SAGE=0 |
|---|---|---|---|---|
| 4090 SAGE=0 | $0.34 | 585 | $0.055 | 1.00× |
| 4090 + FA2 | $0.34 | ~450 | **$0.043** | 0.78× (22 % cheaper) |
| H100 SAGE=0 | $2.50 | 155 | $0.108 | 1.96× |
| H100 + FA3 | $2.50 | ~105 | **$0.073** | 1.33× |

Dollar-winners are in **bold**. 4090 + FA2 is the cost leader; H100 + FA3 is the latency leader.

### 10.3 Implied throughput (clips per $)

| config | clips / $1 |
|---|---|
| 4090 + FA2 | ~23 |
| 4090 SAGE=0 | 18 |
| H100 + FA3 | ~14 |
| H100 SAGE=0 | 9 |

The 4090 + FA2 combination gets you ~60 % more clips per dollar than H100 + FA3.

---

<a id="11"></a>

## 11. Cost per clip and GPU selection

Given §10 numbers, the recommendation tree:

### 11.1 If you care about cost-per-clip above all

**Ship 4090 + FA2.** Expected $0.04 – 0.05 / clip. You sacrifice latency (~7-10 min / clip) but throughput-per-dollar is best. Appropriate for:
- Batch generation pipelines
- Video libraries being prepared in advance
- Non-interactive customer-facing features

### 11.2 If you care about latency above all

**Ship H100 + FA3.** Expected ~105 s / clip at $0.07-0.08. Appropriate for:
- Interactive "generate in real-time" UX
- Customer-facing submit-and-wait flows where 90 s feels instant and 10 min feels broken
- Premium tier or time-sensitive applications

### 11.3 If you care about both

Run both in parallel. Route batch jobs to 4090 + FA2; route interactive jobs to H100 + FA3. The image builds are separate (different Dockerfiles) but the pipeline code is the same.

### 11.4 Do not pick

- **A100 40/80 GB.** LTX-2's `fp8_cast` Triton kernel requires `fp8e4nv`, which sm_80 can't emit. Stage 2 fails on LoRA fusion unconditionally. Separate compatibility problem from FA; FA doesn't fix it.
- **RTX 6000 Ada / L40S 48 GB.** Same Ada architecture as 4090 but more expensive; doesn't unlock streaming-off (we measured this — the 46 GB BF16 load peak OOMs on 48 GB cards). No advantage.
- **H200 141 GB.** Expensive, and we don't use the extra VRAM at 1080p / 121 frames. Only relevant if we move to 4K or long-form.

---

<a id="12"></a>

## 12. Troubleshooting

### 12.1 "ImportError: flash_attn_interface" at pipeline init

- Verify the FA3 wheel was built for sm_90 (not another arch) by inspecting the builder's log.
- Verify the wheel was installed in the runtime stage (`pip list | grep flash`).
- Verify `TORCH_CUDA_ARCH_LIST="9.0"` was set at build time. If it was `"8.9;9.0"`, see the SageAttention branch's debugging history for the "wgmma instruction not supported on sm_89" pitfall — you must build FA3 for a single arch.

### 12.2 Brown-noise output after switching to FA3

Unexpected — FA3 is exact attention. If this happens, the likely cause is:
1. The mask-fallback patch is **not** applied, cross-attn is passing a mask, and NotImplementedError is being caught and swallowed somewhere upstream with a uniform-noise default result. Check pod logs for `NotImplementedError`.
2. The FA3 wheel is corrupt or built for the wrong arch.

Fallback: set `LTX_ATTENTION_BACKEND=default`, redeploy. If the problem goes away, FA3 is the culprit.

### 12.3 No speedup observed

Phase 1 outcome is most likely — xformers was already routing to FA2 on 4090. Explicit FA2 via `flash-attn` can sometimes shave another few percent but not always. On H100, lack of speedup over SDPA suggests the FA3 wheel either didn't install or isn't being selected — check the `attention_type` gets into the configurator.

### 12.4 Significantly slower than baseline

Indicates the mask-fallback is firing on almost every call (i.e. cross-attention does pass masks) and we're paying double the overhead (FA3 setup + fallback to xformers per call). Detection: §7.2 warning log fires at high rate. Resolution: route cross-attn through `XFORMERS` explicitly via per-module backends (Path B in §4.2 gives per-layer control).

### 12.5 FA3 build fails in CI with "out of memory"

Lower `MAX_JOBS` from 4 to 2 (or 1). Build time goes up linearly but RAM pressure drops. nvcc at MAX_JOBS=4 on a 16 GB runner peaks around 12 – 14 GB; at MAX_JOBS=1 it's under 4 GB.

### 12.6 FA3 build fails with "Feature 'wgmma.fence' not supported on .target 'sm_89'"

You set `TORCH_CUDA_ARCH_LIST="8.9;9.0"` instead of `"9.0"`. FA3's `hopper/` tree has sm_90-only sources that can't compile for sm_89. Same failure mode we hit with SageAttention earlier this week. Drop sm_89 from the arch list.

### 12.7 Output differs from the xformers baseline

Expected. Different attention kernels = different rounding in BF16 = slightly different outputs at the same seed. If pixel L2 < 5 %, accept. If > 10 %, investigate — possibly the wrong attention is being selected (e.g. FA3 enum got resolved back to PyTorch default somewhere).

---

<a id="13"></a>

## 13. Timeline estimate

Working through the phases at a reasonable pace:

| phase | duration | calendar time |
|---|---|---|
| Phase 1 xformers audit (Ada) | ~30 min one engineer | half a day |
| Implement Ada path (§2.1), open PR | ~30 min | half a day incl. review |
| Deploy Ada + Phase 3 validation | ~1 h | half a day |
| Implement Hopper Dockerfile (§6.2), first build | ~2 h including CI debugging | one day |
| Implement mask-fallback (§5.2) + configurator override (§4.1) | ~2 h | half a day |
| Phase 2 mask audit | ~1 h | half a day |
| Deploy Hopper + Phase 3 validation | ~1 h | half a day |
| Docs updates, release notes, one round of review | ~2 h | one day |

**Total ≈ 3 – 4 working days** end-to-end, assuming no CI surprises or unexpected shape incompatibilities. The Ada path alone is ~1 day; the Hopper path adds ~2 – 3 days.

---

## Appendix A — file/line references

- LTX-2 `AttentionFunction` enum: `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py:119-136`
- LTX-2 `FlashAttention3` class: `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py:94-116`
- LTX-2 `XFormersAttention` class: `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py:49-91`
- LTX-2 configurator: `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py:53` (audio-video) and `:112` (video-only)
- Attention.forward call site: `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py:232`
- Current main-branch Dockerfile: `/Users/anand/Documents/my-projects/deploy-ltx-on-rent-gpu/Dockerfile`
- Reference SageAttention Dockerfile (multi-stage builder, to mirror for FA3): sister branch `fix/interference-mod-api-optimized-version`
- Reference empirical runs: sister-branch commit `7dd4120` and its Supabase outputs `ltx_361b4675e4fc.mp4` (4090), `ltx_954d135fb6b7.mp4` (H100 SAGE=0), `ltx_988d63db487f.mp4` (H100 SAGE=1 noise).

## Appendix B — upstream PRs worth opening

- Dao-AILab/flash-attention: request a PyPI pre-built wheel for the `hopper/` subpackage to save every downstream project the 5-min nvcc build. Low probability of acceptance but zero cost to ask.
- Lightricks/LTX-2: the mask-fallback (§5.1) is broadly useful. Clean PR, likely merged.

## Appendix C — what to carry forward to the implementation PR

- This doc, as-is, in `docs/FlashAttention_LTX23_Complete_Guide.md`.
- The compatibility analysis in `docs/FlashAttention_Compatibility_Analysis.md`.
- A minimal sample `src/attention_override.py` illustrating Path A + C from §4.
- A CHANGELOG entry describing the new env var `LTX_ATTENTION_BACKEND` and its allowed values.
- Regression test frames (t=0 extracted PNGs from each backend × arch combo) so visual parity can be inspected at review time.
