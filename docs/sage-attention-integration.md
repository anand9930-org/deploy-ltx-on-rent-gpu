# SageAttention Integration — As-Is vs To-Be

## Goal

Replace the auto-selected XFormers / PyTorch SDPA attention backend with **thu-ml SageAttention 2.2.0** so fresh container deployments on **Hopper H100 80 GB** get a step-time + accuracy win with zero operator action — no flags, no env vars.

### Why Hopper + FP8

- On sm_90, `sageattn` auto-dispatches to `sageattn_qk_int8_pv_fp8_cuda_sm90` (INT8 QK + FP8 PV). thu-ml's README reports this kernel matches FlashAttention3-FP8 speed with better accuracy on video-DiT benchmarks.
- The existing `QuantizationPolicy.fp8_cast()` stores weights in FP8 and upcasts to BF16 inside `Fp8CastLinear.forward`. Sage wants FP16/BF16 inputs — BF16 activations from the upcast satisfy Sage's input contract with no extra plumbing.
- H100 is 80 GB, so `src/pipeline.py:182` (`streaming = 2 if gpu_vram_gb < 40 else None`) already resolves to pure-GPU (`streaming=None`, `max_batch_size=1`). No streaming / offload changes needed.

### CUDA / container versions

| Component | Required by Sage 2.2.0 | Current in repo | Status |
|---|---|---|---|
| CUDA runtime | ≥12.3 (Hopper FP8) | 12.8 | ✔ clears with headroom |
| PyTorch | ≥2.3.0 | 2.8.0 | ✔ |
| Triton | ≥3.0.0 | transitively via torch 2.8 | ✔ |
| Python | ≥3.9 | 3.11 (container) / 3.12 (local) | ✔ |

Base image stays `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`. A CUDA-13 image would only matter for Blackwell; we're deploying on Hopper.

---

## Implementation approach — runtime monkey-patch

Key constraint: **`LTX-2-ref/` in the repo is gitignored** and the Dockerfile clones a fresh `Lightricks/LTX-2` inside the container. That means direct edits to `attention.py` in the reference tree would never reach production. We therefore ship a small `src/sage_patch.py` that, on import, injects a `SageAttention` class into the already-loaded `ltx_core.model.transformer.attention` module and overrides `AttentionFunction.DEFAULT.to_callable()` to return it.

Every `Attention(...)` construction site inside `ltx-core` (`model.py:323`, `transformer.py:45/55/70/80/96/108`, plus the top-level `Transformer(attention_function=AttentionFunction.DEFAULT)` default at `transformer.py:32`) flows through `AttentionFunction.DEFAULT` — so redefining what DEFAULT resolves to flips Sage on everywhere with zero call-site changes.

Import order (in `src/pipeline.py.__init__`):

```python
from . import sage_patch  # installs SageAttention before pipeline import
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
```

`sage_patch._install()` does:
1. `try: from sageattention import sageattn; except ImportError: log and return "unchanged"`.
2. `from ltx_core.model.transformer import attention as _attn` (the module is loaded lazily when the pipeline imports it; by capturing it here, we can mutate its namespace).
3. Define `SageAttention(AttentionCallable)` that reshapes to `(B, H, N, D)` and calls `sageattn(..., tensor_layout="HND", is_causal=False)`.
4. Attach `_attn.SageAttention = SageAttention` and `_attn.sageattn = sageattn`.
5. Wrap `AttentionFunction.to_callable` so `DEFAULT` returns `SageAttention()` and all other enum values delegate to the original method.

If the `sageattention` wheel import fails (no wheel for macOS, no CUDA, etc.), the patch logs and no-ops — XFormers/PyTorch remain the default.

---

## As-Is vs To-Be (per file)

### 1. `src/sage_patch.py` — NEW

**As-is**: does not exist.

**To-be**: 80-line module that performs the injection described above. Pure monkey-patch; no upstream edits.

### 2. `src/pipeline.py`

**As-is**: first import inside `LTXVideoGenerator.__init__` is `from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline`.

**To-be**: one-line addition immediately above it — `from . import sage_patch`. Nothing else changes.

### 3. `Dockerfile`

**As-is**:
```dockerfile
RUN git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2 \
    && uv pip install --system --no-cache \
        -e /app/LTX-2/packages/ltx-core \
        -e /app/LTX-2/packages/ltx-pipelines
```

**To-be**: added a new layer immediately after the LTX-2 install:

```dockerfile
# ---- SageAttention (Hopper sm_90 FP8 kernel auto-dispatch) -----------------
RUN uv pip install --system --no-cache sageattention==2.2.0 --no-build-isolation
```

Base image unchanged.

### 4. `bentofile.yaml`

**As-is**: `python.packages` has 5 entries.

**To-be**: appended `sageattention==2.2.0` so Bento-packaged builds match the Dockerfile.

### 5. `pyproject.toml`

**As-is**: 5 core deps + `[project.optional-dependencies].test`.

**To-be**: **unchanged**. `sageattention` has no macOS wheel; listing it here made `uv run pytest` fail resolution on dev machines. The container installs it via Dockerfile and the Bento bundle via `python.packages`, which covers both ship targets.

### 6. `.env.example`

**As-is / to-be**: **unchanged** — no new environment variables, per direction.

### 7. `README.md`

**To-be**: added a short paragraph under the GPU-requirement line noting that SageAttention 2.2.0 is installed and auto-selected on GPUs where the wheel imports (Hopper → `sageattn_qk_int8_pv_fp8_cuda_sm90`), with XFormers → PyTorch SDPA as fallback.

### 8. `src/pipeline.py` streaming / VRAM path

**As-is**: `streaming = 2 if gpu_vram_gb < 40 else None; max_batch_size = 4 if streaming else 1` (lines 181–193).

**To-be**: **unchanged**. On H100 80 GB this already yields `streaming=None`, `max_batch_size=1` — aligned with the "don't raise max_batch_size on big GPUs" memory note.

### 9. Precision

**As-is**: `QuantizationPolicy.fp8_cast()` (`src/pipeline.py:53-59`).

**To-be**: **unchanged**. Produces the BF16 activations Sage expects on input.

---

## Runtime behaviour matrix

| Scenario | As-is resolved backend | To-be resolved backend |
|---|---|---|
| H100 container, sage wheel installed | `XFormersAttention` | **`SageAttention` → sm_90 INT8 QK + FP8 PV** |
| H100 container, sage wheel import fails | `XFormersAttention` | `XFormersAttention` (fallback) |
| 24–32 GB GPU (L4 / 4090) | `XFormersAttention` + streaming | `SageAttention` + streaming (streaming unaffected) |
| macOS local dev, pytest | `PytorchAttention` (unreachable — tests don't load pipeline) | same — `sage_patch` never runs |

---

## Expected payoff

Lower bound from RTX 5090 Laptop, 121 frames, 8 steps (HF benchmark thread):

| config | s/it | total |
|---|---|---|
| FP8, no Sage | 7.08 | 56 s |
| FP8 + Sage (KJ patch) | 5.95 | 47 s |

That's ~16 % step-time reduction at FP8. On H100 the reported win is primarily accuracy at matched speed vs FA3-FP8 (thu-ml CogVideoX1.5 benchmark). Today our production backend is XFormers (FA2-class), so the H100 gains should include both a step-time cut and the quality improvement.

---

## Verification

1. `uv run pytest tests/test_pipeline.py` → 11 passed (confirmed on macOS; `sage_patch` is not exercised because tests don't construct the generator).
2. In the container on H100, in a Python shell:
   ```python
   from src import sage_patch
   print(sage_patch.status)                  # expected: "installed"
   from ltx_core.model.transformer.attention import AttentionFunction
   print(type(AttentionFunction.DEFAULT.to_callable()).__name__)
   # expected: SageAttention
   ```
3. End-to-end on H100 (1024×1536, 121 frames, 30 steps, FP8): compare wall-clock against a run where the Sage wheel is uninstalled (forces `status = "unchanged"`, falls back to XFormers). Expect ≥10 % step-time reduction.
4. Visual parity with fixed seed: Sage's INT8-QK / FP8-PV is reported lossless on CogVideoX; drift should be imperceptible.

## Out of scope

- NVFP4 weights (separate `Lightricks/LTX-2.3-nvfp4` checkpoint).
- CUDA 13 / SageAttention3 (Blackwell only).
- `torch.compile` wrapping of the DiT.
- Removing streaming code (kept as 24–32 GB fallback).
