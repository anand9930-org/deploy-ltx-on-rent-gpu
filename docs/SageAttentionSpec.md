# SageAttention Spec — what ships in production

> This is the post-deployment reference for the SageAttention integration. For design rationale and as-is-vs-to-be, see [`sage-attention-integration.md`](sage-attention-integration.md).

## 1. Wheel

| Field | Value |
|---|---|
| Package | `sageattention` |
| Version | `2.2.0+cu128torch2.8` |
| Source | [Comfy-Org/wheels](https://github.com/Comfy-Org/wheels) (prebuilt) |
| ABI | cu128 / torch2.8 / cp311 / manylinux_2_34 |

Installed via direct URL in `Dockerfile` L28–32:

```dockerfile
RUN uv pip install --system --no-cache \
    "https://github.com/Comfy-Org/wheels/releases/download/sageattention-latest/sageattention-2.2.0%2Bcu128torch2.8-cp311-cp311-manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl"
```

Why not PyPI: thu-ml upstream does not publish v2.x to PyPI. Comfy-Org ships the ABI-matched wheel for our base image (`pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`).

## 2. Kernel

**Selected: `sageattn_qk_int8_pv_fp8_cuda` with `pv_accum_dtype="fp32+fp16"`** — this is SageAttention2++, the precision-safe variant recommended by the thu-ml README for precision-sensitive models.

Fallback chain (`src/sage_patch.py` L50–62):

```python
try:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda as _sage_fn
    _sage_kwargs = {"pv_accum_dtype": "fp32+fp16"}
except ImportError:
    try:
        from sageattention import sageattn_qk_int8_pv_fp16_triton as _sage_fn
        _sage_kwargs = {}
    except ImportError:
        from sageattention import sageattn as _sage_fn
        _sage_kwargs = {}
```

### Why Sage2++ and not the plain sm_90 FP8-PV kernel

`sageattn` auto-dispatches on sm_90 to `sageattn_qk_int8_pv_fp8_cuda_sm90` (pure FP8-PV). On LTX-2.3 22B combined with `QuantizationPolicy.fp8_cast()` weights, FP8-PV accumulation saturates, the DiT trajectory diverges, and the VAE decodes black / noisy video. Sage2++ keeps the INT8-QK speed win but moves PV accumulation into mixed fp32+fp16 accumulators, which fixes the saturation at a ~5–10 % step-time cost vs the raw sm_90 kernel.

## 3. How it hooks into LTX

`src/sage_patch.py` is a runtime monkey-patch. On import, `_install()`:

1. Imports `ltx_core.model.transformer.attention as _attn`.
2. Defines a `SageAttention(AttentionCallable)` class that reshapes Q/K/V to HND layout and calls `_sage_fn(...)`.
3. Overrides `AttentionFunction.to_callable` so `AttentionFunction.DEFAULT` resolves to `SageAttention()` (L149–156):

```python
_original_to_callable = _attn.AttentionFunction.to_callable

def to_callable(self):
    if self is _attn.AttentionFunction.DEFAULT:
        return SageAttention()
    return _original_to_callable(self)

_attn.AttentionFunction.to_callable = to_callable
```

Every `Attention(...)` construction inside LTX uses `attention_function=AttentionFunction.DEFAULT`, so redirecting what DEFAULT resolves to flips Sage on at every DiT block — including the spatial upscaler — with zero call-site changes.

Loaded before the pipeline is built in `src/pipeline.py`:

```python
if os.getenv("USE_SAGE_ATTENTION", "1") != "0":
    from . import sage_patch  # installs SageAttention before pipeline import
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
```

## 4. Call routing

- **Unmasked calls (every DiT block)** → Sage. (`src/sage_patch.py` L119–124)
- **Masked calls (Gemma text-encoder `embeddings_connector`)** → `PytorchAttention` / torch SDPA. (`src/sage_patch.py` L96–97)

Sage cannot consume padding masks; routing masked calls through torch SDPA is correct and costs nothing — the text encoder is ≪ 1 % of step time.

## 5. Observability

All logs use the `[SAGE_PATCH]` prefix for greppability in BentoML pod stdout.

**At install time** (`src/sage_patch.py` L65–70):

```
[SAGE_PATCH] kernel=<kernel_name> device_capability=(9, 0) sageattention_version=2.2.0+cu128torch2.8
[SAGE_PATCH] installed SageAttention as DEFAULT attention backend
```

**On first un-masked call** (`src/sage_patch.py` L107–117):

```
[SAGE_PATCH] first call: heads=... head_dim=... seq_len=... q.dtype=... k.dtype=... v.dtype=... kernel=...
```

A `WARNING` fires if `head_dim ∉ {64, 96, 128}` — Sage's CUDA-supported head-dim set. Outside that set the kernel silently falls back to Triton (slower) or miscomputes, so the warning is the first signal something is off.

**On first output** (`src/sage_patch.py` L131–142):

```
[SAGE_PATCH] first output finite: min=... max=... abs_max=...
```

If the tensor is non-finite, an `ERROR`-level log fires naming the saturating kernel — post-mortem becomes trivial.

## 6. Hardware assumption

- **Target**: sm_90 (Hopper H100 80 GB).
- **Graceful degradation**: if `sageattention` is unimportable (no CUDA, wrong arch, missing wheel) `_install()` logs and returns `"unchanged"` — LTX's native XFormers / PyTorch SDPA resolver stays in place.

## 7. Verified outputs

On H100, 1920×1088, 121 frames, 30 steps, `seed=42`:

| Job | Scenario | Time |
|---|---|---|
| `ltx_4dc6cc188c42.mp4` | Cold start (first request after pod boot) | 237 s |
| `ltx_f16a079daf19.mp4` | Warm | 142 s |
| `ltx_2f678934fcaf.mp4` | Warm (UGC-style prompt) | 141 s |

All three passed the `first output finite` check and produced valid non-black video.

## 8. Measuring perf — A/B with vs without Sage

The import is gated by `USE_SAGE_ATTENTION` in `src/pipeline.py`. Unset or `1` means Sage loads (default); `0` skips the import and LTX falls back to its native attention resolver (XFormers → PyTorch SDPA).

**Protocol:**

1. **Deploy** the image once. No rebuild needed to flip sides.
2. **Run with Sage on** (default): submit ~5 identical `POST /generate/submit` requests. Same prompt, same seed, all other params fixed. Note `generation_time_seconds` from each response. Discard the first (cold-start dominated by pipeline init). Record the median of the rest.
3. **Restart the pod with `USE_SAGE_ATTENTION=0`** set in RunPod's environment panel. Confirm pod stdout does **not** show `[SAGE_PATCH] installed ...` on boot — its absence confirms the fallback path.
4. **Re-run the same prompts/seeds.** Record the median again.
5. **Compare.** Expect Sage to be faster per step; the step-time delta scales with `num_inference_steps` so keep that fixed. Sage2++ is ~5–10 % slower than the raw sm_90 FP8-PV kernel but much more accurate — the comparison here is Sage2++ vs XFormers, not vs FP8-PV.

**Caveats:**

- The env var is read *once at module import*; flipping it requires a pod restart.
- Seeds and all generation params must match for the comparison to be meaningful. The service already defaults to `seed=42`.
- VAE decode and video encode are not attention-bound; they'll contribute the same time on both sides. The `generation_time_seconds` in the response includes them, but the delta still reflects the attention kernel change.
