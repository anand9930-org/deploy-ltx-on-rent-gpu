# SageAttention Deep Dive — why Sage2++ is slower than XFormers on our shape

## TL;DR

On H100 at 1920×1088 / 121 frames / 30 steps / seed 42, Sage2++ warm = **141 s** vs XFormers warm = **133 s** (~6 % slower). The Sage2++ install is correct — kernel fires, head_dim is supported, all Attention sites route through it. The slowdown stacks from three independent factors:

1. **Our baseline is already FA-class.** XFormers on H100+BF16 dispatches into FlashAttention 2/3 internally. We're not beating vanilla SDPA; we're racing hand-tuned FA3 on Hopper.
2. **`pv_accum_dtype="fp32+fp16"` is a precision-safety mode, not a speed mode.** The raw `sageattn_qk_int8_pv_fp8_cuda_sm90` kernel is what benchmarks faster than FA3 — and that's the one that saturated and produced black video on LTX-2.3 22B.
3. **Our shape is outside Sage's winning regime.** thu-ml's H200 benchmark at B=4, N=16 k, D=128 already shows Sage ~2 % behind FA3-FP8. At B=1 (ours), the fixed INT8-QK quantization overhead (~4 % per call) amortizes worse.

The pieces are all correct. Sage just isn't the right backend for this specific shape on H100.

## 1. Kernel is firing correctly

| Check | Value | File:Line |
|---|---|---|
| `head_dim` | 128 (in Sage CUDA-supported set {64, 96, 128}) | `LTX-2-ref/.../model.py:42` |
| `num_attention_heads` | 32 | `LTX-2-ref/.../model.py:41` |
| `inner_dim` | 32 × 128 = 4096 | — |
| `seq_len` at first call | ≈ 130,560 tokens (240 × 34 × 16) | derived from 1920×1088×121 |
| Attention construction sites routed through `DEFAULT` | all 6 per DiT block + Gemma connector | `transformer.py:38,48,63,73,89,101`, `embeddings_connector.py:26` |
| Cached-callable overhead | `to_callable()` resolved **once at `Attention.__init__`**, not per forward | `attention.py:154` |
| Spatial upscaler | same `Transformer` class, same `DEFAULT` — Sage applies there too | — |

No silent Triton fallback. No re-resolution cost. The monkey-patch hits every DiT block in both the 22 B transformer and the spatial upscaler.

## 2. The baseline is FA3, not SDPA

`XFormersAttention.__call__` (`attention.py:65–89`) calls `xformers.ops.memory_efficient_attention(q, k, v, attn_bias=mask, p=0.0)` in BMH layout (`[B, M, H, K]`). On sm_90 with BF16 inputs, xformers dispatches internally to FA2/FA3.

`PytorchAttention.__call__` (`attention.py:44`) calls `F.scaled_dot_product_attention`, which on H100+BF16 uses cuDNN's FA3 backend.

So "without Sage" on H100 means "with FA3." That is not a soft baseline. thu-ml's own benchmarks show Sage beating FA3 only at **larger batch** and/or **longer sequences** than ours.

## 3. Sage2++'s fp32+fp16 accumulation is the speed tax

Per the Sage2++ paper (arXiv 2505.21136):

- `pv_accum_dtype="fp32"` → `mma.f32.f8.f8.f32` → ~2× faster than FP16 accum
- `pv_accum_dtype="fp32+fp16"` → `mma.f16.f8.f8.f16` + periodic FP32 spill → ~4× faster than FP16 accum, but slower than default FP8-PV
- Default FP8-PV (`sageattn_qk_int8_pv_fp8_cuda_sm90`) — the fastest and the one that benchmarked at or above FA3-FP8 in thu-ml numbers — is exactly the one that saturated on LTX combined with `QuantizationPolicy.fp8_cast()` weights, producing black video.

**The precision fix and the speed win are the same thing, flipped.** We can't keep both without a different kernel.

## 4. Wrapper overhead in `src/sage_patch.py`

Our `SageAttention.__call__` (L96–L144):
- 3× `.view(...).transpose(1, 2)` on Q/K/V (HND layout)
- 2× `.to(v.dtype)` casts
- Kernel call
- Output `.transpose(1, 2).reshape(...)`

`XFormersAttention.__call__` (`attention.py:65–90`):
- 1× `.view(...)` on Q/K/V (BMH layout)
- 2× `.to(v.dtype)` casts (same as ours)
- Kernel call
- Output `.reshape(...)`

**Delta: we do 4 extra transposes** per attention forward. On a 130 k-token × 4096-dim BF16 tensor (~1 GB per Q/K/V), that's real memory I/O, though probably not the dominant factor — a few ms per call, 6 calls per block, 28 blocks, 30 steps ≈ a handful of seconds across the whole generation. Real but not the whole story.

## 5. Shape regime

Sage benchmark defaults (README / paper):
- Batch: 4
- Heads: 32
- Head dim: 128
- Seq lens tested: 1 k, 2 k, 4 k, 8 k, 16 k, 32 k

Published data point at **B=4, N=16 k, D=128 on H200**: Sage ~882 TFLOPS vs FA3-FP8 ~898 TFLOPS → Sage **−2 %**.

Our runtime shape: **B=1, N≈130 k, D=128**. The long sequence is favourable (normally Sage's strength), but B=1 makes the fixed per-call INT8-QK quantization + smoothing cost (~4 % of kernel time per the Sage1 paper) amortize worse. Benchmarks at B=1 are not published.

## 6. BF16 vs FP16 marshalling

LTX's `Fp8CastLinear` upcasts FP8 weights to `input.dtype` (BF16). All Q/K/V are BF16. Our `_sage_fn(q.to(v.dtype), k.to(v.dtype), v, ...)` keeps BF16 end-to-end at the wrapper level, but Sage's FP16-PV variants may insert implicit BF16→FP16 conversions inside the kernel. XFormers accepts BF16 natively and runs it end-to-end without any such conversion.

Not fully confirmed without instrumenting the kernel, but it's a plausible extra cost specific to Sage in this pipeline.

## 7. Why we still kept Sage enabled previously

Sage2++ is still the **numerically safest fast-attention option** we have; the FP8-PV crash on the plain `_sm90` kernel is documented and real. The earlier decision to ship Sage was correct under the assumption "Sage beats XFormers on H100." The A/B on 2026-04-22 falsified that assumption for this shape — but only for this shape.

## 8. Options, ranked by effort vs expected win

| # | Action | Effort | Expected outcome |
|---|---|---|---|
| 1 | Flip `USE_SAGE_ATTENTION` default to `"0"` in `src/pipeline.py`. Keep the patch code so it can be re-enabled with an env var. | 1-line diff | Ships the 6 % faster path today. |
| 2 | Try `sageattn_qk_int8_pv_fp16_cuda` with `pv_accum_dtype="fp16"` — a precision-safe non-FP8-PV CUDA kernel that avoids the fp32+fp16 spill. | ~5 min, one-line kernel swap in `sage_patch.py` | Might narrow or close the gap. Worth one A/B run before shelving Sage. |
| 3 | Switch to `tensor_layout="NHD"` and drop the transposes — match what XFormers does. | ~10 min | Saves a few ms per call. Won't flip the outcome on its own. |
| 4 | Re-benchmark at larger shapes (higher res, more frames, B>1 if supported). | 0 code | If we ever run bigger shapes, Sage may win. Useful data before removing the patch entirely. |
| 5 | Keep Sage for quality-sensitive configs, XFormers for speed-sensitive ones. | Medium | Over-engineering for one service with one shape. |

Recommendation: **(1) + (2)**. Ship XFormers as default immediately; keep one experiment on the FP16-PV CUDA kernel. If (2) doesn't beat XFormers either, Sage can be removed entirely.

## 9. A/B measurement details

Same prompt (UGC beach vlog), seed=42, 1920×1088, 121 frames, 30 steps, warm pipeline:

| Backend | Job | Warm time |
|---|---|---|
| Sage2++ (`sageattn_qk_int8_pv_fp8_cuda`, `pv_accum="fp32+fp16"`) | `ltx_2f678934fcaf.mp4` | 141 s |
| XFormers (`memory_efficient_attention`, BF16) | `ltx_08e16810a63e.mp4` | 133 s |

Single-shot measurement. Typical run-to-run noise on this pod is ±2–3 s. The 8 s delta is outside noise but a 3–5 sample median per side would tighten the estimate.

## 10. References

- thu-ml SageAttention README: https://github.com/thu-ml/SageAttention
- SageAttention paper (v1): https://arxiv.org/html/2410.02367v5
- SageAttention 2: https://arxiv.org/html/2411.10958v5
- SageAttention 2++: https://arxiv.org/html/2505.21136v1
- Our install: `Dockerfile` L28–32, `src/sage_patch.py`
- Our toggle: `src/pipeline.py` L46 (`USE_SAGE_ATTENTION` env var, default `"1"`)
