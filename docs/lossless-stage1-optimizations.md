# Lossless Stage-1 Optimization Plan (L40S, Pure-GPU, HQQ4)

This document captures the deep-research output for reducing Stage-1
diffusion latency on an L40S (Ada sm_89, 48 GB) when running in
pure-GPU mode with `GEMMA_POST_LOAD_QUANT=hqq4`. Every mitigation
listed here is **lossless** (or near-lossless within ≤1% VBench drop on
video DiTs per source-of-truth research) — no schedule changes, no
checkpoint swaps, no guidance tricks.

The current pipeline runs at ~250 s end-to-end. Stage 1 (the 30-step
22B FP8 DiT with 4 guidance passes per step) accounts for ~150–200 s,
or roughly 75–85% of wall time. Everything below targets that phase.

---

## 1. Where Stage-1 time actually goes

Three independent sources of overhead, ranked by share. Each is
attackable with a different strategy.

| Source | Share of Stage-1 | Lossless mitigation | Already in repo? |
|---|---|---|---|
| **A. Diffusion schedule** — 30 steps × 4 guidance passes = 120 transformer forwards | ~80% | Step-skip caching (TeaCache / FBCache / MagCache) | TeaCache implemented in `src/teacache.py`, default OFF |
| **B. FP8↔BF16 circular cast** — `Fp8CastLinear.forward` does `weight.to(input.dtype)` on every Linear (~14,400 upcasts/gen) | ~12–18% | `torch._scaled_mm` native FP8 matmul on Ada | No — currently uses `fp8-cast` backend |
| **C. Kernel launch / Python overhead** — ~14k Linears + ~500 attention calls per gen | ~3–7% | `torch.compile(mode="reduce-overhead")` for CUDA graph capture | Partial — `torch_compile=True` is on but in default mode, not reduce-overhead |

`max_batch_size=4` (pipeline.py:866) is correctly always-on after
commit 7477b2e and is **not** a source of redundancy. It batches the
4 guidance passes into a single transformer call, eliminating 3
serial B=1 forwards per step (~75% reduction in launch overhead for
the guidance branch). Do not change it.

---

## 2. Lossless mitigations, ranked by leverage

### Tier 1 — High leverage, ship now

#### 2.1 Flip on TeaCache (zero code change)

Already implemented in `src/teacache.py`, default off in `.env.example`.

**Source of truth:** `ali-vilab/TeaCache` —
[`TeaCache4LTX-Video/README.md`](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4LTX-Video/README.md)
explicitly publishes:

| Threshold | Speedup | Quality note |
|---|---|---|
| `0.00` | 1.0× (baseline) | reference |
| `0.03` | **1.6×** | Their lossless point |
| `0.05` | 2.1× | "Without much visual degradation" |

The polynomial in `src/teacache.py:44` is the LTX-Video-fitted curve
(coefficients copied verbatim from upstream). It applies as-is to
LTX-2.3 because the residual structure is the same family of DiT.

**Action:** Set `ENABLE_TEACACHE=1`, `TEACACHE_THRESHOLD=0.03` in the
deploy environment. The implementation already logs
`computes / skips / skip_rate` per stage (see
`src/teacache.py:213`), so we get observability for free.

**Caveat from existing memory:** TeaCache × `BatchSplitAdapter` has a
correctness concern when `max_batch_size=1`. Our pure-GPU path
runs `max_batch_size=4` always (commit 7477b2e), so this is a
non-issue here.

**Expected end-to-end win:** ~1.6× Stage-1.

---

#### 2.2 Eliminate the FP8 circular cast — `_scaled_mm` swap

**Source of truth:** Lightricks ships two FP8 backends in `ltx-core`:

- `fp8-cast` (current) — stores weights in FP8, upcasts to BF16 on
  every forward. The matmul runs in BF16, so we never use Ada's
  native FP8 tensor cores.
- `fp8-scaled-mm` — calls `cublas_scaled_mm` for native FP8 matmul,
  but Lightricks only supports it via the `fp8-trtllm` extra
  (TensorRT-LLM build, informally Hopper-targeted). See
  [LTX-2 issue #181](https://github.com/Lightricks/LTX-2/issues/181)
  for the shape-mismatch surface in their TRT-LLM path.

**The PyTorch-native escape hatch:** `torch._scaled_mm` supports
`e4m3fn × e4m3fn → bf16` with per-tensor and per-row scaling on Ada
sm_89 since PyTorch 2.5. The L40S is sm_89. We do not need TRT-LLM.

**Action:** Patch `Fp8CastLinear.forward` (in
`LTX-2/packages/ltx-core/src/ltx_core/quantization/fp8_cast.py`) to
call `torch._scaled_mm` directly when the GPU is Ada+ and PyTorch is
≥ 2.5. The weights are already FP8 on disk, so the upcast disappears
entirely and Ada's FP8 tensor cores (~330 dense TFLOPS) handle the
matmul.

**Expected win:** 1.3–1.6× Stage-1 (compounds on top of TeaCache).
Lossless within FP8 precision (same weights, just no round-trip
through BF16).

**Risk:** Per-tensor scale calibration. Mitigation: cache the scales
at boot, validate via the existing magnitude hook
(`pipeline.py:409`).

**Effort:** ~1 day, fork-and-mount via the same Dockerfile sed/patch
pattern used for the Gemma `use_fast=True` patch
(`Dockerfile:30-32`).

---

### Tier 2 — Strong wins, more porting

#### 2.3 `torch.compile(mode="reduce-overhead")`

**Source of truth:** PyTorch
[CUDA Graphs blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/) +
[torch.compile + Diffusers guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/).

`reduce-overhead` mode wraps each compiled region in a captured CUDA
graph. Replay cost is ~10 µs vs ~5–10 µs *per kernel* in the eager
path. With ~14k Linears × 120 forwards we're paying several seconds of
pure CPU launch stall per generation.

**Action:** Patch the per-block compile call in
`LTX-2/packages/ltx-core/src/ltx_core/transformer/compiling.py` to
pass `mode="reduce-overhead", fullgraph=True`. DiT blocks are
uniform-shape for a fixed (H, W, frames), so capture works without
recompile.

**Expected win:** 5–15% Stage-1 (small but additive; compounds with
2.1 and 2.2).

**Risk:** CUDA graphs are sensitive to dynamic shapes. Mitigation:
fix (H, W, frames) per pod or use guard-keyed graph cache. First-call
compile time grows; cached for the rest of the pod's lifetime.

---

#### 2.4 Upgrade TeaCache → MagCache (NeurIPS 2025, future work)

**Source of truth:**
[`Zehong-Ma/MagCache`](https://github.com/Zehong-Ma/MagCache),
[arXiv 2506.09045](https://arxiv.org/html/2506.09045v1).

| Cache | Reported lossless speedup on video DiT | Calibration |
|---|---|---|
| TeaCache | 1.6× (LTX-Video, threshold 0.03) | Polynomial fit (already done) |
| FBCache | 1.5–2× (FLUX, video DiT thresholds 0.08–0.12) | None |
| **MagCache** | **2.68× (Wan2.1), 2.82× (HunyuanVideo)** | Single random sample |

MagCache uses the magnitude ratio of successive residual outputs
(monotone-decreasing across timesteps, sharp at the end of the
schedule) as the skip indicator. Single-sample calibration — far
lighter than TeaCache's polynomial fit.

**Action:** Port `magcache_calibration` + `magcache_forward` into a
new `src/magcache.py` mirroring the structure of `src/teacache.py`.
Their repo provides a PR template for new model integrations. Run
the single-sample calibration once at pod boot for LTX-2.3 and
hardcode the resulting magnitude ratios.

**Critical:** Do NOT stack MagCache + TeaCache. Both gate skips on
similar input similarity; stacking double-counts and degrades
quality. Pick one.

**Expected win:** Replaces TeaCache's 1.6× with ~2.5–2.8× at the
same lossless quality bar.

**Effort:** 1–2 days.

---

### Tier 3 — Smaller, free-ish wins

#### 2.5 Batch the two-prompt Gemma encode

LTX-2's `PromptEncoder` calls Gemma sequentially for positive and
negative prompts. Batching into a single `B=2` forward halves the
encode time.

**Expected win:** 5–15 s off the 15–30 s Gemma phase (~3–6% end-to-end).

**Effort:** Small upstream patch in LTX-2's `PromptEncoder.__call__`.

---

#### 2.6 Gate `_install_stage2_cleanup_hook` on `not pure_gpu_mode`

The hook in `pipeline.py:484` forces a `torch.cuda.synchronize()` +
`empty_cache()` + `_host_emptyCache()` at the Stage 1 → Stage 2
boundary. It exists to fix a pin_memory crash on 24 GB cards. In
pure-GPU mode there is no streaming and no pin_memory pressure, so
the synchronize is pure stall (a few hundred ms).

**Expected win:** ~0.1–0.5 s per generation. Tiny but free.

**Effort:** One-line conditional.

---

## 3. What is excluded from the lossless plan

| Approach | Why excluded |
|---|---|
| NVFP4 | Blackwell-only |
| FP4 attention (SageAttention 3) | Blackwell-only |
| Distilled checkpoint (4–8 step) | Changes schedule + LoRA strength → not lossless |
| CFG distillation / single-pass guidance | Lossy by definition |
| Layer-wise quant beyond FP8 | Requires QAT |
| Context parallelism | Single-pod L40S deploy |
| SageAttention2 INT8 attention | Reported quality artifacts on some video DiTs (multiple ComfyUI issues). Near-lossless but not strictly lossless — A/B required before adopting |

---

## 4. Recommended landing sequence

| # | Change | Lossless? | Wall-time win | Effort | Order rationale |
|---|---|---|---|---|---|
| 1 | `ENABLE_TEACACHE=1`, `TEACACHE_THRESHOLD=0.03` | Yes (per ali-vilab) | ~1.6× Stage-1 | env flip | Zero code, biggest fast win |
| 2 | `_scaled_mm` swap in `Fp8CastLinear` for Ada | Yes (FP8-equivalent) | ~1.4× Stage-1, multiplies on #1 | ~1 day fork patch | Eliminates the largest non-cache redundancy |
| 3 | `torch.compile(mode="reduce-overhead")` | Yes | ~1.1× Stage-1 | small kwarg change | Cheap compile-mode upgrade |
| 4 | Gate Stage-2 cleanup hook on `not pure_gpu_mode` | Yes | ~0.1–0.5 s | 1 line | Trivial cleanup |
| 5 | Port MagCache, replace TeaCache once validated | Yes (per paper) | upgrades #1 from 1.6× → ~2.5× | 1–2 days | Bigger lever, more porting work |
| 6 | Batch Gemma 2-prompt encode | Yes | 5–15 s | medium | Independent of DiT path |

**Stacked, lossless target:** 150–200 s Stage-1 → ~60–80 s.
End-to-end 250 s → **~120–140 s** with no schedule, checkpoint, or
guidance changes.

---

## 5. Instrumentation prerequisite

Before landing #2 onward, add per-phase `time.perf_counter()` blocks
in `LTXVideoGenerator.generate()` (`pipeline.py:828–888`) for:

- Gemma encode (positive + negative)
- Stage 1 diffusion
- Stage 2 cleanup hook
- Spatial upscaler
- Stage 2 diffusion
- VAE decode + H.264 encode

Without these we're optimizing against estimates instead of measured
data. The numbers in §1 above are derived from the bottleneck-analysis
memory and the LTX-2 architecture; pod-specific reality may differ
±20%.

---

## 6. Sources

- [ali-vilab/TeaCache — TeaCache4LTX-Video README](https://github.com/ali-vilab/TeaCache/blob/main/TeaCache4LTX-Video/README.md)
- [Lightricks LTX-2 ltx-core README](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/README.md)
- [Lightricks LTX-2 issue #181 — fp8-scaled-mm shapes](https://github.com/Lightricks/LTX-2/issues/181)
- [PyTorch `_scaled_mm` API reference (drisspg)](https://gist.github.com/drisspg/783616821043ab4594b9784f556c6714)
- [PyTorch native FP8 datatypes](https://medium.com/data-science/pytorch-native-fp8-fedc06f1c9f7)
- [ParaAttention First Block Cache](https://deepwiki.com/chengzeyi/ParaAttention/2.2-first-block-cache)
- [Comfy-WaveSpeed FBCache docs](https://deepwiki.com/chengzeyi/Comfy-WaveSpeed/2-first-block-cache-(fbcache))
- [MagCache (NeurIPS 2025) repo](https://github.com/Zehong-Ma/MagCache)
- [MagCache paper (arXiv 2506.09045)](https://arxiv.org/html/2506.09045v1)
- [thu-ml SageAttention](https://github.com/thu-ml/SageAttention)
- [PyTorch CUDA Graphs blog](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)
- [PyTorch torch.compile + Diffusers guide](https://pytorch.org/blog/torch-compile-and-diffusers-a-hands-on-guide-to-peak-performance/)
- [AdaCache (ICCV 2025)](https://github.com/AdaCache-DiT/AdaCache)
- [Cache method comparison: FBCache vs TeaCache vs AdaCache](https://sahirp.com/posts/caching/)
