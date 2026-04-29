# FP8 vs BF16 — 5 s clip comparison (live, paired)

**Date:** 2026-04-29
**GPU:** H100 SXM5 80 GB, RunPod community
**Workload:** 1920×1088 landscape, **121 frames (5 s @ 24 fps)**, 30 inference steps, cfg=3.0, stg=1.0
**Common config (both stacks):** FA3 ON · TeaCache ON · torch.compile ON · same prompts and seeds
**Endpoint:** `https://8byl84g6drpvl1-8000.proxy.runpod.net/generate` (re-deployed per branch)

## Stack identification

| Label | Branch | Compute dtype | Weight storage | GEMM kernel |
|---|---|---|---|---|
| **FP8** | `feature/fp8-h100` | FP8 (e4m3) | FP8 (e4m3) | TRT-LLM `cublas_scaled_mm` (W8A8) |
| **BF16** | `feat/fa3-teacache` | **BF16** | FP8 (cast on load via `QuantizationPolicy.fp8_cast()` — `src/pipeline.py:181`) | torch BF16 matmul (W8A16) |

Both pods exported `LTX_ATTENTION_TYPE=flash_attention_3` and `ENABLE_TEACACHE=1`. The "BF16" label refers to **compute dtype**, not weight storage — weights are still FP8 on disk and cast to BF16 per forward pass.

## Results — 5 s, paired by seed

| Seed | Prompt | FP8 wall | FP8 gen | BF16 wall | BF16 gen | Δ gen | Winner |
|---|---|---|---|---|---|---|---|
| 101 | Skincare serum vlog | 73.19 s | 55.17 s | — | 48.71 s\* | −6.46 s | BF16 1.13× |
| 202 | Portable espresso review | 70.23 s | 54.29 s | 72.38 s | 49.09 s | −5.20 s | BF16 1.11× |
| 303 | Productivity tip vlog | 69.55 s | 51.86 s | 68.25 s | 47.58 s | −4.28 s | BF16 1.09× |
| **Mean** | | **70.99 s** | **53.77 s** | **70.31 s**† | **48.46 s** | **−5.31 s** | **BF16 1.11×** |

\* BF16 seed 101 was the first request after pod boot but did not show a cold-start outlier — gen tracks the warm steady-state of seeds 202/303 (Δ < 1.5 s). `wall` not captured for this seed (recovered manually via `/generate/get` after the original poller missed the `completed` status alias).
† BF16 wall mean averages seeds 202/303 only; seed 101 wall not recorded.

## Cost — H100 SXM5 ≈ $2.99 / hr ($0.000831 / s)

| Stack | $/clip (gen-time) | $/clip (wall) |
|---|---|---|
| FP8 (W8A8) | $0.0447 | $0.0590 |
| **BF16 (W8A16)** | **$0.0403** | **$0.0584** |

Per-clip savings on BF16: **~$0.0044 gen / ~$0.0006 wall**. Over a 1 M-clip run that is **$4,400 gen / $600 wall**.

## Why is BF16-compute faster than FP8 here?

Three plausible drivers, ordered by likely contribution:

1. **`cublas_scaled_mm` per-call overhead.** TRT-LLM's FP8 GEMM path carries quantization-scaling fixed cost per matmul (input scale → fp8 cast → fp8 GEMM → output rescale). At LTX-2.3's working shapes (121 frames @ 1080p), this tax is non-trivial relative to the GEMM itself, especially for the many smaller projection matmuls in attention Q/K/V/out and FFN gates. BF16 matmul on Hopper has zero such overhead.

2. **TeaCache hit-rate likely differs across stacks.** TeaCache decides whether to skip a transformer step based on a relative-L1 distance threshold on hidden states between consecutive timesteps. FP8 quantization injects per-step noise into those hidden states, which can inflate the measured distance and cause TeaCache to skip *fewer* steps than under BF16. To confirm, grep pod logs for TeaCache skip-counts on each stack — if BF16 skips noticeably more steps, that fully explains the gap.

3. **Hopper BF16 tensor cores are extremely well-tuned at LTX's GEMM shapes.** A single cast-on-load (FP8→BF16) followed by many BF16 matmuls amortizes near peak Tensor Core throughput. FP8 W8A8 trades a memory-bandwidth win on weight loads for a recurring rescale tax on every matmul — the trade only pays off when bandwidth is the actual bottleneck. At 1080p / 121 frames, compute dominates over bandwidth, so the trade goes the wrong way.

## When does FP8 still win?

Memory headroom, not speed. Today's session also tested 1920×1088 / **241 frames** (10 s clips):

| Stack | Seed 401 | Seed 402 | Seed 403 |
|---|---|---|---|
| FP8 (W8A8) | ✓ 94.93 s | ✓ 93.80 s | ✓ 93.46 s |
| BF16 (W8A16) | ✗ Stage-2 OOM | ✗ Stage-2 OOM | ✗ Stage-2 OOM |

The BF16-compute stack still hits the OOM ceiling that `docs/latency_report.md` issue #6 documented at 1080p / 241 frames. FA3's attention-memory savings alone are not enough to clear it — the working set on activations is still too large with BF16 compute. FP8-W8A8 lifts that ceiling cleanly.

## Conclusion

For the 5 s / 121-frame / 1080p workload (the most common clip length for UGC), **BF16-compute (W8A16, fp8_cast) is the speed-optimal config — ~11% faster gen time than FP8-W8A8** at identical FA3-ON, TeaCache-ON settings. Reach for FP8-W8A8 specifically when you need the memory ceiling: 10 s clips at 1080p, longer clips, higher resolutions, or multi-batch.

**Default recommendation:**

| Clip length | Resolution | Recommended stack |
|---|---|---|
| ≤ 5 s | ≤ 1080p | **BF16 (W8A16, fp8_cast)** |
| ≥ 10 s | ≥ 1080p | **FP8 (W8A8, scaled_mm)** |

## Output URLs (Supabase signed, valid through 2026-05-06)

**FP8 5 s:** `ltx_d1287326e72f.mp4` · `ltx_898b4f1310ba.mp4` · `ltx_051d9caaeb16.mp4`
**BF16 5 s:** `ltx_03b0c0985469.mp4` (101) · `ltx_11358728d5a2.mp4` (202) · `ltx_2d97d74bf319.mp4` (303)

## Cross-reference

- `docs/fp8_fa3_ab_results.md` — FA3 ON vs OFF on the FP8 stack (1.04–1.08×).
- `docs/latency_report.md` §5 — historical BF16-stack reference (31.87 s FA3-on / 40.7 s FA3-off → 1.28×); today's BF16 measurement at 48.46 s is ~16 s slower than that historical number, likely due to image and dependency drift since the report. Investigate separately if the gap matters.
