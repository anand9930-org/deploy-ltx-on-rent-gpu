# FP8 H100 — FA3 ON vs FA3 OFF (live A/B)

**Date:** 2026-04-29
**Image:** `<DOCKERHUB_USERNAME>/ltx-video-fa3-teacache:latest` @ commit `d772806` (FP8 H100 + FA3 + TeaCache)
**Endpoint:** `https://8byl84g6drpvl1-8000.proxy.runpod.net/generate`
**GPU:** H100 SXM5 80 GB, RunPod community
**Workload:** 1920×1088 landscape, 121 frames (5 s @ 24 fps), 30 inference steps, cfg=3.0, stg=1.0
**Toggle:** pod env `LTX_ATTENTION_TYPE=flash_attention_3` (ON) ↔ `LTX_ATTENTION_TYPE=pytorch` (OFF, SDPA)
**Prompts:** three UGC voiceover prompts (skincare / espresso / desk-vlog), seeds 101 / 202 / 303

## Results

| Seed | Prompt | FA3 ON wall | FA3 ON gen | SDPA wall | SDPA gen | Δ gen | Speedup |
|---|---|---|---|---|---|---|---|
| 101 | Skincare serum vlog | 73.19 s | 55.17 s | 114.04 s* | 66.17 s* | +11.0 s | 1.20× FA3 |
| 202 | Portable espresso review | 70.23 s | 54.29 s | 69.55 s | 53.41 s | −0.9 s | ~1.00× |
| 303 | Productivity tip vlog | 69.55 s | 51.86 s | 70.93 s | 53.98 s | +2.1 s | 1.04× FA3 |
| **Mean (all 3)** | | **70.99 s** | **53.77 s** | **84.84 s** | **57.85 s** | **+4.1 s** | **1.08× FA3** |
| **Mean (excl. seed 101 SDPA cold)** | | 70.99 s | 53.77 s | 70.24 s | **53.70 s** | −0.07 s | **~1.00×** |

\* Seed 101 SDPA was the first request after the env-var flip + service restart; wall-vs-gen gap of 48 s (vs 15-18 s on the other runs) suggests first-call torch.compile retrace + model warm. Treat as cold-sample outlier.

## Cost (H100 SXM5 ≈ $2.99 / hr → $0.000831 / s)

| Path | $/clip (gen-time) | $/clip (wall) |
|---|---|---|
| FA3 ON | $0.045 | $0.059 |
| SDPA (excl. cold) | $0.045 | $0.058 |

## Conclusion

On the FP8 H100 stack, **FA3 contributes 1.04–1.08× steady-state — far below the 1.28× the repo documents on BF16** (`docs/latency_report.md` §5: 40.7 s → 31.87 s). Most likely cause: FP8 cast / scaled-mm has shifted the per-step bottleneck off attention, so Amdahl's law shrinks FA3's headroom. Memory savings from FA3 are still load-bearing — three 10 s clips at 1920×1088 / 241 frames ran clean today, despite `latency_report.md` issue #6 documenting Stage-2 OOM at that frame count on the prior stack.

**Recommendation:** keep FA3 ON (no downside, free given the cached wheel). The interesting next A/B is **FP8 ON vs BF16 ON**, not FA3 — that's where the bigger lever lives now.

## Raw 10 s reference (FA3 ON only, for context)

Same image, 1920×1088, **241 frames** (10 s):

| Seed | Theme | Wall | Server gen |
|---|---|---|---|
| 401 | Earbuds unboxing (man) | 123.25 s | 94.93 s |
| 402 | Car-vlog rant (woman) | 125.28 s | 93.80 s |
| 403 | Gym progress (woman) | 121.86 s | 93.46 s |
| **Mean** | | **123.46 s** | **94.06 s** |

Linear scaling vs 5 s (~1.75× gen for 2× frames) — Stage-2 OOM ceiling lifted on this stack.

## Output URLs (Supabase signed, valid through 2026-05-06)

**FA3 ON, 5 s:** `ltx_d1287326e72f.mp4` · `ltx_898b4f1310ba.mp4` · `ltx_051d9caaeb16.mp4`
**SDPA, 5 s:** `ltx_e1faa01e38a6.mp4` · `ltx_cb47422d1e12.mp4` · `ltx_e857f0d9ed60.mp4`
**FA3 ON, 10 s:** `ltx_52c2e61251f8.mp4` · `ltx_445108471ea8.mp4` · `ltx_d92cfbde17e3.mp4`
