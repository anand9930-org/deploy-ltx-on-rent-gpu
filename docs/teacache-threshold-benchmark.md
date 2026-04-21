# TeaCache Threshold A/B Benchmark — L40S Pure-GPU

End-to-end benchmark comparing `TEACACHE_THRESHOLD=0.03` vs `0.05` on the
production deployment (L40S 48 GB, secure cloud, HQQ4 Gemma, pure-GPU
mode). Same prompts, same seeds, sequential execution.

Both runs used commit `59b1157` (post cross-job VRAM-leak fix) and
`max_concurrency=1`.

## Test setup

| Param | Value |
|---|---|
| Hardware | NVIDIA L40S, 48 GB (RunPod secure cloud) |
| Resolution | 1088 × 1920 (vertical UGC, 64-snapped from 1080×1920) |
| Duration | 5.04 s (121 frames @ 24 fps) |
| Steps | 30 (Stage 1) + 3 (Stage 2 distilled) |
| Guidance | cfg=3.0, stg=1.0, rescale=0.7 |
| Gemma quant | HQQ4 (post-load, ~9 GB resident) |
| DiT quant | FP8 (`fp8_cast` backend) |
| Attention | PytorchAttention (SDPA — xformers compile mismatch on this image) |
| `torch.compile` | ON, regional per transformer block |
| TeaCache stages | `stage_1` only |
| Sequential, not concurrent | Yes (3 jobs A → B → C, fresh pod each run) |

## Prompts (same across both runs)

| Tag | Seed | Persona | Spoken line |
|---|---|---|---|
| A | 101 | Bedroom matcha | *"Okay you guys, I just tried this matcha latte recipe and I am literally obsessed, you have to try it."* |
| B | 202 | Bathroom skincare | *"Day fourteen of using this serum and oh my god, my skin has never looked this good."* |
| C | 303 | Lisbon travel vlog | *"You guys, I cannot believe I am actually here in Lisbon right now, this place is unreal."* |

All three are vertical 1088×1920 UGC selfie-style framings with explicit
spoken lines included in the prompt for lip-sync alignment.

## Per-job timings

`gen_time_s` is the server-reported pipeline time; `wall` is end-to-end
including Supabase upload and 30 s polling resolution.

| Job | Seed | **Threshold 0.05** | **Threshold 0.03** | Δ gen_time | Slowdown |
|---|---|---|---|---|---|
| A — matcha (cold) | 101 | 193.24 s / 246 s wall | 217.63 s / 245 s wall | +24.4 s | +12.6% |
| B — skincare (warm) | 202 | 155.18 s / 184 s wall | 186.75 s / 215 s wall | +31.6 s | +20.4% |
| C — Lisbon (warm) | 303 | 155.68 s / 186 s wall | 186.23 s / 213 s wall | +30.6 s | +19.6% |
| **Steady-state (warm B+C avg)** | | **155.4 s / 185 s wall** | **186.5 s / 214 s wall** | **+31.1 s** | **+20.0%** |

The cold-job penalty (~40 s above warm steady-state) comes from
`torch.compile` JIT for both Stage 1 and Stage 2 transformer blocks.
This is paid once per pod boot; amortized to <1 s/job over 100+ jobs.

## Cross-validation against published numbers

`ali-vilab/TeaCache` publishes for LTX-Video baseline:
- `0.05` → **2.1×** speedup vs no cache
- `0.03` → **1.6×** speedup vs no cache (their lossless point)

Predicted slowdown ratio 0.05 → 0.03: **2.1 / 1.6 = 1.31×**, i.e.
~31% slower.

We measured **+20%** on warm jobs. The gap (20% vs 31%) is explained
by the fixed-cost overhead that doesn't scale with the cache (Stage 2
distilled denoise, VAE decode, MP4 encode, end-of-job cleanup,
Supabase upload). TeaCache only affects Stage 1 transformer forwards,
which is a smaller share of total wall time once those fixed costs
are included.

## Cost impact (L40S secure cloud, $1.19/hr)

| Threshold | Per-warm-video cost | Throughput | 3-job test total |
|---|---|---|---|
| 0.05 | $0.0612 (~6.1¢) | ~19.5 videos/hr | $0.2046 (~20.5¢) |
| 0.03 | $0.0707 (~7.1¢) | ~16.8 videos/hr | $0.2196 (~22.0¢) |

Cost delta: **+$0.01 per video** (~16% more in $/video) for the
higher-fidelity 0.03 setting. Throughput drops by ~14%.

## Quality interpretation

- **0.05** is ali-vilab's "no much visual degradation" point — small
  artifacts may appear but are typically not noticeable in
  customer-facing UGC.
- **0.03** is their lossless point — visually indistinguishable from
  no-cache baseline in their VBench evaluation.

The actual perceptual difference between the 6 videos (3 prompts × 2
thresholds, same seeds → same composition) is the next thing to verify.
If 0.03 looks tangibly cleaner, the +1¢/video premium is worth it for
production. If they're indistinguishable, 0.05 is the production
setting.

## Recommendation matrix

| Use case | Recommended threshold |
|---|---|
| Customer-facing UGC where occasional artifacts are unacceptable | **0.03** |
| Internal previews, A/B candidates, drafts | **0.05** |
| Speed-critical (live demos, batch generation at scale) | **0.05** or higher (0.08) |

## Test methodology

Single bash script, sequential execution, full poll-to-terminal
between submissions:

1. POST `/generate/submit` with prompt + params
2. Poll `/generate/status?task_id=...` every 30 s until terminal
3. GET `/generate/get?task_id=...` for the result + Supabase URL
4. Repeat for next job

The script also aborts on first failure to avoid the cross-job
VRAM-leak cascade that was present before commit `59b1157`. With the
fix in place, all three sequential jobs completed cleanly on both
runs — the prior "Job C OOMs after A and B succeed" pattern is gone.

## Related

- `docs/lossless-stage1-optimizations.md` — full landing sequence for
  remaining Stage-1 speedups (FP8 `_scaled_mm` swap, MagCache port,
  CUDA-graph compile mode)
- Commit `59b1157` — cross-job VRAM-leak fix that made stable
  sequential testing possible
- `src/teacache.py` — TeaCache integration; threshold and stage list
  configurable via env (`TEACACHE_THRESHOLD`, `TEACACHE_STAGES`)
