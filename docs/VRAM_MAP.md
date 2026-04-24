# VRAM Map — LTX-2.3 22B on H100 80 GB

Measured from `src/memory_profile.py` instrumentation (commit `14a61a4`) on a
live RunPod H100 pod, image `feat/fa3-teacache` branch. Config:
`LTX_QUANT=fp8_cast`, `LTX_ATTENTION_TYPE=flash_attention_3`, TeaCache on,
streaming disabled. Two back-to-back 5 s requests (1024×1536×121, 30 steps).

Purpose: stop guessing at resident vs. transient splits and lock in the
real numbers before picking a 10 s lever.

---

## 1. Measured snapshots

GPU ceiling: **85.05 GB** total on this pod.

| # | Checkpoint | Alloc | Reserved | Peak alloc | Peak res |
|---|---|---:|---:|---:|---:|
| 0 | `00_init_entry` | 0.00 | 0.00 | 0.00 | 0.00 |
| 1 | `01_after_pipeline_init` | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | `20_generate_start` (cold) | 0.00 | 0.00 | 0.00 | 0.00 |
| 3 | `10_stage2_boundary_before_cleanup` (cold) | 55.93 | 56.03 | 56.22 | 56.34 |
| 4 | `11_stage2_boundary_after_cleanup` (cold) | 55.93 | 56.00 | 56.22 | 56.34 |
| 5 | `30_generate_end` (cold) | 57.02 | 57.10 | **79.57** | 80.47 |
| 6 | `20_generate_start` (warm) | 57.02 | 57.52 | 57.02 | 57.52 |
| 7 | `10_stage2_boundary_before_cleanup` (warm) | 57.06 | 57.18 | 59.34 | 59.83 |
| 8 | `30_generate_end` (warm) | 57.02 | 57.18 | **80.71** | 81.61 |

All numbers in GB. Cold = first request after pod boot; warm = second request
with weights still resident.

### Key deltas

| Metric | Value |
|---|---|
| Resident at Stage 2 entry (warm) | **57.06 GB** |
| Stage 1 transient peak delta | 59.34 − 57.06 = **2.28 GB** |
| Stage 2 transient peak delta | 80.71 − 57.06 = **23.65 GB** |
| Free headroom at Stage 2 peak | 85.05 − 80.71 = **4.34 GB** |

Stage 2's one-shot surge owns virtually all of the transient cost. Stage 1
barely shows up despite running 30 sampler steps — TeaCache + FA3 keep the
working set tiny.

---

## 2. Visual representation

### Timeline across a warm request (GB, against 85.05 GB ceiling)

```
              0 GB                                                    85 GB
              |---------|---------|---------|---------|---------|------>
 init entry   │░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│   0.0 GB
 pipeline init│░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░│   0.0 GB
 gen start    │████████████████████████████████████████░░░░░░░░░░░░░░░│  57.0 GB
 stg1 peak    │██████████████████████████████████████████░░░░░░░░░░░░░│  59.3 GB
 stg2 entry   │████████████████████████████████████████░░░░░░░░░░░░░░░│  57.1 GB
 stg2 PEAK    │████████████████████████████████████████████████████████│  80.7 GB  ← 4.3 GB free
 gen end      │████████████████████████████████████████░░░░░░░░░░░░░░░│  57.0 GB
              |---------|---------|---------|---------|---------|------>
                       20        40        60        80   ceiling=85
```

### Stage 2 peak decomposition (80.71 GB total)

```
┌──────────────────────────────────────────────────────────────────┐
│  Resident transformer weights (BF16 shadow + LoRA fuse)          │
│  ████████████████████████████████████████████████████  55.9 GB   │  69%
├──────────────────────────────────────────────────────────────────┤
│  Allocator / fragmentation / misc                                │
│  █                                                      1.1 GB   │   1%
├──────────────────────────────────────────────────────────────────┤
│  Stage 2 transient (attention + RoPE split buffers + act)        │
│  █████████████████████                                 23.7 GB   │  29%
├──────────────────────────────────────────────────────────────────┤
│  Free                                                            │
│  ████                                                   4.3 GB   │   5%
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Top-25 live allocations (identical at every checkpoint)

All 25 originate from `sft_loader.py:36` — the streaming safetensors loader
that materialises transformer weight shards. Seeing every slot pin to one
callsite is strong evidence that the resident 55.9 GB is the loaded model,
not leaked intermediates.

| Rank | Size | Callsite |
|---:|---:|---|
| #1 | 2013.8 MB | `sft_loader.py:36` |
| #2 | 1541.4 MB | `sft_loader.py:36` |
| #3 | 770.7 MB | `sft_loader.py:36` |
| #4 | 302.0 MB | `sft_loader.py:36` |
| #5 | 226.5 MB | `sft_loader.py:36` |
| #6–#25 (×20) | 134.2 MB each | `sft_loader.py:36` |

Top-25 sum: **7.54 GB**. The remaining ~48 GB is in sub-134 MB blocks below
the instrumentation cutoff (per-layer attention/FFN weights, norm params,
embedding tables, allocator pages).

---

## 4. Correcting the earlier resident estimate

Previous planning documents claimed resident ≈ 23 GB, assuming
`LTX_QUANT=fp8_cast` meant weights live on-device as FP8 E4M3. **That is
wrong.** `fp8_cast` loads BF16 weights from disk and casts to FP8 *inside the
matmul*, so the BF16 tensor stays pinned in HBM. The resident ≈ 55.9 GB
figure matches the size of the BF16 checkpoint (~46 GB file) + LoRA fusion
materialisation + allocator overhead.

Implication: every planning document that assumed a 22-GB resident headroom
needs to be reread against the 55.9-GB number.

---

## 5. Extrapolation to 10 s (1024×1536×241)

Stage 2 transient scales roughly linearly with frame count because the
attention and RoPE buffers carry S = frames × spatial_tokens. Frames
double, spatial stays fixed.

| Workload | Stage 2 transient | Resident | Projected total | Headroom |
|---|---:|---:|---:|---:|
| 5 s (measured) | 23.7 GB | 57.1 GB | 80.7 GB | +4.3 GB |
| 10 s (linear projection) | ~47.3 GB | 57.1 GB | ~104 GB | **−19 GB** |

This matches the observed 10 s OOM — the allocator asks for 372 MB with
only 243 MB free after already sitting at ~79 GB.

---

## 6. Candidate 10 s levers — ranked by GB saved

| # | Lever | Est. GB saved | Complexity | Latency cost | Risk |
|---|---|---:|---|---|---|
| 1 | **Stage 2 temporal windowing** (tile=80 frames, overlap=24) | ~24 GB | medium | +10–15% | low — community-proven pattern |
| 2 | **XFormers attention on Stage 2 only** | ~2–4 GB | medium | +10–15% on Stage 2 | low |
| 3 | **`fp8_scaled_mm` native FP8 matmul** (Py3.12 base image) | ~1–3 GB | high (base image swap, tensorrt_llm cp312 wheel) | neutral | medium — Lightricks #181 open |
| 4 | **FFN chunking on Stage 2 forward** | ~1–2 GB | low | +2–5% | low |
| 5 | Smaller RoPE dtype for split buffer | ~0.5 GB | low | neutral | low |

Lever #1 is the only one that solves 10 s by itself. Levers #2–#5 are
stackable for 15 s+ workloads but can't close a 19 GB deficit on their own.

---

## 7. Recommended sequence

1. **Land Stage 2 temporal windowing** — tile=80 frames, overlap=24, gated
   on `num_frames > 200`. Port from diffusers `LTXI2VLongMultiPromptPipeline`
   (PR #12614) or ComfyUI `looping_sampler`. Wrap Stage 2 only; Stage 1
   stays whole because its transient is already negligible.
2. **Rerun the instrumentation** on a 10 s request and confirm Stage 2 peak
   < 75 GB. Capture a new row for this table.
3. **Keep levers #2–#5 on the shelf** for 15 s / 20 s stretch goals.
4. **Do not** pursue `fp8_scaled_mm` until measurement shows the residual
   transient after windowing is the binding constraint — the base-image
   swap is too expensive to chase 1–3 GB speculatively.

---

## 8. Verification

After any lever lands, rerun with the same instrumentation:

```bash
# on the pod
curl -X POST https://<pod>.proxy.runpod.net/generate/submit \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"<same seed prompt>","seed":42,"duration":10,...}'
```

Check pod logs for `[VRAM]` lines. Required evidence:

- `30_generate_end` peak_alloc < 75 GB for 10 s (≥ 10 GB headroom).
- `10_stage2_boundary_before_cleanup` resident unchanged (~57 GB).
- `sft_loader.py:36` still dominates top-25 — confirms no new leak.
- Pickles in `/tmp/vram_snapshots/` viewable via `torch.cuda._memory_viz`
  if a regression appears.

---

## 9. Source files

- `src/memory_profile.py` — the instrumentation helpers.
- `src/pipeline.py` — snapshot call sites at init, pipeline construct,
  Stage 2 boundary (before/after cleanup), generate start, generate end.
- Raw snapshot pickles: `/tmp/vram_snapshots/` on the pod.
- Pod log capture: source of the numbers in §1.
