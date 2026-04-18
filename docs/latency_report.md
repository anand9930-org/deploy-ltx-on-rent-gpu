# LTX-2.3 Latency & Cost Report

**Date of measurements:** 2026-04-17 through 2026-04-18
**Pipeline under test:** `TI2VidTwoStagesPipeline` from LTX-2 main, wrapped by `src/pipeline.py` on branch `feat/teacache`
**Workload:** 1920×1088, 121 frames, 30 inference steps, seed 300, audio enabled, Supabase upload enabled
**Prompt:** "a red panda eating bamboo in a misty forest, cinematic" (+ 7 varied prompts for warm-pod validation)

---

## 1. Executive summary

- **Production target: H100 80 GB** (SXM, NVL, or PCIe). Warm steady-state is **40.7 s ± 1.1 s per clip** (8 consecutive runs across 8 different prompts). Cost at RunPod community pricing ($2.50/hr): **~$0.028 / clip**.
- **4090 24 GB is a viable secondary path** at **~520 s / $0.049 per clip** with TeaCache only. torch.compile does **not** stack on the 4090 — it conflicts with LTX-2's layer-streaming path and crashes at the Stage 1 → Stage 2 boundary.
- **Every SageAttention configuration produced either a failed wedge or corrupted output** and has been abandoned.
- **A100, RTX 6000 Ada, L40S, RTX 5090 are not production-viable** for this pipeline at 1920×1088 in their current builds — specific blockers listed in §3.

## 2. Production-ready configuration

The stack that ships:

| component | status | where | commit |
|---|---|---|---|
| TeaCache diffusion-step caching | opt-in via `ENABLE_TEACACHE=1` | `src/teacache.py`, `src/pipeline.py` | `6c0e746` |
| Stage 2 pin_memory cleanup hook | always-on correctness fix | `src/pipeline.py` | `79cb570` |
| torch.compile (regional per block) | opt-in via `ENABLE_TORCH_COMPILE=1`; auto-off on <40 GB cards | `src/pipeline.py` | `4ac6f09` + streaming-guard follow-up |
| Gemma `use_fast=True` image processor | Dockerfile sed on editable install | `Dockerfile` | `33025bf` |

Recommended env for production H100 deployment:

```
ENABLE_TEACACHE=1
TEACACHE_THRESHOLD=0.03
TEACACHE_STAGES=stage_1
ENABLE_TORCH_COMPILE=1
```

Recommended env for production 4090 deployment:

```
ENABLE_TEACACHE=1
TEACACHE_THRESHOLD=0.03
TEACACHE_STAGES=stage_1
ENABLE_TORCH_COMPILE=0   # streaming-guard auto-disables it anyway
```

## 3. Hardware compatibility matrix

| GPU | VRAM | Arch | full stack runs? | $/hr† | verdict |
|---|---|---|---|---|---|
| **H100 SXM** | 80 GB | Hopper sm_90 | **yes, 40.7 s warm** | ~$2.80 | **primary production target** |
| **H100 NVL / PCIe** | 80 – 94 GB | Hopper sm_90 | yes, same as SXM | ~$2.40 – $2.80 | interchangeable with SXM |
| **RTX 4090** | 24 GB | Ada sm_89 | partially — TeaCache only (520 s) | ~$0.34 | secondary / cost-optimised |
| RTX 6000 Ada | 48 GB | Ada sm_89 | **no** — OOM during BF16 weight load (46 GB transient peak + residue > 48 GB) | ~$0.74 | not viable for streaming-off path |
| L40 / L40S | 48 GB | Ada sm_89 | **no** — same failure mode as RTX 6000 Ada | ~$0.69 – $0.99 | not viable |
| A100 80 GB | 80 GB | Ampere sm_80 | **no** — LTX-2's `fp8_cast` Triton kernel uses `fp8e4nv`, which sm_80 can't emit. Stage 2 LoRA fusion fails unconditionally with `ValueError("type fp8e4nv not supported in this architecture")` | ~$1.79 | not viable |
| H200 | 141 GB | Hopper sm_90 | presumed yes (untested; same arch as H100, more VRAM we don't need) | ~$3.59 – $4.31 | expensive, no workload-specific benefit |
| RTX 5090 | 32 GB | Blackwell sm_120 | cold 650 s, warm unvalidated — streaming-active path, unclear cold-start breakdown | TBD | not recommended; Blackwell support unproven on this stack |

† RunPod community-cloud pricing, approximate, 2026-04-18.

**Not tested:** MI300X / other non-NVIDIA, consumer Blackwell (5080 / 5070 Ti), Jetson.

## 4. Latency measurements — H100 warm-pod validation

Same pod, same compile cache, same TeaCache state reset per request, 8 different prompts:

| # | prompt | gen_time (s) |
|---|---|---|
| 1 | red panda eating bamboo (baseline) | 38.86 |
| 2 | red panda eating bamboo (re-run, different seed-drift) | 41.50 |
| 3 | astronaut on a red Martian plain | 40.99 |
| 4 | busy Tokyo street at night, rain | 41.42 |
| 5 | young woman vlog intro | 41.32 |
| 6 | Parisian cafe at golden hour voiceover | 39.27 |
| 7 | woman on beach pier at sunset voiceover | 41.43 |
| 8 | marine biologist on research boat | 39.72 |

**Mean 40.7 s, stdev 1.1 s, p95 ≤ 41.5 s, max 41.5 s.**

The H.264 bitrate of outputs varied from 2.6 to 8.9 Mbps depending on scene complexity (Tokyo rain/neon was highest, cafe lowest). This reflects content complexity the encoder can't predict-compress — not noise. File sizes between 1.8 MB and 5.7 MB for 5-second clips.

## 5. Full latency history

### H100 80 GB

| configuration | state | gen_time (s) | $/clip @ $2.50/hr | notes |
|---|---|---|---|---|
| SAGE=0 baseline | cold | 154.68 | $0.108 | pre-optimisation reference |
| SAGE=1 (SageAttention v2, sm_90 INT8/FP8 kernel) | cold | 143.04 | — (output unusable) | **output is brown static noise** — sm_90 kernel tile-alignment bug on LTX-2's 5500/22000/2200 seq_len. Abandoned. |
| TeaCache only | cold | 161.00 | $0.112 | model loaded fresh |
| TeaCache only | warm | 54.00 | $0.038 | steady-state for TeaCache alone |
| TeaCache + compile + use_fast | cold (warm volume) | 75.87 | $0.053 | HF cache already populated; compile warmup ~30 – 60 s absorbed |
| TeaCache + compile + use_fast | cold (fresh volume) | 166.00 | $0.115 | first deploy, includes 80 GB model download |
| **TeaCache + compile + use_fast** | **warm** | **40.7 ± 1.1** | **$0.028** | **production target, 8-sample mean** |

### RTX 4090 24 GB

| configuration | state | gen_time (s) | $/clip @ $0.34/hr | notes |
|---|---|---|---|---|
| SAGE=0 baseline | cold | 585.25 | $0.055 | reference baseline |
| SAGE=1 (SageAttention sm_89 kernel) | — | — (wedged) | — | Triton JIT wedge + eventual pin_memory crash at Stage 2. Abandoned. |
| **TeaCache only** | **cold** | **520.28** | **$0.049** | **production path for 4090**, ~26 % Stage 1 speedup diluted by fixed overhead to ~11 % end-to-end |
| TeaCache + compile | cold | **failed at t+380 s** | — | 4090 streams weights (VRAM < 40 GB); torch.compile and LayerStreamingWrapper conflict at Stage 2 boundary. Streaming-guard now auto-disables compile on <40 GB cards. |
| TeaCache + compile | warm (after run 1) | **failed at t+20 s** | — | CUDA context poisoned from prior failure |

**4090 quality note on TeaCache alone:** output differs from SAGE=0 baseline at the same seed (30 – 35 % relative pixel L2) but is visually equivalent — clean panda, prompt-adherent, no artefacts. TeaCache's approximate cache slightly perturbs the sampling trajectory. Human-perceptual quality is intact; bit-exact seed reproducibility is lost.

## 6. Known issues & their fixes

### Fixed in this branch

1. **SageAttention pip install silently mutated torch / triton** (`--no-deps` missing). Observed as torch.compile-like errors that only appeared after adding the SageAttention wheel. Fixed by adding `--no-deps` to the wheel install in the SageAttention branch Dockerfile. Not an issue on `feat/teacache` because we don't install SageAttention.

2. **Stage 2 pin_memory() crash** on 24 GB streaming deploys. Intermittent CUDA error: invalid argument at `ltx_core/layer_streaming.py:63` (`tensor.data.pin_memory()`) when Stage 2's LayerStreamingWrapper tried to pin transformer weights after Stage 1 teardown. Fixed with `_install_stage2_cleanup_hook()` in `src/pipeline.py` — forces `gc.collect()` + `torch.cuda.synchronize()` + `torch.cuda.empty_cache()` + `torch._C._host_emptyCache()` before Stage 2 opens.

### Unfixed, mitigated

3. **torch.compile × layer-streaming crash on <40 GB cards.** Reproducible on 4090 with full stack. Fails at t+380 s (Stage 1 → Stage 2 boundary), then CUDA context poisoned so subsequent requests die at t+20 s. Mitigation: streaming-guard in `src/pipeline.py` auto-disables torch.compile when VRAM < 40 GB. 4090 therefore runs TeaCache-only (520 s, known-good). Full fix would require either patching LTX-2's `LayerStreamingWrapper` to not mutate `.data` in place, or stable-pointer-pinning the compiled kernels — neither is easy.

4. **LTX-2.3 mesh/web artefact around close-up portraits.** Visible on any close-up subject at 1920×1088 (vlog, cafe, pier, reef biologist). Not caused by our optimisation stack — reproduced with the unmodified `ti2vid_two_stages` pipeline. Known quirk of LTX-2.3 at this framing. Upstream issue.

### Rejected paths

5. **SageAttention on Hopper** — sm_90 kernel (`sageattn_qk_int8_pv_fp8_cuda_sm90`) produces corrupted output for LTX-2's non-128-aligned seq_lens. Confirmed with 4.7× per-frame byte ratio vs baseline and visual inspection showing pure brown static. Upstream issue; likely requires either a fix to the kernel or a completely different attention backend. **Do not enable on H100.**

6. **SageAttention on 4090** — Triton JIT + VRAM pressure wedges the BentoML runner; pre-warm fix was insufficient. Abandoned.

7. **FlashAttention** — research completed (see companion branch `research/flash-attention-ltx23` and `docs/FlashAttention_*.md`). Determined to be lower ROI than TeaCache for this pipeline. Not implemented.

## 7. Cost per clip summary

| config | latency | $/clip | clips per $ | clips per hour |
|---|---|---|---|---|
| 4090 SAGE=0 (pre-optimisation) | 585 s | $0.055 | 18 | 6 |
| 4090 + TeaCache | 520 s | $0.049 | 20 | 7 |
| H100 SAGE=0 warm (estimated) | ~90 s | $0.063 | 16 | 40 |
| **H100 + full stack warm** | **41 s** | **$0.028** | **37** | **~88** |

**H100 is the best option on both axes — cost and latency — once warm.** One day ago the 4090 was the $/clip winner; torch.compile flipped the economics completely.

## 8. Cold-start considerations

H100 cold-start breaks into two cases:

1. **Fresh pod, fresh network volume** — ~75 s of model download (80 GB from HuggingFace) + 90 s of model-to-GPU load + 10 s of torch.compile warmup on first generation = roughly **160 – 180 s** for the very first request.
2. **Fresh pod, warm network volume** (`/workspace/models` retained across restarts) — ~15 s to mount + 60 s model load + compile warmup = **~75 s** for first request.
3. **Any subsequent request on the same pod** — warm steady-state **~41 s**.

For production autoscaling: keep at least one pod warm per service to ensure no user pays the cold-start tax. If the autoscaler scales to zero, the first user after scale-up pays ~75 s additional latency on top of the normal 41 s.

## 9. Reproducibility

### Branch state when this report was written

```
branch: feat/teacache
base:   fix/inference-mode-api (commit 5855a7c)
commits ahead:
  33025bf  Force use_fast=True on Gemma AutoImageProcessor
  4ac6f09  Enable torch.compile on DiffusionStage (regional, per block)
  79cb570  Fix Stage 2 pin_memory crash: flush allocators at stage boundary
  6c0e746  Add TeaCache: diffusion-step caching for 1.6-2× lossless speedup
```

### Standard reproduction request

```http
POST /generate/submit
Content-Type: application/json

{
  "prompt":"a red panda eating bamboo in a misty forest, cinematic",
  "width": 1920,
  "height": 1088,
  "num_frames": 121,
  "num_inference_steps": 30,
  "seed": 300,
  "upload_to_supabase": true
}
```

### Reference output URLs (from test runs)

- 4090 + TeaCache: `https://qbcnoeoqzmjaxtusytjt.supabase.co/storage/v1/object/public/ltx-videos/ltx_7f59c99a70a7.mp4`
- H100 SAGE=0 (clean baseline): `…/ltx-videos/ltx_954d135fb6b7.mp4`
- H100 SAGE=1 (**noise reference — DO NOT SHIP**): `…/ltx-videos/ltx_988d63db487f.mp4`
- H100 + full stack warm: `…/ltx-videos/ltx_653ee52e575b.mp4`
- H100 warm, cafe prompt: `…/ltx-videos/ltx_8dd43904e3c3.mp4`

## 10. Observations worth flagging for product

1. **Audio is generated by default on every clip.** Every output in this test series included a 128 kbps AAC audio track, structurally distinct from silence. Voice-directed prompts (e.g. "says: 'hello world'") produce speech-shaped audio. Most competitors (Sora, Kling, Runway) don't do synced audio in the same pass.
2. **Prompt adherence is strong on cinematic framing prompts** (cafe, beach, documentary). Cinematic close-ups pull the mesh/web artefact (item 4 above); wide / medium shots don't.
3. **Warm-pod latency is prompt-independent** — shape, not content, determines timing. All 8 warm runs landed within ±1.5 s regardless of prompt complexity.
4. **Seed reproducibility is lost with TeaCache.** Same seed on same config still produces the same output (verified — run 1 and run 2 on H100 full stack matched byte-for-byte), but same seed with TeaCache on vs off produces different (equally-valid) outputs. If your tests pin specific seed outputs, plan for one-time re-baselining.

## 11. Next experiments, ranked

1. **`TI2VidTwoStagesHQPipeline` with Res2s sampler (15 Stage 1 steps instead of 30).** Projected to roughly halve remaining Stage 1 time on H100 — warm could land in the 25 – 30 s range. Worth one validation cycle. Requires pipeline class swap (~15 lines) and quality A/B.
2. **`torch.compile` streaming-compatible variant** (upstream fix or workaround for LayerStreamingWrapper). Would extend full-stack benefit to the 4090 path. Non-trivial; probably worth waiting for upstream.
3. **TeaCache threshold tuning** — try 0.05 (aggressive) on a handful of prompts to see if perceptible quality drop is acceptable in exchange for another 10 – 20 % speedup.
4. **Lower-res warm preview path** — add a `/generate/preview` endpoint at 768×432 / 25 frames for thumbnails. Projected <10 s per preview on warm H100.

## 12. Open questions for the platform team

- Is `/workspace/models` persistent across pod restarts on the deployment platform? If yes, cold-starts are 75 s. If no, cold-starts are 160+ s.
- Is autoscaling configured with `min_replicas ≥ 1`? If scale-to-zero, first user after scale-up pays cold-start.
- Supabase signed URL expiry is 7 days (configurable via `SUPABASE_URL_EXPIRY_SECONDS`). If clips need longer retention or shorter (for abuse resistance), tune that.

---

**Report compiled from direct measurements during test sessions 2026-04-17 and 2026-04-18. All latency numbers are medians or means of observed runs (typically one sample per config except H100 warm where we have 8 samples).**
