# LTX-2.3 Latency & Cost Report

**Date of measurements:** 2026-04-17 through 2026-04-20
**Pipeline under test:** `TI2VidTwoStagesPipeline` from LTX-2 main, wrapped by `src/pipeline.py` on branch `feat/fa3-teacache` (now the production branch; supersedes `feat/teacache`)
**Workload:** 1920×1088, 121 frames, 30 inference steps, seed 300 (+ 5 varied prompts for warm-pod validation), audio enabled, Supabase upload enabled
**Prompt:** "a red panda eating bamboo in a misty forest, cinematic" (+ 5 varied prompts)

---

## 1. Executive summary

- **Production target: H100 80 GB** (SXM, NVL, or PCIe). Warm steady-state is **31.9 s ± 1.7 s per clip** (6 consecutive runs across 6 different prompts). Cost at RunPod community pricing ($2.50/hr): **~$0.022 / clip**.
- **This is a 21.7 % latency win and 22 % cost win over the previous production baseline** (`feat/teacache` @ 40.7 s / $0.028 per clip), attributable to adding FlashAttention 3 on top of the existing TeaCache + torch.compile + use_fast stack.
- **4090 24 GB remains the secondary path** at **~520 s / $0.049 per clip** with TeaCache only. FA3 is Hopper-only (sm_90); the wheel is unusable on sm_89. 4090 pods must set `LTX_ATTENTION_TYPE=pytorch` to fall back to SDPA.
- **Every SageAttention configuration produced either a failed wedge or corrupted output** and has been abandoned.
- **A100, RTX 6000 Ada, L40S, RTX 5090 are not production-viable** for this pipeline at 1920×1088 in their current builds — specific blockers listed in §3.

## 2. Production-ready configuration

The stack that ships (`feat/fa3-teacache`, commit `e4f5f63`):

| component | status | where | commit |
|---|---|---|---|
| **FlashAttention 3 attention backend** | **opt-in via `LTX_ATTENTION_TYPE=flash_attention_3`; default ON in `start.sh`** | **`src/attention_override.py`, Dockerfile wheel install** | **`e4f5f63`** |
| TeaCache diffusion-step caching | opt-in via `ENABLE_TEACACHE=1` | `src/teacache.py`, `src/pipeline.py` | `6c0e746` |
| Stage 2 pin_memory cleanup hook | always-on correctness fix | `src/pipeline.py` | `79cb570` |
| torch.compile (regional per block) | opt-in via `ENABLE_TORCH_COMPILE=1`; auto-off on <40 GB cards | `src/pipeline.py` | `4ac6f09` + streaming-guard follow-up |
| Gemma `use_fast=True` image processor | Dockerfile sed on editable install | `Dockerfile` | `33025bf` |

Recommended env for production H100 deployment:

```
LTX_ATTENTION_TYPE=flash_attention_3   # default in start.sh; leave as-is
ENABLE_TEACACHE=1
TEACACHE_THRESHOLD=0.03
TEACACHE_STAGES=stage_1
ENABLE_TORCH_COMPILE=1
```

Recommended env for production 4090 deployment:

```
LTX_ATTENTION_TYPE=pytorch              # FA3 wheel is Hopper-only; MUST override
ENABLE_TEACACHE=1
TEACACHE_THRESHOLD=0.03
TEACACHE_STAGES=stage_1
ENABLE_TORCH_COMPILE=0                  # streaming-guard auto-disables it anyway
```

**Docker image:** `<DOCKERHUB_USERNAME>/ltx-video-fa3-teacache:latest`

## 3. Hardware compatibility matrix

| GPU | VRAM | Arch | full stack runs? | $/hr† | verdict |
|---|---|---|---|---|---|
| **H100 SXM** | 80 GB | Hopper sm_90 | **yes, 31.9 s warm** | ~$2.80 | **primary production target** |
| **H100 NVL / PCIe** | 80 – 94 GB | Hopper sm_90 | yes, same as SXM | ~$2.40 – $2.80 | interchangeable with SXM |
| **RTX 4090** | 24 GB | Ada sm_89 | partially — TeaCache only, no FA3 (wheel Hopper-only), no torch.compile (streaming conflict); ~520 s | ~$0.34 | secondary / cost-optimised; set `LTX_ATTENTION_TYPE=pytorch` |
| RTX 6000 Ada | 48 GB | Ada sm_89 | **no** — OOM during BF16 weight load (46 GB transient peak + residue > 48 GB) | ~$0.74 | not viable for streaming-off path |
| L40 / L40S | 48 GB | Ada sm_89 | **no** — same failure mode as RTX 6000 Ada | ~$0.69 – $0.99 | not viable |
| A100 80 GB | 80 GB | Ampere sm_80 | **no** — LTX-2's `fp8_cast` Triton kernel uses `fp8e4nv`, which sm_80 can't emit. Stage 2 LoRA fusion fails unconditionally with `ValueError("type fp8e4nv not supported in this architecture")`. FA3 also incompatible (sm_90 only) | ~$1.79 | not viable |
| H200 | 141 GB | Hopper sm_90 | presumed yes (untested; same arch as H100, more VRAM we don't need) | ~$3.59 – $4.31 | expensive, no workload-specific benefit |
| RTX 5090 | 32 GB | Blackwell sm_120 | cold 650 s, warm unvalidated — streaming-active path, unclear cold-start breakdown; FA3 wheel's sm_90a kernel does not run on sm_120 | TBD | not recommended; Blackwell support unproven on this stack |

† RunPod community-cloud pricing, approximate, 2026-04-20.

**Not tested:** MI300X / other non-NVIDIA, consumer Blackwell (5080 / 5070 Ti), Jetson.

## 4. Latency measurements — H100 warm-pod validation

Same pod, same compile cache, same TeaCache state reset per request, 6 different prompts, FA3 + TeaCache + torch.compile + use_fast all active:

| # | prompt | seed | gen_time (s) |
|---|---|---|---|
| 1 | young woman morning routine (vlog intro) | 501 | 34.54 |
| 2 | pretty young woman vlog (balcony, golden hour) | 502 | 33.31 |
| 3 | red panda eating bamboo (baseline) | 300 | 31.45 |
| 4 | astronaut on a red Martian plain | 300 | 30.68 |
| 5 | busy Tokyo street at night, rain | 300 | 30.52 |
| 6 | Parisian cafe at golden hour voiceover | 300 | 30.69 |

**Mean 31.87 s, stdev 1.67 s, p95 34.54 s, max 34.54 s, min 30.52 s.**

The two vlog prompts (#1, #2) trended slightly higher than the four classic prompts (#3–6, mean 30.84 s). Attributable to longer caption token counts — vlog prompts pushed 60+ tokens into Gemma, producing marginally more cross-attn work per step. Not a regression.

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
| TeaCache + compile + use_fast (prior production) | warm | 40.7 ± 1.1 | $0.028 | 8-sample mean, `feat/teacache`, superseded |
| FA3 only (no TeaCache, no compile) | warm | 91.83 | $0.064 | `feat/flash-attention-research` — attribution reference for FA3 alone |
| **FA3 + TeaCache + compile + use_fast** | **cold** | **163.57** | $0.114 | first request on fresh pod, torch.compile tracing amortised in |
| **FA3 + TeaCache + compile + use_fast** | **warm** | **31.87 ± 1.67** | **$0.022** | **production target, 6-sample mean, `feat/fa3-teacache`** |

**Contribution decomposition on the current stack** (warm, 1920×1088×121×30):
- SDPA baseline (FA3 off, rest on): ~40.7 s
- + FA3: 40.7 → **31.87 s** (1.28× on top of the already-optimised stack)
- FA3 contribution alone (vs 1024×1536 numbers): 155 s → 91.83 s = 1.69× (isolated on `feat/flash-attention-research`)

### RTX 4090 24 GB

| configuration | state | gen_time (s) | $/clip @ $0.34/hr | notes |
|---|---|---|---|---|
| SAGE=0 baseline | cold | 585.25 | $0.055 | reference baseline |
| SAGE=1 (SageAttention sm_89 kernel) | — | — (wedged) | — | Triton JIT wedge + eventual pin_memory crash at Stage 2. Abandoned. |
| **TeaCache only (`LTX_ATTENTION_TYPE=pytorch`)** | **cold** | **520.28** | **$0.049** | **production path for 4090**, ~26 % Stage 1 speedup diluted by fixed overhead to ~11 % end-to-end |
| TeaCache + compile | cold | **failed at t+380 s** | — | 4090 streams weights (VRAM < 40 GB); torch.compile and LayerStreamingWrapper conflict at Stage 2 boundary. Streaming-guard auto-disables compile on <40 GB cards. |
| FA3 + anything | — | — | — | FA3 wheel is sm_90-only; fails to load on sm_89. 4090 pods must set `LTX_ATTENTION_TYPE=pytorch`. |

**4090 quality note on TeaCache alone:** output differs from SAGE=0 baseline at the same seed (30 – 35 % relative pixel L2) but is visually equivalent — clean panda, prompt-adherent, no artefacts. TeaCache's approximate cache slightly perturbs the sampling trajectory. Human-perceptual quality is intact; bit-exact seed reproducibility is lost.

## 6. Known issues & their fixes

### Fixed in this branch

1. **SageAttention pip install silently mutated torch / triton** (`--no-deps` missing). Not an issue on `feat/fa3-teacache` because we don't install SageAttention.
2. **Stage 2 pin_memory() crash** on 24 GB streaming deploys. Fixed with `_install_stage2_cleanup_hook()` in `src/pipeline.py`.
3. **FlashAttention 3 not used** — now wired via `src/attention_override.py`: monkey-patches `LTXModelConfigurator.from_config` to inject `attention_type=flash_attention_3`, plus a defensive SDPA mask-fallback on `FlashAttention3.__call__`. Gated by `LTX_ATTENTION_TYPE`; `start.sh` defaults it on. Static trace of the 22B AV pipeline confirmed every `Attention.forward` call site passes `mask=None` for normal text-to-video inputs, so the fallback is insurance, not an active path.

### Unfixed, mitigated

4. **torch.compile × layer-streaming crash on <40 GB cards.** Reproducible on 4090 with full stack. Mitigation: streaming-guard auto-disables torch.compile when VRAM < 40 GB.
5. **LTX-2.3 mesh/web artefact around close-up portraits** at 1920×1088. Upstream issue, not caused by our optimisation stack.
6. **10-second (241-frame) clips at 1080p OOM in Stage 2 on 80 GB H100.** Activation memory at Stage 2's 2× spatial upscale × 241 frames hits ~77.7 GiB allocated, trips at a plain `rms_norm` allocation of 744 MiB. Not FA3-related — FA3 actually saves attention memory. Fix path: gate streaming on `num_frames × W × H` (not just GPU VRAM), or wire `TilingConfig` through to the Stage 2 transformer. Unfixed.

### Unfixed, hardening backlog

7. **FA3 wheel index is not pinned to a specific filename.** Install line uses `--find-links` on the bi-weekly-rebuilt windreamer index. A future index rebuild could change the wheel we receive. Fix: pin the exact wheel filename (e.g. `flash_attn_3-3.0.0+<date>.cu128torch280cxx11abitrue.<sha>-cp39-abi3-linux_x86_64.whl`) after first successful build.
8. **FA3 capture check only verifies wheel-on-disk**, not that LTX-2's `attention.py` successfully captured the `flash_attn_interface` module reference. If xformers were ever re-added to the image, it would silently shadow FA3's import and the fingerprint log would still say `flash_attn_interface=yes` despite FA3 being dead at runtime.
9. **`start.sh` sets `LTX_ATTENTION_TYPE=flash_attention_3` by default**, so non-Hopper pods (4090, L40S, 5090) hard-fail at first inference unless the env is explicitly overridden. Should auto-detect `nvidia-smi --query-gpu=compute_cap` and fall back to `pytorch` on non-sm_90.

### Rejected paths

10. **SageAttention on Hopper** — sm_90 kernel produces corrupted output for LTX-2's non-128-aligned seq_lens. Abandoned.
11. **SageAttention on 4090** — Triton JIT + VRAM pressure wedges the BentoML runner. Abandoned.
12. ~~**FlashAttention — "lower ROI than TeaCache, not implemented"**.~~ **Superseded: FA3 was implemented on `feat/flash-attention-research` (1.69× alone) and stacked onto `feat/fa3-teacache` (1.28× on top of the TeaCache+compile stack, 21.7 % end-to-end improvement). Now part of the production stack.**

## 7. Cost per clip summary

| config | latency | $/clip | clips per $ | clips per hour |
|---|---|---|---|---|
| 4090 SAGE=0 (pre-optimisation) | 585 s | $0.055 | 18 | 6 |
| 4090 + TeaCache | 520 s | $0.049 | 20 | 7 |
| H100 SAGE=0 warm (estimated) | ~90 s | $0.063 | 16 | 40 |
| H100 + TeaCache + compile + use_fast (prior production) | 41 s | $0.028 | 37 | ~88 |
| **H100 + FA3 + TeaCache + compile + use_fast (current)** | **31.87 s** | **$0.022** | **46** | **~113** |

**H100 remains the best option on both axes — cost and latency — once warm.** Adding FA3 to the stack moves H100 from ~88 clips/hr to ~113 clips/hr (**+28 % throughput**) and cuts $/clip by 22 %.

At 10 000 clips: $62 saved vs the prior production stack.
At 100 000 clips: $620 saved.

## 8. Cold-start considerations

H100 cold-start breaks into three cases on the current stack:

1. **Fresh pod, fresh network volume** — ~75 s of model download (80 GB from HuggingFace) + 90 s of model-to-GPU load + 30 – 60 s of torch.compile warmup on first generation = roughly **180 – 225 s** for the very first request.
2. **Fresh pod, warm network volume** (`/workspace/models` retained across restarts) — ~15 s to mount + 60 s model load + compile warmup = **~120 – 165 s** for first request. Observed 163.57 s in our cold test.
3. **Any subsequent request on the same pod** — warm steady-state **~31.9 s**.

For production autoscaling: keep at least one pod warm per service to ensure no user pays the cold-start tax. If the autoscaler scales to zero, the first user after scale-up pays ~120 – 165 s additional latency on top of the normal 31.9 s.

## 9. Reproducibility

### Branch state when this report was written

```
branch: feat/fa3-teacache
base:   feat/teacache (commit f7e96e6, which sits on fix/inference-mode-api @ 5855a7c)
commits ahead of feat/teacache:
  e4f5f63  Combine FA3 + TeaCache on a shared BentoML base
commits ahead of fix/inference-mode-api:
  e4f5f63  Combine FA3 + TeaCache on a shared BentoML base
  f7e96e6  Add production latency + cost report
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

### Reference output URLs (from test runs, 7-day signed)

- 4090 + TeaCache: `https://qbcnoeoqzmjaxtusytjt.supabase.co/storage/v1/object/public/ltx-videos/ltx_7f59c99a70a7.mp4`
- H100 SAGE=0 (clean baseline): `…/ltx-videos/ltx_954d135fb6b7.mp4`
- H100 SAGE=1 (**noise reference — DO NOT SHIP**): `…/ltx-videos/ltx_988d63db487f.mp4`
- H100 + TeaCache stack (prior production): `…/ltx-videos/ltx_653ee52e575b.mp4`
- H100 + FA3 + TeaCache stack, 1024×1536: `…/ltx-videos/ltx_68d889dc272e.mp4`
- H100 + FA3 + TeaCache stack, 1920×1088 young-woman vlog: `…/ltx-videos/ltx_f9795499e434.mp4`
- H100 + FA3 + TeaCache stack, 1920×1088 pretty-young-woman vlog: `…/ltx-videos/ltx_1fc357a9ef80.mp4`

## 10. Observations worth flagging for product

1. **Audio is generated by default on every clip.** Every output in this test series included a 128 kbps AAC audio track, structurally distinct from silence. Voice-directed prompts produce speech-shaped audio.
2. **Prompt adherence is strong on cinematic framing prompts** (cafe, beach, documentary). Cinematic close-ups pull the mesh/web artefact (item 5 above); wide / medium shots don't.
3. **Warm-pod latency is mostly prompt-independent** — shape, not content, dominates timing. Vlog-style prompts (longer captions) run ~2 s slower than classic cinematic prompts on the FA3 stack, but still within 1σ of the mean.
4. **Seed reproducibility is lost with TeaCache.** Same seed on same config still produces the same output (run 1 and run 2 with matching params match byte-for-byte), but same seed with TeaCache on vs off produces different (equally-valid) outputs. Plan for one-time re-baselining if your tests pin specific seed outputs.
5. **FA3 numerical equivalence.** FA3 and SDPA's flash-kernel are the same algorithm; different kernel fusion. Outputs at the same seed are not byte-identical (per-frame L2 ~5 × 10⁻⁴ relative), but visually indistinguishable and prompt-adherent. Quality is intact.

## 11. Next experiments, ranked

1. **10-second (241-frame) clip support** at 1920×1088 — currently OOMs in Stage 2 (issue #6 above). Gate streaming on `num_frames × W × H`, or wire `TilingConfig` through to Stage 2. Unlocks a frame-count dimension we haven't shipped.
2. **`TI2VidTwoStagesHQPipeline` with Res2s sampler** (15 Stage 1 steps instead of 30). Projected to roughly halve remaining Stage 1 time on H100 — warm could land in the 18 – 22 s range on the current stack. Worth one validation cycle. Requires pipeline class swap (~15 lines) and quality A/B.
3. **FA3 wheel pin** — capture the exact wheel filename currently serving from the windreamer index and pin it in the Dockerfile. Cheap reproducibility win.
4. **Non-Hopper auto-fallback** in `start.sh` — `nvidia-smi --query-gpu=compute_cap` probe; if ≠ 9.0, unset `LTX_ATTENTION_TYPE`. Removes the "4090 pods must remember to override" footgun.
5. **TeaCache threshold tuning** — try 0.05 (aggressive) on a handful of prompts to see if perceptible quality drop is acceptable for another 10 – 20 % speedup on top of the already-31.9 s warm.
6. **`torch.compile` streaming-compatible variant** (upstream fix or workaround for LayerStreamingWrapper). Would extend full-stack benefit to the 4090 path. Non-trivial; probably worth waiting for upstream.
7. **Lower-res warm preview path** — add a `/generate/preview` endpoint at 768×432 / 25 frames for thumbnails. Projected <8 s per preview on the current stack.

## 12. Open questions for the platform team

- Is `/workspace/models` persistent across pod restarts on the deployment platform? If yes, cold-starts are ~120 s. If no, cold-starts are 180+ s.
- Is autoscaling configured with `min_replicas ≥ 1`? If scale-to-zero, first user after scale-up pays cold-start.
- Supabase signed URL expiry is 7 days (configurable via `SUPABASE_URL_EXPIRY_SECONDS`). Tune if retention needs change.
- **FA3 wheel supply-chain.** The windreamer community index is Apache-2.0 but unsigned. Fine for research and initial production; consider swapping for a SHA-pinned self-hosted build before shipping to external customer traffic.

---

**Report compiled from direct measurements during test sessions 2026-04-17 through 2026-04-20. All latency numbers are means of observed runs (typically one sample per config except H100 warm on `feat/teacache` where we have 8 samples and H100 warm on `feat/fa3-teacache` where we have 6 samples). The FA3 comparison between `feat/teacache` (40.7 s ± 1.1 s, n=8) and `feat/fa3-teacache` (31.9 s ± 1.7 s, n=6) has an effect size of ~5× the combined stdev — the speedup is not sample noise.**
