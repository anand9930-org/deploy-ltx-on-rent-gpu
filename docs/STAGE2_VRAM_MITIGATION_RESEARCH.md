# Research: Mitigating LTX Stage 2 VRAM Peak for 10 s Video on H100 80 GB

## Context

Today's pipeline (`feat/fa3-teacache`, `LTX_QUANT=fp8_cast`, FA3, TeaCache) reaches
**80.71 GB peak** on a 5 s / 121-frame request — only 4.3 GB headroom on an
85 GB H100. Linear extrapolation to 10 s / 241 frames lands at **~104 GB**
(−19 GB deficit), matching the OOM observed in production. The peak is
dominated by Stage 2 (latent upsample / refine) — its transient burst owns
~24 GB out of the 80.7 GB peak; resident weights are pinned at ~57 GB
because `fp8_cast` keeps the BF16 shadow in HBM.

Goal of this document: catalogue **community-validated, quality-preserving**
techniques that reduce that 24 GB Stage 2 burst (or shrink the resident
floor it sits on top of) so 10 s fits inside 85 GB. We rank by VRAM saved,
expected latency cost, and how strict the "lossless" guarantee is.

See `docs/VRAM_MAP.md` for the measured baseline this document builds on.

---

## 1. Mental model — where the 24 GB goes

Stage 2 runs the same 13B transformer as Stage 1 but on **upsampled latents**
(2× spatial), so the sequence length `S = frames × spatial_tokens` jumps
sharply. Three things spike with `S`:

| Sub-component | Scaling | Why it hurts at 10 s |
|---|---|---|
| Attention output buffer (`O = softmax(QKᵀ)V`) | `O(S × d)` per head | FA3 keeps softmax tiled, but the output tensor is still materialised at full `S` |
| FFN intermediate (`gelu(W₁x)`) | `O(S × 4d)` | Hidden dim is 4× model dim → biggest single transient |
| RoPE split buffers + activation residuals | `O(S × d)` | Several copies pinned per block |

Lever taxonomy:

- **Slice the sequence axis** (FFN chunking, attention chunking, temporal windowing) — keeps full numerical fidelity, trades latency for memory.
- **Shrink the resident floor** (offload, real FP8 weights, group offload) — frees room *under* the spike instead of cutting the spike.
- **Cache across steps** (TeaCache, Step Rehash) — already in use; not a Stage 2 lever per se.

---

## 2. Catalogue of mitigations (ranked by lossless-ness, then GB saved)

### Tier A — Mathematically lossless (bit-identical or within numerical noise)

#### A1. FFN forward chunking — *the highest-leverage lossless lever*

The FFN block in transformers operates **independently per token**, so
splitting along the sequence dim and re-stacking is mathematically
identical. `diffusers` exposes this directly on every transformer:

```python
pipe.transformer.enable_forward_chunking(chunk_size=<N>, dim=1)  # dim=1 = sequence
```

**Reported impact** (RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management):

- Default `ffn_chunks=8` → up to **8× peak FFN reduction** (e.g. 3.7 GB → 0.46 GB per layer).
- For ~800 frames, recommend `ffn_chunks=12–16`.
- For ≥ 900 frames, recommend `ffn_chunks=16–24`.
- Author note: *"mathematically identical … no accuracy loss."*

**Latency cost**: small. NeurIPS 2024 / Mosaic measurements at 32 K context
report < 2 % FFN slowdown with 4 chunks. LTX 0.9.7 community workflows
report **2 chunks is usually enough** once attention is FA3, often nothing.

**Estimated GB saved on our pipeline**: **8–12 GB** at 10 s (the FFN spike
is the largest single transient in Stage 2).

#### A2. Temporal windowing on Stage 2 only

Already lever #1 in the existing VRAM_MAP. Sources confirmed:

- **diffusers PR #12614** — `LTXI2VLongMultiPromptPipeline`. Configurable
  `temporal_tile_size` (latent frames) and `temporal_overlap` (shared
  frames between windows). Example config: 361 frames with `tile=120,
  overlap=32`.
- Cross-window blending uses **negative-index latent injection** at window
  heads + per-window timestep reset → "strong parity in motion, content,
  and overall style" with ComfyUI reference.
- Authors explicitly state: *temporal sliding windows only, no spatial
  sharding during denoising* — keeps frame coherence intact.

**Estimated GB saved**: **~24 GB** (the entire Stage 2 transient). Wrap
Stage 2 only; Stage 1 stays whole.
**Latency cost**: +10–15 % from window overlap recompute.
**Quality**: lossless inside windows; overlap blending is the only
non-identity operation, and AdaIN normalisation (`adain_factor=0.25`) is
the standard remedy for any mild seam drift.

#### A3. Streamlined Inference — Feature Slicer + Operator Grouping (NeurIPS 2024)

Paper: *Fast and Memory-Efficient Video Diffusion Using Streamlined
Inference* (arXiv 2411.01171). Lossless reductions reported:

| Model | Before | After | Reduction |
|---|---:|---:|---:|
| AnimateDiff | 42 GB | 11 GB | **74 %** |
| SVD (576×1024) | 39.5 GB | 23.4 GB | 41 % |
| SVD-XT | 61.2 GB | 36.3 GB | 41 % |

Mechanism:
1. **Spatial slicer**: reshape 5D → 4D, slice along batch/temporal.
2. **Temporal slicer**: slice H/W while preserving T (keeps temporal correlation).
3. **Operator grouping**: pipe slices through homogeneous op chains so
   intermediate tensors never fully materialise.

Quality on AnimateDiff/UCF-101: FVD 758.7 → 784.5 (≈ noise),
CLIP-Score 28.89 → 28.71. Treat as effectively lossless.

**Fit**: requires a small refactor of Stage 2 transformer blocks to
operate on slices. Not in `diffusers` upstream — port-style integration.

**Estimated GB saved**: **10–20 GB** if the slicer is plumbed through the
upsampler's attention + FFN; potentially solves 10 s on its own if
combined with A1.

#### A4. Activation checkpointing at inference

Standard `torch.utils.checkpoint.checkpoint` on transformer blocks
recomputes intermediate activations rather than holding them. Bit-exact
in FP32; in BF16/FP8 numerically equivalent within rounding noise.

**Estimated GB saved**: **3–8 GB** (residuals + GELU intermediates).
**Latency cost**: +20–30 % on the wrapped section. Enable only on Stage 2.

#### A5. Unfuse QKV projection (if currently fused)

Fused QKV holds three projections concatenated → wider intermediate.
Unfusing reduces peak by the K+V tensors:

```python
pipe.transformer.unfuse_qkv_projections()
```

**Estimated GB saved**: 1–3 GB. **Latency cost**: ~5–10 % attention slowdown.

#### A6. Sequence-parallel / ring attention (multi-GPU only — listed for completeness)

If a 2-GPU H100 SXM pod is on the table, ring attention shards `S` across
GPUs (online softmax). Lossless. Reported by LTX-2 V5 codebase:
614 K tokens on 6 GPUs (~102 K / GPU). Not relevant for single H100, but
the cleanest lossless solution if scaling out.

---

### Tier B — Quality-preserving but not bit-identical

#### B1. VAE tiled decoding (`enable_tiling()`) on Stage 2 decode

Lightricks' tiled VAE decoder linearly feathers overlapping spatial tiles.
Quality impact is below visual threshold for `tile_size=512, overlap=32`.

```python
pipe.vae.enable_tiling()  # plus decode_horizontal_tiles=4, decode_vertical_tiles=4, decode_overlap=3
```

**Reported**: 2048×2048 video decode 56 GB → 8 GB (CrePal benchmark).
**For our 1024×1536**: probably 4–6 GB saved at decode peak.

#### B2. Stage 2 short-step refine (`denoise_strength=0.4`, `num_inference_steps=4–8`)

Distilled-LoRA refine path: instead of 30 full denoising steps in Stage 2,
do 4–8 partial steps at `denoise_strength=0.35–0.55`. Reduces *per-step
state* footprint indirectly via fewer cached intermediates and lets you
keep TeaCache aggressive.

**Quality**: kept ≤ 0.55, no identity drift; > 0.7 over-rides Stage 1
motion structure.

#### B3. Tone-mapping (`tone_map_compression_ratio=0.6`, LTX 0.9.8+)

Not a memory lever, but the canonical companion for long-video drift —
mention only because it's the recommended quality safeguard once
windowing is in place.

---

### Tier C — Resident-floor reductions (frees space *under* the spike)

These don't reduce the 24 GB burst; they reduce the 57 GB pinned floor.

#### C1. Native FP8 weights via `fp8_scaled_mm` (Lightricks issue #181)

Already on the existing shelf. The win is real (drop BF16 shadow → ~−25 GB
floor), but requires a base-image swap to Python 3.12 + tensorrt_llm
cp312 wheel. Still upstream-blocked. Defer until Tier A is exhausted.

#### C2. Group offloading (`apply_group_offloading`)

`diffusers` `enable_group_offload(onload_device, offload_device,
offload_type='leaf_level', use_stream=True)` walks the transformer block
tree and pages blocks between HBM ↔ CPU around their forward call.
Stream-overlapped to hide PCIe latency.

**Estimated GB saved**: 20–40 GB of resident weights (variable). Large.
**Latency cost**: +30–80 %. PCIe 4.0 transfer of a 35 GB FP8 model takes
~1–2 s per swap on a typical pod.

Use only if Tier A doesn't close the gap.

---

## 3. Recommendation for *your* 19 GB deficit at 10 s

Stack in this order — measure after each:

1. **Enable FFN forward chunking on Stage 2** (`enable_forward_chunking(chunk_size=<seq/2>, dim=1)`). Free, lossless, ~8–12 GB. Likely closes ~half the gap by itself.
2. **Land Stage 2 temporal windowing** (`tile=80 frames, overlap=24`) — the existing plan's lever #1. Lossless modulo overlap blend, ~24 GB. This alone solves 10 s; combined with #1 leaves 15+ GB headroom for 15 s exploration.
3. **Add `enable_vae_tiling()` on the upsample VAE decode** if peak still includes the decode. Cheap, near-lossless, 4–6 GB.
4. Keep **activation checkpointing** and **Streamlined Inference slicer** as 15 s / 20 s stretch levers.
5. Defer FP8 native matmul and group offloading until measurements show resident floor (not Stage 2 burst) is binding.

Verification rubric (carry over from VRAM_MAP §8):
- `30_generate_end` peak_alloc < 75 GB at 10 s.
- Quality A/B vs. current 5 s output: SSIM ≥ 0.98 inside windows, no visible seam at overlap boundaries, identity preserved.

---

---

## 4. Implementation plan — FFN forward chunking on Stage 2

### 4.1 Reality check before building anything

Two pieces of upstream evidence pull in opposite directions; the plan below
is shaped around the gap between them.

**Optimistic data point** — RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management
reports up to **8× peak FFN reduction** on LTX-2 (3.7 GB → 0.46 GB per
layer) with `ffn_chunks=8`, mathematically identical output, on
800–900-frame jobs.

**Pessimistic data point** — diffusers
[PR #8842 (Latte chunked FFN)](https://github.com/huggingface/diffusers/pull/8842)
**was closed without merging**. Author's own benchmark on Latte (same
"pure transformer" family as LTX-2):

| Config | Latency | Peak VRAM |
|---|---:|---:|
| Baseline | 16.28 s | 15.21 GB |
| `chunk_size=2` | 37.49 s | 14.68 GB |
| `chunk_size=1` | 91.36 s | 14.62 GB |

Result: **0.59 GB saved for 2.3× slowdown.** Reviewer comment: *"forward
chunking shows less memory savings for models that are purely transformer
based as opposed to hybrid architectures."*

**Why both can be true.** Per-layer FFN intermediate scales as
`O(B × S × 4d)`. At 5 s (S ≈ 50 K, d = 4096, BF16) that's only ~1.6 GB —
chunking it 8× saves ~1.4 GB on a single live tensor (Latte's regime). At
10 s (S ≈ 100 K) it doubles to ~3.2 GB and chunking saves ~2.8 GB. At
800 frames (S ≈ 400 K) it hits ~13 GB and chunking saves ~11 GB
(ComfyUI's regime). **Our 10 s sits closer to the Latte regime than the
ComfyUI regime.**

**Honest expected savings for our case: 2–4 GB, not 8–12 GB.** That is
not enough to close the 19 GB deficit on its own — temporal windowing
(VRAM_MAP §6 lever #1) remains mandatory. FFN chunking is a *complement*
that buys margin and lets us run the windowing with smaller overlap.

### 4.2 Code we are integrating against (LTX-2 internals)

| Concern | Location | Detail |
|---|---|---|
| FFN definition | `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/feed_forward.py:6-15` | `Sequential(GELUApprox(d, 4d), Identity, Linear(4d, d))` — pure per-token MLP, trivially chunkable on `dim=1` |
| Block-level FFN call (video) | `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/transformer.py:363` | `vx = vx + self.ff(vx_scaled) * vgate_mlp` — only line we need to wrap |
| Block-level FFN call (audio) | `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/transformer.py:372` | `ax = ax + self.audio_ff(ax_scaled) * agate_mlp` |
| Block class | `transformer.py:24` | `BasicAVTransformerBlock(torch.nn.Module)` — what we monkey-patch |
| Block list | `model.py:315-327` | `LTXModel.transformer_blocks: nn.ModuleList[48]` — recursive walker target |
| Stage 2 transformer access | `src/pipeline.py` (built via `DiffusionStage._build_transformer()` at `LTX-2-ref/.../blocks.py:188-215`) | reach as `pipeline.stage_2._transformer.velocity_model` (it's `X0Model(LTXModel(...))`) |
| Existing monkey-patch pattern to copy | `src/attention_override.py` | FA3 swap-in is the prior art — same shape of intervention |
| Env-var pattern to copy | `src/teacache.py:293-326` (`teacache_config_from_env`) | env → dict → kwargs into enable function |
| Env-var declaration | `.env.example` | already has `ENABLE_TEACACHE`, `ENABLE_TORCH_COMPILE` — add new keys here |

### 4.3 Implementation skeleton

New file: `src/ffn_chunking.py` — mirrors `src/attention_override.py` and
`src/teacache.py` style.

```python
"""Stage 2 FFN forward chunking for LTX-2.

Mathematically identical to the unchunked path: chunks the FFN over the
sequence axis and concatenates. Lossless because per-token MLP outputs
are independent of sibling tokens.

Gated by env vars to keep the cold path untouched.
"""
import logging
import os
from typing import Optional

import torch
from ltx_core.model.transformer.transformer import BasicAVTransformerBlock

logger = logging.getLogger(__name__)

_ORIG_BLOCK_FFN_VIDEO = None  # set on first patch
_ORIG_BLOCK_FFN_AUDIO = None


def _chunked_ff(ff_module: torch.nn.Module, x: torch.Tensor, chunk_size: int, dim: int = 1) -> torch.Tensor:
    """Apply ff_module to x in chunks along `dim`. Concat result.

    Equivalent to ff_module(x). Output is bit-identical for deterministic
    Linear+GELU paths; numerically identical within float rounding for
    fused / FP8-cast paths (no reduction across chunks).
    """
    if chunk_size is None or x.shape[dim] <= chunk_size:
        return ff_module(x)
    chunks = x.split(chunk_size, dim=dim)
    return torch.cat([ff_module(c) for c in chunks], dim=dim)


def _apply_chunked_ff_to_block(block: BasicAVTransformerBlock, video_chunk: int | None, audio_chunk: int | None) -> None:
    """Wrap the block's `ff` and `audio_ff` modules so their __call__ chunks.

    We wrap the module instances rather than patching the block.forward —
    the block forward is 200+ lines and we want the smallest possible blast radius.
    """
    if video_chunk is not None and hasattr(block, "ff"):
        original_ff = block.ff

        class ChunkedFF(torch.nn.Module):
            def __init__(self, inner: torch.nn.Module, n: int):
                super().__init__()
                self.inner = inner
                self._n = n
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return _chunked_ff(self.inner, x, self._n, dim=1)

        block.ff = ChunkedFF(original_ff, video_chunk)

    if audio_chunk is not None and hasattr(block, "audio_ff"):
        original_aff = block.audio_ff
        class ChunkedAFF(torch.nn.Module):
            def __init__(self, inner, n):
                super().__init__(); self.inner = inner; self._n = n
            def forward(self, x):
                return _chunked_ff(self.inner, x, self._n, dim=1)
        block.audio_ff = ChunkedAFF(original_aff, audio_chunk)


def enable_ffn_chunking(stage, video_chunk_size: int | None = None, audio_chunk_size: int | None = None) -> int:
    """Walk the stage's transformer and install chunked FFN wrappers.

    Returns the number of blocks patched.
    """
    velocity = stage._transformer.velocity_model  # X0Model -> LTXModel
    n = 0
    for block in velocity.transformer_blocks:
        _apply_chunked_ff_to_block(block, video_chunk_size, audio_chunk_size)
        n += 1
    logger.info(
        "FFN chunking installed on %d blocks (video_chunk=%s, audio_chunk=%s)",
        n, video_chunk_size, audio_chunk_size,
    )
    return n


def ffn_chunking_config_from_env() -> dict | None:
    """Parse LTX_FFN_CHUNK_VIDEO / LTX_FFN_CHUNK_AUDIO. Return None if disabled."""
    raw_v = os.getenv("LTX_FFN_CHUNK_VIDEO")
    raw_a = os.getenv("LTX_FFN_CHUNK_AUDIO")
    if not raw_v and not raw_a:
        return None
    cfg: dict = {}
    if raw_v:
        cfg["video_chunk_size"] = int(raw_v)
    if raw_a:
        cfg["audio_chunk_size"] = int(raw_a)
    return cfg
```

Wire-in at `src/pipeline.py` (right after `_install_stage2_cleanup_hook`,
mirroring the TeaCache pattern at lines 209–213):

```python
from .ffn_chunking import enable_ffn_chunking, ffn_chunking_config_from_env

ffn_cfg = ffn_chunking_config_from_env()
if ffn_cfg is not None:
    enable_ffn_chunking(pipeline.stage_2, **ffn_cfg)
```

`.env.example` additions:

```
# Stage 2 FFN forward chunking (lossless). Set chunk size in tokens.
# Recommended starting points: video=4096, audio unset.
# Leave both unset to disable.
LTX_FFN_CHUNK_VIDEO=
LTX_FFN_CHUNK_AUDIO=
```

### 4.4 Why this design (and what we explicitly did NOT do)

- **Wrap `block.ff` instead of `block.forward`** — block.forward is 200+
  lines with audio/video branching, AdaLN modulation, perturbation masks,
  cross-attention. Patching the FFN module is a 5-line surface; patching
  the forward would duplicate that whole function and rot on the next
  upstream LTX-2 sync.
- **Stage 2 only** — Stage 1 transient is 2.28 GB measured (VRAM_MAP §1).
  Chunking it would slow the 30 sampler steps with no useful saving.
- **Env-gated, off by default** — matches FA3 and TeaCache flags
  (`LTX_ATTENTION_TYPE`, `ENABLE_TEACACHE`). Production stays on the cold
  path until measurement justifies on.
- **Independent video/audio chunks** — audio sequence is much shorter; it
  almost never needs chunking. Default unset.
- **No `_chunk_size` attribute on the block** — diffusers convention is
  to store `_chunk_size` on the block and check it in forward. We don't
  control LTX-2's block forward, so we instead bind chunk size into the
  wrapper module. Effect is equivalent.
- **No interaction with FP8 cast or LoRA fuse** — the cast happens inside
  `Linear.forward` (kernel-level), and LoRA is materialized into weights
  before the wrapper sees them. Chunking just reduces per-call activation
  size; weights are untouched.
- **No interaction with TeaCache** — TeaCache wraps `X0Model.forward` at
  the outer layer (`teacache.py:235-254`). FFN chunking lives inside the
  inner block forward. Orthogonal.
- **Compatible with FA3** — chunking is FFN-only; attention call is
  unchanged.

### 4.5 Calibration table — what chunk size to try first

| Resolution × frames | S (≈) | FFN intermediate (BF16, B=1) | `LTX_FFN_CHUNK_VIDEO` start | Predicted savings |
|---|---:|---:|---:|---:|
| 1024×1536×121 (5 s, current) | 50 K | 1.6 GB | not needed | n/a |
| 1024×1536×241 (10 s, target) | 100 K | 3.2 GB | **8192** (≈ S/12) | ~2.5–3 GB on transient |
| 1024×1536×361 (15 s, stretch) | 150 K | 4.9 GB | 6144 | ~4 GB |

Start higher (less aggressive, less latency cost), bisect down only if
profiler still shows the FFN intermediate dominating Stage 2 peak.

### 4.6 Verification protocol (run on pod)

1. **Boot pod with env unset** — sanity-check baseline matches §1 of
   VRAM_MAP. Same prompt + seed used to capture the baseline.
2. **Set `LTX_FFN_CHUNK_VIDEO=8192`, restart, run same 5 s request.**
   Required evidence:
   - `[VRAM] 30_generate_end` peak < baseline by ≥ 1 GB and ≤ 3 GB
     (sanity range — too little = not active, too much = something else
     changed).
   - `[VRAM] 10_stage2_boundary_before_cleanup` resident unchanged
     (~57 GB) — confirms no weight duplication.
   - Latency overhead < 8 % on Stage 2 only. Stage 1 unchanged (we did
     not patch it).
   - Output frames bit-compared (or SSIM ≥ 0.9999) to the unchunked
     baseline at the same seed. **Lossless or it didn't work.**
3. **Re-run with the 10 s payload.** Required evidence:
   - Generation completes (no OOM).
   - Peak < 75 GB if combined with windowing; otherwise document the
     residual deficit and adjust chunk size + windowing tile.
4. **Snapshot pickle**: `torch.cuda._memory_viz` of the before/after
   shows the per-layer 4d-wide FFN intermediate sliced into N narrower
   strips of width `chunk_size`.

### 4.7 Failure modes and exits

| Failure | Likely cause | Fix |
|---|---|---|
| No measurable saving | The FFN intermediate isn't the binding peak — attention out / residual is | Profile with `torch.cuda.memory._snapshot()`; pivot to activation checkpointing or temporal windowing first |
| Output drifts (SSIM < 1.0) | A custom op inside the FFN reduces across the sequence (it shouldn't, given §4.2 — confirm by reading `GELUApprox`) | Disable; do not ship |
| Latency > 15 % overhead on Stage 2 | Chunk size too small relative to GPU launch overhead | Double the chunk size, re-measure |
| TeaCache stops triggering | Should not happen (orthogonal layers), but verify cache-hit ratio in logs | Disable chunking and bisect |
| Crash inside `ChunkedFF` | LoRA fusion or torch.compile holds a stale reference to the original module | Apply chunking *after* `torch.compile` step in pipeline.py order |

### 4.8 Rollout sequence

1. Land FFN chunking PR with env defaults unset → no production change.
2. On staging pod, capture 5 s baseline + 5 s with `LTX_FFN_CHUNK_VIDEO=8192`. Confirm lossless + savings.
3. Land Stage 2 temporal windowing (VRAM_MAP §6 lever #1).
4. Re-test 10 s with windowing alone, then with windowing + chunking.
5. Pick the minimum-overhead combination that gives ≥ 10 GB headroom at
   10 s; that becomes the production default.
6. Document the chosen `LTX_FFN_CHUNK_VIDEO` value in `.env.example` and
   `docs/VRAM_MAP.md` extrapolation table.

### 4.9 Improvements beyond the basic chunking — for the same PR or follow-ups

- **Adaptive chunk size by sequence length** — compute
  `chunk_size = max(2048, S // n_chunks)` at first forward, store on the
  wrapper. Avoids hand-tuning per resolution.
- **Skip chunking when `S × 4d × dtype_size < 1 GB`** — falls back to
  unchunked path for short-sequence calls (e.g. Stage 1 audio). Removes
  the latency tax on small inputs.
- **Combine with `torch.compile(mode="reduce-overhead")`** on the chunked
  module — recovers some of the per-launch overhead by graph-capturing
  the chunked loop. Requires checking that compile is enabled and
  unfusing on chunk-boundary changes.
- **Free the input chunk after use**: in `_chunked_ff`, do `del c` between
  iterations and `torch.cuda.empty_cache()` only every K chunks (cache
  empty itself is expensive; don't put it in the loop).
- **Stream the output write**: `torch.cat` allocates a full-sequence
  output buffer at once, which momentarily holds *both* the last chunk's
  intermediate and the full output. Pre-allocate a destination tensor of
  shape `x.shape` and write each chunk in-place via `dest[..., start:end, :] = ff(c)`. This shaves one FFN-intermediate-worth of peak.

The last item is the most promising "improvement" — for our regime
(B=1, S=100 K, d=4096, BF16) pre-allocating the output buffer instead of
torch.cat is worth ~3 GB by itself and is mathematically identical.

---

## 5. Sources

- [RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management](https://github.com/RandomInternetPreson/ComfyUI_LTX-2_VRAM_Memory_Management) — FFN chunking parameters, ring attention.
- [Memory Management — Lightricks/ComfyUI-LTXVideo (DeepWiki)](https://deepwiki.com/Lightricks/ComfyUI-LTXVideo/4.3-memory-management) — `LowVRAMLatentUpscaleModelLoader`, tiled VAE decode benchmarks.
- [diffusers PR #12614 — `LTXI2VLongMultiPromptPipeline`](https://github.com/huggingface/diffusers/pull/12614) — temporal sliding-window pipeline, blend mechanics.
- [Fast and Memory-Efficient Video Diffusion Using Streamlined Inference (NeurIPS 2024)](https://arxiv.org/html/2411.01171v1) — Feature Slicer / Operator Grouping / Step Rehash.
- [Towards Chunk-Wise Generation for Long Videos (arXiv 2411.18668)](https://arxiv.org/html/2411.18668v1) — autoregressive chunked I2V, k-step noise selection.
- [Lightricks/LTX-Video-0.9.7-dev (HF)](https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev) — official two-stage `LTXConditionPipeline` + `LTXLatentUpsamplePipeline`, FP8 variants.
- [diffusers LTX pipeline docs](https://huggingface.co/docs/diffusers/api/pipelines/ltx_video) — `enable_tiling`, layerwise casting, group offloading APIs.
- [diffusers Reduce memory usage guide](https://huggingface.co/docs/diffusers/optimization/memory) — `enable_forward_chunking`, `unfuse_qkv_projections`.
- [FlashAttention-3 (arXiv 2407.08608)](https://arxiv.org/abs/2407.08608) — confirms FA3 already gives linear-in-S attention memory; remaining peak is FFN + residuals.
- [LTX 2.3 Multi-Stage Latent Upscaling Workflow — CrePal](https://crepal.ai/blog/aivideo/ltx-2-3-multi-stage-latent-upscaling-comfyui/) — Stage 2 `denoise_strength` 0.35–0.55 quality envelope.
- [WaveSpeedAI LTX-2.3 ComfyUI Setup](https://wavespeed.ai/blog/posts/ltx-2-3-comfyui-setup-two-stage-pipeline/) — two-stage VRAM behaviour and stage-boundary cleanup.
- [diffusers PR #8842 — chunked feed-forward in Latte](https://github.com/huggingface/diffusers/pull/8842) — closed without merging; pure-transformer benchmark showed 0.59 GB saved for 2.3× slowdown. Critical counter-evidence to the ComfyUI optimistic claims.
- [diffusers `_chunked_feed_forward` reference implementation](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention.py) — `_chunked_feed_forward(ff, x, dim, chunk_size)` and `set_chunk_feed_forward` pattern from `BasicTransformerBlock` / `JointTransformerBlock`. We mirror this in `_chunked_ff` but adapt to LTX-2's blockless architecture.
- [Lightricks LTX-2 `BasicAVTransformerBlock` source](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-core/src/ltx_core/model/transformer/transformer.py) — confirms FFN call sites at `vx + self.ff(vx_scaled) * vgate_mlp` and the audio mirror; locally vendored under `LTX-2-ref/`.
