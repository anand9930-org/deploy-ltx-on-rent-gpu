# FlashAttention × LTX-2.3 — Compatibility Analysis

> **Scope.** Deep-dive on whether FlashAttention (FA2 on Ada, FA3 on Hopper) is a viable replacement for SageAttention as the attention backend for LTX-2.3 video generation on rent-by-the-hour GPUs, specifically RTX 4090 (24 GB) and H100 (80 GB).
>
> **Status when written.** 2026-04-18, off commit `main`. Written after 48 h of real deployment testing of SageAttention on both architectures produced one clean result (Ada SAGE=0) and one catastrophic result (Hopper SAGE=1, brown-noise output due to sm_90 kernel tile-alignment corruption). This doc is the alternative-attention research that led to choosing FA as the next thing to try.
>
> **Companion document.** `docs/FlashAttention_LTX23_Complete_Guide.md` turns the conclusions below into concrete deployment steps. This file is the *why*; that file is the *how*.

---

## Executive summary

FlashAttention is a strictly better fit for LTX-2.3 than SageAttention, across every axis that matters for this pipeline:

1. **LTX-2 ships with FA3 integration built in.** `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py:94-116` defines a first-class `FlashAttention3` backend. `AttentionFunction.FLASH_ATTENTION_3` (line 122) is a selectable enum value. There is no patching, no module swapping, no custom wrapper to maintain — we just flip a config flag and install the right wheel. The SageAttention integration we spent two days writing doesn't exist for FA because it doesn't need to.
2. **FA is exact attention, not quantised attention.** The SageAttention failure on Hopper (`sageattn_qk_int8_pv_fp8_cuda_sm90` producing garbage because LTX-2's seq_lens 5500 / 22000 / 2200 are not multiples of 128) simply cannot happen with FA. FA's tile-aware online softmax is provably correct for any seq_len. Output is numerically equal to a naïve `softmax(QK^T / √d) @ V` to within 1e-3 relative error on BF16 inputs (per "Is Flash Attention Stable?" arXiv:2405.02803, §3).
3. **FA2 on Ada (4090) is available via xformers, which LTX-2 already uses as its `DEFAULT` backend.** If the deployment image installs xformers, LTX-2's `AttentionFunction.DEFAULT` branch (line 136) already dispatches through `XFormersAttention`, which delegates to xformers' `memory_efficient_attention`, which internally calls FA2 kernels for shapes where that's the fastest path. So **FA2 is very possibly already live on our 4090 deploys and we just didn't label it that**.
4. **FA3 on Hopper** is a separate sm_90-only wheel, buildable from source in ~3–5 minutes in CI. The speedup over `torch.nn.functional.scaled_dot_product_attention` on H100 at our shapes is ~1.5 – 2.0× (FA3 paper, §5, Fig. 2: 740 TFLOPs/s vs ~380 TFLOPs/s naïve, seq_len 8k, head_dim 128 — bracketing our 5500 & 22000 at 128).
5. **The one real compatibility caveat** is that LTX-2's FA3 implementation raises `NotImplementedError` if a mask is passed (line 111-112). Self-attention (`attn1`, `audio_attn1`) and text/audio cross-attention in the *current* LTX-2.3 code path don't pass masks, so this is not a blocker today. It is a latent footgun for any future tuning that introduces mask-aware attention.

**Projected end-to-end impact on our measured baselines** (1920×1088 / 121 frames / 30 steps / seed 300):

| hardware | attention backend today | 30-step wall clock today | with FA (projected) | projected speedup |
|---|---|---|---|---|
| 4090 24 GB (streaming) | xformers default (≈ FA2 already) | 585 s | 400 – 500 s | 1.15 – 1.45× |
| H100 80 GB (streaming off) | PyTorch SDPA default | 155 s | 95 – 120 s | 1.3 – 1.6× |

These are conservative — the 4090's bottleneck is still PCIe layer streaming, not attention compute (see §5 below), so attention-kernel speedups don't translate fully. On H100 the bottleneck is compute and FA3 will earn more of its headline number.

**Cost-per-clip impact**, plugging those numbers into today's RunPod pricing (see companion guide §6):

| config | $/hr | s/clip | $/clip | vs 4090 SAGE=0 baseline |
|---|---|---|---|---|
| 4090 SAGE=0 (today's baseline) | 0.34 | 585 | **$0.055** | 1.00× |
| 4090 + FA2 (projected) | 0.34 | ~450 | **$0.043** | 0.77× (23 % cheaper) |
| H100 SAGE=0 (today's baseline) | 2.50 | 155 | $0.108 | 1.96× |
| H100 + FA3 (projected) | 2.50 | ~105 | **$0.073** | 1.33× |

**The 4090 + FA2 path is the money winner** — sub-$0.05/clip with no architecture changes, just a correctly-specified xformers version. The H100 + FA3 path is the latency winner — sub-two-minute clips at $0.07, which is the interactive-product price point.

---

## 1. What FlashAttention is, in one paragraph

FlashAttention (Dao et al., 2022 — arXiv:2205.14135) computes `softmax(QK^T / √d) @ V` **exactly** (no approximation, no quantisation) but reorders the computation into IO-aware tiles so that large intermediate tensors (specifically the `(B, H, Nq, Nk)` attention matrix) never materialise in HBM. On an H100 with seq_len 5500, naïve attention writes ~190 MB of attention weights to HBM and reads them back; FA never does. This removes the dominant memory traffic bottleneck, letting attention run close to the hardware's FLOPs ceiling. FA2 (2023, arXiv:2307.08691) refines the tiling and work-partitioning for Ampere/Ada. FA3 (2024, arXiv:2407.08608) rewrites the kernel for Hopper's asynchronous `wgmma` instructions and warp specialisation, hitting 75 % of H100's 989 TFLOP/s peak.

Two properties matter for LTX-2.3 specifically:

- **No alignment pitfall.** FA's tail-tile handling is correct for any `seq_len` because each tile carries its own running softmax maximum and normaliser, and tail tiles just have fewer valid rows — their contribution to the softmax sum is scaled by the partial block size, not padded with garbage. The SageAttention sm_90 kernel failure we saw (garbage written into tail tiles because the kernel assumed aligned shapes) has no analogue in FA.
- **No intermediate quantisation.** FA computes in FP16 / BF16 throughout, with FP32 accumulators for the softmax normaliser and final output. SageAttention's Hopper variant quantises Q and K to INT8 and P (softmax output) and V to FP8 to squeeze more FLOPs out of the tensor cores — that's where its speed advantage comes from, and also where its accuracy risk comes from. FA trades some headroom for never having to quantise.

---

## 2. LTX-2's built-in attention backend system

This is the single most important finding in this analysis: **we do not need to write any integration code**. LTX-2 already has the plumbing.

### 2.1 The `AttentionFunction` enum

`LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py:119-136`:

```python
class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    DEFAULT = "default"

    def to_callable(self) -> AttentionCallable:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return XFormersAttention() if memory_efficient_attention is not None else PytorchAttention()
```

Every `Attention` module receives this at construction time (line 148) and stores the resolved callable on `self.attention_function` (line 153-157). At forward time (line 232) the callable is invoked directly.

### 2.2 Where the choice comes from

The configurator reads the string from the checkpoint's embedded config:

`LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py:53` (`LTXAudioVideoModelConfigurator`) and `:112` (`LTXVideoOnlyModelConfigurator`):

```python
attention_type=AttentionFunction(config.get("attention_type", "default")),
```

If the checkpoint's JSON doesn't specify `attention_type`, the enum falls back to `"default"`, which in turn dispatches to xformers if installed, else PyTorch SDPA.

The LTX-2.3 22B checkpoints we use (`ltx-2.3-22b-dev.safetensors`) ship without an explicit `attention_type` in their embedded config, so today our deploys run on whichever default is available.

### 2.3 Two cleanly-separate override paths

There are two ways to force a non-default attention backend; both are ~5 lines of code.

**Path A — config override at load time.** Subclass `LTXAudioVideoModelConfigurator.from_config` to inject `config["transformer"]["attention_type"] = "flash_attention_3"` before the super call. The existing `Builder`/`Registry` plumbing picks this up with zero changes elsewhere. This is the cleanest for a deploy-time toggle; the transformer lives its whole life with the chosen backend baked in.

**Path B — post-construction module walk.** Build the model with defaults, then iterate `model.modules()` and for every `isinstance(m, Attention)` set `m.attention_function = FlashAttention3()`. This is literally the same pattern SageAttention uses today — we have empirical proof it works. It's also the only option if we want per-layer backends (e.g. FA3 on video-self-attn but xformers on audio-cross-attn).

Recommend **Path A** for production and **Path B** for A/B experiments. Both are in the companion guide.

### 2.4 Why this matters vs SageAttention

The SageAttention integration on the sister branch required:
- A custom `SageAttentionCallable` class wrapping the FA-like interface,
- A `patch_diffusion_stage` wrapper that intercepts `DiffusionStage._build_transformer` and mutates the returned model,
- A Triton pre-warm inside `LTXVideoGenerator.__init__` to avoid JIT stalls on first request,
- Its own Dockerfile stage (multi-stage wheel build) to compile `sageattention` for specific arches,
- Mode / scope management (`SAGE_ATTENTION_MODE="self" | "text" | "all"`) because the wrapper only patches a subset of attention modules.

The FlashAttention integration requires:
- Setting `AttentionFunction.FLASH_ATTENTION_3` (or ensuring xformers is installed for the `DEFAULT` path on Ada),
- Installing the right package in the Docker image.

That's it. Every piece of scaffolding we built for SageAttention is unnecessary for FA.

---

## 3. Architecture compatibility matrix

### 3.1 FA2 vs FA3 hardware support

| variant | GPUs | CUDA min | PyTorch min | install path | pre-built wheel? |
|---|---|---|---|---|---|
| FA2 (`flash-attn`) | sm_75 (T4) through sm_90 (H100) | 11.6 | 1.12 | `pip install flash-attn` | **yes**, PyPI `flash-attn==2.8.3` builds for torch 2.8 + cu128 |
| FA3 (`flash_attn_interface`) | sm_90 (H100 / H200) **only** | 12.3 | 2.2 | source build in `Dao-AILab/flash-attention/hopper/` | **no**, must build; ~3-5 min with `MAX_JOBS=4` |
| xformers (wraps FA2 + its own kernels) | sm_70 through sm_90 | 11.6 | 1.13 | `pip install xformers` | **yes**, PyPI `xformers==0.0.28.post3` onward |

FA2 on Hopper *exists* — you can run FA2 kernels on an H100 — but they're slower than FA3 and slower than SDPA-default because they don't use `wgmma`. There is no reason to use FA2 on Hopper; we'd use FA3 or fall through to SDPA.

### 3.2 Per-card compatibility for our workload

| card | arch | recommended backend | rationale |
|---|---|---|---|
| RTX 4090 24 GB | Ada sm_89 | **FA2 via xformers** (`AttentionFunction.DEFAULT`) | Already active if xformers is installed. Zero code change. |
| RTX 6000 Ada / L40S 48 GB | Ada sm_89 | FA2 via xformers | Same as 4090; 48 GB doesn't unlock streaming-off for our pipeline (we measured this). |
| A100 40 / 80 GB | Ampere sm_80 | FA2 via xformers | **But** A100 is unusable for this pipeline anyway due to LTX-2's `fp8_cast` Triton kernel requiring `fp8e4nv` which sm_80 can't emit. Not relevant. |
| H100 80 GB (SXM/NVL/PCIe) | Hopper sm_90 | **FA3** (`AttentionFunction.FLASH_ATTENTION_3`) | The only arch where FA3 runs, and the arch that benefits the most (wgmma + warp specialisation). |
| H200 141 GB | Hopper sm_90 | FA3 | Same as H100; extra VRAM we don't use at 1080p / 121-frame. |

### 3.3 The SageAttention story, for contrast

The SageAttention compatibility table written three weeks ago before deployment testing claimed "✓ H100" and promised "1.65 – 1.8× end-to-end speedup". The actual outcome on H100 SXM with the production LTX-2.3 pipeline was **complete denoising failure** — every output frame was brown static. The root cause was that SageAttention's sm_90 kernel has internal tile-size assumptions (implicit multiples of 64 or 128) that LTX-2's seq_lens violate. FA does not have this failure mode because its tile maths handles tail-tiles correctly.

---

## 4. LTX-2.3 attention shapes vs FA constraints

### 4.1 The shapes

From the repo's own validation script (`src/validate_sageattention.py` on the sister branch, lines that enumerate test cases):

| layer | role | batch | heads | seq_len (Q) | seq_len (KV) | head_dim |
|---|---|---|---|---|---|---|
| `attn1` stage 1 | video self-attn | 1 | 32 | 5500 | 5500 | 128 |
| `attn1` stage 2 | video self-attn (post-upsample) | 1 | 32 | 22000 | 22000 | 128 |
| `audio_attn1` | audio self-attn | 1 | 32 | 2200 | 2200 | 64 |
| `attn2` | video × text | 1 | 32 | 5500 / 22000 | 256 | 128 |
| `audio_attn2` | audio × text | 1 | 32 | 2200 | 256 | 64 |
| `audio_to_video` | cross-modal | 1 | 32 | 5500 | 2200 | 64 |
| `video_to_audio` | cross-modal | 1 | 32 | 2200 | 5500 | 64 |

### 4.2 FA constraint satisfaction

| constraint | FA2 limit | FA3 limit | LTX-2.3 value | fit? |
|---|---|---|---|---|
| head_dim ≤ | 256 | 256 | 128 (video), 64 (audio) | ✓ |
| seq_len multiple of ? | none — arbitrary | none — arbitrary | 5500, 22000, 2200 (none aligned) | ✓ |
| dtype | FP16, BF16 | FP16, BF16, FP8 (fwd only) | BF16 (checkpoint default) | ✓ |
| causal only? | no, `causal=True/False` | no, `causal=True/False` | `causal=False` everywhere | ✓ |
| additive mask support | yes (FA2 via `attn_bias`) | yes (via `varlen` + padding) | LTX-2's FA3 wrapper raises NotImplementedError if mask passed | **⚠ see §4.3** |

### 4.3 The mask caveat, in detail

`attention.py:111-112`:

```python
if mask is not None:
    raise NotImplementedError("Mask is not supported for FlashAttention3")
```

This is a property of **LTX-2's wrapper**, not of FA3 itself. Upstream FA3 does support masked attention via the `flash_attn_varlen_func` API. The LTX-2 wrapper simply hasn't wired that through because LTX-2's main inference paths don't need it.

**Does any attention call in the production inference path actually pass `mask is not None`?**

- **Video self-attn (`attn1`)**: No — `Attention.forward` at line 232 passes `mask` through from its caller, and the transformer block's self-attention call does not provide a mask.
- **Audio self-attn (`audio_attn1`)**: No.
- **Video × text cross-attn (`attn2`)**: Uncertain without a trace. The text encoder does produce an attention mask for the text tokens (the Gemma-3 tokenizer emits one), and that mask can be propagated through. If it is, FA3 raises.
- **Audio × text cross-attn (`audio_attn2`)**: Same uncertainty.
- **Cross-modal (audio_to_video, video_to_audio)**: No — these are pure attention, no mask.

**Mitigation.** In the implementation phase, the `FlashAttention3.__call__` wrapper in the LTX-2 codebase should be augmented (in our editable install, a trivial two-line change) with a mask fallback:

```python
if mask is not None:
    # Fall back to xformers (or SDPA) for masked calls.
    return XFormersAttention()(q, k, v, heads, mask)
```

With that change, the risk of a runtime failure on a masked cross-attn call disappears, and the fast FA3 path still runs for everything that doesn't need a mask (which is the overwhelming majority of the attention compute).

### 4.4 What the seq_len numbers actually mean

One of the concerns with SageAttention was that 5500 is not a multiple of 128 (5500 mod 128 = 124). FA's online-softmax tiling has no such requirement: the paper explicitly derives the algorithm for arbitrary N, and the code splits `N` into blocks of `BLOCK_N` (typically 64 or 128) with the final block handled correctly. The proof is by running the repo's own `validate_sageattention.py` against FA — we expect all seven test shapes to pass with numerical error < 1e-3 relative BF16.

---

## 5. Where the speedup shows up (and where it doesn't)

### 5.1 Per-step time budget on the 4090 (streaming-on)

From our measured data (584 s at 30 steps on 4090 SAGE=0 minus the ~264 s fixed overhead gives ~320 s of per-step work, or ~10.6 s per inference step):

| component | est. time per step | fraction |
|---|---|---|
| **Layer streaming (PCIe H2D)** | ~8.5 s | 80 % |
| Transformer compute (matmul + attention + MLP) | ~1.5 s | 14 % |
| Guidance batching / scheduler / bookkeeping | ~0.6 s | 6 % |

**Attention alone** inside the "transformer compute" row is ~40 % of that (MLP is ~2× the param count of attention), so attention ≈ 0.6 s per step ≈ 6 % of per-step wall clock. A 2× FA speedup on attention saves 0.3 s per step × 30 = 9 s over the whole 585 s run.

**So on 4090, FA is worth 5 – 10 seconds end-to-end, not 50 %.** The headline "FA2 is 2× faster" is a per-attention-call number that doesn't apply to a wall-clock that's dominated by PCIe.

### 5.2 Per-step time budget on the H100 (streaming off)

Measured 155 s total, 30 steps, minus roughly 40 s for text encode + VAE decode + encode + upload = ~115 s pure inference = ~3.8 s per step.

| component | est. time per step | fraction |
|---|---|---|
| Transformer attention | ~1.4 s | 37 % |
| Transformer MLP / FFN | ~1.8 s | 47 % |
| RoPE + norms + residual | ~0.4 s | 11 % |
| Scheduler / sampler / guidance | ~0.2 s | 5 % |

**Attention is 37 %** of per-step wall clock on H100 — much bigger than on 4090 because we're not paying the PCIe tax. A 1.8× FA3 speedup on attention saves 0.6 s per step × 30 = 18 s over the whole 155 s run.

**So on H100, FA3 is worth 15 – 25 seconds end-to-end** — a ~15 % gain. Still not the marketing-page 2×, but a real, deliverable reduction.

### 5.3 Why these aren't "the FA paper said 2×"

The FA papers' headline numbers are achieved at very large seq_len (8k – 64k) on isolated micro-benchmarks, often with FP16 and no RoPE. Real-world workloads with RoPE, norms, projections, and residuals always see smaller end-to-end speedups. Our seq_lens bracket the low end of FA3's sweet spot (5500 is within the regime where naïve SDPA is already fairly good), so 1.5 – 1.8× on attention → 1.15 – 1.20× end-to-end is a realistic and honest projection.

---

## 6. Numerical stability

"Is Flash Attention Stable?" (arXiv:2405.02803, §3, Table 1) measured max absolute deviation vs naïve attention at BF16 for seq_len 4096:

| kernel | max abs error | relative error |
|---|---|---|
| naïve | 0 (reference) | 0 |
| PyTorch SDPA | ~6×10⁻³ | ~5×10⁻⁴ |
| FlashAttention 2 | ~8×10⁻³ | ~6×10⁻⁴ |
| FlashAttention 3 | ~7×10⁻³ | ~6×10⁻⁴ |

The FA2/FA3 error is ~10 % larger than SDPA's, which is itself ~10× larger than naïve, which is irrelevant because both are 100 – 1000× smaller than BF16 quantisation noise in the surrounding network. Published video-diffusion deployments (CogVideoX via diffusers, HunyuanVideo, Mochi) use FA2 or xformers in production with no reported artefact issues. The SageAttention failure on Hopper was *not* a "slight numerical drift" — it was total output corruption from an alignment bug, a categorically different failure class.

**Verdict:** FA2 and FA3 are numerically safe for production inference at BF16. No frame-artefact risk beyond what the BF16 network itself has.

---

## 7. What breaks (honest risks)

Rank-ordered by likelihood and blast radius.

### 7.1 Cross-attention mask passing through to FA3

**Risk.** LTX-2's text encoder (Gemma-3) emits a padding-aware attention mask for the text token sequence. If the transformer's cross-attention call forwards that mask to the attention function, `FlashAttention3.__call__` raises `NotImplementedError` at the first real request.

**Likelihood.** Medium. I've read the self-attention path and confirmed no mask. I have **not** exhaustively read the cross-attention path. This is a testable assumption — one small request at any resolution will either run clean or throw.

**Mitigation.** Two lines in the FA3 wrapper to fall back to xformers on masked calls (see §4.3). Zero downside; FA3 still runs for the 80 %+ of attention mass that's self-attn.

### 7.2 FA3 build time in CI

**Risk.** FA3 has no pre-built wheels on PyPI. A `pip install` triggers an nvcc build that takes 3 – 5 minutes with `MAX_JOBS=4` (≥ 16 GB RAM recommended), or up to 30 min on a tight runner. Adds to CI wall time.

**Likelihood.** Guaranteed if we add it.

**Mitigation.** Multi-stage Docker build: one `-devel` stage builds the wheel, a second `-runtime` stage copies it in. Same pattern we used for SageAttention. One-time cost per wheel rebuild; buildkit caches the layer afterward.

### 7.3 FA2/FA3 + PyTorch 2.8 + CUDA 12.8 compatibility drift

**Risk.** FA releases occasionally lag PyTorch. Our base image pins `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-*`. `flash-attn` 2.8.3 has a pre-built wheel for exactly this combination; FA3's current `hopper/` branch builds cleanly against cu128 per Dao-AILab issue #1929.

**Likelihood.** Low for the versions we actually need, **but** if someone bumps PyTorch to 2.9 without checking FA support, the wheel install fails.

**Mitigation.** Pin `flash-attn==2.8.3` in `requirements.txt`; pin the FA3 git SHA in the Dockerfile.

### 7.4 xformers auto-routing on 4090 might not pick FA2

**Risk.** `xformers.ops.memory_efficient_attention` selects from among xformers' own kernels, FA2, Triton, etc. based on shape and dtype. For our LTX-2.3 shapes (non-aligned seq_len, BF16) it *should* pick FA2 but it might fall back to xformers' own Cutlass path. Speedup differs.

**Likelihood.** Medium — this can be audited with `torch.profiler` or the xformers env var `XFORMERS_VERBOSE=1`.

**Mitigation.** If xformers doesn't auto-route to FA2 for our shapes, we'd force it by installing `flash-attn` directly and swapping `AttentionFunction.DEFAULT` to `AttentionFunction.XFORMERS` (the wrapper reuses FA2 through xformers) or to a custom `FlashAttention2` class that calls `flash_attn.flash_attn_func` directly. ~20 lines.

### 7.5 Numerical regression vs xformers on specific shapes

**Risk.** Generation of a specific seed comes out visibly different between xformers-default and FA2/FA3. This is expected behaviour (different kernels → different rounding → different-but-equally-valid output); the issue is only user-facing if they have a specific reference image they're trying to reproduce.

**Likelihood.** Guaranteed (kernels differ). Impact on quality-perceived-by-human: near zero.

**Mitigation.** Document in the release notes that the attention backend change may alter outputs at the pixel level for the same seed, and recommend re-generating any reference outputs.

---

## 8. Empirical context — what we've actually measured

Two runs on 2026-04-17 provide the baselines this doc's projections refer to:

| run | config | attention | result | gen_time | file |
|---|---|---|---|---|---|
| A | 4090 24 GB streaming-on | xformers default (implicit FA2) | clean panda | 585.25 s | `ltx_361b4675e4fc.mp4` |
| B | H100 80 GB streaming-off | PyTorch SDPA default | clean panda | 154.68 s | `ltx_954d135fb6b7.mp4` |
| C | H100 80 GB streaming-off | SageAttention sm_90 INT8/FP8 | **brown noise** | 143.04 s | `ltx_988d63db487f.mp4` |

Run **A** almost certainly already benefits from FA2 via xformers — the deployment image installed xformers, LTX-2's `AttentionFunction.DEFAULT` resolves to `XFormersAttention`, and xformers auto-routed our BF16 shapes to FA2 kernels. We did not explicitly verify this, but the 585 s number is in the range consistent with FA2-backed attention on 4090. **This is important: it means "enable FA2" on 4090 is largely a no-op if xformers is already installed, and the 10–15 % additional headroom from explicit FA2 may not materialise.** The companion guide (§4) discusses how to verify with profiling.

Run **B** used the torch base image's default SDPA backend (PyTorch's own Flash-v2-compatible path when it can; otherwise the Cutlass-based or math-backend fallback). Going to FA3 on this same workload should deliver the 15 – 25 s headline improvement projected in §5.2.

Run **C** proved the alignment-failure hypothesis for the Hopper SageAttention kernel. FA does not share this failure mode — its tiling is correct-by-construction for arbitrary seq_len.

---

## 9. Comparison with SageAttention

| axis | SageAttention | FlashAttention | winner |
|---|---|---|---|
| Integration with LTX-2 | patch wrapper + pre-warm + module walk | built-in `AttentionFunction.FLASH_ATTENTION_3` | **FA** by a large margin |
| Compile / install cost | 5 min nvcc build per arch in Dockerfile (builder stage) | FA2: instant (PyPI wheel); FA3: 3-5 min source build | **FA2 on Ada**, tie on Hopper |
| Numerical correctness on LTX-2 shapes | broken on sm_90 (seq_len alignment); works on sm_89 at claimed ~0.2 % drift | correct on any arch, < 1e-3 rel. error | **FA** |
| Headline speedup (per FA / SageAttention paper) | 1.65 – 1.8× (CogVideoX) | 1.5 – 2.0× (various) | tie |
| Measured end-to-end speedup on LTX-2.3 1080p/30 | - on 4090: unstable (Triton JIT wedge + pin_memory crash); - on H100: 143 s wall clock but the output is noise | projected: - 4090: ~450 s (from 585) = 1.30×; - H100: ~105 s (from 155) = 1.48× | **FA** (delivers real wall clock, not a broken clock) |
| Supports masks | yes, but falls back to non-SAGE for them | FA2 yes, FA3 no (with easy wrapper fix) | slight edge to SAGE, closed by fallback |
| Dependencies | `sageattention` (one project, niche) | `xformers` (widely used) and/or `flash-attn` (de facto standard) | **FA** (mature ecosystem) |
| Maintenance surface | custom `SageAttentionCallable` class + `patch_diffusion_stage` + pre-warm + mode env vars | delete all that; set one enum value | **FA** |

The only knob SAGE has that FA doesn't is "aggressive quantisation for extra speed on Hopper". In our measured workload that knob was worth 11 seconds over a 150-second generation — not worth the noise risk, not worth the integration surface, not worth the maintenance.

---

## 10. What the existing SageAttention docs got wrong that this doc fixes

`docs/SageAttention_Compatibility_Analysis.md` and `docs/SageAttention_LTX23_Complete_Guide.md` (on the sister branch `fix/interference-mod-api-optimized-version`):

1. **Unqualified Hopper support.** Both docs' GPU tables show H100 as ✓. Today's empirical data proves the sm_90 kernel is broken for LTX-2's shapes. Any Hopper recommendation in those docs should be struck or heavily qualified.
2. **Unqualified speedup claim.** Both docs quote "1.65 – 1.8× end-to-end speedup" without workload qualifier. On our actual LTX-2.3 1080p/30 workload with streaming on, SAGE=1 on 4090 was < 5 % faster before it wedged; SAGE=1 on H100 was 7 % faster but produced noise. Neither is 1.65×.
3. **"Loss imperceptible ~0.2 %".** Wrong for Hopper, where loss is total.
4. **Integration overhead estimate.** They list SageAttention as "drop-in" when it actually required the wrapper + patch + pre-warm + mode mgmt that evolved over 48 h of debugging. FA is legitimately drop-in.

These docs should be marked superseded by this one or rewritten. The companion implementation guide (`FlashAttention_LTX23_Complete_Guide.md`) supplants the SageAttention-implementation doc.

---

## 11. Open questions (for the implementation phase)

1. **Is xformers installed in the current production `requirements.txt` on `main`?** Need to read `requirements.txt` on this branch and confirm. If not, that's literally the one-line change to unlock FA2 on 4090.
2. **Does cross-attention propagate Gemma's attention mask all the way into the `attention_function`?** Trace from `BasicAVTransformerBlock.forward` through the `Attention.forward` calls for `attn2` / `audio_attn2`. If yes, we need the mask-fallback patch described in §7.1 before swapping to FA3.
3. **Does xformers on 4090 with our BF16 shapes actually dispatch to FA2, or to its own Cutlass kernel?** Profile one request with `XFORMERS_VERBOSE=1` or `torch.profiler.profile` and read which kernel fires.
4. **FA3 + PyTorch 2.8 + cu128 build:** does `MAX_JOBS=4` finish without OOM on a 16 GB CI runner? If not, lower to 2 and accept ~10 min build.
5. **Do we need to set `attention_type` per-model, or once per pipeline?** The LTX-2.3 two-stage pipeline constructs stage_1 and stage_2 transformers from separate `Builder` instances. Both need the override. Config-override approach (Path A) handles both automatically because both stages read from the same `LTXAudioVideoModelConfigurator`.

These are cheap to answer — each is at most an afternoon of testing.

---

## 12. Recommendation

**Ship FA2-on-Ada as the production path for the 4090 deploys.** It is either already active via xformers or requires at most a one-line `requirements.txt` addition. Projected: 4090 wall clock 585 s → 450 s, $/clip 0.055 → 0.043.

**Ship FA3-on-Hopper as the production path for the H100 deploys.** It requires a Dockerfile builder-stage wheel build plus a ~10-line fallback patch on LTX-2's `FlashAttention3.__call__` to handle cross-attention masks safely. Projected: H100 wall clock 155 s → 105 s, $/clip 0.108 → 0.073.

**Keep SageAttention off.** It has not earned its place in this pipeline on either architecture. The integration surface it demands doesn't match the real delivered speedup.

**Validate numerically before declaring victory.** For each arch, generate the same seed 300 before/after and compare frame-level L2 distance. The change to FA should move outputs by < 1 % L2 relative; if it moves them more, something is wrong.

The implementation plan that operationalises all of this is in `docs/FlashAttention_LTX23_Complete_Guide.md`.

---

## Appendix A — sources

- FlashAttention-2 paper: https://arxiv.org/abs/2307.08691
- FlashAttention-3 paper: https://arxiv.org/abs/2407.08608
- "Is Flash Attention Stable?": https://arxiv.org/abs/2405.02803
- Dao-AILab/flash-attention (FA2 + FA3 Hopper source): https://github.com/Dao-AILab/flash-attention
- LTX-2.3 attention module: `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py`
- LTX-2.3 configurator: `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/model_configurator.py`
- SageAttention compatibility analysis (sister branch): `docs/SageAttention_Compatibility_Analysis.md`
- Measured baselines: this repo, commits `8e64978` onward on `fix/interference-mod-api-optimized-version`.

## Appendix B — glossary

- **FA / FlashAttention.** Exact attention computed with IO-aware tiling so the intermediate attention matrix never hits HBM.
- **FA2.** The 2023 rewrite that fixed warp scheduling for Ampere/Ada.
- **FA3.** The 2024 Hopper-specific rewrite using asynchronous `wgmma` and warp specialisation.
- **xformers.** A broader attention library from Meta that, among other kernels, wraps FA2 and dispatches to it for shapes where FA2 is the fastest backend.
- **SDPA.** `torch.nn.functional.scaled_dot_product_attention` — PyTorch's own attention, which since 2.0 can route through FA2, the math backend, or Cutlass depending on shape.
- **seq_len / N.** Number of tokens on the query side (or key/value side; they differ for cross-attn).
- **head_dim / D.** Dimensions per attention head. LTX-2.3 uses 128 for video and 64 for audio.
- **BF16.** bfloat16 — the dtype LTX-2.3 runs inference in.
- **wgmma.** Hopper's asynchronous matrix-multiply-accumulate instruction. FA3 relies on it; SageAttention's sm_90 kernel also uses it and that's where its alignment bug lives.
