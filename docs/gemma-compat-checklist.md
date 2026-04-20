# Gemma 3 compatibility + VRAM checklist

Working branch: `feat/gemma-compat` off `fix/inference-mode-api`.
Goal: shrink Gemma-3-12B from ~26 GB BF16 down to a footprint that lets
the whole pipeline run in **pure-GPU mode on a 48 GB L40S / RTX 6000
Ada / A6000**, without producing black output.

Target footprint: `Gemma ≤13 GB + DiT 22 GB + VAE 2 GB + activations ~10 GB = ≤47 GB`.

---

## What's been tried

### On-disk checkpoint swaps (replace `gemma_root` directory)

- [x] ~~**RedHatAI GPTQ W4A16** (`RedHatAI/gemma-3-12b-it-quantized.w4a16`)~~ — ❌ **Boot OK, black video.** Tokenizer plumbing fixed via `35beb13` (pulls `tokenizer.model` from Google repo), full generation completes in 162 s on Blackwell B6000, but the resulting MP4 is ~60 KB for a 5 s 1024×1536 clip — i.e. flat black. **Root cause:** LTX-2's `PromptEncoder` uses a custom safetensors loader (`ltx_pipelines/utils/blocks.py:329` → `ltx_core/text_encoders/gemma/encoders/base_encoder.py`) that bypasses `transformers.AutoModel`, so `compressed-tensors` decoding of the W4A16 packed weights never runs. The DiT receives zero-ish conditioning → outputs near-zero latents → VAE decodes flat color. **Verdict:** any checkpoint with a `quantization_config` block will fail the same way until the loader path is fixed.

### Runtime behavior

- [x] **Pure-GPU mode plumbing** (commit `d17c132`) — ✅ **Working.** On GPUs ≥40 GB VRAM, `StateDictRegistry` and per-layer streaming are bypassed. Confirmed on 97 GB Blackwell: boot log shows `Pure-GPU mode ENABLED`. No CPU↔GPU swapping observed during generation.
- [x] **BF16 default restored** (commit `bb28188`) — ✅ **Working.** `GEMMA_QUANT=bf16` fetches Google's QAT-dequantized copy and loads normally through LTX-2. Known-good output. Fits on 80 GB only (26 GB Gemma + 22 GB DiT + activations > 48 GB).

---

## What's left to try (ranked by priority)

### Post-load quantization (RECOMMENDED — sidesteps the loader problem)

Load Gemma as BF16 through LTX-2's normal path, then replace `nn.Linear`
modules in-place with quantized equivalents. On-disk stays BF16 (loader
happy); runtime memory drops.

- [~] **HQQ 4-bit post-load** — ✅ **plumbed** (commit `69a75ba`), `GEMMA_POST_LOAD_QUANT=hqq4`. Live end-to-end test pending.
- [~] **HQQ 8-bit post-load** — ✅ **plumbed** (commit `69a75ba`), `GEMMA_POST_LOAD_QUANT=hqq8`. Backup if hqq4 drifts.
- [ ] **BitsAndBytes NF4 post-load** — dependency `bitsandbytes`, ~7 GB runtime, `compute_dtype=bfloat16` preserves activations. Backup if HQQ has issues.
- [ ] **BitsAndBytes 8-bit (LLM.int8) post-load** — ~13 GB runtime. Conservative floor — essentially always works, effectively lossless. Keep as safety net.
- [ ] **TorchAO `Int8WeightOnlyConfig`** via `torchao.quantization.quantize_(gemma, config)` — ~13 GB runtime, modern PyTorch-native API.
- [ ] **TorchAO `Int4WeightOnlyConfig`** — ~7 GB runtime. Raw INT4 without calibration; expected quality drop vs HQQ/NF4.

### Off-GPU strategies (not strictly pure-GPU but very cheap wins)

- [ ] **Gemma post-encode eviction to CPU** — Gemma runs once per job; evict after encode, diffusion hot path is pure GPU.
- [ ] **Negative-prompt embedding precompute** — `DEFAULT_NEGATIVE_PROMPT` is constant; encode once at boot. Saves ~50% of Gemma time every job.
- [ ] **Service-layer prompt embedding cache** — LRU + safetensors on disk. Near-100% hit rate on seed/aspect sweeps.

### Loader-level fixes (enables on-disk quant paths)

- [ ] **Monkey-patch `PromptEncoder.__init__`** to route through `transformers.AutoModel.from_pretrained(gemma_root)`. Lets on-disk W4A16 / FP8 / NF4 checkpoints load correctly. Higher complexity; only worth it if post-load quant isn't enough.
- [ ] **Upstream fork of LTX-2's loader** — clean fix, merge-worthy PR. Highest effort.

### On-disk checkpoints that *might* work with BnB integration

These ship with `bitsandbytes` quantization metadata that some loaders respect even without `AutoModel`. Low-confidence.

- [ ] **Unsloth BnB NF4 (QAT-warm-start)** (`unsloth/gemma-3-12b-it-qat-int4-unsloth-bnb-4bit`) — only worth testing if post-load NF4 fails. Same loader risk as W4A16.
- [ ] **Unsloth BnB 8-bit** — same caveat.

### Stretch / probably not useful

- [ ] **RedHatAI FP8-dynamic** (`RedHatAI/gemma-3-12b-it-FP8-dynamic`) — same compressed-tensors format as the W4A16 we just ruled out. Will fail identically. Skip unless loader is fixed.
- [ ] **Pre-dequantize W4A16 → BF16 on disk** — no VRAM win (still 26 GB BF16 at runtime). Only useful as a quality-control A/B to verify GPTQ's dequantized weights match Google's directly.
- [ ] **Google GGUF Q4_0 → PyTorch conversion** — requires a GGUF loader. Large effort for weights that effectively match the QAT-dequantized BF16 we already have. Skip.
- [ ] **DiT post-load quantization (FP8 → INT4)** — LTX-2's `QuantizationPolicy.fp8_cast()` is the shipped FP8 path; further quant on the DiT risks visible video quality loss and is high-effort. Not a Gemma problem, park it.
- [ ] **Replace Gemma with a smaller encoder** (T5-XXL, CLIP-L) — requires retraining LTX-2. Blocked.

---

## Next concrete change (HQQ post-load path)

### Files to modify

| File | Change |
|---|---|
| `pyproject.toml` | Add `hqq>=0.2.0` to `dependencies` (drop `compressed-tensors` — unused with post-load path). |
| `src/pipeline.py` | After `self._pipeline = TI2VidTwoStagesPipeline(**pipeline_kwargs)`, locate the Gemma module handle (likely `self._pipeline.prompt_encoder.gemma` or similar — needs one-time probe), walk its submodules, replace every `nn.Linear` with `hqq.core.quantize.HQQLinear(..., quant_config=BaseQuantizeConfig(nbits=4, group_size=64))`. Run pipeline as normal. |
| `src/download_models.py` | Revert the `_GEMMA_REPOS` dict to only `bf16` — post-load quant works on the BF16 checkpoint. Keep the `tokenizer.model` fallback (harmless, always present on BF16 anyway). |
| `.env.example` | Replace `GEMMA_QUANT` doc with `GEMMA_POST_LOAD_QUANT={none,hqq4,nf4,int8}`. |

### Order of commits

1. **Add `hqq` dependency** + remove `compressed-tensors` from `pyproject.toml`.
2. **Add post-load quant dispatcher** in `src/pipeline.py` — env-flagged, default `none` so BF16 remains the safe default.
3. **Probe + record the correct Gemma module handle** in `src/pipeline.py` (may need a one-shot debug commit that logs `type(self._pipeline.prompt_encoder)` and its `__dict__` keys, so we know exactly what attribute to walk).
4. **Wire the Linear-replacement walker** — apply HQQLinear after pipeline construction.
5. **Add a `max_abs` / `mean_abs` hidden-state log** at the first Gemma forward so we can detect zero-output silent failures (never again produce a 60 KB black MP4 without knowing why).
6. **Live test on 48 GB L40S / Blackwell.** Confirm video is not black, measure VRAM peak, validate runtime.

### Sanity-check gates (must pass before marking HQQ as working)

- [ ] Boot log shows `Gemma post-load HQQ4 quant applied: <N> Linear layers replaced`.
- [ ] First-encode hidden state has `max_abs > 0.01` and `mean_abs > 0.001` (not zero-valued tensors).
- [ ] Output MP4 for a 5 s 1024×1536 generation is **>500 KB** (not flat-color). Below 200 KB → abort and inspect.
- [ ] VRAM peak during Stage 1 ≤ 45 GB on a 48 GB card.
- [ ] Warm generation time within 2× of BF16 baseline on the same GPU.

---

## Memory budget reference

| Config | Gemma | DiT | VAE+up | Activations | **Total** | Fits 48 GB? |
|---|---|---|---|---|---|---|
| BF16 Gemma + FP8 DiT (current default) | 26 | 22 | 2 | 10 | **60** | No |
| **HQQ4 / NF4 Gemma + FP8 DiT** | **7** | 22 | 2 | 10 | **41** | **Yes, comfortable** |
| INT8 Gemma + FP8 DiT | 13 | 22 | 2 | 10 | **47** | Yes, tight |
| BF16 Gemma evicted post-encode + FP8 DiT | 0 | 22 | 2 | 10 | **34** | Yes but not "pure GPU" |

---

## Cross-references

- W4A16 failure evidence: commit `bb28188` message, this doc's first row.
- Pure-GPU mode code: `src/pipeline.py:40-70` (`_pure_gpu_mode` flag), `src/pipeline.py:97-106` (registry skip), `src/pipeline.py:219-232` (streaming skip).
- LTX-2 loader path to fix (if we go G1): `/app/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:329` → `PromptEncoder.__init__` → `base_encoder.py` `module_ops_from_gemma_root`.
- Full strategy survey: conversation history 2026-04-20, "all the possible ways for a pure gpu pipeline that could run on 48gb ram".

---

## Optimizations ported from `feat/fa3-teacache`

Imported in one pass after HQQ4 landed. All new features are orthogonal to HQQ — they target the DiT transformer, not Gemma.

- [x] **TeaCache** — `src/teacache.py` ported verbatim. Gated on `ENABLE_TEACACHE=1`, threshold 0.03 lossless / 0.05 aggressive. Patches `DiffusionStage._transformer_ctx` so the streaming + batch-split machinery is untouched.
- [x] **Stage 1→2 cleanup hook** — `_install_stage2_cleanup_hook`. Always on; flushes allocators at the stage boundary. Harmless on 48 GB+ cards, essential on 24 GB.
- [x] **torch.compile (regional)** — `ENABLE_TORCH_COMPILE=1` default on. Per-block compile with inductor; 15–30 % Stage-1 speedup on Ada + Hopper.
- [x] **Attention fingerprint log** — `_log_attention_fingerprint` boots a one-shot diagnostic reporting `xformers` / `flash_attn_interface` availability and which callable LTX-2 resolved to.
- [x] **Gemma `use_fast=True` sed** in Dockerfile — shaves 1–2 s off text encoding by forcing the Rust tokenizer.
- [x] **xformers install** — LTX-2's `AttentionFunction.DEFAULT` auto-picks `XFormersAttention` when xformers is importable. Works on Ada sm_89 (no monkey-patching needed). Replaces the FA3 override which was Hopper-only.

### Deliberately NOT ported (Hopper-only)

- [ ] ~~FA3 (`flash_attn_interface`)~~ — Hopper sm_90 only, doesn't run on L40S. Skipped.
- [ ] ~~`src/attention_override.py` FA3 monkey-patch~~ — dependent on the wheel above.
- [ ] ~~windreamer wheel install in Dockerfile~~ — same reason.

### Deferred follow-ups

- [ ] **SageAttention 2** — INT8/FP8 attention kernels for Ada, ~2–3× over FA2. Needs a monkey-patch similar to the old FA3 one. Highest-ceiling optimization remaining on this branch.
