# LTX-2.3 FP8 on H100 — Integration Spec

## Context

This branch (`FA3TCache-FP8`) is forked from `feat/fa3-teacache` to integrate the **pre-quantized** `Lightricks/LTX-2.3-fp8` checkpoint and run it through `QuantizationPolicy.fp8_scaled_mm()` (W8A8 — real FP8 GEMM via TRT-LLM `cublas_scaled_mm`) on H100.

The production branch (`feat/fa3-teacache`) currently:
- Downloads the 46 GB BF16 `Lightricks/LTX-2.3` checkpoint.
- Runs `QuantizationPolicy.fp8_cast()` at load time (W8A16 — weights FP8, activations upcast to BF16).
- Uses FA3 + TeaCache for attention speedup.

This approach works on any FP8-capable GPU (Ada / Hopper / Blackwell) but wastes download bandwidth and does not use H100's native FP8 compute. On H100 we can do better: ~30 GB download (vs 46 GB), lower VRAM, and measurable step-time speedup on the matmul path — while keeping FA3 + TeaCache unchanged.

## Two available FP8 paths (keep both; route by GPU)

| Path | Matmul | Attention | Checkpoint | GPU | Extra | Notes |
|------|--------|-----------|------------|-----|-------|-------|
| `fp8_cast` (current prod) | BF16 (weights upcast per-fwd) | FA3 in BF16 | BF16 **or** FP8 safetensors | Ada / Hopper / Blackwell | `ltx-core[xformers]` OK | Memory savings only; no compute speedup |
| `fp8_scaled_mm` (**this branch**) | **FP8 e4m3fn** via TRT-LLM `cublas_scaled_mm` | FA3 in BF16 | **Pre-quantized FP8 only** | **H100 / H200** (SM 9.0) | `ltx-core[fp8-trtllm]` (mutex with `xformers`) | W8A8; per-tensor amax scales embedded in checkpoint |

Decision: on this branch, **default to `fp8_scaled_mm` when `torch.cuda.get_device_capability() == (9, 0)`**, and fall back to `fp8_cast` otherwise so the same image still boots on 4090 / 5090.

## HuggingFace artifacts

Source: `Lightricks/LTX-2.3-fp8` — **weights-only** repo (no `config.json`, no tokenizer). The FP8 checkpoints contain the 22B **DiT only** — VAE encoder/decoder, audio decoder, vocoder, image encoder, and the embeddings processor all live inside the BF16 `Lightricks/LTX-2.3` checkpoint and are extracted via `*_COMFY_KEYS_FILTER` state-dict ops by the non-DiT pipeline blocks.

| File | Size | Role |
|------|------|------|
| `Lightricks/LTX-2.3/ltx-2.3-22b-dev.safetensors` | 46 GB | **Still required on every path.** Sole source of VAE / audio / image-encoder / embeddings-processor weights consumed by `ImageConditioner`, `VideoDecoder`, `AudioDecoder`, `PromptEncoder`, `VideoUpsampler`. On the `cast` path this also backs the two DiffusionStages. |
| `Lightricks/LTX-2.3-fp8/ltx-2.3-22b-dev-fp8.safetensors` | 29.1 GB | Stage 1 DiT on `scaled_mm`. |
| `Lightricks/LTX-2.3-fp8/ltx-2.3-22b-distilled-fp8.safetensors` | 29.5 GB | Stage 2 DiT on `scaled_mm` (distilled weights pre-fused). |
| `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | ~1 GB | Unchanged. |
| Gemma 3 12B text encoder | ~26 GB | Unchanged. |

Net disk on the `scaled_mm` path: ~104 GB (46 + 29 + 29 + 1 + Gemma). Download bandwidth goes **up**, not down, because the BF16 file cannot be dropped. The win is runtime compute (native H100 FP8 GEMM) and ~30 GB lower peak DiT VRAM during inference.

**LoRA caveat.** `fp8_scaled_mm` cannot fuse a BF16 LoRA into pre-quantized FP8 weights at load time. On the H100 path, Stage 2 uses the distilled-fp8 checkpoint (distilled weights already merged) and receives `distilled_lora=()`.

## Required kernels / libs

- `torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor` — activation quant (TRT-LLM op)
- `torch.ops.trtllm.cublas_scaled_mm` — FP8 GEMM, fp32 scales in, BF16 out (TRT-LLM op)
- `flash_attn_interface.flash_attn_func` (FA3) — BF16 attention on H100 (already active on base branch)
- **Not** used on this path: xformers, TransformerEngine, torchao float8, custom Triton kernels
- The Triton stochastic-rounding kernel in `LTX-2-ref/packages/ltx-core/src/ltx_core/quantization/fp8_cast.py` is specific to the `fp8_cast` branch and is irrelevant here.

## Dependency pins

From `LTX-2-ref/packages/ltx-core/pyproject.toml`:

```
torch ~= 2.7                       (wheel index: pytorch.org/whl/cu129  → CUDA 12.9)
transformers >= 4.52
safetensors, accelerate, einops, scipy >= 1.14
tensorrt-llm == 1.0.0              [fp8-trtllm extra]
onnx >= 1.16.0, < 1.20.0           [fp8-trtllm extra]
openmpi                            [fp8-trtllm extra]
flash-attn (with FA3 interface)    — installed separately; not pinned in pyproject
```

Runtime: CUDA ≥ 12.8 (12.9 recommended), H100 SM 9.0, driver ≥ 550, Python ≥ 3.10 (ruff target: 3.11).

Note: `ltx-core[xformers]` and `ltx-core[fp8-trtllm]` are declared as **conflicting extras** in upstream `pyproject.toml`. Regenerate `uv.lock` on this branch with `fp8-trtllm` selected.

## Implemented changes (deploy wrapper only; no patch to `LTX-2-ref`)

### 1. `src/download_models.py`
- Added a helper `_fp8_mode()` that reads `LTX_FP8_MODE` (`cast` | `scaled_mm`, default `cast`). The env flag is used at download time because the GPU isn't queryable during pod-image warm-up.
- BF16 dev checkpoint and spatial upscaler are always downloaded.
- `scaled_mm` additionally downloads `ltx-2.3-22b-dev-fp8.safetensors` and `ltx-2.3-22b-distilled-fp8.safetensors` from `Lightricks/LTX-2.3-fp8`, and skips the 7.6 GB distilled LoRA (the pre-fused distilled-fp8 checkpoint supersedes it).
- `cast` keeps the existing BF16 + distilled-LoRA download set unchanged.

### 2. `src/pipeline.py`
- New `_select_fp8_mode()` picks the path in this order: `LTX_FP8_MODE` env override → `torch.cuda.get_device_capability() == (9, 0)` → `cast` fallback.
- `scaled_mm` uses `QuantizationPolicy.fp8_scaled_mm()` and passes an empty `distilled_lora`; the pipeline is still constructed against the BF16 `checkpoint_path` so the non-DiT blocks find their VAE / encoder / vocoder / embeddings-processor weights.
- Immediately after construction, both `self._pipeline.stage_1` and `self._pipeline.stage_2` are rebuilt as fresh `DiffusionStage` instances pointing at the FP8 DiT files — Stage 1 at `ltx-2.3-22b-dev-fp8.safetensors`, Stage 2 at `ltx-2.3-22b-distilled-fp8.safetensors`. `DiffusionStage` defers weight load to its first `__call__`, so the override is safe.
- The existing Stage 1 → Stage 2 cleanup hook is installed **after** the stage rebuild so it wraps the correct `_transformer_ctx`.
- On `cast`, behaviour is byte-identical to the production branch.

### 3. `Dockerfile` / `setup.sh`
- Installs `-e "/app/LTX-2/packages/ltx-core[fp8-trtllm]"` with `--extra-index-url https://pypi.nvidia.com`, which pulls `tensorrt-llm==1.0.0`, `onnx`, and `openmpi`. This replaces the no-extra install. The `[fp8-trtllm]` extra is mutex with `[xformers]` in upstream `pyproject.toml`; since LTX-2-ref's attention dispatcher falls back to torch SDPA when xformers is absent and FA3 is installed directly from the windreamer wheel index, dropping xformers is safe.
- Dockerfile adds `libopenmpi-dev` to apt so the openmpi wheel can dlopen its shared libs; sets `ENV LTX_FP8_MODE=scaled_mm` so `download_models.py` pulls the FP8 variants by default on this branch.
- `setup.sh` mirrors the install command for non-Docker deploys.

### 4. `src/attention_override.py` (unchanged)
- FA3 runs in BF16 regardless of matmul path.
- The attention dispatcher in `LTX-2-ref/packages/ltx-core/src/ltx_core/model/transformer/attention.py` picks xformers only when explicitly requested; with xformers absent the path resolves to FA3 or SDPA. Verified with the existing boot-time fingerprint log in `LTXVideoGenerator.__init__`. TeaCache caches BF16 activations and is unaffected by the W8A8 matmul swap.

## Critical upstream files to reference (read-only)

- `LTX-2-ref/packages/ltx-core/src/ltx_core/quantization/policy.py` — `QuantizationPolicy.fp8_scaled_mm` factory.
- `LTX-2-ref/packages/ltx-core/src/ltx_core/quantization/fp8_scaled_mm.py` — `FP8Linear` (lines 11–73), `_apply_fp8_prepare_to_model` (145–168), `EXCLUDED_LAYER_SUBSTRINGS` (111–126). Important: the first block (`transformer_blocks.0.`) and the last five (`43–47`) stay BF16; so do `adaln_single`, `caption_projection`, `proj_out`, `patchify_proj` and their audio twins.
- `LTX-2-ref/packages/ltx-core/src/ltx_core/loader/single_gpu_model_builder.py` — threads `sd_ops` + `module_ops` from the policy into model load.
- `LTX-2-ref/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_two_stages.py` — the call site that needs a Stage-2 checkpoint override.
- `LTX-2-ref/packages/ltx-pipelines/src/ltx_pipelines/utils/args.py:262` — confirms upstream supports `--quantization {fp8-cast,fp8-scaled-mm}` at CLI level.

## Verification

1. **Download check** — `python -m src.download_models` fetches both fp8 safetensors; confirm sizes match HF.
2. **Load check** — instantiate the model with `QuantizationPolicy.fp8_scaled_mm()`; assert that a middle transformer block (e.g. `transformer_blocks[10]`) contains `FP8Linear` instances, and that `transformer_blocks[0]` / `transformer_blocks[43..47]` remain `nn.Linear`.
3. **Numeric sanity** — generate a 5 s 768×512 video with a fixed seed on both `fp8_cast` and `fp8_scaled_mm`; PSNR between outputs > 30 dB is the acceptance bar.
4. **Perf** — measure step time on H100 for a 10 s generation; target ≥ 1.3× speedup vs. the `fp8_cast` baseline (torchao float8 reports 1.27–1.54× on comparable DiTs).
5. **FA3 confirmation** — log which attention backend resolved in Stage 1 and Stage 2; both must be `FLASH_ATTENTION_3`.
6. **VRAM** — expect ~30 GB weight footprint plus activations; fits H100 80 GB comfortably with TeaCache enabled and layer-streaming disabled.

## Open questions

- **Stage-2 checkpoint plumbing** — upstream `TI2VidTwoStagesPipeline` does not yet expose a separate Stage 2 checkpoint path; this deploy handles it locally by replacing `pipeline.stage_1` / `pipeline.stage_2` after construction. If upstream adds the parameter, migrate to it.
- **FP8 FA3** — out of scope for v1; attention stays BF16. Revisit once Dao-AILab `flash-attention#1848` (silent BF16 fallback in the FP8 FA3 path) is closed.
- **uv.lock** — the `xformers` ↔ `tensorrt-llm` mutex means the lockfile differs per deploy target. Maintain `uv.lock` on this branch pinned to `fp8-trtllm`; keep the base-branch `uv.lock` pinned to `xformers`.
