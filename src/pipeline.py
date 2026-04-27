"""Shared LTX-2.3 pipeline wrapper.

All VRAM optimisation techniques live here.
"""

import functools
import gc
import logging
import os
import re
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, oversaturated"
)


def _round_to(value: int, divisor: int) -> int:
    return (value // divisor) * divisor


def _round_frames(n: int) -> int:
    return ((n - 1) // 8) * 8 + 1


def _round_user_inputs(width: int, height: int, num_frames: int) -> tuple[int, int, int]:
    """Round user-supplied dims to pipeline grid and log when we change anything.

    LTX-2.3 requires width/height divisible by 64 (latent stride × patch size)
    and frame counts of the form 8k+1. Silently clamping surprises callers when
    a request that "worked" locally comes back at a different resolution; the
    log line makes the adjustment discoverable from pod output.
    """
    w, h, f = _round_to(width, 64), _round_to(height, 64), _round_frames(num_frames)
    if (w, h, f) != (width, height, num_frames):
        logger.info(
            "Input rounded to pipeline grid: %dx%d×%d → %dx%d×%d",
            width, height, num_frames, w, h, f,
        )
    return w, h, f


def _bmgjet_stage_2_sigmas(num_frames: int, prompt: str) -> torch.Tensor:
    """Frame-count-aware Stage-2 sigma schedule.

    Ported from bmgjet/ComfyUi-LTX23Sigmas. Mitigates late-frame chroma
    drift on long clips that the upstream STAGE_2_DISTILLED_SIGMAS
    constant (3-step schedule, tuned for 121f/5s) does not handle.
    See HF Lightricks/LTX-2.3 discussion #13.
    """
    def _interp(v: int, points: list[tuple[int, float]]) -> float:
        prev = points[0]
        if v <= prev[0]:
            return prev[1]
        for p in points[1:]:
            if v <= p[0]:
                t = (v - prev[0]) / (p[0] - prev[0])
                return prev[1] + (p[1] - prev[1]) * t
            prev = p
        return points[-1][1]

    s1 = _interp(num_frames, [(121, 0.84), (241, 0.85), (361, 0.90), (481, 0.95), (601, 0.98)])
    s2 = _interp(num_frames, [(121, 0.78), (241, 0.78), (361, 0.85), (481, 0.89), (601, 0.91), (841, 0.92)])
    s3 = _interp(num_frames, [(121, 0.735), (961, 0.735), (1081, 0.740)])
    s4 = _interp(num_frames, [(601, 0.96), (841, 0.97), (961, 0.98), (10000, 0.98)])

    word_count = max(1, len(prompt.split()))
    offset = min(0.0075, max(0.0, (num_frames / word_count) - 10) * 0.0002)
    # Threshold deviates from upstream bmgjet (`<= 241`). Empirical
    # test on 241f / 1920x1088 (pod or0nihrlpzu1nn, branch
    # feat/bmgjet-stage2-sigmas commit 683beb8) showed the 3-step
    # short regime made tail SAT *worse* (29.81 vs 27.70 baseline).
    # Pushing 241 into the 4-step medium regime gives Stage 2 one
    # extra denoising pass at sigma 0.78, which is the direction the
    # data points to.
    regime = (num_frames < 241) * 1 + (num_frames >= 601) * 2
    schedules = [
        [max(s1, min(1.0, s1 + offset)), max(s2, min(1.0, s2 + offset * 0.5)), s3, 0.445, 0.0],
        [0.85, 0.725, 0.4219, 0.0],
        [s4, 0.935, 0.9, 0.725, 0.445, 0.0],
    ]
    return torch.tensor(schedules[regime], dtype=torch.float32)


def _install_stage2_cleanup_hook(pipeline) -> None:
    """Flush device + host allocators at the Stage 1 → Stage 2 boundary.

    LTX-2's weight-offloading path (``OffloadMode.CPU`` / ``DISK`` via
    ``block_streaming`` — formerly ``layer_streaming``; renamed in PR
    #201, 2026-04-23) pins transformer blocks in host memory during
    inference. PR #201 also *removed* the best-effort
    ``torch._C._host_emptyCache()`` that used to run in the streaming
    teardown, so pinned pages are no longer returned to the OS between
    stages. On <40 GB pods that intermittently exhausts the pinned arena
    and the next ``tensor.data.pin_memory()`` call (inside the Stage 2
    pool builder) raises

        torch.AcceleratorError: CUDA error: invalid argument

    We force a synchronous cleanup cycle (Python GC → device
    empty_cache → CUDA sync → host empty_cache) right before Stage 2's
    transformer context manager enters, so the pinned arena is in a
    known-drained state. On H100 (``OffloadMode.NONE``) the streaming
    path is inactive and the hook is a cheap sync; the actual Stage 1
    GPU weight release happens inside upstream ``gpu_model.__exit__``
    (which calls ``model.to("meta")``), and only takes effect because
    the DiT stages are built without a ``StateDictRegistry`` — see the
    ``registry=None`` rebuilds in ``LTXVideoGenerator.__init__``. The
    hook's ``empty_cache`` returns the freed CUDA blocks to the OS so
    Stage 2's build sees a near-empty pool.
    Idempotent; safe to call multiple times.
    """
    stage = getattr(pipeline, "stage_2", None)
    if stage is None:
        logger.warning("Stage 2 cleanup hook: pipeline has no stage_2")
        return
    if getattr(stage, "_stage2_cleanup_hooked", False):
        return

    original_ctx = stage._transformer_ctx

    @functools.wraps(original_ctx)
    def hooked_ctx(*args, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            alloc_before = torch.cuda.memory_allocated(0) / 1e9
            res_before = torch.cuda.memory_reserved(0) / 1e9
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if hasattr(torch._C, "_host_emptyCache"):
                try:
                    torch._C._host_emptyCache()
                except Exception:  # pragma: no cover — best effort
                    logger.debug("Stage 2 cleanup: _host_emptyCache raised", exc_info=True)
            alloc_after = torch.cuda.memory_allocated(0) / 1e9
            res_after = torch.cuda.memory_reserved(0) / 1e9
            alloc_delta = alloc_before - alloc_after
            res_delta = res_before - res_after
            alloc_note = (
                "" if alloc_delta > 0.01
                else " (unchanged; stage 1 already released via gpu_model.__exit__)"
            )
            logger.info(
                "Stage 2 boundary cleanup: alloc %.2f → %.2f GB%s, "
                "reserved %.2f → %.2f GB (%.2f GB returned to pool)",
                alloc_before, alloc_after, alloc_note,
                res_before, res_after, res_delta,
            )
        else:
            logger.info("Stage 2 boundary cleanup done (no CUDA)")
        return original_ctx(*args, **kwargs)

    stage._transformer_ctx = hooked_ctx
    stage._stage2_cleanup_hooked = True


def _install_build_transformer_audit(stage, label: str) -> None:
    """Wrap ``DiffusionStage._build_transformer`` with timing + dtype audit.

    The transformer build is what dominates cold-start: disk read of the
    FP8 safetensors → SDOps transpose → ``load_state_dict(..., assign=True)``
    → the FP8 ``_prepare`` swap. We have no instrumentation for that
    wall time today; the cold-start gap (job submit → first sampler step)
    is a 7-min black box. This wrapper logs:

    1. Wall time for the whole build (single number — easy to compare
       cold vs warm).
    2. Post-load dtype histogram over `transformer_blocks.*` parameters.
       This is the GROUND TRUTH for "did the FP8 swap actually take?".
       The per-block log inside `_prepare` only reflects what the
       matcher decided to swap; if the matcher picked the wrong module
       names, the actual weight buffers stay BF16 — and only this
       post-load histogram surfaces the discrepancy.
    3. First few BF16 names in blocks ≥ 2 — sanity check that BF16
       isn't bleeding into the body of the network.
    4. Blocks fully BF16 (every weight in the block is BF16) — should
       be {0, 1, 46, 47} or similar per upstream's published design.

    Idempotent; safe to call once per stage.
    """
    if getattr(stage, "_build_transformer_audited", False):
        return
    if not hasattr(stage, "_build_transformer"):
        logger.warning(
            "Audit hook [%s]: stage has no _build_transformer; skipping",
            label,
        )
        return

    original_build = stage._build_transformer

    @functools.wraps(original_build)
    def audited_build(*args, **kwargs):
        logger.info("[%s] _build_transformer start", label)
        if torch.cuda.is_available():
            alloc_before = torch.cuda.memory_allocated(0) / 1e9
            res_before = torch.cuda.memory_reserved(0) / 1e9
            prior_process_peak = torch.cuda.max_memory_allocated(0) / 1e9
            torch.cuda.reset_peak_memory_stats(0)
        else:
            alloc_before = res_before = prior_process_peak = 0.0
        t_start = time.perf_counter()
        model = original_build(*args, **kwargs)
        elapsed = time.perf_counter() - t_start
        if torch.cuda.is_available():
            alloc_after = torch.cuda.memory_allocated(0) / 1e9
            res_after = torch.cuda.memory_reserved(0) / 1e9
            this_stage_peak = torch.cuda.max_memory_allocated(0) / 1e9
        else:
            alloc_after = res_after = this_stage_peak = 0.0
        logger.info(
            "[%s] _build_transformer done in %.1fs (load+swap+assign). "
            "Live alloc %.2f → %.2f GB, reserved %.2f → %.2f GB, "
            "this-stage peak %.2f GB (prior process peak %.2f GB).",
            label, elapsed,
            alloc_before, alloc_after, res_before, res_after,
            this_stage_peak, prior_process_peak,
        )

        try:
            from collections import Counter
            dtype_counts: Counter = Counter()
            block_dtype_sets: dict[int, set[str]] = {}
            bf16_in_body: list[str] = []
            # `_build_transformer` returns `X0Model(builder.build(...))`,
            # which wraps the inner DiT under an attribute prefix. So
            # `named_parameters()` yields names like
            # `inner.transformer_blocks.<n>....` — search the regex
            # anywhere in the name rather than anchoring at the start.
            block_re = re.compile(r"transformer_blocks\.(\d+)\.")
            for pname, p in model.named_parameters():
                m = block_re.search(pname)
                if m is None:
                    continue
                dt_name = str(p.dtype).removeprefix("torch.")
                dtype_counts[dt_name] += 1
                bi = int(m.group(1))
                block_dtype_sets.setdefault(bi, set()).add(dt_name)
                if (
                    dt_name == "bfloat16"
                    and bi >= 2
                    and len(bf16_in_body) < 5
                    and pname.endswith(".weight")
                ):
                    bf16_in_body.append(pname)

            total = sum(dtype_counts.values())
            fp8_n = dtype_counts.get("float8_e4m3fn", 0)
            logger.info(
                "FP8 audit [%s]: %d FP8 weight tensors loaded "
                "(should match the prepare-step swapped count above). "
                "Param-level dtype mix follows; denominator includes "
                "non-Linear params (norms, biases, FP8 scales) so any "
                "percentage from it is NOT the module-level swap rate.",
                label, fp8_n,
            )
            logger.info(
                "FP8 audit [%s]: param-level dtype histogram "
                "(transformer_blocks.*, total=%d): %s",
                label, total, dict(dtype_counts),
            )
            if bf16_in_body:
                logger.info(
                    "FP8 audit [%s]: first %d BF16 weight names in blocks ≥ 2: %s",
                    label, len(bf16_in_body), bf16_in_body,
                )
            fully_bf16_blocks = sorted(
                bi for bi, ds in block_dtype_sets.items()
                if "float8_e4m3fn" not in ds
            )
            if fully_bf16_blocks:
                logger.info(
                    "FP8 audit [%s]: blocks with NO FP8 params: %s",
                    label, fully_bf16_blocks,
                )
            else:
                logger.info(
                    "FP8 audit [%s]: every block has at least one FP8 param",
                    label,
                )
        except Exception:
            logger.exception("FP8 audit [%s]: dtype histogram failed", label)

        return model

    stage._build_transformer = audited_build
    stage._build_transformer_audited = True


# The Lightricks FP8 safetensors store DiT weights with this prefix on
# disk; `LTXV_MODEL_COMFY_RENAMING_MAP` strips it when the loader reads
# them. Our probe reads RAW keys so we strip it too — otherwise the
# exclusion names we emit won't match the in-memory module paths.
_COMFY_PREFIX = "model.diffusion_model."

# torch.compile wraps each transformer block in an OptimizedModule whose
# inner attributes are reached through `_orig_mod`. After compile,
# `model.named_modules()` yields paths like
# `transformer_blocks.1._orig_mod.attn1.to_gate_logits`. We strip the
# injected segment before the substring check so exclusions can use
# the plain (pre-compile) form regardless of whether compile ran.
_ORIG_MOD_SEGMENT_RE = re.compile(r"\._orig_mod(?=\.|$)")


def _strip_comfy_prefix(name: str) -> str:
    return name[len(_COMFY_PREFIX):] if name.startswith(_COMFY_PREFIX) else name


def _normalize_for_match(name: str) -> str:
    """Canonicalize a module/key path for exclusion matching.

    Removes the `_orig_mod.` segment injected by torch.compile AND the
    `model.diffusion_model.` prefix that may persist on raw safetensors
    keys. The result is the canonical in-memory module path used by
    `EXCLUDED_LAYER_SUBSTRINGS` and by our probe output.
    """
    return _ORIG_MOD_SEGMENT_RE.sub("", _strip_comfy_prefix(name))


def _probe_fp8_exclusions(paths: list[str]) -> tuple[str, ...]:
    """Return module paths that must stay as nn.Linear (BF16).

    Opens each `.safetensors` header (no tensor data loaded) and
    collects every `.weight` key whose dtype is not `float8_e4m3fn`.
    Each name is canonicalized by stripping `model.diffusion_model.`
    so it matches the post-rename module path the LTX loader produces.

    Why we probe: the Lightricks `*-fp8.safetensors` checkpoints leave
    a subset of transformer block 1 in BF16 (AV cross modules, a few
    MLPs) that upstream's hard-coded `EXCLUDED_LAYER_SUBSTRINGS` does
    not cover. Without these extras, `_apply_fp8_prepare_to_model`
    swaps those modules to FP8Linear and `load_state_dict` fails with
    size mismatches at first inference.

    Logs a dtype histogram per file, a sample of transformer-block
    extras, and a warning if zero transformer-block BF16 entries were
    found (which would mean the crash that motivated this probe isn't
    actually being prevented).
    """
    from safetensors import safe_open  # lazy — downloads pull safetensors

    bare: set[str] = set()
    for path in paths:
        dtype_counts: dict[str, int] = {}
        bf16_samples: list[str] = []
        file_bare_adds = 0
        with safe_open(path, framework="pt") as f:
            for key in f.keys():
                if not key.endswith(".weight"):
                    continue
                dt = f.get_slice(key).get_dtype()
                dtype_counts[dt] = dtype_counts.get(dt, 0) + 1
                if dt != "F8_E4M3":
                    normalized = _strip_comfy_prefix(key[: -len(".weight")])
                    bare.add(normalized)
                    file_bare_adds += 1
                    if len(bf16_samples) < 3:
                        bf16_samples.append(f"{key} [{dt}]")
        logger.info(
            "FP8 probe: %s — dtype histogram=%s, non-FP8 weights=%d",
            os.path.basename(path),
            sorted(dtype_counts.items()),
            file_bare_adds,
        )
        if bf16_samples:
            logger.info("FP8 probe: %s — raw BF16 samples: %s",
                        os.path.basename(path), bf16_samples)

    block_entries = sorted(n for n in bare if n.startswith("transformer_blocks."))
    if not block_entries:
        logger.warning(
            "FP8 probe: NO transformer_blocks.* BF16 weights found across %d "
            "checkpoint(s). Either the Lightricks FP8 format changed or the "
            "prefix/dtype probe logic is stale. Block-1 load is expected to "
            "fail with size mismatches.",
            len(paths),
        )
    else:
        logger.info(
            "FP8 probe: %d transformer_blocks.* BF16 entries (first 10): %s",
            len(block_entries),
            block_entries[:10],
        )

    return tuple(sorted(bare))


def _build_scaled_mm_policy(extras: tuple[str, ...]):
    """Wrapper-only equivalent of `QuantizationPolicy.fp8_scaled_mm()`
    that accepts a per-call exclusion list AND normalizes names before
    substring matching.

    The upstream factory has two problems for the Lightricks
    `*-fp8.safetensors` format:

    1. Its `EXCLUDED_LAYER_SUBSTRINGS` doesn't cover the BF16 modules
       Lightricks left in transformer block 1. So we append `extras`
       produced by `_probe_fp8_exclusions`.

    2. `_should_skip_layer` in upstream is a raw substring check. With
       regional `torch.compile` active, module names pick up a
       `._orig_mod.` segment that breaks substring matching against
       bare exclusions. We replace `_apply_fp8_prepare_to_model` and
       `_create_transpose_kv_operation` with local versions that
       normalize `_orig_mod` out before the check.

    We also can't edit upstream directly: the Dockerfile (line 73)
    clones `Lightricks/LTX-2` fresh at build, so any change in
    `LTX-2-ref/` is local-only and never reaches the pod.
    """
    from ltx_core.loader.module_ops import ModuleOps
    from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
    from ltx_core.model.transformer import LTXModel
    from ltx_core.quantization import QuantizationPolicy
    from ltx_core.quantization.fp8_scaled_mm import (
        FP8_PREPARE_MODULE_OPS,
        FP8Linear,
        FP8_TRANSPOSE_SD_OPS,
        _linear_to_fp8linear,
    )

    try:
        import tensorrt_llm  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "tensorrt_llm not installed — fp8-trtllm extra missing"
        ) from e

    if not extras:
        # No checkpoint-driven extras — but we still want `_orig_mod`
        # normalization, because upstream's baseline exclusions like
        # `"transformer_blocks.0."` happen to tolerate compile (they
        # stop at the dot, so substring matches regardless). Keeping
        # the upstream primitives here is safe; no crash has been
        # observed without extras.
        return QuantizationPolicy(
            sd_ops=FP8_TRANSPOSE_SD_OPS,
            module_ops=(FP8_PREPARE_MODULE_OPS,),
        )

    # Exact match against the probe result, NOT substring match against
    # upstream's baseline. Baseline includes block-level catch-alls like
    # `"transformer_blocks.0."` and `"transformer_blocks.43..47."` that
    # substring-match *every* module in those blocks. Lightricks' FP8
    # checkpoint keeps some modules in those blocks as FP8 — the
    # substring rule over-excluded them, leaving FP8 weights loaded into
    # plain nn.Linear modules and crashing F.linear on first forward.
    # `extras` is the ground truth: the safetensors header's BF16 keys.
    exclusions = frozenset(extras)

    def _should_skip(name: str) -> bool:
        return _normalize_for_match(name) in exclusions

    import torch
    from torch import nn

    def _prepare(model):
        """Drop-in replacement for `_apply_fp8_prepare_to_model`.

        Walks named_modules; for each nn.Linear (not already FP8Linear)
        whose canonicalized name does NOT match an exclusion, swaps it
        to FP8Linear. Logs per-block swap/skip totals for ALL blocks
        encountered so we can verify swap coverage end-to-end (the
        upstream design keeps block 0 and the late blocks in BF16; we
        need visibility into blocks 2..N to confirm they are FP8).
        """
        from collections import defaultdict

        replacements: list[tuple[nn.Module, str, nn.Linear]] = []
        block_totals: dict[int, int] = defaultdict(int)
        block_swapped: dict[int, int] = defaultdict(int)
        block_skipped: dict[int, int] = defaultdict(int)
        block_swap_samples: dict[int, list[str]] = defaultdict(list)
        block_skip_samples: dict[int, list[str]] = defaultdict(list)

        _BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")

        def _block_idx(n: str) -> int | None:
            m = _BLOCK_RE.match(n)
            return int(m.group(1)) if m else None

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear) or isinstance(module, FP8Linear):
                continue
            normalized = _normalize_for_match(name)
            bi = _block_idx(normalized)
            if bi is not None:
                block_totals[bi] += 1
            if _should_skip(name):
                if bi is not None:
                    block_skipped[bi] += 1
                    if len(block_skip_samples[bi]) < 3:
                        block_skip_samples[bi].append(normalized)
                continue
            if bi is not None:
                block_swapped[bi] += 1
                if len(block_swap_samples[bi]) < 3:
                    block_swap_samples[bi].append(normalized)
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            replacements.append((parent, attr_name, module))

        for parent, attr_name, linear in replacements:
            setattr(parent, attr_name, _linear_to_fp8linear(linear))

        # Aggregate roll-up
        total_in_blocks = sum(block_totals.values())
        total_swapped = sum(block_swapped.values())
        total_skipped = sum(block_skipped.values())
        pct_swapped = (100.0 * total_swapped / total_in_blocks) if total_in_blocks else 0.0
        logger.info(
            "FP8 prepare: aggregate over %d transformer blocks — "
            "linears=%d, swapped=%d (%.1f%%), skipped=%d",
            len(block_totals), total_in_blocks, total_swapped, pct_swapped, total_skipped,
        )

        # Per-block summary, sorted; flag fully-BF16 blocks for fast scan.
        all_bf16_blocks = []
        for bi in sorted(block_totals.keys()):
            tag = ""
            if block_totals[bi] > 0 and block_swapped[bi] == 0:
                tag = " [ALL-BF16]"
                all_bf16_blocks.append(bi)
            logger.info(
                "FP8 prepare: block-%02d count=%d, swapped=%d, skipped=%d%s",
                bi, block_totals[bi], block_swapped[bi], block_skipped[bi], tag,
            )
        if all_bf16_blocks:
            logger.info(
                "FP8 prepare: blocks fully BF16 (swapped=0): %s",
                all_bf16_blocks,
            )

        # Spot-check samples from a few representative blocks (first,
        # mid, last) to confirm the matcher is producing sensible names.
        seen = sorted(block_totals.keys())
        spot_blocks = []
        if seen:
            spot_blocks = [seen[0]]
            mid = seen[len(seen) // 2]
            if mid not in spot_blocks:
                spot_blocks.append(mid)
            if seen[-1] not in spot_blocks:
                spot_blocks.append(seen[-1])
        for bi in spot_blocks:
            if block_swap_samples[bi]:
                logger.info(
                    "FP8 prepare: block-%02d swap samples: %s",
                    bi, block_swap_samples[bi],
                )
            if block_skip_samples[bi]:
                logger.info(
                    "FP8 prepare: block-%02d skip samples: %s",
                    bi, block_skip_samples[bi],
                )
        return model

    def _transpose_op(key: str, value: torch.Tensor):
        if not key.endswith(".weight"):
            return [KeyValueOperationResult(key, value)]
        if value.dim() != 2 or value.dtype != torch.float8_e4m3fn:
            return [KeyValueOperationResult(key, value)]
        layer_name = key.rsplit(".weight", 1)[0]
        if _should_skip(layer_name):
            return [KeyValueOperationResult(key, value)]
        return [KeyValueOperationResult(key, value.t())]

    sd_ops = SDOps("fp8_transpose_weights_normalized").with_kv_operation(
        _transpose_op,
        key_prefix="transformer_blocks.",
        key_suffix=".weight",
    )
    module_ops = ModuleOps(
        name="fp8_prepare_for_loading_normalized",
        matcher=lambda m: isinstance(m, LTXModel),
        mutator=_prepare,
    )
    return QuantizationPolicy(sd_ops=sd_ops, module_ops=(module_ops,))


def _select_fp8_mode() -> str:
    """Pick the FP8 compute path.

    ``scaled_mm`` is the H100-optimised W8A8 path (real FP8 GEMM via
    TRT-LLM ``cublas_scaled_mm``), ``cast`` is the Ada/Blackwell W8A16
    path (weights FP8, activations upcast to BF16 per forward).

    Order of precedence:
      1. ``LTX_FP8_MODE=scaled_mm|cast`` env override (matches
         download_models.py so downloads and runtime agree).
      2. GPU capability auto-detect: H100/H200 (SM 9.0) → ``scaled_mm``,
         otherwise → ``cast``.
      3. ``cast`` fallback when CUDA is unavailable.
    """
    override = os.getenv("LTX_FP8_MODE", "").strip().lower()
    if override in ("scaled_mm", "cast"):
        return override
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability(0)
        if cap == (9, 0):
            return "scaled_mm"
    return "cast"


class LTXVideoGenerator:
    """Initialises the LTX-2.3 two-stage pipeline and runs inference."""

    def __init__(self, model_dir: str = "/models") -> None:
        # BF16 DiT checkpoint. Also holds VAE / audio decoder / vocoder /
        # image encoder / embeddings processor weights — loaded by the
        # non-DiT blocks through *_COMFY_KEYS_FILTER, so this file is
        # required on every path (including scaled_mm).
        checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-dev.safetensors")
        spatial_upsampler_path = os.path.join(
            model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        distilled_lora_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        dev_fp8_path = os.path.join(model_dir, "ltx-2.3-22b-dev-fp8.safetensors")
        distilled_fp8_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-fp8.safetensors"
        )
        gemma_root = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")

        logger.info("Initializing LTX-2.3 pipeline ...")
        self._log_vram("before pipeline init")

        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.media_io import encode_video
        from ltx_pipelines.utils.types import OffloadMode

        self._encode_video = encode_video

        # Weight placement. Upstream PR #201 (2026-04-23) replaced the
        # per-call `streaming_prefetch_count` kwarg with a constructor
        # `offload_mode` enum on `TI2VidTwoStagesPipeline` and
        # `DiffusionStage`. OffloadMode.NONE keeps all weights on GPU
        # (fastest); CPU streams layers from pinned host RAM for <40 GB
        # pods; DISK re-reads from disk each pass for tiny cards.
        gpu_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0
        )
        offload_mode = OffloadMode.NONE if gpu_vram_gb >= 40 else OffloadMode.CPU
        self._offload_mode = offload_mode
        logger.info(
            "Offload mode: %s (GPU=%.0f GB)", offload_mode.value, gpu_vram_gb,
        )

        # Pick FP8 path. scaled_mm = H100 W8A8 through TRT-LLM
        # cublas_scaled_mm; cast = W8A16 runtime downcast/upcast.
        fp8_mode = _select_fp8_mode()
        self._fp8_mode = fp8_mode

        from ltx_core.quantization import QuantizationPolicy
        if fp8_mode == "scaled_mm":
            if not (os.path.exists(dev_fp8_path) and os.path.exists(distilled_fp8_path)):
                raise RuntimeError(
                    f"LTX_FP8_MODE=scaled_mm requires pre-quantized FP8 checkpoints at "
                    f"{dev_fp8_path} and {distilled_fp8_path}. Run download_models.py "
                    f"with LTX_FP8_MODE=scaled_mm."
                )
            # Checkpoint-driven exclusion list — PER STAGE. The dev and
            # distilled FP8 checkpoints do NOT keep the same modules in
            # BF16 (distilled has ~34 extra BF16 entries that are FP8 in
            # dev). Merging their exclusions and reusing one policy for
            # both stages leaves dev-stage modules as nn.Linear while
            # loading FP8 weights into them → BF16×FP8 dtype error on
            # the first forward. Probe each file on its own.
            extras_dev = _probe_fp8_exclusions([dev_fp8_path])
            extras_distilled = _probe_fp8_exclusions([distilled_fp8_path])
            logger.info(
                "FP8 checkpoint probe: dev=%d, distilled=%d non-FP8 weight modules",
                len(extras_dev),
                len(extras_distilled),
            )
            for label, extras in (("dev", extras_dev), ("distilled", extras_distilled)):
                if extras:
                    preview = ", ".join(extras[:5])
                    more = f" (+{len(extras) - 5} more)" if len(extras) > 5 else ""
                    logger.info("FP8 probe extras (%s, first 5): %s%s", label, preview, more)
            quantization_dev = _build_scaled_mm_policy(extras_dev)
            quantization_distilled = _build_scaled_mm_policy(extras_distilled)
            # Placeholder fed to the pipeline constructor; the stage
            # rebuild below overwrites both stages with per-checkpoint
            # policies. DiffusionStage defers weight load so this
            # placeholder never actually touches a DiT checkpoint.
            quantization = quantization_dev
            logger.info(
                "FP8 mode: scaled_mm (W8A8, TRT-LLM cublas_scaled_mm, H100-optimised)"
            )
        else:
            quantization = QuantizationPolicy.fp8_cast()
            quantization_dev = None
            quantization_distilled = None
            logger.info("FP8 mode: cast (W8A16, weights FP8 / activations BF16)")

        # Upstream PR #201 constraint: DiffusionStage rejects
        # `offload_mode != NONE` combined with either `quantization` or
        # `torch_compile`. On H100 (our primary target) offload_mode is
        # NONE so this is a no-op. On a <40 GB pod we'd otherwise hit
        # a ValueError at pipeline init; drop quantization + compile so
        # the pipeline still boots in streaming BF16 mode.
        if offload_mode != OffloadMode.NONE and quantization is not None:
            logger.warning(
                "Offload mode %s requires non-quantized BF16 weights — "
                "dropping FP8 quantization and torch.compile for this run. "
                "This branch is tuned for ≥40 GB GPUs; consider a larger pod.",
                offload_mode.value,
            )
            quantization = None
            quantization_dev = None
            quantization_distilled = None
            fp8_mode = "bf16"
            self._fp8_mode = fp8_mode

        # CPU weight caching — only one model on GPU at a time
        registry = None
        try:
            from ltx_core.loader import StateDictRegistry
            registry = StateDictRegistry()
            logger.info("Using StateDictRegistry (CPU weight caching)")
        except ImportError:
            logger.warning("StateDictRegistry not available")

        # Distilled LoRA — only meaningful on the cast path. scaled_mm
        # cannot fuse a BF16 LoRA into pre-quantized FP8 weights at load
        # time; we instead swap Stage 2 to the pre-fused distilled-fp8
        # checkpoint further down.
        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        if fp8_mode == "cast":
            distilled_lora = [
                LoraPathStrengthAndSDOps(
                    path=distilled_lora_path,
                    strength=0.8,
                    sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
                )
            ]
        else:
            distilled_lora = []

        # Build pipeline. On scaled_mm we pass the BF16 checkpoint so
        # PromptEncoder / ImageConditioner / VideoDecoder / AudioDecoder /
        # VideoUpsampler see their VAE + encoder keys; Stage 1 and Stage 2
        # DiffusionStages are rebuilt below to point at the FP8 DiT files.
        pipeline_kwargs = dict(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
            offload_mode=offload_mode,
        )
        if quantization is not None:
            pipeline_kwargs["quantization"] = quantization
        if registry is not None:
            pipeline_kwargs["registry"] = registry

        # torch.compile — LTX-2's regional compile (per transformer block,
        # not whole model) via COMPILE_TRANSFORMER SDOps in
        # ltx_core/model/transformer/compiling.py. Each transformer block
        # gets wrapped with torch.compile(m); small blocks compile fast
        # and are cached, so the effective cost is paid once per pod boot.
        # Expected ~15-30% Stage 1 speedup on Ada + Hopper; transparent to
        # numerics. Default ON; set ENABLE_TORCH_COMPILE=0 to disable.
        torch_compile_enabled = os.getenv("ENABLE_TORCH_COMPILE", "1").strip().lower() not in (
            "0", "false", "no", "off", "",
        )
        if torch_compile_enabled and offload_mode == OffloadMode.NONE:
            pipeline_kwargs["torch_compile"] = True
            logger.info("torch.compile ENABLED (regional per transformer block)")
        elif torch_compile_enabled:
            logger.warning(
                "torch.compile requested but offload_mode=%s disallows it "
                "(upstream DiffusionStage guard); running uncompiled.",
                offload_mode.value,
            )
        else:
            logger.info("torch.compile disabled via ENABLE_TORCH_COMPILE=%s", os.getenv("ENABLE_TORCH_COMPILE"))

        # Optional FA3 enablement. When `LTX_ATTENTION_TYPE=flash_attention_3`
        # is set, patch LTX's model configurator so every Attention module
        # constructed by the pipeline routes through the FA3 wrapper. Must
        # run before the TI2VidTwoStagesPipeline's Builder constructs any
        # transformer, which happens lazily on the first __call__.
        _requested_attn = os.environ.get("LTX_ATTENTION_TYPE", "").lower()
        if _requested_attn == "flash_attention_3":
            from src.attention_override import enable_flash_attention_3
            enable_flash_attention_3()

        _t0 = time.perf_counter()
        self._pipeline = TI2VidTwoStagesPipeline(**pipeline_kwargs)
        logger.info(
            "Init timing: TI2VidTwoStagesPipeline construction took %.2fs",
            time.perf_counter() - _t0,
        )
        self._log_vram("after pipeline init")

        # scaled_mm: swap Stage 1 and Stage 2 DiffusionStages to point at
        # the pre-quantized FP8 DiT checkpoints. The upstream
        # TI2VidTwoStagesPipeline exposes a single checkpoint_path that it
        # feeds to every block, which is fine for the cast path but wrong
        # here — we need BF16 for the VAE/encoder blocks and FP8 for the
        # two DiT stages. Rebuilding after construction is safe because
        # DiffusionStage defers weight load to its first __call__.
        if fp8_mode == "scaled_mm":
            from ltx_pipelines.utils.blocks import DiffusionStage
            # No registry on the DiT stages: state dicts load straight to
            # GPU and are kept alive by registry refs even after
            # `gpu_model.__exit__` runs `model.to("meta")`. With both
            # ~40 GB FP8 stages pinned, the Stage 2 build OOMs on H100
            # 80 GB. Builder defaults to DummyRegistry → no caching.
            _t1 = time.perf_counter()
            self._pipeline.stage_1 = DiffusionStage(
                checkpoint_path=dev_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization_dev,
                registry=None,
                torch_compile=pipeline_kwargs.get("torch_compile", False),
                offload_mode=offload_mode,
            )
            logger.info(
                "Init timing: DiffusionStage(stage_1) constructor took %.3fs "
                "(weight load deferred to first __call__)",
                time.perf_counter() - _t1,
            )
            # Stage 2 uses the distilled-fp8 checkpoint (distilled weights
            # already fused in) instead of base + distilled LoRA.
            _t2 = time.perf_counter()
            self._pipeline.stage_2 = DiffusionStage(
                checkpoint_path=distilled_fp8_path,
                dtype=self._pipeline.dtype,
                device=self._pipeline.device,
                loras=(),
                quantization=quantization_distilled,
                registry=None,
                torch_compile=pipeline_kwargs.get("torch_compile", False),
                offload_mode=offload_mode,
            )
            logger.info(
                "Init timing: DiffusionStage(stage_2) constructor took %.3fs "
                "(weight load deferred to first __call__)",
                time.perf_counter() - _t2,
            )
            logger.info(
                "scaled_mm: rebuilt stage_1 → %s, stage_2 → %s (distilled pre-fused)",
                os.path.basename(dev_fp8_path),
                os.path.basename(distilled_fp8_path),
            )

            # Wrap each stage's _build_transformer to log wall time +
            # post-load dtype histogram. This is the only place we can
            # observe the actual swapped-vs-skipped result, because the
            # swap happens inside Builder.build(...) at the END of the
            # weight load — well after _prepare(model) returns, when
            # the safetensors values get assigned via load_state_dict
            # (assign=True). A meta module renamed wrong by _prepare
            # would still log "swapped" there but the actual weight
            # buffer would land on a stale nn.Linear. The histogram
            # below reads `module.weight.dtype` after the load → ground
            # truth.
            _install_build_transformer_audit(self._pipeline.stage_1, "stage_1")
            _install_build_transformer_audit(self._pipeline.stage_2, "stage_2")

        # Aggressive allocator flush at the Stage 1 → Stage 2 boundary.
        # Fixes intermittent
        #   torch.AcceleratorError: CUDA error: invalid argument
        # out of block_streaming.py on small (<40 GB) cards: the streaming
        # path pins transformer blocks in host memory and PR #201 removed
        # the host-side empty_cache that used to drain that arena. This
        # hook restores that drain.
        #
        # Skip on OffloadMode.NONE (H100): the streaming path is inactive,
        # Stage 1 is already released by gpu_model.__exit__ before the
        # hook runs, and the allocator has already returned reserved
        # bytes. Empirically (cold + warm runs on H100 80 GB):
        #   alloc 32.42 → 32.42 GB (unchanged), reserved 32.92 → 32.92 GB
        # — i.e. exactly zero work for the hook to do, and a misleading
        # log line. Drop it on the NONE path.
        if offload_mode != OffloadMode.NONE:
            _t3 = time.perf_counter()
            _install_stage2_cleanup_hook(self._pipeline)
            logger.info(
                "Init timing: _install_stage2_cleanup_hook took %.3fs",
                time.perf_counter() - _t3,
            )
        else:
            logger.info(
                "Stage 2 cleanup hook skipped: offload_mode=NONE "
                "(streaming path inactive; gpu_model.__exit__ already "
                "drains stage 1 before stage 2 builds)"
            )

        # TeaCache — opt-in via ENABLE_TEACACHE=1. Skips the transformer
        # forward on diffusion steps where the input hasn't changed
        # enough (rescaled relative-L1 below TEACACHE_THRESHOLD). Works
        # across any GPU + attention backend — it never touches the
        # attention kernel, just caches the module's output. Validated
        # on LTX-Video with ~1.6–2.1× lossless speedup.
        from src.teacache import enable_teacache, teacache_config_from_env
        teacache_cfg = teacache_config_from_env()
        self._teacache_enabled = teacache_cfg is not None
        if self._teacache_enabled:
            _t4 = time.perf_counter()
            enable_teacache(self._pipeline, **teacache_cfg)
            logger.info(
                "Init timing: enable_teacache took %.3fs",
                time.perf_counter() - _t4,
            )

        # Boot-time attention-backend fingerprint. LTX-2's
        # `AttentionFunction.DEFAULT` resolves to `PytorchAttention` (torch
        # SDPA dispatcher) in this image. We probe ltx_core's actual
        # module-level binding of `flash_attn_interface` (not our env) so
        # the log reflects what the pipeline will use.
        try:
            from ltx_core.model.transformer import attention as _ltx_attn
            from ltx_core.model.transformer.attention import AttentionFunction
            _has_fa3_live = _ltx_attn.flash_attn_interface is not None
            _attn_resolved = type(AttentionFunction.DEFAULT.to_callable()).__name__
        except Exception as _e:
            _has_fa3_live = None
            _attn_resolved = f"unknown ({type(_e).__name__})"
        logger.info(
            "Attention fingerprint — ltx_core.fa3=%s, "
            "LTX default resolves to: %s, requested: %s",
            _has_fa3_live, _attn_resolved, _requested_attn or "default",
        )
        if _requested_attn == "flash_attention_3" and _has_fa3_live is False:
            logger.warning(
                "FA3 requested but ltx_core.attention.flash_attn_interface is None "
                "— first attention call will fail."
            )

        # Optional components (guiders, tiling)
        self._MultiModalGuiderParams = None
        try:
            from ltx_core.components.guiders import MultiModalGuiderParams
            self._MultiModalGuiderParams = MultiModalGuiderParams
        except ImportError:
            pass

        self._TilingConfig = None
        self._get_video_chunks_number = None
        try:
            from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
            self._TilingConfig = TilingConfig
            self._get_video_chunks_number = get_video_chunks_number
        except ImportError:
            pass

        # Token-count-aware Stage-1 sigma schedule. The standard
        # TI2VidTwoStagesPipeline (ti2vid_two_stages.py:155-157) calls
        # self._scheduler.execute(steps=N) without a `latent=` arg, so
        # the scheduler falls back to MAX_SHIFT_ANCHOR=4096 tokens for
        # its sigma-shift formula (schedulers.py:32). The HQ variant
        # (ti2vid_two_stages_hq.py:168-170) passes the actual latent
        # shape to get token-count-dependent shift. We mirror HQ here:
        # for our 1920x1088 renders, real Stage-1 token counts are 8K
        # (5s) or 16K (10s) — the 4096-anchor under-shifts by 1.7x and
        # 3.0x respectively, which biases the schedule toward fine-detail
        # denoising and starves the high-noise structure-forming regime
        # most acutely on long clips. Result: late-frame drift on 10s.
        self._VideoPixelShape = None
        self._VideoLatentShape = None
        try:
            from ltx_core.types import VideoPixelShape, VideoLatentShape
            self._VideoPixelShape = VideoPixelShape
            self._VideoLatentShape = VideoLatentShape
        except ImportError:
            pass

        logger.info("Pipeline ready.")

    def _log_vram(self, label: str) -> None:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated(0) / 1e9
            res = torch.cuda.memory_reserved(0) / 1e9
            logger.info("VRAM %s: %.2f GB allocated, %.2f GB reserved", label, alloc, res)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
        width: int = 1024,
        height: int = 1536,
        num_frames: int = 121,
        num_inference_steps: int = 30,
        seed: int = 42,
        frame_rate: float = 24.0,
        cfg_scale: float = 3.0,
        stg_scale: float = 1.0,
        rescale_scale: float = 0.7,
    ) -> dict:
        """Run inference and encode MP4. Returns dict with output_path, output_filename, etc."""
        width, height, num_frames = _round_user_inputs(width, height, num_frames)
        job_id = uuid.uuid4().hex[:12]

        logger.info(
            "Job %s: prompt=%r, %dx%d, %d frames, %d steps, seed=%d",
            job_id, prompt[:80], width, height, num_frames, num_inference_steps, seed,
        )

        try:
            # Guidance params
            video_guider_params = None
            audio_guider_params = None
            if self._MultiModalGuiderParams is not None:
                video_guider_params = self._MultiModalGuiderParams(
                    cfg_scale=cfg_scale, stg_scale=stg_scale,
                    rescale_scale=rescale_scale, modality_scale=3.0, stg_blocks=[28],
                )
                audio_guider_params = self._MultiModalGuiderParams(
                    cfg_scale=7.0, stg_scale=1.0, rescale_scale=0.7,
                    modality_scale=3.0, stg_blocks=[28],
                )

            # Tiling config
            tiling_config = None
            video_chunks_number = None
            if self._TilingConfig and self._get_video_chunks_number:
                tiling_config = self._TilingConfig.default()
                video_chunks_number = self._get_video_chunks_number(num_frames, tiling_config)

            # Token-count-aware Stage-1 sigma schedule (mirrors
            # ti2vid_two_stages_hq.py:168-170). Without this, the
            # standard pipeline anchors to 4096 tokens regardless of
            # actual latent size, under-shifting the schedule on long
            # clips. Stage 1 runs at half-res, so the latent we hand
            # to the scheduler must reflect that.
            stage_1_sigmas = None
            scheduler = getattr(self._pipeline, "_scheduler", None)
            if (
                self._VideoPixelShape is not None
                and self._VideoLatentShape is not None
                and scheduler is not None
            ):
                stage_1_pixel_shape = self._VideoPixelShape(
                    batch=1,
                    frames=num_frames,
                    width=width // 2,
                    height=height // 2,
                    fps=frame_rate,
                )
                stage_1_latent_shape = self._VideoLatentShape.from_pixel_shape(
                    stage_1_pixel_shape
                )
                empty_stage_1_latent = torch.empty(stage_1_latent_shape.to_torch_shape())
                stage_1_sigmas = scheduler.execute(
                    latent=empty_stage_1_latent,
                    steps=num_inference_steps,
                )
                logger.info(
                    "Job %s: stage_1 sigma schedule — tokens=%d (lat %dx%dx%d), "
                    "first=%.4f last=%.4f",
                    job_id,
                    stage_1_latent_shape.token_count(),
                    stage_1_latent_shape.frames,
                    stage_1_latent_shape.height,
                    stage_1_latent_shape.width,
                    float(stage_1_sigmas[0]),
                    float(stage_1_sigmas[-2]) if stage_1_sigmas.numel() >= 2 else float("nan"),
                )

            # Frame-count-aware Stage-2 sigma schedule. Upstream
            # STAGE_2_DISTILLED_SIGMAS is a fixed 3-step schedule tuned
            # for 121f / 5s clips; long clips drift at the tail without
            # this override. Disable via DISABLE_BMGJET_STAGE2_SIGMAS=1.
            stage_2_sigmas = None
            if os.getenv("DISABLE_BMGJET_STAGE2_SIGMAS", "0").strip().lower() in ("0", "false", "no", "off", ""):
                stage_2_sigmas = _bmgjet_stage_2_sigmas(num_frames, prompt)
                regime_name = "short" if num_frames < 241 else ("medium" if num_frames < 601 else "long")
                logger.info(
                    "Job %s: stage_2 sigma schedule (bmgjet) — frames=%d, regime=%s, sigmas=[%s]",
                    job_id, num_frames, regime_name,
                    ", ".join(f"{x:.4f}" for x in stage_2_sigmas.tolist()),
                )

            # Run pipeline
            start_time = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats(0)

            # Weight placement was decided at __init__ time via
            # `offload_mode` (PR #201 upstream refactor — `streaming_prefetch_count`
            # no longer exists on `__call__`). `max_batch_size=1` is the
            # upstream default and is correct on big GPUs; raising it
            # above 1 has previously caused issues on ≥40 GB pods.
            max_batch_size = 1
            logger.info(
                "Job %s: offload_mode=%s, max_batch_size=%d, teacache=%s",
                job_id, self._offload_mode.value, max_batch_size,
                getattr(self, "_teacache_enabled", False),
            )

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate, num_inference_steps=num_inference_steps,
                images=[],
                max_batch_size=max_batch_size,
            )
            if video_guider_params is not None:
                call_kwargs["video_guider_params"] = video_guider_params
            if audio_guider_params is not None:
                call_kwargs["audio_guider_params"] = audio_guider_params
            if tiling_config is not None:
                call_kwargs["tiling_config"] = tiling_config
            if stage_1_sigmas is not None:
                call_kwargs["stage_1_sigmas"] = stage_1_sigmas
            if stage_2_sigmas is not None:
                call_kwargs["stage_2_sigmas"] = stage_2_sigmas

            result = self._pipeline(**call_kwargs)
            video, audio = result if isinstance(result, tuple) else (result, None)
            generation_time = time.time() - start_time
            if torch.cuda.is_available():
                peak = torch.cuda.max_memory_allocated(0) / 1e9
                logger.info(
                    "Job %s: generation took %.1fs (peak VRAM %.2f GB, audio=%s)",
                    job_id, generation_time, peak, audio is not None,
                )
            else:
                logger.info("Job %s: generation took %.1fs", job_id, generation_time)

            # Encode video
            output_filename = f"ltx_{job_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            encode_kwargs = dict(video=video, fps=int(frame_rate), output_path=output_path)
            if audio is not None:
                encode_kwargs["audio"] = audio
            if video_chunks_number is not None:
                encode_kwargs["video_chunks_number"] = video_chunks_number
            encode_start = time.time()
            self._encode_video(**encode_kwargs)
            logger.info(
                "Job %s: mp4 encode %.1fs → %s",
                job_id, time.time() - encode_start, output_path,
            )

            return {
                "output_path": output_path,
                "output_filename": output_filename,
                "generation_time_seconds": round(generation_time, 2),
                "parameters": {
                    "width": width, "height": height, "num_frames": num_frames,
                    "num_inference_steps": num_inference_steps, "seed": seed,
                    "frame_rate": frame_rate, "cfg_scale": cfg_scale,
                    "stg_scale": stg_scale, "rescale_scale": rescale_scale,
                },
            }

        except Exception:
            logger.exception("Job %s failed", job_id)
            torch.cuda.empty_cache()
            raise
