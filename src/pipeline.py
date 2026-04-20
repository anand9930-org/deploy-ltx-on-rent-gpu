"""Shared LTX-2.3 pipeline wrapper.

All VRAM optimisation techniques live here.
"""

import logging
import os
import tempfile
import time
import uuid

import torch

logger = logging.getLogger(__name__)

DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, inconsistent motion, blurry, jittery, distorted, "
    "low resolution, watermark, text, oversaturated"
)

# Post-load Gemma quantization. Gemma is always loaded from disk as BF16
# (LTX-2's custom safetensors loader can't decode compressed-tensors / GPTQ
# formats — see docs/gemma-compat-checklist.md). After the pipeline is
# built, we walk the prompt-encoder subtree and replace every nn.Linear
# with an HQQ-quantized equivalent. Activations stay BF16; the DiT sees an
# unchanged interface. Supported modes:
#   none  — default, BF16 runtime (~26 GB Gemma), known-good
#   hqq4  — 4-bit HQQ, ~7 GB Gemma, enables 48 GB pure-GPU
#   hqq8  — 8-bit HQQ, ~13 GB Gemma, conservative floor
_POST_LOAD_QUANT_MODES = ("none", "hqq4", "hqq8")


def _resolve_post_load_quant() -> str:
    raw = os.getenv("GEMMA_POST_LOAD_QUANT", "none").strip().lower()
    if raw not in _POST_LOAD_QUANT_MODES:
        logger.warning("Unknown GEMMA_POST_LOAD_QUANT=%r; falling back to none", raw)
        raw = "none"
    return raw


def _find_nn_modules_in(obj, max_depth: int = 6):
    """Recursively walk an arbitrary object graph and yield every nn.Module
    reachable via ``vars()`` / ``named_children()`` — including attributes
    that start with an underscore. Used to locate the Gemma nn.Module that
    LTX-2's PromptEncoder (a plain Python class, not an nn.Module) stashes
    somewhere inside itself.

    Yields (path, module) pairs with cycle detection.
    """
    import torch.nn as nn

    visited: set[int] = set()

    def _walk(node, depth: int, path: str):
        node_id = id(node)
        if node_id in visited or depth > max_depth:
            return
        visited.add(node_id)
        if isinstance(node, nn.Module):
            yield path, node
            # Don't descend into the Module's children here — caller handles
            # that via named_modules(). We still descend through any plain
            # Python attributes the Module might hold (unusual but cheap).
            try:
                obj_vars = vars(node)
            except TypeError:
                return
            for name, attr in list(obj_vars.items()):
                if isinstance(attr, nn.Module):
                    continue  # reachable via named_children
                if _is_container_like(attr):
                    yield from _walk(attr, depth + 1, f"{path}.{name}")
            return
        # Non-nn.Module container: scan attrs for nn.Modules + sub-wrappers
        try:
            obj_vars = vars(node)
        except TypeError:
            return
        for name, attr in list(obj_vars.items()):
            child_path = f"{path}.{name}" if path else name
            if isinstance(attr, nn.Module):
                yield from _walk(attr, depth + 1, child_path)
            elif _is_container_like(attr):
                yield from _walk(attr, depth + 1, child_path)

    yield from _walk(obj, 0, "prompt_encoder")


def _is_container_like(obj) -> bool:
    """True if obj is worth recursing into (has ``__dict__`` and isn't a
    trivial leaf). Excludes tensors, callables, and primitive types."""
    import types
    if obj is None:
        return False
    if isinstance(obj, (str, int, float, bool, bytes, bytearray)):
        return False
    if torch.is_tensor(obj):
        return False
    if isinstance(obj, (types.FunctionType, types.MethodType, type)):
        return False
    return hasattr(obj, "__dict__") or isinstance(obj, (list, tuple, dict))


def _dump_prompt_encoder_structure(pipeline) -> None:
    """One-shot diagnostic: log the nn.Modules reachable from
    ``pipeline.prompt_encoder`` so we can see where Gemma is nested."""
    prompt_encoder = getattr(pipeline, "prompt_encoder", None)
    if prompt_encoder is None:
        logger.info("prompt_encoder: (missing)")
        return
    logger.info("prompt_encoder top-level type: %s", type(prompt_encoder).__name__)
    found = list(_find_nn_modules_in(prompt_encoder))
    if not found:
        logger.info("prompt_encoder: no nn.Module descendants found")
        return
    for path, mod in found[:12]:  # cap log volume
        try:
            n_params = sum(p.numel() for p in mod.parameters(recurse=False))
            n_params_total = sum(p.numel() for p in mod.parameters())
        except Exception:
            n_params = n_params_total = -1
        logger.info(
            "  %s = %s  (own_params=%d, total_params=%d)",
            path, type(mod).__name__, n_params, n_params_total,
        )
    if len(found) > 12:
        logger.info("  ... and %d more nn.Modules", len(found) - 12)


def _apply_hqq_post_load_quant(pipeline, nbits: int, group_size: int = 64) -> tuple[int, list]:
    """Walk the pipeline's prompt_encoder subtree and replace every
    ``nn.Linear`` with an ``HQQLinear``. Returns (count_replaced, nn_module_roots).

    Recurses through arbitrary wrapper objects (LTX-2's PromptEncoder is a
    plain Python class holding the Gemma nn.Module deeper down), so on-disk
    BF16 weights stay compatible with LTX-2's custom loader but runtime
    Linear layers get quantized. ``compute_dtype=bfloat16`` keeps the DiT
    interface unchanged.
    """
    from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
    import torch.nn as nn

    prompt_encoder = getattr(pipeline, "prompt_encoder", None)
    if prompt_encoder is None:
        raise RuntimeError(
            "LTX-2 pipeline has no `prompt_encoder` attribute — "
            "cannot locate the Gemma quant target"
        )

    quant_config = BaseQuantizeConfig(
        nbits=nbits,
        group_size=group_size,
        quant_zero=False,
        quant_scale=False,
        offload_meta=False,
        view_as_float=False,
    )

    # Locate every nn.Module root reachable from prompt_encoder. Quantize
    # the Linear layers inside each. (Most pipelines have a single big
    # Gemma root; some may also expose a small head/projection sibling.)
    nn_roots: list[tuple[str, nn.Module]] = list(_find_nn_modules_in(prompt_encoder))
    if not nn_roots:
        raise RuntimeError(
            "Post-load quant: no nn.Module found anywhere under prompt_encoder. "
            "LTX-2's PromptEncoder wrapper does not expose its Gemma model via "
            "any attribute reachable by vars(). See boot log structure dump."
        )

    # De-duplicate roots: if module A is a descendant of module B via
    # named_modules(), we only need to walk B. Keep the roots with the
    # shortest path that cover everything.
    root_ids: set[int] = set()
    unique_roots: list[tuple[str, nn.Module]] = []
    for path, root in nn_roots:
        if id(root) in root_ids:
            continue
        # Skip roots that are contained inside an already-picked root.
        if any(
            id(root) in {id(sub) for sub in picked.modules()}
            for _, picked in unique_roots
        ):
            continue
        unique_roots.append((path, root))
        root_ids.add(id(root))

    replaced: list[str] = []

    def _quant_linears(module: nn.Module, path: str) -> None:
        for name, child in list(module.named_children()):
            child_path = f"{path}.{name}" if path else name
            if isinstance(child, nn.Linear) and not isinstance(child, HQQLinear):
                hqq_layer = HQQLinear(
                    child,
                    quant_config=quant_config,
                    compute_dtype=torch.bfloat16,
                    device=child.weight.device,
                    initialize=True,
                    del_orig=True,
                )
                setattr(module, name, hqq_layer)
                replaced.append(child_path)
            else:
                _quant_linears(child, child_path)

    for path, root in unique_roots:
        logger.info("Quantizing Linears under %s (%s)", path, type(root).__name__)
        _quant_linears(root, path)

    return len(replaced), [root for _, root in unique_roots]


def _install_first_encode_magnitude_hook(nn_roots) -> None:
    """Register a one-shot forward hook on the largest nn.Module root we
    identified (that's Gemma). Logs the magnitude of the first forward's
    output — catches near-zero silent-failure mode that would otherwise
    only surface as a flat-black MP4.
    """
    if not nn_roots:
        logger.warning("Magnitude hook: no nn.Module roots to attach to")
        return

    # Largest root by total parameter count ≈ Gemma (12B params vs
    # anything else in the prompt encoder).
    def _param_count(m):
        try:
            return sum(p.numel() for p in m.parameters())
        except Exception:
            return 0

    gemma_candidate = max(nn_roots, key=_param_count)
    logger.info(
        "Magnitude hook target: %s (%d params)",
        type(gemma_candidate).__name__, _param_count(gemma_candidate),
    )

    state = {"logged": False}

    def _collect_tensors(obj):
        if torch.is_tensor(obj):
            yield obj
        elif isinstance(obj, (tuple, list)):
            for item in obj:
                yield from _collect_tensors(item)
        elif isinstance(obj, dict):
            for v in obj.values():
                yield from _collect_tensors(v)

    def hook(module, args, output):
        if state["logged"]:
            return
        state["logged"] = True
        saw_tensor = False
        for i, tensor in enumerate(_collect_tensors(output)):
            if not tensor.is_floating_point():
                continue
            saw_tensor = True
            try:
                max_abs = tensor.detach().abs().max().item()
                mean_abs = tensor.detach().abs().mean().item()
            except Exception:
                logger.exception("Magnitude hook: failed to read output[%d]", i)
                continue
            logger.info(
                "First Gemma encode output[%d]: shape=%s dtype=%s max_abs=%.4g mean_abs=%.4g",
                i, tuple(tensor.shape), tensor.dtype, max_abs, mean_abs,
            )
            if max_abs < 1e-3:
                logger.error(
                    "FIRST-ENCODE OUTPUT[%d] IS NEAR-ZERO (max_abs=%.2e) — "
                    "quantization likely produced garbage; expect black video",
                    i, max_abs,
                )
        if not saw_tensor:
            logger.warning("Magnitude hook: no floating-point tensors in output")

    gemma_candidate.register_forward_hook(hook)


def _round_to(value: int, divisor: int) -> int:
    return (value // divisor) * divisor


def _round_frames(n: int) -> int:
    return ((n - 1) // 8) * 8 + 1


class LTXVideoGenerator:
    """Initialises the LTX-2.3 two-stage pipeline and runs inference."""

    def __init__(self, model_dir: str = "/models") -> None:
        checkpoint_path = os.path.join(model_dir, "ltx-2.3-22b-dev.safetensors")
        spatial_upsampler_path = os.path.join(
            model_dir, "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        )
        distilled_lora_path = os.path.join(
            model_dir, "ltx-2.3-22b-distilled-lora-384-1.1.safetensors"
        )
        gemma_root = os.path.join(model_dir, "gemma-3-12b-it-qat-q4_0-unquantized")

        self._post_load_quant = _resolve_post_load_quant()
        logger.info("Gemma post-load quant: %s", self._post_load_quant)

        # Pure-GPU mode: post-load hqq4/hqq8 shrinks Gemma to 7–13 GB, so
        # the CPU<->GPU swapping machinery (StateDictRegistry + per-layer
        # streaming) becomes unnecessary on any 40 GB+ GPU. With BF16 Gemma
        # (26 GB), pure-GPU mode needs 80 GB+ and we stick with the
        # streaming fallback for safety on smaller cards.
        gpu_vram_gb = (
            torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0
        )
        self._gpu_vram_gb = gpu_vram_gb
        quant_reduces_vram = self._post_load_quant != "none"
        self._pure_gpu_mode = quant_reduces_vram and gpu_vram_gb >= 40
        if self._pure_gpu_mode:
            logger.info(
                "Pure-GPU mode ENABLED: post-load quant=%s + %.0f GB VRAM, "
                "registry and layer streaming will be bypassed",
                self._post_load_quant, gpu_vram_gb,
            )

        logger.info("Initializing LTX-2.3 pipeline ...")
        self._log_vram("before pipeline init")

        from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
        from ltx_pipelines.utils.media_io import encode_video

        self._encode_video = encode_video

        # FP8 quantization — downcasts BF16 weights to FP8 on the fly,
        # upcasts back to BF16 during forward. ~40% VRAM reduction.
        try:
            from ltx_core.quantization import QuantizationPolicy
            quantization = QuantizationPolicy.fp8_cast()
            logger.info("Using FP8 quantization (fp8_cast)")
        except ImportError:
            quantization = None
            logger.warning("QuantizationPolicy not available")

        # CPU weight caching — only one model on GPU at a time.
        # Skipped in pure-GPU mode since all models fit resident.
        registry = None
        if self._pure_gpu_mode:
            logger.info("Pure-GPU mode: skipping StateDictRegistry")
        else:
            try:
                from ltx_core.loader import StateDictRegistry
                registry = StateDictRegistry()
                logger.info("Using StateDictRegistry (CPU weight caching)")
            except ImportError:
                logger.warning("StateDictRegistry not available")

        # Distilled LoRA
        from ltx_core.loader import (
            LTXV_LORA_COMFY_RENAMING_MAP,
            LoraPathStrengthAndSDOps,
        )
        distilled_lora = [
            LoraPathStrengthAndSDOps(
                path=distilled_lora_path,
                strength=0.8,
                sd_ops=LTXV_LORA_COMFY_RENAMING_MAP,
            )
        ]

        # Build pipeline — pass registry and quantization at construction time
        # so VRAM is managed correctly from the start
        pipeline_kwargs = dict(
            checkpoint_path=checkpoint_path,
            distilled_lora=distilled_lora,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root=gemma_root,
            loras=[],
        )
        if quantization is not None:
            pipeline_kwargs["quantization"] = quantization
        if registry is not None:
            pipeline_kwargs["registry"] = registry

        self._pipeline = TI2VidTwoStagesPipeline(**pipeline_kwargs)
        self._log_vram("after pipeline init")

        # Diagnostic: dump the nn.Module roots reachable from prompt_encoder
        # so we know where Gemma sits inside LTX-2's custom wrapper. Logged
        # once at init; useful for quant-path debugging.
        _dump_prompt_encoder_structure(self._pipeline)

        # Post-load Gemma quantization. Runs after BF16 weights are in
        # place; replaces nn.Linear → HQQLinear in the prompt-encoder
        # subtree. Guarded with a magnitude hook that logs the first
        # encoded output so a broken quant can't silently produce black
        # video the way on-disk W4A16 did (see docs/gemma-compat-checklist.md).
        nn_roots_for_hook: list = []
        if self._post_load_quant != "none":
            nbits = {"hqq4": 4, "hqq8": 8}[self._post_load_quant]
            logger.info("Applying post-load HQQ quant (nbits=%d) to prompt encoder ...", nbits)
            try:
                n_replaced, nn_roots_for_hook = _apply_hqq_post_load_quant(
                    self._pipeline, nbits=nbits,
                )
                logger.info(
                    "Post-load HQQ%d quant applied: %d Linear layers replaced",
                    nbits, n_replaced,
                )
                if n_replaced == 0:
                    logger.error(
                        "Post-load quant replaced 0 Linear layers — check the "
                        "structure dump above. Gemma will still run but VRAM win "
                        "was not realised."
                    )
                self._log_vram("after post-load quant")
            except Exception:
                logger.exception(
                    "Post-load HQQ quant failed; falling back to BF16 Gemma "
                    "(pure-GPU mode may OOM on <80 GB GPUs)"
                )
                self._post_load_quant = "none"
                self._pure_gpu_mode = False
        else:
            # Even without quant, attach the magnitude hook so BF16 runs get
            # observability on Gemma output magnitude — helps compare against
            # a quantized run later.
            nn_roots_for_hook = [m for _, m in _find_nn_modules_in(self._pipeline.prompt_encoder)]

        # One-shot magnitude check on the first Gemma encode. Fires only
        # once; logs ERROR if output is near-zero (silent-failure guard).
        _install_first_encode_magnitude_hook(nn_roots_for_hook)

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
        width = _round_to(width, 64)
        height = _round_to(height, 64)
        num_frames = _round_frames(num_frames)
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

            # Run pipeline
            start_time = time.time()

            # Streaming: builds models on CPU, streams layers to GPU on demand.
            # Required for <48GB GPUs — without it, LoRA fusion OOMs because
            # the 22B transformer + LoRA deltas exceed GPU memory.
            # With streaming, build+fuse happens on CPU (plenty of RAM),
            # then only 2-3 layers live on GPU at any time during inference.
            # max_batch_size=4 batches guidance passes to reduce PCIe round-trips.
            #
            # Pure-GPU mode (post-load hqq4/hqq8 + >=40 GB VRAM) bypasses
            # streaming entirely: DiT + Gemma + VAE + upscaler all stay resident.
            if self._pure_gpu_mode:
                streaming = None
                logger.info(
                    "Job %s: pure-GPU mode — streaming disabled (%.0f GB VRAM)",
                    job_id, self._gpu_vram_gb,
                )
            else:
                streaming = 2 if self._gpu_vram_gb < 40 else None
                if streaming:
                    logger.info(
                        "Job %s: streaming enabled (GPU VRAM: %.0f GB < 40 GB)",
                        job_id, self._gpu_vram_gb,
                    )

            call_kwargs = dict(
                prompt=prompt, negative_prompt=negative_prompt, seed=seed,
                height=height, width=width, num_frames=num_frames,
                frame_rate=frame_rate, num_inference_steps=num_inference_steps,
                images=[],
                streaming_prefetch_count=streaming,
                max_batch_size=4 if streaming else 1,
            )
            if video_guider_params is not None:
                call_kwargs["video_guider_params"] = video_guider_params
            if audio_guider_params is not None:
                call_kwargs["audio_guider_params"] = audio_guider_params
            if tiling_config is not None:
                call_kwargs["tiling_config"] = tiling_config

            result = self._pipeline(**call_kwargs)
            video, audio = result if isinstance(result, tuple) else (result, None)
            generation_time = time.time() - start_time
            logger.info("Job %s: generation took %.1fs", job_id, generation_time)

            # Encode video
            output_filename = f"ltx_{job_id}.mp4"
            output_path = os.path.join(tempfile.gettempdir(), output_filename)
            encode_kwargs = dict(video=video, fps=int(frame_rate), output_path=output_path)
            if audio is not None:
                encode_kwargs["audio"] = audio
            if video_chunks_number is not None:
                encode_kwargs["video_chunks_number"] = video_chunks_number
            self._encode_video(**encode_kwargs)

            # Black-MP4 guard: an h264-encoded flat-color 1024x1536 clip of
            # 5+ seconds is typically <200 KB. Real content is >500 KB.
            # This is the second line of defence if the magnitude hook
            # above missed a silent quant failure.
            try:
                mp4_bytes = os.path.getsize(output_path)
            except OSError:
                mp4_bytes = -1
            if mp4_bytes >= 0 and mp4_bytes < 200_000 and num_frames >= 25:
                logger.error(
                    "Job %s: output MP4 is suspiciously small (%d bytes) — "
                    "likely a flat-black/constant-colour video. Check Gemma "
                    "quant + pipeline numerics.",
                    job_id, mp4_bytes,
                )
            else:
                logger.info("Job %s: output MP4 size=%d bytes", job_id, mp4_bytes)

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
