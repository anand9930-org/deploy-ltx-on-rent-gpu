"""ComfyUI runtime bootstrap for the ``triple_stages_comfyui`` pipeline.

That pipeline runs the real ComfyUI **core** node classes (``comfy_extras.*``,
``nodes``; cloned into the image at ``COMFYUI_PATH`` — see the Dockerfile). It
does NOT use ComfyUI-LTXVideo's custom nodes — their modules use relative
imports that need the full ``nodes.init_extra_nodes()`` runtime, and the two
nodes the workflow used from there have core equivalents
(``LTXVImgToVideoInplace`` and ``round()``; see the graph module). Those core
node classes still import a fair amount of ComfyUI (``comfy.cli_args``,
``comfy.model_management``, ``folder_paths``, …), so before any of them can be
imported or called we have to (1) tell ComfyUI to parse a *clean* argv — it
would otherwise choke on the ``bentoml serve …`` args — and (2) point
``folder_paths`` at our ``/models`` tree and at temp input/output dirs
(``LoadImage`` reads from the input dir).

We do NOT run ComfyUI's ``nodes.init_extra_nodes()`` / ``PromptServer`` / its
asyncio loop — the node classes are imported and called directly (see
``src/pipeline/triple_stages_comfyui_graph.py``), which needs only the above.

Owns:

* :func:`bootstrap_once` — idempotent; call from the pipeline ``__init__``.
* :func:`unload_models` — free ComfyUI's resident weights (called by
  ``LTXVideoGenerator._ensure_mode`` when switching away to another pipeline,
  so the ``ltx_pipelines`` path gets the GPU back). No-op if never bootstrapped.
* :func:`reset_to_clean_gpu` — gc + ``torch.cuda.empty_cache()``; call right
  before constructing the ComfyUI pipeline.
* :func:`input_dir` — the temp dir ``LoadImage`` reads from (stage images here).

No ComfyUI imports at module level (only inside the functions), so importing
this module is free even on a box without ComfyUI installed.
"""

from __future__ import annotations

import gc
import logging
import os
import sys
import tempfile

logger = logging.getLogger(__name__)

# Process-global, one-shot bootstrap state (ComfyUI's folder_paths / cli_args /
# model cache are all process-global).
_bootstrapped = False
_input_dir: str | None = None

# ``folder_paths`` categories the 3mljpp workflow's loader nodes resolve
# against. CheckpointLoaderSimple / LoraLoaderModelOnly / LTXVAudioVAELoader →
# "checkpoints"+"loras"; LatentUpscaleModelLoader → "latent_upscale_models";
# LTXAVTextEncoderLoader → "text_encoders" plus an "embeddings" dir for
# comfy.sd.load_clip's embedding_directory argument.
_MODEL_FOLDERS_AT_ROOT = ("checkpoints", "loras", "latent_upscale_models")


def _patch_comfyui_vae_inplace(comfyui_path: str) -> None:
    """Workaround for ComfyUI PR #13028 (commit 735a046, 2026-03-18) which
    introduced in-place ops on a tensor returned from ComfyUI's internal
    ``torch.inference_mode()`` block at ``comfy/sd.py`` line ~95. On torch
    2.10+ that combination raises::

        RuntimeError: Inplace update to inference tensor outside InferenceMode
                      is not allowed.

    during VAE decode. We rewrite the expression to its non-in-place pre-#13028
    form (``torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)``). The ~3 GB peak-RAM
    regression per PR #13028's own profiling is negligible on the 96 GB
    Blackwell pod. Must run BEFORE any ``comfy.*`` import — Python caches
    modules in ``sys.modules`` after first import, so modifying the .py file
    afterward has no effect on this process. Idempotent: a second call finds
    the BAD pattern already gone and returns silently.
    """
    sd_py = os.path.join(comfyui_path, "comfy", "sd.py")
    if not os.path.isfile(sd_py):
        return  # let the natural import-time error surface
    bad = "image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)"
    good = "torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)"
    with open(sd_py, "r", encoding="utf-8") as f:
        src = f.read()
    if bad not in src:
        # Either already patched or the pattern shifted upstream. We do NOT
        # rely on this patch being the only protection: see the instance-level
        # `vae.process_output = ...` reassignment in
        # src/pipeline/triple_stages_comfyui_graph.py, added after a 2026-05-18
        # incident where this file-content patch silently no-op'd on a fresh
        # GPU Hub pod. Log loudly here so we never silently fall back again.
        logger.warning(
            "comfy/sd.py at %s does not contain the expected in-place pattern; "
            "skipping file-content patch (instance-level patch in "
            "triple_stages_comfyui_graph.py is load-bearing)",
            sd_py,
        )
        return
    with open(sd_py, "w", encoding="utf-8") as f:
        f.write(src.replace(bad, good))
    logger.info("Patched %s: in-place VAE post-process -> non-in-place clamp", sd_py)


def bootstrap_once(comfyui_path: str, model_dir: str, cpu_only: bool = False) -> None:
    """Make the cloned ComfyUI checkout importable and point ``folder_paths``
    at the model dir + temp input/output dirs. Idempotent.

    The triple-stages-comfyui-graph pipeline uses only ComfyUI **core** nodes
    (``comfy_extras.nodes_lt*`` etc., absolute imports), so this is all the
    "runtime" needed — no ``nodes.init_extra_nodes()`` / ``PromptServer`` / asyncio.

    ``cpu_only``: force ComfyUI into CPU mode (``comfy.cli_args.args.cpu = True``)
    before anything imports ``comfy.model_management`` — which, at *import* time,
    runs ``torch.cuda.current_device()`` and raises on a GPU-less box unless
    ``args.cpu`` is set. Only the build-time node-contract check in the Dockerfile
    passes ``cpu_only=True`` (the GHCR builder has no GPU); at pod boot we want the
    GPU, so the default is ``False``.
    """
    global _bootstrapped, _input_dir
    if _bootstrapped:
        return

    if not os.path.isdir(comfyui_path):
        raise RuntimeError(
            f"COMFYUI_PATH={comfyui_path!r} is not a directory — the ComfyUI "
            "checkout is missing from the image (see Dockerfile)."
        )

    # Pre-comfy torch knobs (no-op on cpu_only build-time check).
    # - cudnn.benchmark: autotunes the conv algorithm per input shape. Our
    #   3-stage cascade runs identical shapes on every request, so requests
    #   2..N reuse the cached plans. First-request cost is a one-time
    #   ~100-400 ms autotune per unique shape; subsequent calls hit cache.
    #   cudnn picks among numerically-equivalent algorithms (selection by
    #   wall-clock, not output), so the chosen kernel is deterministic at
    #   fixed shape — no quality regression. Avoided via `--fast autotune`
    #   because ComfyUI issue #9779 reports a 123 s first-run stall on long
    #   workflows under that umbrella; setting the torch flag directly is
    #   surgical and won't drift if upstream rebundles `autotune`.
    # - set_float32_matmul_precision("high"): no-op on the BF16/FP8 cascade
    #   today (no FP32 matmul nodes), but cheap insurance + documents intent
    #   if a future node introduces an FP32 path. Already the torch 2.11
    #   default on Ampere+; explicit setting equals status quo.
    if not cpu_only:
        import torch

        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    # Patch comfy/sd.py BEFORE any comfy.* import (see helper docstring for why).
    _patch_comfyui_vae_inplace(comfyui_path)

    # Put the cloned ComfyUI checkout on sys.path BEFORE any comfy import.
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    # ComfyUI's cli_args parses sys.argv at first import; under `bentoml serve`
    # that's the wrong argv. Feed a clean one for the duration of the bootstrap.
    #
    # On the runtime path (cpu_only=False), inject:
    #   --gpu-only              Pin the entire pipeline (UNet, VAE, text encoder)
    #                           to GPU and disable ComfyUI's dynamic model
    #                           offload. Strictly stronger than `--highvram`:
    #                           both map to VRAMState.HIGH_VRAM in
    #                           comfy/model_management.py (lines 423-429 at SHA
    #                           64b8457f), but `--gpu-only` additionally forces
    #                           the Gemma 12B text encoder / CLIP to stay GPU-
    #                           resident across requests instead of getting
    #                           bounced back to CPU by the smart-memory path.
    #                           Single-pipeline BentoML worker on a 96 GB pod —
    #                           zero risk, removes one CPU↔GPU round-trip per
    #                           request.
    #   --reserve-vram 0.5      Headroom for transient allocations. ComfyUI's
    #                           Linux default is 400 MB; we previously carried
    #                           2 GB without measured justification. On a 96 GB
    #                           pod with ~60 GB resident cascade footprint,
    #                           dropping to 0.5 GB returns 1.5 GB of transient
    #                           headroom and never approaches the residual
    #                           ~30 GB free pool.
    #
    # `--fast fp8_matrix_mult` was tried in Phase 1.6d (commit e27c941) for a
    # ~16% speedup on the FP8 matmul path, then rolled back in Phase 1.6g after
    # live verification showed it regresses I2V image conditioning at frame
    # counts >= 241. The flag routes activations through torch._scaled_mm with
    # a per-tensor FP8 cast (E4M3fn, range ~[6e-8, 448]); at longer sequence
    # lengths the activation dynamic range saturates and cross-attention loses
    # the image-conditioning signal — the model "free-runs" from the prompt.
    # The proper future-work FP8 path is cu130 + ComfyUI's `comfy_kitchen` CUDA
    # backend (gated at `torch.version.cuda >= (13,)` in comfy/quant_ops.py),
    # which has per-module FP8 enable lists that skip cross-attention. Whether
    # the kitchen path is already routing FP8 matmuls on our cu130 stack (i.e.
    # whether the LTX-2.3 safetensors carries `quant_config` metadata) is a
    # pending probe — see plan mutable-orbiting-hickey.md Phase 2.
    #
    # The build-time node-contract check (cpu_only=True) doesn't load weights,
    # so these flags are no-ops there — keep the clean argv to avoid surprises.
    saved_argv = sys.argv
    sys.argv = ["comfyui"] if cpu_only else ["comfyui", "--gpu-only", "--reserve-vram", "0.5"]
    try:
        import comfy.options  # must precede any other comfy import (so cli_args parses our clean argv)

        comfy.options.enable_args_parsing()

        if cpu_only:
            # Set the parsed flag programmatically *before* folder_paths / any
            # node module pulls in comfy.model_management (which probes the GPU
            # at import time via torch.cuda.current_device() unless args.cpu).
            import comfy.cli_args

            comfy.cli_args.args.cpu = True

        import folder_paths

        for category in _MODEL_FOLDERS_AT_ROOT:
            folder_paths.add_model_folder_path(category, model_dir)
        # The Gemma single-file text encoder comes from Comfy-Org/ltx-2 via
        # src/download_models.py, which lands it under <model_dir>/split_files/text_encoders/.
        # Register that and a plain <model_dir>/text_encoders/ so either layout resolves.
        for te_dir in (
            os.path.join(model_dir, "text_encoders"),
            os.path.join(model_dir, "split_files", "text_encoders"),
        ):
            os.makedirs(te_dir, exist_ok=True)
            folder_paths.add_model_folder_path("text_encoders", te_dir)
        emb_dir = os.path.join(model_dir, "embeddings")
        os.makedirs(emb_dir, exist_ok=True)
        folder_paths.add_model_folder_path("embeddings", emb_dir)

        _input_dir = tempfile.mkdtemp(prefix="comfyui_in_")
        folder_paths.set_input_directory(_input_dir)
        folder_paths.set_output_directory(tempfile.mkdtemp(prefix="comfyui_out_"))
        if hasattr(folder_paths, "set_temp_directory"):
            folder_paths.set_temp_directory(tempfile.mkdtemp(prefix="comfyui_tmp_"))
    finally:
        sys.argv = saved_argv

    _bootstrapped = True
    logger.info(
        "ComfyUI runtime bootstrapped: path=%s, models=%s, input_dir=%s, cpu_only=%s",
        comfyui_path, model_dir, _input_dir, cpu_only,
    )


def is_bootstrapped() -> bool:
    return _bootstrapped


def input_dir() -> str:
    """The ComfyUI input dir (``LoadImage`` reads from here). Bootstrap first."""
    if _input_dir is None:
        raise RuntimeError("ComfyUI runtime not bootstrapped yet.")
    return _input_dir


def unload_models() -> None:
    """Free ComfyUI's resident model weights so another pipeline can use the
    GPU. No-op if the runtime was never bootstrapped; never raises."""
    if not _bootstrapped:
        return
    try:
        import comfy.model_management

        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
    except Exception:  # noqa: BLE001 — diagnostic, never fail a mode switch
        logger.warning("ComfyUI unload_models failed", exc_info=True)
    gc.collect()


def reset_to_clean_gpu() -> None:
    """gc + ``torch.cuda.empty_cache()`` — call right before constructing the
    ComfyUI pipeline so it starts from a clean GPU."""
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
