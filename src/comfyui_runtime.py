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


def bootstrap_once(comfyui_path: str, model_dir: str) -> None:
    """Make the cloned ComfyUI checkout importable and point ``folder_paths``
    at the model dir + temp input/output dirs. Idempotent.

    The triple-stages-comfyui-graph pipeline uses only ComfyUI **core** nodes
    (``comfy_extras.nodes_lt*`` etc., absolute imports), so this is all the
    "runtime" needed — no ``nodes.init_extra_nodes()`` / ``PromptServer`` / asyncio.
    """
    global _bootstrapped, _input_dir
    if _bootstrapped:
        return

    if not os.path.isdir(comfyui_path):
        raise RuntimeError(
            f"COMFYUI_PATH={comfyui_path!r} is not a directory — the ComfyUI "
            "checkout is missing from the image (see Dockerfile)."
        )

    # Put the cloned ComfyUI checkout on sys.path BEFORE any comfy import.
    if comfyui_path not in sys.path:
        sys.path.insert(0, comfyui_path)

    # ComfyUI's cli_args parses sys.argv at first import; under `bentoml serve`
    # that's the wrong argv. Feed a clean one for the duration of the bootstrap.
    saved_argv = sys.argv
    sys.argv = ["comfyui"]
    try:
        import comfy.options  # must precede any other comfy import (so cli_args parses our clean argv)

        comfy.options.enable_args_parsing()

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
        "ComfyUI runtime bootstrapped: path=%s, models=%s, input_dir=%s",
        comfyui_path, model_dir, _input_dir,
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
