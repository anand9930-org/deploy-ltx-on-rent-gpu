"""Anti-corruption layer for ``ltx_core`` / ``ltx_pipelines``.

The single chokepoint: every upstream symbol in this codebase is imported
here, then re-exported via ``from src.upstream import X``. When upstream
renames a path the diff is bounded to this file. The Dockerfile runs
``python -c "import src.upstream"`` so a rename fails the build, not a
generation.

Three import styles, each load-bearing:

  Style 1 — direct class/const re-export (``from X import Cls``).
            Safe when consumers instantiate or method-patch ``Cls``;
            method patches mutate the class in place.
  Style 2 — module reference (``import X as ltx_X``). REQUIRED when an
            override reassigns a module-level attribute (e.g.
            ``compile_override`` reassigns ``ltx_compiling.compile_transformer``).
            ``from X import Y`` would bind ``Y`` at import time and miss
            later reassignments.
  Style 3 — optional symbol (try/except + ``HAS_*`` flag). For symbols
            that may legitimately be absent on a given pinned SHA.
"""

# === Style 2 — module references (module-level patch targets) ===

from ltx_core.model.transformer import compiling as ltx_compiling
from ltx_core.model.transformer import attention as ltx_attention
from ltx_core.model.transformer import model_configurator as ltx_model_configurator

try:
    from ltx_pipelines.utils import blocks as ltx_blocks
except ImportError:  # pragma: no cover — defensive; ltx_pipelines should be installed
    ltx_blocks = None


# === Style 1 — class / constant re-exports ===

from ltx_core.model.transformer.attention import AttentionFunction, FlashAttention3
from ltx_core.model.transformer import LTXModel
from ltx_core.loader import (
    LTXV_LORA_COMFY_RENAMING_MAP,
    LoraPathStrengthAndSDOps,
    StateDictRegistry,
)
from ltx_core.loader.module_ops import ModuleOps
from ltx_core.loader.sd_ops import KeyValueOperationResult, SDOps
from ltx_core.quantization import QuantizationPolicy
from ltx_core.quantization.fp8_scaled_mm import (
    FP8_PREPARE_MODULE_OPS,
    FP8_TRANSPOSE_SD_OPS,
    FP8Linear,
    _linear_to_fp8linear,
)
from ltx_pipelines.ti2vid_two_stages import TI2VidTwoStagesPipeline
from ltx_pipelines.ic_lora import ICLoraPipeline
from ltx_pipelines.utils.blocks import DiffusionStage
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import OffloadMode
from ltx_pipelines.utils.args import ImageConditioningInput


# === Style 3 — optional symbols (gated by HAS_*) ===

try:
    from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
    HAS_TILING = True
except ImportError:
    TilingConfig = None
    get_video_chunks_number = None
    HAS_TILING = False

try:
    from ltx_core.components.guiders import MultiModalGuiderParams
    HAS_GUIDERS = True
except ImportError:
    MultiModalGuiderParams = None
    HAS_GUIDERS = False


# Explicit public surface — declares every re-export as intentional so
# ruff F401 / mypy don't flag them and downstream `from src.upstream import X`
# is the documented contract.
__all__ = [
    # Style 2 — module references
    "ltx_compiling", "ltx_attention", "ltx_model_configurator", "ltx_blocks",
    # Style 1 — class / constant re-exports
    "AttentionFunction", "FlashAttention3", "LTXModel",
    "LTXV_LORA_COMFY_RENAMING_MAP", "LoraPathStrengthAndSDOps", "StateDictRegistry",
    "ModuleOps", "KeyValueOperationResult", "SDOps",
    "QuantizationPolicy",
    "FP8_PREPARE_MODULE_OPS", "FP8_TRANSPOSE_SD_OPS", "FP8Linear", "_linear_to_fp8linear",
    "TI2VidTwoStagesPipeline", "ICLoraPipeline", "DiffusionStage",
    "encode_video", "OffloadMode", "ImageConditioningInput",
    # Style 3 — optional symbols (gated by HAS_* flags)
    "TilingConfig", "get_video_chunks_number", "HAS_TILING",
    "MultiModalGuiderParams", "HAS_GUIDERS",
]
