"""Anti-corruption layer for the upstream Lightricks/LTX-2 packages.

This is the ONE file in this codebase that imports directly from
``ltx_core`` and ``ltx_pipelines``. Everything else in ``src/`` consumes
upstream symbols via ``from src.upstream import X``.

Why a single chokepoint:

  * Upstream ``main`` moves often. When a class is renamed or a module
    path changes, the diff to absorb that rename is bounded to this file
    instead of fanning out across the codebase.
  * The pinned ``LTX2_UPSTREAM_SHA`` in ``Dockerfile`` plus this file
    define our exact contract surface with upstream — what's listed here
    is what we depend on; anything not listed we're free of.

Three import styles are used, picked deliberately per symbol.

Style 1 — class / constant re-exports
    ``from upstream.module import Cls`` is fine when consumers either
    instantiate ``Cls`` or patch one of its METHODS. Patching a class
    method mutates the class object in place, so every previously-imported
    reference to that class sees the patched method automatically.

Style 2 — module references (``import upstream.module as ltx_X``)
    Required when an override module reassigns a MODULE-LEVEL attribute
    on the upstream module (e.g. ``compile_override`` reassigns
    ``ltx_core.model.transformer.compiling.compile_transformer`` and
    ``COMPILE_TRANSFORMER``). With ``from X import Y``, the local name
    ``Y`` is bound at import time and a later ``X.Y = ...`` reassignment
    is invisible to it. Going through a module reference (``X.Y``) reads
    the attribute lazily at call time, so the patch is visible.

Style 3 — optional symbols (try/except + ``HAS_*`` flag)
    Used for symbols that may legitimately not exist on a given pinned
    upstream SHA. Importers gate on the ``HAS_*`` flag.

If any Style-1 or Style-2 import fails, this module raises at import
time. That failure surfaces in the Docker build (see the
``python -c "import src.upstream"`` line in the Dockerfile) rather than
30 minutes into a generation as a cryptic ``AttributeError`` deep in a
pipeline call.
"""

# === Style 2 — module references (load-bearing for module-level patches
# in src/compile_override.py and attribute access in src/pipeline.py) ===

from ltx_core.model.transformer import compiling as ltx_compiling
from ltx_core.model.transformer import attention as ltx_attention
from ltx_core.model.transformer import model_configurator as ltx_model_configurator

try:
    from ltx_pipelines.utils import blocks as ltx_blocks
except ImportError:  # pragma: no cover — defensive; ltx_pipelines should be installed
    ltx_blocks = None


# === Style 1 — class / constant re-exports (safe under method-level
# monkey-patching; class objects are mutated in place) ===

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


# === Style 3 — optional symbols (gated by HAS_* flag) ===

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
