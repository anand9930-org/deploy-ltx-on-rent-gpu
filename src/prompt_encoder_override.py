"""Filter unknown kwargs on upstream callables invoked by the vendored
triple-stages pipeline.

``src/vendor/ti2vid_triple_stages.py`` is byte-identical to the eisneim
fork (April 2026), authored against a newer LTX-2 SHA than the one we pin
in ``Dockerfile``. The skew surfaces as e.g.::

    TypeError: PromptEncoder.__call__() got an unexpected keyword
    argument 'streaming_prefetch_count'

Editing the vendored file is forbidden (treat as upstream — see
``src/vendor/README.md``); bumping the upstream SHA would ripple into
T2V/I2V/V2V. Instead we wrap the affected ``__call__`` methods at boot
to drop kwargs whose names are absent from the bound signature. The
dropped kwargs are layer-streaming hints — only meaningful with offload
enabled, which triple-stages forbids (``OffloadMode.NONE``).

Targets split by call timing — class-method patches mutate the class
in place (visible after vendored import; called from
``LTXVideoGenerator.__init__``); free-function patches must run BEFORE
the vendored ``from X import Y`` runs, otherwise the vendored module
binds the un-patched function locally at its own import time.

  Method-level (post-import OK):
    * ``PromptEncoder.__call__``    — drops ``streaming_prefetch_count``
    * ``DiffusionStage.__call__``   — drops ``streaming_prefetch_count``

  Function-level (pre-import REQUIRED — wired in ``src/upstream.py``):
    * ``ltx_pipelines.utils.helpers.assert_resolution`` — drops ``is_two_stage``
    * ``ltx_pipelines.utils.helpers.combined_image_conditionings``
      — drops ``preprocessed_images``

  Watch-list (NOT patched — filtering would silently break data flow,
  so failures here need a rename map instead of a kwarg drop):
    * ``ModalitySpec(initial_latent=...)`` — stage 2/3 latent chain
    * ``LTX2Scheduler().execute(steps=...)`` — sigma schedule kwarg
    * ``VideoPixelShape(batch=..., fps=..., ...)`` — dataclass fields

All patches are sig-aware: no-op when upstream already accepts the kwarg
or declares ``**kwargs``. Idempotent via per-callable attribute markers
plus module-level flags — safe to call repeatedly.
"""

import functools
import inspect
import logging

logger = logging.getLogger(__name__)

_applied = False
_helper_patches_applied = False


def enable_triple_stages_kwarg_filter() -> None:
    """Wrap upstream callables that the vendored triple-stages pipeline
    invokes with newer-API kwargs. Idempotent."""
    global _applied
    if _applied:
        return

    from src.upstream import ltx_blocks
    if ltx_blocks is None:  # pragma: no cover — defensive
        logger.warning(
            "ltx_blocks unavailable; triple-stages kwarg filter not installed"
        )
        return

    for cls_name in ("PromptEncoder", "DiffusionStage"):
        cls = getattr(ltx_blocks, cls_name, None)
        if cls is None:  # pragma: no cover — upstream rename guard
            logger.warning(
                "ltx_blocks.%s missing; cannot install kwarg filter", cls_name
            )
            continue
        _wrap_with_kwarg_filter(cls)

    _applied = True


def _wrap_with_kwarg_filter(cls) -> None:
    """Wrap ``cls.__call__`` to silently drop kwargs not in its signature.

    No-op when the existing signature already accepts the kwargs the
    vendored fork passes (i.e. upstream caught up) or when the method
    declares ``**kwargs``."""
    original = cls.__call__
    if getattr(original, "_kwarg_filtered", False):
        return  # already wrapped

    sig = inspect.signature(original)
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kw:
        logger.info(
            "%s.__call__ already accepts **kwargs; kwarg filter not needed",
            cls.__name__,
        )
        return

    accepted = {
        name for name, p in sig.parameters.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    @functools.wraps(original)
    def wrapper(self, *args, **kwargs):
        unknown = [k for k in kwargs if k not in accepted]
        if unknown:
            for k in unknown:
                kwargs.pop(k)
            if not getattr(wrapper, "_warned", False):
                logger.info(
                    "%s.__call__ kwarg filter active — dropping %s on first "
                    "call (vendored triple-stages passes newer-API kwargs "
                    "absent from pinned upstream signature)",
                    cls.__name__, unknown,
                )
                wrapper._warned = True
        return original(self, *args, **kwargs)

    wrapper._kwarg_filtered = True
    cls.__call__ = wrapper
    logger.info(
        "Installed kwarg filter on %s.__call__ (signature accepts %d named params)",
        cls.__name__, len(accepted),
    )


def enable_triple_stages_helper_kwarg_filter() -> None:
    """Patch free helper functions that the vendored triple-stages module
    imports via ``from ltx_pipelines.utils.helpers import assert_resolution,
    combined_image_conditionings``. MUST run BEFORE that vendored import:
    ``from X import Y`` binds ``Y`` at vendored-import time, so a post-hoc
    module patch would not propagate to the local reference. Wired in
    ``src/upstream.py`` immediately before the vendored module import.
    Idempotent."""
    global _helper_patches_applied
    if _helper_patches_applied:
        return

    try:
        from ltx_pipelines.utils import helpers as ltx_helpers
    except ImportError:  # pragma: no cover — defensive
        logger.warning(
            "ltx_pipelines.utils.helpers unavailable; "
            "triple-stages helper kwarg filter not installed"
        )
        return

    for fn_name in ("assert_resolution", "combined_image_conditionings"):
        _wrap_function_with_kwarg_filter(ltx_helpers, fn_name)

    _helper_patches_applied = True


def _wrap_function_with_kwarg_filter(module, fn_name: str) -> None:
    """Wrap ``module.<fn_name>`` to silently drop kwargs not in its signature.

    No-op when the function already accepts the fork's kwargs or declares
    ``**kwargs``. Idempotent via a per-callable attribute marker."""
    original = getattr(module, fn_name, None)
    if original is None:  # pragma: no cover — upstream rename guard
        logger.warning(
            "%s.%s missing; cannot install function kwarg filter",
            module.__name__, fn_name,
        )
        return
    if getattr(original, "_kwarg_filtered", False):
        return  # already wrapped

    sig = inspect.signature(original)
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_kw:
        logger.info(
            "%s.%s already accepts **kwargs; kwarg filter not needed",
            module.__name__, fn_name,
        )
        return

    accepted = {
        name for name, p in sig.parameters.items()
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }

    @functools.wraps(original)
    def wrapper(*args, **kwargs):
        unknown = [k for k in kwargs if k not in accepted]
        if unknown:
            for k in unknown:
                kwargs.pop(k)
            if not getattr(wrapper, "_warned", False):
                logger.info(
                    "%s.%s kwarg filter active — dropping %s on first "
                    "call (vendored triple-stages passes newer-API kwargs "
                    "absent from pinned upstream signature)",
                    module.__name__, fn_name, unknown,
                )
                wrapper._warned = True
        return original(*args, **kwargs)

    wrapper._kwarg_filtered = True
    setattr(module, fn_name, wrapper)
    logger.info(
        "Installed kwarg filter on %s.%s (signature accepts %d named params)",
        module.__name__, fn_name, len(accepted),
    )
