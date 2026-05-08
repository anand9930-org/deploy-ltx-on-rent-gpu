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

Targets (read off the vendored file, see ``vendor/ti2vid_triple_stages.py``):
  * ``PromptEncoder.__call__`` — receives ``streaming_prefetch_count``
  * ``DiffusionStage.__call__`` — receives ``streaming_prefetch_count``

Patch is sig-aware: a no-op when upstream already accepts the kwarg or
declares ``**kwargs``. Idempotent via a per-function attribute marker
plus a module-level flag — safe to call on every ``LTXVideoGenerator``
construction.
"""

import functools
import inspect
import logging

logger = logging.getLogger(__name__)

_applied = False


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
