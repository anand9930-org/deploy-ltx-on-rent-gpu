"""Unit tests for ``src/prompt_encoder_override.py`` — verifies the kwarg
filter we install at boot to absorb the version skew between the vendored
triple-stages fork and our pinned upstream ``PromptEncoder`` /
``DiffusionStage`` signatures, plus the free-function helpers
``assert_resolution`` / ``combined_image_conditionings``.

Tests run pure-Python — no upstream / GPU needed. They exercise the
private ``_wrap_with_kwarg_filter`` and ``_wrap_function_with_kwarg_filter``
directly on toy classes / fake modules that reproduce the relevant
signature shapes."""

import logging
import types

from src.prompt_encoder_override import (
    _wrap_function_with_kwarg_filter,
    _wrap_with_kwarg_filter,
)


class _NarrowSig:
    """Fake whose ``__call__`` does NOT accept ``streaming_prefetch_count``."""

    def __call__(self, prompts, *, enhance_first_prompt=False):
        return ("p", "n", prompts, enhance_first_prompt)


class _AlreadyAccepts:
    """Fake whose ``__call__`` already accepts the kwarg the fork passes."""

    def __call__(self, prompts, *, streaming_prefetch_count=None):
        return ("ok", streaming_prefetch_count)


class _VarKwargs:
    """Fake declaring ``**kwargs`` — patch must be a no-op."""

    def __call__(self, prompts, **kwargs):
        return ("ok", kwargs)


class TestKwargFilter:
    def test_drops_unknown_kwarg_and_forwards_known(self):
        _wrap_with_kwarg_filter(_NarrowSig)
        out = _NarrowSig()(
            ["a", "b"],
            enhance_first_prompt=True,
            streaming_prefetch_count=4,  # not in signature
        )
        assert out == ("p", "n", ["a", "b"], True)

    def test_idempotent(self):
        _wrap_with_kwarg_filter(_NarrowSig)
        first = _NarrowSig.__call__
        _wrap_with_kwarg_filter(_NarrowSig)
        assert _NarrowSig.__call__ is first  # not double-wrapped

    def test_noop_when_signature_already_accepts(self):
        original = _AlreadyAccepts.__call__
        _wrap_with_kwarg_filter(_AlreadyAccepts)
        # Signature accepts the kwarg → wrapper is still installed (we don't
        # know what kwargs the fork *might* pass), but a present-kwarg call
        # passes through unchanged with the original value preserved.
        assert _AlreadyAccepts()(["x"], streaming_prefetch_count=7) == ("ok", 7)

    def test_noop_when_var_kwargs(self):
        original = _VarKwargs.__call__
        _wrap_with_kwarg_filter(_VarKwargs)
        # **kwargs → wrapper not installed; method object is unchanged.
        assert _VarKwargs.__call__ is original

    def test_drop_logged_once(self, caplog):
        # Fresh class so the wrapper's first-call ``_warned`` flag is clean.
        class _Fresh:
            def __call__(self, prompts):
                return prompts
        _wrap_with_kwarg_filter(_Fresh)
        with caplog.at_level(logging.INFO, logger="src.prompt_encoder_override"):
            _Fresh()(["a"], streaming_prefetch_count=1)
            _Fresh()(["b"], streaming_prefetch_count=2)
        drops = [r for r in caplog.records if "kwarg filter active" in r.getMessage()]
        assert len(drops) == 1  # one-shot first-call log


def _make_module(name: str, **attrs):
    """Build a throwaway module-like object for free-function patch tests."""
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    return mod


class TestFunctionKwargFilter:
    def test_drops_unknown_kwarg_and_forwards_known(self):
        def assert_resolution(*, height, width):
            return (height, width)
        mod = _make_module("fake_helpers", assert_resolution=assert_resolution)

        _wrap_function_with_kwarg_filter(mod, "assert_resolution")

        # Vendored fork passes is_two_stage=True; pinned upstream lacks it.
        out = mod.assert_resolution(height=512, width=512, is_two_stage=True)
        assert out == (512, 512)

    def test_idempotent(self):
        def fn(x):
            return x
        mod = _make_module("fake_helpers", fn=fn)

        _wrap_function_with_kwarg_filter(mod, "fn")
        first = mod.fn
        _wrap_function_with_kwarg_filter(mod, "fn")
        assert mod.fn is first  # not double-wrapped

    def test_noop_when_var_kwargs(self):
        def fn(x, **kwargs):
            return (x, kwargs)
        mod = _make_module("fake_helpers", fn=fn)
        original = mod.fn

        _wrap_function_with_kwarg_filter(mod, "fn")
        # **kwargs → wrapper not installed; module attribute is unchanged.
        assert mod.fn is original

    def test_missing_function_logs_and_returns(self, caplog):
        mod = _make_module("fake_helpers")  # no attribute "missing_fn"
        with caplog.at_level(logging.WARNING, logger="src.prompt_encoder_override"):
            _wrap_function_with_kwarg_filter(mod, "missing_fn")
        warnings = [r for r in caplog.records if "missing" in r.getMessage()]
        assert warnings  # rename guard logged
        assert not hasattr(mod, "missing_fn")

    def test_combined_image_conditionings_shape(self):
        # Reproduces the second high-risk vendored call: positional + the
        # newer-API kwarg ``preprocessed_images`` that pinned upstream lacks.
        def combined_image_conditionings(*, images, height, width):
            return (len(images), height, width)
        mod = _make_module(
            "fake_helpers",
            combined_image_conditionings=combined_image_conditionings,
        )

        _wrap_function_with_kwarg_filter(mod, "combined_image_conditionings")

        out = mod.combined_image_conditionings(
            images=["a", "b"], height=480, width=640, preprocessed_images=None,
        )
        assert out == (2, 480, 640)
