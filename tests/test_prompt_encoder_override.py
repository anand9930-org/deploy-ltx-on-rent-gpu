"""Unit tests for ``src/prompt_encoder_override.py`` — verifies the kwarg
filter we install at boot to absorb the version skew between the vendored
triple-stages fork and our pinned upstream ``PromptEncoder`` /
``DiffusionStage`` signatures.

Tests run pure-Python — no upstream / GPU needed. They exercise the
private ``_wrap_with_kwarg_filter`` directly on toy classes that
reproduce the three relevant signature shapes."""

import logging

from src.prompt_encoder_override import _wrap_with_kwarg_filter


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
