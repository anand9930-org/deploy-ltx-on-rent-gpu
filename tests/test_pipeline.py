"""Tests for pipeline helper functions (no GPU needed)."""

from src.pipeline import DEFAULT_NEGATIVE_PROMPT, _round_frames, _round_to


class TestRoundTo:
    def test_exact_multiple(self):
        assert _round_to(128, 64) == 128

    def test_rounds_down(self):
        assert _round_to(130, 64) == 128

    def test_rounds_down_large(self):
        assert _round_to(1920, 64) == 1920

    def test_rounds_down_odd(self):
        assert _round_to(1000, 64) == 960


class TestRoundFrames:
    def test_valid_frame_count(self):
        assert _round_frames(121) == 121  # (121-1)/8 = 15 → 15*8+1 = 121

    def test_rounds_to_8k_plus_1(self):
        assert _round_frames(100) == 97  # (100-1)/8 = 12 → 12*8+1 = 97

    def test_minimum(self):
        assert _round_frames(9) == 9  # (9-1)/8 = 1 → 1*8+1 = 9

    def test_not_8k_plus_1(self):
        assert _round_frames(10) == 9  # (10-1)/8 = 1 → 1*8+1 = 9

    def test_large(self):
        assert _round_frames(257) == 257  # (257-1)/8 = 32 → 32*8+1 = 257


class TestDefaults:
    def test_negative_prompt_not_empty(self):
        assert len(DEFAULT_NEGATIVE_PROMPT) > 0

    def test_negative_prompt_is_string(self):
        assert isinstance(DEFAULT_NEGATIVE_PROMPT, str)
