"""Tests for pipeline helper functions and the public package surface."""

from src.pipeline import DEFAULT_NEGATIVE_PROMPT
from src.pipeline.triple_stages_comfyui import (
    LANDSCAPE_BUCKET,
    PORTRAIT_BUCKET,
    _resolve_aspect_ratio,
    _round_frames_8k1,
)


class TestRoundFrames:
    def test_valid_frame_count(self):
        assert _round_frames_8k1(121) == 121  # (121-1)/8 = 15 → 15*8+1 = 121

    def test_rounds_to_8k_plus_1(self):
        assert _round_frames_8k1(100) == 97  # (100-1)/8 = 12 → 12*8+1 = 97

    def test_minimum(self):
        assert _round_frames_8k1(9) == 9  # (9-1)/8 = 1 → 1*8+1 = 9

    def test_not_8k_plus_1(self):
        assert _round_frames_8k1(10) == 9  # (10-1)/8 = 1 → 1*8+1 = 9

    def test_large(self):
        assert _round_frames_8k1(257) == 257  # (257-1)/8 = 32 → 32*8+1 = 257


class TestBuckets:
    def test_landscape_bucket_is_128_aligned(self):
        gen_w, gen_h, out_w, out_h = LANDSCAPE_BUCKET
        assert gen_w % 128 == 0 and gen_h % 128 == 0
        assert (out_w, out_h) == (1920, 1080)
        # the bucket must contain the output dims so the center-crop is loss-free.
        assert gen_w >= out_w and gen_h >= out_h

    def test_portrait_bucket_is_128_aligned(self):
        gen_w, gen_h, out_w, out_h = PORTRAIT_BUCKET
        assert gen_w % 128 == 0 and gen_h % 128 == 0
        assert (out_w, out_h) == (1080, 1920)
        assert gen_w >= out_w and gen_h >= out_h


class TestResolveAspectRatio:
    def test_explicit_landscape_wins_over_image(self, tmp_path):
        # Even with no image, "16:9" → landscape bucket.
        assert _resolve_aspect_ratio("16:9", None) == LANDSCAPE_BUCKET

    def test_explicit_portrait_wins_over_image(self):
        assert _resolve_aspect_ratio("9:16", None) == PORTRAIT_BUCKET

    def test_auto_t2v_falls_back_to_landscape(self):
        assert _resolve_aspect_ratio("auto", None) == LANDSCAPE_BUCKET

    def test_auto_landscape_image_resolves_to_landscape(self, tmp_path):
        from PIL import Image

        path = str(tmp_path / "land.png")
        Image.new("RGB", (1920, 1080), (0, 0, 0)).save(path)
        assert _resolve_aspect_ratio("auto", path) == LANDSCAPE_BUCKET

    def test_auto_portrait_image_resolves_to_portrait(self, tmp_path):
        from PIL import Image

        path = str(tmp_path / "port.png")
        Image.new("RGB", (1080, 1920), (0, 0, 0)).save(path)
        assert _resolve_aspect_ratio("auto", path) == PORTRAIT_BUCKET

    def test_auto_square_image_resolves_to_landscape(self, tmp_path):
        # tie-breaker: square (W == H) → landscape, matching derive_orientation.
        from PIL import Image

        path = str(tmp_path / "sq.png")
        Image.new("RGB", (512, 512), (0, 0, 0)).save(path)
        assert _resolve_aspect_ratio("auto", path) == LANDSCAPE_BUCKET

    def test_explicit_override_beats_image_orientation(self, tmp_path):
        # Caller asks for 9:16 even though the image is landscape — the
        # explicit value wins.
        from PIL import Image

        path = str(tmp_path / "land.png")
        Image.new("RGB", (1920, 1080), (0, 0, 0)).save(path)
        assert _resolve_aspect_ratio("9:16", path) == PORTRAIT_BUCKET


class TestDefaults:
    def test_negative_prompt_not_empty(self):
        assert len(DEFAULT_NEGATIVE_PROMPT) > 0

    def test_negative_prompt_is_string(self):
        assert isinstance(DEFAULT_NEGATIVE_PROMPT, str)
