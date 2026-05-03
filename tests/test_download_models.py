"""Tests for download_models (no actual downloads)."""

import os
from unittest.mock import patch, MagicMock

import pytest


class TestEnsureModelsDownloaded:
    def test_raises_without_hf_token(self, tmp_path):
        from src.download_models import ensure_models_downloaded

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="HF_TOKEN"):
                ensure_models_downloaded(str(tmp_path))

    @pytest.mark.parametrize("fp8_mode", ["", "scaled_mm"])
    def test_skips_existing_files(self, tmp_path, fp8_mode):
        """If all model files exist, no downloads should happen.

        Parametrized over LTX_FP8_MODE so the FP8 selector branch is exercised
        — when ``LTX_FP8_MODE=scaled_mm`` the FP8 distilled file must also be
        present on disk to satisfy the additional download path.
        """
        from src.download_models import ensure_models_downloaded

        # Create fake model files
        (tmp_path / "ltx-2.3-22b-distilled-1.1.safetensors").touch()
        (tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
        (tmp_path / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors").touch()
        if fp8_mode == "scaled_mm":
            (tmp_path / "ltx-2.3-22b-distilled-fp8.safetensors").touch()
        gemma_dir = tmp_path / "gemma-3-12b-it-qat-q4_0-unquantized"
        gemma_dir.mkdir()
        (gemma_dir / "model.safetensors").touch()

        env = {"HF_TOKEN": "hf_test"}
        if fp8_mode:
            env["LTX_FP8_MODE"] = fp8_mode
        with patch.dict(os.environ, env, clear=True):
            with patch("src.download_models.hf_hub_download") as mock_dl, \
                 patch("src.download_models.snapshot_download") as mock_snap:
                ensure_models_downloaded(str(tmp_path))

        # No downloads should have been triggered
        mock_dl.assert_not_called()
        mock_snap.assert_not_called()

    def test_creates_model_dir(self, tmp_path):
        """Should create the model directory if it doesn't exist."""
        from src.download_models import ensure_models_downloaded

        model_dir = tmp_path / "subdir" / "models"
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
            with patch("src.download_models.hf_hub_download"), \
                 patch("src.download_models.snapshot_download"):
                ensure_models_downloaded(str(model_dir))

        assert model_dir.exists()
