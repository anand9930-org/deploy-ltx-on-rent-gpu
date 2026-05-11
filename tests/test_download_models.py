"""Tests for download_models (no actual downloads)."""

import os
from unittest.mock import patch

import pytest


class TestEnsureModelsDownloaded:
    def test_raises_without_hf_token(self, tmp_path):
        from src.download_models import ensure_models_downloaded

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(RuntimeError, match="HF_TOKEN"):
                ensure_models_downloaded(str(tmp_path))

    def test_skips_existing_files(self, tmp_path):
        """If all model files exist, no downloads happen."""
        from src.download_models import ensure_models_downloaded

        (tmp_path / "ltx-2.3-22b-dev-fp8.safetensors").touch()
        (tmp_path / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors").touch()
        (tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()

        comfy_te_dir = tmp_path / "split_files" / "text_encoders"
        comfy_te_dir.mkdir(parents=True)
        (comfy_te_dir / "gemma_3_12B_it.safetensors").touch()

        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=True):
            with patch("src.download_models.hf_hub_download") as mock_dl:
                ensure_models_downloaded(str(tmp_path))

        mock_dl.assert_not_called()

    def test_creates_model_dir(self, tmp_path):
        """Should create the model directory if it doesn't exist."""
        from src.download_models import ensure_models_downloaded

        model_dir = tmp_path / "subdir" / "models"
        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
            with patch("src.download_models.hf_hub_download"):
                ensure_models_downloaded(str(model_dir))

        assert model_dir.exists()

    def test_downloads_all_four_models(self, tmp_path):
        """All four model files should be downloaded when none are cached."""
        from src.download_models import ensure_models_downloaded

        with patch.dict(os.environ, {"HF_TOKEN": "hf_test"}, clear=True):
            with patch("src.download_models.hf_hub_download") as mock_dl:
                ensure_models_downloaded(str(tmp_path))

        called_files = [c.kwargs.get("filename") for c in mock_dl.call_args_list]
        assert "ltx-2.3-22b-dev-fp8.safetensors" in called_files
        assert "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" in called_files
        assert "ltx-2.3-spatial-upscaler-x2-1.1.safetensors" in called_files
        assert "split_files/text_encoders/gemma_3_12B_it.safetensors" in called_files
