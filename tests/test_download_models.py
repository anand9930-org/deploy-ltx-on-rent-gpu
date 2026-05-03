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

    @pytest.mark.parametrize("fp8_mode", ["", "cast", "scaled_mm"])
    def test_skips_existing_files(self, tmp_path, fp8_mode):
        """If all model files for the given LTX_FP8_MODE exist, no downloads
        happen. Parametrized over the three boot modes the wrapper supports —
        each mode pulls a different union of T2V + unified files.
        """
        from src.download_models import ensure_models_downloaded

        # Always-present files (T2V dev BF16 base + unified base + IC-LoRA +
        # spatial upscaler).
        (tmp_path / "ltx-2.3-22b-dev.safetensors").touch()
        (tmp_path / "ltx-2.3-22b-distilled-1.1.safetensors").touch()
        (tmp_path / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors").touch()
        (tmp_path / "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors").touch()

        # cast/no-FP8 path also pulls the T2V distilled-LoRA.
        if fp8_mode != "scaled_mm":
            (tmp_path / "ltx-2.3-22b-distilled-lora-384-1.1.safetensors").touch()

        # scaled_mm pulls the dev FP8 DiT in addition.
        if fp8_mode == "scaled_mm":
            (tmp_path / "ltx-2.3-22b-dev-fp8.safetensors").touch()

        # Both FP8 paths share the distilled FP8 DiT.
        if fp8_mode in ("scaled_mm", "cast"):
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

    def test_scaled_mm_pulls_dev_fp8(self, tmp_path):
        """scaled_mm mode must trigger dev-fp8 download but skip distilled-LoRA."""
        from src.download_models import ensure_models_downloaded

        with patch.dict(
            os.environ,
            {"HF_TOKEN": "hf_test", "LTX_FP8_MODE": "scaled_mm"},
            clear=True,
        ):
            with patch("src.download_models.hf_hub_download") as mock_dl, \
                 patch("src.download_models.snapshot_download"):
                ensure_models_downloaded(str(tmp_path))

        called_files = [c.kwargs.get("filename") for c in mock_dl.call_args_list]
        assert "ltx-2.3-22b-dev-fp8.safetensors" in called_files
        assert "ltx-2.3-22b-distilled-fp8.safetensors" in called_files
        # Distilled LoRA only needed on the cast/bf16 path.
        assert "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" not in called_files

    def test_cast_pulls_distilled_lora_not_dev_fp8(self, tmp_path):
        """cast mode pulls the distilled LoRA + distilled-fp8 but no dev-fp8."""
        from src.download_models import ensure_models_downloaded

        with patch.dict(
            os.environ,
            {"HF_TOKEN": "hf_test", "LTX_FP8_MODE": "cast"},
            clear=True,
        ):
            with patch("src.download_models.hf_hub_download") as mock_dl, \
                 patch("src.download_models.snapshot_download"):
                ensure_models_downloaded(str(tmp_path))

        called_files = [c.kwargs.get("filename") for c in mock_dl.call_args_list]
        assert "ltx-2.3-22b-distilled-lora-384-1.1.safetensors" in called_files
        assert "ltx-2.3-22b-distilled-fp8.safetensors" in called_files
        assert "ltx-2.3-22b-dev-fp8.safetensors" not in called_files
