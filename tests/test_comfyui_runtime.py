"""Headless tests for the ComfyUI runtime bootstrap.

``src/comfyui_runtime.py`` does all its ComfyUI imports *inside* the functions,
so the module must import on a box without ComfyUI, and the pre-bootstrap state
of its accessors must be well-defined (the test suite never runs
``bootstrap_once`` — that needs the ComfyUI checkout + a GPU at runtime).

Also pins the ``cpu_only`` parameter (the build-time node-contract check in the
Dockerfile passes ``cpu_only=True`` so ComfyUI doesn't probe ``torch.cuda`` at
import time on the GPU-less GHCR builder) — a rename / drop would break the build.
"""

import inspect

import pytest

from src import comfyui_runtime


def test_module_imports_without_comfyui():
    # No top-level ``import comfy`` / ``import nodes`` — importing this module is
    # free even where ComfyUI isn't installed.
    assert hasattr(comfyui_runtime, "bootstrap_once")
    assert hasattr(comfyui_runtime, "unload_models")
    assert hasattr(comfyui_runtime, "reset_to_clean_gpu")
    assert hasattr(comfyui_runtime, "input_dir")
    assert hasattr(comfyui_runtime, "is_bootstrapped")


def test_bootstrap_once_accepts_cpu_only():
    params = inspect.signature(comfyui_runtime.bootstrap_once).parameters
    assert "cpu_only" in params, "Dockerfile's build-time check passes cpu_only=True"
    assert params["cpu_only"].default is False, "runtime (pod boot) wants the GPU"


def test_pre_bootstrap_state():
    # The suite never bootstraps, so these reflect the cold state.
    assert comfyui_runtime.is_bootstrapped() is False
    with pytest.raises(RuntimeError, match="not bootstrapped"):
        comfyui_runtime.input_dir()


def test_unload_models_is_a_noop_before_bootstrap():
    # Must not raise when ComfyUI was never bootstrapped (called from
    # LTXVideoGenerator._ensure_mode on every mode switch).
    comfyui_runtime.unload_models()


def test_reset_to_clean_gpu_is_safe_without_cuda():
    # gc + (conditional) torch.cuda.empty_cache(); a no-CUDA box just gc's.
    comfyui_runtime.reset_to_clean_gpu()


# ---------------------------------------------------------------------------
# _patch_comfyui_vae_inplace — workaround for ComfyUI PR #13028 on torch 2.10+
# ---------------------------------------------------------------------------

_BAD_VAE_LINE = (
    "        self.process_output = lambda image: "
    "image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)\n"
)
_GOOD_VAE_LINE = (
    "        self.process_output = lambda image: "
    "torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)\n"
)


def _write_fake_sd_py(tmp_path, body: str) -> str:
    """Stage a minimal ``comfy/sd.py`` under ``tmp_path`` and return its path."""
    comfy_dir = tmp_path / "comfy"
    comfy_dir.mkdir()
    sd_py = comfy_dir / "sd.py"
    sd_py.write_text(body, encoding="utf-8")
    return str(sd_py)


def test_patch_comfyui_vae_inplace_replaces_bad_pattern(tmp_path):
    sd_py = _write_fake_sd_py(tmp_path, _BAD_VAE_LINE)
    comfyui_runtime._patch_comfyui_vae_inplace(str(tmp_path))
    contents = open(sd_py, "r", encoding="utf-8").read()
    assert _GOOD_VAE_LINE in contents
    assert "add_(1.0).div_(2.0).clamp_(0.0, 1.0)" not in contents


def test_patch_comfyui_vae_inplace_is_idempotent(tmp_path):
    # Two consecutive calls leave the file in the same state — no double-patch
    # corruption, no exceptions on a file that's already patched.
    sd_py = _write_fake_sd_py(tmp_path, _BAD_VAE_LINE)
    comfyui_runtime._patch_comfyui_vae_inplace(str(tmp_path))
    after_first = open(sd_py, "r", encoding="utf-8").read()
    comfyui_runtime._patch_comfyui_vae_inplace(str(tmp_path))
    after_second = open(sd_py, "r", encoding="utf-8").read()
    assert after_first == after_second
    assert _GOOD_VAE_LINE in after_second


def test_patch_comfyui_vae_inplace_missing_file_is_silent(tmp_path):
    # No comfy/sd.py present (e.g. partial checkout) — the helper must not raise.
    # The natural import error will surface from the comfy.* imports later.
    comfyui_runtime._patch_comfyui_vae_inplace(str(tmp_path))


def test_patch_comfyui_vae_inplace_already_safe_is_silent(tmp_path):
    # Upstream ComfyUI may have reverted the in-place op (or we're pointed at
    # an older SHA): the file has the GOOD expression already, nothing to do.
    sd_py = _write_fake_sd_py(tmp_path, _GOOD_VAE_LINE)
    comfyui_runtime._patch_comfyui_vae_inplace(str(tmp_path))
    assert open(sd_py, "r", encoding="utf-8").read() == _GOOD_VAE_LINE
