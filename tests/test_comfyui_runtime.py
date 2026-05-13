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
