"""Unit tests for :mod:`src.diagnostics` — boot-time host/image diagnostics.

These tests run without a GPU and without ``nvidia-smi`` actually present;
``subprocess.run`` is monkeypatched everywhere ``host_machine`` calls out.
"""

from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src import diagnostics


class TestParseCudaVersion:
    def test_major_minor(self) -> None:
        assert diagnostics._parse_cuda_version("13.0") == (13, 0)

    def test_major_minor_patch_drops_patch(self) -> None:
        assert diagnostics._parse_cuda_version("12.8.1") == (12, 8)

    def test_major_only(self) -> None:
        assert diagnostics._parse_cuda_version("12") == (12, 0)

    def test_empty_string(self) -> None:
        assert diagnostics._parse_cuda_version("") is None

    def test_none(self) -> None:
        assert diagnostics._parse_cuda_version(None) is None

    def test_garbage(self) -> None:
        assert diagnostics._parse_cuda_version("not-a-version") is None


class TestCompatibilityVerdict:
    def test_equal_versions_ok(self) -> None:
        ok, msg = diagnostics.compatibility_verdict(
            {"cuda_tuple": (13, 0), "cuda_toolkit": "13.0"},
            {"host_cuda_tuple": (13, 0)},
        )
        assert ok is True
        assert "supports CUDA 13.0" in msg

    def test_host_newer_ok(self) -> None:
        ok, msg = diagnostics.compatibility_verdict(
            {"cuda_tuple": (12, 8), "cuda_toolkit": "12.8"},
            {"host_cuda_tuple": (13, 1)},
        )
        assert ok is True
        assert "12.8" in msg
        assert "13.1" in msg

    def test_host_older_fails_with_driver_too_old(self) -> None:
        ok, msg = diagnostics.compatibility_verdict(
            {"cuda_tuple": (13, 0), "cuda_toolkit": "13.0"},
            {"host_cuda_tuple": (12, 8)},
        )
        assert ok is False
        assert "DRIVER TOO OLD" in msg
        assert "13.0" in msg
        assert "12.8" in msg

    def test_image_cuda_unparseable(self) -> None:
        ok, msg = diagnostics.compatibility_verdict(
            {"cuda_tuple": None, "cuda_toolkit": "huh"},
            {"host_cuda_tuple": (13, 0)},
        )
        assert ok is False
        assert "unparseable" in msg

    def test_host_cuda_missing(self) -> None:
        ok, msg = diagnostics.compatibility_verdict(
            {"cuda_tuple": (13, 0), "cuda_toolkit": "13.0"},
            {"host_cuda_tuple": None},
        )
        assert ok is False
        assert "nvidia-smi" in msg


class TestHostMachine:
    """Monkeypatch subprocess.run so host_machine() parses canned outputs."""

    @staticmethod
    def _patched_run(commands: dict[str, str]):
        """Return a fake subprocess.run that returns prepared stdout per command."""

        def fake_run(cmd, *args, **kwargs):  # noqa: ANN001 — mirrors subprocess.run signature
            key = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            for needle, output in commands.items():
                if needle in key:
                    return SimpleNamespace(stdout=output, returncode=0)
            return SimpleNamespace(stdout="", returncode=0)

        return fake_run

    def test_parses_nvidia_smi_output(self) -> None:
        nvidia_smi_text = (
            "Wed May 14 06:17:24 2026       \n"
            "+-----------------------------------------------------------------------------+\n"
            "| NVIDIA-SMI 570.86.10    Driver Version: 570.86.10    CUDA Version: 12.8     |\n"
            "+-----------------------------------------------------------------------------+\n"
        )
        nvidia_smi_csv = "NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 570.86.10, 12.0\n"

        canned = {
            "nvidia-smi --query-gpu": nvidia_smi_csv,
            "nvidia-smi": nvidia_smi_text,
            "free -h": (
                "              total        used        free      shared  buff/cache   available\n"
                "Mem:           250Gi        12Gi       100Gi       1.0Gi       138Gi       236Gi\n"
            ),
            "df -h /workspace": (
                "Filesystem      Size  Used Avail Use% Mounted on\n"
                "/dev/nvme0n1    500G   80G  400G  17% /workspace\n"
            ),
        }
        with patch.object(subprocess, "run", side_effect=self._patched_run(canned)):
            host = diagnostics.host_machine()

        assert host["host_cuda_max"] == "12.8"
        assert host["host_cuda_tuple"] == (12, 8)
        assert host["gpu_name"] == "NVIDIA RTX PRO 6000 Blackwell Server Edition"
        assert host["vram_total_mib"] == "97887"
        assert host["driver_version"] == "570.86.10"
        assert host["compute_cap"] == "12.0"
        assert host["ram_total"] == "250Gi"
        assert host["disk_free_workspace"] == "400G"

    def test_handles_missing_nvidia_smi_gracefully(self) -> None:
        """If nvidia-smi is unavailable, host_machine returns 'unknown' fields."""
        # subprocess.run failure modes used by _run:
        with patch.object(subprocess, "run", side_effect=FileNotFoundError):
            host = diagnostics.host_machine()
        assert host["gpu_name"] == "unknown"
        assert host["host_cuda_max"] == "unknown"
        assert host["host_cuda_tuple"] is None
        assert host["driver_version"] == "unknown"
        # platform.* is not subprocess-dependent and should still populate:
        assert "os" in host and host["os"]


class TestExitCode:
    """End-to-end: feed image + host dicts, drive main(), check exit code."""

    @staticmethod
    def _stub_image_and_host(monkeypatch: pytest.MonkeyPatch, img: dict, host: dict) -> None:
        monkeypatch.setattr(diagnostics, "image_requirements", lambda: img)
        monkeypatch.setattr(diagnostics, "host_machine", lambda: host)

    def test_success_exits_zero(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        self._stub_image_and_host(
            monkeypatch,
            img={
                "torch": "2.8.0+cu128",
                "torchvision": "0.23.0+cu128",
                "torchaudio": "2.8.0+cu128",
                "cuda_toolkit": "12.8",
                "cuda_tuple": (12, 8),
                "min_driver": "570.86.10",
                "arch_flags": ["sm_120"],
                "target_arch": "sm_120 (Blackwell)",
                "vram_resident_gb": 64,
                "vram_peak_gb": 70,
            },
            host={
                "os": "Linux",
                "kernel": "6.x",
                "python": "3.12.0",
                "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "vram_total_mib": "97887",
                "driver_version": "570.86.10",
                "compute_cap": "12.0",
                "host_cuda_max": "12.8",
                "host_cuda_tuple": (12, 8),
                "ram_total": "250Gi",
                "disk_free_workspace": "400G",
            },
        )
        assert diagnostics.main() == 0
        out = capsys.readouterr().out
        assert "IMAGE REQUIREMENTS" in out
        assert "HOST MACHINE" in out
        assert "COMPATIBILITY" in out
        assert "sm_120 Blackwell" in out

    def test_driver_too_old_exits_two(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
        self._stub_image_and_host(
            monkeypatch,
            img={
                "torch": "2.10.0+cu130",
                "torchvision": "0.25.0+cu130",
                "torchaudio": "2.10.0+cu130",
                "cuda_toolkit": "13.0",
                "cuda_tuple": (13, 0),
                "min_driver": "580.65.06",
                "arch_flags": ["sm_120"],
                "target_arch": "sm_120 (Blackwell)",
                "vram_resident_gb": 64,
                "vram_peak_gb": 70,
            },
            host={
                "os": "Linux",
                "kernel": "6.x",
                "python": "3.12.0",
                "gpu_name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
                "vram_total_mib": "97887",
                "driver_version": "570.86.10",
                "compute_cap": "12.0",
                "host_cuda_max": "12.8",
                "host_cuda_tuple": (12, 8),
                "ram_total": "250Gi",
                "disk_free_workspace": "400G",
            },
        )
        assert diagnostics.main() == 2
        out = capsys.readouterr().out
        assert "DRIVER TOO OLD" in out
        assert "Remediation" in out
        assert "cu128" in out  # rollback hint references the host's CUDA
        assert "CUDA 13.0+" in out


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
