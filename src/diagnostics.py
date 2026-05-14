"""Boot-time diagnostics for the LTX-2.3 Blackwell deployment.

Prints structured "image requirements" vs "host machine reality" so any
driver/runtime mismatch is obvious from the first lines of pod logs — not
buried inside a Python traceback. Carefully avoids any ``torch.cuda.*`` call
that triggers CUDA initialization: driver-too-old errors raise from
:func:`torch.cuda._lazy_init`, which we want to *pre-empt* not propagate.

Run as a module from start.sh::

    cd /app
    python3 -m src.diagnostics

Exit codes:
* ``0`` — host driver supports the image's CUDA toolkit AND the GPU's compute
  capability is acceptable (Blackwell sm_120 = primary target; sm_12X or
  Hopper sm_9X = log-only warnings).
* ``2`` — driver vs. runtime CUDA mismatch (e.g. cu130 wheel on a 12.8
  driver). Remediation hints printed.
* ``3`` — compute capability unsupported (pre-Hopper).
"""

from __future__ import annotations

import importlib.metadata
import platform
import re
import subprocess
import sys

logger = None  # not used; this module is invoked as a script and prints directly


# Known minimum NVIDIA Linux driver versions per CUDA toolkit major.minor.
# Source: NVIDIA's CUDA Toolkit / Driver Compatibility tables. Used as
# informational output for operators — the actual gating decision is made by
# comparing ``torch.version.cuda`` (what the wheel needs) to the host's max
# CUDA reported by ``nvidia-smi`` (what the driver supports).
_MIN_DRIVER_FOR_CUDA: dict[tuple[int, int], str] = {
    (12, 0): "525.60.13",
    (12, 4): "550.54.14",
    (12, 6): "560.35.03",
    (12, 8): "570.86.10",
    (12, 9): "575.51.02",
    (13, 0): "580.65.06",
    (13, 1): "585.x",
    (13, 2): "590.x",
}


def _run(cmd: list[str]) -> str:
    """Run a shell command; return stdout, or empty string on failure.

    No exception is raised — the caller decides what to do with missing data
    (typically: show ``"unknown"`` in the printed report).
    """
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return ""


def _parse_cuda_version(value: str | None) -> tuple[int, int] | None:
    """Parse a CUDA version like ``"13.0"`` or ``"12.8.1"`` -> ``(major, minor)``.

    Returns ``None`` for unparseable or missing values so the caller can render
    them as ``"unknown"`` and continue.
    """
    if not value:
        return None
    try:
        parts = value.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except (ValueError, IndexError):
        return None


def _section(title: str) -> None:
    """Print a left-padded section header with consistent width."""
    bar = "=" * 28
    print(f"{bar} {title} {bar}", flush=True)


def _kv(key: str, value: str, ok: bool | None = None) -> None:
    """Print a single key/value line; optional ✓/✗ marker for verdicts."""
    if ok is None:
        marker = "  "
    elif ok:
        marker = "✓ "  # check mark
    else:
        marker = "✗ "  # cross mark
    print(f"  {marker}{key:<24} {value}", flush=True)


def image_requirements() -> dict[str, object]:
    """Return facts about *this image* — what the build targets.

    Safe to call without a working CUDA driver: only reads version strings and
    the compile-time arch flags baked into the torch wheel.
    """
    import torch  # noqa: PLC0415 — deferred so the module imports cleanly even without torch

    cuda_ver = torch.version.cuda  # e.g. "13.0"
    cuda_tuple = _parse_cuda_version(cuda_ver)
    arch_flags = (torch._C._cuda_getArchFlags() or "").split()  # noqa: SLF001 — private API, stable across torch 1.x/2.x

    return {
        "torch": torch.__version__,
        "torchvision": importlib.metadata.version("torchvision"),
        "torchaudio": importlib.metadata.version("torchaudio"),
        "cuda_toolkit": cuda_ver,
        "cuda_tuple": cuda_tuple,
        "min_driver": _MIN_DRIVER_FOR_CUDA.get(cuda_tuple, "unknown"),
        "arch_flags": arch_flags,
        "target_arch": "sm_120 (Blackwell)",
        "vram_resident_gb": 64,
        "vram_peak_gb": 70,
    }


def host_machine() -> dict[str, object]:
    """Inspect the host via ``nvidia-smi`` (NVML) and platform module.

    Uses NVML — completely separate from the CUDA runtime — so it works even
    when the host driver can't support the image's CUDA toolkit (which is
    exactly when this diagnostic matters most).
    """
    smi_text = _run(["nvidia-smi"])
    cuda_match = re.search(r"CUDA Version:\s+(\d+(?:\.\d+)?)", smi_text)
    host_cuda = cuda_match.group(1) if cuda_match else None
    host_cuda_tuple = _parse_cuda_version(host_cuda)

    smi_csv = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap",
            "--format=csv,noheader,nounits",
        ]
    ).strip()
    if smi_csv:
        parts = [s.strip() for s in smi_csv.split(",")]
        parts = (parts + ["unknown"] * 4)[:4]
    else:
        parts = ["unknown"] * 4
    gpu_name, vram_mib, driver_version, compute_cap = parts

    free_lines = _run(["free", "-h"]).splitlines()
    ram_total = free_lines[1].split()[1] if len(free_lines) >= 2 else "unknown"

    df_lines = _run(["df", "-h", "/workspace"]).splitlines()
    disk_free = df_lines[-1].split()[3] if len(df_lines) >= 2 else "unknown"

    return {
        "os": platform.platform(),
        "kernel": platform.release(),
        "python": platform.python_version(),
        "gpu_name": gpu_name,
        "vram_total_mib": vram_mib,
        "driver_version": driver_version,
        "compute_cap": compute_cap,
        "host_cuda_max": host_cuda or "unknown",
        "host_cuda_tuple": host_cuda_tuple,
        "ram_total": ram_total,
        "disk_free_workspace": disk_free,
    }


def compatibility_verdict(
    image: dict[str, object], host: dict[str, object]
) -> tuple[bool, str]:
    """Compare image's CUDA toolkit requirement to host's max CUDA support.

    Returns ``(ok, message)``. ``ok=False`` always carries an actionable
    message (the caller prints remediation hints after this).
    """
    img_cuda = image.get("cuda_tuple")
    host_cuda = host.get("host_cuda_tuple")
    if not isinstance(img_cuda, tuple):
        return False, f"image torch.version.cuda is unparseable: {image.get('cuda_toolkit')!r}"
    if not isinstance(host_cuda, tuple):
        return False, "host CUDA version not found in nvidia-smi output"
    if host_cuda >= img_cuda:
        return (
            True,
            f"host driver supports CUDA {host_cuda[0]}.{host_cuda[1]} "
            f"(image needs CUDA {img_cuda[0]}.{img_cuda[1]}+)",
        )
    return (
        False,
        f"DRIVER TOO OLD: image needs CUDA {img_cuda[0]}.{img_cuda[1]}+, "
        f"host supports CUDA {host_cuda[0]}.{host_cuda[1]} max",
    )


def _print_requirements(img: dict[str, object]) -> None:
    _section("IMAGE REQUIREMENTS")
    _kv("torch", str(img["torch"]))
    _kv("torchvision", str(img["torchvision"]))
    _kv("torchaudio", str(img["torchaudio"]))
    _kv("CUDA toolkit (wheel)", str(img["cuda_toolkit"]))
    _kv("Min NVIDIA driver", str(img["min_driver"]))
    _kv("Target GPU arch", str(img["target_arch"]))
    arch = img.get("arch_flags") or []
    arch_str = " ".join(arch) if isinstance(arch, list) else str(arch)
    _kv("Built for archs", arch_str or "unknown")
    _kv("VRAM resident (est)", f"{img['vram_resident_gb']} GB")
    _kv("VRAM peak (est)", f"{img['vram_peak_gb']} GB")


def _print_host(host: dict[str, object]) -> None:
    _section("HOST MACHINE")
    _kv("OS", str(host["os"]))
    _kv("Kernel", str(host["kernel"]))
    _kv("Python", str(host["python"]))
    _kv("GPU", str(host["gpu_name"]))
    _kv("VRAM", f"{host['vram_total_mib']} MiB")
    _kv("NVIDIA driver", str(host["driver_version"]))
    _kv("Compute capability", str(host["compute_cap"]))
    _kv("Host driver max CUDA", str(host["host_cuda_max"]))
    _kv("Total RAM", str(host["ram_total"]))
    _kv("Disk free /workspace", str(host["disk_free_workspace"]))


def _print_compatibility(img: dict[str, object], host: dict[str, object]) -> int:
    _section("COMPATIBILITY")
    ok, msg = compatibility_verdict(img, host)
    _kv("Verdict", msg, ok=ok)
    if not ok:
        host_cuda = host.get("host_cuda_tuple")
        img_cuda = img.get("cuda_tuple")
        print(flush=True)
        print("  Remediation:", flush=True)
        if isinstance(host_cuda, tuple) and isinstance(img_cuda, tuple):
            host_label = f"cu{host_cuda[0]}{host_cuda[1]}"
            img_label = f"CUDA {img_cuda[0]}.{img_cuda[1]}+"
            print(
                f"  - Roll back the image to a {host_label} build, OR",
                flush=True,
            )
            print(
                f"  - Move to a pod with an NVIDIA driver supporting {img_label}",
                flush=True,
            )
            print(
                "    (RunPod Secure Cloud generally has newer drivers than Community Cloud)",
                flush=True,
            )
        else:
            print("  - Inspect nvidia-smi output and the wheel's CUDA version manually.", flush=True)
        return 2

    cap_tuple = _parse_cuda_version(str(host.get("compute_cap")))
    if cap_tuple == (12, 0):
        _kv("Compute target", "sm_120 Blackwell (expected target)", ok=True)
    elif cap_tuple and cap_tuple[0] == 12:
        _kv(
            "Compute target",
            f"other Blackwell variant sm_{cap_tuple[0]}{cap_tuple[1]} — should still run",
            ok=True,
        )
    elif cap_tuple and cap_tuple[0] >= 9:
        _kv(
            "Compute target",
            f"non-Blackwell sm_{cap_tuple[0]}{cap_tuple[1]} — Blackwell-tuned but should still run",
            ok=True,
        )
    else:
        _kv(
            "Compute target",
            f"unsupported sm_{cap_tuple[0] if cap_tuple else '?'}{cap_tuple[1] if cap_tuple else '?'} (pre-Hopper)",
            ok=False,
        )
        return 3

    return 0


def main() -> int:
    img = image_requirements()
    host = host_machine()
    _print_requirements(img)
    _print_host(host)
    return _print_compatibility(img, host)


if __name__ == "__main__":
    sys.exit(main())
