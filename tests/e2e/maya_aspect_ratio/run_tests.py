#!/usr/bin/env python3
"""End-to-end aspect_ratio test runner for the LTX-2.3 ComfyUI service.

Submits four prepared payloads against a running pod, polls each task to
completion (handles BentoML's "completed" status, not "success"), downloads
the resulting MP4, ffprobes its actual dimensions, and writes a summary.

Usage:
    uv run --with httpx python3 -u run_tests.py <POD_URL>
    # e.g. https://<RUNPOD_ID>-8000.proxy.runpod.net

Layout this script expects (relative to its own location):
    payloads/T{1..4}_*.json       — full submit bodies (image_b64 inline)
    inputs/                       — reference copies of the input images
    outputs/                      — MP4s land here, named per test
    results/                      — per-test result JSON from /generate/get
    summary.json                  — top-level pass/fail matrix
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent

# (test_name, payload_filename, expected_output_dims, output_filename)
TESTS = [
    ("T1_auto_portrait",  "T1_auto_portrait.json",  (1080, 1920), "T1_auto_portrait_1080x1920.mp4"),
    ("T2_auto_landscape", "T2_auto_landscape.json", (1920, 1080), "T2_auto_landscape_1920x1080.mp4"),
    ("T3_override_169",   "T3_override_169.json",   (1920, 1080), "T3_override_169_1920x1080.mp4"),
    ("T4_override_916",   "T4_override_916.json",   (1080, 1920), "T4_override_916_1080x1920.mp4"),
]

POLL_INTERVAL = 5.0
POLL_TIMEOUT = 30 * 60.0
HTTP_TIMEOUT = 120.0
TERMINAL_STATUSES = ("completed", "success", "failure", "failed", "cancelled")


def ffprobe_dims(mp4: Path) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(mp4)],
        check=True, capture_output=True, text=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    return s["width"], s["height"]


def run_one(client: httpx.Client, base: str, name: str, payload_file: str,
            expected: tuple[int, int], output_filename: str) -> dict:
    print(f"\n──── {name} ────")
    payload = json.loads((HERE / "payloads" / payload_file).read_text())
    print(f"  aspect_ratio={payload['aspect_ratio']!r}, "
          f"num_frames={payload['num_frames']}, seed={payload['seed']}")

    t0 = time.time()
    r = client.post(f"{base}/generate/submit", json=payload, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    body = r.json()
    task_id = body.get("task_id") or body.get("id")
    print(f"  submitted → task_id = {task_id}")

    last_status = None
    while True:
        r = client.get(f"{base}/generate/status", params={"task_id": task_id}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        b = r.json()
        s = b.get("status", "unknown")
        if s != last_status:
            print(f"    [{time.time() - t0:6.1f}s] status={s}")
            last_status = s
        if s in TERMINAL_STATUSES:
            break
        if time.time() - t0 > POLL_TIMEOUT:
            return {"name": name, "ok": False, "task_id": task_id, "reason": "timeout"}
        time.sleep(POLL_INTERVAL)

    if last_status in ("failure", "failed", "cancelled"):
        return {"name": name, "ok": False, "task_id": task_id, "reason": f"status={last_status}"}

    r = client.get(f"{base}/generate/get", params={"task_id": task_id}, timeout=HTTP_TIMEOUT)
    r.raise_for_status()
    res = r.json()
    (HERE / "results" / f"{name}_result.json").write_text(json.dumps(res, indent=2))

    p = res.get("parameters", {})
    print(f"  parameters: aspect_ratio={p.get('aspect_ratio')!r}, "
          f"width={p.get('width')}, height={p.get('height')}, "
          f"gen_time={res.get('generation_time_seconds')}s")

    video_url = res.get("video_url")
    if not video_url:
        return {"name": name, "ok": False, "task_id": task_id,
                "reason": "no video_url in result", "parameters": p}

    out_mp4 = HERE / "outputs" / output_filename
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    with client.stream("GET", video_url, timeout=HTTP_TIMEOUT) as st:
        st.raise_for_status()
        with out_mp4.open("wb") as f:
            for chunk in st.iter_bytes(1 << 20):
                f.write(chunk)
    dims = ffprobe_dims(out_mp4)
    ok = tuple(dims) == expected
    print(f"  downloaded → {out_mp4.name} ({out_mp4.stat().st_size:,} bytes)")
    print(f"  MP4 dims = {dims[0]}x{dims[1]}  expected {expected[0]}x{expected[1]}  "
          + ("✓ PASS" if ok else "✗ FAIL"))

    return {
        "name": name, "ok": ok, "task_id": task_id,
        "expected": list(expected), "actual_dims": list(dims),
        "parameters": p, "generation_time_seconds": res.get("generation_time_seconds"),
        "output_file": str(out_mp4.relative_to(HERE)),
        "elapsed_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    if len(sys.argv) != 2:
        sys.exit("usage: run_tests.py <POD_URL>")
    base = sys.argv[1].rstrip("/")

    with httpx.Client(follow_redirects=True) as client:
        try:
            r = client.get(f"{base}/readyz", timeout=10.0)
            print(f"readyz → {r.status_code}")
        except httpx.HTTPError as e:
            print(f"readyz check failed (continuing): {e}")

        results = []
        for name, payload_file, expected, output_filename in TESTS:
            try:
                results.append(run_one(client, base, name, payload_file, expected, output_filename))
            except Exception as e:
                print(f"  ✗ EXCEPTION: {e!r}")
                results.append({"name": name, "ok": False, "reason": repr(e)})

    print("\n──── summary ────")
    for r in results:
        flag = "✓ PASS" if r.get("ok") else "✗ FAIL"
        extra = (f"  {r['actual_dims'][0]}x{r['actual_dims'][1]} (expected {r['expected'][0]}x{r['expected'][1]})"
                 if "actual_dims" in r else f"  {r.get('reason','?')}")
        print(f"  {flag}  {r['name']}{extra}")

    (HERE / "summary.json").write_text(json.dumps({"results": results}, indent=2))
    print(f"\nfull results → {HERE / 'summary.json'}")
    return 0 if all(r.get("ok") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
