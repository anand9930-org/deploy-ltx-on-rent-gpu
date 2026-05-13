# Maya I2V aspect_ratio end-to-end test pack

End-to-end test bundle for the new `aspect_ratio` API on branch
`feature/comfyui-bentoml-graph-resolution-management`
(commit `be2d223`, image `<DOCKERHUB_USERNAME>/ltx-video-fa3-teacache:latest`).

Every video the service emits is in exactly one of two buckets:

| `aspect_ratio` | Internal gen size (`/128`) | Output (post-crop) |
|---|---|---|
| `"16:9"` | 1920 × 1152 | **1920 × 1080** |
| `"9:16"` | 1152 × 1920 | **1080 × 1920** |
| `"auto"` | derived from input image orientation; T2V → `"16:9"` |  |

This pack exercises all four resolver paths.

## Test matrix

| # | Test | Input | `aspect_ratio` | Resolver path | Expected output |
|---|---|---|---|---|---|
| T1 | `T1_auto_portrait`  | `inputs/maya_portrait_1280x1920.jpg`      | `"auto"` | image is portrait → `"9:16"` | 1080 × 1920 |
| T2 | `T2_auto_landscape` | `inputs/maya_landscape_mid_1280x720.jpg`  | `"auto"` | image is landscape → `"16:9"` | 1920 × 1080 |
| T3 | `T3_override_169`   | `inputs/maya_portrait_1280x1920.jpg`      | `"16:9"` | explicit override beats portrait orientation | 1920 × 1080 |
| T4 | `T4_override_916`   | `inputs/maya_landscape_top_1280x720.jpg`  | `"9:16"` | explicit override beats landscape orientation; reframes to portrait | 1080 × 1920 |

The two landscape inputs come from the same portrait original:
- **mid** crop is rows `[600, 1320)` — center band, used for the "auto detects landscape" case.
- **top** crop is rows `[0, 720)` — preserves Maya's face, used for the "force-portrait" reframe where keeping the face matters.

## Run

From the repo root:

```bash
uv run --with httpx python3 -u tests/e2e/maya_aspect_ratio/run_tests.py \
    https://<POD_ID>-8000.proxy.runpod.net
```

(Or `cd tests/e2e/maya_aspect_ratio/` first and drop the path prefix —
`run_tests.py` locates `payloads/`, `inputs/`, `outputs/` etc. relative to
its own location, not the cwd.)

The runner submits each test sequentially, polls `/generate/status` (handling
the `"completed"` enum BentoML actually returns — *not* `"success"`),
downloads the resulting MP4 from the Supabase signed URL, runs `ffprobe` to
read its real dimensions, and finally writes a `summary.json` with the
verdict matrix.

Expected dimensions are encoded in `TESTS`; a mismatch is a FAIL.

## Layout

```
inputs/                       — three Maya input variants
  maya_portrait_1280x1920.jpg     # original portrait
  maya_landscape_mid_1280x720.jpg # center 16:9 crop (face partial)
  maya_landscape_top_1280x720.jpg # top 16:9 crop (face fully visible)

payloads/                     — full submit bodies (image_b64 inline)
  T1_auto_portrait.json
  T2_auto_landscape.json
  T3_override_169.json
  T4_override_916.json

outputs/                      — MP4 results from the last run
  T2_auto_landscape_1920x1080.mp4
  T3_override_169_1920x1080.mp4
  T4_override_916_1080x1920.mp4

results/                      — per-test /generate/get response JSON
  T2_auto_landscape_result.json
  T3_override_169_result.json
  T4_override_916_result.json

summary.json                  — verdict matrix (gen times, dims, pass/fail)
run_tests.py                  — the runner
```

## Last successful run (2026-05-13)

All three locally-verified tests passed against pod
`fegju09lfrzdqh-8000.proxy.runpod.net` on the freshly-built
`be2d223` image (warm-pod generation time was 67–88s per 121-frame clip):

| Test | Output | Verdict |
|---|---|---|
| T1 auto + portrait | (not downloaded locally; verified done on pod side) | ✓ pod-side |
| T2 auto + landscape | 1920 × 1080 | ✓ PASS |
| T3 override → 16:9 | 1920 × 1080 | ✓ PASS |
| T4 override → 9:16 | 1080 × 1920 | ✓ PASS |

T1's local download was missed by the original runner because of a status-
string bug (waited for `"success"`, pod returns `"completed"`). The shipped
`run_tests.py` here is the fixed version — running it again will capture T1
properly.

## Notes

- Each payload's `image_b64` is the same Maya image variant the test name
  implies. Payload size is ~400 KB (landscape) or ~960 KB (portrait).
- `num_frames=121` (5s @ 24fps) chosen to keep the four-test loop under
  ~10 min on a warm pod. Bump to `241` (10s) for the canonical workload.
- The pod must already be warm; first cold-start adds ~5–10 min for model
  load. Hit `/readyz` once and wait for `200` before running the suite.
- Result JSONs carry `parameters.aspect_ratio` (the *resolved* value) plus
  the canonical `width`/`height` — verify both against the table above when
  triaging a failure.
