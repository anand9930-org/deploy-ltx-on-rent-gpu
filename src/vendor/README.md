# `src/vendor/` — vendored upstream files

Byte-identical copies of third-party Python files we treat as upstream
modules. Imported through `src/upstream.py`; application code never imports
from `src.vendor` directly.

**Do not modify these files.** They are the canonical source of behavior. If
upstream changes and we want the new version, replace the file verbatim and
bump the SHA below.

## Provenance

| File | Source | Commit |
|---|---|---|
| `ti2vid_triple_stages.py` | https://github.com/eisneim/LTX-2_3_stage_sampling_i2v/blob/main/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_triple_stages.py | `190d474564e78589fd1b7b7109103ded3fab8adb` (main, 2026-04-08) |

## How to refresh a file

```bash
curl -fsSL https://raw.githubusercontent.com/eisneim/LTX-2_3_stage_sampling_i2v/<SHA>/packages/ltx-pipelines/src/ltx_pipelines/ti2vid_triple_stages.py \
    -o src/vendor/ti2vid_triple_stages.py
```

Then update the commit column above and run `pytest tests/`.

## Why vendored, not installed

The triple-stages pipeline lives in a fork branch that we don't track via
the pinned upstream SHA in the Dockerfile. Vendoring the single file (which
imports only public `ltx_core.*` / `ltx_pipelines.utils.*` symbols) is
simpler than swapping the whole package and lets us keep our existing FA3
+ FP8 + compile patches against the official pin.
