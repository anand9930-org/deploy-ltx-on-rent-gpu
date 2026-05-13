#!/bin/bash
# BentoML setup script — runs during Docker image build.
#
# No-op on the Blackwell branch: the ComfyUI graph pipeline
# (`src/pipeline/triple_stages_comfyui_graph.py`) runs ComfyUI **core** nodes
# and explicitly opts out of `ltx-core` / `ltx-pipelines` (see its module
# docstring). Installing them only drags in `tensorrt-llm==1.0.0`, which is
# H100/Hopper (sm_90) only and unavailable for Blackwell sm_122. The H100 W8A8
# path lives on `main`.
set -e

echo "(setup.sh no-op on Blackwell branch — ComfyUI path does not use ltx-core)"
