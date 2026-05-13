#!/bin/bash
# BentoML setup script — runs during Docker image build.
# Clones the official LTX-2 repo and installs ltx-core / ltx-pipelines.
# The [fp8-trtllm] extra pulls tensorrt-llm==1.0.0 for the H100 W8A8
# scaled_mm path; fetched from pypi.nvidia.com.
set -e

git lfs install
git clone --depth 1 https://github.com/Lightricks/LTX-2.git /app/LTX-2
pip install --no-cache-dir \
    --extra-index-url https://pypi.nvidia.com \
    -e "/app/LTX-2/packages/ltx-core[fp8-trtllm]" \
    -e /app/LTX-2/packages/ltx-pipelines
