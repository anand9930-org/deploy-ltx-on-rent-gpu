#!/usr/bin/env bash
# Idempotent FA3 wheel cache for the NGC pytorch base image.
#
# Runs natively inside an NGC pytorch container (where torch.__version__
# contains "+nv"). The orchestrator (scripts/runpod_build_fa3_wheel.sh)
# spawns such a pod for you; on an existing pod, copy this script over
# and run directly.
#
# Inputs (env vars, optional — defaults pinned to current Dockerfile state):
#   FA3_COMMIT  Upstream Dao-AILab/flash-attention SHA to build at.
#   NGC_TAG     nvcr.io/nvidia/pytorch base image tag (e.g. 25.06-py3).
#   GH_REPO     owner/name of the repo whose Releases store the cache.
#               Default: derived from `gh repo view`.
#
# Output: a copy/paste-able snippet of the two `ARG` lines for Dockerfile,
# plus machine-readable lines `FA3_WHEEL_URL=...` / `FA3_WHEEL_SHA256=...`
# (no leading whitespace) so a parent orchestrator can grep them out.
#
# Why GitHub Releases: public HTTPS, no auth in the Docker build path,
# tag-as-cache-key matches the (FA3 SHA, NGC tag) tuple that determines
# ABI compatibility. See docs/fa3-wheel-process.md.

set -euo pipefail

FA3_COMMIT="${FA3_COMMIT:-6c73fb506fa84424c3fa04880f85c789b4a498e2}"
NGC_TAG="${NGC_TAG:-25.06-py3}"
SHORT="${FA3_COMMIT:0:8}"
TAG="fa3-ngc${NGC_TAG%-py3}-${SHORT}"

if [[ -z "${GH_REPO:-}" ]]; then
  GH_REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
fi
echo "[fa3-cache] repo=${GH_REPO} tag=${TAG} ngc=${NGC_TAG} fa3=${FA3_COMMIT}"

public_url() {
  local name="$1"
  # GitHub's browser_download_url URL-encodes `+` as `%2B`. Mirror that
  # so the Dockerfile's `basename | sed 's/%2B/+/g'` decoder remains
  # accurate (curl handles either form, but consistency aids debugging).
  local encoded="${name//+/%2B}"
  printf 'https://github.com/%s/releases/download/%s/%s' "$GH_REPO" "$TAG" "$encoded"
}

# ---- Cache check -----------------------------------------------------------
if gh release view "$TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
  echo "[fa3-cache] cache hit — Release '${TAG}' already exists"
  WHEEL_NAME=$(gh release view "$TAG" --repo "$GH_REPO" --json assets \
        -q '.assets[] | select(.name | endswith(".whl")) | .name' | head -n1)
  SHA=$(gh release view "$TAG" --repo "$GH_REPO" --json body -q '.body' \
        | grep -oE '[a-f0-9]{64}' | head -n1)
  if [[ -z "${WHEEL_NAME}" || -z "${SHA}" ]]; then
    echo "[fa3-cache] ERROR: existing Release '${TAG}' is missing the wheel" >&2
    echo "            asset or sha256 in its body. Delete it and re-run:" >&2
    echo "            gh release delete '${TAG}' --repo '${GH_REPO}' --yes" >&2
    exit 1
  fi
  URL=$(public_url "$WHEEL_NAME")

  cat <<EOF

=== Dockerfile snippet — paste under '# ---- FlashAttention 3 (sm_90 only) ----' ===

ARG FA3_WHEEL_URL=${URL}
ARG FA3_WHEEL_SHA256=${SHA}

EOF
  # Machine-readable trailer (column-1 anchored — orchestrators can grep these).
  echo "FA3_WHEEL_URL=${URL}"
  echo "FA3_WHEEL_SHA256=${SHA}"
  exit 0
fi

# ---- Cache miss — build path -----------------------------------------------
echo "[fa3-cache] cache miss — wheel build required (~25 min)"

# Sanity check: refuse to build outside an NGC pytorch container — the
# whole point of this script is to link against NGC's torch ABI.
if ! command -v python >/dev/null || \
   ! python -c "import sys, torch; sys.exit(0 if '+nv' in torch.__version__ else 1)" 2>/dev/null; then
  echo "[fa3-cache] ERROR: not inside an NGC pytorch container (torch.__version__ lacks '+nv')." >&2
  echo "            Run this script on a RunPod pod started from nvcr.io/nvidia/pytorch:${NGC_TAG}," >&2
  echo "            or use scripts/runpod_build_fa3_wheel.sh to spawn one." >&2
  exit 1
fi

# MAX_JOBS = min(cpu_cores, mem_aware_cap, 16). The memory cap reflects
# FA3's heaviest CUTLASS backward kernels (flash_bwd_hdim192_*), where
# `cicc` peaks at ~6-8 GB per process. We reserve 8 GB for OS+ninja+linker
# overhead and divide the rest by 8. Without this, a 64 GB CPU pod with
# nproc=32 hits the 16-cap and OOM-kills cicc on the bwd_hdim192 unit.
JOBS_CPU="${MAX_JOBS:-$(nproc 2>/dev/null || echo 4)}"
MEM_GB=$(awk '/^MemTotal:/ {printf "%d\n", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo 64)
JOBS_MEM=$(( (MEM_GB - 8) / 8 ))
(( JOBS_MEM < 1 )) && JOBS_MEM=1
JOBS=$(( JOBS_CPU < JOBS_MEM ? JOBS_CPU : JOBS_MEM ))
(( JOBS > 16 )) && JOBS=16
echo "[fa3-cache] MAX_JOBS=${JOBS} (cpu=${JOBS_CPU} mem_cap=${JOBS_MEM} total_gb=${MEM_GB})"

OUT="$(mktemp -d)"
trap 'rm -rf "$OUT"' EXIT

apt-get update && apt-get install -y --no-install-recommends git ninja-build
rm -rf /tmp/fa3
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/fa3
( cd /tmp/fa3 && git checkout "${FA3_COMMIT}" )
( cd /tmp/fa3/hopper \
    && TORCH_CUDA_ARCH_LIST='9.0' MAX_JOBS="${JOBS}" \
       python setup.py bdist_wheel -d "${OUT}" )

WHEEL=$(ls "${OUT}"/*.whl 2>/dev/null | head -n1)
if [[ -z "${WHEEL}" || ! -f "${WHEEL}" ]]; then
  echo "[fa3-cache] ERROR: build did not produce a .whl in ${OUT}" >&2
  exit 1
fi
SHA=$(sha256sum "$WHEEL" | awk '{print $1}')
echo "[fa3-cache] wheel=$(basename "$WHEEL") sha256=${SHA}"

# ---- Upload to GitHub Releases (BEFORE smoke) ------------------------------
# The wheel is the expensive artifact. Upload immediately so a smoke
# failure can't cost us another 25-30 min of compute on the next run.
gh release create "$TAG" "$WHEEL" --repo "$GH_REPO" \
  --title "FA3 wheel — NGC ${NGC_TAG} @ ${SHORT}" \
  --notes "Built inside nvcr.io/nvidia/pytorch:${NGC_TAG}.

FA3 commit: ${FA3_COMMIT}
sha256: ${SHA}

Auto-generated by scripts/build_fa3_wheel.sh — do not edit the body
without preserving the sha256 line, the script parses it on cache hits."

URL=$(public_url "$(basename "$WHEEL")")

# ---- Smoke test (best-effort, post-upload) ---------------------------------
# Failure here is logged but does NOT abort the script — wheel is already
# uploaded and reachable to the orchestrator's cache logic. Smoke failure
# may be ergonomic (PEP 668, missing import-time deps) rather than ABI.
if pip install --break-system-packages --no-deps "$WHEEL" 2>&1; then
  python -c 'import flash_attn_interface as f; \
    assert hasattr(f, "flash_attn_func"); \
    print("[fa3-cache] build-side smoke OK:", getattr(f, "__version__", "unknown"))' \
    || echo "[fa3-cache] WARN: smoke import failed; wheel was uploaded anyway" >&2
else
  echo "[fa3-cache] WARN: smoke pip install failed; wheel was uploaded anyway" >&2
fi

cat <<EOF

=== Dockerfile snippet — paste under '# ---- FlashAttention 3 (sm_90 only) ----' ===

ARG FA3_WHEEL_URL=${URL}
ARG FA3_WHEEL_SHA256=${SHA}

EOF
echo "FA3_WHEEL_URL=${URL}"
echo "FA3_WHEEL_SHA256=${SHA}"
