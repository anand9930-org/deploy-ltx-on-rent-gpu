#!/usr/bin/env bash
# Drive the FA3 wheel build on a one-shot RunPod CPU pod, from this Mac.
# Idempotent: cache-checks GitHub Releases before spawning, so re-runs
# after a successful build cost $0 and finish in ~1 second.
#
# CPU-only by design: FA3 compilation is 100% CPU work (nvcc/ptxas →
# sm_90a PTX, no device required), so a CPU pod runs the build for
# ~80% less than an H100. See docs/fa3-wheel-process.md.
#
# Flow on cache miss:
#   1. Read RunPod API key from ~/.runpod/config.toml (apikey = "...").
#   2. Read GitHub token from `gh auth token` (or $GH_TOKEN if set).
#   3. Read SSH keypair from ~/.runpod/ssh/RunPod-Key-Go (or $RUNPOD_SSH_KEY_PATH).
#   4. Confirm spend with the user (skip via --yes).
#   5. Spawn a CPU pod via RunPod GraphQL deployCpuPod with NGC
#      pytorch:25.06-py3 image and inline dockerArgs that installs
#      openssh-server + authorizes our public key + starts sshd.
#   6. Trap EXIT to podTerminate — the pod is killed on success, on
#      script error, on Ctrl-C, on a kernel panic on this Mac.
#   7. Poll until desiredStatus=RUNNING and SSH port is reachable.
#   8. scp build_fa3_wheel.sh + a small init wrapper onto the pod;
#      ssh-exec with GH_TOKEN. Stream output via tee.
#   9. Parse FA3_WHEEL_URL/SHA from the streamed output.
#  10. Print the Dockerfile snippet (and machine-readable trailer).
#
# Cost: ~$0.20-0.40 for one cold CPU build (build takes ~25 min). On
# cache hit, $0.
#
# Flags:
#   --yes              Skip the spend-confirmation prompt.
#   --dry-run          Print the GraphQL spawn payload + skip API calls.
#   --instance <id>    Override RunPod CPU instance id (default:
#                      "cpu3g-32-128" — 32 vCPU + 128 GB RAM, general).
#                      cpu3c (compute, 2 GB/vCPU) caps at 64 GB and
#                      OOMs cicc on FA3's bwd_hdim192 instantiations.
#   --secure-cloud     Spawn on RunPod Secure Cloud (cloudType: SECURE)
#                      instead of Community. Bigger inventory but ~30-50%
#                      pricier. Use when COMMUNITY hits SUPPLY_CONSTRAINT.
#   --keep-pod         Don't terminate the pod on exit (for debugging).
#                      You'll be charged until you `runpodctl stop pod`.

set -euo pipefail

# ============================================================================
# Defaults & arg parsing
# ============================================================================
FA3_COMMIT_DEFAULT="6c73fb506fa84424c3fa04880f85c789b4a498e2"
NGC_TAG_DEFAULT="25.06-py3"
CPU_INSTANCE_DEFAULT="cpu3g-32-128"   # 32 vCPU, 128 GB RAM (cpu3g general @ 4 GB/vCPU).

YES=0
DRY_RUN=0
KEEP_POD=0
SECURE_CLOUD=0
CPU_INSTANCE="$CPU_INSTANCE_DEFAULT"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yes) YES=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --keep-pod) KEEP_POD=1; shift ;;
    --secure-cloud) SECURE_CLOUD=1; shift ;;
    --instance) CPU_INSTANCE="$2"; shift 2 ;;
    -h|--help) sed -n '2,/^set -/p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

CLOUD_TYPE="COMMUNITY"
(( SECURE_CLOUD )) && CLOUD_TYPE="SECURE"

FA3_COMMIT="${FA3_COMMIT:-$FA3_COMMIT_DEFAULT}"
NGC_TAG="${NGC_TAG:-$NGC_TAG_DEFAULT}"
SHORT="${FA3_COMMIT:0:8}"
TAG="fa3-ngc${NGC_TAG%-py3}-${SHORT}"

# ============================================================================
# Prerequisite checks
# ============================================================================
need() { command -v "$1" >/dev/null || { echo "missing required tool: $1" >&2; exit 1; }; }
need jq; need curl; need ssh; need scp; need gh; need nc

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_SCRIPT="$REPO_ROOT/scripts/build_fa3_wheel.sh"
[[ -f "$BUILD_SCRIPT" ]] || { echo "missing $BUILD_SCRIPT" >&2; exit 1; }
# `gh repo view` reads cwd's git config — pin to the repo this script
# lives in so it doesn't return a sibling repo when invoked from elsewhere.
cd "$REPO_ROOT"

GH_REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
echo "[runpod] repo=${GH_REPO} tag=${TAG} fa3=${FA3_COMMIT} ngc=${NGC_TAG}"

# ============================================================================
# Cache check (free path)
# ============================================================================
if gh release view "$TAG" --repo "$GH_REPO" >/dev/null 2>&1; then
  echo "[runpod] cache hit — Release '${TAG}' already exists. No pod needed."
  exec env GH_REPO="$GH_REPO" FA3_COMMIT="$FA3_COMMIT" NGC_TAG="$NGC_TAG" \
    bash "$BUILD_SCRIPT"
fi

# ============================================================================
# Secrets — RunPod API key, GitHub token, SSH keypair
# ============================================================================
if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
  RUNPOD_CONFIG="${RUNPOD_CONFIG:-$HOME/.runpod/config.toml}"
  [[ -f "$RUNPOD_CONFIG" ]] || {
    echo "no RUNPOD_API_KEY in env and no $RUNPOD_CONFIG — run 'runpodctl config'" >&2
    exit 1
  }
  RUNPOD_API_KEY="$(awk -F'=' '
    $1 ~ /^[[:space:]]*apikey[[:space:]]*$/ {
      gsub(/^[[:space:]"'"'"']+|[[:space:]"'"'"']+$/, "", $2); print $2; exit
    }' "$RUNPOD_CONFIG")"
fi
[[ -n "$RUNPOD_API_KEY" ]] || { echo "could not load RunPod apikey" >&2; exit 1; }

GH_TOKEN="${GH_TOKEN:-$(gh auth token 2>/dev/null || true)}"
[[ -n "$GH_TOKEN" ]] || { echo "no gh token (run 'gh auth login')" >&2; exit 1; }

SSH_KEY="${RUNPOD_SSH_KEY_PATH:-$HOME/.runpod/ssh/RunPod-Key-Go}"
[[ -f "$SSH_KEY" && -f "$SSH_KEY.pub" ]] || {
  echo "missing SSH keypair at $SSH_KEY(.pub)" >&2; exit 1; }
PUBLIC_KEY="$(cat "$SSH_KEY.pub")"

# ============================================================================
# GraphQL helper
# ============================================================================
RUNPOD_API_URL="https://api.runpod.io/graphql"

rp_gql() {
  # $1 = query string (with $varname placeholders), rest = jq --argjson/--arg pairs.
  local query="$1"; shift
  local payload
  payload=$(jq -nc --arg q "$query" "$@" '{query: $q, variables: $ARGS.named | del(.q)}')
  local resp
  resp=$(curl -fsSL -H 'content-type: application/json' \
        -H "authorization: Bearer $RUNPOD_API_KEY" \
        --data-binary "$payload" "$RUNPOD_API_URL")
  if echo "$resp" | jq -e '.errors' >/dev/null 2>&1; then
    echo "[runpod] GraphQL error:" >&2
    echo "$resp" | jq '.errors' >&2
    return 1
  fi
  echo "$resp"
}

# ============================================================================
# Build the dockerArgs — install sshd, authorize key, start sshd in foreground.
#
# We pass the init body as a base64 blob to dockerize cleanly: base64
# output is alphanumeric + `+/=` only, so it survives any tokenizer
# RunPod's runtime might apply to the dockerArgs string. The container
# decodes and executes via `bash -c "echo <b64> | base64 -d | bash"`.
# Avoid single quotes inside the body so a future shell-form CMD parse
# can't trip on them.
# ============================================================================
read -r -d '' DOCKER_ARGS_BODY <<'BODY' || true
set -e
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq openssh-server >/dev/null
mkdir -p /run/sshd /root/.ssh /etc/ssh/sshd_config.d
chmod 700 /root/.ssh
printf "%s\n" "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
chmod 600 /root/.ssh/authorized_keys
ssh-keygen -A
{
  echo "PermitRootLogin prohibit-password"
  echo "PasswordAuthentication no"
} > /etc/ssh/sshd_config.d/99-runpod.conf
exec /usr/sbin/sshd -D -e
BODY
DOCKER_ARGS_B64=$(printf '%s' "$DOCKER_ARGS_BODY" | base64 | tr -d '\n')
DOCKER_ARGS="bash -c \"echo ${DOCKER_ARGS_B64} | base64 -d | bash\""

# ============================================================================
# Cost confirmation
# ============================================================================
ESTIMATED_COST="~0.40"  # 32 vCPU CPU pod ≈ $0.40-0.80/hr × ~30 min

cat <<EOF

About to spawn a CPU pod ($CPU_INSTANCE, cloudType=$CLOUD_TYPE)
running nvcr.io/nvidia/pytorch:$NGC_TAG to build the FA3 wheel and
publish it as the GitHub Release '$TAG' on $GH_REPO.

Estimated cost: \$$ESTIMATED_COST (build ~25-30 min + provisioning).
Pod is auto-terminated on script exit (success OR failure).

EOF

if (( ! YES && ! DRY_RUN )); then
  read -r -p "Proceed? [y/N] " ans
  [[ "$ans" =~ ^[Yy]$ ]] || { echo "aborted"; exit 1; }
fi

# ============================================================================
# Spawn pod
# ============================================================================
# `deployCpuPod` uses `instanceId` (e.g. "cpu3g-32-128") in place of
# gpuTypeId/gpuCount/bidPerGpu/supportPublicIp. SSH routes through the
# proxy when `startSsh: true` + `ports: "22/tcp"` — same wire format on
# the pod-query response (`runtime.ports[]`).
SPAWN_QUERY='mutation Spawn($pk: String!, $da: String!, $img: String!, $name: String!, $iid: String!, $cloud: CloudTypeEnum!) {
  deployCpuPod(input: {
    cloudType: $cloud,
    instanceId: $iid,
    containerDiskInGb: 60,
    name: $name,
    imageName: $img,
    dockerArgs: $da,
    ports: "22/tcp",
    startSsh: true,
    env: [{ key: "PUBLIC_KEY", value: $pk }]
  }) { id desiredStatus costPerHr }
}'
SPAWN_FIELD="deployCpuPod"

if (( DRY_RUN )); then
  echo "=== dry-run: would POST ==="
  jq -n --arg q "$SPAWN_QUERY" \
        --arg pk "$PUBLIC_KEY" --arg da "$DOCKER_ARGS" \
        --arg img "nvcr.io/nvidia/pytorch:${NGC_TAG}" \
        --arg name "fa3-build-${SHORT}" \
        --arg iid "$CPU_INSTANCE" \
        --arg cloud "$CLOUD_TYPE" \
        '{query: $q, variables: ($ARGS.named | del(.q))}'
  echo
  echo "=== dry-run: would scp & ssh-run ==="
  echo "scp -i $SSH_KEY -P <port> $BUILD_SCRIPT root@<ip>:/root/"
  echo "ssh -i $SSH_KEY -p <port> root@<ip> '<wrapper that sets GH_TOKEN, runs build script>'"
  exit 0
fi

RESP=$(rp_gql "$SPAWN_QUERY" \
  --arg pk "$PUBLIC_KEY" --arg da "$DOCKER_ARGS" \
  --arg img "nvcr.io/nvidia/pytorch:${NGC_TAG}" \
  --arg name "fa3-build-${SHORT}" \
  --arg iid "$CPU_INSTANCE" \
  --arg cloud "$CLOUD_TYPE")
POD_ID=$(echo "$RESP" | jq -r ".data.${SPAWN_FIELD}.id")
COST_HR=$(echo "$RESP" | jq -r ".data.${SPAWN_FIELD}.costPerHr // \"?\"")
[[ -n "$POD_ID" && "$POD_ID" != "null" ]] || {
  echo "[runpod] pod creation returned no id" >&2; echo "$RESP" >&2; exit 1; }
echo "[runpod] pod ${POD_ID} provisioned at \$${COST_HR}/hr"

# ============================================================================
# Trap-based cleanup — fire podTerminate no matter how we exit
# ============================================================================
cleanup() {
  local rc=$?
  if (( KEEP_POD )); then
    echo "[runpod] --keep-pod set; pod $POD_ID is still running. Stop it manually:"
    echo "         runpodctl stop pod $POD_ID  (or use the web UI)"
  elif [[ -n "${POD_ID:-}" ]]; then
    echo "[runpod] terminating pod $POD_ID (exit code $rc)"
    rp_gql 'mutation T($id: String!) { podTerminate(input: { podId: $id }) }' \
      --arg id "$POD_ID" >/dev/null 2>&1 || \
      echo "[runpod] WARN: podTerminate failed; please check the web UI" >&2
  fi
}
trap cleanup EXIT INT TERM

# ============================================================================
# Poll until SSH is reachable
# ============================================================================
# Secure Cloud H100 pods commonly take 4-6 min just to get to RUNNING; give
# them more headroom than COMMUNITY before declaring failure.
SSH_TIMEOUT_S="${SSH_TIMEOUT_S:-720}"
echo "[runpod] waiting for pod to reach RUNNING + SSH up (timeout ${SSH_TIMEOUT_S}s)…"
DEADLINE=$(( $(date +%s) + SSH_TIMEOUT_S ))
SSH_IP=""; SSH_PORT=""
LAST_STATUS=""
LAST_PORTS_JSON=""
while (( $(date +%s) < DEADLINE )); do
  RESP=$(rp_gql 'query S($id: String!) { pod(input: { podId: $id }) {
    desiredStatus runtime { ports { ip isIpPublic privatePort publicPort type } }
  } }' --arg id "$POD_ID") || true
  STATUS=$(echo "$RESP" | jq -r '.data.pod.desiredStatus // "UNKNOWN"')
  PORTS_JSON=$(echo "$RESP" | jq -c '.data.pod.runtime.ports // null')
  SSH_IP=$(echo "$RESP" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | .ip' | head -n1)
  SSH_PORT=$(echo "$RESP" | jq -r '.data.pod.runtime.ports[]? | select(.privatePort==22 and .isIpPublic==true) | .publicPort' | head -n1)
  if [[ "$STATUS" != "$LAST_STATUS" || "$PORTS_JSON" != "$LAST_PORTS_JSON" ]]; then
    echo
    echo "[runpod] status=${STATUS} ports=${PORTS_JSON}"
    LAST_STATUS="$STATUS"; LAST_PORTS_JSON="$PORTS_JSON"
  fi
  if [[ "$STATUS" == "RUNNING" && -n "$SSH_IP" && "$SSH_IP" != "null" ]]; then
    if nc -z -w 2 "$SSH_IP" "$SSH_PORT" 2>/dev/null; then
      echo "[runpod] pod is up: ssh -i $SSH_KEY -p $SSH_PORT root@$SSH_IP"
      # nc-zero says TCP-accept succeeded; sshd's auth/banner exchange
      # can lag by a couple seconds, so give it a moment before scp.
      sleep 3
      break
    fi
  fi
  printf '.'
  sleep 5
done
echo
if [[ -z "$SSH_IP" || -z "$SSH_PORT" || "$SSH_IP" == "null" ]]; then
  echo "[runpod] SSH did not come up." >&2
  echo "[runpod] last status=${LAST_STATUS}" >&2
  echo "[runpod] last ports=${LAST_PORTS_JSON}" >&2
  exit 1
fi

# ============================================================================
# Stage build script + init wrapper, then ssh-exec
# ============================================================================
TMPDIR_LOCAL="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_LOCAL"; cleanup' EXIT INT TERM

cat > "$TMPDIR_LOCAL/run.sh" <<'INIT'
#!/usr/bin/env bash
# Pod-side init: install gh CLI, then run the build script.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

if ! command -v gh >/dev/null; then
  apt-get update -qq
  apt-get install -y -qq curl ca-certificates gnupg git >/dev/null
  curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
    | gpg --dearmor -o /usr/share/keyrings/githubcli-archive-keyring.gpg
  echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
    > /etc/apt/sources.list.d/github-cli.list
  apt-get update -qq
  apt-get install -y -qq gh >/dev/null
fi

# gh CLI auto-uses GH_TOKEN when set
gh auth status >/dev/null

cd /root
exec bash /root/build_fa3_wheel.sh
INIT
chmod +x "$TMPDIR_LOCAL/run.sh"

SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
          -o ServerAliveInterval=30 -o ServerAliveCountMax=20)

echo "[runpod] uploading build script + init wrapper"
scp "${SSH_OPTS[@]}" -P "$SSH_PORT" "$BUILD_SCRIPT" "$TMPDIR_LOCAL/run.sh" \
    "root@$SSH_IP:/root/"

# Compose the remote command with safely-escaped env vars.
REMOTE_CMD=$(printf 'export GH_TOKEN=%q GH_REPO=%q FA3_COMMIT=%q NGC_TAG=%q; bash /root/run.sh' \
  "$GH_TOKEN" "$GH_REPO" "$FA3_COMMIT" "$NGC_TAG")

echo "[runpod] running build (~25 min). Streaming pod stdout…"
echo "─── pod output ───"
LOG="$TMPDIR_LOCAL/build.log"
set +e
ssh "${SSH_OPTS[@]}" -p "$SSH_PORT" "root@$SSH_IP" "$REMOTE_CMD" 2>&1 | tee "$LOG"
SSH_RC=${PIPESTATUS[0]}
set -e
echo "─── /pod output ──"

if (( SSH_RC != 0 )); then
  echo "[runpod] build failed (ssh rc=$SSH_RC). Pod will be terminated by trap." >&2
  exit "$SSH_RC"
fi

# ============================================================================
# Parse machine-readable trailer from build script output
# ============================================================================
URL=$(grep -E '^FA3_WHEEL_URL=' "$LOG" | tail -n1 | cut -d= -f2-)
SHA=$(grep -E '^FA3_WHEEL_SHA256=' "$LOG" | tail -n1 | cut -d= -f2-)
[[ -n "$URL" && -n "$SHA" ]] || {
  echo "[runpod] could not parse FA3_WHEEL_URL/SHA256 from build output" >&2
  exit 1; }

cat <<EOF

✓ FA3 wheel built and cached.

=== Dockerfile snippet ===

ARG FA3_WHEEL_URL=${URL}
ARG FA3_WHEEL_SHA256=${SHA}

FA3_WHEEL_URL=${URL}
FA3_WHEEL_SHA256=${SHA}
EOF
