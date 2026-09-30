#!/usr/bin/env bash
# Start the cookbook's vLLM server (Qwen3.8-27B-NVFP4 served as "primary") on NVIDIA DGX Spark.
# EXPERIMENTAL.
#
#   ./serve.sh --dry-run          print the checks and the docker run command, start nothing
#   ./serve.sh                    start the tested configuration (MTP 3 + prefix caching)
#   ./serve.sh --mtp-off          fallback: no speculative decoding (needed on an image without PR #50021)
#   ./serve.sh --wait             start, then wait until the container reports healthy (first start ~10 min)
#
# Safety: this script only ever creates ONE new container, named $NAME. It refuses to start when the port
# is already in use, when a container called $NAME already exists (running or stopped), or when another
# vLLM container is running (override: --allow-second-server). It never stops, removes, restarts or
# changes any other container, never pulls an image and, unless ALLOW_DOWNLOAD=1, never downloads weights.
#
# Every setting below is an environment variable with the tested value as its default.
set -euo pipefail

# --- image and container ------------------------------------------------------------------------
IMAGE="${IMAGE:-spark-vllm-agent:0.30-pr50021}"   # built by ./build.sh
NAME="${NAME:-vllm-agent}"                        # container name
RESTART_POLICY="${RESTART_POLICY:-unless-stopped}"
# The tested server ran --privileged. It should not need it: the GPU comes in through the NVIDIA
# container runtime (--gpus all). Running without it is the default here but UNTESTED on a Spark; set
# PRIVILEGED=1 (opt-in) to reproduce the tested server exactly if the GPU is not visible without it.
PRIVILEGED="${PRIVILEGED:-0}"

# --- network ------------------------------------------------------------------------------------
PORT="${PORT:-8000}"
# The tested server bound 0.0.0.0 (all interfaces). The default here is loopback only, because vLLM has
# no authentication unless API_KEY is set. Use HOST=0.0.0.0 (plus API_KEY) to serve other machines;
# a non-loopback HOST without API_KEY is refused.
HOST="${HOST:-127.0.0.1}"
# Passed to the container as the env var VLLM_API_KEY (vLLM reads it), never as --api-key on a command
# line, where it would show in `ps`. The key only guards /v1/*: /metrics and /health stay
# unauthenticated, so firewall the port when HOST is not loopback.
API_KEY="${API_KEY:-}"

# --- model and Hugging Face cache ---------------------------------------------------------------
MODEL="${MODEL:-nvidia/Qwen3.8-27B-NVFP4}"      # HF repo id (resolved from the cache) or a path inside the container
# Pinned to the commit the tested server runs, so a later push to the repo can't silently change the
# weights under the published numbers. Empty = no pin (vLLM uses "main"). Ignored when MODEL is a path.
MODEL_REVISION="${MODEL_REVISION:-482ca0f3832238542f8f5295dde86b5f22711d80}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-primary}"
HF_CACHE_HOST="${HF_HOME:-$HOME/.cache/huggingface}"   # host HF_HOME; must contain hub/models--nvidia--Qwen3.8-27B-NVFP4
HF_CACHE_CONTAINER="${HF_CACHE_CONTAINER:-/hf-cache}"  # mount point; set as HF_HOME inside the container
ALLOW_DOWNLOAD="${ALLOW_DOWNLOAD:-0}"           # 1 = let vLLM download missing weights (about 21 GB)

# --- tested vLLM flags --------------------------------------------------------------------------
MAX_MODEL_LEN="${MAX_MODEL_LEN:-131072}"                    # 128k context
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
KV_CACHE_MEMORY_BYTES="${KV_CACHE_MEMORY_BYTES:-21474836480}"   # KV cache pinned at 20 GiB (~520k tokens)
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.55}"    # startup free-memory check only (see README)
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
REASONING_PARSER="${REASONING_PARSER:-qwen3}"
LOAD_FORMAT="${LOAD_FORMAT:-fastsafetensors}"               # empty = vLLM default loader
MTP_TOKENS="${MTP_TOKENS:-3}"                               # MTP speculative tokens; 0 = off
PREFIX_CACHING="${PREFIX_CACHING:-1}"                       # 1 = --enable-prefix-caching
MAMBA_CACHE_MODE="${MAMBA_CACHE_MODE:-align}"               # used with prefix caching (hybrid GDN layers)
LANGUAGE_MODEL_ONLY="${LANGUAGE_MODEL_ONLY:-1}"             # required with MTP on vLLM 0.30 (vLLM #58203)
EXTRA_ARGS="${EXTRA_ARGS:-}"                                # extra `vllm serve` flags, split on spaces
# The PR #50021 head commit build.sh pins (its PR50021_HEAD); the image label should carry it.
PR50021_HEAD="71d7c782ca4230b556ced08f78fcac865b64a15d"

# --- healthcheck --------------------------------------------------------------------------------
HEALTH_START_PERIOD="${HEALTH_START_PERIOD:-900s}"          # first start takes about 9-10 min

DRY_RUN=0
WAIT=0
ALLOW_SECOND=0
FORCE_MTP=0

usage() {
  sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Options:
  --dry-run              print the checks and the command; start nothing
  --mtp-off              no speculative decoding (same as MTP_TOKENS=0)
  --no-prefix-caching    --no-enable-prefix-caching (same as PREFIX_CACHING=0)
  --wait                 after starting, wait for the container to become healthy
  --allow-second-server  start even though another vLLM container is running (memory!)
  --force-mtp            allow MTP on an image without the PR #50021 label (not recommended)
  -h, --help             this help

Exit codes: 0 started (or dry run ok), 1 refused (a check failed), 2 unknown option, other: docker run failed.
  With --wait also: 3 the container stopped or vanished before it became healthy (last log lines
  are printed), 4 still not healthy after 20 min (the container keeps running; check its logs).

Current settings: IMAGE=${IMAGE} NAME=${NAME} PORT=${PORT} HOST=${HOST} MODEL=${MODEL}
                  MODEL_REVISION=${MODEL_REVISION:-none} PRIVILEGED=${PRIVILEGED}
                  HF cache (host)=${HF_CACHE_HOST}
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --mtp-off) MTP_TOKENS=0 ;;
    --no-prefix-caching) PREFIX_CACHING=0 ;;
    --wait) WAIT=1 ;;
    --allow-second-server) ALLOW_SECOND=1 ;;
    --force-mtp) FORCE_MTP=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
die() { printf 'refusing to start: %s\n' "$*" >&2; exit 1; }

case "$PORT" in ''|*[!0-9]*) die "PORT must be a number (got '${PORT}')." ;; esac

# Host, port and API key have their own variables: in EXTRA_ARGS they would bypass the checks below
# (and an --api-key there would show in `ps`).
read -r -a extra_check <<<"$EXTRA_ARGS"
for a in "${extra_check[@]}"; do
  case "$a" in
    --host|--host=*) die "EXTRA_ARGS must not set --host; use HOST=... instead." ;;
    --port|--port=*) die "EXTRA_ARGS must not set --port; use PORT=... instead." ;;
    --api-key|--api-key=*) die "EXTRA_ARGS must not set --api-key; use API_KEY=... instead (passed as the env var VLLM_API_KEY)." ;;
  esac
done

# 0. Never serve beyond loopback without an API key.
case "$HOST" in
  127.*|::1|localhost) ;;
  *)
    [ -n "$API_KEY" ] || die "HOST=${HOST} is not a loopback address and no API_KEY is set; vLLM would serve
  without authentication. Set API_KEY=... (and firewall the port: /metrics stays unauthenticated), or use
  HOST=127.0.0.1."
    warn "HOST=${HOST}: the API key guards /v1/* only; /metrics and /health stay unauthenticated. Firewall port ${PORT}." ;;
esac

# --- checks (all read-only) ----------------------------------------------------------------------
arch="$(uname -m)"
if [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; then
  warn "this cookbook is tested only on DGX Spark (aarch64); this machine is ${arch}."
fi
command -v docker >/dev/null 2>&1 || die "docker not found."
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon."

# 1. The container name must be free (any state). We never replace an existing container.
if docker container inspect "$NAME" >/dev/null 2>&1; then
  state="$(docker container inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || echo unknown)"
  die "a container named '${NAME}' already exists (state: ${state}). This script never replaces it.
  If it is this cookbook's server: 'docker start ${NAME}' starts it again with its old settings.
  Otherwise choose another name: NAME=vllm-agent-2 $0"
fi

# 2. The port must be free.
port_in_use() {
  if command -v ss >/dev/null 2>&1; then
    [ -n "$(ss -Hltn "sport = :${PORT}" 2>/dev/null)" ]
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1
  elif command -v netstat >/dev/null 2>&1; then
    netstat -ltn 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${PORT}\$"
  else
    return 2
  fi
}
rc=0; port_in_use || rc=$?
if [ "$rc" -eq 0 ]; then
  die "port ${PORT} is already in use. If that is a vLLM server, adopt it (README 'Adopt an existing
  server') or pick another port: PORT=8001 $0"
elif [ "$rc" -eq 2 ]; then
  die "cannot check whether port ${PORT} is free (need ss, lsof or netstat)."
fi

# 3. Another vLLM container running? Two 27B servers rarely fit next to each other on one Spark.
others="$(docker ps --no-trunc --format '{{.Names}}|{{.Image}}|{{.Command}}' | grep -i 'vllm' | cut -d'|' -f1 || true)"
if [ -n "$others" ]; then
  if [ "$ALLOW_SECOND" -eq 0 ]; then
    die "another vLLM container is running: $(echo "$others" | paste -sd, -)
  This script does not touch it. Adopt it instead, or re-run with --allow-second-server if memory allows."
  fi
  warn "another vLLM container is running ($(echo "$others" | paste -sd, -)); continuing (--allow-second-server)."
fi

# 4. The image must be local. We never pull.
docker image inspect "$IMAGE" >/dev/null 2>&1 || die "image '${IMAGE}' is not available locally.
  Build it with ./build.sh, or pull a fallback image yourself (README 'Fallback')."

# 5. MTP only on an image that carries PR #50021.
if [ "$MTP_TOKENS" != "0" ]; then
  pr_label="$(docker image inspect -f '{{ index .Config.Labels "org.spark-vllm-agent-cookbook.vllm.pr50021" }}' "$IMAGE" 2>/dev/null || true)"
  if [ -z "$pr_label" ] || [ "$pr_label" = "<no value>" ]; then
    if [ "$FORCE_MTP" -eq 0 ]; then
      die "image '${IMAGE}' has no PR #50021 label. MTP without that patch can crash or wedge the engine
  with Qwen3.8 (see docs/field-notes.md). Use --mtp-off, or --force-mtp if you know the image has the fix."
    fi
    warn "MTP on an image without the PR #50021 label (--force-mtp)."
  elif [ "$pr_label" != "$PR50021_HEAD" ]; then
    warn "image '${IMAGE}' carries PR #50021 at '${pr_label}', not the pinned ${PR50021_HEAD}; this MTP setup is untested with it."
  fi
fi

# 6. Weights: present in the host cache? (Only checked for an HF repo id.)
case "$MODEL" in
  /*) say "MODEL is a container path (${MODEL}); make sure it is inside the mounted cache." ;;
  */*)
    repo_dir="${HF_CACHE_HOST}/hub/models--${MODEL//\//--}"
    if ! compgen -G "${repo_dir}/snapshots/${MODEL_REVISION:-*}/config.json" >/dev/null; then
      if [ "$ALLOW_DOWNLOAD" = "1" ]; then
        warn "weights${MODEL_REVISION:+ at revision ${MODEL_REVISION}} not found in ${repo_dir}; vLLM will download them on start (about 21 GB)."
      else
        die "weights for ${MODEL}${MODEL_REVISION:+ at revision ${MODEL_REVISION}} not found under ${HF_CACHE_HOST}/hub. Download them first (about 21 GB):
  HF_HOME=\"${HF_CACHE_HOST}\" hf download ${MODEL}${MODEL_REVISION:+ --revision ${MODEL_REVISION}}
  or re-run with ALLOW_DOWNLOAD=1 to let vLLM download them."
      fi
    fi ;;
esac

# 7. Memory: vLLM checks at startup that GPU_MEMORY_UTILIZATION x total memory is free.
if command -v free >/dev/null 2>&1; then
  avail_gb="$(free -g | awk '/^Mem:/ {print $7}')"
  if [ -n "${avail_gb:-}" ] && [ "$avail_gb" -lt 70 ]; then
    warn "only ${avail_gb} GB memory available; the startup check needs about 66 GB free on a Spark (0.55 x ~121 GB)."
  fi
fi

# --- command ------------------------------------------------------------------------------------
# The eugr base (and the image built from it) needs `vllm serve` as the command; the official
# vllm/vllm-openai images already have `vllm serve` as their entrypoint.
entrypoint="$(docker image inspect -f '{{json .Config.Entrypoint}}' "$IMAGE" 2>/dev/null || echo null)"
if printf '%s' "$entrypoint" | grep -q '"vllm"'; then
  serve_cmd=()
else
  serve_cmd=(vllm serve)
fi

docker_args=(run -d --name "$NAME" --pull never --gpus all --network host --ipc host
  --restart "$RESTART_POLICY"
  -v "${HF_CACHE_HOST}:${HF_CACHE_CONTAINER}"
  -e "HF_HOME=${HF_CACHE_CONTAINER}"
  --health-cmd "python3 -c \"import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:${PORT}/health').status==200 else 1)\" || exit 1"
  --health-interval 30s --health-timeout 5s --health-start-period "$HEALTH_START_PERIOD" --health-retries 20
  --label org.spark-vllm-agent-cookbook.cookbook=vllm-dgx-spark)
[ "$PRIVILEGED" = "1" ] && docker_args+=(--privileged)
[ "$ALLOW_DOWNLOAD" = "1" ] || docker_args+=(-e HF_HUB_OFFLINE=1)
# Name only: docker copies the value from its own environment, so the key is not on any command line.
if [ -n "$API_KEY" ]; then
  export VLLM_API_KEY="$API_KEY"
  docker_args+=(-e VLLM_API_KEY)
fi

vllm_args=("$MODEL"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$HOST" --port "$PORT"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --kv-cache-memory-bytes "$KV_CACHE_MEMORY_BYTES"
  --max-model-len "$MAX_MODEL_LEN"
  --kv-cache-dtype "$KV_CACHE_DTYPE"
  --reasoning-parser "$REASONING_PARSER"
  --enable-auto-tool-choice --tool-call-parser "$TOOL_CALL_PARSER"
  --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
if [ -n "$MODEL_REVISION" ] && [ "${MODEL#/}" = "$MODEL" ]; then   # repo id, not a container path
  vllm_args+=(--revision "$MODEL_REVISION" --tokenizer-revision "$MODEL_REVISION")
fi
if [ "$PREFIX_CACHING" = "1" ]; then
  vllm_args+=(--enable-prefix-caching --mamba-cache-mode "$MAMBA_CACHE_MODE")
else
  vllm_args+=(--no-enable-prefix-caching)
fi
[ -n "$LOAD_FORMAT" ] && vllm_args+=(--load-format "$LOAD_FORMAT")
if [ "$MTP_TOKENS" != "0" ]; then
  vllm_args+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}")
fi
[ "$LANGUAGE_MODEL_ONLY" = "1" ] && vllm_args+=(--language-model-only)
if [ -n "$EXTRA_ARGS" ]; then
  read -r -a extra <<<"$EXTRA_ARGS"
  vllm_args+=("${extra[@]}")
fi

say "checks passed: name '${NAME}' and port ${PORT} are free; image ${IMAGE}; MTP ${MTP_TOKENS}; prefix caching ${PREFIX_CACHING}"
printf '+ docker'
for a in "${docker_args[@]}" "$IMAGE" "${serve_cmd[@]}" "${vllm_args[@]}"; do printf ' %q' "$a"; done
printf '\n'

if [ "$DRY_RUN" -eq 1 ]; then
  say "dry run: nothing started."
  exit 0
fi

docker "${docker_args[@]}" "$IMAGE" "${serve_cmd[@]}" "${vllm_args[@]}"
say "started '${NAME}'. First start compiles and autotunes for about 9-10 min."
say "logs: docker logs -f ${NAME}    health: docker inspect -f '{{.State.Health.Status}}' ${NAME}"

if [ "$WAIT" -eq 1 ]; then
  say "waiting for '${NAME}' to become healthy (up to 20 min)..."
  for _ in $(seq 1 120); do
    status="$(docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "$NAME" 2>/dev/null || echo gone)"
    case "$status" in
      "running healthy") say "healthy. Verify with: python3 verify.py --base-url http://localhost:${PORT}/v1 --model ${SERVED_MODEL_NAME}"; exit 0 ;;
      running*) sleep 10 ;;
      *) echo "container is '${status}'. Last log lines:" >&2; docker logs --tail 60 "$NAME" >&2 || true; exit 3 ;;
    esac
  done
  echo "not healthy after 20 min; check: docker logs --tail 100 ${NAME}" >&2
  exit 4
fi
