#!/usr/bin/env bash
# Build the cookbook's vLLM image for NVIDIA DGX Spark locally. EXPERIMENTAL.
#
#   ./build.sh --dry-run      print every command, run nothing
#   ./build.sh                build (pulls the pinned base, about 11 GB compressed, if it is not local)
#
# The image stays local. This script has no push step, and you must not publish the image: the base
# carries third-party software under its own licences. No model weights go into the image.
set -euo pipefail

TAG="${TAG:-spark-vllm-agent:0.30-pr50021}"
BASE_IMAGE="eugr/spark-vllm@sha256:d1e9d16ad8958ddc2e88686416cefdc1ab679047fcdeacb747a6a0d2385ad932"
PR50021_BASE="fba47397f304993741d2b76d73fb42e5be3e8eac"
PR50021_HEAD="71d7c782ca4230b556ced08f78fcac865b64a15d"
PR50021_SHA256="ab9c8597eea7a7c171217886dbcdc98656a9ed85e968d6863f37b17edc967349"
PR50021_URL="https://github.com/vllm-project/vllm/compare/${PR50021_BASE}...${PR50021_HEAD}.diff"
MIN_FREE_GB="${MIN_FREE_GB:-40}"

DRY_RUN=0
REBUILD=0
FORCE_ARCH=0

usage() {
  cat <<EOF
Usage: $0 [--dry-run] [--rebuild] [--tag NAME:TAG] [--force-arch]

  --dry-run     print the commands without running them
  --rebuild     build again even if ${TAG} already exists (an existing tag without the pinned
                PR #50021 label is refused unless you pass this)
  --tag         local image tag (default: ${TAG}; env TAG works too)
  --force-arch  build on a machine that is not aarch64 (unsupported; the base image is arm64 only)
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --rebuild) REBUILD=1 ;;
    --tag) TAG="${2:?--tag needs a value}"; shift ;;
    --force-arch) FORCE_ARCH=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CTX="${HERE}/.build"
DIFF="${CTX}/pr50021.diff"

say() { printf '==> %s\n' "$*"; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }
# Print a command (shell-quoted), then run it unless --dry-run.
run() {
  printf '+'; printf ' %q' "$@"; printf '\n'
  if [ "$DRY_RUN" -eq 0 ]; then "$@"; fi
}

# --- checks (read-only) ------------------------------------------------------------------------
arch="$(uname -m)"
if [ "$arch" != "aarch64" ] && [ "$arch" != "arm64" ]; then
  if [ "$FORCE_ARCH" -eq 0 ]; then
    die "this cookbook targets DGX Spark (aarch64); this machine is ${arch}. See README.md 'Other GPUs'."
  fi
  say "WARNING: building on ${arch} (--force-arch). The base image is arm64 only."
fi

command -v docker >/dev/null 2>&1 || die "docker not found. Install Docker Engine and the NVIDIA Container Toolkit first."
command -v sha256sum >/dev/null 2>&1 || die "sha256sum not found (coreutils)."

if [ "$DRY_RUN" -eq 0 ]; then
  docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon (is it running, and may this user use it?)."
  if docker image inspect "$TAG" >/dev/null 2>&1 && [ "$REBUILD" -eq 0 ]; then
    label="$(docker image inspect -f '{{ index .Config.Labels "org.spark-vllm-agent-cookbook.vllm.pr50021" }}' "$TAG" 2>/dev/null || true)"
    [ "$label" = "<no value>" ] && label=""
    if [ "$label" = "$PR50021_HEAD" ]; then
      say "${TAG} already exists with the pinned PR #50021 label (${label}). Nothing to do; use --rebuild to build again."
      exit 0
    fi
    # The tag is taken by an image that does not carry the pinned patch: never reuse or overwrite it silently.
    die "${TAG} already exists, but its PR #50021 label is '${label:-missing}', not the pinned ${PR50021_HEAD}.
  Rebuild it with --rebuild (replaces the tag), or build under another tag: --tag spark-vllm-agent:<name>"
  fi
  root_dir="$(docker info -f '{{.DockerRootDir}}' 2>/dev/null || echo /var/lib/docker)"
  free_gb="$(df -P -BG "$root_dir" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')"
  if [ -n "${free_gb:-}" ] && [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
    die "only ${free_gb} GB free under ${root_dir}; the image needs about 25 GB (plus the pull). Set MIN_FREE_GB to override."
  fi
  if docker image inspect "$BASE_IMAGE" >/dev/null 2>&1; then
    say "base image is already local: ${BASE_IMAGE}"
  else
    say "base image is not local; the build will pull ${BASE_IMAGE} (about 11 GB compressed, about 25 GB on disk)."
  fi
fi

# --- PR #50021 diff, pinned by commit and sha256 -----------------------------------------------
run mkdir -p "$CTX"
if [ "$DRY_RUN" -eq 0 ] && [ -f "$DIFF" ] && echo "${PR50021_SHA256}  ${DIFF}" | sha256sum -c --status -; then
  say "PR #50021 diff already downloaded and verified: ${DIFF}"
else
  command -v curl >/dev/null 2>&1 || die "curl not found."
  run curl -fsSL --retry 3 -o "${DIFF}.part" "$PR50021_URL"
  if [ "$DRY_RUN" -eq 0 ]; then
    if ! echo "${PR50021_SHA256}  ${DIFF}.part" | sha256sum -c --status -; then
      rm -f "${DIFF}.part"
      die "sha256 mismatch for the PR #50021 diff (expected ${PR50021_SHA256}). Not building."
    fi
  else
    printf '+ echo %q | sha256sum -c -\n' "${PR50021_SHA256}  ${DIFF}.part"
  fi
  run mv "${DIFF}.part" "$DIFF"
fi
run cp "${HERE}/Dockerfile" "${CTX}/Dockerfile"

# --- build (local tag only; there is deliberately no push) --------------------------------------
run docker build \
  --build-arg "PR50021_BASE=${PR50021_BASE}" \
  --build-arg "PR50021_HEAD=${PR50021_HEAD}" \
  --build-arg "PR50021_SHA256=${PR50021_SHA256}" \
  -t "$TAG" "$CTX"

run docker image inspect -f '{{ index .Config.Labels "org.spark-vllm-agent-cookbook.vllm.pr50021" }}' "$TAG"
say "done: ${TAG} (local only). Next: download the weights, then ./serve.sh --dry-run (see README.md)."
