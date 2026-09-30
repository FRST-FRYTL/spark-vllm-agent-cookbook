# syntax=docker/dockerfile:1
#
# Reference vLLM image for agent tool-calling on NVIDIA DGX Spark (GB10, aarch64, sm_121). EXPERIMENTAL.
#
#   base:  eugr/spark-vllm nightly-20260927 (vLLM 0.30.1rc1.dev220, CUDA 13.0.2, torch 2.13, arch 12.1a)
#   patch: vllm-project/vllm PR #50021 "Bound accepted-token state lookups in GDN/KDA spec decode"
#          at PR head 71d7c782ca4230b556ced08f78fcac865b64a15d (runtime files under vllm/ only)
#
# Build it with ./build.sh, which downloads the PR diff, checks its sha256 and passes it in the build
# context. This file contains no weights and nothing is pushed anywhere. Do not publish the result:
# the base image carries third-party software (CUDA and others) under its own licences.
#
# WARNING: the base is a *nightly* tag on Docker Hub, pinned here by digest. Nightly tags and their
# digests can be deleted from Docker Hub at any time. If the pull fails with "manifest unknown", this
# exact build can no longer be reproduced. Use the fallback in README.md (stock vLLM, MTP off, no
# patch), or move the pin to a newer nightly and re-run `python3 verify.py` before relying on it.
FROM eugr/spark-vllm@sha256:d1e9d16ad8958ddc2e88686416cefdc1ab679047fcdeacb747a6a0d2385ad932

# PR #50021, pinned. The diff is the GitHub compare of the PR's fork point and its head commit, so it
# does not move when the PR is pushed again. build.sh fetches it from:
#   https://github.com/vllm-project/vllm/compare/<PR50021_BASE>...<PR50021_HEAD>.diff
ARG PR50021_BASE=fba47397f304993741d2b76d73fb42e5be3e8eac
ARG PR50021_HEAD=71d7c782ca4230b556ced08f78fcac865b64a15d
ARG PR50021_SHA256=ab9c8597eea7a7c171217886dbcdc98656a9ed85e968d6863f37b17edc967349

COPY pr50021.diff /opt/pr50021.diff

# Apply only the Python/Triton runtime files (vllm/**) to the installed vLLM tree; the PR's test files
# are skipped. Triton kernels are JIT-compiled, so no CUDA compile is needed and the build takes seconds.
# The final grep checks that the bounded state-index load is in place.
RUN set -eu; \
    echo "${PR50021_SHA256}  /opt/pr50021.diff" | sha256sum -c -; \
    site="$(python3 -c "import importlib.util, os; print(os.path.dirname(os.path.dirname(importlib.util.find_spec('vllm').origin)))")"; \
    echo "installed vLLM tree: ${site}"; \
    cd "${site}"; \
    git apply --check --include='vllm/**' /opt/pr50021.diff; \
    git apply --include='vllm/**' /opt/pr50021.diff; \
    grep -q "idx_in_row = (i_t >= 0) & (i_t < stride_indices_seq)" \
        vllm/third_party/flash_linear_attention/ops/fused_recurrent.py; \
    echo "vllm-project/vllm PR #50021 at ${PR50021_HEAD} applied" > /opt/pr50021.applied

# serve.sh reads this label: MTP is only enabled by default on an image that carries the patch.
LABEL org.spark-vllm-agent-cookbook.cookbook="vllm-dgx-spark" \
      org.spark-vllm-agent-cookbook.vllm.base="eugr/spark-vllm@sha256:d1e9d16ad8958ddc2e88686416cefdc1ab679047fcdeacb747a6a0d2385ad932" \
      org.spark-vllm-agent-cookbook.vllm.pr50021="71d7c782ca4230b556ced08f78fcac865b64a15d"

# The base image's entrypoint and working directory are kept; serve.sh passes `vllm serve ...`.
