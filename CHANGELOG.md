# Changelog

All notable changes to spark-vllm-agent-cookbook. Versions follow
[Semantic Versioning](https://semver.org/); git tags are `v<version>`. Each release is one squashed
commit exported from the development workspace.

## [0.1.1] - 2026-10-01

Docs only. The README links [PAN](https://github.com/FRST-FRYTL/pan-agent), the Hermes memory add-on
this server is the tested stack of, now that it is published. Image, flags, pins and `verify.py` are
unchanged from 0.1.0.

## [0.1.0] - 2026-09-29

First public release, a research preview. Tag `v0.1.0`. Experimental: the fresh install is not yet
tested end to end, and the scripts, flags and checks may change.

**Tested stack:** NVIDIA DGX Spark (GB10, aarch64); base image `eugr/spark-vllm` `nightly-20260927`
pinned by digest (vLLM `0.30.1rc1.dev220`); vLLM PR #50021 at head `71d7c782`, applied at build time;
`nvidia/Qwen3.8-27B-NVFP4` at revision `482ca0f3`; MTP 3 + prefix caching, served as `primary`.

### What's in it
- `build.sh` and `Dockerfile`: build the patched image locally. The PR diff is pinned by commit and
  sha256; the build checks that the patch is in place; there is no push step.
- `serve.sh`: starts one container with the tested flags. It defaults to loopback, refuses a
  non-loopback host without an API key, refuses to replace or collide with existing containers and
  ports, never pulls images, and refuses MTP on an image without the PR label. `--mtp-off` and
  `--no-prefix-caching` fallbacks.
- `verify.py`: endpoint checks for agent use (`models`, `tool_calls`, `tool_corruption`,
  `json_schema`), with `--json` output and exit codes for scripts. Unit tests in `tests/`.
- `docs/field-notes.md`: tool-call corruption with prefix caching + MTP, the GDN engine crash and
  PR #50021, repeat TTFT (#53670), structured output, thinking mode, memory flags on unified memory,
  a stall watchdog, rollback.
- The `install-vllm-cookbook` Claude Code skill and a plugin marketplace manifest
  (`vllm-dgx-spark@spark-vllm-agent-cookbook`).
- CI: tests on Python 3.11 and 3.13, `bash -n` and shellcheck, gitleaks, DCO check.

### Known limitations
- Tested on one DGX Spark only. The 24 h soak of the patched image under real load passed (0 restarts,
  0 engine faults, 37 benchmark runs); the fresh-install path is not yet tested end to end.
- The base image is a nightly tag; if its digest disappears from Docker Hub, this exact build cannot
  be reproduced (see the README fallback).
- PR #50021 is unmerged upstream.
