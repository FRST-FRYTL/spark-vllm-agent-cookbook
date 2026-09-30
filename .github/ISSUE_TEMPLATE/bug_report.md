---
name: Bug report
about: Something in the cookbook does not work as documented
labels: bug
---

> spark-vllm-agent-cookbook is a research preview maintained in spare time. Issues are read and
> handled **best-effort**, with no response-time or support guarantee. Security problems: please use
> private vulnerability reporting (see SECURITY.md), not an issue.

**What happened, and what did you expect?**

**Which step failed?** (build / download / serve / verify / using it from a harness)

**Steps to reproduce**

**Environment**
- Cookbook version or commit:
- Hardware (`uname -m`, GPU name from `nvidia-smi`):
- Driver and CUDA version (`nvidia-smi` header):
- Docker version, NVIDIA Container Toolkit version:
- Image: tag and PR label (`docker image inspect -f '{{ index .Config.Labels "org.spark-vllm-agent-cookbook.vllm.pr50021" }}' <image>`):
- Non-default settings (env vars or flags you changed for `serve.sh`):

**Relevant output**
- the `--dry-run` output of the script that failed;
- `python3 verify.py --json`, if the server starts;
- the relevant `docker logs` lines (for example the last 60).

**Redact anything private** first: host names, addresses, paths, API keys, prompt content.
