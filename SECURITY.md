# Security policy

## Versions

spark-vllm-agent-cookbook is a research preview. Fixes, if any, go into the latest release only.

## Reporting a vulnerability

Please **do not open a public issue** for security problems. Report them privately through GitHub's
[private vulnerability reporting](https://github.com/FRST-FRYTL/spark-vllm-agent-cookbook/security/advisories/new)
(the Security tab → "Report a vulnerability").

Include:
- the affected version or commit;
- a description of the problem and its impact;
- the steps to reproduce it.

You should get an acknowledgement within 7 days. This is a one-person project, so a fix may take
longer. You will be credited in the advisory unless you prefer otherwise.

## Scope

In scope are this repository's own files:

- `serve.sh` exposing the server more widely than documented: for example binding beyond loopback
  without an API key, or leaking the API key onto a command line or into logs;
- `build.sh` or the `Dockerfile` accepting a patch or base image other than the pinned ones (a
  bypass of the digest or sha256 checks);
- `serve.sh` or `build.sh` changing, removing or replacing containers, images or files they don't
  own;
- `verify.py` sending the API key anywhere but the base URL it was given;
- documentation whose advice leads to an insecure setup.

Out of scope: vulnerabilities in vLLM, the `eugr/spark-vllm` base image, CUDA, Docker or the model
itself. Please report those to the respective upstream project.

## What to know when you run it

- vLLM has **no authentication** unless an API key is set. `serve.sh` binds to `127.0.0.1` by
  default and refuses a non-loopback `HOST` without `API_KEY`.
- The API key guards `/v1/*` only. **`/metrics` and `/health` stay unauthenticated**, so firewall
  the port when the server listens beyond loopback.
- The key is passed to the container as an environment variable, so anyone who can use Docker on
  the machine can read it with `docker inspect`.
- `PRIVILEGED=1` runs the container with `--privileged`; use it only if the GPU is not visible
  without it.
- Everything you send to the server (prompts, tool results) stays on your machine, but vLLM can log
  request content, depending on its logging flags. Treat the container logs accordingly.
