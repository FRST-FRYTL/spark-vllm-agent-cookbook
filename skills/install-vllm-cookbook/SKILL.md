---
name: install-vllm-cookbook
description: "Set up a local agent model server on an NVIDIA DGX Spark from the spark-vllm-agent-cookbook (Qwen3.8-27B-NVFP4 on a patched vLLM, served as `primary`, tested for agent tool calling), or adopt a vLLM server that already runs, then verify it with `python3 verify.py`. Optionally points an existing PAN or Hermes Agent install at it, after asking. Use when the user wants a local model server for an agent on a DGX Spark, asks to install this vLLM cookbook, or wants to check whether a local OpenAI-compatible endpoint is fit for agent tool calling."
license: MIT
platforms: [linux]
metadata:
  author: FRST-FRYTL
  homepage: https://github.com/FRST-FRYTL/spark-vllm-agent-cookbook
  tags: [vllm, dgx-spark, model-server, tool-calling, agents, install, setup, experimental]
---

# Install the vLLM agent cookbook (DGX Spark, experimental)

This skill drives the scripts at the root of the spark-vllm-agent-cookbook repo (`build.sh`,
`serve.sh`, `verify.py`) and decides *what* to run; the scripts, `README.md` and
`docs/field-notes.md` do the work and hold the details. All paths below are relative to that repo's
root. It works without any particular agent: the result is an OpenAI-compatible endpoint that any
client can use. The cookbook is **experimental**, tested only on NVIDIA DGX Spark (GB10, aarch64,
about 121 GB usable unified memory), and **its full fresh-install path has not yet been tested end to end.** Say
so to the user at the start.

## Hard rules
These rules bind **you, the agent**: they limit which commands you run. Some steps below are for the
**user to run themselves** (marked "user runs"): you may show those commands and explain them, but
never run them for the user.
- **Never stop, restart, remove, replace or reconfigure an existing model server or container**, and
  never touch any container this skill did not create. That includes `docker stop`, `rm`,
  `restart`, `update` and `rename`. If the user wants the old server gone, they do it themselves (user
  runs); point them to the rollback advice in `docs/field-notes.md` (keep the old container stopped,
  not deleted).
- **Ask before each of these, one at a time, with the size and time:** cloning the repo, pulling or
  building an image (base pull about 11 GB, about 25 GB on disk), downloading the weights (about
  21 GB), starting a container, and writing any agent config (PAN, Hermes or other). A "yes" to one
  is not a "yes" to the next.
- **Dry-run first.** Run `./build.sh --dry-run` and `./serve.sh --dry-run` and show their output
  before the real run.
- **Never redistribute.** No `docker push`, no `docker save` for others, no uploading weights. Users
  build locally and download the model themselves from Hugging Face.
- **Read-only probing only** for servers you did not start: `ss`, `docker ps`, `docker inspect`,
  `curl …/v1/models`, `curl …/health`. `verify.py` sends a handful of synthetic test requests (about
  a dozen, well under a minute on a warm server), so ask before running it against someone else's
  server.
- **Never run `pan setup`, `hermes setup` or any installer of an agent from here**, and never create
  an agent config that does not exist yet. Configuring an agent is optional (step 8) and only edits
  an existing config, after asking.
- If anything here disagrees with `README.md`, `./build.sh --help`, `./serve.sh --help` or
  `python3 verify.py --help` of the checked-out version, the scripts and README win. Say so and follow
  them.

## Dry-run mode
If the user asks for a dry run, a walk-through or "just show me": **run no commands at all**, not even
the read-only checks (unless the user allows those). Go through steps 1–9 in order and print every
command you would run, each with the question you would ask before it and what you would do with the
answer. End with the full list of commands. Change nothing.

## 1. Get the cookbook
- Look for an existing checkout first: a directory with `serve.sh`, `build.sh` and `verify.py` side
  by side (the current directory, or ask the user where they keep it). Otherwise ask, then:
  ```bash
  git clone https://github.com/FRST-FRYTL/spark-vllm-agent-cookbook
  cd spark-vllm-agent-cookbook
  ```
  If the user wants a fixed release, add `--branch <tag>` with a tag from the repo's releases.
- Read `README.md` and `docs/field-notes.md` before going on; tell the user about the experimental
  status and the field notes that matter to them (repeat TTFT, tool-call corruption on other builds,
  engine stalls).
- Read the defaults: `./serve.sh --help` prints the current settings (`IMAGE`, `NAME`, `PORT`,
  `HOST`, `MODEL`, `MODEL_REVISION`) and `./build.sh --help` the image tag. Use those values
  wherever this skill says `$IMAGE`, `$NAME`, `$MODEL`, `$MODEL_REVISION` or `$PORT`.

## 2. Check prerequisites (read-only)
```bash
uname -m                                         # aarch64
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
nvidia-smi | head -5                             # "CUDA Version: 13.0" or higher
docker version --format '{{.Server.Version}}'    # daemon reachable as this user
docker info --format '{{json .Runtimes}}'        # an "nvidia" runtime, or:
nvidia-ctk --version                             # NVIDIA Container Toolkit
df -h "$(docker info -f '{{.DockerRootDir}}')" "${HF_HOME:-$HOME/.cache/huggingface}"
free -g
python3 --version                                # Python 3.10+, for verify.py
```
- Need: aarch64 with an `NVIDIA GB10`, docker usable by the user, the container toolkit, about **45 GB
  free** disk (image + weights; 60 GB is comfortable), and about **45–49 GB** of memory for vLLM. At
  startup vLLM also wants about 66 GB free (0.55 of total), so other large GPU jobs must be stopped
  first. That is the user's call; don't stop them yourself.
- Not a Spark (other arch or GPU): say the cookbook is untested there, show the README's notes for
  other GPUs, and continue only if the user wants to adapt it themselves.
- Missing docker access, toolkit or driver: stop and explain what to install; don't install system
  packages yourself unless the user asks.

## 3. Detect an existing server (read-only)
Target port: `PORT` (default from `./serve.sh --help`, usually 8000; ask if the user wants another).
```bash
ss -Hltnp "sport = :$PORT"
docker ps -a --format '{{.Names}}\t{{.Image}}\t{{.Status}}' | grep -i vllm
curl -s --max-time 5 "http://localhost:$PORT/v1/models"
docker ps -a --filter label=org.spark-vllm-agent-cookbook.cookbook=vllm-dgx-spark --format '{{.Names}}\t{{.Status}}'
```
(If `serve.sh` sets a different `--label` on its container, filter on that one.)
If something already listens on the port or a vLLM container runs, **don't touch it**. Offer:
- **(a) Adopt the existing server:** keep it as it is, verify it (step 7, ask first) with the model id
  from `/v1/models`, and optionally point an agent at it (step 8). Skip steps 4–6.
- **(b) Use a different port:** continue with `PORT=<free port>`. Warn that two 27B servers rarely fit
  in one Spark's memory, so `serve.sh` needs `--allow-second-server` and the start may fail.

If the running container carries this cookbook's label, it is this cookbook's server from an earlier
run: treat it as "already done" (step 4), not as a conflict.

## 4. Work out what is already done (idempotent)
Check each item and show a short status table before planning:
| Item | Check | Done when |
|---|---|---|
| image | `docker image inspect -f '{{ index .Config.Labels "org.spark-vllm-agent-cookbook.vllm.pr50021" }}' "$IMAGE"` (the label key `build.sh` checks) | prints the PR #50021 head commit that `build.sh` pins |
| weights | `ls "${HF_HOME:-$HOME/.cache/huggingface}/hub/models--${MODEL//\//--}/snapshots/$MODEL_REVISION/config.json"` | file exists |
| container | `docker ps -a --filter name="^$NAME\$" --format '{{.Status}}'` | `Up … (healthy)` |
| verified | `python3 verify.py --base-url "http://localhost:$PORT/v1"` (step 7) | exit code 0 |

- A stopped `$NAME` container from an earlier run: offer `docker start "$NAME"` (ask; it keeps its
  old settings). Don't recreate it; `serve.sh` refuses to while the name exists. If the user wants new
  settings, they remove it themselves first.
- Re-running this skill after a finished install should end in step 7 with nothing to change.

## 5. Show the plan
List only the steps that are not done yet, each with its cost, for example:
1. build the image: base pull about 11 GB (can take 20+ min on a slow link), build seconds;
2. download the weights: about 21 GB;
3. start `$NAME` on port `$PORT`: first start about 9–10 min;
4. `python3 verify.py`, about a minute;
5. optional: point an existing agent install at the server.

Get a yes for the plan, then still ask before each step.

## 6. Build, download, serve (ask before each)
```bash
./build.sh --dry-run          # show the output
./build.sh                    # ask first: pulls the pinned base image, applies vLLM PR #50021
```
- `build.sh` verifies the PR diff's sha256 and fails otherwise; if the base digest is gone from
  Docker Hub ("manifest unknown"), stop and offer the README's fallback (stock vLLM, `--mtp-off`).

Weights (ask first; mention the model licence: Apache 2.0 per NVIDIA's model card, which the user
should read):
```bash
hf download "$MODEL" --revision "$MODEL_REVISION"
# without the CLI installed:
uvx --from huggingface_hub hf download "$MODEL" --revision "$MODEL_REVISION"
```
The revision is pinned to the commit the tested server runs, so the weights can't change under the
tested setup; `serve.sh` passes the same one to vLLM (`MODEL_REVISION`).
Run it as the user (not in a root container), so the cache stays theirs. If it stalls, re-run it; it
resumes.

Serve (ask first):
```bash
PORT=$PORT ./serve.sh --dry-run    # show the checks and the docker run command
PORT=$PORT ./serve.sh --wait       # starts only the new container "$NAME"
```
- `serve.sh` binds `127.0.0.1` by default. If the user needs other machines to reach it, suggest
  `HOST=0.0.0.0` plus `API_KEY=…` (it refuses a non-loopback `HOST` without a key); never echo the key
  back. `/metrics` stays unauthenticated, so tell them to firewall the port.
- If `serve.sh` refuses (port in use, name exists, other vLLM running, image missing, no PR label for
  MTP, weights missing), report the reason and go back to step 3 or 4. **Don't** work around a refusal
  by stopping something.
- While it starts, `docker logs -f "$NAME"`. A container that exits: show the last log lines, check
  `docs/field-notes.md`, and offer `--mtp-off` as the fallback.

## 7. Verify with `verify.py`
```bash
python3 verify.py --base-url "http://localhost:$PORT/v1" --model primary
```
(Use the adopted server's model id in case (a); without `--model` it takes the first listed id. If
the server has an API key, set `VLLM_API_KEY` in the environment rather than passing `--api-key`, so
the key stays out of the shell history; never echo it.) It is a single stdlib-only file and checks:
`models` (reachable, model listed), `tool_calls` (6 streamed tool-call probes), `tool_corruption`
(one agent-style request sent 5 times, which exercises the prefix cache) and `json_schema`
(structured output for agent side calls such as memory gates or routers; a `json_object` fallback is
a warning). Show the result. `--json` gives machine-readable output.
- Exit code 0: all ok or warnings. 1: a check failed. 2: the endpoint is unreachable (still starting?
  wrong port or `--base-url`?).
- Each failure prints a `fix (<check>):` hint, most of them naming a section of
  `docs/field-notes.md`; read that section and explain it to the user.
- A `tool_corruption` failure: recommend `--no-prefix-caching` or `--mtp-off` (a new container: `serve.sh` refuses while the old one holds the name, the port or runs as
  another vLLM server, so the user stops and removes the old `$NAME` themselves, or keeps it
  stopped for rollback and starts the new one with another `NAME`; running both at once needs
  another `PORT`, another `NAME` and `--allow-second-server`, and rarely fits in memory), then
  re-run `verify.py`.
- A `json_schema` warning: the server rejects `json_schema` but `json_object` works. Note it for
  step 8 (clients with structured side calls should use `json_object`).
- Re-run `verify.py` after any image or flag change.

## 8. Optional: point an existing agent at the server
The server is done after step 7. This step is **optional**; skip it when the user only wanted the
server, and for users without PAN or Hermes. Detect (read-only):
```bash
command -v pan hermes
ls "${HERMES_HOME:-$HOME/.hermes}/config.yaml" "${HERMES_HOME:-$HOME/.hermes}/pan/config.yaml" 2>/dev/null
```
**Neither found:** show the generic settings any OpenAI-compatible client needs and stop there:
base URL `http://localhost:$PORT/v1`, model `primary` (or the adopted id), API key `EMPTY` (or the
server's key), and whether `json_schema` or only `json_object` works (from step 7).

**Found:** offer to point it at the endpoint, and **ask first**. Ask which profile (`HERMES_HOME`,
default `~/.hermes`), show the exact change as a diff before writing it, and edit only the keys named
here; keep the rest of the file.
- **Hermes Agent, main model** (`$HERMES_HOME/config.yaml`): Hermes' custom OpenAI-compatible
  provider (`vllm` is an alias of `custom`):
  ```bash
  hermes config set model.provider custom
  hermes config set model.base_url "http://localhost:$PORT/v1"
  hermes config set model.default primary
  ```
  `model.api_key` only if the server has an API key.
- **PAN, memory gate** (only when `$HERMES_HOME/pan/config.yaml` already exists):
  ```yaml
  models:
    gate:
      base_url: http://localhost:8000/v1   # with the user's PORT
      model: primary                        # or the adopted server's model id
      api_key: EMPTY                        # or their API key; never echo it
      # structured: json_object             # only if verify.py warned on json_schema
  ```
  Edit only `models.gate`. `pan` on PATH but no `pan/config.yaml` in that profile: PAN isn't set up
  there; say so and leave it (setting PAN up is not this skill's job).
- The main model and the gate can share this one server.

## 9. Hand-over
Tell the user:
- the container name (`$NAME`), port, image tag, and that it restarts with the machine
  (`--restart unless-stopped`); stopping it is `docker stop "$NAME"` (user runs, their call);
- the field notes that matter (`docs/field-notes.md`): repeat TTFT about 1.1 s with MTP + prefix
  caching (vLLM #53670); rare engine wedges keep `/health` green, so a stall watchdog is advised
  (sketch in the field notes); the pinned nightly base may disappear from Docker Hub, so keep the
  built image;
- rollback when replacing an older server (user runs; you never do): keep the old container stopped
  with `docker update --restart=no`, don't delete it;
- re-run `python3 verify.py` after any image or flag change, and whenever an agent's tool calls start
  to look wrong.
