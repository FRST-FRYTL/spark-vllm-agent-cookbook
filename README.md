# spark-vllm-agent-cookbook

A cookbook for running **Qwen3.8-27B-NVFP4** on an **NVIDIA DGX Spark** with a patched **vLLM**,
tuned and tested for **agent tool calling**. It builds one local image, starts one server with the
tested flags, and checks it with `verify.py`: model id, streamed tool calls, tool-call corruption on
prefix-cache hits, and `json_schema` structured output.

It is for anyone who runs agents against a local model on a Spark: [Hermes Agent](https://github.com/NousResearch/hermes-agent)
or any other harness that talks to an OpenAI-compatible endpoint. The server listens at
`http://localhost:8000/v1` and serves the model as `primary`.

> **Status:** ongoing personal project, research preview, release `v0.1.1`. Experimental. The fresh
> install is not yet tested end to end: the image, flags and checks are the ones that run the tested
> server, but nobody has yet gone from a clean machine to a working server with exactly these
> scripts. Issues are handled best-effort, with no support guarantees. Please report what breaks.

**This repo never redistributes images or weights.** You build the image locally from a public base image
plus a public vLLM patch that is downloaded at build time, and you download the model yourself from
Hugging Face. Don't push the built image to a public registry: the base contains third-party
software (CUDA and others) under its own licences.

## The tested stack

| | |
|---|---|
| Hardware | NVIDIA DGX Spark: GB10, aarch64, about 121 GB usable unified memory, driver 580 (CUDA 13.0) |
| Base image | [`eugr/spark-vllm`](https://hub.docker.com/r/eugr/spark-vllm) `nightly-20260927`, pinned by digest `sha256:d1e9d16ad8958ddc2e88686416cefdc1ab679047fcdeacb747a6a0d2385ad932`; built by [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) for sm_121 (CUDA 13.0.2, torch 2.13) |
| vLLM | `0.30.1rc1.dev220+g24c9772d1` (from the base image) |
| Patch | [vLLM PR #50021](https://github.com/vllm-project/vllm/pull/50021), pinned at head commit `71d7c782ca4230b556ced08f78fcac865b64a15d`, diff checked by sha256, runtime files only. It bounds a state lookup that makes MTP speculative decoding crash or wedge on Qwen3.8's GDN layers |
| Model | [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4) at revision `482ca0f3832238542f8f5295dde86b5f22711d80`, NVIDIA's NVFP4/FP8 quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B) |
| Key flags | MTP 3 (`--speculative-config`), prefix caching with `--mamba-cache-mode align`, 128k context, fp8 KV cache pinned at 20 GiB, `qwen3_coder` tool parser, `qwen3` reasoning parser, `--language-model-only` |
| Served as | `primary` on `127.0.0.1:8000` |

## Tested numbers

One machine, the author's measurements, small samples (3–8 requests per inference row). They hold
only for this stack; treat them as orientation, not as a benchmark.

| Measurement | Result | Setup |
|---|---|---|
| Single-stream decode | about 25 tok/s (p50 25.3 with thinking on, 26.4 off) | short chat, 256 output tokens |
| Agent-shaped request, 1 at a time | TTFT 1.09 s, decode about 21 tok/s | ~7.5k-token prompt, 512 output tokens |
| 2 concurrent agent requests | 44.3 tok/s aggregate | same prompts |
| 4 concurrent agent requests | 70.2 tok/s aggregate, TTFT p50 3.7 s | same prompts, 8 requests |
| Repeat TTFT on a ~7.5k-token prefix | p50 1.09 s (cold 2.97 s) | see [#53670](docs/field-notes.md#repeat-ttft-with-mtp--prefix-caching-53670) |
| MTP | mean acceptance length 3.02 (per position 0.81 / 0.68 / 0.53) | 3 speculative tokens |
| Tool-call corruption, MTP + prefix caching on | 0 corrupted in 914 responses | Hermes-shaped prompts, 20 tool schemas, 4–6 concurrent |
| 24 h soak under real load | **passed**: 0 restarts, 0 engine faults, 0 Xid, 37 benchmark runs, max stall 30 s (watchdog threshold 180 s) | agent benchmark load |
| KV cache capacity | 520,234 tokens (about 3.97 sequences at 128k) | 20 GiB pinned |
| First start | about 9.5 min until healthy (mostly GEMM autotuning) | |

## Quickstart

### 1. Prerequisites

- **Hardware:** NVIDIA DGX Spark (GB10, aarch64). Check with `uname -m` (→ `aarch64`) and
  `nvidia-smi` (→ `NVIDIA GB10`).
- **Driver:** the DGX OS driver with CUDA 13.0 support (`nvidia-smi` shows `CUDA Version: 13.0` or
  higher). The tested machine ran driver 580.
- **Docker** Engine, usable by your user (`docker info`), and the **NVIDIA Container Toolkit**
  (`nvidia-ctk --version`; `docker info` lists an `nvidia` runtime or CDI devices). DGX OS ships both.
- **Disk:** about **45 GB**: the image is about 25 GB on disk (an 11 GB compressed pull), the weights
  about 21 GB. 60 GB free is comfortable.
- **Memory:** vLLM uses about **45–49 GB** of the unified memory (weights about 21 GB, KV cache
  pinned at 20 GiB, plus overhead). At startup it checks that 55 % of total memory (about 66 GB) is
  free, so stop other large GPU jobs first. See [memory flags](docs/field-notes.md#memory-flags-on-unified-memory).
- **Tools:** `bash`, `curl`, `sha256sum`, `ss` (iproute2), Python 3.10+ for `verify.py`, and
  the Hugging Face CLI `hf` for the download (`uvx --from huggingface_hub hf …` works without
  installing it).

```bash
git clone --branch v0.1.1 https://github.com/FRST-FRYTL/spark-vllm-agent-cookbook
cd spark-vllm-agent-cookbook
```

### 2. Build the image

```bash
./build.sh --dry-run      # prints every command, runs nothing
./build.sh                # tags spark-vllm-agent:0.30-pr50021 locally
```

`build.sh` checks the architecture, Docker and free disk space. It downloads the PR #50021 diff
pinned by commit and refuses it unless its sha256 matches. The `Dockerfile` applies the diff with
`git apply --include='vllm/**'` to the installed vLLM tree (Python and Triton files only; no CUDA
compile, so the build takes seconds after the base pull) and checks that the bounded state-index load
is in place. There is no push step. Running it again is a no-op once the tag exists with the pinned
PR label; an existing tag with another or no label is refused (`--rebuild` replaces it).

### 3. Download the weights (pinned revision)

About 21 GB, into your normal Hugging Face cache (`HF_HOME`, default `~/.cache/huggingface`):

```bash
hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3832238542f8f5295dde86b5f22711d80
# or, without installing the CLI:
uvx --from huggingface_hub hf download nvidia/Qwen3.8-27B-NVFP4 --revision 482ca0f3832238542f8f5295dde86b5f22711d80
```

The revision is pinned to the commit the tested server runs, so a later push to the model repo
can't silently change the weights behind the numbers above. `serve.sh` passes the same revision to
vLLM (`MODEL_REVISION`). Download as your own user, so the cache doesn't end up owned by root.
`serve.sh` mounts this cache and starts vLLM with `HF_HUB_OFFLINE=1`, so it never downloads anything
unless you set `ALLOW_DOWNLOAD=1`.

### 4. Serve

```bash
./serve.sh --dry-run      # runs the checks, prints the docker run command, starts nothing
./serve.sh --wait         # starts container "vllm-agent" and waits until it is healthy
```

`serve.sh` **refuses to start** when the port is in use, when a container with the same name exists
(running or stopped), when another vLLM container is running (`--allow-second-server` overrides
that one), or when the weights are missing. It only creates its own container, never stops, removes
or changes any other, and never pulls an image. The first start takes about 9–10 minutes (GEMM
autotuning, `torch.compile`, CUDA graphs); follow it with `docker logs -f vllm-agent`.
`EXTRA_ARGS` must not contain `--host`, `--port` or `--api-key`; use `HOST`, `PORT` and `API_KEY`.

Exit codes: `0` started (or the dry run passed), `1` refused, `2` unknown option. With `--wait`
also `3`: the container stopped or vanished before it became healthy (the last log lines are
printed), and `4`: still not healthy after 20 minutes (the container keeps running; check its logs).

### 5. Verify

```bash
python3 verify.py                                   # http://localhost:8000/v1, first model /models lists (primary with serve.sh defaults)
python3 verify.py --base-url http://localhost:8001/v1 --model primary --json
python3 verify.py --no-thinking                     # sends chat_template_kwargs.enable_thinking=false
```

It runs four checks with synthetic prompts: `models` (reachability and the model id), `tool_calls`
(streamed structured tool calls), `tool_corruption` (a repeated agent-shaped tool-call request, so
later attempts hit the prefix cache) and `json_schema` (strict structured output). `--skip` leaves
checks out, `--timeout` sets the per-request limit. For the API key, prefer the env vars
`VLLM_API_KEY` / `OPENAI_API_KEY`; `--api-key` also works, but a key on the command line is visible
to other users in `ps`. Exit codes: `0` all checks passed, `1` a check failed, `2`
the server is unreachable. A failed check prints a hint that points to the matching section of
[docs/field-notes.md](docs/field-notes.md).

Re-run it after every image or flag change, and before and after any benchmark run.

## The tested configuration

Every value is an environment variable of `serve.sh`, with the tested value as its default:

| Variable | Default | `vllm serve` flag |
|---|---|---|
| `SERVED_MODEL_NAME` | `primary` | `--served-model-name` |
| `MTP_TOKENS` | `3` | `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (`0` or `--mtp-off` = off) |
| `PREFIX_CACHING`, `MAMBA_CACHE_MODE` | `1`, `align` | `--enable-prefix-caching --mamba-cache-mode align` (`--no-prefix-caching` = off) |
| `KV_CACHE_MEMORY_BYTES` | `21474836480` (20 GiB) | `--kv-cache-memory-bytes` |
| `GPU_MEMORY_UTILIZATION` | `0.55` | `--gpu-memory-utilization` (startup check only) |
| `KV_CACHE_DTYPE` | `fp8` | `--kv-cache-dtype` |
| `MAX_MODEL_LEN` | `131072` | `--max-model-len` (128k) |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | `--max-num-batched-tokens` |
| `TOOL_CALL_PARSER` | `qwen3_coder` | `--enable-auto-tool-choice --tool-call-parser` |
| `REASONING_PARSER` | `qwen3` | `--reasoning-parser` |
| `LANGUAGE_MODEL_ONLY` | `1` | `--language-model-only` (required with MTP on vLLM 0.30) |
| `LOAD_FORMAT` | `fastsafetensors` | `--load-format` (empty = vLLM default) |
| `PORT`, `HOST` | `8000`, `127.0.0.1` | `--port`, `--host` |
| `API_KEY` | empty | env `VLLM_API_KEY` in the container (same effect as `--api-key`) |
| `MODEL` | `nvidia/Qwen3.8-27B-NVFP4` | the model argument |
| `MODEL_REVISION` | `482ca0f3…` (full hash in `serve.sh`) | `--revision`, `--tokenizer-revision` (empty = no pin) |
| `HF_HOME` | `$HOME/.cache/huggingface` | host cache, mounted into the container |
| `IMAGE`, `NAME` | `spark-vllm-agent:0.30-pr50021`, `vllm-agent` | container settings |
| `RESTART_POLICY`, `PRIVILEGED` | `unless-stopped`, `0` | container settings (`PRIVILEGED=1` adds `--privileged`) |
| `EXTRA_ARGS` | empty | extra `vllm serve` flags |

The container runs with `--gpus all --network host --ipc host`. Two deliberate differences from the
tested server:

- The tested server listened on `0.0.0.0`; `serve.sh` defaults to `127.0.0.1`, because vLLM has no
  authentication without an API key. For access from other machines, set `HOST=0.0.0.0` and
  `API_KEY`; `serve.sh` refuses a non-loopback `HOST` without `API_KEY`. The key goes into the
  container as the env var `VLLM_API_KEY`, not onto a command line where `ps` would show it (anyone
  with Docker access can still read it with `docker inspect`). It only guards `/v1/*`: **`/metrics`
  and `/health` stay unauthenticated**, so firewall the port.
- The tested server ran `--privileged`. The GPU comes in through the NVIDIA container runtime
  (`--gpus all`), so `serve.sh` runs without it by default, but that is **untested on a Spark**. If
  vLLM finds no GPU, try `PRIVILEGED=1`, which reproduces the tested server exactly.

### Adopt an existing server

If a vLLM server already runs on the port, don't replace it. Check what it serves
(`curl -s http://localhost:8000/v1/models`), run `python3 verify.py --model <its model id>` against
it, and point your harness at it. To run this cookbook's server next to it, pick another port
(`PORT=8001 ./serve.sh --allow-second-server`), but two 27B servers rarely fit in one Spark's memory.
To switch servers, stop the old one yourself and keep it for [rollback](#rollback).

## Fallback: MTP off

Use this when the pinned nightly is gone, the build fails, or you see engine faults with MTP. Without
PR #50021, **MTP must stay off**: `serve.sh` refuses MTP on an image without the PR label.

- **Your built image, MTP off:** `./serve.sh --mtp-off`.
- **Same base, no patch** (vLLM 0.30.1rc1 for sm_121). Pull it yourself, then:

  ```bash
  docker pull eugr/spark-vllm@sha256:d1e9d16ad8958ddc2e88686416cefdc1ab679047fcdeacb747a6a0d2385ad932
  IMAGE=eugr/spark-vllm@sha256:d1e9d16ad8958ddc2e88686416cefdc1ab679047fcdeacb747a6a0d2385ad932 \
    ./serve.sh --mtp-off --dry-run
  ```

- **Official vLLM release image.** `vllm/vllm-openai:v0.30.0` is published for arm64 (the tag is
  multi-arch; `v0.30.0-aarch64` is the arm64-only tag). Pinned below by its multi-arch index digest
  `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90` (arm64 manifest
  `sha256:4864d46625cbc3307623e29ac742030655e27249feba7b97ec925ce4cc4dfb56`). Its kernels are built
  for the SM12x family, so NVFP4 should run on GB10, but **it is not tested on a Spark**. Its
  entrypoint is already `vllm serve`, which `serve.sh` detects. If the `fastsafetensors` loader is
  missing there, set `LOAD_FORMAT=`:

  ```bash
  docker pull vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90
  IMAGE=vllm/vllm-openai:v0.30.0@sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90 \
    LOAD_FORMAT= ./serve.sh --mtp-off --dry-run
  ```

Prefix caching stays on in the fallback: the upstream prefix-cache corruption needed MTP, and
without MTP it did not reproduce in our tests (0 corrupted in 810 checked responses on an older
build). The cost of MTP off is about half the decode speed: on an older vLLM build of the same model
and flags, single-stream decode was 11.8 tok/s and the 4-way aggregate 30.8 tok/s. Repeat TTFT
improves (about 0.55 s, no #53670 block drop). Run `verify.py` on the fallback as well.

## Known issues

Short list; symptoms, detection and fixes are in [docs/field-notes.md](docs/field-notes.md).

- **[Tool-call corruption with prefix caching + MTP](docs/field-notes.md#tool-call-corruption-with-prefix-caching--mtp)**
  on vLLM builds before 0.28 (#43559, fixed by #51113 and #51812). The tested build is clean, and
  `verify.py`'s `tool_corruption` check shows it on yours.
- **[Engine crash with MTP on GDN layers](docs/field-notes.md#engine-crash-with-mtp-on-gdn-layers-pr-50021)**
  without PR #50021. Rare wedges and faults with MTP are still reported upstream on newer builds.
- **[Repeat TTFT about 1.1 s instead of 0.55 s](docs/field-notes.md#repeat-ttft-with-mtp--prefix-caching-53670)**
  with MTP + prefix caching (#53670): the last cached block is recomputed. Latency, not correctness.
- **Engine wedges keep `/health` green.** `serve.sh` uses `--restart unless-stopped`, which handles
  crashes; add a [stall watchdog](docs/field-notes.md#watchdog-and-restart-policy) for wedges.
- **Memory:** keep the KV cache pinned and leave headroom for the host
  ([memory flags](docs/field-notes.md#memory-flags-on-unified-memory)).
- **The nightly base digest may disappear from Docker Hub.** Keep your built image, and use the
  fallback if it is gone ([details](docs/field-notes.md#the-nightly-base-image-may-disappear)).

## Rollback

When you replace an older server, keep the previous container **stopped, not deleted**:
`docker stop <old>` and `docker update --restart=no <old>`, then start this one. To roll back,
`docker rm -f vllm-agent`, then `docker update --restart=unless-stopped <old>` and `docker start <old>`.
Keep the old image too. The full sequence is in the
[field notes](docs/field-notes.md#rollback-keep-the-previous-server-stopped-not-deleted).

## Using it from agent harnesses

**Any OpenAI-compatible client.** Base URL `http://localhost:8000/v1`, model `primary`, API key
`EMPTY` (or your `API_KEY`). Tool calling uses the standard `tools` / `tool_choice: "auto"` fields;
the context window is 131,072 tokens.

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=EMPTY          # or your API_KEY
curl -s "$OPENAI_BASE_URL/chat/completions" -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H 'Content-Type: application/json' -d '{
    "model": "primary",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{"type": "function", "function": {"name": "get_weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
    "chat_template_kwargs": {"enable_thinking": false}
  }'
```

Harness tips from the field notes: put the system message first and keep it stable (prefix-cache
hits), don't rely on `tool_choice: "required"` or a named tool (#54808), send
`chat_template_kwargs` only to servers that accept it, and retry requests that fail while the server
restarts.

**Hermes Agent.** Use Hermes' custom OpenAI-compatible provider (`vllm` is an alias of `custom`),
in `$HERMES_HOME/config.yaml`:

```yaml
model:
  provider: custom
  base_url: http://localhost:8000/v1
  default: primary
  # api_key: <your API_KEY>     # only if you set one
  # context_length: 131072      # optional; Hermes normally detects it
```

or with `hermes config set model.provider custom`, `hermes config set model.base_url
http://localhost:8000/v1` and `hermes config set model.default primary`.

**PAN.** [PAN](https://github.com/FRST-FRYTL/pan-agent), a memory add-on for Hermes, uses this server as
its tested stack and for its memory gate. Check the server with
`pan doctor --base-url http://localhost:8000/v1 --model primary`,
then set `models.gate.base_url: http://localhost:8000/v1` and `models.gate.model: primary` in
`$HERMES_HOME/pan/config.yaml`. The gate and Hermes' main model can share this one server.

## Claude Code skill and plugin

A Claude Code skill walks through these steps with you, or adopts a server that already runs:
[`skills/install-vllm-cookbook`](skills/install-vllm-cookbook/SKILL.md). Install it as a plugin:

```text
/plugin marketplace add FRST-FRYTL/spark-vllm-agent-cookbook
/plugin install vllm-dgx-spark@spark-vllm-agent-cookbook
```

## Other GPUs (generic notes, untested)

- The pinned base image is arm64 and built for sm_121 only. On x86_64 or another GPU, start from an
  official `vllm/vllm-openai` release image, with `IMAGE=… ./serve.sh --mtp-off` as the starting
  point.
- NVFP4 weights need a Blackwell-class GPU. On other GPUs, use another quantization of Qwen3.8-27B
  and check that vLLM supports it.
- Size `KV_CACHE_MEMORY_BYTES` and `GPU_MEMORY_UTILIZATION` for your card. On a discrete GPU, the
  unified-memory reasoning above does not apply.
- Keep MTP off unless your vLLM contains PR #50021 (or its merged successor). Whatever you run, check
  it with `verify.py`. The numbers above hold only for the tested stack.

## Files

| Path | What it is |
|---|---|
| [`Dockerfile`](Dockerfile), [`build.sh`](build.sh) | the local image: pinned base + PR #50021, no push |
| [`serve.sh`](serve.sh) | starts one container with the tested flags; refuses to touch anything else |
| [`verify.py`](verify.py), `tests/` | the endpoint checks and their tests (`python3 -m pytest -q tests`) |
| [`docs/field-notes.md`](docs/field-notes.md) | pitfalls, upstream issues, watchdog, memory, rollback |
| [`skills/install-vllm-cookbook`](skills/install-vllm-cookbook/SKILL.md), `.claude-plugin/` | the Claude Code skill and plugin manifest |

Contributions: [CONTRIBUTING.md](CONTRIBUTING.md). Security: [SECURITY.md](SECURITY.md). Changes:
[CHANGELOG.md](CHANGELOG.md).

## Credits and licences

- **Base image:** [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) (MIT) builds the
  `eugr/spark-vllm` images this cookbook starts from. The image itself contains third-party software
  (CUDA, PyTorch, vLLM and others) under their own licences; it is pulled from Docker Hub by you, not
  redistributed here.
- **vLLM:** [vllm-project/vllm](https://github.com/vllm-project/vllm), Apache-2.0. The PR #50021
  diff is fetched from GitHub at build time and checked by sha256; it is not redistributed here.
- **Model:** [`nvidia/Qwen3.8-27B-NVFP4`](https://huggingface.co/nvidia/Qwen3.8-27B-NVFP4), NVIDIA's
  quantization of [`Qwen/Qwen3.8-27B`](https://huggingface.co/Qwen/Qwen3.8-27B). NVIDIA's model card
  states **Apache 2.0**. Read the model card (and the base model's) before use; you accept those terms
  when you download it. The weights are not redistributed here.
- **This cookbook's own files:** MIT, see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Built and maintained by FRST-FRYTL.
