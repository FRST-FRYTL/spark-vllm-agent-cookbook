# Field notes: Qwen3.8-27B-NVFP4 on vLLM for agent tool calling (DGX Spark)

These notes come from running agent harnesses (Hermes Agent, plus a memory add-on that makes its own
structured-output calls) against one local vLLM server on one NVIDIA DGX Spark. They apply to the
tested stack: `eugr/spark-vllm` `nightly-20260927` (vLLM `0.30.1rc1.dev220`) plus vLLM PR #50021,
serving `nvidia/Qwen3.8-27B-NVFP4` with MTP 3 and prefix caching. Where a note refers to an older
build, it says so.

Most of these failures are **silent**: the server stays healthy while the agent gets worse. Each
section gives the symptom, the cause, how to detect it, and what to do. Re-check them after any change
to the image or the flags, and re-run `python3 verify.py` after every change. When `verify.py` fails
a check, its hint points to one of the sections below.

All numbers are the author's measurements on one machine, with small sample sizes. Treat them as
orientation, not as a benchmark.

- [Tool-call corruption with prefix caching + MTP](#tool-call-corruption-with-prefix-caching--mtp)
- [Engine crash with MTP on GDN layers (PR #50021)](#engine-crash-with-mtp-on-gdn-layers-pr-50021)
- [Repeat TTFT with MTP + prefix caching (#53670)](#repeat-ttft-with-mtp--prefix-caching-53670)
- [Structured output: json_schema vs json_object](#structured-output-json_schema-vs-json_object)
- [Thinking on or off per request](#thinking-on-or-off-per-request)
- [Memory flags on unified memory](#memory-flags-on-unified-memory)
- [Watchdog and restart policy](#watchdog-and-restart-policy)
- [The nightly base image may disappear](#the-nightly-base-image-may-disappear)
- [Rollback: keep the previous server stopped, not deleted](#rollback-keep-the-previous-server-stopped-not-deleted)
- [Reading tool-call test results](#reading-tool-call-test-results)
- [Smaller items](#smaller-items)

## Tool-call corruption with prefix caching + MTP

**Symptom.** The model prints tool calls as plain text, produces garbled `<tool_call>` blocks
(missing `function=`, nested or foreign formats), sends empty arguments, or announces a call and
stops. In streaming mode the tool parser silently drops a malformed block, so the client gets an empty
reply. From the agent's side it looks like an agent bug or a weak model.

**Cause.** On vLLM 0.21–0.25, MTP speculative decoding **combined with** prefix caching corrupted
the outputs of hybrid Qwen3.x models (Qwen3.8's linear-attention GDN layers run through vLLM's Mamba
cache path) on a prefix-cache hit. Measured on an older build (vLLM 0.22) with Hermes-shaped requests:

| Condition | Structured tool calls |
|---|---|
| prefix-cache hit, MTP 3 on | about 19 %; the probability of `<tool_call>` as the first token fell from 0.97–0.99 to 0.30–0.67 |
| fresh prefill (caching off, or a new `cache_salt`) | 120 of 121 |
| prefix caching on, MTP off | 0 corrupted in 810 checked responses |

The tool parser (`qwen3_coder` vs `qwen3_xml`), streaming, thinking mode and temperature were ruled
out. Upstream tracking issue: [vLLM #43559](https://github.com/vllm-project/vllm/issues/43559). It
was closed by [PR #51113](https://github.com/vllm-project/vllm/pull/51113) ("Keep mamba align
prefill chunks block-aligned past last_cache_position"), with the related
[PR #51812](https://github.com/vllm-project/vllm/pull/51812) ("Align Qwen GDN gates with speculative
tokens"). Both are in vLLM **0.28.0 and later**, and the tested image contains both.

**On the tested image**, with MTP 3 and prefix caching both on, the same test found **0 corrupted tool
calls in 914 responses**: 359 in a 300-request run at 4-way concurrency, and 555 in a 15-minute soak
with 6 workers. The test used Hermes-shaped prompts of about 7.1–7.6k tokens with 20 native tool
schemas, 6 shared prefix groups, multi-turn tool histories, streaming and non-streaming, and
thinking on and off. Every response was checked for a structured call to a known tool, valid JSON
arguments, schema validity, and tool-call text leaking into `content`.

**Detect.** `verify.py` runs two checks for this:
- `tool_calls`: streamed requests that must each come back as a structured call with the right
  function name and JSON arguments;
- `tool_corruption`: one agent-shaped request (persona, tool-use rules, several realistic tool
  schemas) sent repeatedly, so the second and later requests hit the prefix cache. Every answer must
  be a clean call: valid JSON arguments, no tool-call markup in `content`, the same function each
  time.

Run it after every image or flag change. If you benchmark agents, run it before and after each run:
a run that fails it at either end is not worth scoring.

**What to do.**
- Use a vLLM build with #51113 and #51812 (0.28.0 or later). The tested image has them.
- On an older build, turn off one of the two. `./serve.sh --mtp-off` keeps prefix caching and was
  clean in the test above; it halves single-stream decode speed. `./serve.sh --no-prefix-caching`
  keeps MTP; it roughly doubled time to first token on long agent prompts in our tests.
- Re-baseline after the fix. The bug makes every agent on that server look worse, so comparisons
  measured before the fix don't carry over.

## Engine crash with MTP on GDN layers (PR #50021)

**Symptom.** Under sustained load the engine crashes or hangs. On an older build without the patch
(vLLM 0.22) this happened three times in about 33 hours of benchmark load: two hangs with 2 and 4
concurrent requests, and one crash with an illegal memory access in the GDN prefill kernels. Short
stress runs (about 26 minutes in total, up to 8 workers) did not reproduce it.

**Cause.** The likely cause, inferred and not proven locally: in the GDN speculative-decoding kernels
(`fla/ops/fused_recurrent.py`, `fused_sigmoid_gating.py`), `i_t = num_accepted_tokens - 1` indexes
the recurrent-state index table with no bound, and a garbage index gets dereferenced. That gives an
illegal memory access (Xid 31 in `dmesg`) or a wedged engine. CUDA reports faults asynchronously, so
the kernel named in the traceback is not necessarily the one that faulted. MTP with 1 speculative
token runs the same unbounded path, so it is not a workaround.
[vLLM PR #50021](https://github.com/vllm-project/vllm/pull/50021) ("Bound accepted-token state
lookups in GDN/KDA spec decode") masks that load. In its thread, an unpatched build reproduces the
Xid 31 fault with Qwen3.8-27B-NVFP4 + MTP 3 on an RTX PRO 6000, and a patched build does not. The PR
was unmerged and in no release or prebuilt image when this cookbook was written. Related reports for
the same family: #36613 (closed), #40756, #55775, #58422.

**Detect.** `illegal memory access` or `EngineCore encountered a fatal error` in `docker logs`, `Xid`
lines in `dmesg`, or a stalled engine (see [Watchdog and restart policy](#watchdog-and-restart-policy)).

**What to do.**
- **Apply the patch** (what this cookbook does). It touches only Python and Triton files, so it
  layers onto a recent image without a CUDA compile. `build.sh` pins both inputs: the base image by
  digest, and the PR diff by commit (`71d7c782`) and sha256. The `Dockerfile` checks after applying
  it that the bounded load is in place. `serve.sh` refuses MTP on an image without the PR label.
- **Or turn MTP off** for stability: `./serve.sh --mtp-off`. On the older build, single-stream decode
  fell from 23.7–25.8 to 11.8 tok/s.
- **Gate the switch** on the tool-call check above plus a long soak under real load. The incidents
  were rare, so a short test cannot rule them out. On the tested image a 24-hour soak under real
  benchmark load (37 runs) showed no engine fault, stall or restart (max stall 30 s). The kernel logged 8
  transient `NV_ERR_NO_MEMORY` messages in that window: 3 while a co-tenant process held about 20 GB, and 5
  unexplained. None affected the server.
- Keep a restart policy and a stall watchdog anyway: wedges and faults with MTP on hybrid models are
  still reported upstream on newer builds (#58422, #55775, #40756), and on GB10 a FULL-CUDA-graph
  fault with padded batches is reported
  ([local-inference-lab/vllm#823](https://github.com/local-inference-lab/vllm/issues/823)).
- Keep the old server stopped, not deleted, as the
  [rollback](#rollback-keep-the-previous-server-stopped-not-deleted).

## Repeat TTFT with MTP + prefix caching (#53670)

**Symptom.** A repeated long prompt, such as an agent's system prompt, does not reuse its whole cached
prefix. Repeat time to first token is about 1.1 s instead of about 0.55 s, and the prefix-cache hit
rate is about 64 % instead of about 81 %.

**Cause.** With MTP on, vLLM drops the last prefix-cache block and recomputes it. On this model vLLM
raises the attention block size to **1600 tokens** (so that the attention page covers the Mamba
page). A ~7.1k-token shared prefix therefore fills 4 full blocks, but only 3 are reused, even on an
identical repeat:

| Request (same ~7.1k-token system prompt) | Prompt tokens | Cached | Recomputed | TTFT |
|---|---|---|---|---|
| 1, cold | 7141 | 0 | 7141 | 2.85 s |
| 2, same prefix, new question | 7147 | 4800 | 2347 | 0.93 s |
| 3, identical repeat of 2 | 7147 | 4800 | 2347 | 0.94 s |
| 4, same prefix, new question | 7144 | 4800 | 2344 | 0.94 s |

At about 2500 prompt tokens/s of prefill, the recomputed block explains the extra ~0.4 s. Upstream:
[vLLM #53670](https://github.com/vllm-project/vllm/issues/53670). The behaviour matches that issue;
its root cause was not checked here.

**Detect.** The prefix-cache counters in `/metrics` (`vllm:prefix_cache_hits_total` over
`vllm:prefix_cache_queries_total`), `usage.prompt_tokens_details.cached_tokens` in responses (when
enabled), or the time to first token of two identical long requests.

**What to do.** It is a latency cost, not corruption. MTP roughly doubles decode speed, so MTP 3 +
prefix caching was still the fastest configuration end to end for an agent turn. Estimates for one
turn with a ~7.5k-token prompt (the MTP-only row was not measured on this image):

| Configuration | Repeat TTFT | Decode | 100-token tool call | 512-token answer |
|---|---|---|---|---|
| prefix caching only (older build, MTP off) | 0.55 s | 11.8 tok/s | ~9 s | ~44 s |
| MTP only (prefix caching off) | ~3.7 s | ~25 tok/s | ~7.7 s | ~24 s |
| MTP 3 + prefix caching (tested) | ~1.1 s | ~25 tok/s | ~5 s | ~21 s |

If you prefer low TTFT over decode speed (short answers, many turns), run `./serve.sh --mtp-off`.

## Structured output: json_schema vs json_object

**Symptom.** A client that asks for `response_format: {"type": "json_schema", ...}` gets HTTP 400,
or gets text that does not match the schema, and falls back to something worse.

**Cause.** Not every OpenAI-compatible server supports strict `json_schema`. vLLM does (structured
outputs, backend `auto`), and the tested server accepted strict schemas for a structured-output
workload of about 1,500 calls with no fallbacks, parse failures or truncations. Older servers,
other runtimes and some hosted APIs only accept `json_object`, or nothing.

**Detect.** `verify.py`'s `json_schema` check sends one request with a strict schema and validates
the answer against it. If the server rejects `json_schema`, try `json_object` for that client.

**What to do.**
- On this stack, use `json_schema` with `strict: true`.
- Where only `json_object` works, ask for it, put the schema in the prompt, and validate the output
  yourself. Where neither works, parse strictly and treat malformed output as a failure, not as data.
- With the `qwen3` reasoning parser, thinking goes into `reasoning_content` and the JSON into
  `content`. For short structured calls, turn thinking off (next section) or leave enough
  `max_tokens` for both parts, and check `finish_reason`: `length` means the JSON is cut off.

## Thinking on or off per request

**Symptom.** Agent turns are slow, or a client gets HTTP 400 from one server but not from another
with the same code.

**Cause.** Qwen3-style chat templates think by default. vLLM and SGLang accept
`chat_template_kwargs: {"enable_thinking": false}` in the request body to switch the reasoning phase
off per request. It is a server extension, not part of the OpenAI API, and some hosted APIs reject
unknown fields with HTTP 400.

**Detect.** `python3 verify.py --no-thinking` sends `chat_template_kwargs.enable_thinking=false`
with every request. Run it that way when your harness sends the field, and without it when it
doesn't.

**What to do.** On this server, sending `enable_thinking: false` makes tool-calling turns much
faster, and the tool-call results above held with thinking on and off. Send the field only to
servers that accept it.

## Memory flags on unified memory

The Spark has about 121 GB of unified memory shared by the CPU and the GPU, so vLLM's memory settings
also decide whether the host swaps.

- **Pin the KV cache.** Without `--kv-cache-memory-bytes`, vLLM misestimated CUDA-graph memory on the
  Spark, took about 87 GB, and the host started swapping. With the KV cache pinned at 20 GiB
  (`KV_CACHE_MEMORY_BYTES=21474836480`), the server uses about 45–49 GB in total. On the tested image
  that is **520,234 KV tokens**, about 3.97 concurrent sequences at the full 131,072-token context.
  The startup log warns that 3 padding layers may waste up to 6.25 % of the KV cache; that is expected.
- **Keep `--gpu-memory-utilization 0.55`.** Even with a pinned KV cache, vLLM checks at startup that
  this share of total memory (about 66 GB on a Spark) is free. `--kv-cache-memory-bytes` alone fails
  at startup, because the default of 0.92 is still checked. `serve.sh` warns when less than about
  70 GB is available.
- **Watch co-tenants.** Another GPU process of about 20 GB next to vLLM left about 25 GB available,
  pushed the host into swap, and the kernel logged `NVRM: ... Out of memory` lines. vLLM itself kept
  running, but leave headroom for the host and don't run a second large model next to it. Two 27B
  servers rarely fit on one Spark.
- **Startup memory spikes** on DGX Spark are reported upstream
  ([vLLM #56824](https://github.com/vllm-project/vllm/issues/56824)); another reason to pin the KV
  cache and start with plenty of free memory.
- Check it with `free -g` (the `available` column) and `nvidia-smi` while the server runs.

## Watchdog and restart policy

**Symptom.** The server stops producing tokens, but the process keeps running and `/health` keeps
answering 200. Requests hang until they time out.

**Cause.** A **crash** ends the process, and a container restart policy (`--restart unless-stopped`,
the `serve.sh` default) brings it back. A **wedge** does not: the engine loop is stuck, nothing
exits, and `/health` stays green. Wedges and faults with MTP are rare, so a short test cannot rule
them out.

**Detect.** Watch progress, not `/health`: when requests are running (`vllm:num_requests_running`
> 0) but `vllm:generation_tokens_total` has not moved for a few minutes, the engine is wedged. Also
watch `docker logs` for `illegal memory access` or `EngineCore encountered a fatal error`, and
`dmesg` for `Xid`.

**What to do.** Run a restart policy for crashes and a stall watchdog for wedges. An example
watchdog (a sketch; run it under your own supervisor, and adjust `NAME`, `PORT` and the limit):

```bash
#!/usr/bin/env bash
# Restart the cookbook container when requests are running but no tokens were generated for 180 s.
NAME=vllm-agent PORT=8000 LIMIT=180 STEP=30
prev=""; stalled=0
while sleep "$STEP"; do
  m="$(curl -sf --max-time 10 "http://localhost:${PORT}/metrics")" || continue
  running="$(awk '/^vllm:num_requests_running/ {s+=$2} END {printf "%d", s}' <<<"$m")"
  gen="$(awk '/^vllm:generation_tokens_total/ {s+=$2} END {printf "%d", s}' <<<"$m")"
  if [ "$running" -gt 0 ] && [ "$gen" = "$prev" ]; then stalled=$((stalled + STEP)); else stalled=0; fi
  prev="$gen"
  if [ "$stalled" -ge "$LIMIT" ]; then
    echo "$(date -u +%FT%TZ) engine stalled ${stalled}s with ${running} running; restarting ${NAME}"
    docker restart "$NAME"; stalled=0; prev=""
  fi
done
```

- `/metrics` is unauthenticated even with an API key, so the watchdog needs no key.
- Clients should retry a request that fails during a restart. A restart takes minutes (the first
  start of a new container about 9–10 minutes), so use a long retry budget or queue work.
- Agent benchmark harnesses should treat a stall as an invalid run, restart the server, and re-run
  the affected units, rather than score the timeouts.

## The nightly base image may disappear

- The base is `eugr/spark-vllm` `nightly-20260927`, pinned by digest
  `sha256:d1e9d16ad8958ddc2e88686416cefdc1ab679047fcdeacb747a6a0d2385ad932`. Nightly tags can be
  removed from Docker Hub. When the digest is gone, `build.sh` fails with "manifest unknown", and
  this exact build cannot be reproduced.
- If you have built it once, keep the local image (`docker image ls spark-vllm-agent`) and don't prune
  it. You may back it up for yourself with `docker save`, but don't publish it: the base contains
  third-party software under its own licences.
- Otherwise: use the README's fallback (stock vLLM, MTP off, no patch), or move the `FROM` digest in
  the `Dockerfile` and `BASE_IMAGE` in `build.sh` to a newer eugr nightly, check that the PR still
  applies (the build fails if it does not), and re-run `verify.py` plus a soak before trusting it.
- Once PR #50021 (or a successor) is merged and released, this cookbook should move to a stock vLLM
  release.
- Upstream: [eugr/spark-vllm on Docker Hub](https://hub.docker.com/r/eugr/spark-vllm) ·
  [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) ·
  [vLLM PR #50021](https://github.com/vllm-project/vllm/pull/50021)

## Rollback: keep the previous server stopped, not deleted

When you replace an older server with this one, stop the old container and disable its restart
policy, but don't remove it. `serve.sh` never does this for you; it refuses to start while the port
is taken or another vLLM container is running.

```bash
docker stop <old-container>
docker update --restart=no <old-container>     # so a reboot does not start both servers
./serve.sh --wait
python3 verify.py
# rollback:
docker rm -f vllm-agent
docker update --restart=unless-stopped <old-container>
docker start <old-container>
```

Keep the old image too (no `docker image prune -a` while you might need it). Record which image,
flags and model revision each container ran, so a rollback restores a known state.

## Reading tool-call test results

Two things look like corruption and are not:

- **Truncation.** A tool call cut off at `max_tokens` has unparseable arguments. Check
  `finish_reason`: `length` means truncation, not corruption. Long arguments (for example a
  `delegate_task` goal) need a larger `max_tokens`. The only "bad JSON" result in the 914-response
  test above was one of these.
- **Batch nondeterminism.** At temperature 0, two identical requests can still differ under
  concurrent load (in one test, 26 % of repeat pairs differed in content, for example one tool call vs
  two), because batching changes the numerics. Both answers were valid calls. Positional
  corruption looks different: malformed calls, markup in `content`, empty replies.

Also check that a benchmark's "no tokens streamed" errors are not the same artifact: a tool call
truncated at a small `max_tokens` with `ignore_eos` can end with nothing in `content` at all.

## Smaller items

- **`--language-model-only` is required with MTP on vLLM 0.30.** Qwen3.8-27B-NVFP4 keeps its vision
  tower, and multimodal + native MTP crashes in `profile_run`
  ([vLLM #58203](https://github.com/vllm-project/vllm/issues/58203)). With the flag, the server is
  text-only. `--limit-mm-per-prompt '{"image":0,"video":0}'` is the other documented workaround.
- **First start is slow:** engine init took 533 s on the tested image, about 9.5 minutes until the
  container reported healthy. Most of it is FlashInfer FP4/FP8 GEMM autotuning, plus `torch.compile`
  and CUDA graphs. `serve.sh` sets a 900 s health start period.
- **Batch ≥ 4 with MTP:** hybrid 27B models can collapse to about 3 running sequences
  ([vLLM #55533](https://github.com/vllm-project/vllm/issues/55533)). It was not seen at 4-way
  concurrency on the tested image; watch the aggregate throughput at your concurrency.
- **Tool parser:** on vLLM 0.30, `qwen3_coder` maps to a new parser implementation
  (`Qwen3EngineToolParser`). Open upstream issues include `tool_choice: "required"` or a named tool
  being ignored ([#54808](https://github.com/vllm-project/vllm/issues/54808)) and a dropped last
  parameter when `</parameter>` is missing
  ([#57699](https://github.com/vllm-project/vllm/issues/57699)). Harnesses that rely on forced tool
  choice should validate the answer.
- **System message first:** the chat template requires the system message at the start. vLLM merges
  later system messages into the leading system prompt, which changes the prefix and lowers cache
  reuse. Keep the system prompt stable and append-only if you want prefix-cache hits.
- **Sampling:** the model's `generation_config` sets temperature 1.0, top_k 20, top_p 0.95 when the
  client sends nothing. Set what you want explicitly.
- **`vllm serve --help=all` fails without a GPU** in this image ("Failed to infer device type"). To
  check flags while the GPU is busy, read `vllm/engine/arg_utils.py` in the image instead.
- **The GDN kernels use Triton/FLA** (the only backend on sm_121), and vLLM keeps
  `cudagraph_mode=FULL_AND_PIECEWISE` with MTP on this image; older builds fell back to `PIECEWISE`.
- **Downloads:** if `hf download` stalls, run it again; it resumes. Download as your own user so the
  cache is not owned by root.
