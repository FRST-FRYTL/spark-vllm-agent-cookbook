# Contributing to spark-vllm-agent-cookbook

Thanks for your interest. This cookbook is a research preview maintained by one person in their
spare time. Issues and pull requests are welcome and handled **best-effort**. There are no
response-time or support promises. Small, focused changes are the easiest to review.

Especially useful: reports from a **fresh install** (it is not yet tested end to end), results on
other DGX Spark units or driver versions, and updates when an upstream fix (PR #50021, #53670) lands
in a vLLM release.

## How changes flow

This public repo is published as **one squashed commit per release**. Development happens in a
private workspace. Here is what happens to a pull request:

1. You open a PR against `main` here. CI runs the tests, `bash -n` and shellcheck on the scripts,
   a secret scan and a DCO check.
2. Review happens on the PR as usual.
3. When it is accepted, the maintainer applies your commits to the development workspace with your
   authorship and `Signed-off-by` intact. The PR is then closed, with a pointer to the release that
   contains it. It is not merged here.
4. Your change ships in the next release, and the CHANGELOG credits it.

So your PR shows as "closed", not "merged". That is expected.

## Developer Certificate of Origin (DCO)

Every commit must be signed off, which certifies the [DCO](https://developercertificate.org/):

```bash
git commit -s -m "serve.sh: explain why X"
```

The `Signed-off-by:` line must match the commit author. CI rejects PRs with unsigned commits. To fix
the last commit: `git commit --amend -s`. To fix several: `git rebase --signoff main`.

## Development setup

No GPU is needed for the checks CI runs:

```bash
python3 -m pip install pytest
python3 -m pytest -q tests                 # verify.py tests (no network, no GPU)
bash -n build.sh serve.sh
shellcheck -S warning build.sh serve.sh
./build.sh --dry-run                       # prints the commands, runs nothing
./serve.sh --dry-run                       # needs Docker; runs the checks, starts nothing
```

On a Spark, a change to the image, the flags or the scripts should be checked end to end: build,
serve, then `python3 verify.py`. Say in the PR what you ran it on (hardware, driver, image digest).

## Ground rules

- **Pin everything.** Base images by digest, patches by commit and sha256, model weights by
  revision. A change that makes a build depend on a moving tag will be rejected.
- **Never redistribute images or weights.** No `docker push` steps, no image or weight downloads
  from anywhere but their original sources, and nothing vendored from them.
- **Safe defaults stay.** `serve.sh` must keep refusing to replace other containers, to take a used
  port, to serve beyond loopback without an API key, and to enable MTP on an unpatched image.
- **Numbers come with their scope.** When you add or change a measured number, say what hardware,
  image, flags and sample size it comes from.
- **No private data** in code, docs or fixtures: no real host names, internal addresses, paths
  under your home directory, or tokens. Use `localhost`, `example.com` and RFC 5737 addresses. CI
  runs gitleaks for secrets; everything else is checked in review.
- Keep the style of the surrounding code. Tests go with every behaviour change of `verify.py`.

## Reporting bugs

Open an issue (the bug report template lists what to include): the step that failed, the output of
the `--dry-run` of that script, `python3 verify.py --json` if the server starts, the hardware and
driver, and the relevant `docker logs` lines. Redact anything private first. For security issues,
see [SECURITY.md](SECURITY.md).
