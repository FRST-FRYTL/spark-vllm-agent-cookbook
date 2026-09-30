"""verify.py against a fake OpenAI-compatible server on 127.0.0.1 (no real endpoint, no GPU).

Run from the repo root:  python3 -m pytest -q tests
"""

from __future__ import annotations

import ast
import json
import re
import socket
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import verify  # noqa: E402
from verify import FAIL, OK, SKIP, WARN, Endpoint  # noqa: E402


class FakeEndpoint:
    """Serves ``/v1/models`` and ``/v1/chat/completions`` (SSE and JSON) and records request bodies.

    ``corrupt``: None | "always" | "repeat" (only from the second identical request on): the reply then
    carries tool-call markup in ``content`` instead of structured ``tool_calls``.
    ``json_schema``: "ok" | "reject" (HTTP 400; json_object still works) | "invalid" (ignores the schema).
    ``chat_status``: answer every chat request with this HTTP status instead (e.g. 500, an engine crash).
    """

    def __init__(self, models=("fake-model",), corrupt: str | None = None, json_schema: str = "ok",
                 chat_status: int | None = None) -> None:
        self.models = list(models)
        self.corrupt = corrupt
        self.json_schema = json_schema
        self.chat_status = chat_status
        self.requests: list[dict[str, Any]] = []
        self.headers: list[dict[str, str]] = []
        self._seen: dict[str, int] = {}
        self._lock = threading.Lock()
        self._server: ThreadingHTTPServer | None = None

    def __enter__(self) -> "FakeEndpoint":
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()

    @property
    def base_url(self) -> str:
        assert self._server is not None
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    # -- scripted model --------------------------------------------------------------------------

    def _repeat_count(self, body: dict[str, Any]) -> int:
        key = json.dumps(body, sort_keys=True)
        with self._lock:
            self.requests.append(body)
            self._seen[key] = self._seen.get(key, 0) + 1
            return self._seen[key]

    @staticmethod
    def _answer_call(body: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """The call a well-behaved model would make for the offered tools and the user turn."""
        prompt = body["messages"][-1]["content"]
        names = [t["function"]["name"] for t in body["tools"]]
        if "read_file" in names:
            return "read_file", {"path": re.search(r"(\S+\.md)", prompt).group(1)}
        if names == ["add"]:
            a, b = (float(x) for x in re.findall(r"\d+(?:\.\d+)?", prompt)[:2])
            return "add", {"a": a, "b": b}
        return "get_time", {"timezone": re.search(r"[A-Z][a-z]+/[A-Za-z_]+", prompt).group(0)}

    def _handler(self):
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _json(self, status: int, payload: Any) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self) -> None:
                fake.headers.append(dict(self.headers))
                if self.path.rstrip("/") == "/v1/models":
                    self._json(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in fake.models]})
                else:
                    self._json(404, {"error": {"message": "not found"}})

            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
                fake.headers.append(dict(self.headers))
                if self.path.rstrip("/") != "/v1/chat/completions":
                    self._json(404, {"error": {"message": "not found"}})
                    return
                n = fake._repeat_count(body)
                if fake.chat_status:
                    self._json(fake.chat_status, {"error": {"message": "EngineCore encountered an issue"}})
                    return
                if not body.get("tools"):
                    self._structured(body)
                    return
                name, args = fake._answer_call(body)
                corrupt = fake.corrupt == "always" or (fake.corrupt == "repeat" and n > 1)
                content = f"<tool_call>\n<function={name}>\n{json.dumps(args)}" if corrupt else None
                calls = [] if corrupt else [{"id": f"call_{n}", "type": "function",
                                             "function": {"name": name, "arguments": json.dumps(args)}}]
                if body.get("stream"):
                    self._stream(content, calls)
                else:
                    msg = {"role": "assistant", "content": content, **({"tool_calls": calls} if calls else {})}
                    self._json(200, {"id": "c1", "object": "chat.completion", "choices": [
                        {"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}]})

            def _structured(self, body: dict[str, Any]) -> None:
                kind = (body.get("response_format") or {}).get("type")
                if kind == "json_schema" and fake.json_schema == "reject":
                    self._json(400, {"error": {"message": "response_format json_schema is not supported"}})
                    return
                if kind == "json_schema" and fake.json_schema == "invalid":
                    content = "Sure! I would record this."
                else:
                    content = json.dumps({"decision": "record", "reason": "A lasting preference."})
                self._json(200, {"id": "c2", "object": "chat.completion", "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}]})

            def _stream(self, content: str | None, calls: list[dict[str, Any]]) -> None:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Connection", "close")
                self.end_headers()

                def chunk(delta: dict[str, Any], finish: str | None = None) -> None:
                    payload = {"id": "s1", "object": "chat.completion.chunk",
                               "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                    self.wfile.flush()

                chunk({"role": "assistant", "content": ""})
                if content:
                    for i in range(0, len(content), 7):
                        chunk({"content": content[i:i + 7]})
                for i, call in enumerate(calls):
                    name, args = call["function"]["name"], call["function"]["arguments"]
                    chunk({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                           "function": {"name": name[:2], "arguments": ""}}]})
                    chunk({"tool_calls": [{"index": i, "function": {"name": name[2:]}}]})
                    third = max(1, len(args) // 3)
                    for piece in (args[:third], args[third:2 * third], args[2 * third:]):
                        chunk({"tool_calls": [{"index": i, "function": {"arguments": piece}}]})
                chunk({}, "tool_calls" if calls else "stop")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                self.close_connection = True

        return Handler


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """urllib honours *_proxy variables; keep every request on 127.0.0.1. No API key from the caller's env."""
    for var in ("http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                *verify.API_KEY_ENV):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def _by_name(results: list[verify.CheckResult]) -> dict[str, verify.CheckResult]:
    return {r.name: r for r in results}


def _run(fake: FakeEndpoint, model: str = "fake-model", **kw: Any) -> list[verify.CheckResult]:
    return verify.run_checks(Endpoint(fake.base_url, model=model, timeout=10), **kw)


def _closed_port_url() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    return f"http://127.0.0.1:{port}/v1"


# -- checks -----------------------------------------------------------------------------------------

def test_all_checks_pass():
    with FakeEndpoint() as fake:
        results = _run(fake)
    got = _by_name(results)
    assert [r.name for r in results] == list(verify.CHECKS)
    assert all(r.status == OK for r in results), [(r.name, r.detail) for r in results]
    assert verify.exit_code(results) == 0
    assert got["tool_calls"].data["passed"] == 6 and got["tool_calls"].data["ttft_median_s"] is not None
    assert got["tool_corruption"].data == {**got["tool_corruption"].data, "passed": 5, "corrupted": False}
    streamed = [r for r in fake.requests if r.get("stream")]
    repeated = [r for r in fake.requests if r.get("tools") and not r.get("stream")]
    assert len(streamed) == 6 and all(len(r["tools"]) == 1 for r in streamed)
    assert len(repeated) == 5 and all(r == repeated[0] for r in repeated)
    assert 6 <= len(repeated[0]["tools"]) <= 8 and repeated[0]["temperature"] == 0
    assert all("chat_template_kwargs" not in r for r in fake.requests)   # only with --no-thinking
    assert fake.requests[-1]["response_format"]["type"] == "json_schema"


def test_model_not_listed():
    with FakeEndpoint(models=["served-a", "served-b"]) as fake:
        results = _run(fake, model="missing", skip=["tool_calls", "tool_corruption", "json_schema"])
    models = _by_name(results)["models"]
    assert models.status == FAIL and "served-a, served-b" in models.detail and "--model" in models.hint
    assert verify.exit_code(results) == 1


def test_no_model_given_uses_first_listed():
    with FakeEndpoint(models=["served-a", "served-b"]) as fake:
        ep = Endpoint(fake.base_url, model="", timeout=10)
        results = verify.run_checks(ep, skip=["tool_calls", "tool_corruption"])
    assert _by_name(results)["models"].status == OK and "served-a, served-b" in results[0].detail
    assert ep.model == "served-a" and fake.requests[-1]["model"] == "served-a"
    assert verify.exit_code(results) == 0


def test_markup_in_content_is_corruption():
    with FakeEndpoint(corrupt="always") as fake:
        results = _run(fake, skip=["json_schema"])
    got = _by_name(results)
    assert got["tool_calls"].status == FAIL and "markup in content" in got["tool_calls"].detail
    assert "--tool-call-parser" in got["tool_calls"].hint
    corr = got["tool_corruption"]
    assert corr.status == FAIL and corr.data["corrupted"] and corr.data["passed"] == 0
    assert "fresh request" in corr.hint
    assert verify.exit_code(results) == 1


def test_corruption_only_on_repeated_requests():
    with FakeEndpoint(corrupt="repeat") as fake:
        results = _run(fake, skip=["tool_calls", "json_schema"])
    corr = _by_name(results)["tool_corruption"]
    assert corr.status == FAIL and corr.data["corrupted"]
    assert [a["ok"] for a in corr.data["attempts"]] == [True, False, False, False, False]
    assert "#1:ok #2:FAIL" in corr.detail
    assert "--mtp-off" in corr.hint and "--no-prefix-caching" in corr.hint
    assert f'docs/field-notes.md, "{verify.NOTE_CORRUPTION}"' in corr.hint
    assert "fix (tool_corruption)" in verify.render(results, Endpoint(fake.base_url))


def test_json_schema_rejected_falls_back_to_json_object():
    with FakeEndpoint(json_schema="reject") as fake:
        results = _run(fake, skip=["tool_calls", "tool_corruption"])
    js = _by_name(results)["json_schema"]
    assert js.status == WARN and "json_object" in js.hint and verify.NOTE_STRUCTURED in js.hint
    assert js.data == {"structured": "json_object"}
    assert [r["response_format"]["type"] for r in fake.requests] == ["json_schema", "json_object"]
    assert verify.exit_code(results) == 0


def test_json_schema_ignored_fails():
    with FakeEndpoint(json_schema="invalid") as fake:
        results = _run(fake, skip=["tool_calls", "tool_corruption"])
    js = _by_name(results)["json_schema"]
    assert js.status == FAIL and "not JSON" in js.detail
    assert verify.exit_code(results) == 1


def test_server_errors_point_to_crash_notes():
    with FakeEndpoint(chat_status=500) as fake:
        results = _run(fake)
    got = _by_name(results)
    assert got["models"].status == OK
    for name in ("tool_calls", "tool_corruption", "json_schema"):
        assert got[name].status == FAIL and verify.NOTE_CRASH in got[name].hint, name
    assert verify.exit_code(results) == 1


class _Redirector:
    """Answers every request with a 302 to ``target`` and records whether it was hit."""

    def __init__(self, target: str) -> None:
        self.target = target
        self.hits = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args: Any) -> None:
                pass

            def _redirect(self) -> None:
                outer.hits += 1
                length = int(self.headers.get("Content-Length") or 0)
                if length:
                    self.rfile.read(length)
                self.send_response(302)
                self.send_header("Location", outer.target + self.path.removeprefix("/v1"))
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = do_POST = _redirect

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self._server.server_address[1]}/v1"

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


def test_redirects_are_not_followed():
    with FakeEndpoint() as target:
        redirector = _Redirector(target.base_url)
        try:
            ep = Endpoint(redirector.base_url, model="fake-model", api_key="secret-key", timeout=10)
            with pytest.raises(verify.RequestError) as exc:
                ep.get_json("/models")
            results = verify.run_checks(ep)
        finally:
            redirector.close()
    assert exc.value.status == 302 and "redirect" in str(exc.value) and "refused" in str(exc.value)
    assert redirector.hits >= 1
    assert target.headers == [] and target.requests == []   # the Authorization header never reached it
    got = _by_name(results)
    assert got["models"].status == FAIL and "redirect" in got["models"].detail
    assert verify.exit_code(results) == 1


def test_unreachable_endpoint():
    results = verify.run_checks(Endpoint(_closed_port_url(), model="m", timeout=5))
    assert results[0].unreachable and results[0].status == FAIL
    assert [r.status for r in results[1:]] == [SKIP] * 3
    assert verify.exit_code(results) == 2


def test_ttft_note_is_informational(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(verify, "TTFT_NOTE_S", 0.0)
    with FakeEndpoint() as fake:
        results = _run(fake, skip=["tool_corruption", "json_schema"])
    tc = _by_name(results)["tool_calls"]
    assert tc.status == OK and verify.NOTE_TTFT in tc.note
    assert "note (tool_calls)" in verify.render(results, Endpoint(fake.base_url))
    assert verify.exit_code(results) == 0


# -- stream parsing and judging ---------------------------------------------------------------------

def test_read_stream_reassembles_tool_call_deltas():
    lines = [b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"ad","arguments":""}}]}}]}\n',
             b"\n", b": keep-alive\n", b"\n",
             b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"name":"d","arguments":"{\\"a\\": 1,"}}]}}]}\n',
             b"\n",
             b'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":" \\"b\\": 2}"}}]},'
             b'"finish_reason":"tool_calls"}]}\n', b"\n", b"data: [DONE]\n", b"\n"]
    res = verify.read_stream(lines, 0.0)
    assert res.tool_calls == [{"name": "add", "arguments": '{"a": 1, "b": 2}'}]
    assert res.finish_reason == "tool_calls" and res.ttft_s is not None


@pytest.mark.parametrize("calls, content, finish, problem", [
    ([{"name": "add", "arguments": '{"a": 1, "b": 2}'}], None, "tool_calls", None),
    ([{"name": "add", "arguments": '{"a": 1, "b": '}], "", "length", "truncated"),
    ([{"name": "add", "arguments": '{"a": 1}'}], "", "tool_calls", "missing argument"),
    ([{"name": "sum", "arguments": '{"a": 1, "b": 2}'}], "", "tool_calls", "wrong function"),
    ([{"name": "add", "arguments": '{"a": 1, "b": "<|im_end|>"}'}], "", "tool_calls", "garbled"),
    ([], "[TOOL_CALLS] add", "stop", "markup"),
    ([], "The answer is 3.", "stop", "no structured tool call"),
])
def test_judge_call(calls, content, finish, problem):
    got = verify.judge_call(calls, content, finish, name="add", required=["a", "b"], allowed=["a", "b"])
    assert (got is None) if problem is None else (problem in got)


def test_validate_side_call():
    assert verify.validate_side_call('{"decision": "drop", "reason": "small talk"}') is None
    assert "not in record|drop" in verify.validate_side_call('{"decision": "maybe", "reason": "x"}')
    assert "missing" in verify.validate_side_call('{"decision": "drop"}')


# -- CLI --------------------------------------------------------------------------------------------

def test_cli_json_output_and_flags(capsys):
    with FakeEndpoint() as fake:
        rc = verify.main(["--base-url", fake.base_url, "--model", "fake-model", "--api-key", "k-test",
                          "--json", "--no-thinking", "--timeout", "10", "--skip", "tool_calls,json_schema"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["exit_code"] == 0 and out["model"] == "fake-model"
    assert out["base_url"] == fake.base_url
    status = {c["name"]: c["status"] for c in out["checks"]}
    assert status == {"models": OK, "tool_calls": SKIP, "tool_corruption": OK, "json_schema": SKIP}
    assert all(r["chat_template_kwargs"] == {"enable_thinking": False} for r in fake.requests)
    assert all(h.get("Authorization") == "Bearer k-test" for h in fake.headers)


@pytest.mark.parametrize("env, want", [
    ({}, "EMPTY"),
    ({"OPENAI_API_KEY": "k-openai"}, "k-openai"),
    ({"OPENAI_API_KEY": "k-openai", "VLLM_API_KEY": "k-vllm"}, "k-vllm"),
])
def test_cli_api_key_from_env(monkeypatch: pytest.MonkeyPatch, capsys, env, want):
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with FakeEndpoint() as fake:
        rc = verify.main(["--base-url", fake.base_url, "--skip", "tool_calls,tool_corruption,json_schema"])
    assert rc == 0 and "fake-model" in capsys.readouterr().out
    assert fake.headers[0]["Authorization"] == f"Bearer {want}"


def test_cli_repeated_skip_and_default_model(capsys):
    with FakeEndpoint(models=["primary"]) as fake:
        rc = verify.main(["--base-url", fake.base_url, "--skip", "tool_calls", "--skip", "tool_corruption"])
    out = capsys.readouterr().out
    assert rc == 0 and "model primary" in out and "result: ok" in out
    assert [r["model"] for r in fake.requests] == ["primary"]


def test_cli_unreachable_exit_code(capsys):
    rc = verify.main(["--base-url", _closed_port_url(), "--model", "m", "--timeout", "5"])
    out = capsys.readouterr().out
    assert rc == 2 and "unreachable" in out and "fix (models)" in out


def test_cli_rejects_unknown_check(capsys):
    with pytest.raises(SystemExit) as exc:
        verify.main(["--skip", "nope"])
    assert exc.value.code == 2 and "unknown check" in capsys.readouterr().err


def test_default_base_url():
    assert verify.build_parser().parse_args([]).base_url == "http://localhost:8000/v1"


# -- single-file script -----------------------------------------------------------------------------

def test_stdlib_only():
    tree = ast.parse((ROOT / "verify.py").read_text(encoding="utf-8"))
    mods = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    mods |= {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
    assert mods <= set(sys.stdlib_module_names) | {"__future__"}, mods - set(sys.stdlib_module_names)


def test_runs_as_a_script(tmp_path: Path):
    """Copied alone into an empty directory (as with curl), it runs with --help and against the fake."""
    script = tmp_path / "verify.py"
    script.write_text((ROOT / "verify.py").read_text(encoding="utf-8"), encoding="utf-8")
    env = {"PATH": "/usr/bin:/bin", "no_proxy": "127.0.0.1,localhost", "PYTHONDONTWRITEBYTECODE": "1"}
    helped = subprocess.run([sys.executable, str(script), "--help"], capture_output=True, text=True,
                            cwd=tmp_path, env=env, timeout=30)
    assert helped.returncode == 0 and "--base-url URL" in helped.stdout and "exit codes" in helped.stdout
    with FakeEndpoint() as fake:
        run = subprocess.run([sys.executable, str(script), "--base-url", fake.base_url, "--json"],
                             capture_output=True, text=True, cwd=tmp_path, env=env, timeout=60)
    assert run.returncode == 0, run.stdout + run.stderr
    assert {c["status"] for c in json.loads(run.stdout)["checks"]} == {OK}
