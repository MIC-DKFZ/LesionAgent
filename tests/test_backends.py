"""CPU integration tests for backend selection, Codex app-server, and the Antigravity MCP bridge."""
import asyncio
import base64
import json
import sys
import urllib.error
import urllib.request

import pytest

from lesionagent import config
from lesionagent.runtime import LLM, LLMError, make_session, resolve_llm


def test_backend_selection(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_ACTIVE", config.Config())
    monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")

    assert resolve_llm(check=False).backend == "claude"
    assert resolve_llm(backend="codex").label == "Codex"
    assert resolve_llm(backend="agy").label == "Antigravity"

    (tmp_path / "lesionagent.toml").write_text('[llm]\nbackend = "agy"\n')
    config.load()
    assert resolve_llm().backend == "agy"

    for backend in ("codex", "agy"):
        with pytest.raises(LLMError):
            resolve_llm(backend=backend, url="http://localhost:8000")
        with pytest.raises(LLMError):
            resolve_llm(backend=backend, api_key_file=tmp_path / "key")
    with pytest.raises(LLMError):
        resolve_llm(backend="other")


def test_codex_turn_with_image(study, tmp_path, monkeypatch):
    from lesionagent.workspace import RecistWorkspace

    ws = RecistWorkspace(study, tmp_path / "read")
    script = tmp_path / "fake_codex.py"
    script.write_text("""
import base64, json, sys
send = lambda o: (sys.stdout.write(json.dumps(o) + "\\n"), sys.stdout.flush())
for line in sys.stdin:
    msg = json.loads(line)
    m = msg.get("method")
    if m == "initialize":
        send({"id": msg["id"], "result": {}})
    elif m == "thread/start":
        send({"id": msg["id"], "result": {"thread": {"id": "t1"}, "model": "gpt-test"}})
    elif m == "turn/start":
        send({"id": msg["id"], "result": {"turn": {"id": "u1"}}})
        send({"id": "c1", "method": "item/tool/call", "params": {
            "tool": "view", "arguments": {"visit": 1, "center": [6, 24, 24]}}})
        res = json.loads(sys.stdin.readline())["result"]
        imgs = [c for c in res["contentItems"] if c["type"] == "inputImage"]
        assert res["success"] and base64.b64decode(imgs[0]["imageUrl"].split(",", 1)[1]).startswith(b"\\x89PNG")
        send({"method": "item/completed", "params": {"item": {"type": "agentMessage", "text": "done"}}})
        send({"method": "turn/completed", "params": {"turn": {"id": "u1", "status": "completed"}}})
""")
    monkeypatch.setattr("lesionagent.codex.command", lambda: [sys.executable, str(script)])
    events = []
    session = make_session(ws, events.append, LLM(model=None, effort=None, backend="codex"))

    async def run():
        try:
            await session.send("inspect")
        finally:
            await session.close()

    asyncio.run(run())
    assert [e.kind for e in events] == ["tool_call", "tool_result", "text", "done"]
    assert session.observed_model == "gpt-test"


def test_agy_bridge_auth_and_image(study, tmp_path):
    from lesionagent.agy import _LoopbackMcpBridge
    from lesionagent.tools import TOOLS, call_tool
    from lesionagent.workspace import RecistWorkspace

    ws = RecistWorkspace(study, tmp_path / "read")

    def post(url, body, token=None):
        headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {token}"} if token else {})}
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, None

    async def run():
        bridge = _LoopbackMcpBridge(lambda name, args: asyncio.to_thread(call_tool, ws, name, args))
        url, token = await bridge.start()
        try:
            assert (await asyncio.to_thread(post, url, {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}))[0] == 401
            st, listed = await asyncio.to_thread(post, url, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, token)
            assert st == 200 and {t["name"] for t in listed["result"]["tools"]} == {t.name for t in TOOLS}
            call = {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                    "params": {"name": "view", "arguments": {"visit": 1, "center": [6, 24, 24]}}}
            st, view = await asyncio.to_thread(post, url, call, token)
            imgs = [c for c in view["result"]["content"] if c["type"] == "image"]
            assert st == 200 and not view["result"]["isError"]
            assert base64.b64decode(imgs[0]["data"]).startswith(b"\x89PNG")
        finally:
            await bridge.close()

    asyncio.run(run())
