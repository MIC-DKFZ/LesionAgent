"""Antigravity CLI (`agy`) session with a zero-dependency, token-protected loopback MCP bridge.

Runs `agy --input-format stream-json --output-format stream-json` in an isolated temporary working directory,
connected to a minimal stdlib-only Streamable-HTTP MCP server bound to `127.0.0.1:0` and guarded by a per-session
32-byte Bearer token. Because `agy` offloads MCP image results to `~/.gemini/antigravity-cli/brain/<id>/...` and
opens them with `read_file` (one extra step per image), shell/write/URL tools are denied in `--settings` and a
`PreToolUse` hook in `.agents/hooks.json` restricts `read_file` strictly to `.png` files under this conversation's
`.system_generated/steps/` folder (denying all other paths). `close()` validates the conversation id and removes
its `brain/`, `conversations/<id>.db`, and `presence/<id>.lock` files (also after errors or interrupts). Raw
events and stderr are logged to `agy.jsonl` and `agy.stderr.log`; token usage is written to `usage.jsonl`.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import secrets
import shutil
import sys
import tempfile
import time
from importlib.metadata import version
from pathlib import Path

from .runtime import SERVER, AgentEvent, LLMError, ReadingSession
from .tools import TOOLS

TOOL_TIMEOUT_SECONDS = 1800
_DENIED_PERMISSIONS = ["command(*)", "write_file(*)", "read_url(*)", "execute_url(*)"]
_AGY_DATA_DIR = Path.home() / ".gemini" / "antigravity-cli"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_READ_GUARD_SCRIPT = """import json, pathlib, re, sys
data = json.load(sys.stdin)
cid = data.get("conversationId") or ""
raw = ((data.get("toolCall") or {}).get("args") or {}).get("Path") or ""
ok = False
if re.fullmatch(r"[A-Za-z0-9_-]+", cid) and raw:
    base = (pathlib.Path(sys.argv[1]) / "brain" / cid / ".system_generated" / "steps").resolve()
    target = pathlib.Path(raw).resolve(strict=False)
    ok = target.is_relative_to(base) and target.suffix == ".png"
msg = "Only offloaded MCP images for this session may be read."
print(json.dumps({"decision": "allow"} if ok else {"decision": "deny", "reason": msg}))
"""


def command(llm, url: str | None = None) -> list[str]:
    exe = shutil.which("agy")
    if exe is None:
        raise LLMError("install the Antigravity CLI (`agy`) and sign in first")
    settings = json.dumps({"permissions": {"allow": [f"mcp({SERVER}/*)"] if url else [], "deny": _DENIED_PERMISSIONS}})
    args = [exe, "--input-format", "stream-json", "--output-format", "stream-json",
            "--disable-slash-commands", "--settings", settings]
    if llm.model:
        args += ["--model", llm.model]
    if llm.effort:
        args += ["--effort", llm.effort]
    return args


class _LoopbackMcpBridge:
    """Minimal stateless Streamable-HTTP MCP server on 127.0.0.1 using only the Python stdlib."""

    def __init__(self, invoke):
        self._invoke = invoke
        self.token = secrets.token_urlsafe(32)
        self._auth = f"Bearer {self.token}".encode()
        self._server: asyncio.AbstractServer | None = None
        self.url: str = ""

    async def start(self) -> tuple[str, str]:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.url = f"http://127.0.0.1:{self._server.sockets[0].getsockname()[1]}/mcp"
        return self.url, self.token

    async def close(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    @staticmethod
    async def _reply(writer: asyncio.StreamWriter, status: int, payload: dict | None = None) -> None:
        reasons = {200: "OK", 202: "Accepted", 401: "Unauthorized", 405: "Method Not Allowed"}
        body = json.dumps(payload).encode() if payload is not None else b""
        head = (f"HTTP/1.1 {status} {reasons[status]}\r\nContent-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n").encode()
        writer.write(head + body)
        await writer.drain()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            lines = head.decode("latin-1").split("\r\n")
            method, path = lines[0].split(" ", 2)[:2]
            headers = {k.strip().lower(): v.strip() for l in lines[1:] if ":" in l for k, v in [l.split(":", 1)]}
            if path.split("?", 1)[0] != "/mcp" or not secrets.compare_digest(
                headers.get("authorization", "").encode(), self._auth
            ):
                return await self._reply(writer, 401)
            if method != "POST":
                return await self._reply(writer, 405)
            length = int(headers.get("content-length", "0"))
            req = json.loads(await reader.readexactly(length) if length > 0 else b"{}")
            resp = await self._dispatch(req)
            await self._reply(writer, 202 if resp is None else 200, resp)
        except Exception:  # noqa: BLE001
            pass
        finally:
            writer.close()

    async def _dispatch(self, req: dict) -> dict | None:
        if "id" not in req:
            return None
        req_id, method, params = req["id"], req.get("method", ""), req.get("params") or {}
        if method == "initialize":
            return {"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": params.get("protocolVersion") or "2025-03-26",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "lesionagent", "version": version("lesionagent")},
            }}
        if method == "tools/list":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": [
                {"name": t.name, "description": t.description, "inputSchema": t.input_schema,
                 "annotations": {"readOnlyHint": t.read_only}} for t in TOOLS
            ]}}
        if method == "tools/call":
            out = await self._invoke(params.get("name", ""), params.get("arguments") or {})
            content: list[dict] = [{"type": "text", "text": out.text}]
            content.extend(
                {"type": "image", "data": base64.b64encode(p).decode("ascii"), "mimeType": "image/png"}
                for p in out.images
            )
            return {"jsonrpc": "2.0", "id": req_id, "result": {"content": content, "isError": out.is_error}}
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"method not found: {method}"}}


class Session(ReadingSession):
    """Multi-turn Antigravity CLI session backed by a zero-dependency loopback MCP bridge."""

    def __init__(self, ws, on_event, llm=None):
        super().__init__(ws, on_event, llm)
        self._bridge: _LoopbackMcpBridge | None = None
        self._run_dir: tempfile.TemporaryDirectory | None = None
        self._stderr = None
        self._proc: asyncio.subprocess.Process | None = None
        self._with_tools = True
        self._first_turn = True
        self._interrupted = False
        self._conv_ids: set[str] = set()

    async def start(self, *, with_tools: bool = True) -> None:
        if self._proc is not None:
            return
        self._with_tools = with_tools
        self._first_turn = True
        self._interrupted = False
        log_dir = self.ws.workdir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        if self._run_dir is None:
            self._run_dir = tempfile.TemporaryDirectory(prefix="lesionagent-agy-")
            cfg_dir = Path(self._run_dir.name) / ".agents"
            cfg_dir.mkdir()
            guard = cfg_dir / "guard_read.py"
            guard.write_text(_READ_GUARD_SCRIPT)
            (cfg_dir / "hooks.json").write_text(json.dumps({
                "read-guard": {"PreToolUse": [{
                    "matcher": "read_file|view_file",
                    "hooks": [{"type": "command", "command": f"{sys.executable} {guard} {_AGY_DATA_DIR}",
                               "timeout": 10}],
                }]}
            }))
        url: str | None = None
        if with_tools:
            if self._bridge is None:
                self._bridge = _LoopbackMcpBridge(self._tool)
                await self._bridge.start()
            url = self._bridge.url
            cfg = {"mcpServers": {SERVER: {
                "serverUrl": url, "headers": {"Authorization": f"Bearer {self._bridge.token}"},
                "timeoutSeconds": TOOL_TIMEOUT_SECONDS, "tools": {t.name: {"eager": True} for t in TOOLS},
            }}}
            (Path(self._run_dir.name) / ".agents" / "mcp_config.json").write_text(json.dumps(cfg))
        if self._stderr is None:
            self._stderr = open(log_dir / "agy.stderr.log", "a")
        self._proc = await asyncio.create_subprocess_exec(
            *command(self.llm, url), cwd=self._run_dir.name, stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE, stderr=self._stderr, limit=64 * 1024 * 1024,
        )

    async def send(self, text: str) -> None:
        t0 = time.time()
        try:
            if self._proc is None:
                await self.start(with_tools=self._with_tools)
            assert self._proc is not None and self._proc.stdin is not None and self._proc.stdout is not None
            content = (
                f"{self.system_prompt()}\n\nUse only the {SERVER} MCP tools for this read.\n\n{text}"
                if self._first_turn and self._with_tools else text
            )
            self._first_turn = False
            self._proc.stdin.write((json.dumps({"event": "user", "message": {"content": content}}) + "\n").encode())
            await self._proc.stdin.drain()
            stopped = await self._consume_turn(self._proc.stdout)
            if stopped:
                proc, self._proc = self._proc, None
                await self._stop_process(proc)
        except LLMError as exc:
            self._record_error(exc, "Antigravity failed")
            raise
        self.on_event(AgentEvent("done", "stopped" if stopped else "success", duration_s=round(time.time() - t0, 1)))

    async def _consume_turn(self, stdout: asyncio.StreamReader) -> bool:
        log_path = self.ws.workdir.resolve() / "agy.jsonl"
        emitted_text, chunks = False, {}
        async for raw in stdout:
            if not raw.strip():
                continue
            try:
                event = json.loads(raw)
            except ValueError as e:
                raise LLMError(f"Antigravity emitted invalid JSON; see agy.stderr.log: {e}") from e
            with open(log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
            if cid := event.get("conversation_id"):
                self._conv_ids.add(cid)
            kind = event.get("event")
            if kind == "step_update":
                su = event.get("step_update") or {}
                if su.get("step_type") == "agent_response":
                    idx = su.get("step_index", -1)
                    if su.get("text_delta"):
                        chunks.setdefault(idx, []).append(su["text_delta"])
                    if su.get("state") == "DONE":
                        text = "".join(chunks.pop(idx, []))
                        if text.strip():
                            self.on_event(AgentEvent("text", text))
                            emitted_text = True
                        if su.get("usage"):
                            self._log_usage({"kind": "call", "step_index": idx,
                                             "duration_s": su.get("duration_seconds"), "usage": su["usage"]})
            elif kind == "result":
                res = event.get("result") or {}
                if res.get("usage") or res.get("num_turns") is not None:
                    self._log_usage({"kind": "result", "status": res.get("status"), "num_turns": res.get("num_turns"),
                                     "duration_s": res.get("duration_seconds"), "usage": res.get("usage")})
                if res.get("status") != "SUCCESS":
                    if self._interrupted:
                        return True
                    raise LLMError(f"Antigravity failed ({res.get('status')}): {res.get('error') or res}")
                if not emitted_text and (resp := (res.get("response") or "")).strip():
                    self.on_event(AgentEvent("text", resp))
                return False
        if self._interrupted:
            return True
        raise LLMError("Antigravity ended without completing a turn; see agy.jsonl")

    async def interrupt(self) -> None:
        self._interrupted = True
        proc, self._proc = self._proc, None
        await self._stop_process(proc)

    def _cleanup_agy_data(self) -> None:
        specs = (
            (_AGY_DATA_DIR / "brain", "", lambda p: shutil.rmtree(p, ignore_errors=True)),
            (_AGY_DATA_DIR / "conversations", ".db", lambda p: p.unlink(missing_ok=True)),
            (_AGY_DATA_DIR / "presence", ".lock", lambda p: p.unlink(missing_ok=True)),
        )
        for cid in list(self._conv_ids):
            if not _SAFE_ID.fullmatch(cid):
                continue
            for root_path, suffix, rm in specs:
                root = root_path.resolve()
                target = (root / f"{cid}{suffix}").resolve()
                if target.is_relative_to(root) and target != root:
                    rm(target)
        self._conv_ids.clear()

    async def close(self) -> None:
        try:
            await self.interrupt()
        finally:
            if self._stderr is not None:
                self._stderr.close()
                self._stderr = None
            if self._run_dir is not None:
                self._run_dir.cleanup()
                self._run_dir = None
            if self._bridge is not None:
                await self._bridge.close()
                self._bridge = None
            self._cleanup_agy_data()
