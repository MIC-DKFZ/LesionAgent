"""Codex session via `codex app-server --stdio` dynamic tools (no MCP server, no open port, no extra deps).

Talks newline-delimited JSON-RPC 2.0 to `codex app-server --stdio` in an isolated temporary working directory
with `sandbox="read-only"`, `approvalPolicy="never"`, `ephemeral=True`, web search off, and built-in shell/file/
browser/subagent tools disabled (keeping code mode enabled). The 18 RECIST tools are registered as `dynamicTools`
on `thread/start` (`capabilities.experimentalApi=True`), instructions are sent as `developerInstructions`, and
rendered PNGs are returned inline as `inputImage` data URIs. Raw events and stderr are logged to `codex.jsonl`
and `codex.stderr.log` in the study workdir; token usage is written to `usage.jsonl`.
"""
from __future__ import annotations

import asyncio
import base64
import json
import shutil
import tempfile
import time
from importlib.metadata import version

from .runtime import AgentEvent, LLMError, ReadingSession, _LIMIT_RE
from .tools import TOOLS

_DISABLED_FEATURES = (
    "shell_tool", "unified_exec", "view_image", "apps", "plugins", "remote_plugin", "hooks",
    "multi_agent_v2", "image_generation", "browser_use", "in_app_browser", "computer_use",
    "goals", "tool_suggest", "skill_search", "sleep_tool",
)


def command() -> list[str]:
    exe = shutil.which("codex")
    if exe is None:
        raise LLMError("install the Codex CLI (`codex`) and sign in with `codex login` first")
    args = [
        exe, "app-server", "--stdio",
        "-c", 'web_search="disabled"',
        "-c", "agents.enabled=false",
        "-c", "skills.include_instructions=false",
        "-c", "tools.experimental_request_user_input.enabled=false",
    ]
    for feature in _DISABLED_FEATURES:
        args.extend(["--disable", feature])
    return args


class Session(ReadingSession):
    """Multi-turn Codex session over `codex app-server --stdio`."""

    def __init__(self, ws, on_event, llm=None):
        super().__init__(ws, on_event, llm)
        self._run_dir: tempfile.TemporaryDirectory | None = None
        self._stderr = None
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._tasks: set[asyncio.Task] = set()
        self._write_lock = asyncio.Lock()
        self._next_id = 1
        self._pending: dict[int, asyncio.Future] = {}
        self._turn_done: asyncio.Future | None = None
        self._turn_usage: dict | None = None
        self._thread_id: str | None = None
        self._turn_id: str | None = None

    async def start(self, *, with_tools: bool = True) -> None:
        if self._proc is not None:
            return
        self._run_dir = tempfile.TemporaryDirectory(prefix="lesionagent-codex-")
        log_dir = self.ws.workdir.resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        self._stderr = open(log_dir / "codex.stderr.log", "a")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *command(), cwd=self._run_dir.name, stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE, stderr=self._stderr, limit=64 * 1024 * 1024,
            )
            self._reader = asyncio.create_task(self._read_loop())
            await self._request("initialize", {
                "clientInfo": {"name": "lesionagent", "version": version("lesionagent")},
                "capabilities": {"experimentalApi": True},
            })
            await self._write({"method": "initialized"})
            params: dict = {
                "cwd": self._run_dir.name, "sandbox": "read-only", "approvalPolicy": "never",
                "ephemeral": True,
                "developerInstructions": self.system_prompt() if with_tools else "Answer in one word.",
            }
            if self.llm.model:
                params["model"] = self.llm.model
            if with_tools:
                params["dynamicTools"] = [
                    {"type": "function", "name": t.name, "description": t.description, "inputSchema": t.input_schema}
                    for t in TOOLS
                ]
            res = await self._request("thread/start", params)
            self._thread_id = res["thread"]["id"]
            self.observed_model = res.get("model") or self.observed_model
        except BaseException:
            await self.close()
            raise

    async def send(self, text: str) -> None:
        if self._turn_done is not None:
            raise LLMError("A Codex turn is already active")
        t0 = time.time()
        try:
            if self._proc is None:
                await self.start()
            self._turn_done = asyncio.get_running_loop().create_future()
            self._turn_usage = None
            params: dict = {"threadId": self._thread_id, "input": [{"type": "text", "text": text}]}
            if self.llm.effort:
                params["effort"] = self.llm.effort
            res = await self._request("turn/start", params)
            self._turn_id = res["turn"]["id"]
            status = await self._turn_done
        except LLMError as exc:
            self._record_error(exc, "Codex failed")
            raise
        finally:
            self._turn_id = None
            self._turn_done = None
        label = "usage_limit" if self.limit_hit else ("stopped" if status == "interrupted" else "success")
        self.on_event(AgentEvent("done", label, is_error=bool(self.limit_hit), duration_s=round(time.time() - t0, 1)))

    async def interrupt(self) -> None:
        if self._thread_id and self._turn_id and self._proc and self._proc.returncode is None:
            try:
                await asyncio.wait_for(
                    self._request("turn/interrupt", {"threadId": self._thread_id, "turnId": self._turn_id}),
                    timeout=5,
                )
            except Exception:  # noqa: BLE001
                pass

    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        await self._stop_process(self._proc)
        self._proc = None
        if self._reader is not None and not self._reader.done():
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
        self._reader = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None
        if self._run_dir is not None:
            self._run_dir.cleanup()
            self._run_dir = None

    async def _write(self, msg: dict) -> None:
        if self._proc is None or self._proc.stdin is None:
            raise LLMError("codex app-server is not running")
        data = (json.dumps(msg) + "\n").encode()
        async with self._write_lock:
            self._proc.stdin.write(data)
            await self._proc.stdin.drain()

    async def _request(self, method: str, params: dict) -> dict:
        req_id = self._next_id
        self._next_id += 1
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self._write({"id": req_id, "method": method, "params": params})
        return await fut

    def _fail_all(self, exc: Exception) -> None:
        for fut in list(self._pending.values()):
            if not fut.done():
                fut.set_exception(exc)
        self._pending.clear()
        if self._turn_done and not self._turn_done.done():
            self._turn_done.set_exception(exc)

    async def _read_loop(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        log_path = self.ws.workdir.resolve() / "codex.jsonl"
        try:
            async for raw in self._proc.stdout:
                if not raw.strip():
                    continue
                try:
                    msg = json.loads(raw)
                except ValueError as exc:
                    self._fail_all(LLMError(f"codex app-server emitted invalid JSON: {exc}"))
                    return
                with open(log_path, "a") as f:
                    f.write(json.dumps(msg, default=str) + "\n")
                if "id" in msg and "method" not in msg:
                    fut = self._pending.pop(msg["id"], None)
                    if fut and not fut.done():
                        if msg.get("error") is not None:
                            fut.set_exception(LLMError(f"codex app-server RPC error: {msg['error'].get('message')}"))
                        else:
                            fut.set_result(msg.get("result") or {})
                elif "id" in msg and "method" in msg:
                    task = asyncio.create_task(self._handle_request(msg))
                    self._tasks.add(task)
                    task.add_done_callback(self._tasks.discard)
                elif "method" in msg:
                    self._handle_notification(msg["method"], msg.get("params") or {})
            rc = await self._proc.wait()
            if self._pending or (self._turn_done and not self._turn_done.done()):
                self._fail_all(LLMError(f"codex app-server exited unexpectedly (status {rc})"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._fail_all(LLMError(f"codex app-server reader failed: {exc}"))

    async def _handle_request(self, msg: dict) -> None:
        req_id, method, params = msg["id"], msg["method"], msg.get("params") or {}
        if method != "item/tool/call":
            await self._write({"id": req_id, "error": {"code": -32601, "message": f"unsupported: {method}"}})
            return
        args = params.get("arguments") if isinstance(params.get("arguments"), dict) else {}
        out = await self._tool(params.get("tool", ""), args)
        content: list[dict] = [{"type": "inputText", "text": out.text}]
        content.extend(
            {"type": "inputImage", "imageUrl": f"data:image/png;base64,{base64.b64encode(p).decode('ascii')}"}
            for p in out.images
        )
        await self._write({"id": req_id, "result": {"contentItems": content, "success": not out.is_error}})

    def _handle_notification(self, method: str, params: dict) -> None:
        if method == "item/completed":
            item = params.get("item") or {}
            if item.get("type") == "agentMessage" and (text := (item.get("text") or "").strip()):
                if _LIMIT_RE.search(text) and len(text) < 300:
                    self.limit_hit = text
                    self.on_event(AgentEvent("error", f"LLM usage limit: {text}", is_error=True))
                else:
                    self.on_event(AgentEvent("text", item["text"]))
            elif item.get("type") == "reasoning" and (summary := "\n".join(item.get("summary") or []).strip()):
                self.on_event(AgentEvent("thinking", summary))
        elif method == "thread/tokenUsage/updated" and params.get("tokenUsage"):
            self._turn_usage = params["tokenUsage"]
        elif method == "turn/completed":
            turn = params["turn"]
            status = turn["status"]
            if self._turn_usage:
                self._log_usage({"kind": "result", "usage": self._turn_usage})
            if self._turn_done and not self._turn_done.done():
                if status in ("completed", "interrupted"):
                    self._turn_done.set_result(status)
                else:
                    detail = (turn.get("error") or {}).get("message") or status
                    if _LIMIT_RE.search(detail):
                        self.limit_hit = detail
                    self._turn_done.set_exception(LLMError(f"Codex turn failed: {detail}"))
