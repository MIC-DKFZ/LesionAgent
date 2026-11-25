"""The agent runtime: one tool list, one skill, one harness (the Claude Agent SDK), any model.

The agent is an in-process Claude Agent SDK session. It sees only the recist tools (no shell, no file access,
no code execution) and our own system prompt (host context + skill + references). Which model answers is decided
the way Claude Code decides it (`resolve_llm`):

  Claude       the Claude login of this machine, or an Anthropic API key (model alias, e.g. opus / sonnet)
  own server   any model behind an Anthropic-compatible /v1/messages endpoint, e.g. an open model on vLLM

The session emits AgentEvent objects (text, thinking, tool_call, tool_result, done, error) to a callback and
writes per-call token usage to <workdir>/usage.jsonl.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import config
from .tools import TOOLS, call_tool
from .workspace import RecistWorkspace

SKILL_DIR = Path(__file__).parent / "skill"
SERVER = "recist"
EFFORTS = ("low", "medium", "high", "xhigh", "max")
DEFAULT_MODEL, DEFAULT_EFFORT = "opus", "high"
DEFAULT_CONTEXT_TOKENS = 131072  # for a model on an own server; `max_context_tokens` in the settings file
_LIMIT_RE = re.compile(r"(hit your (session|usage|weekly) limit|usage limit|rate limit reached|failed to authenticate|oauth session expired)", re.I)
# Variables a user sets to configure Claude Code; everything else named CLAUDE_CODE_* belongs to a running session.
_KEEP_PREFIXES = ("CLAUDE_CODE_USE_", "CLAUDE_CODE_SKIP_")
_KEEP = ("CLAUDE_CODE_OAUTH_TOKEN", "CLAUDE_CODE_MAX_OUTPUT_TOKENS", "CLAUDE_CODE_MAX_CONTEXT_TOKENS",
         "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC")  # fmt: skip


class LLMError(RuntimeError):
    """The chosen LLM cannot be used; the message says why and what to do."""


@dataclass
class AgentEvent:
    kind: str  # text | thinking | tool_call | tool_result | done | error | info
    text: str = ""
    tool: str = ""
    args: dict = field(default_factory=dict)
    images: list = field(default_factory=list)
    is_error: bool = False
    cost_usd: float | None = None
    duration_s: float | None = None


@dataclass
class LLM:
    """The model that drives the agent, and the environment of the Claude Code process that talks to it."""

    model: str = DEFAULT_MODEL
    effort: str | None = DEFAULT_EFFORT  # None: the server takes no effort setting
    env: dict = field(default_factory=dict, repr=False)  # may hold a key
    label: str = "Claude"
    report_cost: bool = True  # False when Claude Code's prices do not apply (an own server)


# ---------------------------------------------------------------------------------------------- LLM
def resolve_llm(model: str | None = None, effort: str | None = None, url: str | None = None,
                api_key_file: str | Path | None = None, check: bool = True) -> LLM:  # fmt: skip
    """The LLM for a read. Arguments win over the `[llm]` table of the settings file (config.py).

    Without `url`: Claude, signed in the way Claude Code is (its login, `ANTHROPIC_API_KEY`, or the key in
    `api_key_file`). With `url`: the model `model` on that Anthropic-compatible server; `check` asks the server
    first whether it has the model and which efforts it takes. Raises LLMError."""
    c = config.get().llm
    model, effort, url = model or c.model, effort or c.effort or DEFAULT_EFFORT, url or c.url
    key_file = api_key_file or c.api_key_file
    if effort not in EFFORTS:
        raise LLMError(f"unknown effort {effort!r}; one of {', '.join(EFFORTS)}")
    if not url:
        key = _read_key(key_file, "")
        return LLM(model or DEFAULT_MODEL, effort, {"ANTHROPIC_API_KEY": key} if key else {},
                   "Claude (API key)" if key or os.environ.get("ANTHROPIC_API_KEY") else "Claude")  # fmt: skip
    base = url.rstrip("/").removesuffix("/v1")  # Claude Code appends /v1/messages
    key = _read_key(key_file, "LLM_API_KEY") or "none"
    if check:
        model, effort = _check_server(base, model, effort, key)
    elif not model:
        raise LLMError(f"give the model that {base} serves (--model, or `model` in the settings file)")
    return LLM(model, effort, _server_env(base, model, key, c.max_context_tokens or DEFAULT_CONTEXT_TOKENS),
               f"server {base.split('//')[-1]}", report_cost=False)  # fmt: skip


def _server_env(base: str, m: str, key: str, context_tokens: int) -> dict:
    """Claude Code pointed at an Anthropic-compatible /v1/messages endpoint: every model alias is `m`."""
    return {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key, "ANTHROPIC_API_KEY": key, "ANTHROPIC_MODEL": m,
            "ANTHROPIC_SMALL_FAST_MODEL": m, "ANTHROPIC_DEFAULT_HAIKU_MODEL": m, "ANTHROPIC_DEFAULT_SONNET_MODEL": m,
            "ANTHROPIC_DEFAULT_OPUS_MODEL": m, "CLAUDE_CODE_SUBAGENT_MODEL": m,
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "CLAUDE_CODE_MAX_CONTEXT_TOKENS": str(context_tokens),
            "API_TIMEOUT_MS": "1800000"}  # fmt: skip


def _check_server(base: str, model: str | None, effort: str, key: str) -> tuple[str, str | None]:
    """(model, effort) that `base` accepts: the model must be served, and the effort is the nearest one the server
    takes (Claude Code sends an effort, and servers accept different ones), or None if it takes none."""
    headers = {"x-api-key": key, "authorization": f"Bearer {key}", "anthropic-version": "2023-06-01"}
    status, body = _http(f"{base}/v1/models", headers)
    ids = [m.get("id") for m in (body.get("data") or []) if m.get("id")] if status == 200 and isinstance(body, dict) else []
    if model is None and len(ids) == 1:
        model = ids[0]
    if model is None or (ids and model not in ids):
        raise LLMError(f"choose a model of {base} with --model" + (f": {', '.join(ids)}" if ids else ""))
    takes = []
    for e in EFFORTS[:4]:
        payload = {"model": model, "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}],
                   "output_config": {"effort": e}}  # fmt: skip
        status, body = _http(f"{base}/v1/messages", headers, payload)
        if status == 200:
            takes.append(e)
        elif status in (404, 405):
            raise LLMError(f"{base} has no Anthropic-compatible /v1/messages endpoint")
        elif not any(w in str(body).lower() for w in ("effort", "output_config")):  # servers refuse with 400 or 500
            raise LLMError(f"{base}/v1/messages answered {status} for {model}: {str(body)[:300]}")
    if not takes:
        return model, None
    if effort not in takes:  # the nearest one, the lower one on a tie
        rank = EFFORTS.index(effort)
        effort = min(takes, key=lambda e: (abs(EFFORTS.index(e) - rank), EFFORTS.index(e)))
    return model, effort


def _read_key(path, var: str) -> str | None:
    key = Path(path).expanduser().read_text().strip() if path else (os.environ.get(var, "").strip() if var else "")
    return key or None


def _http(url: str, headers: dict, payload: dict | None = None, timeout: float = 120) -> tuple:
    """(status, parsed JSON or text); LLMError if the server cannot be reached."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data, {**headers, "content-type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            status, raw = r.status, r.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    except (urllib.error.URLError, OSError) as e:
        raise LLMError(f"cannot reach {url}: {getattr(e, 'reason', e)}") from e
    try:
        return status, json.loads(raw)
    except ValueError:
        return status, raw.decode(errors="replace")


# ---------------------------------------------------------------------------------------------- prompts
def skill_prompt() -> str:
    """SKILL.md body + its reference files, inlined (the agent has no file access)."""
    text = (SKILL_DIR / "SKILL.md").read_text()
    body = text.split("---", 2)[2] if text.startswith("---") else text
    refs = [f"\n\n# Reference: {p.name}\n\n{p.read_text()}" for p in sorted((SKILL_DIR / "reference").glob("*.md"))]
    return body + "".join(refs)


HOST_PROMPT = """
# Context

You are a RECIST 1.1 reading agent. The study (one patient, all visits) is already loaded into the workspace:
start with `study_info`. You drive the tools; the numbers (diameters, sums, response) are computed by code.
You have no shell, file or code access: everything goes through the tools. Work autonomously through all phases
(`recist_report` saves the result) and end with a short report: targets and why, per-visit response, confidence,
and what a radiologist should double-check.
"""


def _clean_env() -> None:
    """When a read is started from inside a Claude Code session, do not pass that session's variables on."""
    for k in list(os.environ):
        session = k.startswith("CLAUDE_CODE_") and not k.startswith(_KEEP_PREFIXES) and k not in _KEEP
        if session or k in ("CLAUDECODE", "CLAUDE_PID", "CLAUDE_EFFORT"):
            os.environ.pop(k, None)


# ---------------------------------------------------------------------------------------------- session
class AgentSession:
    """A Claude Agent SDK conversation bound to one workspace."""

    def __init__(self, ws: RecistWorkspace, on_event: Callable[[AgentEvent], None], llm: LLM | None = None):
        self.ws, self.on_event, self.llm = ws, on_event, llm or LLM()
        self.client = None
        self.total_cost = 0.0
        self.limit_hit = None

    def _server(self):
        from claude_agent_sdk import SdkMcpTool, ToolAnnotations, create_sdk_mcp_server

        def make(spec):
            async def handler(args: dict) -> dict:
                out = await asyncio.to_thread(call_tool, self.ws, spec.name, args)
                self.on_event(AgentEvent("tool_result", out.text, spec.name, args, out.images, out.is_error))
                content = [{"type": "text", "text": out.text}]
                content += [{"type": "image", "data": base64.b64encode(p).decode(), "mimeType": "image/png"}
                            for p in out.images]  # fmt: skip
                return {"content": content, "is_error": out.is_error}

            return SdkMcpTool(name=spec.name, description=spec.description, input_schema=spec.input_schema,
                              handler=handler, annotations=ToolAnnotations(readOnlyHint=spec.read_only))  # fmt: skip

        return create_sdk_mcp_server(SERVER, tools=[make(t) for t in TOOLS])

    def _options(self):
        from claude_agent_sdk import ClaudeAgentOptions

        _clean_env()
        env = {**self.llm.env, "MCP_TOOL_TIMEOUT": "1800000"}
        return ClaudeAgentOptions(
            mcp_servers={SERVER: self._server()}, strict_mcp_config=True, tools=[], allowed_tools=[f"mcp__{SERVER}__*"],
            permission_mode="dontAsk", setting_sources=[], system_prompt=self.system_prompt(),
            model=self.llm.model, effort=self.llm.effort, cwd=str(self.ws.workdir),
            max_buffer_size=64 * 1024 * 1024, env=env,
        )  # fmt: skip

    def system_prompt(self) -> str:
        notes = []
        if getattr(self.ws.clinician, "name", "none") != "none":
            notes.append("A clinician is available in this session (`ask_clinician`): confirm the target set with them "
                         "before `set_targets` (see the skill).")  # fmt: skip
        return HOST_PROMPT + skill_prompt() + ("\n\n# This run\n\n" + "\n".join(notes) + "\n" if notes else "")

    def _log_usage(self, rec: dict) -> None:
        try:
            with open(self.ws.workdir / "usage.jsonl", "a") as f:
                f.write(json.dumps({"time": round(time.time(), 2), **rec}, default=str) + "\n")
        except OSError:
            pass

    async def start(self):
        from claude_agent_sdk import ClaudeSDKClient

        self.client = ClaudeSDKClient(self._options())
        await self.client.connect()

    async def send(self, text: str) -> None:
        from claude_agent_sdk import AssistantMessage, ResultMessage, TextBlock, ThinkingBlock, ToolUseBlock

        if self.client is None:
            await self.start()
        t0 = time.time()
        await self.client.query(text)
        result, last_id = None, None
        async for msg in self.client.receive_response():  # ends with the ResultMessage of this turn
            if isinstance(msg, AssistantMessage):
                if msg.usage and msg.message_id != last_id:  # one API call can arrive as several messages
                    last_id = msg.message_id
                    self._log_usage({"kind": "call", "model": msg.model, "message_id": msg.message_id,
                                     "usage": msg.usage, "tools": [b.name.removeprefix(f"mcp__{SERVER}__")
                                                                   for b in msg.content if isinstance(b, ToolUseBlock)]})  # fmt: skip
                for b in msg.content:
                    if isinstance(b, TextBlock) and b.text.strip():
                        if _LIMIT_RE.search(b.text) and len(b.text) < 300:
                            self.limit_hit = b.text.strip()
                            self.on_event(AgentEvent("error", f"LLM usage limit: {b.text.strip()}", is_error=True))
                            continue
                        self.on_event(AgentEvent("text", b.text))
                    elif isinstance(b, ThinkingBlock) and getattr(b, "thinking", "").strip():
                        self.on_event(AgentEvent("thinking", b.thinking))
                    elif isinstance(b, ToolUseBlock):
                        self.on_event(AgentEvent("tool_call", tool=b.name.removeprefix(f"mcp__{SERVER}__"),
                                                 args=dict(b.input)))  # fmt: skip
            elif isinstance(msg, ResultMessage):
                result = msg
                self._log_usage({"kind": "result", "num_turns": msg.num_turns, "duration_ms": msg.duration_ms,
                                 "duration_api_ms": msg.duration_api_ms, "usage": msg.usage,
                                 "model_usage": msg.model_usage, "sdk_cost_usd": msg.total_cost_usd})  # fmt: skip
                if msg.total_cost_usd:
                    self.total_cost = msg.total_cost_usd
        if self.limit_hit:
            result = None
        cost = self.total_cost if self.llm.report_cost else None
        self.on_event(AgentEvent("done", "usage_limit" if self.limit_hit else (result.subtype if result else "ended"),
                                 is_error=bool(self.limit_hit) or bool(result and result.is_error),
                                 cost_usd=cost or None, duration_s=round(time.time() - t0, 1)))  # fmt: skip

    async def interrupt(self):
        if self.client is not None:
            await self.client.interrupt()

    async def close(self):
        if self.client is not None:
            await self.client.disconnect()
            self.client = None


def make_session(ws: RecistWorkspace, on_event, llm: LLM | None = None) -> AgentSession:
    """The agent session for `ws`; `llm` from `resolve_llm` (default: Claude, opus at high effort)."""
    return AgentSession(ws, on_event, llm or resolve_llm())
