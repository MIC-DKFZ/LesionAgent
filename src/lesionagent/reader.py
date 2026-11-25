"""One read: the agent assesses a study from the images to the RECIST 1.1 report.

    from lesionagent import Study, read

    study = Study.from_files(["bl.nii.gz", "fu1.nii.gz"], days=[0, 63])
    result = read(study, out="runs/patient1")
    print(result.report["best_overall_response"])

The output folder holds the report (report.md, report.json), the lesion registry and one label map per visit
(result/), every image the agent saw (renders/), the agent's messages (transcript.jsonl), every tool call
(log.jsonl), the token usage (usage.jsonl) and the settings of the read (run.json).
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

from . import config as settings
from . import tasks
from .clinician import Clinician
from .core.study import Study
from .runtime import LLM, AgentEvent, make_session, resolve_llm
from .workspace import RecistWorkspace


@dataclass
class Read:
    """The result of `read`."""

    out: Path  # the output folder
    report: dict | None  # the RECIST report (report.json); None if the agent did not get to one
    summary: str  # the agent's closing message
    run: dict  # run.json: LLM, model versions, duration, number of tool calls
    workspace: RecistWorkspace  # the registry and every tool, for a look at the lesions or a correction


def task_text(study: Study, task: str | None = None) -> str:
    """The task the agent is given: the full RECIST 1.1 read (default), a built-in task (see tasks/), the content of
    a text file, or `task` itself."""
    if task is None:
        days = all(v.day is not None for v in study.visits)
        visits = ", ".join(f"{v.name} day {v.day}" if days else v.name for v in study.visits)
        return f"Perform a complete RECIST 1.1 assessment of this patient ({len(study.visits)} visits: {visits})."
    if task in tasks.names():
        if len(study.visits) != 1:
            raise ValueError(f"the task '{task}' reads one scan, but {len(study.visits)} images were given")
        return tasks.text(task)
    try:
        if Path(task).expanduser().is_file():
            return Path(task).expanduser().read_text().strip()
    except (OSError, ValueError):  # a long text is not a file name
        pass
    return task


def _short(d: dict, n=160) -> str:
    s = json.dumps(d, default=str)
    return s if len(s) <= n else s[:n] + "..."


def _require_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("LesionAgent needs an NVIDIA GPU (about 24 GB): PyTorch finds no CUDA device")


def read(study: Study, out=None, *, task: str | None = None, llm: LLM | None = None,
         clinician: Clinician | None = None, config=None, log=print) -> Read:  # fmt: skip
    """Let the agent read `study` and write everything to `out` (default: ./runs/<time>). Blocks until it is done.

    `task`: a built-in task ("lesions", "lung-nodules"), a text file or the text itself; default: the full RECIST 1.1
    read. `llm`: from `resolve_llm(model=..., effort=..., url=...)`; default: Claude (opus, high effort) or what the
    settings file says. `clinician`: an `InteractiveClinician` that answers the agent's questions; without one the
    agent decides alone. `config`: a settings file (see config.py). `log`: receives the readable log line by line."""
    if config is not None:
        settings.load(config)
    _require_gpu()
    out = Path(out) if out else Path.cwd() / "runs" / time.strftime("%Y%m%d_%H%M%S")
    if (out / "transcript.jsonl").exists():
        raise FileExistsError(f"{out} already holds a read: choose another output folder")
    llm = llm or resolve_llm()
    text = task_text(study, task)
    ws = RecistWorkspace(study, out, clinician=clinician)
    tr = open(out / "transcript.jsonl", "a")
    t0 = time.time()
    last = {"text": "", "cost": None, "finished": False}

    def on_event(ev: AgentEvent):
        rec = {"t": round(time.time() - t0, 1), "kind": ev.kind, "tool": ev.tool, "args": ev.args, "text": ev.text,
               "images": len(ev.images), "error": ev.is_error, "cost": ev.cost_usd}  # fmt: skip
        tr.write(json.dumps(rec, default=str) + "\n")
        tr.flush()
        pre = f"[{rec['t']:7.1f}s]"
        if ev.kind == "text":
            last["text"] = ev.text.strip()
            log(f"{pre} AGENT: {ev.text.strip()}")
        elif ev.kind == "tool_call":
            log(f"{pre} -> {ev.tool} {_short(ev.args)}")
        elif ev.kind == "tool_result":
            first = ev.text.strip().splitlines()[0][:200] if ev.text.strip() else ""
            log(f"{pre} <- {ev.tool}{' ERROR' if ev.is_error else ''}: {first}"
                + (f" [{len(ev.images)} image(s)]" if ev.images else ""))  # fmt: skip
        elif ev.kind in ("done", "error"):
            last["cost"] = ev.cost_usd or last["cost"]
            last["finished"] = ev.kind == "done" and not ev.is_error
            log(f"{pre} {ev.kind.upper()}: {ev.text[:300]}" + (f" cost ${ev.cost_usd:.2f}" if ev.cost_usd else ""))

    log(f"output {out}\nLLM {llm.label}: model {llm.model}, effort {llm.effort or 'server default'}\nTASK: {text}\n")
    (out / "task.txt").write_text(text)
    sessions = []

    async def go():
        s = make_session(ws, on_event, llm)
        sessions.append(s)
        try:
            await s.send(text)
        finally:
            await s.close()

    run: dict = {}
    try:
        asyncio.run(go())
    finally:
        from .backends import models

        try:
            ws.save()
        except Exception as e:  # noqa: BLE001
            log(f"save failed: {e}")
        run = {"lesionagent": version("lesionagent"), "images": [os.path.abspath(v.path) for v in study.visits],
               "days": [v.day for v in study.visits], "llm": llm.label, "model": llm.model, "effort": llm.effort,
               "models": dict(models.VERSIONS), "seconds": round(time.time() - t0, 1),
               "tool_calls": sum(1 for _ in open(out / "log.jsonl")) if (out / "log.jsonl").exists() else 0,
               "model_calls": ws.n_model_calls, "questions": len(ws.questions),
               "cost_usd": None if last["cost"] is None else round(last["cost"], 2),
               "finished": last["finished"], "usage_limit": bool(sessions and sessions[0].limit_hit)}  # fmt: skip
        (out / "run.json").write_text(json.dumps(run, indent=2))
        (out / "summary.md").write_text(last["text"] + "\n")
        log("\nRUN " + json.dumps(run))
        tr.close()
        ws.close()
    rep = out / "report.json"
    return Read(out, json.loads(rep.read_text()) if rep.exists() else None, last["text"], run, ws)
