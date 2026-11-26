"""`lesionagent setup`: check the GPU, download and load every model once, send one test message to the LLM."""

from __future__ import annotations

import asyncio
from importlib.metadata import version

NC = "CC BY-NC-SA 4.0 (non-commercial)"
TS_TASKS = {"total (fast)": [297, 298], "trunk_cavities": [343], "lung_nodules": [913], "liver_vessels": [8]}


def _step(label: str, fn) -> bool:
    """Run one check and print its result on one line (the model libraries print their own progress in between)."""
    print(f"  {label} ...", flush=True)
    try:
        note = fn()
    except Exception as e:  # noqa: BLE001
        print(f"  {label}: FAILED\n    {type(e).__name__}: {e}", flush=True)
        return False
    print(f"  {label}: ok" + (f" ({note})" if note else ""), flush=True)
    return True


def _gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch finds no CUDA device; install a PyTorch build that matches your NVIDIA driver")
    gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    note = f"PyTorch {torch.__version__}, {torch.cuda.get_device_name(0)}, {gb:.0f} GB"
    return note + ("; less than 24 GB: large scans may not fit" if gb < 23 else "")


def _load(name: str, key: str | None = None):
    def fn():
        from .backends import models

        m = models.get(name)
        if name == "registration":
            m.warm_up()
        note = models.VERSIONS.get(key or name, "")
        models.release()
        return note

    return fn


def _totalsegmentator():
    from totalsegmentator.libs import download_pretrained_weights

    for ids in TS_TASKS.values():
        for i in ids:
            download_pretrained_weights(i)
    return ", ".join(TS_TASKS)


def _ask_llm(llm):
    """One message without tools through the same harness as a read."""
    from .runtime import BACKENDS, LLMError, make_session

    if llm.backend in BACKENDS:
        async def go_backend():
            import tempfile
            from pathlib import Path
            from types import SimpleNamespace

            replies: list[str] = []
            with tempfile.TemporaryDirectory(prefix="lesionagent-probe-") as tmp:
                ws = SimpleNamespace(workdir=Path(tmp), clinician=None)
                on_ev = lambda ev: replies.append(ev.text.strip()) if ev.kind == "text" and ev.text.strip() else None
                session = make_session(ws, on_ev, llm)
                try:
                    await session.start(with_tools=False)
                    await session.send("Reply with the word ready.")
                finally:
                    await session.close()
            if not replies:
                raise LLMError(f"{llm.label} returned no answer to the setup probe")
            return f"answered {replies[-1][:40]!r}"

        return asyncio.run(go_backend())
    from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query

    from .runtime import _clean_env

    async def go():
        _clean_env()
        opts = ClaudeAgentOptions(tools=[], strict_mcp_config=True, setting_sources=[], permission_mode="dontAsk",
                                  max_turns=1, model=llm.model, effort=llm.effort, env=dict(llm.env),
                                  system_prompt="Answer in one word.")  # fmt: skip
        text, result = "", None
        async for msg in query(prompt="Reply with the word ready.", options=opts):
            if isinstance(msg, AssistantMessage):
                text += "".join(b.text for b in msg.content if isinstance(b, TextBlock))
            elif isinstance(msg, ResultMessage):
                result = msg
        if result is None or result.is_error or not text.strip():
            raise RuntimeError(f"no answer from the LLM: {text.strip() or getattr(result, 'result', '') or 'empty reply'}")
        return f"answered {text.strip()[:40]!r}"

    return asyncio.run(go())


def setup(get_llm, models: bool = True, llm: bool = True) -> int:
    """Run the checks; `get_llm()` returns the LLM of the command line and settings file. 0 if all passed."""
    print(f"LesionAgent {version('lesionagent')} setup\n\nGPU")
    ok = _step("PyTorch with CUDA", _gpu)
    if models and ok:
        print("\nModels (downloaded on first use)")
        ok &= _step("VoxTell", _load("voxtell"))
        ok &= _step("nnInteractive", _load("nninteractive"))
        ok &= _step("LongiTrack", _load("longitrack"))
        ok &= _step("uniGradICON", _load("registration"))
        ok &= _step("TotalSegmentator", _totalsegmentator)
        print(f"  The weights of VoxTell, nnInteractive and LongiTrack are licensed under {NC}.")
    if llm:
        print("\nLLM")
        box = {}

        def resolve():
            box["llm"] = get_llm()
            return f"{box['llm'].label}, model {box['llm'].model or 'default'}, effort {box['llm'].effort or 'default'}"

        if _step("settings", resolve):
            ok &= _step("test message", lambda: _ask_llm(box["llm"]))
        else:
            ok = False
    print("\nready" if ok else "\nnot ready: see the failed step(s) above")
    return 0 if ok else 1
