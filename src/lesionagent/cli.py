"""The lesionagent command line.

  lesionagent run --images bl.nii.gz fu1.nii.gz fu2.nii.gz --days 0 63 126 [--out DIR]
                  [--model opus] [--effort high] [--llm-url URL] [--api-key-file FILE] [--config FILE]
  lesionagent run --images ct.nii.gz --task lesions
  lesionagent setup [--skip-models] [--skip-llm]

`run` reads one patient: CT volumes in chronological order, baseline first. It prints a readable log and writes
the report, the lesion registry with one label map per visit, every image the agent saw and the full transcript to
the output folder (default ./runs/<time>). Without --task it is the full RECIST 1.1 read; --task gives another task,
e.g. `lesions`: find and segment every lesion of one scan. `setup` checks the GPU, downloads all model weights and
tests the LLM.
"""

from __future__ import annotations

import argparse
import contextlib
import sys
import threading


class _ReadLog:
    """stdout during a read: the log is printed from the main thread, while the model libraries print their own
    progress from the tool threads. Only the main thread's lines pass."""

    def __init__(self, stream):
        self.stream, self.thread = stream, threading.get_ident()

    def write(self, s: str) -> int:
        return self.stream.write(s) if threading.get_ident() == self.thread else len(s)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def _llm_args(p) -> None:
    p.add_argument("--model", help="LLM: a Claude alias (opus, sonnet), a model id, or the model on --llm-url")
    p.add_argument("--effort", help="reasoning effort: low | medium | high | xhigh (default high)")
    p.add_argument("--llm-url", help="an Anthropic-compatible server instead of Claude, e.g. vLLM at http://localhost:8000")
    p.add_argument("--api-key-file", help="file with the API key (else $ANTHROPIC_API_KEY, or $LLM_API_KEY with --llm-url)")
    p.add_argument("--config", help="settings file: pinned model versions, LLM (default ./lesionagent.toml if present)")


def _llm(a):
    from .runtime import resolve_llm

    return resolve_llm(a.model, a.effort, a.llm_url, a.api_key_file)


def _lesion_table(ws) -> str:
    """The lesions of a read without a RECIST report: one row per lesion, sizes measured on the masks."""
    visits = ws.study.visits
    rows = [f"#{l.id} | {l.name} | {l.organ or '?'} | " + " | ".join(ws._lesion_summary(l, v.index) for v in visits)
            + (f" | confidence {l.confidence}" if l.confidence else "")
            for l in sorted(ws.registry.lesions.values(), key=lambda l: l.id) if l.role != "rejected"]  # fmt: skip
    head = "id | name | organ | " + " | ".join(f"{v.name}: long and short axis (mm), volume" for v in visits)
    return "\n".join([head, *rows]) if rows else "no lesions found"


def cmd_run(a) -> int:
    from .core.study import Study
    from .reader import read

    study = Study.from_files(a.images, a.days)
    if a.days is None and len(a.images) > 1:
        print("note: without --days the report cannot check the minimum interval for stable disease", file=sys.stderr)
    llm = _llm(a)
    with contextlib.redirect_stdout(_ReadLog(sys.stdout)):
        result = read(study, a.out, task=a.task, llm=llm, log=lambda s: print(s, flush=True))
    if result.report is not None:
        print(f"\n{(result.out / 'report.md').read_text()}\nreport and masks: {result.out}")
        return 0
    if a.task is None or not result.run["finished"]:  # the default task ends with the RECIST report
        print(f"\n{'no report: ' if a.task is None else ''}the agent did not finish the read "
              f"(see {result.out / 'transcript.jsonl'})", file=sys.stderr)  # fmt: skip
        return 1
    print(f"\n{_lesion_table(result.workspace)}\n\nlesions and masks: {result.out / 'result'}")
    return 0


def cmd_setup(a) -> int:
    from .check import setup

    return setup(lambda: _llm(a), models=not a.skip_models, llm=not a.skip_llm)


def main(argv=None) -> int:
    from . import tasks

    ap = argparse.ArgumentParser(prog="lesionagent", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="read one patient")
    r.add_argument("--images", nargs="+", required=True, help="CT volumes in chronological order, baseline first")
    r.add_argument("--days", nargs="+", type=int, help="days from baseline, one per image (e.g. 0 63 126)")
    r.add_argument("--out", help="output folder (default ./runs/<time>)")
    r.add_argument("--task", help=f"another task than the full RECIST 1.1 read: {', '.join(tasks.names())}, "
                                  "a text file, or the text itself")  # fmt: skip
    _llm_args(r)
    s = sub.add_parser("setup", help="check the GPU, download the model weights, test the LLM")
    s.add_argument("--skip-models", action="store_true", help="do not download and load the model weights")
    s.add_argument("--skip-llm", action="store_true", help="do not send a test message to the LLM")
    _llm_args(s)
    a = ap.parse_args(argv)
    from claude_agent_sdk import ClaudeSDKError

    from . import config
    from .runtime import LLMError

    try:
        config.load(a.config)  # before any model or the LLM is chosen
        return cmd_run(a) if a.cmd == "run" else cmd_setup(a)
    except (LLMError, FileNotFoundError, FileExistsError, ValueError, RuntimeError) as e:
        print(f"error: {e}", file=sys.stderr)
    except ClaudeSDKError as e:
        print(f"error: the LLM session failed ({type(e).__name__}): {e}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
