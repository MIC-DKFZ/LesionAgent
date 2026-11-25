"""The agent-facing tool layer: the tool list, its schemas and their match with the workspace methods."""
import json
from inspect import signature

import pytest

from lesionagent.core import recist, rules
from lesionagent.tools import TOOLS, TOOLS_BY_NAME, call_tool
from lesionagent.workspace import RecistWorkspace


@pytest.fixture
def ws(study, tmp_path):
    return RecistWorkspace(study, tmp_path / "out")


def test_tool_list():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names)) == 18
    for t in TOOLS:  # every schema is valid JSON and closed
        assert json.loads(json.dumps(t.input_schema))["additionalProperties"] is False


def test_descriptions_quote_the_rules():
    so = f"max({rules.SECOND_OPINION_MIN_MM:g} mm, {100 * rules.SECOND_OPINION_FRACTION:g} %)"
    assert so in TOOLS_BY_NAME["review_candidates"].description
    assert f"too_small (counts {recist.TOO_SMALL_MM:g} mm)" in TOOLS_BY_NAME["update_lesion"].description


def test_every_tool_calls_the_workspace_method_of_its_name(ws):
    for t in TOOLS:
        params = signature(getattr(ws, t.name)).parameters
        assert set(t.properties) | set(t.fixed) <= set(params), t.name
        assert set(t.required) <= set(t.properties), t.name
    assert call_tool(ws, "view_pair", {}).text == "KeyError: 'lesion'"  # a missing required argument
    assert not call_tool(ws, "study_info", {"visit": 1, "unknown": 1}).is_error  # keys outside the schema: ignored
    assert call_tool(ws, "no_such_tool", {}).is_error


def test_study_info_and_log(ws):
    out = call_tool(ws, "study_info", {})
    assert "3 visits" in out.text and "v2 (case_1) day 60" in out.text
    log = [json.loads(line) for line in open(ws.workdir / "log.jsonl")]
    assert log[-1]["tool"] == "study_info" and not log[-1]["error"]


def test_system_prompt(ws):
    from lesionagent.clinician import InteractiveClinician
    from lesionagent.runtime import HOST_PROMPT, AgentSession

    prompt = AgentSession(ws, lambda ev: None).system_prompt()
    assert prompt.startswith(HOST_PROMPT) and "# RECIST 1.1 reading with tools" in prompt
    assert "# Reference: recist-rules.md" in prompt and "# Reference: voxtell-prompts.md" in prompt
    assert "# This run" not in prompt  # no clinician: nothing is added
    ws.clinician = InteractiveClinician(lambda q: None)
    assert "A clinician is available in this session" in AgentSession(ws, lambda ev: None).system_prompt()


def test_read_log_passes_only_the_main_thread(capsys):
    import contextlib
    import sys
    import threading

    from lesionagent.cli import _ReadLog

    with contextlib.redirect_stdout(_ReadLog(sys.stdout)):
        print("agent line")
        t = threading.Thread(target=lambda: print("model library progress"))
        t.start()
        t.join()
    assert capsys.readouterr().out == "agent line\n"
