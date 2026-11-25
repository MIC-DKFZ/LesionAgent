"""The settings file and the choice of the LLM (no network)."""
import os
from pathlib import Path

import pytest

from lesionagent import config
from lesionagent.runtime import LLMError, _clean_env, resolve_llm


@pytest.fixture(autouse=True)
def clean(monkeypatch, tmp_path):
    for k in ("ANTHROPIC_API_KEY", "LLM_API_KEY", "LESIONAGENT_CONFIG"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config, "_ACTIVE", config.Config())
    yield


def test_defaults_are_claude_and_the_newest_models():
    c = config.load()
    assert (c.path, c.models.voxtell, c.models.nninteractive, c.models.longitrack) == (None, None, None, None)
    llm = resolve_llm()
    assert (llm.model, llm.effort, llm.env) == ("opus", "high", {})


def test_file_pins_models_and_llm(tmp_path):
    (tmp_path / "lesionagent.toml").write_text('[llm]\nmodel = "sonnet"\neffort = "medium"\n\n[models]\nlongitrack = "1.0"\n')
    c = config.load()  # ./lesionagent.toml is found
    assert c.models.longitrack == "1.0" and c.models.voxtell is None
    assert (resolve_llm().model, resolve_llm().effort) == ("sonnet", "medium")
    assert resolve_llm(model="opus").model == "opus"  # an argument wins over the file
    (tmp_path / "typo.toml").write_text('[models]\nlongitrak = "1.0"\n')
    with pytest.raises(ValueError, match="longitrak"):
        config.load(tmp_path / "typo.toml")
    with pytest.raises(FileNotFoundError):
        config.load(tmp_path / "missing.toml")


def test_own_server_and_keys(tmp_path):
    key = tmp_path / "key"
    key.write_text("secret\n")
    llm = resolve_llm(model="Qwen/Qwen3.8-27B", effort="medium", url="http://localhost:8000/v1", api_key_file=key, check=False)
    assert llm.env["ANTHROPIC_BASE_URL"] == "http://localhost:8000" and llm.env["ANTHROPIC_API_KEY"] == "secret"
    assert llm.env["ANTHROPIC_DEFAULT_OPUS_MODEL"] == "Qwen/Qwen3.8-27B" and not llm.report_cost
    assert "secret" not in repr(llm)
    assert resolve_llm(api_key_file=key).env == {"ANTHROPIC_API_KEY": "secret"}
    with pytest.raises(LLMError):
        resolve_llm(url="http://localhost:8000", check=False)  # no model given
    with pytest.raises(LLMError):
        resolve_llm(effort="extreme")


def test_session_variables_are_not_passed_on(monkeypatch):
    for k in ("CLAUDECODE", "CLAUDE_CODE_SESSION_ID", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_USE_BEDROCK"):
        monkeypatch.setenv(k, "1")
    _clean_env()
    assert "CLAUDE_CODE_SESSION_ID" not in os.environ and "CLAUDECODE" not in os.environ
    assert os.environ["CLAUDE_CODE_USE_BEDROCK"] == "1"  # the user's provider choice stays


def test_example_file_is_valid():
    c = config.load(Path(__file__).parents[1] / "lesionagent.example.toml")
    assert (c.models.voxtell, c.models.nninteractive, c.models.longitrack) == ("voxtell_v1.1", "nnInteractive_v1.0", "1.0")
    assert (resolve_llm().model, resolve_llm().effort) == ("opus", "high")


def test_file_applies_before_any_model_loads(tmp_path, monkeypatch):
    from lesionagent import cli

    (tmp_path / "lesionagent.toml").write_text('[models]\nlongitrack = "1.0"\n')
    monkeypatch.setattr(config, "_ACTIVE", None)
    assert config.get().models.longitrack == "1.0"  # found in the working folder on first use
    (tmp_path / "other.toml").write_text('[models]\nlongitrack = "1.1"\n')
    monkeypatch.setattr("lesionagent.check._gpu", lambda: "test")
    assert cli.main(["setup", "--skip-models", "--skip-llm", "--config", str(tmp_path / "other.toml")]) == 0
    assert config.get().models.longitrack == "1.1"  # --config wins and is read by setup, too


def test_effort_follows_what_the_server_takes(monkeypatch):
    """A server names the efforts it takes by refusing the others, with status 400 or 500."""
    from lesionagent import runtime

    def http(url, headers, payload=None, timeout=120):
        if payload is None:
            return 200, {"data": [{"id": "qwen"}]}
        ok = payload["output_config"]["effort"] != "high"
        return (200, {}) if ok else (500, {"error": {"message": "Unexpected reasoning effort high."}})

    monkeypatch.setattr(runtime, "_http", http)
    assert runtime._check_server("http://localhost:8000", None, "high", "key") == ("qwen", "medium")
    monkeypatch.setattr(runtime, "_http", lambda url, headers, payload=None, timeout=120: (500, {"error": "out of memory"}))
    with pytest.raises(LLMError, match="answered 500"):
        runtime._check_server("http://localhost:8000", "qwen", "high", "key")
