"""Optional settings file: which LLM reads the scans and which model versions the tools load.

Without a file the LLM is Claude (the login or API key that Claude Code uses) and every perception model is its
newest release. A TOML file pins what should not move; every key is optional:

    [llm]
    model = "opus"                  # a Claude alias or full model id, or the model name on your own server
    effort = "high"
    url = "http://localhost:8000"   # an Anthropic-compatible server (e.g. vLLM) instead of Claude
    api_key_file = "~/.keys/llm"
    max_context_tokens = 131072     # context window of the model on your own server

    [models]
    voxtell = "voxtell_v1.1"
    nninteractive = "nnInteractive_v1.0"
    longitrack = "1.0"

The file is the one given (`--config FILE`, `read(..., config=FILE)`), else `$LESIONAGENT_CONFIG`, else
`./lesionagent.toml`. Command-line options win over the `[llm]` table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path

LOCAL_FILE = "lesionagent.toml"
ENV_VAR = "LESIONAGENT_CONFIG"


@dataclass
class LLMConfig:
    model: str | None = None
    effort: str | None = None
    url: str | None = None
    api_key_file: str | None = None
    max_context_tokens: int | None = None


@dataclass
class ModelConfig:
    """Pinned model versions; None = the newest release."""

    voxtell: str | None = None
    nninteractive: str | None = None
    longitrack: str | None = None


@dataclass
class Config:
    llm: LLMConfig = field(default_factory=LLMConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    path: Path | None = None  # the file it was read from


_ACTIVE: Config | None = None


def get() -> Config:
    """The settings in use. On first use the settings file is looked up as in `find` (no file: the defaults)."""
    return _ACTIVE if _ACTIVE is not None else load()


def find(path=None) -> Path | None:
    """The settings file: `path`, else $LESIONAGENT_CONFIG, else ./lesionagent.toml, else None."""
    if path:
        p = Path(path).expanduser()
        if not p.is_file():
            raise FileNotFoundError(f"settings file {p} not found")
        return p
    env = os.environ.get(ENV_VAR)
    if env:
        return find(env)
    local = Path.cwd() / LOCAL_FILE
    return local if local.is_file() else None


def _table(cls, name: str, raw: dict, path: Path):
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"{path}: unknown key(s) {', '.join(unknown)} in [{name}]; known: {', '.join(sorted(known))}")
    return cls(**{k: (str(v) if k != "max_context_tokens" else int(v)) for k, v in raw.items()})


def load(path=None) -> Config:
    """Read the settings file (see `find`) and make it the settings in use. Without a file: the defaults."""
    global _ACTIVE
    p = find(path)
    if p is None:
        _ACTIVE = Config()
        return _ACTIVE
    import tomllib

    raw = tomllib.loads(p.read_text())
    unknown = sorted(set(raw) - {"llm", "models"})
    if unknown:
        raise ValueError(f"{p}: unknown table(s) {', '.join(unknown)}; known: llm, models")
    _ACTIVE = Config(_table(LLMConfig, "llm", raw.get("llm", {}), p), _table(ModelConfig, "models", raw.get("models", {}), p), p)
    return _ACTIVE
