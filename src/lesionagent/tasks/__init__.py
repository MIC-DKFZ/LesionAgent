"""Built-in tasks: what the agent is asked to do instead of the full RECIST 1.1 read. One text file per task.

    lesions        one scan: find and segment every lesion, of any kind and size
    lung-nodules   one chest CT: find and segment every lung nodule (the task of the paper's LIDC reads)
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def names() -> list[str]:
    return sorted(p.stem for p in _DIR.glob("*.md"))


def text(name: str) -> str:
    """The text of a built-in task. The files are wrapped for reading; each paragraph is sent as one line."""
    return "\n\n".join(" ".join(p.split()) for p in (_DIR / f"{name}.md").read_text().strip().split("\n\n"))
