"""LesionAgent: an LLM agent that reads longitudinal CT end to end, from the images to the RECIST 1.1 response.

    from lesionagent import Study, read

    study = Study.from_files(["bl.nii.gz", "fu1.nii.gz"], days=[0, 63])
    result = read(study, out="runs/patient1")

`RecistWorkspace` and `call_tool` run single tools without an LLM; `resolve_llm` chooses the LLM of a read.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("lesionagent")
except PackageNotFoundError:  # a source checkout that is not installed
    __version__ = "0.0.0"

__all__ = ["Study", "read", "Read", "RecistWorkspace", "call_tool", "resolve_llm", "LLM", "Clinician",
           "InteractiveClinician", "__version__"]  # fmt: skip
_LAZY = {"Study": "core.study", "read": "reader", "Read": "reader", "RecistWorkspace": "workspace",
         "call_tool": "tools", "resolve_llm": "runtime", "LLM": "runtime", "Clinician": "clinician",
         "InteractiveClinician": "clinician"}  # fmt: skip


def __getattr__(name: str):
    if name in _LAZY:
        from importlib import import_module

        return getattr(import_module(f".{_LAZY[name]}", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
