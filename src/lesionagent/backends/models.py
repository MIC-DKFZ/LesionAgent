"""Lazy GPU model manager. Each backend registers a loader; `get(name)` loads on first use, `release` frees.

Models stay resident on the GPU, so switching between them costs nothing. GPU steps run through `run` / `retry`: on a
CUDA out-of-memory error (while loading or running) the other resident models that are not in use are released and
the step runs once more; with nothing left to release the error propagates (another process holds the GPU: the
tool layer waits and retries). After each step the cached activation memory goes back to the driver, so an idle
process holds only the weights. LESIONAGENT_GPU_EXCLUSIVE=1: one model family at a time (loading one releases the others).
"""

from __future__ import annotations

import gc
import os
import threading
from typing import Callable, TypeVar

T = TypeVar("T")
_LOADERS: dict[str, Callable[[], object]] = {}
_MODELS: dict[str, object] = {}
_BUSY: dict[str, int] = {}
_LOCK = threading.RLock()
VERSIONS: dict[str, str] = {}  # model -> the version that was loaded (the default is the newest release)
EXCLUSIVE = os.environ.get("LESIONAGENT_GPU_EXCLUSIVE", "0") not in ("0", "")


def register(name: str, loader: Callable[[], object]) -> None:
    _LOADERS[name] = loader


_MODULES = {"longitrack": "longitrack", "registration": "longitrack", "voxtell": "voxtell", "nninteractive": "interactive"}


def get(name: str, exclusive: bool | None = None):
    if name not in _LOADERS and name in _MODULES:  # loaders register on import of their backend module
        import importlib

        importlib.import_module(f"{__package__}.{_MODULES[name]}")
    with _LOCK:
        if name in _MODELS:
            return _MODELS[name]
        if exclusive if exclusive is not None else EXCLUSIVE:
            release(keep=(name, *(n for n in _MODELS if _BUSY.get(n))))
        try:
            model = _LOADERS[name]()
        except Exception as e:
            if not is_oom(e) or not evict_others(name):
                raise
            model = _LOADERS[name]()
        _MODELS[name] = model
        return model


def retry(name: str, fn: Callable[[], T]) -> T:
    """fn(), a GPU step of model `name`. On CUDA out-of-memory with other resident models (not in use) release them
    and run fn() once more. Cached activation memory is returned to the driver afterwards."""
    with _LOCK:
        _BUSY[name] = _BUSY.get(name, 0) + 1
    try:
        try:
            return fn()
        except Exception as e:
            if not is_oom(e) or not evict_others(name):
                raise
        return fn()
    finally:
        with _LOCK:
            _BUSY[name] -= 1
        _empty_cache()


def run(name: str, fn: Callable[[object], T], exclusive: bool | None = None) -> T:
    """fn(model) with the (loaded on demand) model `name`, as `retry`."""
    return retry(name, lambda: fn(get(name, exclusive)))


def evict_others(name: str) -> bool:
    """After a CUDA out-of-memory error in `name`: release the other resident models that are not in use.
    False if there was none (retrying would not help)."""
    with _LOCK:
        others = [n for n in _MODELS if n != name and not _BUSY.get(n)]
        release(keep=tuple(n for n in _MODELS if n not in others))
    return bool(others)


def release_idle() -> list[str]:
    """Release every resident model that is not in use (e.g. before a GPU subprocess)."""
    with _LOCK:
        return release(keep=tuple(n for n in _MODELS if _BUSY.get(n)))


def is_oom(e: BaseException | None) -> bool:
    """CUDA out-of-memory, also when re-raised from a worker thread (as the cause)."""
    for _ in range(5):
        if e is None:
            return False
        if type(e).__name__ == "OutOfMemoryError" or any(m in str(e).lower() for m in ("out of memory", "alloc_failed")):
            return True
        e = e.__cause__ or e.__context__
    return False


def loaded() -> list[str]:
    return list(_MODELS)


def release(name: str | None = None, keep: tuple = ()) -> list[str]:
    with _LOCK:
        names = [name] if name else [n for n in _MODELS if n not in keep]
        for n in names:
            obj = _MODELS.pop(n, None)
            if obj is not None and hasattr(obj, "close"):
                try:
                    obj.close()
                except Exception:  # noqa: BLE001
                    pass
        gc.collect()
        _empty_cache()
        return names


def _empty_cache() -> None:
    try:
        import torch

        if torch.cuda.is_initialized():
            torch.cuda.empty_cache()
    except ImportError:
        pass

