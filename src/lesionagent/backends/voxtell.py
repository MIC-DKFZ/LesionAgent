"""VoxTell text-prompted segmentation -> one mask per prompt in the SimpleITK grid.

VoxTell needs RAS-reoriented input (nnU-Net NibabelIOWithReorient). Its masks are written back with the same
reader/writer (which undoes the reorientation) and read with SimpleITK, so they land exactly on the visit's
[z, y, x] grid. Up to 16 prompts are packed as bits of one uint16 volume for a single round trip.

The newest VoxTell release is used unless the settings file pins one (`[models] voxtell = "voxtell_v1.1"`).
"""

from __future__ import annotations

import re
import tempfile
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from .. import config
from . import models

VOXTELL_REPO = "mrokuss/VoxTell"
VOXTELL_KNOWN = "voxtell_v1.1"  # used when the Hub cannot be asked for the newest release
_BANK: dict | None = None  # VoxTell's published bank of precomputed prompt embeddings, kept across model reloads
_EMBEDDINGS: dict = {}  # lowercased prompt -> embedding (CPU): every prompt is embedded once per process


def model_name() -> str:
    """The VoxTell model folder in the Hub repo: the pinned one, else the newest `voxtell_v*` release."""
    pinned = config.get().models.voxtell
    if pinned:
        return pinned
    try:
        from huggingface_hub import HfApi

        names = {f.split("/", 1)[0] for f in HfApi().list_repo_files(VOXTELL_REPO) if "/" in f}
    except Exception:  # noqa: BLE001  (offline, Hub not reachable)
        return VOXTELL_KNOWN
    versions = {n: tuple(int(x) for x in re.findall(r"\d+", n)) for n in names if re.fullmatch(r"voxtell_v[\d.]+", n)}
    return max(versions, key=versions.get) if versions else VOXTELL_KNOWN


def _load():
    global _BANK
    import torch
    from huggingface_hub import snapshot_download
    from voxtell.inference.predictor import VoxTellPredictor

    name = models.VERSIONS["voxtell"] = model_name()
    root = Path(snapshot_download(VOXTELL_REPO, allow_patterns=[f"{name}/*"]))
    pred = VoxTellPredictor(str(root / name), device=torch.device("cuda"), embedding_bank=_BANK)
    if _BANK is None and pred.embedding_bank is not None:
        _BANK = dict(pred.embedding_bank)
    return pred


models.register("voxtell", _load)


def predict(image: Path, prompts: list[str]) -> dict[str, np.ndarray]:
    """Binary mask per prompt, in the SimpleITK array grid of `image`."""
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

    out: dict[str, np.ndarray] = {}
    io = NibabelIOWithReorient()
    data, props = io.read_images([str(image)])
    for start in range(0, len(prompts), 16):
        chunk = prompts[start:start + 16]
        seg = models.run("voxtell", lambda pred: _predict_oom_safe(pred, data, _embed(pred, chunk)))  # (P, *reoriented)
        packed = np.zeros(seg.shape[1:], np.uint16)
        for i, m in enumerate(seg):
            packed |= (m.astype(np.uint16) << i)
        with tempfile.TemporaryDirectory() as tmp:
            f = str(Path(tmp) / "packed.nii.gz")
            io.write_seg(packed, f, props)
            back = sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.uint16)
        for i, p in enumerate(chunk):
            out[p] = ((back >> i) & 1).astype(bool)
    return out


def _embed(pred, prompts: list[str]):
    """Text embeddings (1, P, D) as `pred.embed_text_prompts` gives them on a fresh predictor (bank prompts by lookup,
    the others through the Qwen3 text encoder), but computed once per prompt and process. The text encoder (~8 GB)
    is freed right after use instead of staying in RAM."""
    import torch

    keys = [p.lower() for p in prompts]
    todo = [k for k in keys if k not in _EMBEDDINGS]
    if todo:
        for k, e in zip(todo, pred.embed_text_prompts(todo)[0]):
            _EMBEDDINGS[k] = e.cpu()
        if pred.text_backbone is not None:
            pred.text_backbone = pred.tokenizer = None
            _trim()
    with torch.inference_mode():
        return torch.stack([_EMBEDDINGS[k] for k in keys]).to(pred.device).view(1, len(keys), -1)


def _trim() -> None:
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def _predict_oom_safe(pred, data, emb) -> np.ndarray:
    """predict_single_image with the prompt embeddings `emb`. On CUDA OOM release the other resident models and try
    again; then (very large volumes) with half the prompts, and for a single prompt with the logits kept on the
    CPU. Same result, only slower."""
    import torch

    try:
        return pred.predict_single_image(data, text_embeddings=emb)
    except Exception as e:
        if not models.is_oom(e):
            raise
        torch.cuda.empty_cache()
    if models.evict_others("voxtell"):
        return _predict_oom_safe(pred, data, emb)
    if emb.shape[1] > 1:
        h = emb.shape[1] // 2
        return np.concatenate([_predict_oom_safe(pred, data, emb[:, :h]), _predict_oom_safe(pred, data, emb[:, h:])])
    pred.perform_everything_on_device = False
    try:
        return pred.predict_single_image(data, text_embeddings=emb)
    finally:
        pred.perform_everything_on_device = True
        torch.cuda.empty_cache()
