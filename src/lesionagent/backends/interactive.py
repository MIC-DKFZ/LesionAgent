"""nnInteractive (local) for prompt-based segmentation and refinement of a lesion mask on one visit.

One session per visit image, reset per lesion. A refinement starts from the lesion's current mask
(`add_initial_seg_interaction`) and applies positive/negative points, boxes or scribbles. Coordinates [z, y, x].

A scribble is rasterized 2 px wide in its plane and sent as a tight crop with its bounding box, which is what
nnInteractive expects (core/raster.py).
"""

from __future__ import annotations

import os

import numpy as np

from .. import config
from ..core import raster
from . import models


def make_local_session(model_id: str | None = None):
    """An in-process nnInteractiveInferenceSession on cuda:0 (CPU without a GPU) with the given nnInteractive model,
    else the one pinned in the settings file, else nnInteractive's current default. Downloaded on first use."""
    import torch
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
    from nnInteractive.model_management import ensure_model_available, get_default_model_id

    model_id = models.VERSIONS["nninteractive"] = model_id or config.get().models.nninteractive or get_default_model_id()
    folder = ensure_model_available(model_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    session = nnInteractiveInferenceSession(device=device, use_torch_compile=False,
                                            torch_n_threads=os.cpu_count() or 8, verbose=False)  # fmt: skip
    session.initialize_from_trained_model_folder(folder, 0, "checkpoint_final.pth")
    return session


class InteractiveSession:
    def __init__(self):
        self.s = make_local_session()
        self.image_key = None
        self.target = None
        self.shape = None

    def set_image(self, key: str, data: np.ndarray, spacing_zyx) -> None:
        import torch

        if key == self.image_key:
            return
        self.s.set_image(data[None].astype(np.float32), {"spacing": list(map(float, spacing_zyx))})
        self.target = torch.zeros(data.shape, dtype=torch.uint8)
        self.s.set_target_buffer(self.target)
        self.image_key, self.shape = key, tuple(data.shape)

    def start(self, initial_mask: np.ndarray | None = None) -> None:
        """New lesion: empty, or initialised from an existing mask (e.g. a VoxTell candidate) to correct."""
        if initial_mask is not None and initial_mask.any():
            self.s.add_initial_seg_interaction(initial_mask.astype(np.uint8), run_prediction=False)  # also resets
        else:
            self.s.reset_interactions()

    def _gpu(self, fn) -> None:
        """One prompt with its prediction. nnInteractive rolls a failed prompt back, so after a CUDA out-of-memory
        error (other resident models released) the same prompt runs again."""
        models.retry("nninteractive", fn)

    def point(self, zyx, positive: bool = True) -> np.ndarray:
        self._gpu(lambda: self.s.add_point_interaction([int(round(c)) for c in zyx], include_interaction=positive))
        return self.mask()

    def bbox(self, c1, c2, positive: bool = True) -> np.ndarray:
        """2D box from two inclusive corners in one slice (they share one coordinate)."""
        box = [[int(min(a, b)), int(max(a, b)) + 1] for a, b in zip(c1, c2)]
        self._gpu(lambda: self.s.add_bbox_interaction(box, include_interaction=positive))
        return self.mask()

    def scribble_line(self, points, positive: bool = True, axis: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Polyline through >= 2 points in one slice (any plane), drawn 2 px wide. Returns (mask, centerline [N, 3])."""
        pts = [[int(round(c)) for c in p] for p in points]
        plane, index = raster.infer_plane(pts, axis)
        crop, bbox, centerline = raster.scribble_crop(pts, plane, index, self.shape)
        self._gpu(lambda: self.s.add_scribble_interaction(crop, include_interaction=positive, interaction_bbox=bbox))
        return self.mask(), centerline

    def mask(self) -> np.ndarray:
        return self.target.numpy().astype(bool).copy()

    def close(self) -> None:
        self.s = None


models.register("nninteractive", InteractiveSession)
