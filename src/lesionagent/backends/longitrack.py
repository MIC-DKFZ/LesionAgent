"""LongiTrack (point-prompted lesion segmentation with longitudinal context) + uniGradICON point propagation.

The newest LongiTrack release is used unless the settings file pins one (`[models] longitrack = "1.0"`).
All points are SimpleITK array indices [z, y, x].
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import config
from . import models

LONGITRACK_REPO = "ykirchhoff/LongiTrack"


def model_folder() -> Path:
    """The LongiTrack model folder (downloaded on first use): the pinned version, else the newest one."""
    from longitrack_backend.model import resolve_model_folder

    return Path(resolve_model_folder(LONGITRACK_REPO, version=config.get().models.longitrack))


def _load_engine():
    from longitrack_backend.inference import TrackingEngine

    folder = model_folder()
    models.VERSIONS["longitrack"] = folder.name
    eng = TrackingEngine(folder, device="cuda", gpu_cache_bytes=2 * 2**30)
    eng.load()
    return eng


def _load_registration():
    from longitrack_backend.registration import RegistrationService

    return RegistrationService()


models.register("longitrack", _load_engine)
models.register("registration", _load_registration)


@dataclass
class Propagated:
    baseline_point: list
    followup_point: list | None
    out_of_bounds: bool
    error: str | None = None


def propagate(bl_image: Path, fu_image: Path, points_zyx: list, fu_shape: tuple) -> list[Propagated]:
    """Register baseline -> follow-up (uniGradICON, cached per pair) and map the points."""
    pts = [list(map(float, p)) for p in points_zyx]
    res = models.run("registration", lambda reg: reg.propagate(str(bl_image), str(fu_image), pts, tuple(fu_shape)),
                     exclusive=False)  # small: it can stay next to LongiTrack
    return [Propagated(r.baseline_index, r.followup_index, r.out_of_bounds, r.error) for r in res]


def segment(image: Path, point_zyx: list, context_image: Path | None = None, context_point: list | None = None,
            tta: bool = True, pad: str = "zero") -> tuple[np.ndarray, dict]:  # fmt: skip
    """Segment the lesion at `point` in `image`. With a context (the same lesion in another visit) LongiTrack
    uses the longitudinal context (`track`); without one it segments from the click alone (`segment_baseline`).
    Mirroring TTA is on by default, as in LongiSeg's tracking command.
    pad="reflect": mirror-pad the preprocessed scans so each point-centred crop lies inside real or mirrored data
    instead of the network's zero padding (scans with a short z-extent; see _track_reflect).
    Returns (mask in the image's SimpleITK grid, info)."""
    return models.run("longitrack", lambda eng: _segment(eng, image, point_zyx, context_image, context_point, tta, pad))


def _segment(eng, image, point_zyx, context_image, context_point, tta, pad):
    if pad == "reflect" and context_image is not None and context_point is not None:
        t0 = time.perf_counter()
        m = _track_reflect(eng, Path(context_image), list(context_point), Path(image), list(point_zyx), tta)
        info = {"mode": "track_reflect", "seconds": round(time.perf_counter() - t0, 1), "notes": []}
    elif context_image is not None and context_point is not None:
        r = eng.track(str(context_image), list(context_point), str(image), list(point_zyx),
                      segment_baseline=False, disable_tta=not tta)  # fmt: skip
        m = r.followup
        info = {"mode": "track", "seconds": round(r.seconds, 1), "notes": r.notes}
    else:
        m = eng.segment_baseline(str(image), list(point_zyx), disable_tta=not tta)
        info = {"mode": "single_click"}
    mask = m.full_mask() > 0
    info.update({"voxels": int(mask.sum()), "volume_ml": round(m.volume_ml, 3), "empty": bool(m.empty)})
    return mask, info


def _reflect_around(img, point_pre: list, patch: list):
    """Mirror-pad a backend PreprocessedImage (on its device) so that the patch centred on `point_pre` lies inside
    it on every axis. Returns (padded image, per-axis offset added in front)."""
    import torch
    from longitrack_backend.inference import PreprocessedImage

    data, lows = img.data, []
    for ax, (p, n, ps) in enumerate(zip(point_pre, data.shape[1:], patch)):
        lo, hi = max(0, ps // 2 - p), max(0, p + ps - ps // 2 - n)
        lows.append(lo)
        if lo or hi:
            mode = "reflect" if n > 1 else "edge"
            sel = torch.as_tensor(np.pad(np.arange(n), (lo, hi), mode=mode), device=data.device)
            data = torch.index_select(data, ax + 1, sel)
    if not any(lows) and data is img.data:
        return img, lows
    return PreprocessedImage(path=img.path, data=data, seg=None, properties=img.properties,
                             original_shape=img.original_shape, spacing_zyx=img.spacing_zyx), lows  # fmt: skip


def _track_reflect(eng, bl_image: Path, bl_point: list, fu_image: Path, fu_point: list, tta: bool):
    """LongiTrack follow-up prediction exactly as TrackingEngine.track (same cached preprocessing, single-patch
    prediction, patch-only restore), but on mirror-padded preprocessed scans; the prediction is cut back to the
    real voxels before the restore."""
    from longitrack_backend.geometry import snap_to_volume
    from longitrack_backend.inference import LesionMask

    eng.load()
    pred = eng.predictor
    prev = pred.use_mirroring
    pred.use_mirroring = tta
    try:
        bl, fu = eng.preprocess(bl_image), eng.preprocess(fu_image)
        patch = list(pred.configuration_manager.patch_size)
        blp_pt, fup_pt = eng.to_preprocessed_point(bl_point, bl), eng.to_preprocessed_point(fu_point, fu)
        blp, bl_lo = _reflect_around(bl, blp_pt, patch)
        fup, fu_lo = _reflect_around(fu, fup_pt, patch)
        prob, lower, upper = eng._predict(blp, [c + o for c, o in zip(blp_pt, bl_lo)],
                                          fup, [c + o for c, o in zip(fup_pt, fu_lo)])  # fmt: skip
        # padded -> real preprocessed coordinates; keep the part of the patch over real voxels
        lo_r = [max(lw - o, 0) for lw, o in zip(lower, fu_lo)]
        hi_r = [min(up - o, n) for up, o, n in zip(upper, fu_lo, fu.preprocessed_shape)]
        cut = (slice(None), *[slice(a - (lw - o), b - (lw - o)) for a, b, lw, o in zip(lo_r, hi_r, lower, fu_lo)])
        mask, bounds = eng._restore_patch(prob[cut], lo_r, hi_r, fu)
    finally:
        pred.use_mirroring = prev
    return LesionMask(mask=mask, point=snap_to_volume(fu_point, fu.original_shape), spacing_zyx=fu.spacing_zyx,
                      scan=fu.path, properties=fu.properties, bounds=bounds, original_shape=fu.original_shape)  # fmt: skip


# ---------------------------------------------------------------------------------------------- FOV mismatch
def _z_profile(vol, step: float = 5.0) -> np.ndarray:
    """Per-slice lung and bone area (cm^2) along array axis0, resampled to `step` mm, ordered inferior->superior."""
    from scipy import ndimage as ndi

    a = vol.data[:, ::4, ::4]
    sz, sy, sx = vol.spacing
    air = a < -400
    cc, _ = ndi.label(air)
    border = np.unique(np.r_[cc[:, 0].ravel(), cc[:, -1].ravel(), cc[:, :, 0].ravel(), cc[:, :, -1].ravel()])
    lung = air & ~np.isin(cc, border)
    px = 16 * sx * sy / 100
    prof = np.stack([lung.sum((1, 2)) * px, (a > 250).sum((1, 2)) * px], 1)
    if _superior_sign(vol) < 0:
        prof = prof[::-1]
    zmm = np.arange(a.shape[0]) * sz
    grid = np.arange(0, zmm[-1], step)
    return np.stack([np.interp(grid, zmm, prof[:, k]) for k in range(2)], 1)


def _superior_sign(vol) -> int:
    d = np.asarray(vol.direction, float).reshape(3, 3)
    return 1 if d[2, 2] >= 0 else -1  # ITK z index axis (numpy axis0) points superior?


def overlap_crop(long_vol, short_vol, margin_mm: float = 30.0, step: float = 5.0):
    """(z0, z1) array-axis0 range of `long_vol` overlapping `short_vol`, or None if the extents are similar."""
    pl, ps = _z_profile(long_vol, step), _z_profile(short_vol, step)
    if len(pl) < 1.25 * len(ps):
        return None
    norm = lambda x: (x - x.mean(0)) / (x.std(0) + 1e-6)  # noqa: E731
    best = max(range(len(pl) - len(ps) + 1), key=lambda o: float((norm(pl[o:o + len(ps)]) * norm(ps)).sum()))
    sz, n = long_vol.spacing[0], long_vol.shape[0]
    lo = max(int((best * step - margin_mm) / sz), 0)
    hi = min(int(((best + len(ps)) * step + margin_mm) / sz), n)
    if _superior_sign(long_vol) < 0:  # profile was flipped: convert back to array indices
        lo, hi = n - hi, n - lo
    return lo, hi


def propagate_with_crop(study, from_visit, to_visit, points_zyx, crop_dir: Path):
    """propagate(), cropping whichever scan is >= 1.25x longer to the overlap first (for example whole-body
    against chest CT). Points are mapped back to the full grids."""
    import SimpleITK as sitk

    fv, tv = study.volume(from_visit), study.volume(to_visit)
    paths = {"from": Path(from_visit.path), "to": Path(to_visit.path)}
    offs = {"from": 0, "to": 0}
    note = ""
    for tag, lv, sv, vis in (("from", fv, tv, from_visit), ("to", tv, fv, to_visit)):
        rng = overlap_crop(lv, sv)
        if rng is None:
            continue
        z0, z1 = rng
        crop_dir.mkdir(parents=True, exist_ok=True)
        out = crop_dir / f"{vis.case_id}_z{z0}-{z1}.nii.gz"
        if not out.exists():
            sitk.WriteImage(lv.image[:, :, z0:z1], str(out), useCompression=True)
        paths[tag], offs[tag] = out, z0
        note += f" (cropped {vis.name} to z {z0}-{z1} for registration: field-of-view mismatch)"
    pts = [[p[0] - offs["from"], p[1], p[2]] for p in points_zyx]
    shape = list(tv.shape)
    shape[0] = shape[0] if offs["to"] == 0 else sitk.ReadImage(str(paths["to"])).GetSize()[2]
    res = propagate(paths["from"], paths["to"], pts, tuple(shape))
    for r in res:
        if r.followup_point is not None:
            r.followup_point = [int(r.followup_point[0] + offs["to"]), int(r.followup_point[1]), int(r.followup_point[2])]
        r.baseline_point = [int(r.baseline_point[0] + offs["from"]), *map(int, r.baseline_point[1:])]
    return res, note
