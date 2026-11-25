"""RECIST diameters from masks, in the SimpleITK array grid [z, y, x].

The measurement plane is the acquisition plane closest to axial: the array axis whose direction cosine is
most aligned with the patient superior-inferior axis is the slice axis. Diameters are measured between
pixel centres.

ld_mm     longest in-plane diameter over all slices (RECIST long axis for non-nodal lesions)
sa_mm     longest diameter perpendicular to the long axis, on the ld slice
sa_max_mm maximum short axis over all slices (RECIST short axis for lymph nodes)
Endpoints are returned as [z, y, x] voxel coordinates so the measurement lines can be drawn.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull


@dataclass
class Measurement:
    ld_mm: float
    sa_mm: float
    sa_max_mm: float
    ld_slice: int | None
    sa_max_slice: int | None
    ld_line: list | None  # [[z,y,x], [z,y,x]]
    sa_line: list | None  # short axis on the ld slice
    sa_max_line: list | None
    volume_ml: float
    n_voxels: int
    n_components: int
    centroid: list  # [z, y, x]
    slice_axis: int

    def to_dict(self) -> dict:
        return asdict(self)


def slice_axis(direction: np.ndarray | None) -> int:
    """Array axis (0..2 in [z,y,x] order) closest to patient S-I, from a SimpleITK direction (row-major 3x3)."""
    if direction is None:
        return 0
    d = np.asarray(direction, float).reshape(3, 3)  # columns = physical direction of index axes (x, y, z)
    si = np.abs(d[2, :])  # S-I component of each index axis (ITK index order x, y, z)
    itk_axis = int(np.argmax(si))
    return 2 - itk_axis  # numpy order is reversed


def _plane_axes(mask2d: np.ndarray, sr: float, sc: float):
    """Long axis + perpendicular chord in one 2D plane (rows, cols), in mm, with endpoints (row, col)."""
    rr, cc = np.nonzero(mask2d)
    if len(rr) < 2:
        return 0.0, 0.0, None, None
    pts = np.stack([rr * sr, cc * sc], 1).astype(float)
    try:
        hull = pts[ConvexHull(pts).vertices] if len(pts) >= 3 else pts
    except Exception:  # collinear
        hull = pts
    d = np.linalg.norm(hull[:, None] - hull[None], axis=-1)
    i, j = np.unravel_index(d.argmax(), d.shape)
    ld = float(d[i, j])
    if ld == 0:
        return 0.0, 0.0, None, None
    u = (hull[j] - hull[i]) / ld
    v = np.array([-u[1], u[0]])
    a, b = pts @ u, pts @ v
    step = min(sr, sc)
    bins = np.round(a / step).astype(int)
    order = np.argsort(bins)
    bins_s, b_s, pts_s = bins[order], b[order], pts[order]
    cuts = np.flatnonzero(np.diff(bins_s)) + 1
    best, sa_line = 0.0, None
    for seg_b, seg_p in zip(np.split(b_s, cuts), np.split(pts_s, cuts)):
        w = seg_b.max() - seg_b.min()
        if w > best:
            best = float(w)
            sa_line = (seg_p[seg_b.argmin()], seg_p[seg_b.argmax()])
    to_idx = lambda p: [p[0] / sr, p[1] / sc]  # noqa: E731
    ld_line = (to_idx(hull[i]), to_idx(hull[j]))
    sa_idx = (to_idx(sa_line[0]), to_idx(sa_line[1])) if sa_line is not None else None
    return ld, best, ld_line, sa_idx


def measure(mask: np.ndarray, spacing_zyx, direction=None) -> Measurement:
    """Measure one binary lesion mask given in the SimpleITK array grid."""
    mask = np.asarray(mask, bool)
    ax = slice_axis(direction)
    other = [k for k in range(3) if k != ax]
    sp = [float(s) for s in spacing_zyx]
    n = int(mask.sum())
    if n == 0:
        return Measurement(0, 0, 0, None, None, None, None, None, 0.0, 0, 0, [], ax)
    nz = [np.flatnonzero(np.any(mask, axis=tuple(a for a in range(3) if a != k))) for k in range(3)]
    bb = tuple(slice(int(z[0]), int(z[-1]) + 1) for z in nz)
    sub = mask[bb]  # all per-lesion work on the bounding-box crop
    off = [s.start for s in bb]
    best = (0.0, 0.0, None, None, None)
    best_sa = (0.0, None, None)
    for k in range(sub.shape[ax]):
        m2 = np.take(sub, k, axis=ax)
        if not m2.any():
            continue
        ld, sa, ldl, sal = _plane_axes(m2, sp[other[0]], sp[other[1]])
        if ld > best[0]:
            best = (ld, sa, k, ldl, sal)
        if sa > best_sa[0]:
            best_sa = (sa, k, sal)

    def to3d(line, k):
        if line is None:
            return None
        out = []
        for p in line:
            q = [0.0, 0.0, 0.0]
            q[ax] = float(k + off[ax])
            q[other[0]] = round(float(p[0] + off[other[0]]), 1)
            q[other[1]] = round(float(p[1] + off[other[1]]), 1)
            out.append(q)
        return out

    from .raster import center_of_mass_crop

    com = [round(c, 1) for c in center_of_mass_crop(sub, bb)]
    return Measurement(
        ld_mm=round(best[0], 2), sa_mm=round(best[1], 2), sa_max_mm=round(best_sa[0], 2),
        ld_slice=None if best[2] is None else int(best[2] + off[ax]),
        sa_max_slice=None if best_sa[1] is None else int(best_sa[1] + off[ax]),
        ld_line=to3d(best[3], best[2]), sa_line=to3d(best[4], best[2]), sa_max_line=to3d(best_sa[2], best_sa[1]),
        volume_ml=round(n * float(np.prod(sp)) / 1000.0, 4), n_voxels=n,
        n_components=int(ndi.label(sub)[1]), centroid=com, slice_axis=ax,
    )  # fmt: skip
