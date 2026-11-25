"""Geometry helpers for masks and nnInteractive prompts: planes, scribble rasterization, bbox, object centre.

Coordinates are 3D voxel indices [axis0, axis1, axis2] (here: SimpleITK [z, y, x]). A scribble is a polyline
whose points share one coordinate (the plane's fixed axis); it is drawn 2 px wide in that plane — the width
nnInteractive expects — and handed to the session as a tight crop plus its bounding box.
"""

from __future__ import annotations

import numpy as np
from skimage.draw import line as draw_line

SCRIBBLE_WIDTH = 2


def plane_axes(axis: int) -> tuple:
    """The two in-plane axes (row axis, column axis) of the plane with fixed ``axis``."""
    return tuple(a for a in range(3) if a != axis)


def bbox_of(mask: np.ndarray):
    """Half-open [[lo, hi], ...] bbox of the nonzero voxels, or None if empty."""
    out = []
    for axis in range(mask.ndim):
        other = tuple(a for a in range(mask.ndim) if a != axis)
        nz = np.flatnonzero(np.any(mask, axis=other))
        if nz.size == 0:
            return None
        out.append([int(nz[0]), int(nz[-1]) + 1])
    return out


def crop_box(mask: np.ndarray, margin: int = 0) -> tuple | None:
    """Slices of the nonzero bbox grown by `margin` voxels (clipped to the array), or None if empty: per-lesion work
    runs on this crop instead of the whole scan."""
    bb = bbox_of(mask)
    if bb is None:
        return None
    return tuple(slice(max(lo - margin, 0), min(hi + margin, n)) for (lo, hi), n in zip(bb, mask.shape))


def center_of_mass_crop(sub: np.ndarray, box: tuple) -> list:
    """ndi.center_of_mass of the full-scan mask, computed on its crop `sub` = mask[box]: the coordinate sums are
    taken in absolute indices (exact integers, as in the full-scan call) and divided by the same count."""
    idx = np.nonzero(sub)
    n = len(idx[0])
    return [float(np.int64((ix + b.start).sum(dtype=np.int64))) / n for ix, b in zip(idx, box)]


def label_centroids(label_map: np.ndarray) -> dict:
    """{label: centroid [z, y, x]} of every nonzero label in one pass (= ndi.center_of_mass(label_map == i) each)."""
    import cc3d  # one pass for all labels (cc3d.statistics)

    st = cc3d.statistics(np.ascontiguousarray(label_map))
    return {i: np.array(st["centroids"][i], float) for i in range(1, len(st["voxel_counts"])) if st["voxel_counts"][i]}


def object_center(mask: np.ndarray) -> list | None:
    """A point inside the object near its middle: the slice-wise center of the bbox, snapped to the mask."""
    bbox = bbox_of(mask)
    if bbox is None:
        return None
    sub = mask[tuple(slice(lo, hi) for lo, hi in bbox)]
    idx = np.argwhere(sub)
    mid = idx.mean(0)
    best = idx[np.argmin(((idx - mid) ** 2).sum(1))]
    return [int(b + lo) for b, (lo, _) in zip(best, bbox)]


def infer_plane(points: list, axis: int | None = None, prefer: tuple = (0,)) -> tuple:
    """The plane (fixed axis, index) that all points lie in.

    A straight line along one axis lies in two planes; then the first axis in ``prefer`` that qualifies wins
    (default: axial), else the lowest axis.
    """
    if axis is not None:
        values = {p[axis] for p in points}
        if len(values) != 1:
            raise ValueError(f"all points must have the same axis{axis} coordinate, got {sorted(values)}")
        return axis, values.pop()
    shared = [a for a in range(3) if len({p[a] for p in points}) == 1]
    if len(shared) == 1:
        return shared[0], points[0][shared[0]]
    if not shared:
        raise ValueError("the scribble points do not lie in one slice: one coordinate must be the same for all points "
                         "(same z = axial, same y = coronal, same x = sagittal)")  # fmt: skip
    chosen = next((a for a in prefer if a in shared), shared[0])
    return chosen, points[0][chosen]


def _segments(rc: list):
    """Per segment: its pixels and whether it runs mostly along rows (vertical on screen)."""
    out = []
    for k, ((r0, c0), (r1, c1)) in enumerate(zip(rc[:-1], rc[1:])):
        rr, cc = draw_line(int(r0), int(c0), int(r1), int(c1))
        pix = np.stack([rr, cc], axis=1)
        out.append((pix if k == 0 else pix[1:], abs(r1 - r0) >= abs(c1 - c0)))
    return out


def scribble_crop(points3d: list, axis: int, index: int, shape, width: int = SCRIBBLE_WIDTH) -> tuple:
    """Rasterize a polyline in plane ``axis = index``, ``width`` px wide.

    The line is thickened perpendicular to each segment's main direction (a vertical-ish segment gets extra columns,
    a horizontal-ish one extra rows), so it is exactly ``width`` px wide and never extends beyond its end points.

    Returns ``(crop_3d, bbox, centerline_3d)``: the uint8 crop (singleton along ``axis``), its half-open
    ``[[lo, hi], ...]`` bbox in image coordinates (what ``add_scribble_interaction(..., interaction_bbox=bbox)``
    takes), and the centerline pixels as (N, 3) image coordinates.
    """
    ra, ca = plane_axes(axis)
    rc = [(p[ra], p[ca]) for p in points3d]
    if len({tuple(v) for v in rc}) == 1:
        raise ValueError("the scribble points are all the same; give at least two different points")
    offsets = np.arange(width) - (width - 1) // 2
    thick, center = [], []
    for pix, vertical in _segments(rc):
        center.append(pix)
        for o in offsets:
            thick.append(pix + (np.array([0, o]) if vertical else np.array([o, 0])))
    center = np.concatenate(center)
    pix = np.concatenate(thick)
    keep = (pix[:, 0] >= 0) & (pix[:, 0] < shape[ra]) & (pix[:, 1] >= 0) & (pix[:, 1] < shape[ca])
    pix = pix[keep]
    lo, hi = pix.min(0), pix.max(0) + 1
    crop2d = np.zeros(tuple(hi - lo), dtype=np.uint8)
    crop2d[pix[:, 0] - lo[0], pix[:, 1] - lo[1]] = 1
    bbox = [[0, 0], [0, 0], [0, 0]]
    bbox[axis] = [int(index), int(index) + 1]
    bbox[ra] = [int(lo[0]), int(hi[0])]
    bbox[ca] = [int(lo[1]), int(hi[1])]
    centerline = np.zeros((len(center), 3), dtype=int)
    centerline[:, axis] = index
    centerline[:, ra] = center[:, 0]
    centerline[:, ca] = center[:, 1]
    keep = ((centerline >= 0) & (centerline < np.asarray(shape))).all(1)
    return np.expand_dims(crop2d, axis), bbox, centerline[keep]


def bbox_from_corners(c1: list, c2: list) -> tuple:
    """Half-open bbox from two inclusive corners that share one coordinate (2D box); returns (bbox, fixed axis)."""
    flat = [a for a in range(3) if c1[a] == c2[a]]
    if len(flat) != 1:
        raise ValueError("a box is a 2D rectangle: the two corners must share exactly one coordinate (the slice), "
                         f"got {c1} and {c2}")  # fmt: skip
    bbox = [[min(a, b), max(a, b) + 1] for a, b in zip(c1, c2)]
    return bbox, flat[0]
