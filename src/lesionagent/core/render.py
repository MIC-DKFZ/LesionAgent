"""Off-screen renders for the agent: lesion contours with ids/names, measurement lines, prompt/anchor marks.

Conventions: absolute voxel tick labels, physical aspect, radiological orientation with R/L/A/P/S/I letters,
nearest-neighbour voxels. On top of the images:
- lesions of the registry drawn in a fixed colour per lesion id (same colour at every visit) + "#id name"
- an optional highlighted lesion (yellow fill + contour), optional candidates (dashed white)
- RECIST measurement lines: long axis yellow, short axis cyan, with mm labels
- points: prompts (green +/red -), anchors such as propagated points (magenta cross); box prompts (green rectangle)
Coordinates are SimpleITK array indices [z, y, x].
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import numpy as np
from matplotlib import patheffects
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import to_rgba
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from PIL import Image

from .raster import plane_axes

__all__ = ["Overlay", "Mark", "Line", "Box", "panel", "orthogonal", "montage", "projection", "stack", "to_png",
           "fov_around", "lesion_color", "WINDOWS"]  # fmt: skip

# ---------------------------------------------------------------- base helpers
BG = "#161616"
FG = "#e8e8e8"
CURRENT_COLOR = "#ffd400"
OBJECT_COLORS = [
    "#00e5ff", "#ff4fd8", "#ff9f1a", "#7c8cff", "#9dff00", "#ff6b6b", "#00ffa8", "#c58cff",
    "#ffd8a8", "#4dabf7", "#f783ac", "#94d82d",
]  # fmt: skip
_OPPOSITE = {"R": "L", "L": "R", "A": "P", "P": "A", "I": "S", "S": "I"}
_STROKE = [patheffects.withStroke(linewidth=2.2, foreground="black")]
MAX_EDGE = 1568  # keep composite images within what vision models take without downscaling


def _flip(letter, role: str) -> bool:
    """Whether to reverse an axis on screen so medical views follow radiological convention."""
    if letter is None:
        return False
    return letter in ("S", "A") if role == "row" else letter in ("R", "A")


def _nice_step(extent: float, target: int = 7) -> int:
    raw = max(extent / target, 1)
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 5, 10):
        if raw <= m * mag:
            return int(m * mag)
    return int(10 * mag)


def _fill_pinholes(mask: np.ndarray) -> np.ndarray:
    """For display only: close holes of a few pixels so the contour shows the object's outline instead of dozens of
    tiny rings. The fill still shows the true mask."""
    from skimage.morphology import remove_small_holes

    return remove_small_holes(mask, max_size=max(4, int(mask.size * 0.0005)))


def _contour(ax, cc, rr, mask, color, lw):
    if mask.all() or not mask.any():
        if mask.all():
            ax.add_patch(Rectangle((cc[0] - 0.5, rr[0] - 0.5), len(cc), len(rr), fill=False, ec=color, lw=lw))
        return
    pad = np.pad(mask.astype(np.float32), 1)
    ax.contour(np.arange(cc[0] - 1, cc[-1] + 2), np.arange(rr[0] - 1, rr[-1] + 2), pad, levels=[0.5], colors=[color],
               linewidths=lw)  # fmt: skip


def stack(panels: list, ncols: int | None = None, gap: int = 6) -> np.ndarray:
    """Arrange RGB panels in a grid (row-major), padding with the background color."""
    ncols = min(ncols or len(panels), len(panels))  # fewer panels than columns (e.g. one slab on the last page)
    nrows = int(np.ceil(len(panels) / ncols))
    bg = np.array([int(BG[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.uint8)
    col_w = [max(p.shape[1] for p in panels[c::ncols]) for c in range(ncols)]
    row_h = [max(p.shape[0] for p in panels[r * ncols : (r + 1) * ncols]) for r in range(nrows)]
    out = np.empty((sum(row_h) + gap * (nrows - 1), sum(col_w) + gap * (ncols - 1), 3), dtype=np.uint8)
    out[:] = bg
    for i, p in enumerate(panels):
        r, c = divmod(i, ncols)
        y, x = sum(row_h[:r]) + gap * r, sum(col_w[:c]) + gap * c
        out[y : y + p.shape[0], x : x + p.shape[1]] = p
    return out


def to_png(rgb: np.ndarray, max_edge: int = MAX_EDGE) -> bytes:
    img = Image.fromarray(rgb)
    if max(img.size) > max_edge:
        f = max_edge / max(img.size)
        img = img.resize((int(img.size[0] * f), int(img.size[1] * f)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def fov_around(shape, bbox=None, center=None, size=None, margin_frac=0.35, min_size=32) -> list:
    """A 3D field of view: the bbox with a margin, or a box of ``size`` around ``center``."""
    if bbox is not None:
        out = []
        for (lo, hi), s in zip(bbox, shape):
            ext = hi - lo
            pad = int(max(np.ceil(ext * margin_frac), (min_size - ext) / 2, 4))
            out.append([max(0, lo - pad), min(s, hi + pad)])
        return out
    if center is not None and size is not None:
        sizes = size if isinstance(size, (list, tuple)) else [size] * 3
        return [[max(0, int(c - z // 2)), min(s, int(c - z // 2) + int(z))] for c, z, s in zip(center, sizes, shape)]
    return [[0, s] for s in shape]


# ---------------------------------------------------------------- views

WINDOWS = {"soft": (-160.0, 240.0), "lung": (-1350.0, 150.0), "bone": (-400.0, 1200.0), "liver": (-40.0, 210.0),
           "ground_glass": (-1000.0, -300.0)}
LD_COLOR, SA_COLOR, ANCHOR_COLOR, CAND_COLOR = "#ffd400", "#3cf0ff", "#ff3cf5", "#ffffff"
POS, NEG = "#2bff4a", "#ff3b3b"


def lesion_color(lesion_id: int) -> str:
    return OBJECT_COLORS[(int(lesion_id) - 1) % len(OBJECT_COLORS)]


@dataclass
class Mark:
    point: list                  # [z, y, x]
    label: str = ""
    kind: str = "anchor"         # anchor | positive | negative
    radius: int = 3              # drawn on slices within +- radius


@dataclass
class Line:
    p: list                      # [z, y, x]
    q: list
    color: str = LD_COLOR
    label: str = ""


@dataclass
class Box:
    c1: list                     # inclusive corners [z, y, x] that share one coordinate (the box's slice)
    c2: list
    label: str = "box"
    positive: bool = True


@dataclass
class Overlay:
    lesions: np.ndarray | None = None        # instance map (value = lesion id)
    names: dict = field(default_factory=dict)
    highlight: int | None = None             # lesion id drawn with fill
    current: np.ndarray | None = None        # an extra mask drawn with fill (e.g. a fresh segmentation)
    candidates: np.ndarray | None = None     # candidate instance map, dashed
    candidate_names: dict = field(default_factory=dict)
    marks: list = field(default_factory=list)
    lines: list = field(default_factory=list)
    boxes: list = field(default_factory=list)
    show_lesions: bool = True


def _sl(arr, axis, index, rows, cols):
    s = [slice(None)] * 3
    s[axis] = index
    ra, ca = plane_axes(axis)
    s[ra], s[ca] = slice(*rows), slice(*cols)
    return arr[tuple(s)]


def panel(data, spacing, orientation, axis, index, rows, cols, ov: Overlay, window, px=480, title=None,
          cross: dict | None = None) -> np.ndarray:  # fmt: skip
    ra, ca = plane_axes(axis)
    n_r, n_c = rows[1] - rows[0], cols[1] - cols[0]
    phys_h, phys_w = n_r * spacing[ra], n_c * spacing[ca]
    scale = px / max(phys_h, phys_w)
    img_h, img_w = max(int(round(phys_h * scale)), 60), max(int(round(phys_w * scale)), 60)
    left, right, top, bottom = 58, 14, 30, 44
    W, H = img_w + left + right, img_h + top + bottom
    fig = Figure(figsize=(W / 100, H / 100), dpi=100, facecolor=BG)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([left / W, bottom / H, img_w / W, img_h / H])
    ax.set_facecolor("black")
    extent = (cols[0] - 0.5, cols[1] - 0.5, rows[1] - 0.5, rows[0] - 0.5)
    ax.imshow(_sl(data, axis, index, rows, cols).astype(np.float32), cmap="gray", vmin=window[0], vmax=window[1],
              extent=extent, interpolation="nearest", aspect="auto")  # fmt: skip
    rr, cc = np.arange(rows[0], rows[1]), np.arange(cols[0], cols[1])

    if ov.candidates is not None:
        cand = _sl(ov.candidates, axis, index, rows, cols)
        for cid in np.unique(cand):
            if cid == 0:
                continue
            m = cand == cid
            pad = np.pad(m.astype(np.float32), 1)
            ax.contour(np.arange(cc[0] - 1, cc[-1] + 2), np.arange(rr[0] - 1, rr[-1] + 2), pad, levels=[0.5],
                       colors=[CAND_COLOR], linewidths=1.0, linestyles="--")  # fmt: skip
            r_c, c_c = np.argwhere(m).mean(0)
            ax.text(cols[0] + c_c, rows[0] + r_c, ov.candidate_names.get(int(cid), f"c{cid}"), color=CAND_COLOR,
                    fontsize=8, ha="center", va="center", path_effects=_STROKE)  # fmt: skip

    if ov.lesions is not None and ov.show_lesions:
        les = _sl(ov.lesions, axis, index, rows, cols)
        for lid in np.unique(les):
            if lid == 0:
                continue
            m = les == lid
            hl = ov.highlight == lid
            if hl:
                fill = np.zeros((*m.shape, 4), np.float32)
                fill[m] = to_rgba(CURRENT_COLOR, 0.22)
                ax.imshow(fill, extent=extent, interpolation="nearest", aspect="auto")
            _contour(ax, cc, rr, _fill_pinholes(m) if hl else m, CURRENT_COLOR if hl else lesion_color(lid),
                     1.8 if hl else 1.1)  # fmt: skip
            r_c, c_c = np.argwhere(m).mean(0)
            nm = ov.names.get(int(lid), "")
            ax.text(cols[0] + c_c, rows[0] + r_c - max(3, 0.08 * n_r), f"#{lid} {nm}".strip(),
                    color=CURRENT_COLOR if hl else lesion_color(lid), fontsize=8, ha="center", va="bottom",
                    path_effects=_STROKE)  # fmt: skip

    if ov.current is not None:
        cur = _sl(ov.current, axis, index, rows, cols) > 0
        if cur.any():
            fill = np.zeros((*cur.shape, 4), np.float32)
            fill[cur] = to_rgba(CURRENT_COLOR, 0.25)
            ax.imshow(fill, extent=extent, interpolation="nearest", aspect="auto")
            _contour(ax, cc, rr, _fill_pinholes(cur), CURRENT_COLOR, 1.6)

    for ln in ov.lines:
        if round(ln.p[axis]) == index and round(ln.q[axis]) == index:
            ax.plot([ln.p[ca], ln.q[ca]], [ln.p[ra], ln.q[ra]], color=ln.color, lw=1.8, marker="o", ms=3,
                    path_effects=_STROKE)  # fmt: skip
            if ln.label:
                ax.text(ln.q[ca], ln.q[ra], " " + ln.label, color=ln.color, fontsize=8, path_effects=_STROKE)

    for mk in ov.marks:
        d = abs(mk.point[axis] - index)
        if d > mk.radius:
            continue
        col = {"anchor": ANCHOR_COLOR, "positive": POS, "negative": NEG}.get(mk.kind, ANCHOR_COLOR)
        if mk.kind == "anchor":
            ax.plot([mk.point[ca]], [mk.point[ra]], marker="+", ms=14 if d == 0 else 9, mew=2.2, color=col,
                    path_effects=_STROKE)  # fmt: skip
        else:
            ax.plot([mk.point[ca]], [mk.point[ra]], marker="o", ms=7 if d == 0 else 5,
                    mfc=col if d == 0 else "none", mec="black" if d == 0 else col, mew=1.2)  # fmt: skip
        if mk.label:
            ax.text(mk.point[ca] + 2, mk.point[ra] - 2, mk.label, color=col, fontsize=8, fontweight="bold",
                    path_effects=_STROKE)  # fmt: skip

    for bx in ov.boxes:
        lo = [min(a, b) for a, b in zip(bx.c1, bx.c2)]
        hi = [max(a, b) for a, b in zip(bx.c1, bx.c2)]
        flat = [a for a in range(3) if lo[a] == hi[a]]
        col = POS if bx.positive else NEG
        if axis in flat and lo[axis] == index:
            ax.add_patch(Rectangle((lo[ca] - 0.5, lo[ra] - 0.5), hi[ca] - lo[ca] + 1, hi[ra] - lo[ra] + 1, fill=False,
                                   ec=col, lw=2.0))  # fmt: skip
            if bx.label:
                ax.text(lo[ca], lo[ra] - 1.5, bx.label, color=col, fontsize=8, fontweight="bold", path_effects=_STROKE)
        elif flat and lo[axis] <= index <= hi[axis]:  # the box's slice crosses this view: a dashed segment
            if flat[0] == ra:
                ax.plot([lo[ca] - 0.5, hi[ca] + 0.5], [lo[ra], lo[ra]], color=col, lw=1.2, ls="--")
            elif flat[0] == ca:
                ax.plot([lo[ca], lo[ca]], [lo[ra] - 0.5, hi[ra] + 0.5], color=col, lw=1.2, ls="--")

    o_r, o_c = orientation[ra], orientation[ca]
    flip_r, flip_c = _flip(o_r, "row"), _flip(o_c, "col")
    ax.set_xlim(extent[1], extent[0]) if flip_c else ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[3], extent[2]) if flip_r else ax.set_ylim(extent[2], extent[3])
    step_r = _nice_step(n_r, target=max(3, img_h / 65))
    step_c = _nice_step(n_c, target=max(3, img_w / 65))
    ax.set_xticks(np.arange(int(np.ceil(cols[0] / step_c)) * step_c, cols[1], step_c))
    ax.set_yticks(np.arange(int(np.ceil(rows[0] / step_r)) * step_r, rows[1], step_r))
    ax.tick_params(colors=FG, labelsize=8, length=3)
    ax.grid(True, color="white", alpha=0.15, linewidth=0.5)
    for s in ax.spines.values():
        s.set_color("#555555")
    names = ("z", "y", "x")
    ax.set_xlabel(f"{names[ca]} (axis{ca}; index {'<-' if flip_c else '->'})", color=FG, fontsize=8, labelpad=2)
    ax.set_ylabel(f"{names[ra]} (axis{ra}; index {'up' if flip_r else 'down'})", color=FG, fontsize=8, labelpad=2)
    ax.set_title(title or f"{names[axis]} = {index}  (axis{axis})", color=FG, fontsize=10, pad=4)
    for letter, flip, pos in ((o_r, flip_r, "row"), (o_c, flip_c, "col")):
        if letter is None:
            continue
        a, b = (letter, _OPPOSITE[letter]) if flip else (_OPPOSITE[letter], letter)
        kw = dict(transform=ax.transAxes, color="#ffb000", fontsize=10, path_effects=_STROKE)
        if pos == "row":
            ax.text(0.5, 0.985, a, ha="center", va="top", **kw)
            ax.text(0.5, 0.015, b, ha="center", va="bottom", **kw)
        else:
            ax.text(0.012, 0.5, a, ha="left", va="center", **kw)
            ax.text(0.988, 0.5, b, ha="right", va="center", **kw)
    if cross:
        if ra in cross and rows[0] <= cross[ra] < rows[1]:
            ax.axhline(cross[ra], color="#4dd2ff", lw=0.6, alpha=0.5)
        if ca in cross and cols[0] <= cross[ca] < cols[1]:
            ax.axvline(cross[ca], color="#4dd2ff", lw=0.6, alpha=0.5)
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()


def orthogonal(data, spacing, orientation, center, fov, ov, window, axes=(0, 1, 2), px=420, prefix="") -> np.ndarray:
    panels = []
    for a in axes:
        ra, ca = plane_axes(a)
        title = f"{prefix}{('z', 'y', 'x')[a]} = {int(center[a])}"
        panels.append(panel(data, spacing, orientation, a, int(center[a]), fov[ra], fov[ca], ov, window, px=px,
                            title=title, cross={k: center[k] for k in range(3)}))  # fmt: skip
    return stack(panels)


def montage(data, spacing, orientation, axis, indices, fov, ov, window, px=280, ncols=None, prefix="") -> np.ndarray:
    ra, ca = plane_axes(axis)
    panels = [panel(data, spacing, orientation, axis, int(i), fov[ra], fov[ca], ov, window, px=px,
                    title=f"{prefix}{('z', 'y', 'x')[axis]} = {int(i)}") for i in indices]  # fmt: skip
    ncols = ncols or min(len(panels), 4 if len(panels) > 9 else 3)
    return stack(panels, ncols=ncols)


def projection(data, spacing, orientation, axis, ov: Overlay, window, px=640, title="") -> np.ndarray:
    """Max-intensity projection along `axis` with projected lesion (and candidate) outlines + ids."""
    proj = data.max(axis=axis)[None] if axis == 0 else np.moveaxis(data.max(axis=axis), 0, 0)[None]
    ra, ca = plane_axes(axis)
    sp = (1.0, spacing[ra], spacing[ca])
    ori = (None, orientation[ra], orientation[ca])

    def flat(inst):
        if inst is None:
            return None
        out = np.zeros(inst.shape[:axis] + inst.shape[axis + 1:], inst.dtype)
        for i in np.unique(inst):
            if i:
                out[np.any(inst == i, axis=axis) & (out == 0)] = i
        return out[None]

    pov = Overlay(lesions=flat(ov.lesions), names=ov.names, highlight=ov.highlight,
                  candidates=flat(ov.candidates), candidate_names=ov.candidate_names)  # fmt: skip
    win = (window[0], max(window[1], float(np.percentile(proj, 99.5))))
    p = panel(proj, sp, ori, 0, 0, (0, proj.shape[1]), (0, proj.shape[2]), pov, win, px=px,
              title=title or f"max projection along {('z', 'y', 'x')[axis]}")  # fmt: skip
    return p
