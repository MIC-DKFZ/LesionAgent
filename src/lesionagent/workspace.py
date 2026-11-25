"""RecistWorkspace: everything the RECIST agent needs — study, lesion registry, candidates, backends, renders.

Every public method returns a Result (text for the agent, optional PNG renders, structured data), so the same
calls serve the agent tools and a host application. Every state change emits an event
("changed", "activity", "phase", "question") to registered listeners, so a viewer can show live what the agent
and the models are doing.

Coordinates are SimpleITK array indices [z, y, x]. Sizes for views are in mm.
"""

from __future__ import annotations

import json
import threading
import time
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from .clinician import KINDS, Clinician
from .core import raster, render, rules, selection
from .core.raster import bbox_of, center_of_mass_crop, crop_box, label_centroids, object_center, plane_axes
from .core.measure import measure
from .core.registry import ROLES, STATUSES, TYPES, LesionRegistry
from .core.study import Study


class AgentError(Exception):
    """A usage error the agent can fix (unknown lesion, bad coordinates, wrong order)."""


@dataclass
class Result:
    text: str
    images: list = field(default_factory=list)  # PNG bytes
    data: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    __repr__ = __str__


_LPS = [("R", "L"), ("A", "P"), ("I", "S")]


def orientation_letters(direction) -> tuple:
    """Patient direction (LPS) of increasing index along each numpy axis [z, y, x]."""
    d = np.asarray(direction, float).reshape(3, 3)
    out = []
    for k in range(3):
        vec = d[:, 2 - k]
        j = int(np.argmax(np.abs(vec)))
        out.append(_LPS[j][1] if vec[j] > 0 else _LPS[j][0])
    return tuple(out)


def _fmt(x, nd=1):
    return "-" if x is None else (f"{x:.{nd}f}" if isinstance(x, float) else str(x))


class RecistWorkspace:
    def __init__(self, study: Study, workdir, clinician: Clinician | None = None, cache_dir=None):
        self.study = study
        self.workdir = Path(workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.cache_dir = Path(cache_dir) if cache_dir else self.workdir / "cache"
        self.registry = LesionRegistry(study)
        self.clinician = clinician or Clinician()
        self.cands: dict[int, dict] = {}          # visit index -> {"map": uint16, "items": {cid: dict}}
        self.organ_maps: dict[int, object] = {}
        self._undo: dict = {}
        self.scores: dict[int, dict] = {}         # lesion id -> {"flags", "repro_score", "rationale"}
        self.target_justification = ""
        self.non_target_groups: list = []
        self.listeners: list = []
        self.lock = threading.RLock()
        self.n_model_calls = 0
        self.questions: list = []

    # ------------------------------------------------------------------ events
    def emit(self, _event: str, **payload) -> None:
        for fn in list(self.listeners):
            try:
                fn(_event, payload)
            except Exception:  # noqa: BLE001 - a broken listener must not break the agent
                traceback.print_exc()

    def activity(self, text: str, **kw) -> None:
        self.emit("activity", text=text, t=time.time(), **kw)

    def changed(self, visits=(), lesions=(), what: str = "") -> None:
        self.emit("changed", visits=[self.study.visit(v).index for v in visits], lesions=list(lesions), what=what)

    # ------------------------------------------------------------------ helpers
    def _vis(self, v=None):
        try:
            return self.study.visit(v if v is not None else self.study.baseline.index)
        except KeyError as e:
            raise AgentError(str(e)) from e

    def _les(self, ref):
        try:
            return self.registry.get(ref)
        except KeyError as e:
            raise AgentError(f"{e}; known lesions: {sorted(self.registry.lesions)}") from e

    def _vol(self, v):
        return self.study.volume(self._vis(v))

    def _ori(self, v):
        return orientation_letters(self._vol(v).direction)

    def _point(self, p, v) -> list:
        shape = self._vol(v).shape
        if p is None or len(p) != 3:
            raise AgentError(f"points are [z, y, x] (3 numbers), got {p!r}")
        q = [int(round(float(c))) for c in p]
        if any(not 0 <= c < s for c, s in zip(q, shape)):
            raise AgentError(f"point {q} outside the volume {list(shape)} of {self._vis(v).name}")
        return q

    def _fov(self, v, center, size_mm=None, bbox=None, margin_mm=25.0, min_mm=rules.VIEW_MIN_MM):
        vol = self._vol(v)
        sp = np.array(vol.spacing)
        if bbox is not None and size_mm is None:
            lo = np.array([b[0] for b in bbox]); hi = np.array([b[1] for b in bbox])
            ext_mm = np.maximum((hi - lo) * sp + 2 * margin_mm, min_mm)
            center = (lo + hi) / 2
        else:
            ext_mm = np.full(3, float(size_mm or 100.0))
        half = np.ceil(ext_mm / sp / 2).astype(int)
        out = []
        for c, h, s in zip(center, half, vol.shape):  # at the volume edge the window is shifted inwards, not cut
            lo, hi = int(c - h), int(c + h) + 1
            lo, hi = (lo - max(0, hi - s), min(s, hi)) if hi > s else (lo, hi)
            lo, hi = (max(0, lo), min(s, hi - lo)) if lo < 0 else (lo, hi)
            out.append([max(0, lo), min(s, hi)])
        return out

    def _center(self, lesion_id, v):
        m = self.registry.mask(lesion_id, v)
        if m.any():
            return object_center(m)
        f = self.registry.lesions[lesion_id].findings.get(self._vis(v).index)
        if f and f.point is not None:
            return [int(round(c)) for c in f.point]
        return None

    def _new_lesion(self, name="", lesion_type="unknown", organ="", role="candidate"):
        return self.registry.new_lesion(name, _type(lesion_type), organ, _role(role))  # unnamed: lesion_<id>

    def _write_mask(self, l, vis, mask, undo=None, notes="", **finding) -> tuple:
        """Replace lesion `l`'s mask at `vis` (under the lock): keep `undo` (the previous mask) and the status for
        undo, set the given Finding fields (status, provenance, point) and the notes (if any), re-measure. A 'present'
        lesion left without voxels becomes 'not_assessed'; info["note"] tells the agent.
        Returns (set_mask info, finding, measurement)."""
        with self.lock:
            f = self.registry.finding(l.id, vis)
            if undo is not None:
                self._undo[(l.id, vis.index)] = (undo, f.status)
            info = self.registry.set_mask(l.id, vis, mask)
            for k, v in finding.items():
                setattr(f, k, v)
            f.notes = notes or f.notes
            m = self.registry.remeasure(l.id, vis)
            info["note"] = self._no_empty_present(l, vis, f) if not info["voxels"] else ""
        return info, f, m

    def _no_empty_present(self, l, vis, f) -> str:
        """'present' needs a mask (the measurement): an empty one makes the finding 'not_assessed'."""
        if f.status != "present":
            return ""
        f.status = "not_assessed"
        return (f"\n  #{l.id} has no mask at {vis.name}: status set to not_assessed (update_lesion status=absent if "
                f"the lesion is gone, or segment it)")

    def _overlay(self, v, highlight=None, show_candidates=False, lines=(), marks=()):
        vis = self._vis(v)
        names = {l.id: l.name for l in self.registry.lesions.values()}
        cm, cnames = None, {}
        if show_candidates and vis.index in self.cands:
            cm = self.cands[vis.index]["map"]
            cnames = {cid: f"c{cid}" for cid, c in self.cands[vis.index]["items"].items() if c["state"] == "open"}
            cm = np.where(np.isin(cm, list(cnames)), cm, 0).astype(np.uint16) if cnames else None
        return render.Overlay(lesions=self.registry.label_map(vis), names=names, highlight=highlight, candidates=cm,
                              candidate_names=cnames, lines=list(lines), marks=list(marks))  # fmt: skip

    def _png(self, rgb, kind):
        png = render.to_png(rgb)
        d = self.workdir / "renders"
        d.mkdir(exist_ok=True)
        f = d / f"{time.strftime('%H%M%S')}_{kind}_{len(list(d.iterdir())):04d}.png"
        f.write_bytes(png)
        self.emit("render", kind=kind, path=str(f))
        return png

    def _window(self, window, lesion=None):
        if isinstance(window, str):
            if window not in render.WINDOWS:
                raise AgentError(f"window name must be one of {list(render.WINDOWS)} or [low, high]")
            return render.WINDOWS[window]
        if window:
            return tuple(window)
        if lesion is not None and self.registry.lesions[lesion].organ == "lung":
            return render.WINDOWS["lung"]
        return render.WINDOWS["soft"]

    def _measure_lines(self, lesion_id, v):
        f = self.registry.lesions[lesion_id].findings.get(self._vis(v).index)
        if not f or not f.measurement or not f.measurement.get("ld_line"):
            return []
        m = f.measurement
        node = self.registry.lesions[lesion_id].is_node
        out = [render.Line(*m["ld_line"], render.LD_COLOR, f"LD {m['ld_mm']:.1f}")]
        if m.get("sa_line"):
            out.append(render.Line(*m["sa_line"], render.SA_COLOR, f"SA {m['sa_mm']:.1f}"))
        if node and m.get("sa_max_line") and m.get("sa_max_slice") != m.get("ld_slice"):
            out.append(render.Line(*m["sa_max_line"], render.SA_COLOR, f"SAmax {m['sa_max_mm']:.1f}"))
        return out

    def _lesion_summary(self, l, vis_index):
        f = l.findings.get(vis_index)
        if f is None:
            return "-"
        m = f.measurement
        s = f.status if f.status != "present" else ""
        if m and m.get("n_voxels"):
            vol = f"{m['volume_ml']:.2f}ml" if m.get("volume_ml") is not None else ""
            return f"{s} LD {m['ld_mm']:.1f} SA {m['sa_max_mm'] if l.is_node else m['sa_mm']:.1f} {vol}".strip()
        return s or "no mask"

    # ================================================================== study / inspection
    def study_info(self, visit=None) -> Result:
        lines = [self.study.describe(), "", self.progress_text(), "", self.list_lesions(visit).text]
        return Result("\n".join(lines))

    def progress(self) -> dict:
        bl = self.study.baseline.index
        les = list(self.registry.lesions.values())
        base = [l for l in les if l.role != "rejected" and bl in l.findings]
        p = {
            "baseline_lesions": len(base),
            "baseline_candidates_open": sum(c["state"] == "open" for c in self.cands.get(bl, {}).get("items", {}).values()),
            "organs_assigned": sum(bool(l.organ) for l in base),
            "targets": sum(l.role == "target" for l in les),
            "follow_up": {},
        }
        for v in self.study.visits[1:]:
            p["follow_up"][v.name] = {
                "tracked": sum(1 for l in base if v.index in l.findings and l.findings[v.index].status != "not_assessed"),
                "of": len(base),
                "new_lesions": sum(1 for l in les if l.role == "new" and v.index in l.findings),
                "judged": v.index in self.registry.judgements,
            }
        return p

    def progress_text(self) -> str:
        p = self.progress()
        s = [f"Progress: baseline lesions {p['baseline_lesions']} (open candidates {p['baseline_candidates_open']}), "
             f"organs {p['organs_assigned']}/{p['baseline_lesions']}, targets {p['targets']}"]  # fmt: skip
        for k, v in p["follow_up"].items():
            s.append(f"  {k}: tracked {v['tracked']}/{v['of']}, new lesions {v['new_lesions']}, "
                     f"visit judgement {'set' if v['judged'] else 'missing'}")  # fmt: skip
        return "\n".join(s)

    def list_lesions(self, visit=None) -> Result:
        les = sorted(self.registry.lesions.values(), key=lambda l: l.id)
        if not les:
            return Result("No lesions in the registry yet.")
        vis = [self._vis(visit)] if visit is not None else self.study.visits
        head = "id | name | type | organ | role | " + " | ".join(v.name for v in vis)
        rows = [head]
        for l in les:
            rows.append(f"#{l.id} | {l.name} | {l.lesion_type} | {l.organ or '?'} | {l.role} | " +
                        " | ".join(self._lesion_summary(l, v.index) for v in vis))  # fmt: skip
        return Result("Lesions (LD/SA in mm; SA = short axis, max over slices for nodes):\n" + "\n".join(rows),
                      data={"n": len(les)})  # fmt: skip

    def overview(self, visit=None, window=None, show_candidates=True) -> Result:
        vis = self._vis(visit)
        vol = self._vol(vis)
        ov = self._overlay(vis, show_candidates=show_candidates)
        w = self._window(window)
        cor = render.projection(np.clip(vol.data, w[0], w[1] + 400), vol.spacing, self._ori(vis), 1, ov, w, px=560,
                                title=f"{vis.name} coronal MIP (clipped) — lesions #id, candidates c#")  # fmt: skip
        ax = render.projection(np.clip(vol.data, w[0], w[1] + 400), vol.spacing, self._ori(vis), 0, ov, w, px=560,
                               title=f"{vis.name} axial MIP (clipped)")  # fmt: skip
        return Result(f"Overview of {vis.name} ({vis.case_id}). Projection outlines show lesions (#id, colour per id) "
                      "and open candidates (dashed, c#). Use view/montage for real slices.",
                      [self._png(render.stack([cor, ax]), "overview")])  # fmt: skip

    def view(self, visit=None, center=None, lesion=None, candidate=None, size_mm=None, window=None,
             axes=(0, 1, 2), show_candidates=None, measurement=False, mode="slices", axis=0, n=9,
             slice_range=None, slab_mm=10.0, projection=None) -> Result:  # fmt: skip
        """mode 'slices': orthogonal slices; 'montage': up to 16 slices along `axis`; 'overview': MIPs of the visit
        (candidates shown unless show_candidates=False); 'sweep': axial slabs."""
        if mode == "overview":
            return self.overview(visit, window, show_candidates if show_candidates is not None else True)
        if mode == "montage":
            return self.montage(visit, lesion, candidate, axis, n, slice_range, size_mm, window, center)
        if mode == "sweep":
            return self.sweep(visit, slice_range, window, slab_mm, projection)
        if mode != "slices":
            raise AgentError("mode must be 'slices', 'montage', 'sweep' or 'overview'")
        vis = self._vis(visit)
        vol = self._vol(vis)
        hl, bbox, lines, marks = None, None, [], []
        if lesion is not None:
            l = self._les(lesion)
            hl = l.id
            m = self.registry.mask(l.id, vis)
            bbox = bbox_of(m)
            if center is None:
                center = self._center(l.id, vis)
            if measurement:
                lines = self._measure_lines(l.id, vis)
                f = l.findings.get(vis.index)
                if f and f.measurement.get("ld_slice") is not None and center is not None:
                    center = [f.measurement["ld_slice"], *center[1:]]
            f = l.findings.get(vis.index)
            if f and f.point is not None:
                marks.append(render.Mark(f.point, f"#{l.id} {f.provenance}"))
        if candidate is not None:
            c = self._cand(vis, candidate)
            bbox = c["bbox"]
            center = center or [int(round(x)) for x in c["centroid"]]
            show_candidates = True
        if center is None:
            if bbox is None:
                raise AgentError("give `center`, `lesion` or `candidate`")
            center = [(a + b) // 2 for a, b in bbox]
        center = self._point(center, vis)
        fov = self._fov(vis, center, size_mm, bbox if size_mm is None else None)
        ov = self._overlay(vis, hl, show_candidates, lines, marks)
        rgb = render.orthogonal(vol.data, vol.spacing, self._ori(vis), center, fov, ov,
                                self._window(window, hl), axes=tuple(axes), prefix=f"{vis.name} ")  # fmt: skip
        return Result(f"{vis.name} views through {center} (FOV {[b - a for a, b in fov]} voxels).",
                      [self._png(rgb, "view")])  # fmt: skip

    def montage(self, visit=None, lesion=None, candidate=None, axis=0, n=9, slice_range=None, size_mm=None,
                window=None, center=None) -> Result:  # fmt: skip
        vis = self._vis(visit)
        vol = self._vol(vis)
        bbox, hl, show_c = None, None, False
        if lesion is not None:
            hl = self._les(lesion).id
            bbox = bbox_of(self.registry.mask(hl, vis))
            if bbox is None:
                c = self._center(hl, vis)
                if c is None:
                    raise AgentError(f"lesion {hl} has no mask or point at {vis.name}")
                bbox = [[x - 5, x + 6] for x in c]
        elif candidate is not None:
            bbox, show_c = self._cand(vis, candidate)["bbox"], True
        elif center is not None:
            c = self._point(center, vis)
            bbox = [[x - 5, x + 6] for x in c]
        if slice_range is None:
            if bbox is None:
                raise AgentError("give lesion, candidate, center or slice_range")
            lo, hi = bbox[axis]
            slice_range = [max(0, lo - 1), min(vol.shape[axis] - 1, hi)]
        n = int(min(max(n, 1), 16))
        idx = np.unique(np.linspace(slice_range[0], slice_range[1], n).round().astype(int))
        center = [(a + b) // 2 for a, b in bbox] if bbox else [s // 2 for s in vol.shape]
        fov = self._fov(vis, center, size_mm, bbox if size_mm is None else None)
        rgb = render.montage(vol.data, vol.spacing, self._ori(vis), axis, idx, fov, self._overlay(vis, hl, show_c),
                             self._window(window, hl), prefix=f"{vis.name} ")  # fmt: skip
        return Result(f"{vis.name} montage along axis{axis}, slices {idx.tolist()}.", [self._png(rgb, "montage")])

    def _lung_box(self, vis, lo, hi):
        """In-plane box (rows, cols) of the aerated lungs in slices lo..hi: air (< -400 HU) components that do not
        touch the image border (outside air), + 10 mm; whole slice if none."""
        vol = self._vol(vis)
        air = vol.data[lo:hi + 1] < rules.AIR_HU
        lab, n = ndi.label(air)
        border = set(np.unique(np.concatenate([lab[:, 0].ravel(), lab[:, -1].ravel(), lab[:, :, 0].ravel(),
                                               lab[:, :, -1].ravel()])))  # fmt: skip
        sizes = np.bincount(lab.ravel())
        keep = [i for i in range(1, n + 1) if i not in border and sizes[i] * np.prod(vol.spacing) > rules.LUNG_MIN_ML * 1000]
        if not keep:
            return (0, vol.shape[1]), (0, vol.shape[2])
        ys, xs = np.where(np.isin(lab, keep).any(0))
        m = int(round(10 / float(vol.spacing[1])))
        return ((max(0, int(ys.min()) - m), min(vol.shape[1], int(ys.max()) + m + 1)),
                (max(0, int(xs.min()) - m), min(vol.shape[2], int(xs.max()) + m + 1)))  # fmt: skip

    def _body_box(self, vis, lo, hi):
        """In-plane box (rows, cols) of the patient (largest component > -500 HU in slices lo..hi, not the table)."""
        vol = self._vol(vis)
        body = _largest_component(vol.data[lo:hi + 1].max(0) > rules.BODY_HU)
        ys, xs = np.where(body)
        if not len(ys):
            return (0, vol.shape[1]), (0, vol.shape[2])
        return (int(ys.min()), int(ys.max()) + 1), (int(xs.min()), int(xs.max()) + 1)

    def sweep(self, visit=None, slice_range=None, window=None, slab_mm=10.0, projection=None,
              per_image: int = 4) -> Result:  # fmt: skip
        """Read the scan systematically: consecutive axial slabs (slab_mm) through slice_range (default: all), any
        window, projection 'mip' (bright structures: lung nodules = dots, vessels = lines), 'mean' (a thick slice:
        soft tissue, nodes, liver, ground glass) or 'slice' (the slab's middle slice). Cropped to the lungs for lung-type
        windows, else to the body; 4 large panels per image; registry lesions and candidates outlined. Page with
        slice_range."""
        vis = self._vis(visit)
        vol = self._vol(vis)
        nz = vol.shape[0]
        win = self._window(window or "soft")
        lungy = win[0] < rules.LUNG_WINDOW_LOW_HU
        projection = projection or ("mip" if lungy and win[1] > rules.MIP_WINDOW_HIGH_HU else "mean")
        if projection not in ("mip", "mean", "slice"):
            raise AgentError("projection must be 'mip', 'mean' or 'slice'")
        step = max(1, int(round(float(slab_mm) / float(vol.spacing[0]))))
        lo, hi = (0, nz - 1) if slice_range is None else (max(0, int(slice_range[0])), min(nz - 1, int(slice_range[1])))
        starts = list(range(lo, hi + 1, step))[: int(per_image)]
        if not starts:
            raise AgentError(f"empty slice_range {slice_range}")
        end = min(starts[-1] + step, hi + 1) - 1
        rows, cols = (self._lung_box if lungy else self._body_box)(vis, starts[0], end)
        lm, cm = self.registry.label_map(vis), self.cands.get(vis.index, {}).get("map")
        names = {l.id: f"#{l.id}" for l in self.registry.lesions.values()}
        panels = []
        for z0 in starts:
            z1 = min(z0 + step, hi + 1)
            sl = slice(z0, z1)
            img = {"mip": lambda a: a.max(0), "mean": lambda a: a.mean(0), "slice": lambda a: a[len(a) // 2]}[projection](
                vol.data[sl])[None]  # fmt: skip
            ov = render.Overlay(lesions=lm[sl].max(0)[None], names=names,
                                candidates=None if cm is None else cm[sl].max(0)[None])  # fmt: skip
            sp = (float(slab_mm), vol.spacing[1], vol.spacing[2])
            panels.append(render.panel(img, sp, self._ori(vis), 0, 0, rows, cols, ov, win, px=560,
                                       title=f"{vis.name} {projection.upper()} z {z0}-{z1 - 1}"))  # fmt: skip
        more = f" Next: slice_range=[{end + 1}, {hi}]." if end < hi else " This covers the requested range."
        return Result(f"{vis.name} sweep: {len(panels)} axial {projection} slabs of {step} slices ({step * vol.spacing[0]:.1f} mm), "
                      f"window {list(win)}, z {starts[0]} to {end}; outlines = registry lesions, dashed = candidates.{more} "
                      f"A finding at (y, x) in slab z0-z1: check it in `view` slices / montage, then `segment` it (a point, "
                      f"or a box for large or faint lesions; in aerated lung the click snaps along z to the nodule).",
                      [self._png(render.stack(panels, ncols=2), "sweep")])  # fmt: skip

    def _snap_to_nodule(self, vis, p, radius_mm=6.0):
        """A click from a MIP slab or a slightly-off click in aerated lung (< -400 HU): move it along z (+-6 mm,
        and +-1 voxel in-plane) to the brightest voxel, if that one is soft tissue. Returns (point, note)."""
        vol = self._vol(vis)
        if vol.data[tuple(p)] >= rules.AIR_HU:
            return p, ""
        dz = max(1, int(round(radius_mm / float(vol.spacing[0]))))
        z0, z1 = max(0, p[0] - dz), min(vol.shape[0], p[0] + dz + 1)
        y0, y1 = max(0, p[1] - 1), min(vol.shape[1], p[1] + 2)
        x0, x1 = max(0, p[2] - 1), min(vol.shape[2], p[2] + 2)
        sub = vol.data[z0:z1, y0:y1, x0:x1]
        k = np.unravel_index(int(np.argmax(sub)), sub.shape)
        q = [int(z0 + k[0]), int(y0 + k[1]), int(x0 + k[2])]
        if vol.data[tuple(q)] < rules.SNAP_MIN_HU:
            return p, ""
        return q, f" (click in aerated lung moved to the nodule at {q}, HU {vol.data[tuple(q)]:.0f})"

    def view_pair(self, lesion, visits=None, size_mm=80, window=None, coronal=True, sagittal=False,
                  center=None) -> Result:  # fmt: skip
        """The same lesion at several visits side by side (axial through its centre / anchor, + coronal, + sagittal).
        `center` {visit: [z, y, x]} centres a visit's panels on another spot (e.g. the registration point or a
        neighbouring structure) without changing the mask, to compare it with the lesion at the other visits."""
        l = self._les(lesion)
        center = {self._vis(k).index: p for k, p in (center or {}).items()}
        idx = visits or sorted({v.index for v in self.study.visits if v.index in l.findings} | set(center))
        vlist = [self._vis(v) for v in idx]
        if not vlist:
            raise AgentError(f"lesion {l.id} has no finding at any visit yet")
        planes = (0,) + ((1,) if coronal else ()) + ((2,) if sagittal else ())
        rows = {ax: [] for ax in planes}
        info = []
        for vis in vlist:
            vol = self._vol(vis)
            f = l.findings.get(vis.index)
            c = center.get(vis.index) or self._center(l.id, vis)
            if c is None:
                info.append(f"{vis.name}: no mask/point")
                continue
            c = self._point(c, vis)
            reg = (f.qa or {}).get("propagated_point") if f else None
            marks = ([render.Mark(f.point, "pt")] if f and f.point is not None else []) + \
                    ([render.Mark(reg, "reg")] if reg is not None else [])  # fmt: skip
            fov = self._fov(vis, c, size_mm)
            ov = self._overlay(vis, l.id, False, self._measure_lines(l.id, vis), marks)
            w = self._window(window, l.id)
            for ax in planes:
                ra, ca = plane_axes(ax)
                rows[ax].append(render.panel(vol.data, vol.spacing, self._ori(vis), ax, c[ax], fov[ra], fov[ca], ov, w,
                                             px=360, title=f"{vis.name} day {vis.day if vis.day is not None else '?'} "
                                             f"{('z', 'y', 'x')[ax]}={c[ax]}"
                                             + (" (given centre)" if vis.index in center else "")))  # fmt: skip
            info.append(f"{vis.name}: {self._lesion_summary(l, vis.index)} at {c} ({f.provenance if f else ''})"
                        + (" [panels centred on the given point]" if vis.index in center else "")
                        + (f"; registration point {[int(round(x)) for x in reg]}" if reg is not None else ""))
        panels = [p for ax in planes for p in rows[ax]]
        if not panels:
            raise AgentError("nothing to show")
        rgb = render.stack(panels, ncols=max(1, len(rows[0])))
        names = ", ".join(("axial", "coronal", "sagittal")[ax] for ax in planes)
        return Result(f"Lesion #{l.id} {l.name} across visits (rows: {names}; magenta + pt = the lesion's point, "
                      f"reg = the registration point):\n" + "\n".join(info), [self._png(rgb, "pair")])  # fmt: skip

    def inspect(self, visit=None, lesion=None, candidate=None, point=None, radius_mm=5.0) -> Result:
        """Numbers about a lesion / candidate mask or a point (read-only): components, extent, per-slice size, HU,
        organ, neighbours, QA."""
        vis = self._vis(visit)
        if point is not None:
            return self._inspect_point(vis, point, radius_mm)
        vol = self._vol(vis)
        sp = np.array(vol.spacing)
        lm = self.registry.label_map(vis)
        out, data = [], {}
        if lesion is not None:
            l = self._les(lesion)
            mask, title = self.registry.mask(l.id, vis).astype(bool), f"#{l.id} {l.name} at {vis.name}"
            f = l.findings.get(vis.index)
            if f is not None:
                data.update(status=f.status, provenance=f.provenance, point=f.point, qa=f.qa)
        elif candidate is not None:
            c = self._cand(vis, candidate)
            mask, title = self.cands[vis.index]["map"] == int(candidate), f"candidate c{candidate} at {vis.name}"
            data.update(state=c["state"], prompts=c["prompts"])
        else:
            raise AgentError("give `lesion`, `candidate` or `point`")
        if not mask.any():
            return Result(f"{title}: no mask" + (f" (status {data.get('status')}, point {data.get('point')})"
                                                  if "status" in data else ""), data=data)  # fmt: skip
        m = measure(mask, vol.spacing, vol.direction)
        box = crop_box(mask, margin=3)  # per-lesion work on the bounding-box crop
        lab, sizes, order = _components(mask[box])
        n = len(sizes)
        comps = [{"component": i + 1, "voxels": int(sizes[k]), "ml": round(float(sizes[k] * sp.prod() / 1000), 3),
                  "centroid": [int(round(x)) for x in center_of_mass_crop(lab == k + 1, box)]}
                 for i, k in enumerate(order[:8])]  # fmt: skip
        zs = np.where(mask.any((1, 2)))[0]
        per_slice = []
        for z in zs:
            ys, xs = np.where(mask[z])
            per_slice.append({"z": int(z), "voxels": int(len(ys)), "y": [int(ys.min()), int(ys.max())],
                              "x": [int(xs.min()), int(xs.max())]})  # fmt: skip
        if len(per_slice) > 24:  # keep the text short: every k-th slice plus the largest one
            k = int(np.ceil(len(per_slice) / 24))
            big = max(per_slice, key=lambda r: r["voxels"])
            per_slice = [r for i, r in enumerate(per_slice) if i % k == 0 or r is big]
        ring = ndi.binary_dilation(mask[box], iterations=2) & ~mask[box]
        own = self._les(lesion).id if lesion is not None else None
        neighbours = sorted(int(i) for i in np.unique(lm[box][ring]) if i and i != own)
        organ = self._organ_at(vis, comps[0]["centroid"])
        data.update(measurement={k: v for k, v in m.to_dict().items() if not k.endswith("_line")},
                    components=comps, n_components=int(n), z_range=[int(zs[0]), int(zs[-1])], per_slice=per_slice,
                    hu=_pct(vol.data[box][mask[box]]), hu_rim=_pct(vol.data[box][ring]), touching_lesions=neighbours, organ=organ)  # fmt: skip
        out.append(f"{title}: {int(mask.sum())} voxels {m.volume_ml:.2f} ml, {_ld_sa(m.to_dict(), True)}, "
                   f"z {zs[0]}..{zs[-1]}, organ at centre {organ or '?'}")  # fmt: skip
        out.append(f"HU inside {_pct_txt(data['hu'])}; 2-voxel rim outside {_pct_txt(data['hu_rim'])}")
        out.append(f"{n} component(s) (largest first): " + "; ".join(f"#{c['component']} {c['voxels']} vox at "
                                                                    f"{c['centroid']}" for c in comps))  # fmt: skip
        out.append("per slice (z: voxels, y-range, x-range): " + ", ".join(f"{r['z']}: {r['voxels']} {r['y']} {r['x']}"
                                                                          for r in per_slice))  # fmt: skip
        if neighbours:
            out.append(f"touches lesions {neighbours}")
        for k in ("status", "provenance", "point", "state", "prompts"):
            if data.get(k) not in (None, "", []):
                out.append(f"{k}: {data[k]}")
        if data.get("qa"):
            out.append("QA: " + json.dumps({k: v for k, v in data["qa"].items() if k != "organ_ts"}, default=str))
        return Result("\n".join(out), data=data)

    def _inspect_point(self, vis, point, radius_mm) -> Result:
        """HU at a point and in a ball around it; the lesion / candidate / organ containing it."""
        vol = self._vol(vis)
        sp = np.array(vol.spacing)
        p = self._point(point, vis)
        r = np.ceil(radius_mm / sp).astype(int)
        sl = tuple(slice(max(0, c - k), min(n, c + k + 1)) for c, k, n in zip(p, r, vol.shape))
        zz, yy, xx = np.ogrid[sl]
        ball = ((zz - p[0]) * sp[0]) ** 2 + ((yy - p[1]) * sp[1]) ** 2 + ((xx - p[2]) * sp[2]) ** 2 <= radius_mm ** 2
        hu = vol.data[sl][ball]
        lid = int(self.registry.label_map(vis)[tuple(p)])
        cmap = self.cands.get(vis.index, {}).get("map")
        cid = int(cmap[tuple(p)]) if cmap is not None else 0
        data = {"point": p, "hu_point": float(vol.data[tuple(p)]), "hu_ball": _pct(hu), "lesion": lid or None,
                "candidate": cid or None, "organ": self._organ_at(vis, p)}  # fmt: skip
        return Result(f"{vis.name} point {p}: HU {data['hu_point']:.0f}; within {radius_mm:g} mm: "
                      f"{_pct_txt(data['hu_ball'])}; inside lesion {('#%d' % lid) if lid else '-'}, candidate "
                      f"{('c%d' % cid) if cid else '-'}, organ {data['organ'] or '? (run totalsegmentator)'}",
                      data=data)  # fmt: skip

    def _organ_at(self, vis, p):
        om = self.organ_maps.get(vis.index)
        if om is None:
            return None
        k = int(om.total[tuple(int(round(c)) for c in p)])
        return f"{om.total_names[k]} ({om.group_of[k]})" if k else "outside TotalSegmentator organs"

    # ================================================================== phase 1: candidates
    def _cand(self, vis, cid):
        items = self.cands.get(vis.index, {}).get("items", {})
        if int(cid) not in items:
            raise AgentError(f"no candidate c{cid} at {vis.name}; run detect first (known: "
                             f"{sorted(items)})")  # fmt: skip
        return items[int(cid)]

    def detect_candidates(self, visit=None, prompts=None, min_voxels: int = rules.MIN_CANDIDATE_VOXELS) -> Result:
        """VoxTell zero-shot detection -> candidates (connected components of the union of all prompts), ADDED to the
        visit's candidate table: a component that overlaps an earlier candidate adds its prompts to that one, others
        become new c# — so repeated calls with more prompts never lose earlier candidates or decisions."""
        from .backends import voxtell

        vis = self._vis(visit)
        prompts = prompts or ["lesion", "tumor", "metastasis", "lymph node metastasis"]
        self.activity(f"VoxTell on {vis.name}: {len(prompts)} prompts", model="voxtell", visit=vis.index)
        t0 = time.time()
        masks = voxtell.predict(vis.path, list(prompts))
        self.n_model_calls += 1
        union = np.zeros(self._vol(vis).shape, bool)
        for m in masks.values():
            union |= m
        new = self._add_candidates(vis, union, "VoxTell", min_voxels)
        cur = self.cands[vis.index]
        cur["prompts"] = list(dict.fromkeys(cur.get("prompts", []) + list(prompts)))
        hit_ids = set()
        for p, m in masks.items():  # which prompts fired on which candidate
            for cid in np.unique(cur["map"][m]):
                if cid:
                    it = cur["items"][int(cid)]
                    if p not in it["prompts"]:
                        it["prompts"].append(p)
                    hit_ids.add(int(cid))
        for it in cur["items"].values():
            it["prompts"] = [q for q in it["prompts"] if q != "VoxTell"] or it["prompts"]
            it["support"] = len(it["prompts"])
        items = cur["items"]
        rows = [_cand_row(c) for c in sorted(items.values(), key=lambda c: c["id"]) if c["id"] in hit_ids]
        dt = time.time() - t0
        self.activity(f"VoxTell {vis.name}: {len(new)} new candidates in {dt:.0f} s", model="voxtell", visit=vis.index)
        ov = self.overview(vis, show_candidates=True)
        txt = (f"VoxTell on {vis.name} with prompts {list(prompts)}: {len(new)} new candidate(s), {len(hit_ids)} hit by "
               f"these prompts ({len(items)} in this visit's table so far; earlier ones and decisions are kept), {dt:.0f} s. "
               "Inspect each (view candidate=c#, mode slices or montage) and accept / reject it (review_candidates). "
               "VoxTell has false positives (vessels, bowel, cysts) and misses: check organ by organ.\n"
               + "\n".join(rows))  # fmt: skip
        return Result(txt, ov.images, {"candidates": [items[i] for i in sorted(hit_ids)]})

    def detect(self, visit=None, prompts=None, min_voxels: int = rules.MIN_CANDIDATE_VOXELS) -> Result:
        """VoxTell detection. Baseline: candidates c1, c2, ...; follow-up: only candidates not covered by tracked
        lesions, each mapped back to the baseline (new lesion vs missed baseline lesion)."""
        vis = self._vis(visit)
        if vis.index == self.study.baseline.index:
            return self.detect_candidates(vis, prompts, min_voxels)
        r = self.detect_candidates(vis, prompts, min_voxels)
        openc = [c for c in self.cands[vis.index]["items"].values() if c["state"] == "open"]
        txt, strip = self._follow_up_candidates(vis, openc)
        return Result(txt, r.images + strip, {"candidates": openc})

    def review_candidates(self, decisions, visit=None) -> Result:
        """Batch accept / reject. decisions: [{"candidate": 3, "accept": true, "name", "lesion_type", "organ", "role",
        "lesion", "notes"} | {"candidate": 4, "accept": false, "reason": "vessel"}]."""
        vis = self._vis(visit)
        if not decisions:
            raise AgentError("no decisions given")
        out, data, imgs = [], [], []
        for d in decisions:
            cid = d.get("candidate")
            if cid is None or "accept" not in d:
                raise AgentError(f"each decision needs `candidate` and `accept`: {d}")
            try:
                if d["accept"]:
                    r = self.accept_candidate(cid, vis, d.get("name", ""), d.get("lesion_type", "unknown"),
                                              d.get("organ", ""), d.get("role", "candidate"), d.get("notes", ""),
                                              d.get("lesion"))  # fmt: skip
                    data.append(r.data)
                    imgs += r.images
                else:
                    r = self.reject_candidate([cid], vis, d.get("reason", ""))
                out.append(r.text)
            except AgentError as e:
                out.append(f"c{cid}: ERROR {e}")
        return Result("\n".join(out), imgs, data={"accepted": data})

    def accept_candidate(self, candidate, visit=None, name="", lesion_type="unknown", organ="", role="candidate",
                         notes="", lesion=None) -> Result:  # fmt: skip
        """Turn a candidate into a registry lesion (or into the mask of an existing `lesion` at this visit)."""
        vis = self._vis(visit)
        c = self._cand(vis, candidate)
        if c["state"] not in ("open", "existing"):
            raise AgentError(f"candidate c{candidate} is already {c['state']}")
        mask = self.cands[vis.index]["map"] == int(candidate)
        parts, dropped = (self._split_parts(vis, mask) if lesion is None else ([mask], 0))
        texts, imgs, ids, meas = [], [], [], []
        for k, part in enumerate(parts):
            with self.lock:
                if lesion is not None:
                    l = self._les(lesion)
                else:
                    l = self._new_lesion(f"{name}_{k + 1}" if name and len(parts) > 1 else name, lesion_type, organ, role)
                info, _, m = self._write_mask(l, vis, part, notes=notes, status="present", provenance="voxtell",
                                              point=[round(x, 1) for x in (c["centroid"] if len(parts) == 1
                                                                            else object_center(part))])  # fmt: skip
            second, im = self._second_opinion(l.id, vis, part)
            imgs += im
            if vis.index == self.study.baseline.index and l.role == "new":  # present at baseline: tracked, not new
                l.role = "candidate"
                second += f"\n  #{l.id} now has a baseline mask: it is a (missed) baseline lesion, no longer 'new'"
            ids.append(l.id)
            meas.append(m)
            texts.append(f"lesion #{l.id} {l.name} at {vis.name}: {info['voxels']} voxels, "
                         f"LD {m['ld_mm']:.1f} mm, SA {m['sa_mm']:.1f} mm, SA max {m['sa_max_mm']:.1f} mm, "
                         f"{m['n_components']} component(s)" + (f"; overlaps lesions {info['overlaps_with']}" if info["overlaps_with"] else "")
                         + info["note"] + second)  # fmt: skip
        with self.lock:
            c["state"] = "accepted->" + ",".join(f"#{i}" for i in ids)
        self.changed([vis], ids, "accept")
        if len(parts) == 1:
            return Result(f"c{candidate} -> {texts[0]}", imgs, data={"lesion": ids[0], "measurement": meas[0]})
        head = (f"c{candidate} consists of {len(parts)} separate nodules/masses (each >= {rules.SPLIT_MIN_ML} ml"
                + (f"; {dropped} fragment(s) below dropped" if dropped else "") + "): one lesion each. Reject a part "
                "that is not a lesion with update_lesion role=rejected.")  # fmt: skip
        return Result(head + "\n" + "\n".join(f"  -> {t}" for t in texts), imgs,
                      data={"lesion": ids[0], "lesions": ids, "measurement": meas[0]})

    def _split_parts(self, vis, mask) -> tuple[list, int]:
        """A detector candidate made of several separate components (often a cluster of nodules) becomes one lesion per
        component of at least SPLIT_MIN_ML; smaller fragments are dropped. One component (or one large and only
        fragments): unchanged."""
        box = crop_box(mask)
        lab, n = ndi.label(mask[box], structure=np.ones((3, 3, 3)))
        if n < 2:
            return [mask], 0
        ml = ndi.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1)) * np.prod(self._vol(vis).spacing) / 1000
        keep = [i + 1 for i in np.argsort(-ml) if ml[i] >= rules.SPLIT_MIN_ML]
        if len(keep) < 2:
            return [mask], 0
        parts = []
        for i in keep:
            p = np.zeros_like(mask)
            p[box] = lab == i
            parts.append(p)
        return parts, int(n - len(keep))

    def _second_opinion(self, lid, vis, mask, source="VoxTell") -> tuple[str, list]:
        """Every model mask the agent relies on gets a second opinion: an nnInteractive click at the mask centre
        (~0.3 s) segments the same lesion independently. They DISAGREE when the long axes differ by more than
        max(SECOND_OPINION_MIN_MM, SECOND_OPINION_FRACTION) (core.rules); then both masks are rendered side by side and
        the agent must pick one. Leaks (click > LEAK_ML) and fragmented model masks (> FRAGMENTED_COMPONENTS) are
        not compared."""
        try:
            p = object_center(mask)
            m2 = self._gpu(lambda: self._click(vis, p))
            if not m2.any():
                return "", []
            vol = self._vol(vis)
            mm = measure(m2, vol.spacing, vol.direction)
            ld0 = self.registry.lesions[lid].findings[vis.index].measurement.get("ld_mm", 0)
            r = m2.sum() / max(int(mask.sum()), 1)
            ml = float(m2.sum() * np.prod(vol.spacing) / 1000)
            n_vt = ndi.label(mask[crop_box(mask)])[1]
            f = self.registry.finding(lid, vis)
            f.qa["second_opinion"] = {"tool": "nninteractive", "point": p, "voxels": int(m2.sum()), "ld_mm": mm.ld_mm,
                                      "volume_ratio": round(float(r), 2), "voxtell_components": int(n_vt)}  # fmt: skip
            if ml > rules.LEAK_ML:
                return f"\n  second opinion: the nnInteractive click leaked ({ml:.0f} ml); {source} mask kept", []
            if n_vt > rules.FRAGMENTED_COMPONENTS:
                return (f"\n  second opinion not comparable: the {source} mask has {n_vt} components (cluster?); "
                        f"check whether this is one lesion"), []  # fmt: skip
            if abs(mm.ld_mm - ld0) <= max(rules.SECOND_OPINION_MIN_MM, rules.SECOND_OPINION_FRACTION * ld0):
                return f"\n  second opinion (nnInteractive click): LD {mm.ld_mm:.1f} mm — agrees", []
            f.qa["second_opinion"]["disagrees"] = True
            img = self._compare_masks(vis, mask, m2, f"{source}  LD {ld0:.1f} mm", f"nnInteractive click  LD {mm.ld_mm:.1f} mm")
            return (f"\n  ! SECOND OPINION DISAGREES for #{lid} at {vis.name} (LD {ld0:.1f} vs {mm.ld_mm:.1f} mm, image: "
                    f"left {source}, right nnInteractive). Decide which mask covers the lesion. Keep {source}: do "
                    f"nothing. Take the click: segment lesion={lid} visit={vis.index} point={p} method=nninteractive. "
                    f"Neither: fix it with refine_lesion (negative click on the extra part, positive on what is missing)."),\
                [img]  # fmt: skip
        except Exception as e:  # noqa: BLE001
            from .backends import models

            if models.is_oom(e) or "cuda" in str(e).lower():
                raise  # the GPU retry wrapper handles it; everything else only loses the second opinion
            return f"\n  (second opinion unavailable: {type(e).__name__})", []

    def _empty_track_opinion(self, lid, fv, tv, p) -> tuple[str, list]:
        """LongiTrack returned an empty mask: an nnInteractive click at the registration point segments whatever is
        there, for the agent to compare."""
        try:
            m2 = self._gpu(lambda: self._click(tv, p))
            vol = self._vol(tv)
            ml = float(m2.sum() * np.prod(vol.spacing) / 1000)
            if not m2.any() or ml > rules.LEAK_ML:
                return (f"\n  second opinion: an nnInteractive click at the registration point {p} finds "
                        f"{'nothing' if not m2.any() else f'no bounded structure (leak, {ml:.0f} ml)'}"), []  # fmt: skip
            mm = measure(m2, vol.spacing, vol.direction)
            v0 = self.registry.mask(lid, fv).sum() * np.prod(self._vol(fv).spacing) / 1000
            f = self.registry.finding(lid, tv)
            f.qa["second_opinion"] = {"tool": "nninteractive", "point": p, "voxels": int(m2.sum()), "ld_mm": mm.ld_mm,
                                      "volume_ratio": round(float(ml / v0), 2) if v0 else None}  # fmt: skip
            return (f"\n  ! LongiTrack found nothing, but an nnInteractive click at the registration point {p} segments "
                    f"a structure with LD {mm.ld_mm:.1f} mm ({ml:.2f} ml, x{ml / v0:.2f} the {fv.name} volume). Before "
                    f"searching elsewhere, compare it with the lesion: view_pair lesion={lid} center={{\"{tv.index}\": "
                    f"{p}}}. The same lesion -> segment lesion={lid} visit={tv.index} point={p} method=nninteractive."), []
        except Exception as e:  # noqa: BLE001
            from .backends import models

            if models.is_oom(e) or "cuda" in str(e).lower():
                raise  # the GPU retry wrapper handles it; everything else only loses the second opinion
            return f"\n  (second opinion unavailable: {type(e).__name__})", []

    def _compare_masks(self, vis, m1, m2, t1, t2):
        """Two axial panels through the larger mask's centre: m1 (left) and m2 (right)."""
        vol = self._vol(vis)
        big = m2 if m2.sum() > m1.sum() else m1
        c = [int(round(x)) for x in ndi.center_of_mass(big)]
        fov = self._fov(vis, c, None, bbox_of(big | m1))
        w = self._window(None)
        panels = [render.panel(vol.data, vol.spacing, self._ori(vis), 0, c[0], fov[1], fov[2],
                               render.Overlay(current=m.astype(bool)), w, px=360, title=f"{t} (z={c[0]})")
                  for m, t in ((m1, t1), (m2, t2))]  # fmt: skip
        return self._png(render.stack(panels, ncols=2), "second_opinion")

    def reject_candidate(self, candidates, visit=None, reason="") -> Result:
        vis = self._vis(visit)
        ids = candidates if isinstance(candidates, (list, tuple)) else [candidates]
        for cid in ids:
            c = self._cand(vis, cid)
            c["state"], c["reason"] = "rejected", reason
        self.changed([vis], what="candidates")
        return Result(f"rejected {['c%d' % int(i) for i in ids]} at {vis.name}" + (f" ({reason})" if reason else ""))

    def segment(self, point=None, visit=None, lesion=None, method="auto", name="", lesion_type="unknown", organ="",
                role="candidate", notes="", box=None, preview=False) -> Result:  # fmt: skip
        """Segment a lesion from one point (default: the lesion's point at this visit, e.g. the propagated one).
        method 'longitrack': LongiTrack with the lesion's mask at another visit as longitudinal context (tracking);
        'nninteractive': one positive click, no context; 'auto': LongiTrack when context exists, else nnInteractive.
        `box` [[z, y0, x0], [z, y1, x1]] (inclusive corners in one slice) instead of a point: an nnInteractive box
        prompt (constrains the extent where a click leaks or stops short); `preview` draws the box without segmenting,
        and after a box a fit check reports where the prediction sticks out of the box or leaves it empty."""
        vis = self._vis(visit)
        l = self._les(lesion) if lesion is not None else None
        if box is not None:
            return self._segment_box(vis, l, box, method, preview, name, lesion_type, organ, role, notes)
        given = point is not None
        if point is None:
            f0 = l.findings.get(vis.index) if l is not None else None
            if f0 is None or f0.point is None:
                raise AgentError("give `point` (no stored point for this lesion at this visit)")
            point = f0.point
        p = self._point(point, vis)
        has_ctx = l is not None and any(ov.index != vis.index and self.registry.mask(l.id, ov).any()
                                        for ov in self.study.visits)  # fmt: skip
        if method == "auto":
            method = "longitrack" if has_ctx else "nninteractive"
        if method == "longitrack" and has_ctx:
            return self.track(l.id, vis, p if given else None)
        t0 = time.time()
        if method == "longitrack":
            from .backends import longitrack

            self.activity(f"LongiTrack single click at {vis.name} {p}", model="longitrack", visit=vis.index)
            mask, info = longitrack.segment(vis.path, p, None, None)
        elif method == "nninteractive":
            snap = ""
            if given:
                p, snap = self._snap_to_nodule(vis, p)
            s = self._ni(vis)
            s.start(None)
            self.activity(f"nnInteractive point at {vis.name} {p}", model="nninteractive", visit=vis.index)
            mask = s.point(p, True)
            info = {"mode": "nninteractive_point", "snap": snap}
        else:
            raise AgentError("method must be 'longitrack' or 'nninteractive'")
        self.n_model_calls += 1
        if not mask.any():
            return Result(f"{method} returned an EMPTY mask at {p} ({vis.name}). The point is probably not on a lesion: "
                          "look at a view there and try again, or use the other method.", data={"empty": True})  # fmt: skip
        return self._store_segmentation(vis, l, mask, p, method, info, t0, has_ctx, name, lesion_type, organ, role,
                                        notes, extra=info.get("snap", ""))

    def _store_segmentation(self, vis, l, mask, p, method, info, t0, has_ctx, name, lesion_type, organ, role, notes,
                            extra="") -> Result:  # fmt: skip
        with self.lock:
            undo = None if l is None else self.registry.mask(l.id, vis)
            if l is None:
                l = self._new_lesion(name, lesion_type, organ, role)
            ov, f, m = self._write_mask(l, vis, mask, undo, notes, status="present", provenance=method, point=p)
        inside = bool(mask[tuple(p)])
        qa_txt = ""
        if has_ctx:
            qa = self._track_qa(l.id, self._context_visit(l.id, vis), vis, p, mask)
            f.qa.update(qa)
            qa_txt = f"; QA flags vs {self._context_visit(l.id, vis).name}: {qa['flags'] or 'none'}"
        self.changed([vis], [l.id], "segment")
        return Result(f"lesion #{l.id} {l.name} at {vis.name} ({method}, {info.get('mode')}, {time.time() - t0:.1f} s): "
                      f"{ov['voxels']} voxels {m['volume_ml']:.2f} ml, {_ld_sa(m, True)}, {m['n_components']} "
                      f"component(s), point inside mask: {inside}"
                      + (f"; overlaps lesions {ov['overlaps_with']}" if ov["overlaps_with"] else "") + qa_txt + extra + ov["note"],
                      data={"lesion": l.id, "measurement": m})  # fmt: skip

    def _segment_box(self, vis, l, box, method, preview, name, lesion_type, organ, role, notes) -> Result:
        """nnInteractive box prompt (a fresh segmentation of a new lesion, or of `l` at this visit)."""
        if method not in ("auto", "nninteractive"):
            raise AgentError("a box is an nnInteractive prompt: use method 'nninteractive' (or 'auto')")
        if not isinstance(box, (list, tuple)) or len(box) != 2:
            raise AgentError("box is two inclusive corners [[z, y0, x0], [z, y1, x1]] in one slice")
        c1, c2 = self._point(box[0], vis), self._point(box[1], vis)
        try:
            _, axis = raster.bbox_from_corners(c1, c2)
        except ValueError as e:
            raise AgentError(str(e)) from e
        lo, hi = [min(a, b) for a, b in zip(c1, c2)], [max(a, b) for a, b in zip(c1, c2)]
        sp = self._vol(vis).spacing
        ra, ca = plane_axes(axis)
        span = ", ".join(f"{'zyx'[a]} {lo[a]}-{hi[a]} ({hi[a] - lo[a] + 1} voxels, {(hi[a] - lo[a] + 1) * sp[a]:.0f} mm)"
                         for a in (ra, ca))  # fmt: skip
        plane = f"{'zyx'[axis]} = {lo[axis]}"
        if preview:
            vol = self._vol(vis)
            fov = render.fov_around(vol.shape, bbox=[[a, b + 1] for a, b in zip(lo, hi)], margin_frac=0.6, min_size=48)
            ov = self._overlay(vis, l.id if l is not None else None)
            ov.boxes = [render.Box(c1, c2, "new box")]
            rgb = render.panel(vol.data, vol.spacing, self._ori(vis), axis, lo[axis], fov[ra], fov[ca], ov,
                               self._window(None, l.id if l is not None else None), px=520,
                               title=f"{vis.name} box preview ({plane})")  # fmt: skip
            return Result(f"PREVIEW, nothing segmented: box in {plane} covering {span} (green). Each edge should lie "
                          "just outside the lesion (a few voxels) and the box must not cut it anywhere. If it is off, "
                          "preview corrected corners; if it fits, call segment with the same box and preview=false.",
                          [self._png(rgb, "box_preview")])  # fmt: skip
        t0 = time.time()
        s = self._ni(vis)
        s.start(None)
        self.activity(f"nnInteractive box at {vis.name} {plane}", model="nninteractive", visit=vis.index)
        mask = s.bbox(c1, c2, True)
        self.n_model_calls += 1
        if not mask.any():
            return Result(f"nninteractive returned an EMPTY mask for the box in {plane} ({vis.name}): check the "
                          "corners on a view (preview=true).", data={"empty": True})  # fmt: skip
        p = object_center(mask)
        has_ctx = l is not None and any(ov.index != vis.index and self.registry.mask(l.id, ov).any()
                                        for ov in self.study.visits)  # fmt: skip
        fit = "\n  " + _box_fit(mask, axis, lo[axis], lo, hi)
        return self._store_segmentation(vis, l, mask, p, "nninteractive", {"mode": "nninteractive_box"}, t0, has_ctx,
                                        name, lesion_type, organ, role, notes, fit)  # fmt: skip

    @staticmethod
    def _gpu(fn):
        """Run a model call with the tools' GPU retry. For calls after a tool has already changed state (a mask written
        before its second opinion, the earlier lesions of a batch): only this call is repeated on a busy GPU, not the
        whole tool."""
        from .tools import _with_gpu_retry

        return _with_gpu_retry(fn)

    def _click(self, vis, p):
        """One positive nnInteractive click on an empty session (second opinions)."""
        s = self._ni(vis)
        s.start(None)
        return s.point(p, True)

    def _ni(self, vis):
        from .backends import models

        s = models.get("nninteractive")
        vol = self._vol(vis)
        s.set_image(str(vis.path), vol.data, vol.spacing)
        return s

    def refine_lesion(self, lesion, visit=None, points=None, scribbles=None, undo=False) -> Result:
        """nnInteractive correction of an existing mask. points: [{"point": [z,y,x], "positive": bool}];
        scribbles: [{"points": [[z,y,x], ...] (one slice in any plane: same z, y or x), "positive": bool}], drawn 2 px
        wide in that plane. Each prompt is reported with where it lies relative to the current mask (a negative prompt
        outside the mask removes nothing). undo=True reverts the last mask change."""
        if undo:
            return self.undo_lesion(lesion, visit)
        vis = self._vis(visit)
        l = self._les(lesion)
        cur = self.registry.mask(l.id, vis)
        if not cur.any():
            raise AgentError(f"lesion {l.id} has no mask at {vis.name}; use segment")
        # validate every prompt before the model runs
        pts = [(self._point(pr["point"], vis), bool(pr.get("positive", True))) for pr in points or []]
        scs = []
        for sc in scribbles or []:
            q = [self._point(p, vis) for p in sc["points"]]
            try:
                raster.infer_plane(q)
            except ValueError as e:
                raise AgentError(str(e)) from e
            if len({tuple(x) for x in q}) == 1:
                raise AgentError("the scribble points are all the same; give at least two different points")
            scs.append((q, bool(sc.get("positive", True))))
        if not pts and not scs:
            raise AgentError("give points and/or scribbles (or undo=true)")
        s = self._ni(vis)
        s.start(cur)
        before = int(cur.sum())
        mask = cur
        vol = self._vol(vis)
        lab = self.registry.label_map(vis)
        self.activity(f"nnInteractive refine #{l.id} at {vis.name}: {len(pts)} points, {len(scs)} scribbles",
                      model="nninteractive", visit=vis.index)  # fmt: skip
        diag = []
        for p, pos in pts:
            prev = mask
            mask = s.point(p, pos)
            diag.append(f"{'+' if pos else '-'}point {p}: {'inside' if prev[tuple(p)] else 'outside'} the mask, "
                        f"HU {float(vol.data[tuple(p)]):.0f}; {_delta(prev, mask)}"
                        + _prompt_warning(pos, bool(prev[tuple(p)]), 1.0 if prev[tuple(p)] else 0.0))  # fmt: skip
        for q, pos in scs:
            prev = mask
            mask, line = s.scribble_line(q, pos)
            idx = tuple(line.T)
            inside = float(np.mean(prev[idx])) if len(line) else 0.0
            hu = np.percentile(vol.data[idx], (5, 50, 95)) if len(line) else (np.nan,) * 3
            plane, index = raster.infer_plane(q)
            other = sorted({int(i) for i in np.unique(lab[idx])} - {0, l.id})
            diag.append(f"{'+' if pos else '-'}scribble in {'zyx'[plane]} = {index} ({len(line)} px): {inside:.0%} "
                        f"inside the mask, HU along it {hu[1]:.0f} (5-95%: {hu[0]:.0f} to {hu[2]:.0f})"
                        + (f", crosses lesions {other}" if other else "") + f"; {_delta(prev, mask)}"
                        + _prompt_warning(pos, inside > 0, inside))  # fmt: skip
        self.n_model_calls += 1
        prov = (self.registry.finding(l.id, vis).provenance + "+nninteractive").strip("+")
        info, _, m = self._write_mask(l, vis, mask, undo=cur, provenance=prov)
        self.changed([vis], [l.id], "refine")
        return Result(f"lesion #{l.id} at {vis.name}: {before} -> {int(mask.sum())} voxels; {_ld_sa(m)}, "
                      f"{m['n_components']} component(s). `undo=true` reverts.\n  "
                      + "\n  ".join(diag) + info["note"], data={"measurement": m})  # fmt: skip

    def undo_lesion(self, lesion, visit=None) -> Result:
        vis = self._vis(visit)
        l = self._les(lesion)
        prev = self._undo.pop((l.id, vis.index), None)
        if prev is None:
            raise AgentError("nothing to undo for this lesion/visit")
        info, _, _ = self._write_mask(l, vis, prev[0], status=prev[1])
        self.changed([vis], [l.id], "undo")
        return Result(f"lesion #{l.id} at {vis.name} restored ({int(prev[0].sum())} voxels)" + info["note"])

    def set_lesion_info(self, lesion, name=None, lesion_type=None, organ=None, role=None, notes=None,
                        confidence=None) -> Result:  # fmt: skip
        l = self._les(lesion)
        if name:
            l.name = name
        if lesion_type:
            l.lesion_type = _type(lesion_type)
        if organ is not None:
            l.organ = organ
        if role:
            l.role = _role(role)
        if notes:
            l.notes = (l.notes + " | " + notes).strip(" |")
        if confidence:
            l.confidence = confidence
        self.changed(lesions=[l.id], what="info")
        return Result(f"#{l.id}: name={l.name} type={l.lesion_type} organ={l.organ or '?'} role={l.role}")

    def delete_lesion(self, lesion) -> Result:
        l = self._les(lesion)
        self.registry.delete(l.id)
        self.changed(self.study.visits, [l.id], "delete")
        return Result(f"deleted lesion #{l.id} {l.name}")

    def merge_lesions(self, keep, absorb, visit) -> Result:
        vis = self._vis(visit)
        self.registry.merge(self._les(keep).id, self._les(absorb).id, vis)
        m = self.registry.remeasure(keep, vis)
        self.changed([vis], [keep, absorb], "merge")
        return Result(f"at {vis.name}: #{absorb} merged into #{keep} (RECIST: coalesced lesions are measured as one, "
                      f"the absorbed one counts 0). #{keep} now LD {m['ld_mm']:.1f} mm.")  # fmt: skip

    EDIT_OPS = ("keep_largest", "fill_holes", "clip", "subtract", "remove_component")

    def edit_mask(self, lesion, visit=None, ops=None) -> Result:
        """Deterministic mask edits, applied in order: {"op": "keep_largest"} (largest 3D component),
        {"op": "fill_holes"} (3D and per axial slice), {"op": "clip", "axis": 0|1|2, "min": i, "max": j} (keep only
        indices min..max along the axis), {"op": "subtract", "lesion": id} (remove another lesion's voxels),
        {"op": "remove_component", "component": k} (component numbers as listed by `inspect`)."""
        vis = self._vis(visit)
        l = self._les(lesion)
        cur = self.registry.mask(l.id, vis).astype(bool)
        if not cur.any():
            raise AgentError(f"lesion {l.id} has no mask at {vis.name}")
        m = cur.copy()
        done = []
        for o in ops or []:
            op = o.get("op")
            if op == "keep_largest":
                m = _largest_component(m)
            elif op == "fill_holes":
                m = ndi.binary_fill_holes(m)
                for z in np.where(m.any((1, 2)))[0]:
                    m[z] = ndi.binary_fill_holes(m[z])
            elif op == "clip":
                ax = int(o.get("axis", 0))
                lo, hi = int(o.get("min", 0)), int(o.get("max", m.shape[ax] - 1))
                keep = np.zeros(m.shape[ax], bool)
                keep[max(lo, 0):hi + 1] = True
                m &= keep.reshape([-1 if k == ax else 1 for k in range(3)])
            elif op == "subtract":
                m &= ~self.registry.mask(self._les(o["lesion"]).id, vis).astype(bool)
            elif op == "remove_component":
                lab, sizes, order = _components(m)  # numbered as in `inspect`
                k = int(o["component"])
                if not 1 <= k <= len(sizes):
                    raise AgentError(f"component {k} does not exist (mask has {len(sizes)})")
                m &= lab != order[k - 1] + 1
            else:
                raise AgentError(f"unknown op {op!r}; ops: {self.EDIT_OPS}")
            done.append(op)
        if not m.any():
            raise AgentError("the edits would remove the whole mask; nothing changed (use update_lesion status=absent)")
        prov = (self.registry.finding(l.id, vis).provenance + "+edit").strip("+")
        _, _, meas = self._write_mask(l, vis, m, undo=cur, provenance=prov)
        self.changed([vis], [l.id], "edit")
        return Result(f"#{l.id} at {vis.name} after {done}: {int(cur.sum())} -> {int(m.sum())} voxels, "
                      f"{_ld_sa(meas)}, {meas['n_components']} component(s). refine_lesion undo=true reverts.",
                      data={"measurement": meas})  # fmt: skip

    def update_lesion(self, lesion, visit=None, name=None, lesion_type=None, organ=None, role=None, notes=None,
                      confidence=None, status=None, merged_into=None, delete=False) -> Result:  # fmt: skip
        """Registry edits: lesion info (name, type, organ, role, notes, confidence), its status at `visit`
        (present | too_small | absent | merged | not_assessed; merged needs `merged_into`), or delete it."""
        if delete:
            return self.delete_lesion(lesion)
        out = []
        if any(x is not None for x in (name, lesion_type, organ, role, notes, confidence)):
            out.append(self.set_lesion_info(lesion, name, lesion_type, organ, role, notes, confidence).text)
        if merged_into is not None or status == "merged":
            if merged_into is None or visit is None:
                raise AgentError("status merged needs `visit` and `merged_into` (the lesion it coalesced with)")
            out.append(self.merge_lesions(merged_into, lesion, visit).text)
        elif status is not None:
            if visit is None:
                raise AgentError("`status` needs `visit`")
            out.append(self.set_status(lesion, visit, status, notes or "").text)
        if not out:
            raise AgentError("nothing to update")
        return Result("\n".join(out))

    # ================================================================== organs
    def assign_organs(self, visit=None, lesions=None, overwrite=False) -> Result:
        from .backends import organs

        vis = self._vis(visit)
        if vis.index not in self.organ_maps:
            self.activity(f"TotalSegmentator (total fast + trunk_cavities) on {vis.name}", model="totalsegmentator",
                          visit=vis.index)  # fmt: skip
            t0 = time.time()
            self.organ_maps[vis.index] = organs.OrganMaps(vis.path, self.cache_dir / "totalseg", key=vis.case_id)
            self.n_model_calls += 1
            self.activity(f"TotalSegmentator {vis.name} done in {time.time() - t0:.0f} s", model="totalsegmentator")
        om = self.organ_maps[vis.index]
        lm = self.registry.label_map(vis)
        ids = [self._les(i).id for i in lesions] if lesions else [i for i in self.registry.lesions if (lm == i).any()]
        rows = []
        sp = self._vol(vis).spacing
        for i in ids:
            m = lm == i
            if not m.any():
                continue
            a = om.assign(m, sp, lm)
            l = self.registry.lesions[i]
            f = self.registry.finding(i, vis)
            f.qa["organ_ts"] = asdict(a)
            if overwrite or not l.organ:
                l.organ = a.organ if a.organ not in ("none", "vessel", "muscle") else l.organ
            rows.append(f"#{i} {l.name}: {a.organ} ({a.source}; inside {dict(list(a.inside.items())[:3])}) "
                        f"extra_organ={a.extra_organ} cavity={a.cavity} -> organ set to '{l.organ or '?'}'")  # fmt: skip
        self.changed([vis], ids, "organs")
        return Result("Organ assignment (TotalSegmentator). extra_organ=True means the lesion is not inside a "
                      "parenchymal organ: typical for lymph nodes, peritoneal and soft-tissue lesions — decide node vs "
                      "not from the images (shape, fatty hilum, along vessels). 'vessel'/'muscle'/'none' never become "
                      "the organ; set it with update_lesion.\n" + "\n".join(rows))  # fmt: skip

    # ================================================================== phase 2: follow-up
    def propagate(self, to_visit, lesions=None, from_visit=None) -> Result:
        """uniGradICON registration from_visit -> to_visit; stores the propagated point per lesion at to_visit."""
        from .backends import longitrack

        fv, tv = self._vis(from_visit), self._vis(to_visit)
        if fv.index == tv.index:
            raise AgentError("from_visit and to_visit are the same")
        ids = [self._les(i).id for i in lesions] if lesions else [
            l.id for l in self.registry.lesions.values() if l.role != "rejected" and self.registry.mask(l.id, fv).any()]
        pts = [self._center(i, fv) for i in ids]
        self.activity(f"uniGradICON registration {fv.name} -> {tv.name} ({len(ids)} points)", model="registration",
                      visit=tv.index)  # fmt: skip
        t0 = time.time()
        res, note = self._gpu(lambda: longitrack.propagate_with_crop(self.study, fv, tv, pts, self.cache_dir / "reg_crops"))
        self.n_model_calls += 1
        rows = []
        for i, p, r in zip(ids, pts, res):
            f = self.registry.finding(i, tv)
            if r.followup_point is None:
                rows.append(f"#{i}: propagation failed ({r.error})")
                continue
            if f.provenance in ("", "registration"):
                f.point, f.provenance = r.followup_point, "registration"
                f.status = "not_assessed"
            f.qa["propagated_point"] = r.followup_point
            f.qa["propagation_out_of_bounds"] = r.out_of_bounds
            rows.append(f"#{i} {self.registry.lesions[i].name}: {fv.name} {p} -> {tv.name} {r.followup_point}"
                        + (" (OUT OF FIELD OF VIEW: lesion may not be scanned)" if r.out_of_bounds else ""))  # fmt: skip
        self.activity(f"registration done in {time.time() - t0:.0f} s", model="registration")
        self.changed([tv], ids, "propagate")
        return Result(f"Propagated {len(ids)} lesions {fv.name} -> {tv.name} in {time.time() - t0:.0f} s{note}.\n"
                      + "\n".join(rows) + "\n")  # fmt: skip

    def track(self, lesion, to_visit, point=None, from_visit=None) -> Result:
        """LongiTrack: segment the lesion at to_visit, using its mask at from_visit as longitudinal context and
        `point` (default: the propagated point) as the follow-up prompt. Returns QA numbers."""
        from .backends import longitrack

        l = self._les(lesion)
        tv = self._vis(to_visit)
        fv = self._vis(from_visit) if from_visit is not None else self._context_visit(l.id, tv)
        bl_pt = self._center(l.id, fv)
        if bl_pt is None:
            raise AgentError(f"lesion {l.id} has no mask at {fv.name}")
        f = self.registry.finding(l.id, tv)
        p = self._point(point if point is not None else f.point, tv) if (point is not None or f.point is not None) else None
        if p is None:
            raise AgentError(f"no point for lesion {l.id} at {tv.name}: run propagate first or give `point`")
        self.activity(f"LongiTrack #{l.id} {fv.name}->{tv.name} at {p}", model="longitrack", visit=tv.index)
        t0 = time.time()
        # scans with a short z-extent: most of LongiTrack's crop would be zero padding -> mirror padding instead
        # (longitrack.segment pad="reflect"); all other scans keep the model's own zero padding
        thin = min(self._vol(v).shape[0] * self._vol(v).spacing[0] for v in (fv, tv)) < rules.THIN_SCAN_MM
        mask, _ = self._gpu(lambda: longitrack.segment(tv.path, p, fv.path, bl_pt, pad="reflect" if thin else "zero"))
        self.n_model_calls += 1
        with self.lock:
            prev = self.registry.mask(l.id, tv)
            ovl, f, m = self._write_mask(l, tv, mask, undo=prev if prev.any() else None, point=p,
                                         provenance="longitrack" if point is None else "longitrack(agent point)",
                                         status="present" if mask.any() else "not_assessed")  # fmt: skip
        qa = self._track_qa(l.id, fv, tv, p, mask)
        f.qa.update(qa)
        second, imgs = self._second_opinion(l.id, tv, mask, "LongiTrack") if mask.any() else self._empty_track_opinion(l.id, fv, tv, p)
        self.changed([tv], [l.id], "track")
        flags = qa["flags"]
        return Result(f"#{l.id} {l.name} at {tv.name} ({time.time() - t0:.1f} s): {m['n_voxels']} voxels "
                      f"{m['volume_ml']:.2f} ml (x{qa['volume_ratio']} vs {fv.name}), {_ld_sa(m, True)}; QA flags: "
                      f"{flags or 'none'}"
                      + (f"; overlaps lesions {ovl['overlaps_with']}" if ovl["overlaps_with"] else "") + ovl["note"] + second,
                      imgs, data={"qa": qa, "measurement": m})  # fmt: skip

    def _latest_mask_visit(self, lid, tv):
        """The latest visit before `tv` where the lesion has a mask, or None."""
        return next((v for v in reversed(self.study.visits) if v.index < tv.index and self.registry.mask(lid, v).any()),
                    None)  # fmt: skip

    def _context_visit(self, lid, tv):
        prev = self._latest_mask_visit(lid, tv)
        if prev is not None:
            return prev
        other = [v for v in self.study.visits if v.index != tv.index and self.registry.mask(lid, v).any()]
        if not other:
            raise AgentError(f"lesion {lid} has no mask at any other visit to track from")
        return other[0]

    def _track_qa(self, lid, fv, tv, p, mask) -> dict:
        sp = np.array(self._vol(tv).spacing)
        prev = self.registry.mask(lid, fv)
        v0 = prev.sum() * np.prod(self._vol(fv).spacing)
        v1 = mask.sum() * np.prod(sp)
        flags = []
        ratio = round(float(v1 / v0), 2) if v0 else None
        if not mask.any():
            flags.append("empty_mask")
        else:
            if ratio is not None and (ratio > rules.VOLUME_RATIO_MAX or ratio < rules.VOLUME_RATIO_MIN):
                flags.append(f"volume_change_x{ratio}")
            box = crop_box(mask, margin=3)  # per-lesion work on the bounding-box crop
            sub = mask[box]
            c = np.array(center_of_mass_crop(sub, box))
            off = float(np.linalg.norm((c - np.array(p)) * sp))
            if not mask[tuple(p)]:
                flags.append(f"point_not_in_mask(centroid {off:.0f} mm away)")
            n = ndi.label(sub)[1]
            if n > 1:
                flags.append(f"{n}_components")
            others = self.registry.label_map(tv)[box]
            touch = sorted(int(i) for i in np.unique(others[ndi.binary_dilation(sub, iterations=2)]) if i and i != lid)
            if touch:
                flags.append(f"touches_lesions_{touch}")
        return {"volume_ratio": ratio, "flags": flags, "point": list(map(int, p))}

    def track_all(self, to_visit, from_visit=None, lesions=None) -> Result:
        """propagate + track every baseline lesion (not rejected), or only `lesions`, to `to_visit`; QA table. Without
        `from_visit`, a lesion without a baseline mask (new at an earlier follow-up) is registered and tracked from its
        latest earlier visit with a mask."""
        tv, fv = self._vis(to_visit), self._vis(from_visit)
        sel = None if not lesions else [self._les(i).id for i in lesions]
        groups, skipped = {}, []  # registration source visit -> lesion ids (None: all with a mask there)
        if sel is None:
            groups[fv.index] = None
            if from_visit is None:
                for l in self.registry.lesions.values():
                    if l.role != "rejected" and not self.registry.mask(l.id, fv).any():
                        src = self._latest_mask_visit(l.id, tv)
                        if src is not None:
                            groups.setdefault(src.index, []).append(l.id)
        else:
            for i in sel:
                src = fv if self._center(i, fv) is not None else (
                    self._latest_mask_visit(i, tv) if from_visit is None else None)  # fmt: skip
                if src is None:
                    skipped.append(i)
                else:
                    groups.setdefault(src.index, []).append(i)
        rows, imgs = [self.propagate(tv, lesions=ids, from_visit=v).text for v, ids in groups.items()], []
        rows += [f"#{i}: no mask at {fv.name if from_visit is not None else 'an earlier visit'} to track from"
                 for i in skipped] + [""]  # fmt: skip
        for l in sorted(self.registry.lesions.values(), key=lambda l: l.id):
            if sel is not None and (l.id not in sel or l.id in skipped):
                continue
            if l.role == "rejected" or tv.index not in l.findings or l.findings[tv.index].point is None:
                continue
            if l.role == "new" and not any(v < tv.index and self.registry.mask(l.id, v).any() for v in l.findings):
                continue  # new at this very visit: nothing earlier to track from
            try:
                r = self.track(l.id, tv, from_visit=from_visit)
                rows.append(r.text)
                imgs += r.images
            except AgentError as e:
                rows.append(f"#{l.id}: {e}")
        return Result("\n".join(rows) + "\nLook at every lesion with QA flags or a disagreeing second opinion "
                      "(view_pair) before accepting it; an empty mask can mean the lesion resolved (update_lesion "
                      "status=absent) or a wrong point (segment lesion=# point=[...] with a corrected point).", imgs)  # fmt: skip

    def set_status(self, lesion, visit, status, notes="") -> Result:
        l = self._les(lesion)
        vis = self._vis(visit)
        if status not in STATUSES:
            raise AgentError(f"status must be one of {STATUSES}")
        f = self.registry.finding(l.id, vis)
        f.status, f.notes = status, (notes or f.notes)
        if status == "absent":
            self.registry.set_mask(l.id, vis, np.zeros(self._vol(vis).shape, bool))
            f.measurement = {"ld_mm": 0.0, "sa_mm": 0.0, "sa_max_mm": 0.0, "n_voxels": 0}
        note = self._no_empty_present(l, vis, f) if not self.registry.mask(l.id, vis).any() else ""
        self.changed([vis], [l.id], "status")
        return Result(f"#{l.id} at {vis.name}: status {f.status}" + (" (counts 5 mm in the sum)" if status == "too_small" else "")
                      + note)

    def _map_to_baseline(self, vis, pts, what: str) -> list:
        """Registration vis -> baseline of points (a point None where it fails)."""
        from .backends import longitrack

        bl = self.study.baseline
        self.activity(f"registration {vis.name} -> {bl.name} for {len(pts)} {what}", model="registration")
        res, _ = self._gpu(lambda: longitrack.propagate_with_crop(self.study, vis, bl, pts, self.cache_dir / "reg_crops"))
        return [r.followup_point for r in res]

    def _nearest_at_baseline(self):
        """A function point -> (nearest baseline lesion (mm, id), nearest baseline candidate (mm, id, item)), each None
        if there is none; the baseline centroids are computed once, on first use."""
        bl = self.study.baseline
        blm, sp = self.registry.label_map(bl), np.array(self._vol(bl).spacing)
        items = list(self.cands.get(bl.index, {}).get("items", {}).values())
        cent = []

        def nearest(p):
            if not cent:
                cent.append(label_centroids(blm))  # once, not per point x lesion over the whole scan
            p = np.array(p, float)
            dl = [(float(np.linalg.norm((c - p) * sp)), int(i)) for i, c in cent[0].items()]
            dc = [(float(np.linalg.norm((np.array(c["centroid"]) - p) * sp)), c["id"], c) for c in items]
            return (min(dl) if dl else None), (min(dc, key=lambda t: t[:2]) if dc else None)

        return nearest

    def _follow_up_candidates(self, vis, openc) -> tuple[str, list]:
        """Follow-up candidates: map each back to the baseline (registration), nearest baseline lesion / candidate,
        and a THEN / NOW image, to tell a new lesion from a missed baseline lesion."""
        bl = self.study.baseline
        back = self._map_to_baseline(vis, [[int(round(x)) for x in c["centroid"]] for c in openc], "candidate(s)") \
            if openc else None  # fmt: skip
        if back is not None:
            nearest = self._nearest_at_baseline()
            for c, bp in zip(openc, back):
                c["baseline_point"] = bp
                if bp is None:
                    continue
                nl, nc = nearest(bp)
                c["nearest_baseline_lesion"] = None if nl is None else {"lesion": nl[1], "mm": round(nl[0], 1)}
                c["nearest_baseline_candidate"] = None if nc is None else {"c": nc[1], "mm": round(nc[0], 1)}
                c["baseline_hu_at_point"] = float(self._vol(bl).data[tuple(int(round(v)) for v in bp)])
        rows = [f"c{c['id']}: LD {c['ld_mm']:.1f} mm, HU {c['median_hu']:.0f}, at {[round(x) for x in c['centroid']]}, "
                f"baseline point {c.get('baseline_point')}, nearest baseline lesion {c.get('nearest_baseline_lesion')}, "
                f"nearest baseline candidate {c.get('nearest_baseline_candidate')}" for c in openc]  # fmt: skip
        strip = self._then_now(vis, openc)
        return (f"{len(openc)} candidate(s) at {vis.name} not covered by tracked lesions:\n" + "\n".join(rows)
                + ("\nTHEN / NOW image: each candidate (right, " + vis.name + ") next to the same position at "
                   + bl.name + " after registration (left, + = mapped point)." if strip else "")
                + "\nFor each: is it a lesion at all? Was it already there at baseline (look at the THEN panel: "
                "then it was missed -> segment it at baseline, accept this one as its mask here; it is not new)? "
                "Otherwise accept with role new and decide unequivocal vs equivocal in set_visit_judgement."), strip

    def _add_candidates(self, vis, mask, source: str, min_voxels: int = rules.MIN_CANDIDATE_VOXELS) -> list:
        """Merge a detector's binary mask into the visit's candidate table (the one `detect` fills): components that
        overlap an existing candidate add `source` to it; the others become new candidates c#. Returns the new ones."""
        vol = self._vol(vis)
        cur = self.cands.setdefault(vis.index, {"map": np.zeros(vol.shape, np.uint16), "items": {}, "prompts": []})
        existing = self.registry.label_map(vis)
        lab, n = ndi.label(mask, structure=np.ones((3, 3, 3), bool))
        sizes = np.bincount(lab.ravel())
        objs = ndi.find_objects(lab)
        new = []
        for i in sorted((i for i in range(1, n + 1) if sizes[i] >= min_voxels), key=lambda i: -sizes[i]):
            box = objs[i - 1]  # every per-component step on its bounding-box crop
            m = lab[box] == i
            hit, cnt = np.unique(cur["map"][box][m], return_counts=True)
            prev = [(c, int(h)) for h, c in zip(hit, cnt) if h]
            if prev and max(prev)[0] > rules.CANDIDATE_OVERLAP * sizes[i]:
                it = cur["items"][max(prev)[1]]
                if source not in it["prompts"]:
                    it["prompts"].append(source)
                    it["support"] = it.get("support", 0) + 1
                continue
            cid = max(cur["items"], default=0) + 1
            cmap = cur["map"][box]
            cmap[m & (cmap == 0)] = cid
            full = np.zeros(vol.shape, bool)
            full[box] = m
            meas = measure(full, vol.spacing, vol.direction)
            ex = existing[box]
            ov = int((m & (ex > 0)).sum())
            it = {"id": cid, "bbox": [[b.start, b.stop] for b in box], "voxels": int(sizes[i]),
                  "centroid": [round(c, 1) for c in center_of_mass_crop(m, box)], "prompts": [source],
                  "support": 1, "state": "existing" if ov > rules.CANDIDATE_OVERLAP * sizes[i] else "open", "ld_mm": meas.ld_mm,
                  "sa_mm": meas.sa_mm, "sa_max_mm": meas.sa_max_mm, "volume_ml": meas.volume_ml,
                  "median_hu": round(float(np.median(vol.data[box][m])), 1),
                  "overlaps_lesion": sorted(int(x) for x in np.unique(ex[m]) if x)}  # fmt: skip
            cur["items"][cid] = it
            new.append(it)
        self.changed([vis], what="candidates")
        return new

    def totalsegmentator(self, visit=None, task="organs", lesions=None, overwrite=False) -> Result:
        """TotalSegmentator. task 'organs' (total fast + trunk_cavities): organ group per lesion (assign_organs).
        Lesion tasks ('lung_nodules', 'liver_tumors'): a second detector besides VoxTell; its findings join the visit's
        candidate table (new c#, or the source added to an overlapping candidate) for review_candidates."""
        if task == "organs":
            return self.assign_organs(visit, lesions, overwrite)
        from .backends import organs

        if task not in organs.LESION_TASKS:
            raise AgentError(f"task must be 'organs' or one of {list(organs.LESION_TASKS)}")
        vis = self._vis(visit)
        self.activity(f"TotalSegmentator {task} on {vis.name}", model="totalsegmentator", visit=vis.index)
        t0 = time.time()
        mask = organs.lesion_mask(vis.path, self.cache_dir / "totalseg", task, key=vis.case_id)
        self.n_model_calls += 1
        new = self._add_candidates(vis, mask, f"TotalSegmentator:{task}")
        openc = [c for c in new if c["state"] == "open"]
        head = (f"TotalSegmentator {task} on {vis.name} ({time.time() - t0:.0f} s): {len(new)} new candidate(s), "
                f"{len(openc)} not covered by lesions; candidates it shares with VoxTell list both sources.\n")  # fmt: skip
        if vis.index != self.study.baseline.index:
            txt, strip = self._follow_up_candidates(vis, openc)
            return Result(head + txt, strip, {"candidates": openc})
        rows = [_cand_row(c, full=False) for c in openc]
        return Result(head + "\n".join(rows) + "\nInspect and decide them like VoxTell candidates (view candidate=c#, "
                      "review_candidates).", data={"candidates": openc})  # fmt: skip

    def _hu_window(self, v, mask=None, point=None):
        """Window centred on the finding's own density: median HU in the mask (or a 3-voxel ball at the point)
        +-150 HU; lung window in air."""
        vol = self._vol(v)
        box = crop_box(mask) if mask is not None else None
        if box is not None:
            hu = float(np.median(vol.data[box][mask[box]]))  # the same voxels as vol.data[mask]
        else:
            z, y, x = [int(round(c)) for c in point]
            hu = float(np.median(vol.data[max(0, z - 1):z + 2, max(0, y - 3):y + 4, max(0, x - 3):x + 4]))
        return render.WINDOWS["lung"] if hu < rules.AIR_HU else (hu - rules.FINDING_WINDOW_HU, hu + rules.FINDING_WINDOW_HU)

    def _then_now_panels(self, vis, items) -> list:
        """items: (label, point_now, point_then, now_overlay, ld_mm, window). Per finding one row: THEN at the baseline at
        the registered position on 3 slices (-5 mm, 0, +5 mm) and NOW."""
        bl = self.study.baseline
        panels = []
        for label, pn, pb, ov_now, ld, win in items:
            size = max(40.0, 2.5 * ld)
            dz = max(1, int(round(5.0 / float(self._vol(bl).spacing[0]))))
            vb = self._vol(bl)
            for k in (-dz, 0, dz):
                z = int(np.clip(pb[0] + k, 0, vb.shape[0] - 1))
                p = [z, pb[1], pb[2]]
                fov = self._fov(bl, p, size)
                panels.append(render.panel(vb.data, vb.spacing, self._ori(bl), 0, z, fov[1], fov[2],
                                           render.Overlay(marks=[render.Mark(p, "")]), win, px=200,
                                           title=f"THEN {bl.name} z={z}"))  # fmt: skip
            vn = self._vol(vis)
            fov = self._fov(vis, pn, size)
            panels.append(render.panel(vn.data, vn.spacing, self._ori(vis), 0, pn[0], fov[1], fov[2], ov_now, win, px=200,
                                       title=f"NOW {vis.name} {label} LD {ld:.0f} mm z={pn[0]}"))  # fmt: skip
        return [self._png(render.stack(panels, ncols=4), "then_now")] if panels else []

    def _then_now(self, vis, cands, max_n: int = 6, min_ld_mm: float = 5.0) -> list:
        """One image puts every follow-up candidate (>= 5 mm, largest 6) next to the baseline at its registered
        position (3 slices, window at the candidate's density), so the agent sees whether something was already
        there."""
        cs = sorted((c for c in cands if c.get("baseline_point") is not None and c["ld_mm"] >= min_ld_mm),
                    key=lambda c: -c["ld_mm"])[:max_n]  # fmt: skip
        items = []
        for c in cs:
            m = self.cands[vis.index]["map"] == c["id"]
            ov = render.Overlay(candidates=np.where(m, c["id"], 0).astype(np.uint16), candidate_names={c["id"]: f"c{c['id']}"})
            items.append((f"c{c['id']}", [int(round(x)) for x in c["centroid"]], self._point(c["baseline_point"], self.study.baseline),
                          ov, c["ld_mm"], self._hu_window(vis, mask=m)))  # fmt: skip
        return self._then_now_panels(vis, items)

    def _new_lesion_check(self, vis) -> tuple[list, list, list]:
        """Evidence for the unequivocal-new-lesion decision at `vis` (RECIST 1.1: E09 §4.3.5, S16 Q9): for every lesion
        with role `new` present at this visit, the reasons it may be equivocal, and a THEN/NOW image."""
        bl = self.study.baseline
        news = [l for l in self.registry.lesions.values()
                if l.role == "new" and vis.index in l.findings and l.findings[vis.index].status == "present"
                and self.registry.mask(l.id, vis).any()]  # fmt: skip
        if not news:
            msg = "no lesion with role `new` is present at this visit in the registry"
            return [msg], [msg], []
        sp = np.array(self._vol(vis).spacing)
        lm_v = self.registry.label_map(vis)
        boxes = {l.id: crop_box(lm_v == l.id) for l in news}
        cents = {l.id: [int(round(x)) for x in center_of_mass_crop(lm_v[boxes[l.id]] == l.id, boxes[l.id])] for l in news}
        res = self._map_to_baseline(vis, [cents[l.id] for l in news], "new lesion(s)")
        back = {l.id: bp for l, bp in zip(news, res)}
        nearest = self._nearest_at_baseline()
        present = {int(i) for i in np.unique(lm_v) if i}  # = registry.mask(id, vis).any(), one pass
        others = [o for o in self.registry.lesions.values() if o.role not in ("new", "rejected") and o.id in present]
        rows, flags_all, items = [], [], []
        for l in news:
            f, m = l.findings[vis.index], self.registry.mask(l.id, vis)
            meas = f.measurement or self.registry.remeasure(l.id, vis)
            size = meas.get("sa_max_mm", 0) if l.is_node else meas.get("ld_mm", 0)
            flags = []
            if size < rules.NEW_LESION_SMALL_MM:
                flags.append(f"{'short axis' if l.is_node else 'long axis'} {size:.1f} mm < {rules.NEW_LESION_SMALL_MM:g} mm"
                             + (f" (S16: a new node counts only from {rules.NEW_LESION_SMALL_MM:g} mm short axis)" if l.is_node else " (small: the typical equivocal finding)"))  # fmt: skip
            pres = sorted(k for k, g in l.findings.items() if g.status == "present")
            if pres and pres[0] < vis.index:
                m0 = l.findings[pres[0]].measurement or {}
                s0 = m0.get("sa_max_mm", 0) if l.is_node else m0.get("ld_mm", 0)
                if size - s0 < rules.NEW_LESION_GROWTH_MM:
                    flags.append(f"since v{pres[0]} {s0:.1f} -> {size:.1f} mm: no growth beyond measurement error, "
                                 f"so this visit does not confirm it")  # fmt: skip
            for d, o in _lesions_near(m, boxes[l.id], lm_v, others, sp, rules.NEW_LESION_NEAR_MM)[:2]:
                flags.append(f"{'touches' if d <= rules.TOUCH_MM else f'{d:.0f} mm from'} lesion #{o.id} {o.name}: growth/coalescence of "
                             f"that lesion, or a separate lesion?")  # fmt: skip
            bp = back.get(l.id)
            if f.provenance and "candidate" not in f.provenance and "voxtell" not in f.provenance.lower():
                flags.append(f"segmented directly ({f.provenance}): no THEN/NOW check so far"
                             + (" — see the image" if bp is not None else ""))  # fmt: skip
            if bp is not None:
                nl, nc = nearest(bp)
                if nl is not None and nl[0] <= rules.BASELINE_BACKCHECK_MM:
                    flags.append(f"baseline lesion #{nl[1]} lies {nl[0]:.0f} mm from the registered position")
                if nc is not None and nc[0] <= rules.BASELINE_BACKCHECK_MM:
                    d, _, c = nc
                    flags.append(f"baseline candidate c{c['id']} ({c['ld_mm']:.0f} mm, {c['state']}"
                                 + (f": '{c.get('reason')}'" if c.get("reason") else "") + f") lies {d:.0f} mm from the "
                                 f"registered position — was it already there?")  # fmt: skip
                items.append((f"#{l.id}", cents[l.id], self._point(bp, bl),
                              render.Overlay(lesions=np.where(m, l.id, 0).astype(np.uint16), names={l.id: f"#{l.id}"}),
                              meas.get("ld_mm", 0), self._hu_window(vis, mask=m)))  # fmt: skip
            rows.append(f"#{l.id} {l.name} ({l.organ or '?'}, first seen v{pres[0] if pres else vis.index}): "
                        + ("; ".join(flags) if flags else "no reason for doubt found"))  # fmt: skip
            flags_all += flags
        return rows, flags_all, self._then_now_panels(vis, items)

    def set_visit_judgement(self, visit, non_target_status=None, new_lesion=None, equivocal_new_lesion=None,
                            notes="", confirm_reason="", gate=False) -> Result:  # fmt: skip
        """gate (the agent's tool call): an unequivocal new lesion means PD, so `new_lesion=true` first shows the
        evidence for every new lesion (size, growth since first seen, contiguity with known lesions, baseline back-check
        with a THEN/NOW image) and is only recorded when nothing speaks for 'equivocal' or `confirm_reason` says why
        it is unequivocal anyway (RECIST 1.1 has no size limit, but small findings are the typical equivocal ones)."""
        vis = self._vis(visit)
        if non_target_status and non_target_status not in ("CR", "NonCR/NonPD", "PD", "NE", "none"):
            raise AgentError("non_target_status: CR | NonCR/NonPD | PD | NE | none")
        if gate and new_lesion:
            rows, flags, imgs = self._new_lesion_check(vis)
            if flags and not confirm_reason.strip():
                kept = ""
                if non_target_status or equivocal_new_lesion is not None or notes:  # the rest of the call counts
                    j = self._judge(vis, non_target_status, None, equivocal_new_lesion, notes)
                    kept = (f"\nRecorded (everything but the unequivocal new lesion): non-target {j.non_target_status}, "
                            f"equivocal {j.equivocal_new_lesion}" + (", notes" if notes else "") + ".")  # fmt: skip
                return Result(f"{vis.name}: unequivocal new lesion NOT recorded yet. RECIST 1.1: a new lesion means PD only "
                              f"if it is unequivocal (not explained by technique or by something other than tumour); "
                              f"equivocal ones, e.g. small, are followed and PD is back-dated if confirmed.\n"
                              + "\n".join("  " + r for r in rows)
                              + ("\nLook at the THEN/NOW image (baseline at the registered position, 3 slices)." if imgs
                                 else "\nNo THEN/NOW image.")
                              + " Then either "
                                "`set_visit_judgement equivocal_new_lesion=true` (follow it), or, if it is unequivocal "
                                "despite these points, call again with `new_lesion=true` and `confirm_reason`." + kept,
                              imgs)  # fmt: skip
            if confirm_reason.strip():
                notes = (notes + " " if notes else "") + f"[new lesion confirmed despite: {'; '.join(flags)}; reason: {confirm_reason.strip()}]"
        j = self._judge(vis, non_target_status, new_lesion, equivocal_new_lesion, notes)
        return Result(f"{vis.name}: non-target {j.non_target_status}, unequivocal new lesion {j.new_lesion}, "
                      f"equivocal {j.equivocal_new_lesion}")  # fmt: skip

    def _judge(self, vis, non_target_status, new_lesion, equivocal_new_lesion, notes):
        """Record the given parts of a visit judgement."""
        j = self.registry.judgement(vis)
        if non_target_status:
            j.non_target_status = non_target_status
        if new_lesion is not None:
            j.new_lesion = bool(new_lesion)
        if equivocal_new_lesion is not None:
            j.equivocal_new_lesion = bool(equivocal_new_lesion)
        if notes:
            j.notes = notes
        self.changed([vis], what="judgement")
        return j

    # ================================================================== phase 3: selection + report
    def _sel_candidates(self):
        bl = self.study.baseline.index
        st = float(self._vol(bl).spacing[0])
        out = []
        for l in self.registry.lesions.values():
            if l.role in ("rejected", "new") or bl not in l.findings or l.findings[bl].status != "present":
                continue
            m = l.findings[bl].measurement or self.registry.remeasure(l.id, bl)
            sc = self.scores.get(l.id, {})
            out.append(selection.SelCandidate(l.id, l.name, l.organ, l.is_node, m.get("ld_mm", 0), m.get("sa_max_mm", 0),
                                              st, list(sc.get("flags", [])), sc.get("repro_score"), sc.get("rationale", "")))  # fmt: skip
        return out

    def candidate_table(self) -> Result:
        cs = self._sel_candidates()
        if not cs:
            return Result("No baseline lesions yet.")
        rows = []
        bl = self.study.baseline.index
        vol = self._vol(bl)
        for c in sorted(cs, key=lambda c: (c.group, -c.size)):
            ok, why = c.eligible()
            m = self.registry.mask(c.lesion_id, bl)
            hu = float(np.median(vol.data[m])) if m.any() else float("nan")
            auto = []
            if hu < rules.LOW_HU:
                auto.append("low_HU(cystic/necrotic?)")
            meas = self.registry.lesions[c.lesion_id].findings[bl].measurement
            if meas.get("n_components", 1) > 1:
                auto.append(f"{meas['n_components']}_components")
            bb = bbox_of(m)
            if bb and (bb[0][0] == 0 or bb[0][1] == vol.shape[0]):
                auto.append("at_fov_edge(z)")
            rows.append(f"#{c.lesion_id} {c.name} | organ group {c.group} | {'node SA' if c.is_node else 'LD'} {c.size:.1f} mm "
                        f"| eligible: {ok} ({why}) | median HU {hu:.0f} | auto flags {auto or '-'} | agent flags "
                        f"{c.flags or '-'} | repro {c.repro_score if c.repro_score is not None else 'not scored'}")  # fmt: skip
        return Result("Target-selection candidates (baseline lesions). Score the eligible ones that can matter "
                      "(largest per organ, borderline) with score_lesions after looking at view(measurement=true).\n"
                      + "\n".join(rows))  # fmt: skip

    def score_lesions(self, scores) -> Result:
        """Batch of reproducibility judgements: [{"lesion", "repro_score", "flags", "rationale"}]."""
        if not scores:
            raise AgentError("no scores given")
        out = []
        for d in scores:
            l, flags = self._les(d["lesion"]), d.get("flags") or []
            bad = [f for f in flags if f not in selection.PENALTY_FLAGS and f not in selection.EXCLUDING_FLAGS]
            if bad:
                raise AgentError(f"unknown flags {bad}; allowed: "
                                 f"{sorted(selection.PENALTY_FLAGS) + sorted(selection.EXCLUDING_FLAGS)}")  # fmt: skip
            self.scores[l.id] = {"repro_score": float(d["repro_score"]), "flags": list(flags),
                                 "rationale": d.get("rationale", "")}  # fmt: skip
            out.append(f"#{l.id}: reproducibility {d['repro_score']}, flags {flags}")
        return Result("\n".join(out))

    def propose_targets(self, k=5, min_quality=0.4) -> Result:
        """Selection table (eligibility, auto flags, your scores) + the ranked valid target sets."""
        self.measure_all()
        t = self.candidate_table()
        if not self._sel_candidates():
            return t
        sets, log = selection.enumerate_sets(self._sel_candidates(), k=k, min_quality=min_quality)
        names = {l.id: l.name for l in self.registry.lesions.values()}
        rows = [f"set {i + 1}: {[f'#{x} {names[x]}' for x in s.lesion_ids]} | organs {s.organs_covered} | score {s.score} "
                f"| per organ {s.per_organ}" for i, s in enumerate(sets)]  # fmt: skip
        return Result(t.text + "\n\nValid target sets (<= 5, <= 2 per organ, nodes = one organ), best first:\n"
                      + "\n".join(rows) + ("\nNotes:\n" + "\n".join(log) if log else ""),
                      data={"sets": [asdict(s) for s in sets]})  # fmt: skip

    def set_targets(self, lesions, justification="", non_target_groups=None) -> Result:
        ids = [self._les(i).id for i in lesions]
        bl = self.study.baseline.index
        before = {l.id: l.role for l in self.registry.lesions.values()}
        for l in self.registry.lesions.values():
            if l.id in ids:
                l.role = "target"
            elif l.role == "target" or (l.role == "candidate" and bl in l.findings):
                l.role = "non_target"
        viol = self.registry.target_rule_violations()
        if viol:
            for l in self.registry.lesions.values():  # nothing changes on a refused set
                l.role = before[l.id]
            raise AgentError("target set violates RECIST 1.1: " + "; ".join(viol))
        self.target_justification = justification
        self.changed(lesions=ids, what="targets")
        nt = [l.id for l in self.registry.lesions.values() if l.role == "non_target"]
        if non_target_groups:
            self.non_target_groups = []
            for g in non_target_groups:
                self.non_target_groups.append({"label": g["label"], "lesions": [self._les(i).id for i in g["lesions"]]})
        return Result(f"targets {ids}; all other baseline lesions are non-targets {nt}. Rules OK."
                      + (f" Non-target groups: {[g['label'] for g in self.non_target_groups]}" if non_target_groups else ""))

    def measure_all(self, visits=None) -> Result:
        n = 0
        for vis in [self._vis(v) for v in (visits or [v.index for v in self.study.visits])]:
            for l in self.registry.lesions.values():
                if self.registry.mask(l.id, vis).any():
                    self.registry.remeasure(l.id, vis)
                    n += 1
        self.changed(self.study.visits, what="measure")
        return Result(f"re-measured {n} lesion masks.\n" + self.list_lesions().text)

    def recist_report(self, confirm=False) -> Result:
        """Re-measures every mask, computes the report, adds category stability over near-tied target sets, saves."""
        tg = self.registry.targets()
        if not tg:
            raise AgentError("no target lesions set (set_targets); with no measurable disease use non-target only")
        self.measure_all()
        rep = self.registry.report(confirm=confirm)
        bl = self.study.baseline.index
        lines = ["RECIST 1.1 report", "targets: " + ", ".join(
            f"#{l.id} {l.name} ({l.organ}, {'node SA' if l.is_node else 'LD'} "
            f"{l.findings[bl].measurement.get('sa_max_mm' if l.is_node else 'ld_mm', 0):.1f} mm)" for l in tg),
                 f"baseline sum of diameters: {rep['baseline_sld_mm']:.1f} mm"]  # fmt: skip
        if rep["target_rule_violations"]:
            lines.append("RULE VIOLATIONS: " + "; ".join(rep["target_rule_violations"]))
        for t in rep["timepoints"]:
            v = self.study.visit(t["visit"])
            ch = lambda x: "-" if x is None else f"{100 * x:+.1f}%"  # noqa: E731
            when = "" if t["day"] is None else f" day {t['day']}"
            lines.append(f"{v.name}{when}: SLD {_fmt(t['sld_mm'])} mm (baseline {ch(t['change_from_baseline'])},"
                         f" nadir {_fmt(t['nadir_mm'])} -> {ch(t['change_from_nadir'])}); target {t['target_response']},"
                         f" non-target {t['non_target_response']}, new lesion {t['new_lesion']} => OVERALL {t['overall']}"
                         f" | mm to PR {_fmt(t['mm_to_pr'])}, mm to PD {_fmt(t['mm_to_pd'])}"
                         + (f" | {'; '.join(t['notes'])}" if t["notes"] else ""))  # fmt: skip
        lines.append(f"best overall response: {rep['best_overall_response']}" + (" (confirmed)" if confirm else ""))
        if rep.get("pd_backdating"):
            b = rep["pd_backdating"]
            lines.append(f"PD back-dating: an equivocal new finding at v{b['pd_dated_to_visit']} and an unequivocal new "
                         f"lesion at v{b['confirmed_at_visit']}: if they are the same lesion, PD dates to "
                         f"v{b['pd_dated_to_visit']} (timepoint calls above are as judged per visit).")  # fmt: skip
        open_ = {}  # baseline lesions with no usable status at a follow-up: never silently left out
        for v in self.study.visits[1:]:
            ids = [l.id for l in self.registry.lesions.values() if l.role not in ("rejected", "new") and bl in l.findings
                   and (v.index not in l.findings or l.findings[v.index].status == "not_assessed")]
            if ids:
                open_[v.name] = ids
        if open_:
            lines.append("OPEN: baseline lesions without a status at a follow-up (find each one: view_pair, segment, "
                         "or update_lesion status=too_small/absent/merged): "
                         + "; ".join(f"{k}: " + ", ".join(f"#{i}" for i in x) for k, x in open_.items()))
        missing = [v.name for v in self.study.visits[1:] if v.index not in self.registry.judgements]
        if missing:
            lines.append(f"WARNING: no visit judgement (non-target status / new lesions) for {missing}: "
                         "set_visit_judgement; until then a lesion with role 'new' counts as unequivocal.")  # fmt: skip
        mg = rules.DECISION_MARGIN_MM  # short of a threshold or past it: a few mm of measurement decide the category
        close = [t for t in rep["timepoints"] if any(t[k] is not None and abs(t[k]) < mg for k in ("mm_to_pd", "mm_to_pr"))]
        if close:
            lines.append(f"DECISION-CRITICAL: visits within {mg:g} mm of a threshold (either side): "
                         + ", ".join(self.study.visit(t["visit"]).name for t in close)
                         + " — re-check the target measurements there (view_pair, view measurement=true).")  # fmt: skip
        rep["justification"] = self.target_justification
        rep["non_target_groups"] = self.non_target_groups
        try:
            su = self.selection_uncertainty()
            rep["selection_uncertainty"] = su.data
            lines.append("Target-choice stability: " + su.text)
        except Exception as e:  # noqa: BLE001
            lines.append(f"(target-choice stability unavailable: {type(e).__name__}: {e})")
        (self.workdir / "report.json").write_text(json.dumps(rep, indent=2, default=str))
        (self.workdir / "report.md").write_text("\n".join(lines) + "\n")
        self.emit("report", report=rep)
        self._save_state()
        return Result("\n".join(lines) + "\nSaved to result/.", data=rep)

    def selection_uncertainty(self, k=10) -> Result:
        sets, _ = selection.enumerate_sets(self._sel_candidates(), k=k)
        saved = {l.id: l.role for l in self.registry.lesions.values()}

        def ev(ids):
            for l in self.registry.lesions.values():
                l.role = "target" if l.id in ids else ("non_target" if saved[l.id] in ("target", "non_target", "candidate") else saved[l.id])
            try:
                return [t["overall"] for t in self.registry.report()["timepoints"]]
            finally:
                for l in self.registry.lesions.values():
                    l.role = saved[l.id]

        st = selection.category_stability(ev, sets)
        return Result(f"Category stability over {st.get('sets')} near-tied valid target sets: {st.get('stability')}; "
                      f"reference {st.get('reference')}; lesions whose choice flips the category: "
                      f"{st.get('lesions_that_flip')}\nper set: {st.get('per_set')}", data=st)  # fmt: skip

    # ================================================================== clinician, save
    def ask_clinician(self, kind, question, lesion=None, visit=None, point=None, options=None) -> Result:
        if kind not in KINDS:
            raise AgentError(f"kind must be one of {KINDS}")
        req = {"kind": kind, "question": question, "lesion": None if lesion is None else self._les(lesion).id,
               "visit": None if visit is None else self._vis(visit).index, "point": point, "options": options or []}
        if lesion is not None and kind in ("click", "verify_point"):
            l = self.registry.lesions[req["lesion"]]
            ref = next((v for v in self.study.visits if self.registry.mask(l.id, v).any()), None)
            if ref is not None:
                req["reference_visit"], req["reference_point"] = ref.index, self._center(l.id, ref)
        self.emit("question", request=req)
        self.activity(f"asking the clinician ({kind}): {question}", model="clinician")
        ans = self.clinician.ask(req)
        self.questions.append({"request": req, "answer": ans})
        self.emit("answer", request=req, answer=ans)
        return Result(f"clinician ({self.clinician.name}): {json.dumps(ans)}", data=ans)

    def save(self, out_dir=None) -> Result:
        self._save_state(out_dir)
        try:
            self.recist_report()
        except AgentError:
            pass
        return Result(f"saved registry, masks and extras to {Path(out_dir) if out_dir else self.workdir / 'result'}")

    def _save_state(self, out_dir=None) -> None:
        out = Path(out_dir) if out_dir else self.workdir / "result"
        self.registry.save(out)
        (out / "extras.json").write_text(json.dumps({
            "scores": self.scores, "justification": self.target_justification, "non_target_groups": self.non_target_groups,
            "questions": self.questions, "candidates": {str(k): {str(c): {kk: vv for kk, vv in it.items() if kk != "bbox"}
                                                                   for c, it in v["items"].items()} for k, v in self.cands.items()},
        }, indent=2, default=str))  # fmt: skip

    def close(self) -> None:
        from .backends import models

        models.release()


def _ld_sa(m: dict, samax: bool = False) -> str:
    """'LD 12.3 SA 4.5 [SAmax 6.7] mm' of a measurement dict."""
    return f"LD {m['ld_mm']:.1f} SA {m['sa_mm']:.1f}" + (f" SAmax {m['sa_max_mm']:.1f}" if samax else "") + " mm"


def _cand_row(c: dict, full: bool = True) -> str:
    """One row of a candidate table (full: with voxels, state, prompts and overlapping lesions)."""
    row = (f"c{c['id']} | " + (f"{c['voxels']} vox " if full else "") + f"{c['volume_ml']:.2f} ml | LD {c['ld_mm']:.1f} "
           f"SA {c['sa_mm']:.1f} | HU {c['median_hu']:.0f} | at {[round(x) for x in c['centroid']]}")
    if full:
        row += f" | {c['state']} | prompts {c['prompts']}" + (
            f" | overlaps lesion {c['overlaps_lesion']}" if c["overlaps_lesion"] else "")
    return row


def _components(mask) -> tuple:
    """Connected components: (label image, sizes of labels 1..n, label indices (0-based) largest first). Component k
    in `inspect` and edit_mask is label order[k - 1] + 1."""
    lab, _ = ndi.label(mask)
    sizes = np.bincount(lab.ravel())[1:]
    return lab, sizes, np.argsort(-sizes)


def _largest_component(mask):
    lab, n = ndi.label(mask)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n > 1 else mask


def _lesions_near(m, box, lm, others, sp, max_mm: float) -> list:
    """[(mm, lesion)] of the `others` within max_mm of mask m (whose bbox slices are `box`), nearest first. Computed on
    the bbox grown by max_mm (+1 voxel) per axis: every voxel outside is farther, so up to max_mm the crop gives the
    full-scan distance, and farther lesions are not listed either way."""
    grow = [int(np.ceil(max_mm / s_)) + 1 for s_ in sp]
    cb = tuple(slice(max(b.start - k, 0), min(b.stop + k, n_)) for b, k, n_ in zip(box, grow, m.shape))
    lmc, mc = lm[cb], m[cb]
    near = []
    for o in others:
        omc = lmc == o.id
        if not omc.any():
            continue  # farther than max_mm
        d = float(ndi.distance_transform_edt(~omc, sampling=sp)[mc].min())
        if d <= max_mm:
            near.append((d, o))
    return sorted(near, key=lambda t: t[0])


def _delta(prev, mask) -> str:
    return f"change +{int(np.count_nonzero(mask & ~prev))} / -{int(np.count_nonzero(prev & ~mask))} voxels"


def _prompt_warning(positive: bool, touches_mask: bool, inside: float) -> str:
    """A prompt on the wrong side of the mask."""
    if not positive and not touches_mask:
        return " - NOTE: a negative prompt outside the mask removes nothing; put it inside the leak"
    if positive and inside > 0.9:
        return " - NOTE: a positive prompt inside the mask adds little; put it inside the missing part"
    return ""


def _box_fit(mask, axis: int, index: int, lo: list, hi: list) -> str:
    """How the prediction in the box's slice fits the box: a box read off a view at the
    wrong place shows up as prediction sticking out of it, or as a lot of empty box. lo/hi: inclusive corners."""
    ra, ca = plane_axes(axis)
    m = np.take(mask, index, axis=axis) > 0
    lab, _ = ndi.label(m)
    inside = np.unique(lab[lo[ra]: hi[ra] + 1, lo[ca]: hi[ca] + 1])
    m = np.isin(lab, inside[inside > 0])  # the parts of the prediction that touch the box
    if not m.any():
        return f"box check ({'zyx'[axis]} = {index}): no prediction inside the box"
    ext = {ra: np.flatnonzero(m.any(1))[[0, -1]], ca: np.flatnonzero(m.any(0))[[0, -1]]}
    out, loose = [], []
    for a in (ra, ca):
        size = hi[a] - lo[a] + 1
        for side, over, slack in (("low", lo[a] - ext[a][0], ext[a][0] - lo[a]),
                                  ("high", ext[a][1] - hi[a], hi[a] - ext[a][1])):  # fmt: skip
            if over > max(3, round(0.08 * size)):
                out.append(f"{over} voxels past the box at {'zyx'[a]} ({side} side)")
            elif slack > max(6, round(0.25 * size)):
                loose.append(f"{slack} voxels of empty box at {'zyx'[a]} ({side} side)")
    if not out and not loose:
        return f"box check ({'zyx'[axis]} = {index}): the prediction fits the box"
    msg = f"box check ({'zyx'[axis]} = {index}): "
    if out:
        msg += ("WARNING the prediction extends " + "; ".join(out) + ". Either the box cut the lesion there (then "
                "segment again with a corrected box) or it leaks into a neighbour (then refine_lesion with a negative "
                "scribble). Look at that side. ")  # fmt: skip
    if loose:
        msg += "The box may be too large or misplaced: " + "; ".join(loose) + " (or the model stopped early there)."
    return msg.strip()


def _type(t):
    if t not in TYPES:
        raise AgentError(f"lesion_type must be one of {TYPES}")
    return t


def _role(r):
    if r not in ROLES:
        raise AgentError(f"role must be one of {ROLES}")
    return r


def _pct(a) -> dict:
    a = np.asarray(a, float)
    if not a.size:
        return {}
    q = np.percentile(a, [10, 50, 90])
    return {"p10": round(float(q[0]), 1), "median": round(float(q[1]), 1), "p90": round(float(q[2]), 1),
            "mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1), "n": int(a.size)}  # fmt: skip


def _pct_txt(d: dict) -> str:
    return "-" if not d else f"median {d['median']:.0f} (p10 {d['p10']:.0f}, p90 {d['p90']:.0f}, sd {d['std']:.0f})"
