"""A study = one patient's CT visits in chronological order: `Study.from_files([ct_bl, ct_fu, ...], days=[0, 63, ...])`.

Images are read lazily with SimpleITK (array order [z, y, x]). Views, sweeps and the skill take array axis 0 as the
axial (head-foot) axis, so `from_files` refuses volumes that are stored in another orientation.
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mha", ".mhd")


@dataclass
class Volume:
    data: np.ndarray            # [z, y, x] float32 (HU)
    spacing: tuple              # (sz, sy, sx) mm
    direction: tuple            # SimpleITK direction (row-major 3x3)
    origin: tuple
    image: sitk.Image           # geometry reference for writing masks

    @property
    def shape(self) -> tuple:
        return self.data.shape


@dataclass
class Visit:
    index: int                  # 1-based, chronological
    path: Path
    case_id: str                # unique within the study: names the visit's files in the result and the cache
    day: int | None = None      # days from baseline
    date: str | None = None

    @property
    def name(self) -> str:
        return f"v{self.index}"


def _case_ids(paths) -> list[str]:
    """One id per file: the file name without its suffix, prefixed with the visit when two files share a name."""
    ids = []
    for p in paths:
        name = Path(p).name
        ids.append(next((name[: -len(s)] for s in _SUFFIXES if name.lower().endswith(s)), Path(p).stem))
    if len(set(ids)) < len(ids):
        ids = [f"v{i + 1}_{c}" for i, c in enumerate(ids)]
    return ids


def _header(path) -> sitk.ImageFileReader:
    r = sitk.ImageFileReader()
    r.SetFileName(str(path))
    r.ReadImageInformation()
    return r


def check_ct_file(path) -> None:
    """Refuse what the tools cannot read correctly: a missing file, a volume that is not 3D with one value per voxel,
    or one whose array axes are not (head-foot, front-back, left-right)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{p} not found")
    h = _header(p)
    if h.GetDimension() != 3 or h.GetNumberOfComponents() != 1:
        raise ValueError(f"{p.name}: expected one 3D volume, got {h.GetDimension()}D with {h.GetNumberOfComponents()} "
                         f"value(s) per voxel")  # fmt: skip
    d = np.abs(np.asarray(h.GetDirection(), float).reshape(3, 3))
    if any(int(np.argmax(d[:, k])) != k for k in range(3)):
        raise ValueError(f"{p.name}: the volume is not stored in an axial orientation (the last image axis must run "
                         f"head-foot and the first left-right); reorient it first, e.g. to RAS or LPS")  # fmt: skip


class Study:
    def __init__(self, patient_id: str, visits: list[Visit], source: str = "", cache_size: int = 2):
        if not visits:
            raise ValueError("a study needs at least one visit")
        self.patient_id = str(patient_id)
        self.visits = sorted(visits, key=lambda v: v.index)
        self.source = source
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size

    # ------------------------------------------------------------------ construction
    @classmethod
    def from_files(cls, paths, days=None, patient_id: str = "patient", dates=None, check: bool = True) -> "Study":
        """CT volumes in chronological order (baseline first), with the days from baseline if known."""
        paths = [Path(p) for p in paths]
        for what, x in (("days", days), ("dates", dates)):
            if x is not None and len(x) != len(paths):
                raise ValueError(f"{len(paths)} image(s) but {len(x)} {what}")
        if days is not None and list(days) != sorted(days):
            raise ValueError(f"the images must be in chronological order, but the days are {list(days)}")
        if check:
            for p in paths:
                check_ct_file(p)
        visits = [Visit(i + 1, p, c, None if days is None else int(days[i]), None if dates is None else dates[i])
                  for i, (p, c) in enumerate(zip(paths, _case_ids(paths)))]  # fmt: skip
        return cls(patient_id, visits, source="files")

    # ------------------------------------------------------------------ access
    def visit(self, v) -> Visit:
        """Visit by 1-based index, name ('v2') or case id."""
        if isinstance(v, Visit):
            return v
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("v") and v[1:].isdigit():
                v = int(v[1:])
            elif v.isdigit():  # JSON object keys ({"2": ...})
                v = int(v)
            else:
                for x in self.visits:
                    if x.case_id == v:
                        return x
                raise KeyError(f"unknown visit {v!r}")
        for x in self.visits:
            if x.index == int(v):
                return x
        raise KeyError(f"no visit {v}; visits are {[x.index for x in self.visits]}")

    @property
    def baseline(self) -> Visit:
        return self.visits[0]

    def volume(self, v) -> Volume:
        vis = self.visit(v)
        key = str(vis.path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        img = sitk.ReadImage(key)
        vol = Volume(sitk.GetArrayFromImage(img).astype(np.float32), tuple(img.GetSpacing()[::-1]),
                     tuple(img.GetDirection()), tuple(img.GetOrigin()), img)  # fmt: skip
        if float(vol.data[::4, ::4, ::4].min()) > -200:
            warnings.warn(f"{vis.path.name}: no air-like values (minimum {float(vol.data.min()):.0f}); the tools expect "
                          f"CT in Hounsfield units", stacklevel=2)  # fmt: skip
        self._cache[key] = vol
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return vol

    def describe(self) -> str:
        lines = [f"Study {self.source or self.patient_id}: {len(self.visits)} visits"]
        for v in self.visits:
            h = _header(v.path) if v.path.exists() else None
            geo = ""
            if h is not None:
                geo = f" shape[z,y,x]={list(h.GetSize()[::-1])} spacing[z,y,x]={[round(s, 2) for s in h.GetSpacing()[::-1]]}"
            when = f" day {v.day}" if v.day is not None else ""
            lines.append(f"  {v.name} ({v.case_id}){when}{geo}")
        return "\n".join(lines)
