"""Organ assignment for lesions with TotalSegmentator (`total` fast + `trunk_cavities`), RECIST organ groups.

Rule: >= 30 % of the lesion inside one organ group -> that organ; else majority organ in a 3 mm shell (>= 15 % of
the shell); else nearest organ voxel within 10 mm; else "none". Lymph nodes have no TS class: `extra_organ` (lesion
< 30 % inside any parenchymal organ) flags node / soft-tissue / peritoneal suspects, `cavity` gives the compartment.
RECIST: both lungs, both kidneys, both adrenals, all lymph nodes each count as ONE organ (EORTC RECIST FAQ).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi


def ts_group(name: str) -> str:
    if name.startswith("lung_"):
        return "lung"
    if name.startswith("kidney_"):
        return "kidney"
    if name.startswith("adrenal_gland"):
        return "adrenal"
    if name in ("liver", "spleen", "pancreas", "gallbladder", "stomach", "esophagus", "thyroid_gland",
                "urinary_bladder", "prostate", "brain", "heart", "trachea"):  # fmt: skip
        return {"thyroid_gland": "thyroid", "urinary_bladder": "bladder", "trachea": "airway"}.get(name, name)
    if name in ("small_bowel", "duodenum", "colon"):
        return "bowel"
    if name.startswith(("vertebrae_", "rib_", "hip_", "femur_", "humerus_", "scapula_", "clavicula_")) or \
            name in ("sacrum", "sternum", "skull", "costal_cartilages"):  # fmt: skip
        return "bone"
    if name.startswith(("gluteus_", "autochthon_", "iliopsoas_")):
        return "muscle"
    if name == "spinal_cord":
        return "cns"
    if name == "atrial_appendage_left":
        return "heart"
    return "vessel"


PARENCHYMA = {"lung", "liver", "kidney", "adrenal", "spleen", "pancreas", "bone", "thyroid", "brain", "prostate",
              "gallbladder", "stomach", "bowel", "bladder", "esophagus"}  # fmt: skip


@dataclass
class OrganAssignment:
    organ: str
    source: str            # inside | shell | nearest | none
    inside: dict           # organ group -> fraction of lesion inside
    shell: dict            # organ group -> fraction of 3 mm shell
    nearest_mm: float
    extra_organ: bool      # < 30 % inside any parenchymal organ: lymph node / soft tissue / peritoneal suspect
    cavity: str | None     # mediastinum | abdominal_cavity | thoracic_cavity | pericardium | outside | None


# TotalSegmentator tasks that segment lesions (openly licensed): tool task -> (TS task, lesion label)
LESION_TASKS = {"lung_nodules": ("lung_nodules", "lung_nodules"), "liver_tumors": ("liver_vessels", "liver_tumor")}


def run_task(image: Path, cache_dir: Path, task: str, fast: bool = False, key: str | None = None):
    """One TotalSegmentator task on one image (subprocess, cached on disk as `<key>_<task>.nii.gz`; `key` defaults to
    the file name) -> (label array, {id: name})."""
    from totalsegmentator.map_to_binary import class_map

    image, cache_dir = Path(image), Path(cache_dir)
    tag = task + ("_fast" if fast else "")
    out = cache_dir / f"{key or image.name.split('.')[0]}_{tag}.nii.gz"
    if not out.exists():
        import subprocess

        from . import models

        models.release_idle()  # the TotalSegmentator process needs the GPU memory of the resident models
        out.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run([sys.executable, "-m", f"{__package__}.ts_run", str(image), str(out), task,
                            "1" if fast else "0"], capture_output=True, text=True, timeout=900)  # fmt: skip
        if r.returncode or not out.exists():
            raise RuntimeError(f"TotalSegmentator {task} failed: {r.stderr[-1500:]}")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(out))).astype(np.int16)
    return arr, {int(k): v for k, v in class_map[task].items()}


def lesion_mask(image: Path, cache_dir: Path, task: str, key: str | None = None) -> np.ndarray:
    """Binary lesion mask of a TotalSegmentator lesion task (LESION_TASKS) in the image's SimpleITK grid."""
    ts_task, label = LESION_TASKS[task]
    arr, names = run_task(image, cache_dir, ts_task, key=key)
    ids = [k for k, v in names.items() if v == label]
    return np.isin(arr, ids)


class OrganMaps:
    """TotalSegmentator label maps of one visit, in the visit's SimpleITK grid, cached on disk."""

    def __init__(self, image: Path, cache_dir: Path, fast: bool = True, key: str | None = None):
        self.image, self.cache, self.fast, self.key = Path(image), Path(cache_dir), fast, key
        self.total, self.total_names = self._run("total")
        self.cav, self.cav_names = self._run("trunk_cavities")
        self.group_of = {k: ts_group(v) for k, v in self.total_names.items()}

    def _run(self, task: str):
        return run_task(self.image, self.cache, task, fast=self.fast and task == "total", key=self.key)

    def assign(self, mask: np.ndarray, spacing_zyx, other_lesions: np.ndarray | None = None,
               shell_mm: float = 3.0, near_mm: float = 10.0) -> OrganAssignment:  # fmt: skip
        sp = np.asarray(spacing_zyx, float)
        pad = np.ceil(max(near_mm, shell_mm) / sp).astype(int) + 1
        bb = ndi.find_objects(mask.astype(np.uint8))[0]
        sl = tuple(slice(max(s.start - p, 0), min(s.stop + p, n)) for s, p, n in zip(bb, pad, mask.shape))
        L, T = mask[sl].astype(bool), self.total[sl]
        n = max(int(L.sum()), 1)
        g_in: dict = {}
        for k, c in zip(*np.unique(T[L], return_counts=True)):
            if k:
                g = self.group_of.get(int(k), "vessel")
                g_in[g] = g_in.get(g, 0) + c / n
        d_out = ndi.distance_transform_edt(~L, sampling=sp)
        shell = (d_out > 0) & (d_out <= shell_mm)
        if other_lesions is not None:
            shell &= other_lesions[sl] == 0
        g_sh: dict = {}
        for k, c in zip(*np.unique(T[shell], return_counts=True)):
            g = self.group_of.get(int(k), "vessel") if k else "none"
            g_sh[g] = g_sh.get(g, 0) + c / max(int(shell.sum()), 1)
        near_g, near_d = None, float("inf")
        if (T > 0).any():
            d_to, idx = ndi.distance_transform_edt(T == 0, sampling=sp, return_indices=True)
            dl = np.where(L, d_to, np.inf)
            j = np.unravel_index(np.argmin(dl), dl.shape)
            near_d = float(dl[j])
            near_g = self.group_of.get(int(T[tuple(ix[j] for ix in idx)]), "vessel")
        if g_in and max(g_in.values()) >= 0.3:
            organ, how = max(g_in, key=g_in.get), "inside"
        elif any(g != "none" and v >= 0.15 for g, v in g_sh.items()):
            organ, how = max((g for g in g_sh if g != "none"), key=g_sh.get), "shell"
        elif near_d <= near_mm:
            organ, how = near_g, "nearest"
        else:
            organ, how = "none", "none"
        par = sum(v for g, v in g_in.items() if g in PARENCHYMA)
        u, c = np.unique(self.cav[sl][L], return_counts=True)
        cav = None
        if len(u):
            k = int(u[np.argmax(c)])
            cav = "outside" if k == 0 else self.cav_names.get(k, str(k))
        r = lambda d: {k: round(float(v), 3) for k, v in sorted(d.items(), key=lambda x: -x[1])}  # noqa: E731
        return OrganAssignment(organ, how, r(g_in), r(g_sh), round(near_d, 1), par < 0.3, cav)
