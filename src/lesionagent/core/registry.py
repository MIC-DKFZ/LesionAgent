"""LesionRegistry: the agent's structured memory and the single source of truth for the RECIST read.

- one record per lesion (patient-unique id) with type, organ, RECIST role and per-visit findings
- one uint16 instance map per visit (SimpleITK grid of that visit): voxel value = lesion id
- per-visit judgements that only a reader can make (non-target status, unequivocal new lesion)
The agent edits it through tools; `recist_report` turns it into engine inputs. Saved as registry.json +
masks/<case_id>.nii.gz, so a read can be reloaded, reviewed or evaluated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from . import recist
from .measure import measure
from .study import Study

ROLES = ("candidate", "target", "non_target", "new", "rejected")
STATUSES = ("present", "too_small", "absent", "merged", "not_assessed")
TYPES = ("tumor", "lymph_node", "unknown")


@dataclass
class Finding:
    """One lesion at one visit."""

    status: str = "present"
    provenance: str = ""            # what made the mask: voxtell | longitrack | nninteractive | registration ...
    point: list | None = None       # [z, y, x] prompt / seed
    merged_into: int | None = None
    measurement: dict = field(default_factory=dict)
    qa: dict = field(default_factory=dict)
    notes: str = ""


@dataclass
class Lesion:
    id: int
    name: str
    lesion_type: str = "unknown"
    organ: str = ""
    role: str = "candidate"
    findings: dict = field(default_factory=dict)   # visit index (int) -> Finding
    notes: str = ""
    confidence: str = ""                           # agent's self-assessment: high | medium | low

    @property
    def is_node(self) -> bool:
        return self.lesion_type == "lymph_node"


@dataclass
class VisitJudgement:
    non_target_status: str = "NonCR/NonPD"   # CR | NonCR/NonPD | PD | NE | none
    new_lesion: bool = False                 # unequivocal new lesion at this visit
    equivocal_new_lesion: bool = False
    notes: str = ""


class LesionRegistry:
    def __init__(self, study: Study):
        self.study = study
        self.lesions: dict[int, Lesion] = {}
        self.judgements: dict[int, VisitJudgement] = {}
        self._maps: dict[int, np.ndarray] = {}
        self.next_id = 1   # ids are never reused, also not after a deletion

    # ------------------------------------------------------------------ masks
    def label_map(self, v) -> np.ndarray:
        vis = self.study.visit(v)
        if vis.index not in self._maps:
            self._maps[vis.index] = np.zeros(self.study.volume(vis).shape, np.uint16)
        return self._maps[vis.index]

    def mask(self, lesion_id: int, v) -> np.ndarray:
        return self.label_map(v) == lesion_id

    def set_mask(self, lesion_id: int, v, mask: np.ndarray, overwrite_others: bool = False) -> dict:
        """Replace the lesion's voxels at a visit. Voxels of other lesions are kept unless overwrite_others."""
        lm = self.label_map(v)
        mask = np.asarray(mask, bool)
        if mask.shape != lm.shape:
            raise ValueError(f"mask shape {mask.shape} != visit shape {lm.shape}")
        lm[lm == lesion_id] = 0
        clash = mask & (lm > 0) & (lm != lesion_id)
        others = sorted(int(i) for i in np.unique(lm[clash])) if clash.any() else []
        write = mask if overwrite_others else mask & (lm == 0)
        lm[write] = lesion_id
        return {"voxels": int(write.sum()), "overlap_voxels": int(clash.sum()), "overlaps_with": others}

    # ------------------------------------------------------------------ lesions
    def new_lesion(self, name: str = "", lesion_type: str = "unknown", organ: str = "", role: str = "candidate") -> Lesion:
        lid = max(self.next_id, max(self.lesions, default=0) + 1)
        les = Lesion(lid, name or f"lesion_{lid}", _check(lesion_type, TYPES), organ, _check(role, ROLES))
        self.lesions[lid] = les
        self.next_id = lid + 1
        return les

    def get(self, ref) -> Lesion:
        if isinstance(ref, int) or (isinstance(ref, str) and ref.isdigit()):
            if int(ref) in self.lesions:
                return self.lesions[int(ref)]
        else:
            hits = [l for l in self.lesions.values() if l.name == ref]
            if len(hits) == 1:
                return hits[0]
        raise KeyError(f"unknown lesion {ref!r}")

    def finding(self, ref, v, create: bool = True) -> Finding:
        les, vis = self.get(ref), self.study.visit(v)
        if vis.index not in les.findings:
            if not create:
                raise KeyError(f"lesion {les.id} has no finding at {vis.name}")
            les.findings[vis.index] = Finding()
        return les.findings[vis.index]

    def remeasure(self, ref, v) -> dict:
        les, vis = self.get(ref), self.study.visit(v)
        vol = self.study.volume(vis)
        m = measure(self.mask(les.id, vis), vol.spacing, vol.direction).to_dict()
        f = self.finding(les.id, vis)
        f.measurement = m
        if m["n_voxels"] == 0 and f.status == "present":
            f.qa["empty_mask"] = True
        return m

    def delete(self, ref) -> None:
        les = self.get(ref)
        for idx, lm in self._maps.items():
            lm[lm == les.id] = 0
        del self.lesions[les.id]

    def merge(self, keep, absorb, v) -> None:
        """Lesions coalesced at visit v: `absorb` becomes part of `keep` there (RECIST: measure as one, other = 0)."""
        a, b = self.get(keep), self.get(absorb)
        lm = self.label_map(v)
        lm[lm == b.id] = a.id
        f = self.finding(b.id, v)
        f.status, f.merged_into = "merged", a.id

    # ------------------------------------------------------------------ RECIST
    def judgement(self, v) -> VisitJudgement:
        """The visit's judgement for editing (created with default values: the visit counts as judged from now on)."""
        vis = self.study.visit(v)
        return self.judgements.setdefault(vis.index, VisitJudgement())

    def targets(self) -> list[Lesion]:
        return [l for l in self.lesions.values() if l.role == "target"]

    def target_rule_violations(self) -> list[str]:
        bl = self.study.baseline.index
        cands = []
        for l in self.targets():
            m = l.findings.get(bl, Finding()).measurement
            st = self.study.volume(bl).spacing[0]
            cands.append(recist.Candidate(l.id, l.organ or "unknown", l.is_node, m.get("ld_mm", 0), m.get("sa_max_mm", 0), st))
        return recist.validate_targets(cands)

    def engine_inputs(self):
        bl = self.study.baseline.index

        def tm(l, idx):
            f = l.findings.get(idx)
            if f is None:
                return recist.TargetMeasurement(l.id, l.is_node, "not_assessed")
            st = {"merged": "absent"}.get(f.status, f.status)  # merged lesion counts 0, its partner carries it
            m = f.measurement
            if st == "present" and (m.get("n_voxels") == 0 or not m.get("sa_max_mm" if l.is_node else "ld_mm")):
                st = "not_assessed"  # present but no mask / no measurement: not assessable, never 0 mm
            return recist.TargetMeasurement(l.id, l.is_node, st, m.get("ld_mm"), m.get("sa_max_mm"))

        tgts = self.targets()
        base = [tm(l, bl) for l in tgts]
        visits = []
        for vis in self.study.visits[1:]:
            judged = vis.index in self.judgements
            j = self.judgements.get(vis.index, VisitJudgement())  # read only: unjudged stays unjudged
            new_present = any(l.role == "new" and vis.index in l.findings and l.findings[vis.index].status == "present"
                              for l in self.lesions.values())  # fmt: skip
            # the reader's judgement decides (equivocal new lesions are not PD); without one, a new lesion counts
            new = j.new_lesion if judged else new_present
            visits.append(recist.Visit(vis.index, vis.day, [tm(l, vis.index) for l in tgts], j.non_target_status,
                                       new, j.equivocal_new_lesion))  # fmt: skip
        bl_nt = self.judgements.get(bl, VisitJudgement()).non_target_status
        return base, visits, bl_nt

    def report(self, confirm: bool = False) -> dict:
        base, visits, _ = self.engine_inputs()
        res = recist.evaluate(base, visits)
        # RECIST: an equivocal new lesion later confirmed -> PD is dated to the visit where it first appeared
        backdated = None
        first_new = next((r for r in res if r.new_lesion), None)
        if first_new is not None:
            eq = [v for v in visits if v.equivocal_new_lesion and v.visit < first_new.visit]
            if eq:
                backdated = {"confirmed_at_visit": first_new.visit, "pd_dated_to_visit": eq[0].visit,
                             "note": "if the equivocal finding is the lesion confirmed later, PD dates back"}  # fmt: skip
        return {
            "pd_backdating": backdated,
            "target_rule_violations": self.target_rule_violations(),
            "baseline_sld_mm": sum(m.value() or 0 for m in base),
            "timepoints": [asdict(r) for r in res],
            "best_overall_response": recist.best_overall_response(res, confirm=confirm),
        }

    # ------------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return {
            "study": self.study.source,
            "visits": [{"index": v.index, "case_id": v.case_id, "day": v.day, "file": v.path.name} for v in self.study.visits],
            "lesions": [asdict(l) for l in self.lesions.values()],
            "judgements": {str(k): asdict(j) for k, j in self.judgements.items()},
            "next_id": self.next_id,
        }

    def save(self, out_dir) -> Path:
        out = Path(out_dir)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        (out / "registry.json").write_text(json.dumps(self.to_dict(), indent=2, default=_json))
        for idx, lm in self._maps.items():
            vis = self.study.visit(idx)
            img = sitk.GetImageFromArray(lm)
            img.CopyInformation(self.study.volume(vis).image)
            sitk.WriteImage(img, str(out / "masks" / f"{vis.case_id}.nii.gz"), useCompression=True)
        return out

    @classmethod
    def load(cls, study: Study, in_dir) -> "LesionRegistry":
        d = json.loads((Path(in_dir) / "registry.json").read_text())
        reg = cls(study)
        for l in d["lesions"]:
            f = {int(k): Finding(**v) for k, v in l.pop("findings").items()}
            reg.lesions[l["id"]] = Lesion(**l, findings=f)
        reg.judgements = {int(k): VisitJudgement(**v) for k, v in d.get("judgements", {}).items()}
        reg.next_id = int(d.get("next_id", max(reg.lesions, default=0) + 1))
        for vis in study.visits:
            p = Path(in_dir) / "masks" / f"{vis.case_id}.nii.gz"
            if p.exists():
                reg._maps[vis.index] = sitk.GetArrayFromImage(sitk.ReadImage(str(p))).astype(np.uint16)
        return reg


def _check(value: str, allowed: tuple) -> str:
    if value not in allowed:
        raise ValueError(f"{value!r} not in {allowed}")
    return value


def _json(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)
