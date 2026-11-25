"""Deterministic RECIST 1.1 engine (Eisenhauer 2009, Schwartz 2016 clarifications).

The agent supplies judgements (which lesions are targets, per-visit lesion status, non-target status,
whether a new lesion is unequivocal); everything numeric and every category is computed here.

Conventions
- Target sum (SLD) = sum of long axes of non-nodal targets + short axes of nodal targets.
- A target that is present but too small to measure counts as 5 mm; a disappeared lesion counts 0 mm; a target
  reported present without a positive measurement is not assessed (never 0 mm).
- PD (targets): SLD >= 1.2 x nadir AND SLD - nadir >= 5 mm. Nadir = smallest SLD so far, baseline included.
- PR: SLD <= 0.7 x baseline SLD. CR: all non-nodal targets gone and all target nodes < 10 mm short axis.
- Missing target measurements -> NE, unless the measured part alone already meets PD.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MEASURABLE_MM = 10.0       # non-nodal long axis (CT slice thickness <= 5 mm)
NODE_MEASURABLE_MM = 15.0  # nodal short axis
NODE_PATHOLOGIC_MM = 10.0  # nodal short axis; < 10 mm = normal
TOO_SMALL_MM = 5.0
MAX_TARGETS = 5
MAX_TARGETS_PER_ORGAN = 2
PR_THRESHOLD = -0.30
PD_THRESHOLD = 0.20
PD_ABSOLUTE_MM = 5.0


@dataclass
class TargetMeasurement:
    """One target lesion at one visit. status: present | too_small | absent | not_assessed."""

    lesion_id: int
    is_node: bool
    status: str = "present"
    ld_mm: float | None = None
    sa_mm: float | None = None

    def value(self) -> float | None:
        if self.status == "absent":
            return 0.0
        if self.status == "too_small":
            return TOO_SMALL_MM
        if self.status == "not_assessed":
            return None
        v = self.sa_mm if self.is_node else self.ld_mm
        return v if v is not None and v > 0 else None  # present without a measurement: not assessable


@dataclass
class Visit:
    visit: int
    day: int | None
    targets: list[TargetMeasurement]
    non_target_status: str = "NonCR/NonPD"   # CR | NonCR/NonPD | PD | NE | none (no non-target disease)
    new_lesion: bool = False                  # an unequivocal new lesion
    equivocal_new_lesion: bool = False


@dataclass
class TimepointResult:
    visit: int
    day: int | None
    sld_mm: float | None
    change_from_baseline: float | None
    change_from_nadir: float | None
    nadir_mm: float | None
    target_response: str | None
    non_target_response: str
    new_lesion: bool
    overall: str
    notes: list[str] = field(default_factory=list)
    # how far the SLD is from flipping the category (mm): useful for the agent to prioritise checks
    mm_to_pr: float | None = None
    mm_to_pd: float | None = None


def target_value_sum(ms: list[TargetMeasurement]) -> tuple[float | None, float, bool]:
    """(sum or None if any missing, partial sum of the measured ones, any missing)."""
    vals = [m.value() for m in ms]
    partial = float(sum(v for v in vals if v is not None))
    missing = any(v is None for v in vals)
    return (None if missing else partial), partial, missing


def _is_pd(s: float, nadir: float) -> bool:
    return s - nadir >= PD_ABSOLUTE_MM - 1e-9 and s >= (1 + PD_THRESHOLD) * nadir - 1e-9


def target_response(ms: list[TargetMeasurement], base: float, nadir: float) -> str:
    s, partial, missing = target_value_sum(ms)
    if missing:
        return "PD" if _is_pd(partial, nadir) else "NE"
    if _is_pd(s, nadir):
        return "PD"
    if all((m.status == "absent" or (m.value() or 0) < NODE_PATHOLOGIC_MM) if m.is_node else m.status == "absent"
           for m in ms):
        return "CR"
    if base > 0 and s <= (1 + PR_THRESHOLD) * base + 1e-9:
        return "PR"
    return "SD"


def overall_response(t: str | None, nt: str, new: bool) -> str:
    """RECIST 1.1 Tables 1 (targets present) and 2 (non-target disease only)."""
    if new or nt == "PD" or t == "PD":
        return "PD"
    if t is None:  # non-target disease only
        return {"CR": "CR", "NE": "NE", "none": "NE"}.get(nt, "NonCR/NonPD")
    if t == "CR":
        return "CR" if nt in ("CR", "none") else "PR"
    if t == "NE":
        return "NE"
    return t  # PR or SD (with non-target non-PD or NE)


def evaluate(baseline_targets: list[TargetMeasurement], visits: list[Visit]) -> list[TimepointResult]:
    """Timepoint responses for every follow-up visit (chronological). Visits after the first PD are still computed
    (marked in notes) but do not count for the best overall response."""
    has_targets = bool(baseline_targets)
    base, _, missing = target_value_sum(baseline_targets)
    if has_targets and missing:
        raise ValueError("every baseline target needs a measurement")
    base = base or 0.0
    nadir = base
    out = []
    after_pd = False
    for v in visits:
        notes = ["after the first PD: reported, not used for best overall response"] if after_pd else []
        t = None
        s = None
        if has_targets:
            ids = {m.lesion_id for m in baseline_targets}
            got = {m.lesion_id for m in v.targets}
            if ids - got:
                notes.append(f"targets not assessed at this visit: {sorted(ids - got)} -> NE unless PD")
                v.targets = v.targets + [TargetMeasurement(i, False, "not_assessed") for i in sorted(ids - got)]
            t = target_response(v.targets, base, nadir)
            s, partial, miss = target_value_sum(v.targets)
            if miss:
                s = None
        overall = overall_response(t, v.non_target_status, v.new_lesion)
        if v.equivocal_new_lesion and not v.new_lesion:
            notes.append("equivocal new lesion: follow up; if confirmed later, PD is dated to this visit")
        r = TimepointResult(
            v.visit, v.day, s,
            None if s is None or not base else (s - base) / base,
            None if s is None or not nadir else (s - nadir) / nadir,
            nadir if has_targets else None, t, v.non_target_status, v.new_lesion, overall, notes,
        )  # fmt: skip
        if s is not None and has_targets:
            r.mm_to_pr = s - (1 + PR_THRESHOLD) * base          # <= 0 means PR reached
            r.mm_to_pd = max((1 + PD_THRESHOLD) * nadir, nadir + PD_ABSOLUTE_MM) - s  # <= 0 means PD
            nadir = min(nadir, s)
        out.append(r)
        if overall == "PD":
            after_pd = True
    return out


def best_overall_response(results: list[TimepointResult], confirm: bool = False, min_sd_days: int = 42,
                          min_confirm_days: int = 28) -> str:  # fmt: skip
    """Best overall response. confirm=True: CR/PR must be seen again >= min_confirm_days later."""
    if not results:
        return "NE"
    first_pd = next((i for i, r in enumerate(results) if r.overall == "PD"), None)
    if first_pd is not None:
        results = results[: first_pd + 1]
    for rank in ("CR", "PR"):
        for i, r in enumerate(results):
            if r.overall != rank:
                continue
            if not confirm:
                return rank
            ok = ("CR",) if rank == "CR" else ("CR", "PR")
            if any(r2.overall in ok and (r2.day or 0) - (r.day or 0) >= min_confirm_days for r2 in results[i + 1:]):
                return rank
    if any(r.overall in ("CR", "PR", "SD", "NonCR/NonPD") and (r.day is None or r.day >= min_sd_days)
           for r in results):
        return "SD" if any(r.target_response for r in results) else "NonCR/NonPD"
    if any(r.overall == "PD" for r in results):
        return "PD"
    return "NE"


# --------------------------------------------------------------------------- target selection rules
@dataclass
class Candidate:
    lesion_id: int
    organ: str
    is_node: bool
    ld_mm: float
    sa_mm: float
    slice_thickness_mm: float = 5.0


def measurable(c: Candidate) -> tuple[bool, str]:
    if c.is_node:
        if c.sa_mm >= NODE_MEASURABLE_MM:
            return True, "node, short axis >= 15 mm"
        if c.sa_mm >= NODE_PATHOLOGIC_MM:
            return False, "pathological node 10-15 mm short axis: non-target"
        return False, "node < 10 mm short axis: normal, not recorded"
    need = max(MEASURABLE_MM, 2 * c.slice_thickness_mm)
    return (c.ld_mm >= need, f"long axis {'>=' if c.ld_mm >= need else '<'} {need:g} mm")


def validate_targets(targets: list[Candidate]) -> list[str]:
    """Rule violations of a proposed target set (empty list = valid)."""
    errs = []
    if len(targets) > MAX_TARGETS:
        errs.append(f"{len(targets)} targets > {MAX_TARGETS}")
    per = {}
    for c in targets:
        organ = "lymph_nodes" if c.is_node else c.organ  # all nodes count as one organ
        per.setdefault(organ, []).append(c.lesion_id)
        ok, why = measurable(c)
        if not ok:
            errs.append(f"lesion {c.lesion_id} not measurable ({why})")
    for organ, ids in per.items():
        if len(ids) > MAX_TARGETS_PER_ORGAN:
            errs.append(f"{len(ids)} targets in {organ} ({ids}) > {MAX_TARGETS_PER_ORGAN}")
    return errs


def suggest_targets(cands: list[Candidate]) -> list[int]:
    """Largest measurable lesions, <= 2 per organ (nodes = one organ), <= 5 total. A default, not a decision."""
    ok = [c for c in cands if measurable(c)[0]]
    ok.sort(key=lambda c: -(c.sa_mm if c.is_node else c.ld_mm))
    chosen, per = [], {}
    for c in ok:
        organ = "lymph_nodes" if c.is_node else c.organ
        if per.get(organ, 0) < MAX_TARGETS_PER_ORGAN and len(chosen) < MAX_TARGETS:
            chosen.append(c.lesion_id)
            per[organ] = per.get(organ, 0) + 1
    return chosen
