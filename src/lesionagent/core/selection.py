"""Target-lesion selection (RECIST 1.1 §4.2) as a constrained optimisation over agent-scored candidates.

RECIST: up to 5 targets, <= 2 per organ (both lungs/kidneys/adrenals = one organ, all lymph nodes = one organ),
measurable (>= 10 mm or 2x slice thickness; nodes >= 15 mm short axis), "selected on the basis of their size ...
representative of all involved organs ... and lend themselves to reproducible repeated measurements"; if the
largest is not reproducible, take the next largest that is.

Pipeline: candidate table (code) -> hard filters (code) -> agent flags + repro_score per candidate ->
exact enumeration of valid sets, ranked lexicographically by (organs covered, reproducibility-weighted size) ->
the agent picks (normally the top set) and justifies -> `category_stability` over near-tied sets.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from . import recist

ORGAN_ALIASES = {"left lung": "lung", "right lung": "lung", "kidney_left": "kidney", "kidney_right": "kidney",
                 "suprarenal": "adrenal", "adrenal gland": "adrenal", "rib": "bone", "vertebra": "bone"}  # fmt: skip
EXCLUDING_FLAGS = {"simple_cyst", "bone_without_soft_tissue", "effusion", "irradiated", "not_a_lesion"}
PENALTY_FLAGS = {"cystic": 0.3, "necrotic": 0.15, "ill_defined": 0.35, "confluent": 0.35, "motion_prone": 0.15,
                 "partial_volume": 0.2, "at_fov_edge": 0.3, "artefact": 0.4, "equivocal_malignancy": 0.4}  # fmt: skip


def organ_group(organ: str, is_node: bool) -> str:
    if is_node:
        return "lymph_nodes"
    o = (organ or "unknown").strip().lower()
    return ORGAN_ALIASES.get(o, o)


@dataclass
class SelCandidate:
    lesion_id: int
    name: str
    organ: str
    is_node: bool
    ld_mm: float
    sa_mm: float
    slice_thickness_mm: float
    flags: list = field(default_factory=list)
    repro_score: float | None = None   # agent, 0..1 (None = not scored yet)
    rationale: str = ""

    @property
    def group(self) -> str:
        return organ_group(self.organ, self.is_node)

    @property
    def size(self) -> float:
        return self.sa_mm if self.is_node else self.ld_mm

    def measurable(self) -> tuple[bool, str]:
        return recist.measurable(recist.Candidate(self.lesion_id, self.group, self.is_node, self.ld_mm, self.sa_mm,
                                                  self.slice_thickness_mm))  # fmt: skip

    def eligible(self) -> tuple[bool, str]:
        ok, why = self.measurable()
        if not ok:
            return False, why
        bad = EXCLUDING_FLAGS & set(self.flags)
        if bad:
            return False, f"excluded by flags {sorted(bad)}"
        return True, why

    def quality(self) -> float:
        """Reproducibility used for ranking: the agent's score, minus flag penalties when unscored."""
        if self.repro_score is not None:
            return float(np.clip(self.repro_score, 0, 1))
        return float(np.clip(1 - sum(PENALTY_FLAGS.get(f, 0) for f in self.flags), 0.05, 1))


@dataclass
class TargetSet:
    lesion_ids: list
    organs_covered: int
    score: float
    per_organ: dict
    notes: list = field(default_factory=list)


def enumerate_sets(cands: list[SelCandidate], k: int = 5, min_quality: float = 0.4, max_pool: int = 24) -> tuple[list[TargetSet], list[str]]:
    """Top-k valid target sets. Lexicographic: maximise organs covered, then sum(size_rank_weight * quality).
    Candidates with quality < min_quality only enter when their organ has no reproducible alternative
    (RECIST 1.1 §4.2: "the next largest lesion which can be measured reproducibly")."""
    log = []
    elig = []
    for c in cands:
        ok, why = c.eligible()
        if not ok:
            log.append(f"#{c.lesion_id} {c.name}: not eligible ({why})")
        else:
            elig.append(c)
    by_org: dict = {}
    for c in elig:
        by_org.setdefault(c.group, []).append(c)
    pool = []
    for g, cs in by_org.items():
        cs.sort(key=lambda c: -c.size)
        good = [c for c in cs if c.quality() >= min_quality]
        if not good:
            log.append(f"organ {g}: no candidate with reproducibility >= {min_quality}; using the best available")
            good = sorted(cs, key=lambda c: -c.quality())[:2]
        dropped = [c for c in cs if c not in good]
        for c in dropped:
            log.append(f"#{c.lesion_id} {c.name}: skipped, reproducibility {c.quality():.2f} < {min_quality}")
        pool += good[:4]  # more than 4 per organ can never matter for <= 2 per organ
    pool.sort(key=lambda c: -c.size)
    pool = pool[:max_pool]
    if not pool:
        return [], log + ["no eligible target lesions: non-target disease only"]
    n_organs = len({c.group for c in pool})
    max_size = max(c.size for c in pool)
    sets = []
    nmax = min(5, len(pool))
    for r in range(1, nmax + 1):
        for combo in itertools.combinations(pool, r):
            per: dict = {}
            for c in combo:
                per.setdefault(c.group, []).append(c.lesion_id)
            if any(len(v) > 2 for v in per.values()):
                continue
            covered = len(per)
            score = sum((c.size / max_size) * c.quality() for c in combo)
            sets.append(TargetSet(sorted(c.lesion_id for c in combo), covered, round(score, 4),
                                  {g: v for g, v in sorted(per.items())}))  # fmt: skip
    # prefer full organ coverage (up to 5 organs), then more total size*quality, then fewer lesions
    target_cov = min(n_organs, 5)
    sets.sort(key=lambda s: (-min(s.organs_covered, target_cov), -s.score, len(s.lesion_ids)))
    return sets[:k], log


def category_stability(evaluate_set, sets: list[TargetSet], delta: float = 0.15) -> dict:
    """Run the response for every near-tied set (score within `delta` of the best, same coverage).
    `evaluate_set(lesion_ids) -> list of overall responses per follow-up`."""
    if not sets:
        return {"sets": 0}
    best = sets[0]
    near = [s for s in sets if s.organs_covered == best.organs_covered and best.score - s.score <= delta * best.score]
    results = {tuple(s.lesion_ids): evaluate_set(s.lesion_ids) for s in near}
    ref = results[tuple(best.lesion_ids)]
    same = [k for k, v in results.items() if v == ref]
    pivotal = sorted(set(itertools.chain.from_iterable(k for k, v in results.items() if v != ref)) - set(best.lesion_ids))
    return {"sets": len(near), "stability": round(len(same) / len(near), 3), "reference": ref,
            "per_set": {",".join(map(str, k)): v for k, v in results.items()}, "lesions_that_flip": pivotal}  # fmt: skip
