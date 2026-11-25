"""Agent-facing tools: names, descriptions, JSON schemas (one list for every front end).

A front end (the in-process Agent SDK MCP server, a viewer) only translates between its transport and
`call_tool`. The descriptions carry the mechanics; the `recist-assessment` skill carries the strategy.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from inspect import signature

from .workspace import AgentError, RecistWorkspace, Result

PT = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3,
      "description": "voxel index [z, y, x] (read from the tick labels of the views)"}  # fmt: skip
VISIT = {"type": ["integer", "string"], "description": "visit index (1 = baseline) or 'v2'"}
LESION = {"type": ["integer", "string"], "description": "lesion id (e.g. 3) or its unique name"}
WINDOW = {"type": ["string", "array"],
          "description": "'soft' | 'lung' | 'bone' | 'liver' | 'ground_glass' or [low, high] HU"}
IDS = {"type": "array", "items": {"type": ["integer", "string"]}}


@dataclass
class ToolSpec:
    """A tool is the workspace method of the same name, called with the schema's properties as keywords."""

    name: str
    description: str
    properties: dict
    required: list = field(default_factory=list)
    read_only: bool = False
    phase: str = ""
    fixed: dict = field(default_factory=dict)  # keyword arguments of every tool call (not in the schema)

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": self.properties, "required": self.required,
                "additionalProperties": False}  # fmt: skip

    def run(self, ws: RecistWorkspace, args: dict) -> Result:
        """ws.<name>(**args): keys outside the schema are ignored, omitted ones take the method's defaults, and a
        missing argument without a default raises KeyError (like args[key])."""
        fn = getattr(ws, self.name)
        kw = {k: v for k, v in args.items() if k in self.properties}
        for p in signature(fn).parameters.values():
            if p.default is p.empty and p.name not in kw:
                raise KeyError(p.name)
        return fn(**kw, **self.fixed)


@dataclass
class ToolOutput:
    text: str
    images: list = field(default_factory=list)
    is_error: bool = False
    data: dict = field(default_factory=dict)


def T(name, desc, props, req=(), ro=False, phase="", **fixed):
    return ToolSpec(name, desc, props, list(req), ro, phase, fixed)


TYPE = {"type": "string", "enum": ["tumor", "lymph_node", "unknown"]}
ROLE = {"type": "string", "enum": ["candidate", "target", "non_target", "new", "rejected"]}

TOOLS = [
    # ------------------------------------------------------------------ study / inspect (read-only)
    T("study_info", "The study (visits, days, geometry), progress through the RECIST phases, and the lesion table "
      "(LD/SA/volume/status per visit; only `visit` if given).",
      {"visit": VISIT}, ro=True, phase="study"),
    T("view", "Render one visit. mode 'slices' (default): orthogonal slices (z = axial, y = coronal, x = sagittal) "
      "through `center`, a `lesion` or a `candidate`; 'montage': up to 16 slices along `axis` (the fastest 3D check of "
      "a mask); 'overview': coronal + axial MIPs of the whole visit with all outlines (orientation, completeness); "
      "'sweep': read the scan systematically — consecutive axial slabs (slab_mm, default 10) through slice_range, any "
      "window, projection mip / mean / slice, 4 large panels per image (cropped to the lungs for lung windows, else to "
      "the body; page with slice_range): finds lesions detection missed. "
      "Lesions: coloured contours '#id name' (same colour at every visit); highlighted lesion: yellow fill; candidates: "
      "dashed white 'c#'; magenta + = the lesion's point at this visit. `measurement`: RECIST long axis (yellow) and "
      "short axis (cyan) on the long-axis slice. Radiological convention; trust the tick labels (absolute voxel "
      "indices).",
      {"visit": VISIT, "mode": {"type": "string", "enum": ["slices", "montage", "overview", "sweep"]}, "center": PT,
       "projection": {"type": "string", "enum": ["mip", "mean", "slice"],
                      "description": "sweep: slab projection (default mip for the lung window, else mean)"},
       "lesion": LESION, "candidate": {"type": "integer"},
       "size_mm": {"type": "number", "description": "field of view edge in mm (default: lesion + margin, else 100)"},
       "window": WINDOW, "axes": {"type": "array", "items": {"type": "integer", "enum": [0, 1, 2]},
                                  "description": "slices mode: which planes"},
       "axis": {"type": "integer", "enum": [0, 1, 2], "description": "montage: slicing axis (0 = axial)"},
       "n": {"type": "integer", "description": "montage: number of slices (<= 16)"},
       "slice_range": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2},
       "show_candidates": {"type": "boolean"}, "measurement": {"type": "boolean"},
       "slab_mm": {"type": "number", "description": "sweep: slab thickness in mm (default 10)"}},
      ro=True, phase="inspect"),  # fmt: skip
    T("view_pair", "The same lesion at several visits side by side (rows: axial, coronal, optionally sagittal), with "
      "its point, the registration point (reg) and measurement lines. The key view for checking tracking and change. "
      "`center` {visit: [z, y, x]} centres that visit on another spot (the registration point, a neighbouring "
      "structure) without changing the mask: compare candidates for the same lesion before you segment one.",
      {"lesion": LESION, "visits": {"type": "array", "items": VISIT}, "size_mm": {"type": "number"},
       "window": WINDOW, "coronal": {"type": "boolean"}, "sagittal": {"type": "boolean"},
       "center": {"type": "object", "additionalProperties": PT,
                  "description": "visit -> [z, y, x], e.g. {\"2\": [40, 210, 250]}"}},
      ["lesion"], ro=True, phase="inspect"),  # fmt: skip
    T("inspect", "Numbers, no image. For a `lesion` or `candidate` mask at a visit: voxels, volume, LD/SA, z-range, "
      "connected components (size, centroid; numbered largest first), extent per slice, HU inside and in a rim "
      "outside, organ at its centre, touching lesions, status / provenance / QA (incl. the second opinion and tracking "
      "flags). For a `point`: HU at the point and in a ball of `radius_mm`, and which lesion / candidate / organ "
      "contains it.",
      {"visit": VISIT, "lesion": LESION, "candidate": {"type": "integer"}, "point": PT, "radius_mm": {"type": "number"}},
      ro=True, phase="inspect"),  # fmt: skip
    # ------------------------------------------------------------------ lesions (any visit)
    T("detect", "Zero-shot lesion detection with VoxTell (text-prompted 3D segmentation, ~25 s) on one visit. Use "
      "organ-specific prompts ('liver metastasis', 'lung nodule', 'enlarged lymph node', 'adrenal metastasis', "
      "'bone metastasis', 'peritoneal metastasis', 'kidney tumor', 'pancreatic tumor', ...): generic prompts miss "
      "lesions. The union of all prompts is split into candidates c1, c2, ... (largest first) with size, HU and the "
      "prompts that found them; earlier candidates of that visit and their decisions are kept. At a follow-up visit only candidates not "
      "covered by tracked lesions are returned, each mapped back to the baseline (registration): nearest baseline "
      "lesion / candidate, HU there, and a THEN / NOW image (each candidate next to the baseline at its mapped "
      "position), to tell a new lesion from a missed baseline lesion.",
      {"visit": VISIT, "prompts": {"type": "array", "items": {"type": "string"}}, "min_voxels": {"type": "integer"}},
      phase="baseline"),
    T("review_candidates", "Accept or reject candidates of one visit, several per call. Accept turns c# into a lesion "
      "(give name, type, organ if known; role 'new' for a new lesion at a follow-up; `lesion` = use it as that "
      "existing lesion's mask at this visit); a candidate of several separate nodules (each >= 0.1 ml) becomes one "
      "lesion per nodule. Each accepted VoxTell mask is cross-checked by an nnInteractive click "
      "(second opinion): if the long axes differ by > max(5 mm, 30 %), both masks are shown side by side and you must "
      "decide which is right. Reject: not a lesion (vessel, bowel, cyst, normal "
      "tissue, artefact...) with a reason; never a nodule >= 5 mm for being small.",
      {"visit": VISIT, "decisions": {"type": "array", "items": {"type": "object", "properties": {
          "candidate": {"type": "integer"}, "accept": {"type": "boolean"}, "name": {"type": "string"},
          "lesion_type": TYPE, "organ": {"type": "string"}, "role": {"type": "string", "enum": ["candidate", "new"]},
          "lesion": LESION, "notes": {"type": "string"}, "reason": {"type": "string"}},
          "required": ["candidate", "accept"]}}},
      ["decisions"], phase="baseline"),  # fmt: skip
    T("segment", "Segment a lesion at a visit from ONE point inside it (default: the lesion's stored point there, e.g. "
      "the propagated one) — a new lesion, or re-segment `lesion` at this visit. method 'longitrack': LongiTrack "
      "with the lesion's mask at another visit as longitudinal context (tracking; returns QA flags); 'nninteractive': "
      "one positive click, no context (strong on new lesions); 'auto' (default): LongiTrack when the lesion has a "
      "mask at another visit, else nnInteractive. When unsure, try both and keep the one that looks right "
      "(refine_lesion undo=true reverts). `box` instead of `point`: an nnInteractive box prompt, two inclusive corners "
      "in one slice ([[z, y0, x0], [z, y1, x1]]), tight around the lesion where it is largest; it bounds the extent "
      "when a click leaks or stops short. A box read off a view is often off by several voxels: first call with "
      "`preview: true` (draws it, segments nothing); after the box a fit check reports where the mask sticks out of "
      "the box or leaves it empty.",
      {"point": PT, "visit": VISIT, "lesion": LESION,
       "method": {"type": "string", "enum": ["auto", "longitrack", "nninteractive"]}, "name": {"type": "string"},
       "lesion_type": TYPE, "organ": {"type": "string"}, "role": {"type": "string", "enum": ["candidate", "new"]},
       "notes": {"type": "string"},
       "box": {"type": "array", "items": PT, "minItems": 2, "maxItems": 2,
               "description": "two inclusive corners that share one coordinate (the slice)"},
       "preview": {"type": "boolean", "description": "box only: draw it, do not segment"}},
      phase="baseline"),  # fmt: skip
    T("refine_lesion", "Correct a lesion mask with nnInteractive, starting from its current mask: positive points / "
      "scribbles add missing parts, negative ones remove leaks. A scribble is a polyline in ONE slice of any plane "
      "(its points share z = axial, y = coronal or x = sagittal), drawn 2 px wide; each prompt is reported with the "
      "fraction inside the current mask, HU along it and the voxels it changed. `undo`: revert the last mask change "
      "of this lesion at this visit (segment / refine / edit).",
      {"lesion": LESION, "visit": VISIT,
       "points": {"type": "array", "items": {"type": "object", "properties": {"point": PT, "positive": {"type": "boolean"}},
                                             "required": ["point", "positive"]}},
       "scribbles": {"type": "array", "items": {"type": "object", "properties": {
           "points": {"type": "array", "items": PT, "minItems": 2}, "positive": {"type": "boolean"}},
           "required": ["points", "positive"]}},
       "undo": {"type": "boolean"}},
      ["lesion"], phase="baseline"),  # fmt: skip
    T("edit_mask", "Deterministic edits of a lesion mask, applied in order: {op: keep_largest} (largest 3D "
      "component), {op: fill_holes}, {op: clip, axis, min, max} (keep indices min..max along axis 0 = z, 1 = y, "
      "2 = x), {op: subtract, lesion} (remove another lesion's voxels), {op: remove_component, component} (numbering "
      "from `inspect`). For fragments, leaks along one axis and overlaps; refine_lesion undo=true reverts.",
      {"lesion": LESION, "visit": VISIT, "ops": {"type": "array", "items": {"type": "object", "properties": {
          "op": {"type": "string", "enum": ["keep_largest", "fill_holes", "clip", "subtract", "remove_component"]},
          "axis": {"type": "integer", "enum": [0, 1, 2]}, "min": {"type": "integer"}, "max": {"type": "integer"},
          "lesion": LESION, "component": {"type": "integer"}}, "required": ["op"]}}},
      ["lesion", "ops"], phase="baseline"),
    T("update_lesion", "Edit the registry: a lesion's name, type, organ, role, notes or confidence; its status at "
      "`visit`: present | too_small (counts 5 mm) | absent (0 mm) | not_assessed | merged (coalesced with "
      "`merged_into`, measured as one there); or `delete` it at all visits.",
      {"lesion": LESION, "visit": VISIT, "name": {"type": "string"}, "lesion_type": TYPE, "organ": {"type": "string"},
       "role": ROLE, "notes": {"type": "string"}, "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
       "status": {"type": "string", "enum": ["present", "too_small", "absent", "merged", "not_assessed"]},
       "merged_into": LESION, "delete": {"type": "boolean"}},
      ["lesion"], phase="baseline"),  # fmt: skip
    T("totalsegmentator", "TotalSegmentator. task 'organs' (default, ~20 s per visit): assigns each lesion to an organ "
      "group (lung, liver, kidney, adrenal, bone, ...) and reports `extra_organ` (outside parenchymal organs: node / "
      "peritoneal / soft-tissue suspect) and the body cavity; sets the organ of lesions that have none (or all with "
      "overwrite). Lesion tasks — a second detector besides VoxTell: 'lung_nodules', 'liver_tumors' (~30-60 s): their "
      "findings join the visit's candidate table (new c#, or the source added to an overlapping VoxTell candidate), "
      "decided with review_candidates like detect's.",
      {"visit": VISIT, "task": {"type": "string", "enum": ["organs", "lung_nodules", "liver_tumors"]}, "lesions": IDS,
       "overwrite": {"type": "boolean"}},
      phase="baseline"),  # fmt: skip
    # ------------------------------------------------------------------ follow-up
    T("track_all", "Track lesions into a follow-up visit: uniGradICON registration from_visit -> to_visit (~15 s; "
      "field-of-view mismatch handled by cropping) propagates each lesion's point, then LongiTrack segments each "
      "lesion there with its previous mask as context. All baseline lesions, or only `lesions`. Returns size, volume "
      "ratio and QA flags per lesion (empty mask, big volume change, point not in mask, several components, touches "
      "other lesions).",
      {"to_visit": VISIT, "from_visit": VISIT, "lesions": IDS}, ["to_visit"], phase="followup"),
    T("set_visit_judgement", "Reader judgements per follow-up visit: non-target status (CR | NonCR/NonPD | PD | NE | none),"
      " whether there is an UNEQUIVOCAL new lesion, or only an equivocal one — as seen AT THIS VISIT. new_lesion=true first "
      "shows the evidence for every new lesion (size, growth, contiguity, baseline back-check with a THEN/NOW image) and "
      "is recorded only if nothing speaks for 'equivocal' or `confirm_reason` says why it is unequivocal. Do not edit an "
      "earlier visit to back-date PD: the report derives back-dating from equivocal-then-confirmed findings.",
      {"visit": VISIT, "non_target_status": {"type": "string", "enum": ["CR", "NonCR/NonPD", "PD", "NE", "none"]},
       "new_lesion": {"type": "boolean"}, "equivocal_new_lesion": {"type": "boolean"}, "notes": {"type": "string"},
       "confirm_reason": {"type": "string", "description": "why a new lesion is unequivocal despite the points the "
                          "tool raised (size, no growth, contiguity, something at baseline)"}},
      ["visit"], phase="followup", gate=True),  # fmt: skip
    # ------------------------------------------------------------------ targets + RECIST
    T("score_lesions", "Record your judgement of how reproducibly lesions can be measured (repro_score 0..1) with "
      "flags: cystic, necrotic, ill_defined, confluent, motion_prone, partial_volume, at_fov_edge, artefact, "
      "equivocal_malignancy; excluding: simple_cyst, bone_without_soft_tissue, effusion, irradiated, not_a_lesion.",
      {"scores": {"type": "array", "items": {"type": "object", "properties": {
          "lesion": LESION, "repro_score": {"type": "number"}, "flags": {"type": "array", "items": {"type": "string"}},
          "rationale": {"type": "string"}}, "required": ["lesion", "repro_score"]}}},
      ["scores"], phase="selection"),  # fmt: skip
    T("propose_targets", "Re-measures all masks, then lists the baseline lesions for target selection (organ group, "
      "size: LD, nodes short axis; measurability; automatic flags: low HU, components, FOV edge; your flags and "
      "scores) and the exact enumeration of valid target sets (<= 5, <= 2 per organ, all nodes = one organ, "
      "measurable), ranked by organ coverage, then size x reproducibility. Normally take set 1.",
      {"k": {"type": "integer"}, "min_quality": {"type": "number"}}, phase="selection"),  # remeasures: not read-only
    T("set_targets", "Set the target lesions (all other baseline lesions become non-targets), with a short "
      "justification and optional named non-target groups (e.g. 'multiple liver metastases'). Refused if it breaks "
      "RECIST rules.",
      {"lesions": IDS, "justification": {"type": "string"}, "non_target_groups": {"type": "array", "items": {
          "type": "object", "properties": {"label": {"type": "string"}, "lesions": IDS}, "required": ["label", "lesions"]}}},
      ["lesions"], phase="selection"),  # fmt: skip
    T("recist_report", "Compute RECIST 1.1 (code, not you) after re-measuring every mask: sum of diameters, nadir, % "
      "change, target / non-target / new / overall response per visit, best overall response, mm to the PR/PD "
      "thresholds, DECISION-CRITICAL visits, rule violations, and whether the category is stable over "
      "near-equivalent valid target sets (lesions whose choice flips it). Saves registry, masks and report.",
      {"confirm": {"type": "boolean", "description": "require confirmation of CR/PR (non-randomised trials)"}},
      phase="report"),
    # ------------------------------------------------------------------ clinician
    T("ask_clinician", "Ask the clinician, ONLY when the answer can change the target set or the response category. "
      "kind: confirm (yes/no), choose (pick one of `options`), click (they mark a lesion at `visit`; returns a point),"
      " verify_point (they check/move `point`). May be unavailable: then decide yourself.",
      {"kind": {"type": "string", "enum": ["confirm", "choose", "click", "verify_point"]}, "question": {"type": "string"},
       "lesion": LESION, "visit": VISIT, "point": PT, "options": {"type": "array", "items": {"type": "string"}}},
      ["kind", "question"], phase="clinician"),  # fmt: skip
]
TOOLS_BY_NAME = {t.name: t for t in TOOLS}


def phase_of(ws: RecistWorkspace, spec: ToolSpec | None, args: dict) -> str:
    """Lesion tools work at any visit: their phase is 'followup' when they act on a follow-up visit."""
    if spec is None:
        return ""
    if spec.phase == "baseline" and args.get("visit") is not None:
        try:
            if ws.study.visit(args["visit"]).index != ws.study.baseline.index:
                return "followup"
        except (KeyError, ValueError, TypeError):
            pass
    return spec.phase


_GPU_BUSY = ("out of memory", "out of gpu memory", "busy or unavailable", "cuda error")
GPU_RETRIES = int(os.environ.get("LESIONAGENT_GPU_RETRIES", 3))
GPU_RETRY_WAIT_S = 20.0


def _with_gpu_retry(fn, tries: int | None = None, wait_s: float = GPU_RETRY_WAIT_S):
    """Another process may hold the GPU: on a CUDA out-of-memory or busy error free the cache, wait and try again
    before the error goes to the agent. `LESIONAGENT_GPU_RETRIES` sets the number of tries (default 3)."""
    tries = max(1, GPU_RETRIES if tries is None else tries)
    for k in range(tries):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            from .backends import models

            if not (models.is_oom(e) or any(m in str(e).lower() for m in _GPU_BUSY)) or k == tries - 1:
                raise
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            print(f"GPU busy or out of memory ({type(e).__name__}); retry {k + 1} of {tries - 1} in {wait_s:.0f} s",
                  file=sys.stderr, flush=True)  # fmt: skip
            time.sleep(wait_s)


def _release_memory() -> None:
    """Give freed memory back to the OS after a tool call: volumes, masks and model inputs are large, and glibc
    keeps freed blocks per thread arena."""
    import ctypes
    import gc

    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except OSError:
        pass


def call_tool(ws: RecistWorkspace, name: str, args: dict | None) -> ToolOutput:
    args = args or {}
    spec = TOOLS_BY_NAME.get(name)
    phase = phase_of(ws, spec, args)
    t0 = time.time()
    ws.emit("tool_start", tool=name, args=args, phase=phase)
    if spec is None:
        out = ToolOutput(f"unknown tool {name!r}", is_error=True)
    else:
        try:
            res = _with_gpu_retry(lambda: spec.run(ws, args))
            out = ToolOutput(res.text, res.images, data=res.data)
        except (AgentError, ValueError, KeyError, FileNotFoundError, TypeError) as e:
            out = ToolOutput(f"{type(e).__name__}: {e}", is_error=True)
        except Exception as e:  # noqa: BLE001
            out = ToolOutput(f"internal error {type(e).__name__}: {e}\n{traceback.format_exc(limit=4)}", is_error=True)
    _release_memory()
    dt = time.time() - t0
    ws.emit("tool_end", tool=name, args=args, seconds=round(dt, 2), error=out.is_error, text=out.text,
            images=out.images, phase=phase)  # fmt: skip
    try:
        with open(ws.workdir / "log.jsonl", "a") as f:
            f.write(json.dumps({"time": time.strftime("%H:%M:%S"), "tool": name, "args": args, "seconds": round(dt, 2),
                                "error": out.is_error, "text": out.text[:4000], "images": len(out.images)},
                               default=str) + "\n")  # fmt: skip
    except OSError:
        pass
    return out
