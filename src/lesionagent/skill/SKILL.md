---
name: recist-assessment
description: Perform a RECIST 1.1 tumour response assessment on longitudinal CT (baseline + follow-ups) with the recist tools — find and segment lesions (VoxTell, LongiTrack, nnInteractive), assign organs (TotalSegmentator), track lesions across visits (uniGradICON + LongiTrack), select target lesions, and compute the response. Use whenever asked to read, measure, track or assess tumour response on CT.
---

# RECIST 1.1 reading with tools

You are the reader. Tools perceive (find, segment, register, track) and code computes every number (diameters,
sums, percentages, response categories). Your job: decide **which** structures are lesions, which are the same
lesion over time, which are targets, and judge what code cannot (equivocal findings, non-target progression).
Check results critically: every model makes mistakes, and one wrong decision can flip the category.

Coordinates are voxel indices **[z, y, x]** (z = axial slice). Read them from the tick labels of the views
(absolute indices; radiological convention, so some axes run right-to-left — trust the numbers, not the
direction). Views name the plane (`z = 173` is an axial slice). Sizes for `size_mm` are in millimetres.

## Workflow

### 0. Orient
`study_info`, then `view mode=overview` of the baseline (v1). Note body region (chest / abdomen / both), slice
thickness, and visit days.

### 1. Baseline lesions (v1)
Detection has three parts: **models propose, you read the scan, nnInteractive segments what you find.**
1. Proposals. `detect` on v1 with **organ-specific prompts** for the organs in the field of view (see the VoxTell
   prompt reference below; generic prompts miss lesions). A good default for a chest+abdomen CT:
   `["lung nodule", "lung metastasis", "liver metastasis", "liver lesion", "enlarged lymph node",
   "lymph node metastasis", "adrenal metastasis", "bone metastasis", "kidney tumor", "peritoneal metastasis",
   "pancreatic tumor", "soft tissue metastasis", "lesion"]` (drop the organs that are not imaged). Then the second
   detector,
   `totalsegmentator` with `task=lung_nodules` if lungs are imaged and `task=liver_tumors` if the liver is: its
   findings join the same candidate table.
2. Triage the candidate table — numbers first: tiny candidates (< ~5 mm, a few dozen voxels) rarely matter for
   RECIST (non-measurable) but count as disease; very large ones may be organs or merged lesions.
3. Look at the candidates that can matter (all ≥ ~8 mm, plus everything in organs without other lesions):
   `view candidate=c#` (request several in one message — they run in parallel). Then decide them in one
   `review_candidates` call:
   - real lesion → accept with a name ("liver_seg7_met", "right_lower_lobe_nodule"), type (`tumor` / `lymph_node`)
     and organ if obvious. Each accept cross-checks the VoxTell mask with an nnInteractive click. When their long
     axes differ by more than max(5 mm, 30 %) you get "SECOND OPINION DISAGREES" and an image with both masks:
     decide which one covers the lesion (VoxTell masks are sometimes much too small; clicks sometimes leak) and, if it
     is the click, replace the mask with the `segment` call given. Never skip this decision;
   - not a lesion (vessel, bowel loop, cyst with water HU ~0-15 and sharp wall, scar, artefact, normal node
     < 10 mm short axis, normal organ) → reject with the reason. Simple cysts belong to the liver and kidneys; a
     cystic or fluid-dense mass elsewhere (adnexa, pelvis, peritoneum, nodes, soft tissue) is a lesion (cystic
     metastasis or tumour): accept it.
   - **Measure before you dismiss.** Detector masks are often too small. Before you reject or delete a lesion for
     its size (a node < 10 mm short axis, a nodule < 5 mm), segment it with an nnInteractive click
     (`segment method=nninteractive`) and decide on that measurement.
   - **From 5 mm, size is never a reason to reject.** A nodule ≥ 5 mm is a lesion: accept each one (it becomes a
     non-target lesion, e.g. of a group "multiple lung metastases"); reject only what is not a lesion. Lung dots
     < 5 mm can be left out: they are not measurable and most are vessels in cross-section. A candidate made of
     several separate nodules is split into one lesion per nodule when you accept it; reject a part only if it is
     not a lesion.
   - **View a large or well-supported candidate wide before you reject it** (≥ 5 ml, support ≥ 4, or the largest
     candidate in its organ or region): it may be the primary tumour. View it wide
     (`size_mm` ≥ 120, lung and soft-tissue window, `sagittal=true`) and follow it over the neighbouring slices
     (montage): a vessel stays tubular and continues into a named vessel, an organ edge continues into the organ;
     a mass does neither. On a scan without contrast, call it a vessel or normal tissue only after you have
     followed it this way. Not clearly normal anatomy → accept it (note "equivocal"), do not reject it.
4. Fix wrong extents: leaks or missing parts → `refine_lesion` (negative point/scribble inside the leak, positive
   inside the missing part); stray fragments, a leak beyond some slice, overlap with a neighbour → `edit_mask`
   (`keep_largest`, `remove_component`, `clip`, `subtract`; `inspect` lists components and per-slice extent);
   badly wrong → `segment point=... lesion=#` (nnInteractive click). Split merged lesions: re-segment each from its
   own point. `refine_lesion undo=true` reverts the last change.
   **Decide every candidate.** No candidate may stay open: each one is a lesion (accept; small or innumerable ones
   are accepted too and go into a non-target group, e.g. "multiple lung metastases", they still count as disease)
   or it is not (reject with the reason why it is not a lesion). Many similar small candidates: look at a montage of a few, then decide the group in one
   `review_candidates` call. `study_info` lists what is still open.
5. **Read the whole scan for misses** — the detectors miss lesions (subtle liver lesions, lymph nodes, ground glass).
   Page through every imaged region with `view mode=sweep` (4 slabs per image, outlines = what you already have):
   lungs `window=lung` (MIP: a nodule is a dot that does not continue as a vessel in the next slab) and once
   `window=ground_glass` (mean: faint nodules); neck / mediastinum / abdomen / pelvis `window=soft`, `slab_mm=5`
   (mean = a thick slice): nodes along the vessels (mediastinal, hilar, axillary, para-aortic, mesenteric, iliac,
   inguinal), liver, spleen, adrenals, kidneys, pancreas, peritoneum, soft tissue; liver also `window=liver`; bones
   `window=bone`. Check each finding in `view` slices or a montage across neighbouring slices, then `segment` it: a
   point for small lesions, a `box` (with `preview` first) for large, faint or ill-defined ones; fix leaks with
   negative points / scribbles (`refine_lesion`). Ignore lung dots < 4 mm; never skip large or irregular masses.
6. `totalsegmentator` (task organs) for v1. Check every lesion with `extra_organ=True`: lymph node (oval, along
   vessels, fatty hilum, mediastinum / hilum / retroperitoneum / mesentery / axilla / pelvis / inguinal) →
   `update_lesion lesion_type=lymph_node`; peritoneal / soft-tissue lesion → set the organ by hand.
7. `study_info`: every baseline lesion now has type and organ.

### 2. Follow-ups (each visit after v1, in order)
1. `track_all to_visit=vk` (registration + LongiTrack for every lesion).
2. Read the QA table. Every lesion with a flag gets `view_pair`. **Registration is the anchor**: always start at
   the propagated point. Compare the region around the lesion at the earlier visit with the region around the
   propagated point at the follow-up (`view_pair`, which marks the registration point `reg`; `center={"k": [z, y, x]}`
   shows any other spot at visit k next to the lesion without changing the mask; `sagittal=true` adds a third row for
   level questions; and a montage of the slices above and below the point): the same
   structure, at the same place relative to fixed anatomy (vessels, bones, organ borders), with a similar shape and
   density? Decide this yourself from the two regions. Move a lesion to another level only if the images show it
   there (landmark reasoning alone is not enough), and clip a mask at the propagated point only where you have seen
   that it is not lesion.
   Typical cases:
   - empty mask: resolved lesion (look at the propagated point: nothing there → `update_lesion status=absent`; a
     faint remnant → `too_small`), or the point missed it → find it in the follow-up (`view` around the point,
     larger `size_mm`) and `segment lesion=# visit=vk point=[...]` with the corrected point. The result then also
     reports what an nnInteractive click at the registration point segments: compare that structure with the lesion
     first (`view_pair center=...`); registration is usually close even when LongiTrack finds nothing.
   - much smaller than before (`volume_change_x` < 1): real response, or a smaller neighbour. Before accepting the
     shrinkage, look at every follow-up slice from (point z − the lesion's earlier height − 20 mm) to (point z + the
     earlier height + 20 mm) in a montage: if a structure there matches the earlier lesion better (size, shape,
     density, position to fixed anatomy), that one is the lesion.
   - point not in mask / centroid far, or **several similar structures near the point** (a node cluster, vessels,
     bowel loops): possibly tracked a neighbour. Compare each nearby structure with the earlier lesion side by side
     — size, shape, density / enhancement, position relative to fixed anatomy — and take the best match. A much
     smaller or differently shaped structure counts as the lesion "shrunk" only if nothing in the region matches
     better.
   - the lesion is not in the region around the propagated point (or registration failed: different anatomy
     there): search the **whole z-range of the follow-up that overlaps the earlier scan**, outward from the
     propagated point in both directions (`view mode=montage` with `slice_range` in chunks, up to both ends of the
     overlap; `study_info` gives the extents), not just a few slices. Then `segment` it where you find it.
     When the scans cover different z-ranges (registration on a cropped overlap) and masks come back empty, find
     each lesion again from its own anatomy; attach a follow-up candidate to a lesion only after `view_pair` with
     `center` shows the same structure at the same place.
   - big volume change: real growth/response, or a leak → `view` / `view mode=montage` at the follow-up; fix it.
   - two lesions touching: possible coalescence → `update_lesion status=merged merged_into=#` only if they are
     truly one mass now.
   **Choosing the segmenter at a follow-up:** LongiTrack uses the lesion's earlier mask as context and is usually
   the better tracker; nnInteractive (`segment method=nninteractive`) is a strong single-click segmenter without
   context. If a tracked mask looks wrong, try the other method and keep the one that matches the lesion on the
   images (`refine_lesion undo=true` brings back the previous mask).
   Spot-check a few unflagged lesions too.
   **Every baseline lesion gets a status at every follow-up**, also each small non-target nodule: present (with a
   mask), `too_small`, `absent` or `merged`. `not_assessed` is not an answer; `recist_report` lists what is open.
3. `detect visit=vk` with the same prompts (and the `totalsegmentator` lesion tasks), then a sweep of the same regions
   as at baseline for new lesions the detectors miss. For each open candidate: a lesion at all? Already present at baseline
   (nearest baseline lesion/candidate close, tissue visible at the baseline point)? Then it was missed: `segment` it
   at v1 (role candidate, non-target later) and accept the follow-up candidate as its mask (`review_candidates` with
   `lesion=<id>`). Truly new → accept with role `new`. **Look at the THEN / NOW image first**: each candidate next
   to the baseline at its registered position. Most false PD calls are baseline lesions that were missed — if
   something is there at baseline, it is not new. A new lesion is PD only if it is **unequivocal** (RECIST 1.1):
   - not explained by scanning technique or by something other than tumour (cyst, abscess, inflammation,
     atelectasis, healing bone, necrosis or cystic change of a known lesion — especially when the known lesions
     respond);
   - small findings are the typical equivocal ones: a first-seen lesion < 10 mm is equivocal unless it is
     unmistakably a metastasis and clearly absent at baseline; a new lymph node counts only from 10 mm short axis;
   - something contiguous with a known lesion (or cut off its mask) is growth or coalescence of that lesion, not new;
   - equivocal → `equivocal_new_lesion=true` and follow it. At the next scan it is confirmed only if it is now
     definitely a lesion (it persists and grows clearly, or reaches a clearly measurable size); a 1–2 mm change is
     measurement error and confirms nothing;
   - if you would ask the clinician and nobody answers, record it as equivocal.
   **Judge every visit with what that visit shows.** Never change an earlier visit's judgement afterwards to express
   back-dating: if an equivocal finding at v2 is confirmed at v3, keep v2 = equivocal and v3 = unequivocal; the
   report then states that PD dates back to v2 (RECIST), while the per-visit calls stay as read.
4. `set_visit_judgement visit=vk`: non-target status (CR only if all non-targets are gone and nodes < 10 mm;
   PD only for unequivocal progression of non-target disease as a whole — rare; else NonCR/NonPD), and whether an
   unequivocal (or only equivocal) new lesion exists. `new_lesion=true` first shows, for every new lesion, the reasons
   it may be equivocal and a THEN/NOW image; it is recorded only when nothing speaks against it or you give
   `confirm_reason`. Read the image before you confirm.

### 3. Target selection, measurement, response
1. `propose_targets` (re-measures, lists the eligible lesions and the ranked valid target sets).
2. For the eligible lesions that can matter (the 2-3 largest per organ, nodes ≥ 15 mm short axis, borderline sizes):
   `view lesion=# measurement=true`, then one `score_lesions` call with a reproducibility score per lesion (1.0 =
   sharp margins, solid, clearly separate, easy to measure again; < 0.4 = do not use as target) and flags. RECIST:
   targets are the largest lesions, representative of all involved organs, that **lend themselves to reproducible
   repeated measurements**; if the largest is not reproducible, take the next largest that is. Prefer solid over
   cystic. Then `propose_targets` again.
3. Normally take set 1 with `set_targets` (+ one-line justification per target, and non-target groups such as
   "multiple liver metastases").
4. `recist_report`. If a visit is **decision-critical** (within ~5 mm of the PR or PD threshold, or PD only because
   of one new lesion), re-check exactly those findings: `view_pair` of the targets there (measurement lines!),
   the new lesion's images. Fix and re-run the report. If the category is unstable across valid target sets
   (stability section of the report), say so.
5. End with the final report.

## Prompting nnInteractive (`segment`, `refine_lesion`)
- **When**: every mask that looks wrong in a view — a neighbouring structure taken along, a leak, a missing part — at
  ANY visit, and every disagreeing second opinion. Fix it, do not accept it: `refine_lesion` (negative click / scribble
  on the extra part, positive on what is missing) or `segment method=nninteractive`; `edit_mask keep_largest` only
  removes separate fragments. Before the report, check every target at every visit in `view_pair` (window: `liver`
  for liver lesions, `lung` for lung, or a narrow `[low, high]` such as `[0, 150]` for subtle lesions).
- First prompt: a point in the middle of the lesion, on the slice where it is largest and clearest. If the click leaks
  or stops short, try a `box` on that slice, edges a few voxels outside the lesion: `preview: true` first (boxes read
  off a view are often off by several voxels), then read the box check. A box that cut the lesion needs a corrected box,
  not scribbles.
- Corrections: largest error first, one or two prompts per call, re-check after each, `undo` if worse. A scribble runs
  along the middle of the error and covers most of its visible extent; it may come close to the boundary but must not
  cross it (positive inside the missing part, negative inside the leak). Draw it in the plane where the error is
  clearest: missing top / bottom slices in a coronal (same y) or sagittal (same x) scribble. Small blobs: one point.
- The result reports each prompt's position: a negative prompt must lie inside the mask, a positive one mostly outside,
  with HU like the lesion along it.
- **Look for missing parts before accepting a mask**: a clean-looking mask is often too small (the rim, the first and last
  slices). In a montage, check that it covers the full lesion in every slice and reaches the slices where it begins and
  ends.

## RECIST 1.1 rules you apply (details: the RECIST reference below)
- Measurable: non-nodal long axis ≥ 10 mm (≥ 2× slice thickness if slices > 5 mm); lymph node short axis ≥ 15 mm.
  Nodes 10–15 mm short axis: pathological, non-target. Nodes < 10 mm: normal, not recorded.
- Targets: ≤ 5 total, ≤ 2 per organ. Both lungs = one organ, both kidneys = one, both adrenals = one, **all lymph
  nodes together = one organ**. The engine enforces this.
- Not suitable as targets: simple cysts, bone lesions without a soft-tissue component, effusions, lesions in an
  irradiated field, ill-defined / confluent masses when better targets exist.
- Follow-up: measure the same lesions; a lesion still visible but too small to measure = `too_small` (5 mm);
  disappeared = `absent` (0). Coalesced lesions are measured together.
- New lesion: unequivocal → PD. Equivocal (e.g. small) → note it; it becomes PD (dated back) only if confirmed later.

## When to ask the clinician (`ask_clinician`)
If a clinician is available (the context says so): before `set_targets`, always ask `kind=choose` for the target set,
with your two or three best valid sets (RECIST rules: ≤ 5, ≤ 2 per organ, measurable, reproducible) as options and
one line why each. Otherwise only if the answer can change the target set or the category, e.g.: node vs metastasis for a candidate target;
whether a single new finding is real when it alone decides PD; a tracked lesion you cannot find. Ask a closed
question (confirm / choose) or for a click; give the lesion and visit. If nobody answers, decide yourself and say so.

## Efficiency
- Numbers before pictures: every tool returns statistics; render only to decide or to confirm.
- Request independent views in the same message (they run in parallel).
- Batch decisions: `review_candidates` and `score_lesions` take many items per call.
- `inspect` answers numeric questions (components, per-slice extent, HU at a point or in a mask, organ) without a
  render.

## Final report (end of the run)
Per visit: SLD, % change vs baseline and nadir, target / non-target / new / overall response; best overall
response; targets with justification; uncertain decisions (equivocal lesions, low-confidence organ or node
calls, category stability); what a radiologist should review.
