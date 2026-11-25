"""Reading rules and thresholds of the workspace, in one place (mm, ml, HU, voxels).

The RECIST 1.1 arithmetic (sums, categories, measurability) lives in core/recist.py. These are the rules around it
that decide when the agent is asked to look again: a disagreeing second opinion, a tracking QA flag, a doubtful new
lesion, a decision-critical visit. Tool descriptions that quote a value are checked against it in the tests.
"""

# ------------------------------------------------------------------ second opinion (an nnInteractive click per mask)
SECOND_OPINION_MIN_MM = 5.0  # the long axes disagree beyond max(5 mm, 30 % of the model mask's long axis):
SECOND_OPINION_FRACTION = 0.3  # more than reader variability, so the choice changes the measurement
LEAK_ML = 150  # a click that segments more than this leaked: not a lesion, nothing to compare
FRAGMENTED_COMPONENTS = 5  # a model mask with more components (a cluster?) is not compared
SPLIT_MIN_ML = 0.1  # an accepted candidate with several separate components this large: one lesion per component

# ------------------------------------------------------------------ tracking QA (follow-up mask vs context mask)
VOLUME_RATIO_MAX = 3  # volume change beyond x3 or below x0.3 is flagged
VOLUME_RATIO_MIN = 0.3
THIN_SCAN_MM = 200  # scans with a shorter z-extent: LongiTrack's crop is mirror-padded instead of zero-padded

# ------------------------------------------------------------------ views
VIEW_MIN_MM = 100.0  # a view around a lesion / candidate shows at least this much context

# ------------------------------------------------------------------ candidate tables
MIN_CANDIDATE_VOXELS = 20  # smaller detector components are dropped
CANDIDATE_OVERLAP = 0.3  # overlap (fraction of a component's voxels) above which it joins an earlier candidate,
#                          or counts as covered by a registry lesion ('existing')

# ------------------------------------------------------------------ new lesions: reasons for 'equivocal'
NEW_LESION_SMALL_MM = 10.0  # long axis (node: short axis) below: small, the typical equivocal finding
NEW_LESION_GROWTH_MM = 3.0  # growth since first seen below: within measurement error, not a confirmation
NEW_LESION_NEAR_MM = 20.0  # another lesion this close: growth or coalescence of that one?
TOUCH_MM = 2.0  # this close: 'touches'
BASELINE_BACKCHECK_MM = 15.0  # a baseline lesion / candidate this close to the registered position: there before?

# ------------------------------------------------------------------ report
DECISION_MARGIN_MM = 5.0  # a visit this close to the PR or PD threshold is decision-critical

# ------------------------------------------------------------------ HU
AIR_HU = -400  # below: aerated lung / air (lung crop, click snapping, lung window for a finding)
BODY_HU = -500  # above: the patient (body crop of the sweep)
SNAP_MIN_HU = -300  # a click snapped along z must land on at least this (soft tissue, not air)
LUNG_MIN_ML = 20  # an air component inside the body larger than this is lung (not bowel gas)
LUNG_WINDOW_LOW_HU = -500  # a window starting below this is a lung-type window (sweep cropped to the lungs)
MIP_WINDOW_HIGH_HU = -200  # ... and ending above this: MIP slabs (nodules = dots); else mean (ground glass)
FINDING_WINDOW_HU = 150.0  # window around a finding's own median HU (+-)
LOW_HU = 20  # median HU below: cystic / necrotic? (target-selection flag)
