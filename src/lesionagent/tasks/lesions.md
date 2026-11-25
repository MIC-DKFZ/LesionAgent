Lesion detection task: ONE CT scan (a single visit v1; there is no follow-up). Find and segment EVERY lesion in this
scan, of any kind and any size: tumours, metastases, nodules, abnormal lymph nodes, and benign lesions such as cysts
or haemangiomas too. Name each lesion for what it is (e.g. "left_kidney_cyst"). Normal anatomy (vessels, bowel,
normal lymph nodes) and artefacts are not lesions.

Follow phases 0 and 1 of the workflow (detectors, the sweep of every imaged region for missed lesions, organs), but
the RECIST conventions do not apply to this task: there is no minimum size, and nothing is rejected or left out for
being too small, not measurable or benign. Use a small min_voxels in `detect` (e.g. 5), add prompts for benign
lesions (e.g. 'liver cyst', 'kidney cyst'), and look at the small candidates too. A lymph node is a lesion when it
is enlarged or looks abnormal (round, necrotic, no fatty hilum), also below 10 mm.

Report every lesion that you have looked at in a view centred on it: accept it and check its mask. Set `confidence`
with `update_lesion`: medium when you are not fully sure, low when it is only possibly a lesion. Reject a finding
only when it is probably normal anatomy or an artefact; when in doubt, keep it with low confidence, so that the
reader can filter by confidence. In an organ that already has lesions, a further focal finding is more likely a
lesion than not.

This is a detection and segmentation task only: no tracking, no target selection, no scores, no RECIST report. Finish
when every candidate is decided, with a short report: each lesion with organ, kind, size and confidence, and what a
radiologist should double-check.
