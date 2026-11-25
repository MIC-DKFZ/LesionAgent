Lung nodule detection task: ONE chest CT (screening, a single visit v1; there is no follow-up). Find and segment
EVERY lung nodule with a long axis >= 3 mm: solid, part-solid and ground-glass, small ones (3-6 mm) included, also
nodules at the pleura, the fissures or next to vessels; masses > 30 mm count too. Each nodule is its own lesion (one
lesion per nodule; split touching nodules). Use `detect` on v1 with lung prompts (VoxTell prompts such as 'lung
nodule', 'lung tumor', 'lung metastasis', 'ground glass nodule' exist) and a small min_voxels (e.g. 5: a 3 mm nodule
has only ~10-30 voxels). Check the candidates in the views (lung window) and accept the real nodules, reject
vessels, scars, atelectasis and other non-nodules. Then sweep both lungs for missed nodules (`view mode=sweep`, all
slabs from apex to base), ignore dots < 4 mm and vessels (tubes continuing across slabs), never skip large or
irregular masses, and segment each missed one with `segment` (a point inside it). Do not segment anything that is
not a lung nodule (no lymph nodes, no other organs). This is a detection task only: no tracking, no target
selection, no scores, no RECIST report. Finish when every nodule is segmented, with one line: 'nodules: N'.
