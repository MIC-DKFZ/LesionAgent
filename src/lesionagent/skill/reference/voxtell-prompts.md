# VoxTell prompt sets

VoxTell segments whatever the text describes, giving one mask per prompt. `detect` merges all prompts into candidates.

- **Organ-specific prompts find far more lesions** than generic ones (e.g. `liver metastasis` rather than `lesion` or
  `tumor`).
- **Combine** an organ-specific prompt with a generic one. A candidate found by several prompts (high `support`) is usually real.
- **Precomputed embeddings:** common anatomical and lesion prompts are in VoxTell's bank. Unusual wording falls back to the text encoder, which is slower the first time.

## Per region

| Region | Prompts |
|---|---|
| Chest | `lung nodule`, `lung metastasis`, `lung tumor`, `pleural metastasis`, `enlarged mediastinal lymph node`, `hilar lymph node`, `axillary lymph node` |
| Abdomen | `liver metastasis`, `liver lesion`, `adrenal metastasis`, `kidney tumor`, `pancreatic tumor`, `spleen lesion`, `peritoneal metastasis`, `retroperitoneal lymph node`, `mesenteric lymph node` |
| Pelvis | `pelvic lymph node`, `iliac lymph node`, `inguinal lymph node`, `ovarian tumor`, `bladder tumor`, `peritoneal metastasis` |
| Bone and soft tissue | `bone metastasis`, `lytic bone lesion`, `soft tissue metastasis`, `subcutaneous metastasis` |
| Generic (always add) | `lesion`, `tumor`, `metastasis`, `lymph node metastasis` |

## Typical false positives

- vessels (especially portal and hepatic veins in the liver, and pulmonary vessels in cross-section)
- bowel loops
- the gallbladder
- renal cysts
- the adrenal limbs
- the oesophagus
- atelectasis and scarring
- calcified granulomas

Check the HU:
- simple cysts are 0–15 HU;
- calcification is > 200 HU;
- contrast-filled vessels have the same HU as the aorta.
