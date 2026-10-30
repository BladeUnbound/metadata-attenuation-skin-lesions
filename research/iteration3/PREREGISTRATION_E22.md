# Pre-registration: E22, second replication wave and a paired-modality test

*Written 27 Oct 2026, ~21:25 IST, while the ISIC Archive downloads were running and before any image from these collections was opened or any model fitted. Not to be edited after E22 runs.*

## Datasets (ISIC Archive public collections)
| Name | Collection | Images | Licence | Label | Grouping |
|---|---|---|---|---|---|
| HIBA | 251 | 1,616 | CC-BY | malignant vs benign (`diagnosis_1`) | `patient_id` |
| MSKCC 2020 | 216 | 1,295 | CC-BY | malignant vs benign | `lesion_id` if present, else the image |
| MILK10k, dermoscopic | 425 | images with `image_type` = dermoscopic | CC-BY-NC | malignant vs benign | `lesion_id` |
| MILK10k, clinical | 425 | images with `image_type` = clinical | CC-BY-NC | malignant vs benign | `lesion_id` |

Images whose `diagnosis_1` is neither Benign nor Malignant (e.g. Indeterminate) are excluded.

**Auxiliary variables:** age_approx, sex, anatom_site_1. For HIBA also fitzpatrick_skin_type. For MSKCC 2020, `clin_size_long_diam_mm` is analysed as a separate block (below).

**Excluded as label-encoding, decided now:** `diagnosis_confirm_type`, `concomitant_biopsy`, `melanocytic`, every `diagnosis_*` field.

**Missingness:** the pre-registered encoding is the E13 encoding (median imputation + missing indicator; "unknown" level). A complete-case analysis is reported alongside it, because missingness carried the label in all four earlier datasets.

## Protocol
Identical to E13: five-rung frozen ladder, out-of-fold two-stage added value, StratifiedGroupKFold(5) with seeds 0–2, inference on seed 0, grouped bootstrap with B = 2,000.

## Hypotheses
- **H1** (each dataset): Δ(HSV) − Δ(ConvNeXt-T) > 0. **H2:** Δ(ResNet-18) − Δ(ConvNeXt-T) > 0. Holm over all 8 tests (4 datasets × 2).
- **P1, paired modality (MILK10k):** on the *same lesions*, the retained fraction ρ at ConvNeXt-T is **higher for clinical than for dermoscopic images**, because clinical photographs show less lesion detail, so metadata stay less redundant. Tested with a lesion-level paired bootstrap of ρ_clin − ρ_derm, restricted to lesions that have both image types. Falsified if the 95% CI includes 0 or is negative.
- **P2, measured lesion size (MSKCC 2020):** diameter adds value over HSV, and this attenuates with capacity (retained fraction at ConvNeXt-T < 0.5), consistent with E12b's finding that deep image scores track lesion size. Exploratory: n is small.

## What counts as support
- H1 holds in ≥ 3 of 4 datasets (Holm): weak-vs-strong attenuation generalises to the second wave.
- P1 supported: imaging modality modulates the floor, refining the iteration-3 statement that the floor depends on the dataset.
- P1 not supported: the floor is not explained by modality, even within the same lesions.

## Addendum, written before any modelling (after reading HIBA's metadata only)
HIBA contains 340 `clinical: overview` images, which are regional body photographs rather than lesion images, plus 6 clinical close-ups. HIBA analysis is therefore restricted to `image_type = dermoscopic` (1,270 images before label filtering). No image was viewed and no model fitted when this was decided.
