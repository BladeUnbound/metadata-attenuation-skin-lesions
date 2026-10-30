# Pre-registration: confirmatory replication of E10 on independent datasets (E13)

*Written 27 Oct 2026, before any ISIC 2020, ISIC 2019 or PAD-UFES-20 image was downloaded or any model fitted on them. Not to be edited after E13 runs. Deviations go in EXPERIMENTS.md, labelled as such.*

## Datasets and tasks

| Dataset | Task | Grouping unit | Auxiliary blocks | Notes |
|---|---|---|---|---|
| ISIC 2020 (SIIM-ISIC; 256 px JPEG mirror) | Melanoma vs rest | `patient_id` | age, sex, anatomical site | About 1.8% prevalence |
| ISIC 2019, **excluding HAM10000** (`lesion_id` starting `HAM_`) | Melanoma vs rest | `lesion_id`; the image itself if missing | age, sex, anatomical site | Independent of HAM10000 images |
| PAD-UFES-20 (smartphone) | Malignant (BCC, SCC, MEL) vs benign (ACK, SEK, NEV) | `patient_id` | **Block D** (demographic/site): age, gender, region, Fitzpatrick type. **Block S** (symptoms/history): itch, grew, hurt, changed, bleed, elevation | Clinical photographs, not dermoscopy |

Missing values: numeric fields are imputed with the training-fold median plus a missing indicator; categorical fields get an "unknown" level. Any field that encodes how the diagnosis was established (e.g. `biopsed` in PAD-UFES-20, a direct analogue of HAM10000's `dx_type`) is **excluded**.

## Protocol (identical to E10 except as noted)
- Ladder: HSV histogram, ResNet-18, ResNet-50, ConvNeXt-Tiny, ViT-B/16 (frozen ImageNet weights; pooled features; resize shorter side to 224 and centre-crop).
- Image score: out-of-fold L2 logistic regression on standardised features, penalty chosen by grouped inner 3-fold CV inside each training fold.
- Second stage: [score] vs [score + block], out of fold; added value = mean per-image log-loss difference.
- Outer CV: 5-fold StratifiedGroupKFold, 3 fold assignments (seeds 0, 1, 2). Inference uses seed 0; the other seeds check stability.
- Uncertainty: grouped bootstrap (2,000 resamples of grouping units).

## Confirmatory hypotheses (one-sided, Holm-corrected over all six H1/H2 tests at family α = 0.05)
- **H1** (each dataset): AV(HSV) − AV(ConvNeXt-T) > 0, with all auxiliary variables pooled (Block D + S for PAD).
- **H2** (each dataset): AV(ResNet-18) − AV(ConvNeXt-T) > 0.
- p-values from the grouped bootstrap: the share of resamples with a difference ≤ 0, with a +1 correction.

## Mechanism hypotheses (PAD-UFES-20 only; reported separately, uncorrected)
- **M1:** Block S (symptoms) keeps at least half its HSV-level added value at ConvNeXt-T: retained fraction ≥ 0.5.
- **M2:** Block D attenuates more than Block S: retained fraction D < retained fraction S.
- **M3:** Symptoms are poorly recoverable from images: mean AUROC for predicting the symptom fields from ConvNeXt features < 0.65.

If M1–M3 hold, the attenuation in E10 is explained by recoverability: what the image can show, metadata stops adding; what it can't show, metadata keeps adding. If Block S also attenuates fully, recoverability is **not** the explanation, and E09's framework doesn't account for the data.

## What counts as replication
- H1 holds in at least 2 of 3 datasets after Holm correction: **the weak-vs-strong attenuation replicates.**
- H2 holds in at least 2 of 3: **attenuation within deep backbones replicates.** Otherwise, the paper claims only weak-vs-strong attenuation.
- Each failure is reported with its estimate and CI.
