# Experiment registry

Every experiment has a script in `experiments/` and a full JSON record in `results/` (configuration, seed, runtime, peak memory). Rerun any one with `novelty/.venv/bin/python -I research/experiments/<script>.py`.

**Data rule:** results marked *synthetic* validate code and statistics only. They are not clinical evidence.

---

## E01: weight transfer (Keras → PyTorch)
- **Hypothesis:** a faithful port reproduces the trained model; the notebook's cell-18 mapping does not.
- **Data:** `alzheimer_cnn.h5`; 64 random Kaggle slices (seed 0).
- **Result:** the faithful port agrees with Keras on **100%** of predictions, with probabilities within **0.0018** (Keras computes in float16). The notebook mapping agrees on **17%**, below the 25% chance rate. It copied the 13 conv kernels but **no biases and no classifier weights**.
- **Side finding:** the saved model was compiled with `sparse_categorical_crossentropy`, but the notebook's training cell shows `categorical_crossentropy`. So the `.h5` came from a different version of the code than the one in the notebook.
- **Interpretation:** the original SNN conversion ran on a mostly random network. Any future conversion must start from `experiments/vgg_port.py`.
- **Runtime:** 10.8 s, peak 4.7 GB.

## E02: leakage in the image-level split (first attempt failed)
- **Hypothesis:** most validation images have a near-copy in training under an image-level split.
- **Data:** all 44,000 Kaggle images; stratified 80/20 split (seed 42). The notebook's exact split can't be reproduced, because directory listing order differs between systems.
- **Result:** **inconclusive. The method failed.** Globally pooled VGG16 features make every brain slice look alike: cosine ≈ 0.997 even between classes, and 99.3% of different-class neighbours passed the "near-copy" threshold. The 99.9% "near-copy" rate this run printed is meaningless and must not be quoted.
- **One valid number:** 462 exact duplicate files (by MD5).
- **Runtime:** 538 s (embedding 44k images on MPS, with other jobs running).

## E02b: centred embeddings with a discrimination gate
- **Change:** subtract the dataset-mean embedding; add an independent 32×32 pixel embedding; only trust an embedding whose discrimination AUC (self-augmented copies vs different-class neighbours) exceeds 0.95.
- **Result:** **both failed the gate** (VGG 0.72, pixels 0.38). The diagnosis is that my calibration augmentations (flips and ±10° rotations) don't resemble this dataset's augmentations, so the calibration itself was wrong.
- **Valid number:** **128 of the 8,800 validation images (1.5%) have a byte-identical file in the training set.** This is definite leakage and a lower bound.

## E02c: visual check of nearest neighbours
- **Method:** look at validation images next to their nearest same-class and different-class training images in the centred pixel space. That means 8 random pairs plus 8 borderline pairs at similarity 0.93–0.96.
- **Result:** every pair at ≥ 0.93 is visibly **the same slice** with altered brightness or contrast. Pairs at about 0.87 are different brains. Every different-class neighbour is a different brain.

| Pixel similarity threshold | Validation images with a same-class training image above it | Different-class neighbours above it (control) |
|---|---|---|
| 0.93 (visually confirmed) | **67.3%** | 0.0% |
| 0.95 | 65.1% | 0.0% |
| 0.97 | 55.4% | 0.0% |
| 0.99 | 37.8% | 0.0% |

Per class at 0.95: Mild 57%, Moderate 60%, Non 71%, Very mild 70%.
- **Interpretation:** at least two-thirds of the notebook-style validation set has a near-copy of the same slice in training. This is a lower bound: flipped or rotated copies, and other slices from the same patient, are not counted. It agrees with prior audits (Yagis et al. 2021; independent GitHub audits), so it is **known**, not new.

## E08: how much accuracy does near-copy leakage buy?
- **Method:** group near-copies (pixel similarity ≥ threshold, single linkage within class). Train the same multinomial logistic regression on frozen VGG16 features (512-d), 5-fold, once with an image-level split and once with a group-level split.

| Split | Groups | Largest group | Mean accuracy (5 folds) |
|---|---|---|---|
| Image-level (notebook style) | n/a | n/a | **79.3%** (78.9–79.6) |
| Grouped, threshold 0.97 | 22,889 | 1.3% | 77.1% |
| Grouped, threshold 0.93 | 14,862 | 4.7% | **73.6%** |

- **Interpretation:** near-copy leakage alone inflates accuracy by **at least 5.6 points** for this classifier. The true inflation is likely larger: the groups average about 3 images, so slices from the same patient are not linked, and the dataset has no patient IDs. Published subject-level comparisons show far larger drops (95% → 50%, Yagis et al. 2021). This used a linear head on pooled features, not the original model, so it bounds the effect rather than reproducing the 86%.

## E03: does the added-value test hold when the MRI score is learned? (synthetic)
- **Hypothesis:** out-of-fold MRI scores keep the likelihood-ratio test valid; in-sample scores cost power; APOE effects mediated by atrophy trigger rejections without any direct effect.
- **Setup:** 500 cohorts per scenario, n = 400, 200 MRI features (10 informative), L2 logistic MRI model.

| Scenario | in-sample score | out-of-fold score | true atrophy (oracle) |
|---|---|---|---|
| No genetic effect at all | 4.0% | 5.6% | 4.0% |
| APOE acts only through atrophy | 5.6% | **85.6%** | 6.8% |
| Direct genetic effect (power) | **32.0%** | 58.6% | 69.8% |

Simulation standard error at 5% is about 1.0 point.
- **Interpretation:**
  1. Scoring patients with a model trained on them costs about half the power. The protocol must use out-of-fold scores.
  2. With out-of-fold scores, a significant result means "genetics adds information this MRI model missed". It does **not** mean "genetics acts directly". The estimand must be stated in advance.
  3. This could explain why published studies disagree: added value depends on how good each study's MRI model is.
- **Failure cases:** none in the code. The "mediated" result is a property of the test, not a bug.

## E04: calibration metrics (known-answer tests)
- **Result:** with perfect predictions, slope ≈ 1 and intercept ≈ 0. With predicted logit = 2 × true logit, slope = 0.5 ± 0.03. With a +0.8 logit shift, calibration-in-the-large = −0.8 ± 0.05. Brier and ECE behave as expected. See `pipeline/test_pipeline.py`.

## E05: null distribution for saliency-anatomy overlap (synthetic)
- **Hypothesis:** shuffling voxels gives invalid p-values for smooth saliency maps; shifting the mask gives valid ones.
- **Setup:** 200 volumes of 40×48×40 voxels, smoothed noise (σ = 4 voxels), a hippocampus-shaped mask (~750 voxels), 199 null draws each.

| Condition | voxel-shuffle rejection | mask-shift rejection | median enrichment |
|---|---|---|---|
| No relationship (should be 5%) | **34.5%** | **5.0%** | 0.63 |
| Weak true signal in the mask | 61% (not trustworthy) | 15% | 1.46 |

- **Interpretation:** shuffling voxels makes chance overlap look significant about a third of the time, so it must not be used. The valid test has low power at this signal strength: real analyses need strong effects or many subjects.

## E06: M5 hardware benchmark
- **Hypothesis:** MPS gives a large speedup for 3D CNN training; fp16 autocast helps further; memory limits batch size before compute does.
- **Results:** see `HARDWARE.md`. MPS is **about 2×** CPU for 3D training (3.5 vs 1.8 volumes/s), and throughput is flat across batch sizes 2–8, so the GPU is compute-bound. fp16 autocast gave **no speedup** (3.3 volumes/s) and no memory saving. VGG16 2D inference: 126 images/s on MPS vs 31 on CPU.
- **Interpretation:** hypotheses 2 and 3 falsified; hypothesis 1 only partly true. Cost scales with voxels, which motivates compact 3D representations (Track E).

## E07: double machine learning for a direct genetic effect (synthetic)
- **Hypothesis:** partialling out the MRI features with cross-fitted models keeps the false-positive rate near 5% when APOE acts only through atrophy.
- **Setup:** E03 generator; 300 cohorts; fixed penalties (logistic C = 0.1, ridge α = 50).

| n | no effect | mediated only | direct effect |
|---|---|---|---|
| 400 | **20.7%** | **23.7%** | 65.7% |
| 1,600 | **11.7%** | **10.3%** | 99.7% |

- **Result: hypothesis falsified as configured.** The test over-rejects even with no genetic effect at all, less so at larger n.

## E07b: why does double machine learning over-reject?
- **Check 1, oracle helper functions** (implementation test): **6.0%** at n = 400 and 6.0% at n = 1,600. The final-stage code is correct.
- **Check 2, cross-validated penalties** (LogisticRegressionCV, RidgeCV), 300 cohorts each:

| n | no effect | mediated only (should be 5%) | direct effect (power) |
|---|---|---|---|
| 400 | 4.3% | **44%** | 60.3% |
| 1,600 | 4.0% | **28%** | 100% |

- **Interpretation:** tuning fixes the false positives when there's no effect at all. When APOE acts only through atrophy, double machine learning still over-rejects heavily: 44% at n = 400 and 28% at n = 1,600. It improves with n, as theory predicts, but far too slowly for ADNI-sized MCI cohorts. The cause is that removing APOE's imaging footprint requires estimating E[APOE | MRI] well, and APOE's footprint is weak and spread over many features. This is the same barrier E03 exposed, seen from the other side.
- **Decision:** don't build the paper on a double machine learning "direct effect" test at realistic n. Promote candidate 2 (the attenuation curve), which needs no causal claim. A DML analysis could appear only as a sensitivity analysis, with this limitation stated.

## Bug log
- E07b first run: `NameError` in `dml_from_residuals`, introduced by my own refactor. Fixed and rerun; no results came from the broken run.

---

# Iteration 2 (26–27 Oct 2026): dataset pivot

Data used: HAM10000 (Harvard Dataverse, CC BY-NC 4.0; 10,015 images, 7,470 lesions) and VisA (CC BY 4.0). See `iteration2/DATASETS.md`. Every HAM10000 split is grouped by `lesion_id`.

## E09: does an attenuation curve separate mediated from direct auxiliary information? (synthetic)
- **Setup:** n = 3,000, latent dimension 40, 10 repetitions, a ladder of 6 image-noise levels.
- **Result:** mediated added value fell from 0.0024 to about 0, while the auxiliary variable's recoverability from the image rose from 0.10 to 0.58. Direct added value **rose** from 0.029 to 0.044 (I had predicted flat). The mixed case fell to a non-zero floor (0.0016). No effect gave about 0.
- **Interpretation:** the slope's sign is the diagnostic: falling means mediated through the image, rising means direct (non-collapsibility of logistic regression). See `iteration2/MATH_NOTES.md`.

## E10a: HAM10000 metadata audit (real data)
- 45% of images belong to lesions with more than one image, so image-level splits would leak.
- `dx_type` leaks the label: all melanomas are histology-confirmed and follow-up is used only for nevi. On its own it gives AUROC 0.76. Excluded, as pre-registered.
- The image source alone predicts melanoma with AUROC 0.73 (one source has 24 melanomas against 3,720 nevi), a **shortcut risk**.
- Age, sex and site alone: AUROC 0.71.

## E10: metadata added value vs image-model capacity (real data, pre-registered)
Pre-registration: `iteration2/PREREGISTRATION_E10.md`. 5-fold lesion-grouped CV × 3 fold assignments; penalties tuned only inside training folds; 95% CIs from a 2,000-resample lesion bootstrap. Runtime 2.4 min on the M5.

| Rung | Image-only melanoma AUROC (3 seeds) | Metadata added value, mean of 3 seeds | CIs exclude 0? | Age R² from image | 7-class added value |
|---|---|---|---|---|---|
| HSV histogram | 0.765 | 0.0186 | yes, in all 3 | 0.06 | 0.138 |
| ResNet-18 | 0.853 | 0.0037 | no | 0.28 | 0.053 |
| ResNet-50 | 0.851 | 0.0030 | no | 0.32 | 0.045 |
| ConvNeXt-T | 0.889 | 0.0017 | no | 0.39 | 0.027 |
| ViT-B/16 | 0.887 | 0.0022 | no | 0.37 | 0.028 |

**Pre-registered predictions:**
1. Positive added value for the colour model: **confirmed.**
2. Added value falls from rung 2 to the strongest rung: **direction confirmed; paired test in E10b.**
3. A non-zero floor: **not supported.** The deep-rung CIs include 0.
4. Tracks recoverability: **qualitatively confirmed.** Age R² and site AUROC (0.65 → 0.79) rise as added value falls.

**Overfitting control:** feeding full features plus metadata gives the same log-loss as score plus metadata (for example, ConvNeXt 0.228 vs 0.231), so the second-stage design isn't hiding anything.

## E10b: source shortcut control and paired attenuation tests (exploratory, real data)
- **Source-adjusted added value** (source in both models): HSV 0.0153, ResNet-18 0.0047, ResNet-50 0.0040, ConvNeXt 0.0022, ViT 0.0032. Attenuation **persists** after controlling for source.
- **Within-source melanoma AUROC rises with capacity:** vidir_modern 0.61 → 0.78 → 0.77 → 0.83 → 0.84; rosendahl 0.56 → 0.72 → 0.72 → 0.81 → 0.79. The deep models read lesions, not only the source.
- **Paired lesion-bootstrap tests** (same images, seed-0 folds):

| Comparison | Unadjusted Δ added value [95% CI] | Source-adjusted |
|---|---|---|
| HSV − ResNet-18 | 0.0143 [0.0107, 0.0180] | 0.0106 [0.0079, 0.0137] |
| HSV − ConvNeXt | 0.0167 [0.0126, 0.0206] | 0.0132 [0.0096, 0.0167] |
| ResNet-18 − ConvNeXt | **0.0024 [0.0007, 0.0041]** | **0.0026 [0.0009, 0.0040]** |
| ResNet-18 − ViT | 0.0014 [−0.0001, 0.0030] | 0.0015 [−0.0001, 0.0030] |
| ResNet-50 − ConvNeXt | 0.0016 [−0.0001, 0.0032] | 0.0018 [0.0002, 0.0034] |

- **Caveats:** exploratory, 14 comparisons with no multiplicity correction, one fold assignment. The absolute effects among deep models are small: about 1% of log-loss.

## E12: classifier attention vs expert lesion masks (real data)
Frozen ResNet-50 plus a logistic head, out-of-fold. The CAM is exact for a linear head on pooled features. 2,225 images (all melanomas plus a matched sample of others). The p-values use the exact FFT mask-shift null.
- **H1 (enrichment > 1) and H2 (errors have less on-lesion attention), as framed: confounded.** I mapped the *predicted* class's evidence. For "not melanoma" predictions that evidence comes from the skin around the lesion (median enrichment 0.52), while for melanoma predictions it sits on the lesion (1.33). Most errors are missed melanomas, so "errors look off-lesion" was produced by the sign convention.
- **H3: confirmed on real data.** Pixel shuffling declares 29.8% of maps significant; the valid mask-shift test 9.9%.

## E12b: same-sign melanoma-evidence maps (real data)
- Melanoma evidence concentrates on the lesion: median enrichment 1.49, and 84% of images above 1. Significant on 37% of images under the shift null (82% under pixel shuffling).
- **H2b (missed melanomas have less on-lesion evidence): falsified.** Missed 1.57 vs detected 1.33.
- **Lesion size explains it.** Detected melanomas are larger (52% vs 37% of the image, p = 3×10⁻¹³), and the melanoma score rises with lesion size (Spearman 0.31). Within size tertiles, the share of evidence on the lesion barely differs (p = 0.006, 0.47, 0.10). Misses are mostly **small lesions**, not misplaced attention.

## E10c: per-variable attenuation (real data)
- **Prediction (before running):** age and site attenuate strongly; sex least, assuming sex is weakly visible in dermoscopy.

| Variable | Added value, HSV | Added value, deep models (range) | Retained by the two strongest rungs | Recoverability, HSV → strongest |
|---|---|---|---|---|
| Age | 0.0147 | 0.0020–0.0026 | 14% | R² 0.06 → 0.39 |
| Site | 0.0057 | 0.0009–0.0027 | 18% | AUROC 0.65 → 0.79 |
| Sex | 0.0025 | 0.0004–0.0007 | 22% | AUROC 0.57 → **0.77** |

- **Result: inconclusive.** The premise was wrong: sex is fairly recoverable from images (AUROC 0.77), possibly via its correlation with site and age. All three variables attenuate and become recoverable together, so this doesn't separate the recoverability mechanism from alternatives. A sharper test needs a variable whose recoverability does *not* rise with capacity, or a dataset where one exists.

## E11: feature whitening / top-PC removal for kNN anomaly detection (VisA, real data)
- **Protocol:** official VisA `1cls` split (all 12 categories; normal-only training). Wide-ResNet-50 layer 2+3 patch features; random 10% memory bank (identical for all variants); k = 1.
- **Sanity check:** "centred" equals "raw" exactly in every category (Euclidean distance ignores shifts), so the implementation is consistent.

| Variant | Mean image AUROC | Mean pixel AUROC | Image: better than raw in | Wilcoxon p (image) |
|---|---|---|---|---|
| Raw (PatchCore-lite) | 0.900 | 0.976 | n/a | n/a |
| Remove top 16 PCs | 0.904 | 0.977 | 8/12 | 0.10 |
| **Whitened** (shrinkage 0.1) | **0.925** | **0.983** | 8/12 | 0.052 |

- **Per category (image AUROC, raw → whitened):** macaroni2 0.624 → 0.772, macaroni1 0.887 → 0.937, fryum 0.903 → 0.945, pcb1 0.935 → 0.968, pcb2 0.929 → 0.959; slightly worse on candle 0.979 → 0.972, capsules 0.697 → 0.683, pcb3, pcb4.
- **Correction:** my first look, at candle only, suggested "no gain, whitening slightly worse". The full run reverses that. Single-category conclusions are unreliable here.
- **"Gain largest where the baseline is weakest":** Spearman −0.39, p = 0.21. **Not established.**
- **Novelty:** none. This reproduces the published Mahalanobis PatchCore finding (arXiv 2605.27748: whitened global covariance + Euclidean kNN). An open follow-up question is whether the per-category benefit can be predicted from training-feature statistics (e.g. spectral concentration) without test labels.
- **Runtime:** about 2.4 min per category on the M5 (MPS).

---

# Iteration 3 (27 Oct 2026): replication, paper, review cycles

Datasets added: ISIC 2020 (Kaggle `cdeotte/jpeg-melanoma-256x256`; counts match Rotemberg et al. exactly: 33,126 images, 584 melanomas, 2,056 patients), PAD-UFES-20 (images from Kaggle `mahdavi1202/skin-cancer`, CC BY 4.0; **official Mendeley metadata used**; the mirror's metadata matches it exactly apart from formatting, and all 2,298 images are present), ISIC 2019 (official ISIC metadata + 256-px mirror `mnowak061`, licence listed as unknown; HAM10000 lesions excluded). Pre-registration: `iteration3/PREREGISTRATION_REPLICATION.md`.

## E10d: added value in AUROC and Brier units (HAM10000)
| Rung | ΔAUROC [95% CI] | ΔBrier ×10⁻³ [95% CI] |
|---|---|---|
| HSV | +0.043 [0.030, 0.056] | 4.84 [3.02, 6.70] |
| ResNet-18 | +0.007 [0.001, 0.013] | 1.29 [0.23, 2.42] |
| ResNet-50 | +0.005 [−0.001, 0.012] | 1.16 [0.14, 2.23] |
| ConvNeXt-T | +0.003 [−0.002, 0.007] | 0.88 [−0.14, 1.93] |
| ViT-B/16 | +0.004 [−0.001, 0.008] | 0.79 [−0.25, 1.86] |

**Precision note:** ResNet-18's small AUROC and Brier gains are still detectable; only the two strongest rungs are indistinguishable from zero.

## E13: pre-registered replication on PAD-UFES-20 (2,298 images, 1,373 patients, 1,089 malignant)
- **Seed-0 added value, all metadata (nats):** HSV 0.386 [0.352, 0.418] → ResNet-18 0.234 → ResNet-50 0.235 → ConvNeXt-T 0.206 [0.180, 0.233] → ViT-B/16 0.206.
- **H1** (HSV > ConvNeXt): diff 0.180 [0.157, 0.200], one-sided p = 0.0005. **H2** (ResNet-18 > ConvNeXt): diff 0.028 [0.010, 0.046], p = 0.001. Holm correction is applied across datasets once all three are in.
- **Large floor:** metadata keep 0.21 nats of value even for the strongest image models, about 100× HAM10000's.
- **Mechanism hypotheses, all three failed:**
  - M1: symptoms retained 47%, below the 50% threshold.
  - M2: demographics retained 51% vs symptoms 47%, so demographics did not attenuate more.
  - M3: symptoms were recoverable from images (mean AUROC 0.75, against the < 0.65 prediction).

## E16: label-conditional recoverability (diagnosing M3)
- **Hypothesis:** marginal recoverability is inflated because the image predicts the label, and the label correlates with the symptoms.
- **Result:** the hypothesis is **only partly supported.** Within-class recoverability (ConvNeXt) stays high: PAD bleeding 0.75, elevation 0.76, pain 0.72, itch 0.72, change 0.65, age R² 0.27; HAM10000 sex 0.77, age R² 0.36, site 0.76–0.81. The drop from marginal is at most 0.09 AUROC.
- **Interpretation:** the premise of M1–M3, that symptoms are invisible in photographs, was **false**. Bleeding, elevation and even pain are partly visible within a diagnosis class. Neither dataset contains truly invisible auxiliary variables, so the recoverability mechanism can't be isolated with them. This goes in the paper as a limitation and a negative result.

## E13: pre-registered replication on ISIC 2020 (33,126 images, 2,056 patients, 584 melanomas)
- **Added value, mean of 3 fold assignments (×10⁻³ nats):** HSV 3.57 → ResNet-18 1.27 → ResNet-50 1.42 → ConvNeXt-T 0.95 → ViT-B/16 0.72.
- **Image AUROC (3 seeds):** HSV ~0.77; ResNet-18 0.85; ResNet-50 0.84–0.85; ConvNeXt-T 0.86; ViT-B/16 0.87–0.88. The strongest model here (ViT) has the smallest added value.
- **H1:** diff 2.58×10⁻³ [1.78, 3.43], one-sided p = 0.0005.
- **H2:** diff 0.23×10⁻³ [−0.04, 0.50], one-sided p = 0.046 before Holm correction.
- **Seed-0 CIs:** ConvNeXt-T 0.97 [0.27, 1.71] (excludes 0); ViT-B/16 0.60 [−0.10, 1.28].
- Absolute values are small because melanoma prevalence is 1.8%, which keeps log-loss low overall.

## E17: retained fraction of metadata information (exploratory; comparable across datasets)
- **Definition:** ρ_k = Δ_k / I_A, where I_A is the out-of-fold log-loss reduction of a metadata-only model over prevalence; grouped bootstrap CIs; fold assignment 0.

| Dataset | I_A (nats) | HSV | ResNet-18 | ConvNeXt-T | ViT-B/16 |
|---|---|---|---|---|---|
| HAM10000 | 0.026 | 0.71 [0.60, 0.80] | 0.16 [0.00, 0.27] | 0.07 [−0.12, 0.20] | 0.11 [−0.03, 0.21] |
| ISIC 2020 | 0.004 | 0.83 [0.72, 0.95] | 0.28 [0.15, 0.38] | 0.23 [0.08, 0.33] | 0.14 [−0.04, 0.25] |
| PAD-UFES-20 | 0.416 | 0.93 [0.90, 0.96] | 0.56 [0.51, 0.61] | 0.50 [0.45, 0.54] | 0.50 [0.45, 0.54] |

- **Interpretation:** on dermoscopy, strong frozen image models make roughly 77–93% of the metadata's standalone information redundant. On smartphone clinical photographs with symptom and history fields, about half stays non-redundant. The dataset-dependent floor is the main qualification of the attenuation finding.

## E15: interpretability across backbones (HAM10000; review item M7)
- **Exact CAM** for ResNet-18 and ResNet-50, and for ConvNeXt-T through its post-pool LayerNorm. For ConvNeXt the map is exact up to the positive per-image factor 1/σ. **Verified:** reconstructed logits vs the head's logits, r = 1.000 for every fold and backbone.
- 2,222 images (all melanomas plus a matched sample per fold); melanoma-evidence maps; 56×56; exact FFT shift null; 99 pixel permutations.

| Backbone | Enrichment > 1 | Median enrichment | Shift-null sig. | Pixel-perm sig. | Lesion area, detected / missed (p) |
|---|---|---|---|---|---|
| ResNet-18 | 90.3% | 1.55 | 52.9% | 89.2% | 0.49 / 0.37 (5×10⁻¹²) |
| ResNet-50 | 84.1% | 1.50 | 36.0% | 82.0% | 0.52 / 0.37 (3×10⁻¹³) |
| ConvNeXt-T | 91.4% | 1.86 | 56.4% | 89.8% | 0.49 / 0.33 (7×10⁻¹⁸) |

- **Interpretation:** all three E12b findings generalise across backbones. The strongest backbone localises evidence best (median enrichment 1.86).

## E13: pre-registered replication on ISIC 2019 excluding HAM10000 (15,316 images, 6,461 groups, 3,409 melanomas)
- A bug in my adapter (the `_downsampled` suffix) initially dropped 2,074 images; it was fixed before any analysis. A second bug (missing output folder) cost one 13-minute embedding run; also fixed, and nothing was analysed from the failed run.
- **Added value, mean of 3 seeds (×10⁻³ nats):** HSV 19.05 → ResNet-18 10.23 → ResNet-50 10.29 → ConvNeXt-T 9.36 → ViT-B/16 7.34. Seed-0 CIs exclude 0 at every rung.
- **H1:** 10.2×10⁻³ [5.7, 14.4], p ≤ 0.0005. **H2:** 0.6×10⁻³ [−1.1, 2.4], p = 0.24.

## E13: final pre-registered outcome (Holm over all six tests)
| Dataset | H1 Holm p | H2 Holm p |
|---|---|---|
| ISIC 2020 | 0.003 | 0.09 (raw 0.046) |
| ISIC 2019 (non-HAM) | 0.003 | 0.24 |
| PAD-UFES-20 | 0.003 | 0.003 |

0.003 is the floor set by bootstrap resolution (6 × 1/2001).
- **H1 replicates in 3 of 3:** weak-vs-strong attenuation is confirmed.
- **H2 replicates in only 1 of 3:** attenuation *within* deep representations is **not** established, so per the pre-registration the paper claims only weak-vs-strong attenuation.

## E17 update: ISIC 2019 (non-HAM)
- I_A = 0.020 nats. ρ: HSV 0.94 [0.84, 1.05], ResNet-18 0.48, **ConvNeXt-T 0.44 [0.24, 0.62]**, ViT-B/16 0.33.
- **This corrects an earlier framing of mine ("dermoscopy retains little, smartphone retains half").** ISIC 2019 is dermoscopy and retains ρ = 0.44, close to PAD-UFES-20's 0.50. The floor depends on the dataset, not the modality.

## E20: per-variable floors (exploratory) — **PAD-UFES-20 part invalidated by E21's audit**
- At ConvNeXt-T, single-variable added value (×10⁻³ nats): HAM10000 age 1.96, sex 0.36, site 1.10; ISIC 2020 age 1.05, others ≈ 0; ISIC 2019 age 3.18, **site 5.41 [0.05, 10.66]** (retains 76%: site carries ISIC 2019's floor, consistent with BCN20000's hard-to-diagnose sites).
- PAD-UFES-20: sex and Fitzpatrick type each kept about 148×10⁻³ nats. **This was a missingness artefact (see E21), not information about sex or skin type.**

## Missingness-leakage audit (all datasets)
My first audit used `astype(str)`, which in this pandas version keeps NaN as NaN, so missing gender and ISIC values went undetected. **Bug found and fixed;** corrected results:

| Dataset | Field | Missing | Positive rate, missing vs present |
|---|---|---|---|
| PAD-UFES-20 | gender = Fitzpatrick = history fields | 804 | **0.000 vs 0.729** (all benign; none biopsied, so a proxy for the excluded `biopsed`) |
| PAD-UFES-20 | grew / changed = UNK | 402 / 396 | **0.876 / 0.884 vs 0.389** |
| ISIC 2019 (non-HAM) | anatomical site | 945 | 0.074 vs 0.232 |
| ISIC 2020 | age / sex | 68 / 65 | 0.000 vs 0.018 |
| HAM10000 | age / sex | 57 | 0.035 / 0.000 vs 0.112 |

## E21: complete-case re-analysis (deviation from the pre-registration; applies its own leakage rule)
Only images with every metadata field recorded are kept; otherwise E13's protocol, fold assignment 0.

| Dataset | n kept (dropped) | Positive rate | I_A (nats) | ΔHSV → ΔConvNeXt (×10⁻³) | H1 p | H2 p | ρ ConvNeXt |
|---|---|---|---|---|---|---|---|
| HAM10000 | 9,761 (254) | 0.113 | 0.026 | 18.1 → 2.6 | ≤0.0005 | 0.018 | 0.10 |
| ISIC 2020 | 32,531 (595) | 0.018 | 0.004 | 3.36 → 0.79 | ≤0.0005 | 0.001 | 0.19 |
| ISIC 2019 (non-HAM) | 14,156 (1,160) | 0.232 | 0.012 | 13.8 → 6.1 | 0.002 | 0.16 | 0.50 |
| PAD-UFES-20 | 1,089 (1,209) | 0.675 | **0.155** (vs 0.416 with leakage) | 152 → 74 | ≤0.0005 | 0.52 | 0.48 |

- **Robust conclusions:**
  - Weak-vs-strong attenuation (H1) holds in **all four** datasets after leakage correction.
  - Within-deep attenuation (H2) holds in 2 of 4 (HAM10000, ISIC 2020), so it remains inconsistent.
  - Dataset floors: ρ ≈ 0.10–0.19 on HAM10000 and ISIC 2020, ≈ 0.5 on ISIC 2019 and PAD-UFES-20.
- **What changes:** PAD-UFES-20's standalone metadata information was inflated about 2.7× by missingness leakage, and its complete-case sample is a different, mostly biopsied population (positive rate 0.675). ρ is similar, because the leak inflated numerator and denominator alike.
- **Caveat:** complete cases are a selected subsample, so these are sensitivity analyses, not replacements for the pre-registered results.

# Iteration 4 (27–28 Oct 2026): second wave, representation strength, exploratory E30–E34

Run by `overnight/queue2.sh` (00:09–04:01; 36 completed, 0 failed, 1 skipped by the deadline guard: E14b ISIC 2019). Machine-readable results: `results/<ID>.json`; job status: `overnight/STATUS.md`; summary: `iteration4/SUMMARY.md`. E30–E34 were registered in `iteration4/REGISTRY_E30_E34.md` before implementation, as exploratory.

## E13 wave 2 / E22: pre-registered replication on HIBA, MSK 2020, MILK10k (derm., clin.)
Pre-registration: `iteration3/PREREGISTRATION_E22.md` (Holm over 8 tests).

| Image set | H1 diff ×10⁻³ [95% CI] | p_Holm | H2 diff ×10⁻³ [95% CI] | p_Holm |
|---|---|---|---|---|
| HIBA | 69.6 [39.2, 105.5] | 0.004 | 16.3 [7.5, 25.6] | 0.004 |
| MSK 2020 | 13.4 [6.0, 21.4] | 0.004 | 3.5 [−1.6, 8.5] | **0.086** |
| MILK10k derm. | 43.6 [36.9, 50.2] | 0.004 | 6.1 [3.1, 9.2] | 0.004 |
| MILK10k clin. | 52.9 [45.3, 60.2] | 0.004 | 3.1 [1.2, 4.9] | 0.004 |

- H1 4/4, H2 3/4. Over both waves: H1 in 6/6 independent datasets, H2 in 3/6 (MILK10k counted once, holding only if both image sets hold).
- **P1 (pre-registered) failed:** ρ clinical 0.069 vs dermoscopic 0.105, paired difference −0.036 [−0.071, −0.004], opposite to the prediction. Image-only AUROC was nearly identical (0.842 vs 0.845), so the premise (weaker clinical representation) did not hold.
- **P2 (pre-registered, exploratory) supported:** diameter adds 8.6 [1.3, 16.4] ×10⁻³ to HSV and 1.8 [−1.7, 5.1] to ConvNeXt-T; retained 0.21 < 0.5.

## E17 / E21 wave 2
- ρ at ConvNeXt-T, all images: HIBA 0.01, MSK 0.47, MILK derm 0.10, clin 0.07 (range over 8 sets 0.01–0.50).
- Complete cases: H1 8/8, H2 5/8 (fails ISIC 2019, PAD-UFES-20, MSK 2020; uncorrected); ρ range 0.06–0.60.

## E14 / E14b: frozen vs fine-tuned (exploratory)
| Dataset | Backbone | AUROC frozen → FT | Δ frozen → FT ×10⁻³ | Frozen − FT [95% CI] |
|---|---|---|---|---|
| HAM10000 | ResNet-18 | 0.854 → 0.917 | 4.1 → 0.35 | 3.8 [1.7, 5.9] |
| HAM10000 | ConvNeXt-T | 0.887 → 0.930 | 1.7 → 0.8 | 0.9 [−1.1, 2.8] |
| HIBA | ResNet-18 | 0.908 → 0.940 | 18.9 → 1.4 | [6.1, 29.8] |
| HIBA | ConvNeXt-T | 0.929 → 0.951 | 2.6 → −7.2 | [2.6, 17.4] |
| PAD-UFES-20 (all images, leak included) | ResNet-18 | 0.845 → 0.899 | 234 → 173 | [43.8, 79.5] |
| PAD-UFES-20 (all images, leak included) | ConvNeXt-T | 0.874 → 0.928 | 206 → 124 | [65.5, 99.1] |

## E31: representation strength (exploratory; REGISTRY_E30_E34.md)
- **E31a, size within family (frozen, complete cases, HAM/PAD/ISIC 2019):** 0 of 6 smallest-vs-largest differences significant; image AUROC changed −0.016 to +0.009. Size alone neither improves the frozen image model nor attenuates metadata value.
- **E31b, frozen head on 10–100% of training groups:** HAM ResNet-18 8.4 → 4.2; ConvNeXt-T 7.4 → 2.6; ISIC 2019 noisier (ResNet-18 10.2 → 7.0).
- **E31c, fine-tuned ResNet-18 on 10/25/50% (HAM complete cases):** 9.7 [5.4, 14.0], 5.0 [1.4, 8.3], 3.3 [0.4, 6.1] ×10⁻³; AUROC 0.82, 0.86, 0.90. One subsample per fraction. **Follow-up E31d registered (100% on complete cases + second subsample).**

## E18: ViT-L/16 SWAG (304M) vs ConvNeXt-T, frozen, 8 image sets (exploratory)
AUROC change −0.005 to +0.020; added value lower with ViT-L on ISIC 2020 (0.47 [0.17, 0.76] ×10⁻³) and ISIC 2019 (2.4 [0.2, 4.7]); not higher anywhere.

## E19 extension: resolution ladder on HIBA and MILK10k
Run; the manipulation remains weak (AUROC barely changes with resolution). Not used in the paper.

## E30: recoverability vs utility (exploratory)
95 of 160 rows (rows where metadata-only value I_var has a CI excluding 0; this exclusion was fixed in the code before the run but not written in the registry). Spearman(within-class recoverability, retained share) −0.75; partial (given I_var, image AUROC) −0.59, dataset-clustered CI [−0.72, −0.40]; within each representation −0.41 to −0.75. Leave-one-dataset-out MAE of retained share: ridge 0.15 vs median baseline 0.29.

## E32: permutation and shortcut controls (exploratory)
Real metadata beat all 50 group-level permutations on all 8 image sets (complete cases, ConvNeXt-T; p = 0.02, the floor). Implausible-record rate (PAD original): 0.2% real vs 2.3% permuted.

## E33: cross-dataset shift (exploratory)
26 directed pairs (harmonised age/sex/site, ConvNeXt-T, fitted on source only): metadata helped significantly in 14, hurt in 7, unclear in 5; range −0.147 to +0.063 nats. Largest harms involve MSK 2020 (13% positive) → MILK10k (71%): prevalence and label-definition shift, not interpretable as a metadata property. HAM leave-one-source-out: helped for 3 of 4 sources. ISIC-ID overlap between datasets: none.

## E34: rule for which metadata to collect (exploratory) — **negative result**
Leave-one-dataset-out MAE: predict-zero 0.0092; I_var × median retained 0.0082 (Spearman 0.74); ridge 0.0101 (Spearman 0.64). The fitted rule is worse than predicting zero.

## E31d: fine-tuning dose–response on one population (exploratory; registered before running, review item R5-1)
HAM10000 complete cases, ResNet-18, E14 schedule, seed-0 grouped folds. Repeat 0 at 10–50% reused from E31c.

| Fraction | Repeat 0: AUROC / Δ ×10⁻³ [95% CI] | Repeat 1: AUROC / Δ ×10⁻³ [95% CI] | Mean Δ |
|---|---|---|---|
| 10% | 0.820 / 9.7 [5.4, 14.0] | 0.824 / 8.2 [4.1, 12.5] | 8.9 |
| 25% | 0.859 / 5.0 [1.4, 8.5] | 0.861 / 5.9 [2.2, 9.4] | 5.4 |
| 50% | 0.896 / 3.3 [0.5, 6.2] | 0.896 / 3.0 [−0.0, 6.1] | 3.2 |
| 100% | 0.905 / 1.2 [−1.3, 3.8] | — | 1.2 |

- Spearman(fraction, Δ) over 7 runs: −0.97. The prediction (from E31c) held on a single population and with a second subsample.
- The 100% complete-case run gives Δ 1.2 (all-images E14: 0.35); both are the lowest point of their series.
- Paper: Fig. 3 and Section V-G regenerated from E31d.


### E14b ISIC 2019 (non-HAM) and MILK10k dermoscopic, ResNet-18 (registered addendum, REGISTRY_E30_E34.md; exploratory)

Run 2026-10-28 by queue3, both exit 0 (ISIC 2019 54 min, MILK10k 19 min). Same grouped folds (assignment 0), all images, one training run per fold.

| Dataset | AUROC frozen / FT | Δ frozen / FT (×10⁻³ nats) | Δ FT 95% CI | Frozen − FT 95% CI |
|---|---|---|---|---|
| ISIC 2019 (non-HAM) | 0.802 / 0.836 | 9.7 / 7.1 | [1.5, 12.5] | [−0.2, 5.7] |
| MILK10k (dermoscopic) | 0.815 / 0.869 | 14.7 / 13.7 | [6.0, 20.8] | [−2.4, 4.5] |

Interpretation: fine-tuning improved the image model on both, and the point estimate of added value fell on both, but neither drop is distinguishable from zero. Across all 8 frozen-vs-fine-tuned pairs, added value is lower after fine-tuning in every pair; significant on 3 of 5 datasets. The paper's fine-tuning claims were qualified accordingly (REVIEW_LOG R6-9).


### E19 ISIC 2020 completion (registered addendum; exploratory)

Embedded 11:13–11:39 and re-analysed 11:39–11:41 on 2026-10-28 (queue4, both exit 0). Frozen ConvNeXt-T, fold assignment 0, all images.
ISIC 2020: image AUROC 0.850 at 16 px vs 0.859 at 224 px; Δ 1.3 [0.3, 2.1] vs 1.0 [0.2, 1.7] ×10⁻³ nats; Spearman(resolution, Δ) = −0.6, so the pre-stated prediction (Spearman ≤ −0.8 and Δ at 224 below Δ at 16) does not hold.
Across all 7 image sets the prediction holds on 3 (PAD-UFES-20, MILK10k derm., MILK10k clin.). Δ at 224 px is below Δ at 16 px on all 7, but resolution barely moves image AUROC, so the manipulation remains weak. Closes REVIEW_LOG R6-4.

## Iteration 5 (2026-10-28): audit and graph extension

Plan, audit and pre-registrations: research/iteration5/ (AUDIT.md, REGISTRY.md, THEORY_AUDIT.md, LITERATURE.md).

### E40: strictly cross-fitted baseline (registered 21:48 IST; exploratory audit) — done 21:50–22:26
- **Why:** the second stage was trained on out-of-fold scores whose first-stage models had seen the outer test fold's labels (AUDIT R1), and its training rows were imputed with medians that included the test fold (R2).
- **Reproduction:** the original estimator reproduced all 120 stored Δ̂ (8 image sets × 5 rungs × 3 fold assignments) exactly.
- **Change from strict cross-fitting:** at most 0.42 bootstrap CI half-widths, mostly upward at deep rungs (HAM10000 ConvNeXt-T, fold assignment 0: 1.74 → 2.38 ×10⁻³ nats). H1 unchanged everywhere.
- **H2:** ISIC 2020 changes from not holding (Holm 0.092) to holding (Holm 0.028); the difference is 0.36 ×10⁻³ nats, at the edge of bootstrap resolution.
- **ISIC 2019 duplicate merge:** immaterial.
- **Further findings:**
  - Image- vs group-weighted Δ̂ differ materially (HIBA ConvNeXt-T +3.8 vs −0.8).
  - The spread across fold assignments reaches 40–80% of the CI half-width (PAD-UFES-20, MILK10k, HIBA).
  - The fusion model's expected gain K exceeds the realised gain by 1.1–7×.
  - Exploratory: 9–33% of the HSV → ConvNeXt-T attenuation is due to falling image uncertainty, on 7 of 8 sets (61% on PAD-UFES-20).
- **Decision (registered rule):** the corrected estimator is primary from now on; the paper gets a methods note. Details: iteration5/STAGE1_REPORT.md.

### E42: representation graphs and per-lesion metadata benefit (registered 21:59 IST; H1/H2/H4 confirmatory for iteration 5) — done 22:28–22:38
- **H1** (local label entropy ↔ Δ_i, ConvNeXt-T) **fails**: 4 of 7 units (rule ≥ 5).
- **H2** (image predictive entropy ↔ Δ_i) **holds** in 7 of 7; this is a manipulation check implied by u(s) ≤ H(Y|S=s).
- **H4** (graph statistics add held-out predictive value beyond K_i, entropy, margin, norm, density) **fails**: 0 of 7 after Holm.
- **Exploratory H3:** with colour histograms, ambiguity tracks benefit more closely than with ConvNeXt-T on 5 of 8 sets; PAD-UFES-20 is the reverse.
- **Post hoc** (labelled): the fusion model's own expected gain K_i is the best or near-best ranker of realised benefit on 7 of 8 sets. Its top 25% captures 76–105% of total benefit on six sets.
- **Decision:** the registered Stage 6 gate is not met. The graph statistic is not pursued as a method. Details: iteration5/STAGE3_5_REPORT.md.

### E41: alternative labels and complete cases, corrected estimator (registered 22:30 IST; exploratory) — wall clock 22:59–04:38 (Mac asleep most of this time; compute 15.3 + 12.6 min)
- **H1 under the alternative label:** holds on HAM10000 (MAL), ISIC 2019 (MAL), PAD-UFES-20 (MEL), HIBA (MEL, p 0.046) and MSK 2020 (MEL). Fails on MILK10k MEL, where metadata information is 1.8 ×10⁻³ nats.
- **H2 is label-dependent:** ISIC 2019 holds for MAL, fails for MEL.
- **ρ at ConvNeXt-T is label-dependent:** ISIC 2019 0.20 (MAL) vs 0.44–0.52 (MEL).
- **Complete cases, corrected:** H1 8 of 8; H2 6 of 8 uncorrected (fails ISIC 2019, PAD-UFES-20).

### E44: smoothness of Δ_i on representation graphs (registered 21:59 IST; exploratory) — 22:40–22:59
- **ConvNeXt-T:** essentially no local organisation beyond label × uncertainty; stratified-null ratio at most 1.1% below 1, significant on 2 of 8 sets.
- **HSV:** 1–3% below the null on 6 of 8 sets.

### E45: LEACE concept erasure (registered 22:03 IST; exploratory) — 04:38–05:26
- **Age, where decodable** (HAM10000, ISIC 2019, MILK10k): erasing its single linear direction raises Δ_age monotonically, 3–7× (e.g. HAM10000 ConvNeXt-T 2.0 → 13.2 [7.9, 18.3]).
- **Controls:** a same-rank random direction gives no change; performance-matched random subspaces give 2.9 / 3.5 / 35.1.
- **Negative control:** PAD-UFES-20, where age is not decodable (R² 0.02), shows no effect.
- **Sex:** decodable but carries no value, so there is nothing to restore.
- **Site:** less specific.
- **Caveats:** paired CIs against the control were not computed; erasure is linear only.

### E46: E30 re-analysis (registered 22:03 IST; exploratory) — 05:38
- **Filtered rows (n = 95):** partial −0.60, jackknife-stable; mixed model −0.49 [−0.67, −0.31].
- **All rows (n = 160):** partial −0.28 (absolute Δ) or −0.16 (shrunk ratio); mixed-model coefficients not significant.
- **Conclusion:** the claim holds only among variables with standalone value. The filter was set at implementation and is not in the registry.

### E47: missingness and source (registered 22:03 IST; exploratory) — first run crashed (read-only array), re-run 05:55–05:59
- **Removing missing indicators:**
  - halves PAD-UFES-20's Δ̂ (ConvNeXt-T 207 → 112; multiple imputation 91) and lowers ISIC 2019's by 15–25%;
  - raises Δ̂ at ConvNeXt-T on HIBA (3.8 → 12.7), HAM10000 and MILK10k;
  - leaves H1's direction intact everywhere.
- **Source-held-out:** HAM10000 Δ̂ is negative for 2 of 4 held-out sources (ns) and positive for 2. ISIC 2019: BCN → MSK −10.5 (ns), MSK → BCN +29.2.

### E48: mask-swap null (registered 22:03 IST; exploratory) — 05:26–05:28
- **Centred Gaussian control:** shift null 88% significant, swap null 5%.
- **Real CAMs:** shift null 37–57% significant, swap null 10–19%.
- **Conclusion:** the published shift-null shares are mostly centre bias.

Full synthesis: iteration5/FINAL_REPORT.md.

### E49: decomposition uncertainty; record of the E42 post hoc ranking (registered 2026-10-29 09:40 IST; exploratory) — 09:41
- **Share of the HSV → ConvNeXt-T log-attenuation due to image uncertainty (s_H, fold assignment 0, 95% grouped CI):**
  - HAM10000 0.13 [0.06, 0.22]; ISIC 2020 0.09 [0.04, 0.13]; ISIC 2019 0.33 [0.18, 0.58]; PAD-UFES-20 0.61 [0.55, 0.66];
  - HIBA 0.13 [0.07, 0.26] (31% of resamples excluded, Δ_CNX ≤ 0); MSK 2020 0.12 [0.06, 0.17]; MILK10k 0.13 [0.07, 0.19] / 0.11 [0.06, 0.15].
  - Fold assignments 1 and 2 agree.
- **Post hoc ranking (saved to results/E49_decomposition_posthoc.json):** K is the best ranker on 6 of 8 sets, G on ISIC 2020, h(q0) on MILK10k clin.

### E45b: paired inference for the age erasure (registered 2026-10-29 09:39 IST; exploratory) — 09:39–10:03
- **Erasure minus performance-matched random erasure** (mean of 5 random subspaces; ×10⁻³ nats, 95% grouped CI):

  | Dataset | ConvNeXt-T | ResNet-18 |
  |---|---|---|
  | HAM10000 | 10.1 [7.2, 12.9] | 10.4 [7.6, 13.2] |
  | ISIC 2019 | 6.9 [4.6, 9.4] | 6.4 [4.3, 8.8] |
  | MILK10k | 34.7 [28.4, 41.0] | 21.6 [16.7, 27.1] |
  | PAD-UFES-20 (age not decodable) | 0.2 [−2.5, 3.3] | 1.8 [−0.1, 3.9] |

- **Registered claim rule:** restoration beyond matched control holds in 6 of 8 pairs, every pair where age is decodable.
- **Caveat:** on MILK10k the matched control cost less image performance (log-loss 0.50 vs 0.52).
- **Paper:** folded into draft v6.

## Iteration 6 (29–30 Oct 2026): mechanism identification (iteration6/REGISTRY.md, written 2026-10-29 13:49 IST; amendments 14:55, 14:58, 18:44)
Synthesis: iteration6/FINAL_REPORT.md. Tables: iteration6/E50_tables.md, iteration6/ITER6_TABLES.md. Run log: iteration6/STATUS.md. Units ×10⁻³ nats, paired grouped bootstrap 95% CI.

### E50a: age erasure with dose, nested matched random, sham and restoration (confirmatory family + replication) — 14:58–20:36 (Mac asleep ~1 h 50 min during ISIC 2020)
- **Confirmatory (Holm, m = 4).** Only ISIC 2020 age is rejected: T − matched +0.99 [+0.69, +1.29], Holm p = 0.002. HIBA age (−116.6) and MSK 2020 age (−23.8) fail with p = 1.0; the matched control overshoots.
- **G1 = PROCEED.**
- **Replication.** HAM10000 +9.05, ISIC 2019 +6.92, MILK10k derm +9.53, MILK10k clin +11.63; PAD-UFES-20 null.
- **Dose and restoration.** The D2 dose curve is monotone on the large sets. Restoration fraction is 0.71–0.97 targeted vs 0.03–0.10 matched.

### E50b: robustness (`--minimal`) — 29 Oct 23:37 to 30 Oct 01:22
- **Fold assignments 0/1/2:** ISIC 2020 +0.99 / +1.08 / +0.85; HAM10000 +9.05 / +8.74 / +8.65; ISIC 2019 +6.92 / +6.74 / +6.42. All CIs are above 0.
- **ResNet-18:** HAM10000 +8.89, ISIC 2019 +6.22, MILK10k derm +22.19; PAD-UFES-20 +1.89 [−0.16, +4.11].
- **Continuity:** all six E45b claims hold.

### E50c: class-conditional age erasure (amendment 3; exploratory) — 21:16–23:12
Positive and well-matched on ISIC 2020 (+0.86), HAM10000 (+4.05), ISIC 2019 (+2.53), MILK10k derm (+13.62) and clin (+14.05), with site and sex gains unchanged. Null on PAD-UFES-20 and MSK 2020. HIBA (+61) is non-specific because the eraser overfits.

### E51: site erasure and specificity matrix — 20:37–21:15
- **Confirmatory:** HIBA site p = 1.0 (matched overshoots).
- **HAM10000** site T − matched +4.26; **MILK10k derm** +3.83.
- The registered selectivity test holds for both age and site on HAM10000 and MILK10k derm (double dissociation), and fails on HIBA and PAD-UFES-20.

### E52: MLP probe — 23:12–23:17
After targeted linear erasure the MLP still decodes age on the five large sets: R² 0.05–0.23, CI lower bound > 0. It also decodes site on HAM10000 and MILK10k derm. Wording rule triggered: describe the intervention as removing linearly decodable information only.

### E53: estimator robustness — first run failed (pandas read-only array), rerun 23:23–23:29
- H1 holds under score, full-feature and boosted fusion on all 8 sets.
- Whether any value remains at ConvNeXt-T depends on the estimator on 5 of 8 sets.
- Boosted fusion is +4 to +8 above score fusion on 5 sets.
- Missingness share at ConvNeXt-T: PAD-UFES-20 77%, ISIC 2019 25%.

### E53b: E50 contrast under full-feature fusion — 23:29–23:37
ISIC 2020 +1.49, HAM10000 +8.12, ISIC 2019 +5.13, MILK10k derm +23.49, MILK10k clin +25.85; all CIs are above 0. The rebuilt erasers reproduce E50's scores exactly.

### E55: dataset shift and decision rules — first run's summary step crashed after all 12 evaluations; rerun 30 Oct 03:13–03:18 reproduced them exactly
Over 7 dataset-level targets, mean regret: always 3.61, collection rule 3.61 (it chose "use" everywhere), source-validated 3.91, never 9.54. Metadata hurt on PAD-UFES-20 (−19.6 [−36.4, −2.4]), and no source-fitted rule flagged it.

### E54: fine-tuning seeds 1–2 (HIBA, PAD-UFES-20) — 30 Oct 01:22–03:13
- Seed 0 reproduces E14b exactly. Fine-tuned Δ for seeds 0/1/2:
  - HIBA ResNet-18 +1.37 / −2.05 / +1.21
  - HIBA ConvNeXt-T −7.24 / −8.71 / −8.97
  - PAD-UFES-20 ResNet-18 +172.7 / +171.5 / +176.2
  - PAD-UFES-20 ConvNeXt-T +123.9 / +121.0 / +127.8
- The frozen − fine-tuned CI is above 0 for every seed. The seed range is ≤ 6.7.
