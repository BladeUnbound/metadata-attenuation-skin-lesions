# Registry: E30–E34 (requested 27 Oct 2026, ~21:30 IST)

*Written before any of these experiments was implemented or run. The request named them E20–E24; IDs E20–E22 already exist (per-variable floor, complete-case, paired modality), so they are renumbered: request E20 → **E30**, E21 → **E31**, E22 → **E32**, E23 → **E33**, E24 → **E34**.*

**Status of all five: EXPLORATORY.** Results from E10c, E16, E20 and E21 already exist for some datasets, so none of these analyses can honestly be called pre-registered. The hypotheses below are written down now so they can't change after results are seen, but they are not confirmatory.

## Shared rules (all of E30–E34)
- **Datasets:** HAM10000, ISIC 2020, ISIC 2019 (non-HAM), PAD-UFES-20, HIBA (dermoscopic), MSKCC 2020, MILK10k (dermoscopic), MILK10k (clinical). The last four only if E22's downloads complete.
- **Leakage:** label-encoding fields are excluded as before. **Primary analyses use complete cases** (every metadata variable recorded), because missingness carried the label in every earlier dataset (E21 audit). Original-data results appear only where a contrast with complete cases is the point (E32).
- **Splits:** StratifiedGroupKFold(5) by lesion or patient ID, seed 0 unless stated. All tuning happens inside training folds. Out-of-fold image scores are cached under `data/<ds>/oof/` so later experiments reuse identical scores.
- **Uncertainty:** grouped bootstrap, B = 2,000 unless stated.

## E30: recoverability vs incremental utility (request E20)
- **Question:** does how well an image representation predicts a metadata variable explain how much diagnostic value that variable still adds?
- **Units:** dataset × representation (HSV, ResNet-18, ResNet-50, ConvNeXt-T, ViT-B/16, plus ViT-L/16-SWAG if available) × variable.
- **Measures:**
  - recoverability, marginal and within-class: AUROC for binary, macro one-vs-rest AUROC for categorical, R² for numeric; out of fold, grouped; with a Gini-type normalisation (2·AUROC − 1; R² ≥ 0) for pooling;
  - the variable's metadata-only utility I_var;
  - its added value Δ_var over the image score;
  - the full block's Δ;
  - missingness rate (before complete-case filtering), image-only AUROC, and source recoverability where a source field exists.
- **Hypothesis (H30):** controlling for I_var and image-only AUROC, higher within-class recoverability goes with a lower retained fraction Δ_var / I_var.
- **Analysis:** Spearman and partial correlations; a ridge model of retained fraction with dataset-clustered bootstrap CIs; leave-one-dataset-out prediction.
- **Expected negative-result modes:** recoverability may be unrelated, or may track utility only through I_var.
- **Compute:** ~60–90 min CPU (ISIC 2020 dominates).

## E31: controlled representation quality, fine-tuning, data scaling (request E21)
- **(a) Size within one family**, frozen:
  - ResNet-18/34/50/101 and ConvNeXt-T/S/B (torchvision ImageNet weights);
  - on HAM10000, PAD-UFES-20 (complete cases) and ISIC 2019 (non-HAM);
  - GPU ~25 min.
- **(b) Training fraction**, frozen ConvNeXt-T and ResNet-18:
  - fractions 10/25/50/100% of *training groups*, with test folds fixed;
  - 3 subsample repeats;
  - HAM10000 and ISIC 2019;
  - CPU ~30 min.
- **(c) Fine-tuning × fraction**, ResNet-18 on HAM10000:
  - fractions 10/25/50%; 100% reuses E14;
  - E14 schedule, checkpointed per fold;
  - GPU ~50 min.
- **Metrics:** image AUROC, log-loss, Brier, calibration slope and intercept, and metadata Δ with paired CIs. Image-only and image + metadata second stages use the same fixed L2 logistic and the same folds.
- **Hypothesis (H31):** across all of (a)–(c), metadata Δ falls as image-only log-loss improves, whichever factor (architecture, size, fine-tuning, data) produced the improvement.
- **No causal claim:** (a)–(c) are separate manipulations, not one controlled factorial.

## E32: correspondence, permutation and shortcut controls (request E22)
- **Matched vs permuted metadata:**
  - each group's metadata vector is reassigned to another random group, which preserves the joint metadata distribution and missingness patterns;
  - R = 50 permutations give a null distribution of Δ;
  - computed on original data and complete cases;
  - rungs HSV, ConvNeXt-T.
- **Ablations:** single variables and leave-one-variable-out.
- **Source-adjusted models:** HAM10000 (`dataset`); ISIC 2019 (BCN vs other, from the lesion_id prefix).
- **Implausibility check:** how often permutation creates combinations rare in real data, e.g. age < 30 with a malignant lesion, compared with the observed frequency.
- **Hypothesis (H32):** matched Δ exceeds every permuted Δ (empirical p ≤ 1/51) wherever matched Δ > 0; permuted Δ is centred at or below 0.
- **Compute:** ~30–40 min CPU.

## E33: external-source and modality shift (request E23)
- **Provenance matrix:** modality, label definition, metadata, identifiers, missingness, prevalence, source, and **ISIC-ID overlaps between datasets** (overlapping images are removed from test sets).
- **Directed train → test pairs with compatible labels:**
  - melanoma vs rest: HAM10000, ISIC 2019 (non-HAM), ISIC 2020 (all 6 directions);
  - malignant vs benign: HIBA, MSKCC 2020, MILK10k-derm, MILK10k-clin, PAD-UFES-20 (cc);
  - MILK10k derm ↔ clin as a same-lesion modality shift;
  - leave-one-source-out within HAM10000.
- **Harmonised metadata:** age; sex; site mapped to 6 coarse regions.
- **Procedure:** the image head and second stage are fitted on the source only; nothing is tuned on the target.
- **Target metrics:** Δ log-loss, ΔAUROC, sensitivity at 90% specificity (the threshold fixed on the source, plus the target-oracle threshold for description), Brier, calibration slope and intercept; bootstrap over target groups.
- **Hypothesis (H33):** metadata Δ on a shifted target differs from in-distribution Δ. Direction is not pre-specified.
- **Caveat:** any modality difference except MILK10k derm ↔ clin is confounded by population and collection.
- **Compute:** ~30 min CPU.

## E34: predictive rule for which metadata to collect (request E24; only after E30)
- **Training rows:** E30's dataset × variable × representation table.
- **Target:** Δ_var.
- **Features:** image-only AUROC, recoverability (marginal and within-class), I_var, missingness rate, source-shift indicator.
- **Models:**
  - baseline 0, Δ = 0;
  - baseline 1, Δ = I_var × the median retained fraction from the training datasets;
  - ridge regression on standardised features.
- **Evaluation:** leave-one-dataset-out, so no outer-test dataset influences features, scaling or hyperparameters (inner leave-one-dataset-out for the ridge penalty).
- **Metrics:** MAE, Spearman, and a prespecified decision rule ("collect" if predicted Δ ≥ 1×10⁻³ nats) scored against exhaustive evaluation: precision, recall, failure cases.
- **Known limitation:** at most 8 dataset variants, and MILK10k derm and clin share lesions (held out together), so generalisation evidence is weak by construction.
- **Compute:** < 5 min CPU.

## Queue priority (overnight, deadline 05:45)
1. The existing pre-registered second wave (E22: embeddings, analyses, P1/P2).
2. CPU experiments E30 → E32 → E33 → E34.
3. E31(a) family-size embeddings + analysis; E31(b) data scaling.
4. E18 foundation rung.
5. E14b fine-tuning on PAD-UFES-20.
6. E31(c) fine-tuning × fraction.
7. E19 extensions; long fine-tuning runs.

E19 extensions and the long fine-tuning runs were in the earlier queue; they move behind E30–E33 because those answer requested questions using cached embeddings at low cost.

## Addendum, 2026-10-28 07:15 IST: follow-up runs from review round 5 (written before running)
Prompted by the paper's round-5 review (paper/REVIEW_LOG.md, R5-1). All exploratory.

### E31d: fine-tuning dose–response on one population, with a second subsample
- **Problem:** in Fig. 3 of draft v4, the 10/25/50% points (E31c) use HAM10000 complete cases and one subsample of training lesions each, while the 100% point is E14 (all 10,015 images).
- **Protocol:** as E31c (ResNet-18, E14 schedule, complete cases, seed-0 grouped folds). Adds (a) 100% of training lesions on complete cases; (b) a second independent subsample of training lesions at 10/25/50% (subsample seed 314; E31c's subsample is repeat 0, reused from its checkpoints).
- **Report:** added value with 95% grouped-bootstrap CI per fraction and repeat; per-fraction mean over repeats; Spearman between fraction and added value over all runs.
- **Prediction (from E31c, therefore not confirmatory):** added value decreases with the training fraction and is smallest at 100%.
- **Compute:** ~65 min (100%) + ~60 min (repeat 1), GPU (MPS).

### E14b on ISIC 2019 (non-HAM) and MILK10k dermoscopic
- **Purpose:** extend the frozen-vs-fine-tuned comparison beyond HAM10000, HIBA and PAD-UFES-20. The ISIC 2019 run was skipped by the overnight queue's deadline guard.
- **Protocol:** unchanged E14b (ResNet-18; all images; fold assignment 0).
- **Compute:** ~110 min (ISIC 2019) and ~35 min (MILK10k), GPU.

### E19 completion on ISIC 2020 (2026-10-28 09:30 IST, before running)
- **Problem:** E19's registered dataset list includes ISIC 2020, but its multi-resolution embeddings were never computed (the analysis skipped it via its readiness guard), so the published E19 table had 6 of 7 datasets.
- **Protocol:** unchanged E19 (frozen ConvNeXt-T, resolutions 16/32/64/112/224, fold assignment 0, all images). **Compute:** ~40 min GPU.
