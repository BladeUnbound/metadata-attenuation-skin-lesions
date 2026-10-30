# Iteration 6 registry (E50–E55)

Written 2026-10-29 13:49 IST, before any E50–E55 code was written or any of their results computed. The aggregate E45/E45b results on HAM10000, ISIC 2019 non-HAM, MILK10k dermoscopic and PAD-UFES-20 have been seen. E50 on those four sets is therefore labelled **replication (exploratory)**. The confirmatory family below uses only image sets that no erasure experiment has touched.

Shared definitions:
- **Estimator.** E40 strict cross-fitting; second stage logistic on [s] vs [s, A]; the outer test fold is used only for prediction.
- **Per-observation gain.** Δ^A_i = ℓ_i(image only) − ℓ_i(image + A), computed on identical outer test cases in every arm.
- **Grouping.** Patient where recorded (ISIC 2020, PAD-UFES-20, HIBA), otherwise lesion (HAM10000, ISIC 2019, MSK 2020, MILK10k), otherwise image.
- **Inference.** Paired grouped bootstrap (B = 2000) over grouping units; one-sided p = (1 + #{b ≤ 0})/(B + 1).
- **Arm-specific fitting.** Every eraser, probe, scaler, imputation median and penalty is fitted on outer-training rows only. The outer test labels enter nothing except the final log-loss.

## E50: dose–response, sham and restoration for age (Priority 1)

**Hypothesis (redundancy mechanism).** Removing age's linearly decodable trace from frozen image features raises age's added value more than removing a random subspace that costs the image model as much performance.

**Arms** (ConvNeXt-T, fold assignment 0, per image set):
- **T-D1(t).** LEACE eraser for age; S_t = S − t(S − S_erased), t ∈ {0, 0.25, 0.5, 0.75, 1}. This is the E45 definition. It is information-preserving for t < 1 (AUDIT A2).
- **T-D2(t).** Erased coordinates c = (S − μ)WU are replaced by √(1−t)·c + √t·σ·ε, with σ the training-fold SD and ε ~ N(0, 1) independent (fixed seed), t ∈ {0.25, 0.5, 0.75, 1}. Linear decodability of age scales with 1 − t. This is the primary dose–response family.
- **R(t), matched random.** 5 random whitened subspaces at rank r_k, under D2 at every t and under D1 at t = 1. r_k is chosen separately in each outer fold k as the smallest value on the grid {1, 2, 4, …} (refined at √2 steps) whose inner-CV image log-loss on the training fold is at least that of T-D1(1) in that fold. Selection uses training labels only.
- **SR, same-rank random.** 5 subspaces, D1, t = 1.
- **Sham.** 5 LEACE erasers fitted to age permuted within the training fold, D1, t = 1.
- **Restoration.** (i) Features [S_erased, c]: T-D1(1) with the erased coordinates re-appended, and the same for one matched random subspace. (ii) Restoration fraction RF = (Δ(arm) − Δ(0)) / (LL0(arm) − LL0(0)), defined only when the denominator's CI excludes 0. (iii) Image-trace block: Δ from adding c itself, rather than age, in the second stage.

**Recorded in every arm.** Image-only log-loss, AUROC and Brier; image + age log-loss; Δ for age, site, sex and the full block; residual age decodability (nested-α ridge R², fixed α = 100 ridge R², and a site probe); per-fold ranks; chosen C values; seeds.

**Primary contrast (per set).** C = mean_i[Δ^age_i(T-D1(1)) − mean_r Δ^age_i(R_r-D1(1))].

**Confirmatory family F** (Holm, α = 0.05, one-sided, ConvNeXt-T, fold assignment 0):
- (HIBA, age), (MSK 2020, age), (ISIC 2020, age): E50.
- (HIBA, site): E51.

Eligibility was fixed from E30's metadata-only values: lower 95% bound ≥ 2×10⁻³ nats, and the set was never used in an erasure experiment.

**Secondary, pre-specified, not adjusted:**
- S1: targeted − sham.
- S2: targeted − same-rank random.
- S3: D2 and D1 dose curves of targeted − matched, with paired CIs at each t. Prediction: nondecreasing in t under D2.
- S4 (matching check): image-only log-loss, targeted − matched. If its CI lies entirely above 0, matching failed and C for that set is flagged.
- S5: residual decodability in every arm.
- S6: restoration (i)–(iii).
  - (i) Prediction: Δ^age and LL0 return to their t = 0 values within the paired CI.
  - (iii) Prediction under redundancy: Δ_c(T) ≈ Δ^age(T) − Δ^age(0). A Δ_c much larger means the erased coordinate carried label information beyond age, i.e. erasure was not specific.

**Replication (exploratory):** the same contrast on HAM10000, ISIC 2019 non-HAM, MILK10k derm and PAD-UFES-20 complete cases (all E45-seen), and on MILK10k clinical (correlated with derm). E50b adds ResNet-18 and fold assignments 1 and 2 (t = 1 arms only).

**Continuity with E45b.** The E45b claim rule stands as registered. If E50's nested-control contrast CI includes 0 for an E45b pair whose claim held, the manuscript drops that claim, and the report states which control choice drives the change.

**Gate G1 (after E50a):**
- **PROCEED** to the mechanism stages if two conditions hold:
  - at least one confirmatory age test has p ≤ 0.0125 (Holm's first step for m = 4);
  - in that set, targeted − sham has a 95% CI above 0.
- **REVISE** (attribute-specific redundancy not supported; the rise is attributed to performance loss or to the LEACE procedure, and P2/P3 are reported as descriptive) if either holds:
  - only the replication sets show the effect;
  - targeted is distinguishable from random but not from sham.
- **STOP** the mechanism claim if targeted − matched CIs include 0 on all confirmatory and replication sets.

**Compute.** About 50 corrected fits per set (20–35 s each on 10k–33k rows). Roughly 2 h CPU for E50a and 2 h for E50b.

## E51: attribute specificity with anatomical site (Priority 2)

Same arms as E50 at t = 1 (T-D1, 5 matched random, 5 sham, 5 same-rank), with the site one-hot as the concept. LEACE rank is the number of observed categories − 1. Probe: multinomial logistic (fixed C = 0.01, as E45), scored by held-out log-loss reduction over the training-fold prior and by macro one-vs-rest AUROC.

Sets:
- HIBA: confirmatory, in F.
- HAM10000: replication; E45 saw site there.
- PAD-UFES-20 complete cases and MILK10k derm: exploratory; below the 2×10⁻³ standalone threshold.

ISIC 2019, MSK 2020 and ISIC 2020 are excluded because site has no supported standalone value there.

**Specificity matrix.** M(E, V) = targeted-minus-matched contrast for eraser E ∈ {age, site} and gain variable V ∈ {age, site}. M(age, ·) comes from E50.

Selective restoration of E's gain is concluded only if both hold:
- M(E, E)'s CI lies above 0;
- the paired difference M(E, E) − M(E, V≠E), computed per observation and in nats, has a CI above 0.

All four cells are reported whatever their sign.

## E52: residual nonlinear decodability (Priority 3)

**Probe** (fixed now):
- Model: sklearn MLP with one hidden layer of 256 ReLU units; Adam, learning rate 1e-3; L2 α = 1e-3; batch 256; max_iter 200.
- Early stopping on a 10% random split of the training fold (not grouped; training rows only); random_state 0.
- Inputs standardised by the training fold. Age target standardised by training mean/SD; site as a classifier.

**Arms.** t = 0; T-D1(1); T-D2(0.5); matched random subspace 0 at D1(1); sham 0. Both age and site erasers, on all E50/E51 sets.

**Metrics.** Held-out R² for age, with a grouped-bootstrap CI over held-out predictions. Site: log-loss reduction and macro AUROC. The linear probe on the same arms is reported alongside.

**Wording rule.** If the nonlinear R² (or site log-loss reduction) has a CI lower bound above 0 after T-D1(1), the manuscript describes erasure as removing *linearly decodable* attribute information only, never the attribute.

## E53: estimator robustness (Priority 4)

**Setup.** Same outer folds, grouping and test cases as E40 fold assignment 0. Rungs: HSV, ResNet-18 and ConvNeXt-T. All 8 image sets. Full metadata block.

**Fusion estimators** (each paired with its own image-only counterpart, all tuning inside training folds):
- (a) Score fusion: the current E40 corrected estimator.
- (b) Full-feature regularised fusion: L2 logistic on [standardised X, metadata encoding]. C is chosen by grouped inner CV on the training fold from the E10 grid. The counterpart is the same procedure on X alone.
- (c) Small nonlinear fusion: HistGradientBoosting on [inner OOF score, raw metadata with native missing and categorical handling]. Fixed hyperparameters: max_iter 200, learning_rate 0.05, max_depth 3, min_samples_leaf 20, l2 1.0, no early stopping. The counterpart is HGB on [score].
- (d) Missingness-only: logistic on [s, missingness indicators of age/sex/site] vs [s].

**Questions** (descriptive, paired CIs):
- Q1: does H1 (Δ_HSV > Δ_ConvNeXt-T) hold under (b) and (c) wherever it holds under (a)?
- Q2: Δ(b) − Δ(a) and Δ(c) − Δ(a) per set.
- Q3: Δ(d) as a share of Δ(a).

**E53b** (only if G1 = PROCEED). The E50 primary contrast recomputed with estimator (b): 5 matched random subspaces at E50's per-fold ranks, on sets where E50's contrast CI is above 0.

## E54: representation conditions (Priority 5)

Reuse E14, E14b, E31c and E31d as reported; no new backbones.

If time allows after E50–E53: repeat E14b on HIBA and PAD-UFES-20 (ConvNeXt-T and ResNet-18) with training seeds 1 and 2, same schedule and folds.
- Report the per-seed added value and the across-seed spread next to the original seed.
- The second stage is the original estimator, for comparability with E14b. R7-11 (strict cross-fitting of fine-tuning) stays open unless explicitly run.

## E55: dataset shift and decision rules (Priority 6)

**Units.**
- Leave-one-dataset-out within each label family (E33 families):
  - melanoma: HAM10000, ISIC 2019 non-HAM, ISIC 2020;
  - malignant: PAD-UFES-20, HIBA, MSK 2020, MILK10k (derm and clin held out together, derm used).
- Within-dataset source splits only where the held-out source has ≥ 50 positives and ≥ 50 negatives:
  - HAM10000: rosendahl, vidir_modern, vienna_dias (vidir_molemax excluded, 24 positives);
  - ISIC 2019: BCN, MSK.
- Within-dataset splits are reported separately and are not counted as independent datasets.

**Models.** Fitted on sources only: frozen ConvNeXt-T, harmonised age/sex/site, the E33 pipeline.

**Decision rules** (fitted on sources only):
- (R0) never use metadata;
- (R1) always use;
- (R2) source-validated: use metadata iff the mean leave-one-source-out Δ among the training sources is > 0, which needs ≥ 2 training sources;
- (R3) E34 B1 collection rule computed from source data only: predicted Δ = source I × median source retained fraction; use iff ≥ 10⁻³ nats.

**Outcome per target.** Realised target log-loss of the chosen model and its regret against the better of the two on the target, plus decision agreement with the target's realised sign of Δ.

**Limitation stated up front.** At most 7 dataset-level targets (3 melanoma + 4 malignant), so the comparison of rules is descriptive.

## Amendment 1 (written 2026-10-29 14:55 IST, after the PAD-UFES-20 replication shake-down run and before any confirmatory set was run)

**What was seen.** E50 ran end-to-end on PAD-UFES-20 complete cases, a replication set. Its t = 0 and T-D1(1) values reproduce E45/E45b exactly: Δ_age 0.64 and 0.88 ×10⁻³ nats; image log-loss 0.52286.

**Problem 1: overfit eraser coordinate.**
- A label-free eraser diagnostic (features and age only, fold 0) shows that the LEACE coordinate c is overfit when n_train ≈ d:

  | Set | n_train | corr(c, age) train | corr(c, age) test | test SD of c / training SD |
  |---|---|---|---|---|
  | PAD-UFES-20 | 871 | 0.97 | 0.21 | 2.7 |
  | HIBA | 977 | 0.97 | 0.33 | 2.0 |
  | MSK 2020 | 939 | 0.94 | 0.16 | 2.1 |
  | HAM10000 | 7962 | 0.72 | 0.61 | 0.97 |
  | MILK10k derm | 4093 | 0.73 | 0.56 | 1.0 |

- Consequence (i): the registered coordinate block (S6-iii) is computed in-sample on the training rows, so it is biased on the small sets. On PAD it *lowered* fit (Δ_coord = −9.2×10⁻³).
- Consequence (ii): the restoration re-append arm's age probe returned R² = −6.7.

The primary contrast, its arms, the confirmatory family and G1 are **unchanged**. Added before any confirmatory run:

1. **Eraser diagnostics** per fold and arm type, computed from features and attribute only: training and test SD of c, and training and test correlation of c with the attribute.
2. **Secondary eraser variant T-LW.**
   - Definition: LEACE with Ledoit–Wolf-shrunk whitening. Any invertible whitening keeps LEACE's exact zero training cross-covariance; shrinkage only changes which oblique projection is used, so the eraser is less driven by low-variance directions.
   - Arms: D1 at t = 1, plus 5 shams with the same variant.
   - Contrasts: against the same matched random arms and against its own shams.
   - Status: secondary; not in the Holm family and not used in G1.
3. **Cross-fitted image-derived attribute block â** (secondary; replaces the in-sample coordinate block as the S6-iii quantity of interest, while that block is still reported).
   - Definition: â is age predicted from the original features by the nested-α ridge probe. Test rows are scored by a fit on the outer training rows; training rows get inner out-of-fold predictions from the same inner splits.
   - Use: it enters the second stage like a metadata variable, in every arm.
   - Contrasts: Δ_â(T) − Δ_â(R), and Δ_â(T) set against the rise Δ_age(T) − Δ_age(0).
4. **Wording rule.** Restoration (i) is interpreted only on sets where the test/training SD ratio of c lies within [0.8, 1.25].

PAD-UFES-20 is re-run with the amended script. The first output is kept in `logs/E50_pad_ufes.log` and noted as superseded.

## Amendment 2 (written 2026-10-29 14:58 IST, before any confirmatory set was run)

**Finding.** In the amended PAD-UFES-20 run, the registered residual-decodability probe gives an identical held-out R² (−0.0031, same CI) under the clipped and the Ledoit–Wolf erasers. This is a property of the method, not of the data. Any exact LEACE eraser sets the training cross-covariance to zero, so a ridge probe trained on erased training rows has zero weights and predicts the training mean. The registered probe therefore confirms only that the guarantee holds. It cannot show whether the erasure generalises to new images.

**Added for every arm.** A within-held-out-fold probe:
- In each outer test fold (never used to fit that fold's eraser), a ridge probe is cross-validated over grouped 5-fold splits of the fold itself, with α chosen by grouped 3-fold CV inside each probe-training part.
- Inputs: features and age only. Diagnostic labels are never used.
- Output: pooled R² with a grouped-bootstrap CI.

It is reported as "age decodability remaining in new images after a training-fold eraser". The registered probe stays and is reported as "decodability learnable from erased training data".

## Amendment 3 / E50c (written 2026-10-29 18:44 IST, after the three confirmatory E50 sets finished; exploratory, motivated by those results)

**What was seen.**
- ISIC 2020 supports the registered prediction:
  - T − matched = +0.99 [+0.69, +1.29] ×10⁻³ nats, p = 0.0005;
  - matching check passes;
  - sham and same-rank controls are inert.
- HIBA and MSK 2020 do not:
  - targeted rank-1 erasure removes most of the image model's diagnostic signal (HIBA AUROC 0.93 → 0.65), already in inner CV, and also with the shrinkage eraser;
  - no random subspace short of 256–512 dimensions matches that loss, so the matched control overshoots and the primary contrast is negative.
- A descriptive diagnostic (in-sample OOF scores, fold assignment 0) relates this to the age–label association. Age-only AUROC for the label and corr(ConvNeXt-T score, age) are 0.84 and 0.66 in HIBA, 0.72 and 0.39 in MSK 2020, and 0.67 and 0.19 in ISIC 2020.
- Explanation: LEACE removes *all* training covariance with age, including covariance that arises because the label itself depends on age. When that dependence is strong, the intervention is not attribute-specific.

**Status of the registered analysis.** Nothing above changes it. G1 is evaluated as registered, and the Holm family is completed with E51 as registered.

**E50c (exploratory).** Class-conditional erasure.
- **Concept.** The training-fold attribute residualised on the training-fold diagnostic class: Z′ = Z − E_train[Z | Y]. Training labels only; outer test labels are never used.
- **What it removes.** X's within-class covariance with the attribute: the attribute trace that is not explained by the label.
- **Sham.** Z′ permuted within class.
- **Arms.** As the light E50 mode: t0, T, 5 nested matched random, 5 same-rank, 5 sham, re-append, T-LW, 5 LW shams. ConvNeXt-T, fold assignment 0, all 8 E50 sets.
- **Predictions (revised hypothesis).**
  - (a) Class-conditional erasure costs much less image log-loss than marginal erasure on HIBA, MSK 2020 and MILK10k.
  - (b) If the image's within-class age trace is redundant with age metadata, T − matched > 0.
- **Reporting.** Paired CIs; no confirmatory claim; no Holm adjustment.
