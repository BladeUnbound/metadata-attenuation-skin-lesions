# Experiment registry — research_extension

Rules:
- Every experiment gets an entry **before** it runs, with time from `date`: hypothesis, protocol (data, splits, methods, metrics, tests), compute estimate, and status (confirmatory or exploratory).
- An entry is never edited after its run starts. Changes are appended as dated amendments that state what had been seen.
- Confirmatory families declare their Holm correction in advance. Anything else is exploratory, whatever its outcome.
- Outer-test labels never enter transformation fitting, probe fitting, hyperparameter or rank selection, imputation or threshold choice.

## Entries

### SYN-1 — pipeline validation against exact ground truth (linear-Gaussian regimes)
*Registered 2026-10-30 10:54 IST, before any synthetic code existed. Status: exploratory (method validation; no real data).*
- **Purpose.** Check that the empirical pipeline (finite-sample erasers, held-out logistic heads, probes) recovers the exact population quantities of 04 §3 and §7, and separates redundancy removal from destruction of diagnostic information.
- **Data.** Latent factors (A, U, G, nuisance) drawn as Gaussians and mixed into S ∈ ℝ^d by a random rotation plus isotropic noise. Y ~ Bernoulli(σ(b + β_A A + β_U U + β_G G)).
- **Regimes.**
  - R1 redundant (β_U = 0);
  - R2 entangled (β_U = 0.8);
  - R3 irrelevant (β_A = β_U = 0, A ⊥ G);
  - R6 strong age–label (β_A = 2.5, β_G = 0.5);
  - R7 shared component (corr(A, G) = 0.6, β_A = 0).
- **Grid.** n ∈ {1000, 5000, 20000} (50/50 train/test), d ∈ {32, 128}, 5 seeds (seed = 1000·regime + 100·d_index + 10·n_index + rep).
- **Methods** (fitted on the training half). The first six are A-free at inference:
  - none;
  - LEACE (clipped whitening, as E45);
  - LEACE with Ledoit–Wolf whitening;
  - orthogonal removal of Cov(S, A), i.e. the RLACE/OLS closed form;
  - random rank-1 subspace;
  - random subspace matched for cost by an inner-CV search on training rows only;
  - sham (LEACE on permuted A);
  - oracle residual Z = S − β̂A, which uses A at inference (positive control).
- **Metrics.**
  - Exact population C_Y, M_A, L_YA, η and I(A;Z) for the fitted linear map, by quadrature.
  - Held-out estimates from (i) near-unpenalised logistic heads and (ii) E40-style penalised heads (C ∈ logspace(−4,0,5), 3-fold CV).
  - Leakage: ridge R² trained on train; ridge R² within test (5-fold); MLP R² (one hidden layer of 64, train → test).
- **Predictions** (from 04):
  - (a) Fitted LEACE has exact η ≈ 1 in R1, η < 1 in R2, and C = M = 0 in R3.
  - (b) Exact quantities of LEACE, LW-LEACE and orthogonal removal coincide (Proposition 4).
  - (c) Random cost-matched η is far below LEACE's η.
  - (d) Oracle η = 1 up to estimation error.
  - (e) Held-out estimates track the exact values, with bias shrinking in n.
- **Compute.** About 1,200 fits, closed-form erasers and logistic heads on up to 20k × 128: estimated 15–30 min CPU.

### SYN-2 — n/d instability and the head–geometry hypothesis
*Registered 2026-10-30 10:54 IST. Status: exploratory. Motivated by Proposition 4 (04 §7b) and the E50 small-set failures.*
- **Hypothesis H-S2.** At n_train/d near 1, LEACE and orthogonal removal remove information-equivalent content (same estimated kernel). Under penalised heads, LEACE's oblique geometry produces a larger measured diagnostic cost and a matching failure. The size of the excess is predicted by the test/train SD ratio of LEACE's coordinate.
- **Data.** R1 and R2. d ∈ {64, 256, 768}; n_train/d ∈ {1.25, 1.5, 2, 3, 5, 10, 25}; test n = 10,000; 5 seeds.
- **Methods.** none; LEACE (clipped); LEACE (LW); orthogonal; random cost-matched (inner CV on training).
- **Metrics.**
  - Exact population cost (identical by Proposition 4 for the three erasers; checked).
  - Measured costs with penalised and near-unpenalised heads.
  - Coordinate SD ratio; condition number of Σ̂_SS; LW shrinkage intensity.
  - The angle between the estimated and true Cov(S, A) directions.
- **Predictions.**
  - (a) Exact quantities are identical across the three erasers.
  - (b) The measured penalised-head cost of LEACE exceeds that of orthogonal removal as n/d → 1, and the excess correlates with the SD ratio.
  - (c) Near-unpenalised heads remove most of the gap.
- **Compute.** 3 d × 7 ratios × 5 seeds × 5 methods ≈ 525 fits, with tuned heads on d ≤ 768: estimated 30–60 min CPU.

### SYN-3 — nonlinear encoding, nonlinear and task-aware attribute-free erasers, adaptive probes
*Registered 2026-10-30 10:58 IST, after SYN-1/SYN-2 started and before any of their results were seen. Code written (scripts/nonlin.py, scripts/synth3.py) but not run. Status: exploratory.*
- **Purpose.** Directive §§3, 7, 10:
  - whether linear erasure "succeeds" only against linear probes;
  - whether nonlinear adversarial erasure removes information or merely defeats its training adversary;
  - whether a fused-task-preserving transformation (the only MATE form 04 §9 supports) achieves lower irreplaceable cost L_YA than image-task-preserving or pure adversarial erasure at matched leakage.
- **Data.** d = 32; three splits: fit 8k, head/probe 8k, test 20k.
  - Regimes: L1 linear redundant, L2 linear entangled (β_U = 0.8), N1 nonlinear redundant (L1 plus channels A² − 1 and cos 2A, which have zero linear covariance with A), N2 nonlinear entangled.
  - 3 seeds (0, 1, 2).
- **Methods** (all attribute-free at inference):
  - none; LEACE;
  - NAE: residual MLP against an MLP age adversary, λ_A ∈ {0.5, 2, 8}, λ_D = 1;
  - TP-img: as NAE plus an image-only diagnosis head, λ_Y = 1;
  - TP-fused: as NAE plus a diagnosis head on [z, A], λ_Y = 1;
  - sham: NAE against permuted A, λ_A = 8.
- **Evaluation.** Heads and probes are fitted on the head split after the transformation is frozen.
  - C, L, M and η from tuned logistic heads and from gradient-boosted heads.
  - Leakage R² for ridge, MLP (2×128) and HGB probes, and within-class HGB R² (conditional leakage, complementary).
- **Predictions.**
  - (a) N1/N2: after LEACE, linear-probe R² ≈ 0 but MLP/HGB R² stays well above 0.
  - (b) NAE lowers nonlinear leakage, but post-hoc probes recover more than the training adversary implies.
  - (c) At comparable post-hoc leakage, TP-fused has lower L_YA (higher η) than TP-img and NAE. TP-img keeps C_Y lower but leaks more (Proposition 2).
  - (d) In L1/L2, no nonlinear method beats LEACE's η at matched leakage.
  - (e) The sham has C ≈ M ≈ 0.
- **Compute.** About 120 adversarial fits of 10–30 s each, 6 workers in parallel: estimated 10–20 min CPU.

### SYN-3b — measurement amendment to SYN-3 (registered 2026-10-30 11:29 IST, after SYN-3 results were seen)
- **What was seen.** In N1/N2 the extra age channels are even functions of A. They carry |A| but not the sign of A, so R² for signed A is 0 for every probe after LEACE, even though age information remains (within-class HGB R² 0.16 vs 0.75 before erasure). Prediction (a) therefore could not be tested with the registered metric. The nonlinear erasers also did not reach LEACE's leakage level at λ_A ≤ 8.
- **Change.** Same design and seeds. λ_A grid extended to {0.5, 2, 8, 32}. Leakage now also includes:
  - HGB R² for |A|;
  - age-decile classification: an HGB multiclass probe on head-split decile bins, scored by the test log-loss gain over the prior as a fraction of log 10.
- **Status.** Exploratory; SYN-3 results are kept unchanged (results/synthetic/SYN-3.csv). Predictions (a)–(e) unchanged.
- **Compute.** About 4 min.

### SYN-3c — distributional adversary (registered 2026-10-30 11:37 IST, after SYN-3b results were seen)
- **What was seen.** NAE with an MSE adversary for signed A leaves almost all \|A\| information (\|A\| R² 0.59–0.88 vs 0.90 before). LEACE leaves all of it (0.895).
- **Hypothesis.** Nonlinear adversarial erasure removes what its adversary's loss can detect. An adversary trained to classify age deciles (cross-entropy, capped at log 10 in the transform's objective) will remove \|A\| information that the MSE adversary leaves.
- **Prediction.** In N1/N2, \|A\| R² and decile gain fall well below NAE-MSE at matched distortion. In N1 (redundant), η stays ≈ 1 while C_Y grows; in N2, η stays ≤ LEACE's.
- **Design.** As SYN-3b, regimes N1 and N2 only. Methods: NAE-decile and TPfused-decile at λ_A ∈ {0.5, 2, 8, 32}. Same seeds, splits, heads and probes.
- **Status.** Exploratory. **Compute.** About 3 min.

---
## Phase 4/6 — real data (registered 2026-10-30 11:42 IST, after Phases 1–3 and before any new real-data computation)

Common to all REAL entries:
- Frozen embeddings; strict cross-fitted two-stage estimator (E40) reused from research/experiments/e50_dose_restoration.py; fold assignment 0 unless stated.
- Per-observation losses on identical outer-test cases; patient/lesion-grouped paired bootstrap (B = 2000).
- C_Y = R(Z) − R(S) is from the first-stage image log-loss. Δ, M_A and L_YA = R(Z,A) − R(S,A) come from the age second stage.
- η = M_A / C_Y is reported only when the CI of C_Y excludes 0; its CI is the bootstrap ratio CI.
- Leakage is reported as: training-fold ridge R², within-test-fold ridge R², MLP R², and (new) age-decile log-loss gain and \|age − median\| R² from HGB probes fitted on outer-training features.

### REAL-1 — three-objective re-analysis of existing erasure runs (E1, E2, E3, E10, E11). Exploratory.
- **Input.** The saved per-observation arrays of E50a (8 sets: LEACE D1 and D2 doses, 5 nested matched random, 5 shams, re-append), E50b (fold assignments 1–2, ResNet-18), E50c (class-conditional) and E51 (site). Read only; no refitting.
- **Output, per arm.** C_Y, M_A, L_YA and η with CIs; η_T − η_R (paired); three Pareto views; and E11 missing-age risk curves. E11 computes, for a test missingness rate π, the expected risk π·R(Z) + (1 − π)·R(Z,A) under the trained complete-data models, for S and for each Z (this assumes MCAR at test).
- **Questions.** Where does each set sit among the substitution regimes of 05? Does η separate LEACE from random erasure, and how does that relate to age–label dependence?

### REAL-2 — confirmatory: substitution efficiency on an untouched representation (ViT-L/16 SWAG, frozen)
- **Status: CONFIRMATORY.** Erasure has never been run on ViT-L/16 features in this project. The ViT-L/16 embeddings were used only in E18 (added value of all metadata), never with erasure.
- **Hypothesis.** On large datasets, LEACE age erasure is more substitutive than performance-matched random erasure: η_T − η_R > 0.
- **Family.** Holm over 3 tests, α = 0.05, one-sided:
  - HAM10000, ISIC 2019 (non-HAM), ISIC 2020 (fold assignment 0).
  - Statistic: the paired grouped-bootstrap distribution of η_T − η_R, where η_R uses the mean over 5 matched random subspaces; p = (1 + #{b ≤ 0})/(B + 1).
  - A set whose C_Y CI (for T or R) includes 0 counts as a failure (p = 1).
- **Secondary.** T − matched contrast in M_A; the matching check; T − sham.
- **Method.** e50_dose_restoration.py --rung vitl16_swag --minimal, unchanged code path.
- **Compute.** About 1–1.5 h.

### REAL-3 — representation ladder (E6). Exploratory.
- LEACE η, η_T − η_R and the T − matched contrast at HSV, ResNet-18 (E50b), ConvNeXt-T (E50a), ViT-B/16 and ViT-L/16 (REAL-2), on HAM10000 and ISIC 2019 (non-HAM); minimal mode.
- **Question.** Does representation strength change how much age is encoded (leakage before erasure), η, and the size of the substitution?
- **Compute.** About 1 h (HSV and ViT-B/16 on 2 sets).

### REAL-4 — positive control and distributional leakage audit. Exploratory.
- **Positive control (oracle residual).** Z_k = S − β̂_k·A, with β̂_k fitted on outer-training rows, used together with A (attribute-conditional). Population η = 1, so the estimate measures how far finite-sample estimation pulls η̂ below 1.
- **Leakage audit.** Age-decile gain, \|age − median\| R² and within-class R² for t0, LEACE, matched random (j = 0) and sham (j = 0), using the E50 seeds.
- **Sets.** The 4 large sets plus HIBA (stress test).
- **Compute.** About 1 h.

### REAL-5 — E4 nonlinear adversarial audit on the development set (HAM10000, ConvNeXt-T). Exploratory.
- **Arms.** NAE-MSE and NAE-decile (scripts/nonlin.py; attribute only, no diagnostic labels) at λ_A ∈ {2, 8}, one per outer fold.
- **Evaluation.** Strict cross-fitted C_Y, M_A, L_YA and η, plus the full leakage audit; a distortion-matched sham is not available, which is noted as a limitation.
- **Prediction (from SYN-3/3c).** The nonlinear erasers do not exceed LEACE's η, and leave linear leakage above LEACE's.
- **Compute.** About 1 h.

### REAL-6 — n/d scaling and the PCA remedy (E7). Exploratory.
- **Design.** On ISIC 2019 (non-HAM), ConvNeXt-T: subsample outer-training rows (by group) to n_train/d ∈ {1.25, 2, 5, 10} with 3 subsample seeds; keep the full test folds.
- **Arms.** t0, LEACE, nested matched random (5), and PCA(k)-then-LEACE with k = n_train/10 (PCA fitted on training rows).
- **Metrics.** Matching check, η and its CI width, coordinate SD ratio, within-test leakage.
- **Prediction (SYN-2).** At small n/d, η estimates and matching become unstable while point estimates of η remain near the full-data value; PCA narrows the instability.
- **Compute.** About 1.5 h.

### REAL-7 — attribute specificity with sex, and joint removal (E5, E12). Exploratory.
- **Specificity.** Sex erasure (LEACE on the sex indicator, minimal mode) on HAM10000 and MILK10k derm completes a 3×3 matrix (age, site, sex erasers × age, site, sex gains) with E50/E51.
- **Joint removal.** LEACE on [age, sex one-hot, site one-hot] jointly, against single-attribute erasers, on HAM10000 and MILK10k derm. Reports C_Y, M for each attribute and for all metadata, and interactions (joint M minus the sum of single M's).
- **Compute.** About 1.5 h.

### REAL-8 — cross-dataset transfer (E8). Exploratory.
- **Design.** Fit the LEACE eraser on outer-training rows of a source set; apply it to the target set's folds (ConvNeXt-T space). Pairs: HAM10000 ↔ ISIC 2019 (non-HAM), and HAM10000 → ISIC 2020. Target heads are fitted on the target only.
- **Metrics.** Target C_Y, M_A, η and leakage, against the target's own eraser.
- **Compute.** About 1 h.

**Not scheduled, with reasons.**
- **E9 (fine-tuned encoders).** Needs split cross-fitted fine-tuning and penultimate-feature extraction, about 3–5 h of GPU per dataset and backbone.
- **Confirmatory MATE comparison.** Not justified by Phases 2–3 (05 §SYN-3).

Both are recorded in 11_final_research_assessment.md as open work.
