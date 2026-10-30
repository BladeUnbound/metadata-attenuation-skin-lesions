# Iteration 5: repository and estimator audit, staged plan

*Written 2026-10-28 21:48 IST (file creation time), before any iteration-5 experiment was run. Brief: "Graph-conditional metadata utility in strong image models" (user request, same date). Nothing below claims an experiment has succeeded; items marked **Confirmed** were established by reading code or by the read-only checks listed in §3.*

## 1. Pipeline map (as implemented)

```
raw data (research/data/<ds>/)                       adapters: experiments/e13_replication.py::adapter
  │  metadata CSV ──► label y (dataset-specific, §4.3) ; groups g (lesion / patient / image, §4.2)
  │  images ──► load_224 (resize + 224 centre crop) ──► frozen backbones (e10_ham_embed.py)
  │                                                     emb_<rung>.npy cached per dataset (HSV, R18, R50, CNX-T, ViT-B, + E18/E19/E31 extras)
  ▼
outer folds: StratifiedGroupKFold(5, shuffle, seed ∈ {0,1,2}) on (y, g)
  │
  ├─ FIRST STAGE (e10_ham_attenuation.py::oof_scores)
  │    per outer fold k: StandardScaler on train_k ; L2 logistic, C ∈ {1e-4..1} by GroupKFold(3) inside train_k
  │    ──► score s_i for i ∈ test_k   ⇒  s is out-of-fold for every image
  │    (fine-tuned variants E14/E14b/E31c/E31d: per-fold fine-tuning, logits saved per test fold — same pattern)
  │
  ├─ METADATA (e13_replication.py::oof_block / e10::meta_matrix)
  │    numeric: median imputation (median from train_k) + missing indicator ; categorical: one-hot incl. "unknown"
  │    rows of test_k are encoded with fold k's median ⇒ M is assembled fold by fold
  │
  └─ SECOND STAGE (e10_ham_attenuation.py::second_stage) — reused by every later experiment via x_common.py
       per outer fold k (the SAME folds): fit logistic(C=1, standardised) on {s_j, M_j : j ∈ train_k}
       for [s] and for [s, M] ; predict test_k
       ──► per-image log-loss ℓ0_i, ℓ1_i ;  Δ̂ = mean(ℓ0 − ℓ1) ; CI: bootstrap over groups (models held fixed)
```

Downstream consumers of `second_stage`: E10, E10b–d, E13, E14, E14b, E17, E18, E19, E20, E21, E22, E30–E34 (via `x_common.py`). Out-of-fold scores are cached in `data/<ds>/oof/score_<rung>_<all|cc>_s0.npy`.

## 2. Estimator audit

### 2.1 Does the second stage use in-sample first-stage predictions? — No. Does it leak test labels? — **Yes (Confirmed by code reading).**

Every score is out-of-fold, so no image's own label was used to produce its own score. But the second stage for test fold k is trained on scores s_j, j ∈ train_k, and each such s_j was produced by a first-stage model fitted on all folds except j's fold — **which includes fold k**. So the labels of the outer test fold influence the second-stage training features. This violates the brief's requirement that "the outer test fold must remain untouched throughout model fitting". The same pattern holds for the fine-tuned scores (E14, E14b, E31c, E31d).

Direction and size of any bias are not obvious: both arms ([s] and [s, M]) see the same scores, so the leak may largely cancel in Δ̂. It must be measured, not assumed. **E40** (§6, REGISTRY.md) implements strict cross-fitting: within each outer training fold, inner grouped out-of-fold scores train the second stage, and the outer test fold is scored by a first-stage model fitted on the whole outer training fold.

### 2.2 Metadata imputation — **Confirmed, minor.** Training rows of the second stage carry medians computed from other folds' training sets, which include the current test fold's covariates (not labels). E40 encodes every row of outer fold k with fold k's training median only.

### 2.3 What Δ̂ estimates. With fitted conditional models q0(y|s) and q1(y|s,a), held-out log-loss gives
E[ℓ0 − ℓ1] = I(Y;A | s) + KL(p(y|s) ‖ q0) − KL(p(y|s,a) ‖ q1)  (expectations over the test distribution).
So Δ̂ equals I(Y;A | s) only if (i) both second-stage models are well specified, (ii) estimation error is negligible, and (iii) test and training distributions match. Separately, s = h(S) is a one-dimensional score, so the target is I(Y;A | s), which equals I(Y;A | S) only if Y ⊥ S | (s, A) and Y ⊥ S | s (score sufficiency); otherwise Proposition 1 applied to h gives a further R-type term. The paper's limitation (ii) states the first gap but not the score-sufficiency gap; E10's "full features + metadata" control speaks to it only on HAM10000.

### 2.4 Uncertainty. All CIs resample grouping units with the fitted models held fixed, so they cover test-set sampling variability but not training variability. Three fold assignments give only a partial view of the latter. **Risk**, not a bug; E40 reports the seed-to-seed spread next to the bootstrap CI.

### 2.5 Estimand weighting. Δ̂ averages over images, so patients/lesions with many images weigh more (ISIC 2020: 16.1 images per patient). E40 adds the group-weighted Δ̂ as a sensitivity.

## 3. Read-only checks run for this audit (2026-10-28)

| Check | Result |
|---|---|
| Grouping coverage | ISIC 2020: patient ID, 0% missing (2,056 patients). ISIC 2019 non-HAM: lesion ID missing for 13.6% (grouped per image). PAD-UFES-20, HIBA: patient ID complete. MSK 2020: **no lesion or patient ID** (1,174 images = 1,174 groups). MILK10k: lesion ID complete, no patient ID; derm and clinical sets share all 5,117 lesions. HAM10000: lesion ID only. |
| Shared IDs across datasets | No shared image IDs or patient IDs; lesion IDs shared only between the two MILK10k sets (by design). |
| Near-duplicates (ConvNeXt-T cosine) within datasets | MSK 2020: 0 images with a neighbour > 0.95, so image-level grouping is unlikely to split multi-image lesions there. ISIC 2019 non-HAM: 723 pairs > 0.98, 706 in the same group; **17 pairs (34 images, 0.2%) cross groups and folds**, some exact (cosine 1.000) duplicates under different ISIC IDs, 4 involving a missing lesion ID. HAM10000 reference: 5.8% of images have a neighbour > 0.95. |
| Near-duplicates across datasets | Essentially none (all pairs checked ≤ 0.0009 share > 0.95; ISIC 2020 → HAM10000: about 3 images > 0.98, max 0.985). |
| Lesion-mask geometry (HAM10000, 800 masks, 224 centre crop at 56×56) | Lesion occupies 89% of the central 16×16 block and 9% of a 6-pixel border: **strong centre bias**. |
| E30 exclusion rule (rows with I_var CI > 0) | In the code docstring (file saved 2026-10-27 21:32 IST, run started 2026-10-28 00:20 IST) but **not** in the dated registry REGISTRY_E30_E34.md. |

## 4. Methodological risk register

Status key: **Confirmed** (demonstrated), **Risk** (plausible, unquantified), **Checked-OK**, **Limitation** (cannot be resolved with available identifiers).

| # | Issue | Status | Affects | Action / stage |
|---|---|---|---|---|
| R1 | Second-stage training scores depend on test-fold labels (§2.1) | Confirmed; magnitude unknown | every Δ̂ | E40 strict cross-fitting; compare paired (Stage 1) |
| R2 | Second-stage training rows imputed with medians that include test-fold covariates | Confirmed; label-free, expected negligible | every Δ̂ | fixed inside E40 |
| R3 | Δ̂ ≠ I(Y;A\|S): logistic misspecification and score sufficiency (§2.3) | Confirmed gap in paper text | Theory, limitation (ii) | rewrite estimand paragraph (Stage 6.2) |
| R4 | CIs ignore training variability | Risk | all CIs, Holm tests | report seed spread (E40); repeated cross-fitting |
| R5 | Image-weighted estimand; ISIC 2020 16 imgs/patient | Risk | ISIC 2020, ISIC 2019 | group-weighted sensitivity (E40) |
| R6 | MSK 2020 grouped per image | Limitation, partly mitigated (0 near-duplicates > 0.95) | MSK 2020 CIs | state check in paper; no lesion-level claim |
| R7 | ISIC 2019 cross-group duplicates (34 images) | Confirmed, minor | ISIC 2019 | E40 sensitivity with duplicate-merged groups |
| R8 | MILK10k: no patient ID; derm/clin share lesions | Limitation (already counted once) | MILK10k, P1 | keep; state patient-level dependence unknown |
| R9 | HAM10000: lesion ID only, patients unknown | Limitation | HAM10000 | state |
| R10 | Cross-dataset patient overlap untestable (no common patient IDs) | Limitation; image-level overlap Checked-OK | "independent replication" count | state |
| R11 | Targets differ: melanoma-vs-rest (HAM, ISIC 2019, ISIC 2020) vs malignant-vs-benign (PAD, HIBA, MSK, MILK); prevalence 1.8%–71% | Risk to comparability | H1/H2 across datasets, ρ | Stage 2.3: re-run with both label definitions where derivable |
| R12 | Missingness label-informative; complete cases change the population | Known (paper §cc) | all | Stage 9: training-fold multiple imputation without indicators as a further sensitivity |
| R13 | ρ = Δ/I_A unstable when I_A is small | Risk | ρ table, P1 | Stage 2.1: absolute Δ with CIs first; flag I_A lower bound |
| R14 | E30: unregistered filter; 95 rows share datasets/representations/variables; cluster bootstrap with 8 (6 independent) clusters | Confirmed | Discussion "which variables keep value" | Stage 8 re-analysis; likely narrow to exploratory/descriptive |
| R15 | Shift-null p-values assume mask placement is exchangeable under circular shifts; masks are centred (§3) | **Confirmed assumption violation** | Interpretability section | Stage 10: mask-swap null across images, centre-only baseline map |
| R16 | E32 permutation tests A ⊥ (X,Y), not A ⊥ Y \| S | Checked: paper's claim ("not an artefact of extra parameters") matches what it tests | — | optional conditional permutation (Stage 7.4) |
| R17 | "Added value then attenuates as S improves" is not implied by Proposition 1; for nested S1 = h(S2), Δ(S1) − Δ(S2) = R(S1) − R(S2) can take either sign, and the frozen rungs are not nested | Confirmed overstatement (wording) | Theory interpretation | Stage 6.2 rewrite |
| R18 | Fine-tuned scores share R1; correcting needs 25 fine-tunes per dataset | Confirmed pattern | E14, E14b, E31c/d | decide after E40 (if frozen-feature leak is negligible, document; otherwise re-run) |

The Proposition itself is correct as stated: Y is binary, so every term is bounded by log 2; the derivation is two chain-rule expansions plus I(Y;A | X,S) = I(Y;A | X) for S = g(X). The injectivity clause needs a measurable inverse, which holds for injective Borel maps between standard Borel spaces (Lusin–Souslin); this should be stated.

## 5. A theoretical expectation that shapes the graph hypotheses (written before any graph analysis)

For a new lesion the realised benefit is Δ_i = log q1(y_i|s_i,a_i) − log q0(y_i|s_i). If q1 is calibrated, its expectation given (s_i, a_i) is
E[Δ_i | s_i, a_i] = KL( q1(·|s_i,a_i) ‖ q0(·|s_i) ) =: K_i ≥ 0,
which needs no label. So any label-free predictor of Δ_i (graph statistics included) can beat K_i only where the fitted models are miscalibrated or misspecified locally, e.g. if the metadata effect varies across regions of representation space, which a single global logistic second stage cannot express. This makes H4 sharply falsifiable: graph statistics must add to K_i, not merely to margin or entropy. Before metadata are collected, the relevant quantity is the local conditional mutual information I(Y;A | S = s) = E_A[K | S = s], which a graph can estimate nonparametrically from training neighbours. Whether this is new is a Stage 12 question; kNN estimators of conditional mutual information exist.

Consequence: Δ_i is a noisy, label-dependent, per-observation quantity. Analyses that relate it to graph statistics computed with test labels are descriptive. Prospective analyses use only training labels and inductive graphs (test nodes attach to training nodes only).

## 6. Staged plan

Compute figures are estimates from measured runtimes (E13 analysis ≤ 4 min per dataset for 3 seeds × 5 rungs; Apple M5, 10 cores).

| Stage | Experiment | What | Cost | Gate at the end |
|---|---|---|---|---|
| 1 | **E40** | Strict cross-fitted baseline, all 8 image sets × 5 rungs × 3 seeds; paired original-vs-corrected Δ̂; per-observation Δ_i, K_i, calibration, per-class Δ; group-weighted and duplicate-merged sensitivities; H1/H2 re-tested as a sensitivity (pre-registered results stay as reported) | ~1–1.5 h CPU | If no H1/H2 conclusion flips and |change| ≪ CI width → corrected estimator becomes primary for new work, paper gets a short note. If conclusions flip → revise the paper before Stage 3. |
| 2 | E41 | Replication matrix: both label definitions where derivable; ρ stability; training-condition matrix assembled from E14/E31/E18/E19 with leak status | ~1–2 h CPU | Which original claims survive consistent evaluation |
| 3–5 | E42–E44 | Inductive kNN graphs per representation (cosine on standardised features; k ∈ {10, 20, 50}; Gaussian and binary weights; bandwidth = median kNN distance); training-label entropy, margin, density, metadata similarity; H1–H5 pre-registered in REGISTRY.md **before** any graph–Δ association is computed; Dirichlet energy with degree- and scale-normalised versions and two nulls (free permutation; permutation within margin bins) | ~3 h CPU | Proceed to Stage 6 only if graph statistics add to K_i and margin on held-out groups |
| 6 | — | Formal proposal (local CMI on representation graphs) only if Stage 5 supports it; 6.2 theory rewrite regardless; 6.1 PID only if it yields a distinct testable prediction | analysis | — |
| 7 | E45 | Ablation–restoration on cached features: project out linearly decodable metadata directions (iterative nullspace projection), dose = number of directions; random-direction matched control; restore and re-measure | ~1–2 h CPU | Mechanism evidence under stated design assumptions |
| 8 | E46 | E30 re-analysis: no filter, absolute Δ_var, alternative ratios, mixed model with dataset and variable effects, leave-one-dataset-out | ~0.5 h CPU | Keep, narrow, or relabel descriptive |
| 9 | E47 | Missingness: missingness-vs-label/source tables; multiple imputation without indicators; source-aware splits (HAM10000 source; ISIC 2019 BCN vs MSK) | ~1 h CPU | — |
| 10 | E48 | Shift-null audit: recompute CAMs (~20 min GPU), mask-swap null across images matched on lesion area, centre-only baseline map | ~0.5 h GPU+CPU | Narrow or revise interpretability claims |
| 11 | — | Controls consolidated across stages | — | — |
| 12 | — | Literature/novelty audit (web) for local metadata benefit, per-sample fusion gain prediction, local CMI, PID in multimodal learning, graph Dirichlet energy | analysis | Narrow or abandon novelty claims as needed |
| — | deferred | Strictly cross-fitted fine-tuning (R18) | ~10–15 h GPU | only if E40 shows a non-negligible leak |

Integrity rules carried over: preserve the originals (`ORIGINALS.sha256`), never relabel exploratory work as pre-registered, run jobs sequentially, record failures.

*Status updates after Stage 1: see STAGE1_REPORT.md (R1, R2, R4, R5, R7, R18).*
