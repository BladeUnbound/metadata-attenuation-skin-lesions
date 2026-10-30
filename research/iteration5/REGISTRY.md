# Iteration 5 registry

*Each entry is written before the experiment is implemented or run. Entries are never edited after their run starts; changes are appended as dated addenda.*

*Timestamp correction (22:03 IST): the headings originally carried estimated times (21:55 and 22:25). They were replaced with times taken from file metadata: this file was created at 21:48:50 and the E42–E44 entry saved at 21:59:57; e40_crossfit_baseline.py was created at 21:50:01 and first run at 21:50:14; e42_graph.py was created at 22:01:18. No content was changed.*

## E40: strictly cross-fitted baseline (written 2026-10-28 21:48 IST, before implementation)

**Status: exploratory audit.** The original pre-registered analyses (E10, E13, E22) stay as reported; E40 is a sensitivity analysis of their estimator.

**Question.** Does removing the cross-fold label leak (AUDIT.md §2.1, R1) and the imputation leak (R2) change metadata added value or any H1/H2 conclusion?

**Expectation (stated before running).** The leak enters both arms, so |Δ̂_corrected − Δ̂_original| will be small relative to the bootstrap CI width (below half of it) in most cells, and no H1 conclusion will flip. H2 conclusions near the threshold (MSK 2020; complete-case failures) may flip in either direction.

**Data.** HAM10000 and the 7 E13 image sets (ISIC 2020, ISIC 2019 non-HAM, PAD-UFES-20, HIBA, MSK 2020, MILK10k derm, MILK10k clin); all images (pre-registered encoding with missing indicators); rungs HSV, ResNet-18, ResNet-50, ConvNeXt-T, ViT-B/16; fold assignments seeds 0, 1, 2 (identical outer folds to E10/E13).

**Estimators.**
- *Original*: the existing `oof_scores` + `second_stage` + fold-wise metadata encoding, called unchanged. Must reproduce the stored E10/E13 Δ̂ to 5 decimals; any mismatch is reported and stops the run for that dataset.
- *Corrected*: for each outer fold k with training set T_k: (a) first stage fitted on T_k (same tuning as before) scores the test fold; (b) inner StratifiedGroupKFold(5, seed 0) on T_k produces out-of-fold scores for T_k, each from a first stage fitted inside T_k only; (c) metadata for T_k and the test fold are encoded with T_k's medians; (d) both second-stage models (C = 1, standardised) are fitted on T_k's inner scores and metadata and predict the test fold. The test fold is used for nothing but prediction.

**Outputs per dataset × rung × seed.** Image-only and fusion log-loss, AUROC, Brier, calibration intercept and slope; Δ̂ (mean), median Δ_i, Δ̂ by class; group-weighted Δ̂; paired corrected-minus-original difference with grouped-bootstrap CI (B = 2,000, seed 0). Per-observation arrays (Δ_i, ℓ0_i, ℓ1_i, q0_i, q1_i, K_i = KL(q1 ‖ q0), outer fold) saved to `data/<ds>/e40/` for Stages 3–5. H1 (HSV vs ConvNeXt-T) and H2 (ResNet-18 vs ConvNeXt-T) one-sided paired grouped-bootstrap tests recomputed with the corrected estimator, Holm within the original families; reported as a sensitivity only.

**Sensitivity.** ISIC 2019 non-HAM with groups merged over near-duplicate pairs (ConvNeXt-T cosine > 0.98, union-find), corrected estimator, seed 0.

**Decision rule.** If any H1 conclusion flips, or any |change| exceeds half its CI width in more than a quarter of cells, the paper's main results are revised before Stage 3. Otherwise the corrected estimator becomes primary for all new work and the paper gets a short methods note.

**Compute.** About 6× the original first stage: ~1–1.5 h CPU for all datasets, run as one sequential job.

## E42–E44: representation graphs, local ambiguity and metadata benefit (written 2026-10-28 21:59 IST, before implementation)

**State of knowledge when written.** Aggregate Δ̂ from E40 (HAM10000 complete; other datasets running) has been seen. No per-observation Δ_i has been related to any graph, margin, entropy or density statistic. LITERATURE.md shows per-sample prediction of incremental feature value is established (Gadgil et al., ICLR 2024), so these experiments test an empirical question and claim no new estimator.

**Status.** H1, H2 and H4 below are confirmatory for this iteration, with the decision rules fixed here. H3, H5 and the Dirichlet-energy analysis (E44) are exploratory.

**Inputs.** E40 corrected estimator, fold assignment 0. Primary representation ConvNeXt-T; HSV and ResNet-18 for H3; ResNet-50 and ViT-B/16 reported descriptively. Eight image sets; MILK10k derm and clin count as one independent dataset (a hypothesis holds for MILK10k only if it holds in both). Six independent datasets: HAM10000, ISIC 2020, ISIC 2019 non-HAM, PAD-UFES-20, HIBA, MSK 2020 (+ MILK10k = 7 counting units; see decision rules).

**Graph (inductive, prospective).** For outer fold k: nodes are the training images T_k. Features standardised with T_k statistics, then L2-normalised; cosine distance. Each training node connects to its k = 20 nearest training neighbours from *other* groups (same-lesion/patient neighbours excluded); symmetrised by union. Each test image attaches to its 20 nearest training images; no test–test edges. Weights w = exp(−d²/2σ²), σ = median 20-NN distance in T_k. Sensitivities: k ∈ {10, 50}; binary weights. Disconnected components are reported, not repaired.

**Per-observation statistics** (test images, computable without the test label):
- local label entropy LE_i = h(p̃_i), p̃_i = (Σ w_ij y_j + 1)/(Σ w_ij + 2) over training neighbours;
- neighbourhood consistency NC_i = weighted share of neighbours whose label equals the image model's predicted class;
- density: mean distance to the 20 neighbours (log);
- local metadata similarity MS_i = weighted mean over neighbours of [1 − |age_i − age_j|/100 + 1(sex match) + 1(site match)]/3 (missing values count as non-matching);
- model-based baselines: image predictive entropy h(q0_i), margin |logit q0_i|, raw representation norm, K_i = KL(q1_i ‖ q0_i);
- graph-local utility G_i = weighted mean of training neighbours' inner out-of-fold Δ_j, where Δ_j is computed inside T_k only (inner cross-fitting), so the test fold's labels are never used.

**Outcome.** Realised Δ_i from E40 (corrected). Because MSE differences against a noisy target are unbiased for differences in error against E[Δ_i | inputs], realised Δ_i is a valid evaluation target for comparing predictors.

**Hypotheses and decision rules.** Grouped bootstrap (B = 2,000) over the dataset's grouping units; one-sided where a direction is stated; Holm across the 7 counting units within each hypothesis.
- **H1 (local ambiguity):** Spearman(LE_i, Δ_i) > 0 at ConvNeXt-T. Holds overall if it holds in ≥ 5 of 7 units.
- **H2 (decision boundary):** Spearman(h(q0_i), Δ_i) > 0 at ConvNeXt-T. Theory (LITERATURE.md, fact 2) makes this expected; it serves as a manipulation check. Same rule as H1.
- **H3 (representation dependence, exploratory):** the H1 correlation differs between HSV and ConvNeXt-T (two-sided paired bootstrap), and between ResNet-18 and ConvNeXt-T.
- **H4 (incremental predictive value):** for each test fold, a ridge regression (α by inner grouped CV on training nodes, target inner Δ_j) is fitted with (B1) K_i, h(q0_i), margin, representation norm, density; and (B1+G) the same plus LE_i, NC_i, MS_i, G_i. Primary metric: held-out mean squared error of Δ_i; paired per-observation squared-error difference (B1 − B1+G) > 0. Holds if it holds in ≥ 4 of 7 units after Holm. Secondary: utility-capture curve (share of total realised Δ captured when metadata are used only for the top q% of lesions ranked by each predictor, q ∈ {10, 25, 50}), K_i alone vs B1+G.
- **H5 (cross-dataset, exploratory):** leave-one-unit-out: fit B1 and B1+G on the other units (features standardised within dataset), predict the held-out unit; same metrics.

**What would count against the graph framing.** H4 failing, or LE_i and G_i adding nothing beyond K_i and h(q0_i), means graph statistics carry no information about metadata benefit beyond the fitted models. That result is reported as found.

**E44 (exploratory): graph-signal smoothness.** On each dataset's full image graph (transductive, descriptive; same construction with all images, same-group edges removed), normalised Dirichlet energy E(Δ) = ΔᵀLΔ / ΔᵀDΔ (L = D − W; Δ centred), plus the unnormalised and symmetric-normalised versions. Nulls: (i) free permutation of Δ over nodes; (ii) permutation within cells of label × decile of h(q0). Report E_obs / E_null with permutation p (R = 1,000). Interpretation fixed in advance: lower energy than null (i) is expected from label and margin structure alone; only lower energy than null (ii) indicates that metadata benefit is locally organised beyond label and image uncertainty. Energy says nothing about the cause of that organisation.

**Compute.** Inner cross-fitting recomputes E40's inner scores (~1 h CPU, all datasets, seed 0 only); graphs and statistics ~0.5 h; E44 ~0.5 h.

## E45–E48: mechanism, recoverability, missingness and shift-null audits (written 2026-10-28 22:03 IST, before implementation)

All four are **exploratory**. E46 and E47 re-analyse data whose aggregate results have been seen.

### E45: concept erasure, dose–response and restoration (Stage 7)
- **Manipulation.** In each outer training fold, fit LEACE (closed form, Belrose et al. 2023) on training-fold image features with concept Z = one metadata variable (age; sex; site one-hot), using rows where Z is observed. Apply it to all rows. Dose: S_t = S − t(S − S_erased), t ∈ {0, 0.25, 0.5, 0.75, 1}. Restoration is t returning to 0, a sanity check that the pipeline returns to baseline exactly.
- **Manipulation check.** Out-of-fold linear recoverability of Z from S_t (should fall to chance at t = 1), and image-only AUROC and log-loss at each t (side effect).
- **Controls.** (a) Random subspace of the same rank removed; (b) a random subspace whose rank is chosen so that the image-only log-loss rises as much as under concept erasure (matched-performance control); (c) cross-variable specificity matrix (erase age, measure Δ for age, sex and site).
- **Prediction (redundancy mechanism).** At t = 1, Δ for the erased variable rises more than under control (b), and more than Δ for the other variables. If Δ rises equally under (b), the increase reflects a weaker image model, not variable-specific redundancy.
- **Scope.** ConvNeXt-T and ResNet-18; HAM10000, ISIC 2019 non-HAM, PAD-UFES-20 complete cases, MILK10k derm; corrected (E40) estimator; seed 0. Only linear information is erased, so a null result does not exclude nonlinear redundancy. ~1–2 h CPU.

### E46: recoverability re-analysis (Stage 8)
- Re-run E30's association **without** the I_var filter, with (i) absolute Δ_var as outcome; (ii) the original ratio; (iii) Δ_var/(I_var + c) with c = the median I_var. Mixed model: outcome ~ within-class recoverability + I_var + image AUROC + dataset random intercept + variable fixed effects; leave-one-dataset-out jackknife of the recoverability coefficient. Report whether the sign is stable across all variants. The cluster bootstrap with 8 (6 independent) clusters is reported as fragile.

### E47: missingness and source dependence (Stage 9)
- Missingness × label, × source (HAM10000 `dataset`; ISIC 2019 lesion-ID prefix) tables with counts.
- Training-fold multiple imputation without missing indicators (IterativeImputer, m = 5, pooled by averaging predicted probabilities) vs the pre-registered encoding vs complete cases; Δ at HSV and ConvNeXt-T.
- Source-aware evaluation: HAM10000 leave-one-source-out; ISIC 2019 BCN → MSK and MSK → BCN.

### E48: mask-swap null for the CAM overlap test (Stage 10; rationale in THEORY_AUDIT.md §5)
- Recompute melanoma-evidence CAMs (E15 backbones) at 56×56. For image i, compare the mass ratio with its own mask against the masks of 199 other images from the same lesion-area decile; p = (1 + #{M_i(m_j) ≥ M_i(m_i)})/(1 + 199).
- Positive control for the argument: a fixed centred Gaussian map scored under both the shift null and the swap null. Prediction: the shift null declares it significant for most images, the swap null for about 5%.
- Report shares significant under the swap null next to the published shift-null shares.

## E41: replication matrix with alternative label definitions and complete cases (Stage 2; written 2026-10-28 22:30 IST, before implementation)

**Status: exploratory.** It re-analyses datasets whose primary results are known; new label definitions have not been modelled before.

**Label definitions** (fixed from metadata fields, read before any modelling with them):
- Melanoma vs rest (MEL): HAM10000 dx = mel; ISIC 2019 MEL; ISIC 2020 target; PAD-UFES-20 MEL (52 positives, flagged underpowered); HIBA, MSK 2020, MILK10k: diagnosis_2 = "Malignant melanocytic proliferations (Melanoma)".
- Malignant vs benign (MAL): HAM10000 {mel, bcc} (sensitivity: + akiec); ISIC 2019 {MEL, BCC, SCC} (sensitivity: + AK); PAD-UFES-20 {BCC, SCC, MEL} (original); HIBA, MSK 2020, MILK10k diagnosis_1 (original). ISIC 2020 has no distinct MAL label (only melanoma is malignant there).

**Conditions.** E40 corrected estimator, fold assignment 0, all 5 rungs, pre-registered metadata encoding; plus complete cases (every metadata field recorded) under the original labels. Outputs: Δ̂ (image- and group-weighted) with grouped-bootstrap CI; I_A (metadata-only out-of-fold log-loss reduction over prevalence); ρ = Δ̂/I_A at ConvNeXt-T with CI, flagged when I_A's CI lower bound is below 2×10⁻³ nats; H1/H2-type paired tests, unadjusted, descriptive.

**Replication matrix** (deliverable): rows are the paper's claims (H1, H2, the ρ range and its ordering, P1, P2, the HIBA and HAM10000 "almost none" floors); columns are conditions (original estimator; corrected; group-weighted; fold assignments 0/1/2; alternative label; complete cases corrected). Each cell: holds / fails / not derivable.

**Compute.** ~11 label variants and 8 complete-case variants at E40 cost: ~1 h CPU.

## E45b: paired inference for the age-erasure result (written 2026-10-29 09:39 IST, before implementation)

**Status: exploratory follow-up to E45, whose aggregate results have been seen.** Purpose: paired uncertainty for the comparison that E45 reported only as point estimates against a CI.
- **Scope.** Concept = age; HAM10000, ISIC 2019 non-HAM, MILK10k derm, PAD-UFES-20 complete cases; ConvNeXt-T and ResNet-18; corrected estimator; fold assignment 0.
- **Arms.** t = 0; concept erasure at t = 1; same-rank random erasure (5 independent random subspaces); performance-matched random erasure at the rank E45 selected (5 independent random subspaces). Per-observation Δ_age saved for every arm.
- **Inference.** Grouped bootstrap (B = 2,000) of the per-observation paired differences: concept − mean of matched-random arms; concept − mean of same-rank arms; concept − t0. Report the image-only log-loss of every arm. The spread over the 5 random subspaces is reported too.
- **Claim rule (fixed now).** The paper states that erasure restores age's value beyond matched random erasure only for datasets × backbones where the paired 95% CI of (concept − matched) excludes 0.
- **Compute.** ~22 corrected runs per dataset × rung: ~30–45 min CPU.

## E49: uncertainty for the entropy–efficiency decomposition, and a record of the E42 post hoc ranking (written 2026-10-29 09:40 IST, before implementation)

**Status: exploratory.** Both quantities were first computed and seen on 2026-10-28 (STAGE1_REPORT.md finding 4; STAGE3_5_REPORT.md post hoc table). This entry only fixes how they are recorded and given uncertainty.
- **Decomposition.** For each image set, corrected estimator, fold assignment 0, HSV → ConvNeXt-T: share of log-attenuation due to image uncertainty, s_H = log(H_HSV/H_CNX) / log(Δ_HSV/Δ_CNX), with H the image-only held-out log-loss and Δ the mean per-observation benefit. Grouped bootstrap (B = 2,000, same resamples for every quantity). Resamples with Δ_CNX ≤ 0 are counted and excluded; report the share excluded. Repeat the point estimate for fold assignments 1 and 2.
- **Post hoc ranking record.** Spearman of K_i, h(q0_i), B1, B1+G, G and LE with realised Δ_i at ConvNeXt-T (fold assignment 0), and the top-25% capture of K_i, written to results/E49_posthoc_ranking.json and labelled post hoc wherever cited.
