# 10 — Statistical analysis: estimators, uncertainty, multiplicity and their limits

*Phase 7. Written 2026-11-01. It applies to every real-data number in 06 and 08, and to the synthetic numbers in 05 where stated.*

## 1. Estimands and estimators

| Quantity | Population definition (04) | Estimator |
|---|---|---|
| R_Y(V) | Bayes log-loss of Y given V, within the predictor family | mean outer-test log-loss of a penalised logistic head fitted on outer-training rows |
| C_Y | R_Y(Z) − R_Y(S) | mean over test images of l0_i(Z) − l0_i(S) |
| L_YA | R_Y(Z, A) − R_Y(S, A) | mean of l1_i(Z) − l1_i(S) (second stage: image score + age) |
| M_A | Δ_A(Z) − Δ_A(S) = C_Y − L_YA | mean of c_i − l_i |
| η | M_A / C_Y | ratio of means; reported only when C's 95% CI excludes 0 |
| Leakage | how decodable A is from Z | ridge R² (training-fold nested, and within-test), MLP R², \|A − median\| R², age-decile log-loss gain, within-class R² (the last three from REAL-4) |

**Strict cross-fitting (E40)** applies to every arm:
- the eraser, the first-stage head and the second-stage head are all fitted on outer-training rows;
- the outer test fold is used only for prediction;
- patient/lesion groups never cross folds.

**Matched random arms.** Their rank is chosen from inner-CV losses on the training fold only (06 §3).

**Plug-in bias.**
- Each term is a difference of mean held-out losses of *fitted* heads, not of Bayes risks. The heads belong to a restricted family (penalised linear), so all quantities are V-information quantities in the sense of 04 Proposition 1. The identity M = C − L holds exactly for the estimates, by construction.
- The ceiling M ≤ C holds in population only for families closed under the erasure (Proposition 1). Penalised linear heads are not exactly closed under non-orthogonal maps, which is why D1 doses carry a cost (06 §4).

## 2. Uncertainty

**Grouped paired bootstrap.**
- B = 2000 resamples of patient/lesion groups, with the same resample indices (seed 601) for every arm of a run. Arms are therefore compared on identical case resamples, and the CIs of differences reflect the pairing.
- Percentile intervals are reported.

**Ratio CI for η.** Numerator and denominator are recomputed on each resample, and the percentile interval of the ratio is reported.
- When the lower limit of C approaches 0, the ratio's upper tail lengthens. That is why η's CI on ISIC 2019 reaches 2.3 although η > 1 is impossible in population.
- Values of η̂ above 1 (ISIC 2019: 1.05 at ViT-L/16, 1.59 for E50c) are sampling noise in M and C, not violations of the ceiling. Every such case has a CI that includes values below 1.

**What the bootstrap does not capture.**
- It conditions on the fitted erasers and heads, and on one fold assignment. Refitting variability enters only through the fold-assignment replications (fold assignments 1 and 2 for ConvNeXt-T on 3 sets, 06 §8). Their spread (≤ 0.09 in η) is small next to the bootstrap CI widths (0.4–0.9), so the bootstrap is the larger source.
- Random arms average 5 subspace draws per observation, so between-draw variance is mixed into the point estimate, not the CI. The per-draw age-gain values (matched_each_x1e3 in the E50 JSONs) are tightly clustered on the large sets, e.g. HAM ConvNeXt-T 3.8–4.6, ISIC 2019 3.2–3.7, MILK10k derm 57.7–62.0 × 10⁻³ nats, so draw-to-draw variation is small next to the CIs.

**p-values.** One-sided bootstrap p = (1 + #{b : stat_b ≤ 0}) / (B + 1). The minimum attainable value is 0.0005, so "p = 0.0005" means "no resample at or below 0".

## 3. Multiplicity

| Family | Status | Tests | Correction | Result |
|---|---|---|---|---|
| REAL-2 (η_T − η_R > 0, ViT-L/16) | confirmatory, registered 2026-10-30 11:42 IST | 3 (HAM10000, ISIC 2019, ISIC 2020) | Holm, α = 0.05 | 3/3 rejected (Holm p 0.0015 each) |
| Parent paper families (E10–E55) | confirmatory in their own registries | — | as registered there | not re-tested here |
| Everything else (REAL-1, REAL-3 to REAL-8; SYN-1 to SYN-3c) | exploratory | many | none | reported with CIs; no p-value from these is a claim |

**Exploratory multiplicity.**
- REAL-1 and REAL-3 computed 37 η_T − η_R contrasts (results/real/REAL-1_three_objectives.csv), 28 of them on large sets. Among those 28:
  - 24 have p = 0.0005;
  - 2 have p = 0.0015 (ISIC 2019 at ViT-L/16; MILK10k derm, site);
  - 1 has p = 0.011 (ISIC 2019, class-conditional);
  - 1 is p = 1 by the undefined-C rule (ISIC 2019, HSV).
- Holm over all 37 contrasts at α = 0.05 rejects 26, all of them large-set contrasts. The class-conditional ISIC 2019 contrast is not rejected. Bonferroni over 37 (threshold 0.00135) would reject only the 24 at p = 0.0005.
- The *pattern* therefore survives a family-wide correction. It is still exploratory, because these comparisons were not registered as a family.

**What the confirmatory family adds, and what it does not.**
- REAL-2 was registered *after* the ConvNeXt-T exploratory results on the same three datasets had been seen (06; the E50 runs finished 2026-10-29/30).
- It therefore confirms that the effect **generalises to a representation untouched by erasure** (ViT-L/16, used before only for E18's added-value estimate). It is not an independent replication on new patients.
- No external dataset was held back for confirmation. This is a real limitation, recorded in 11.

## 4. Decision rules fixed in advance and applied mechanically

- η is undefined when C's CI includes 0 (registered with REAL-1/2). This applies to PAD-UFES-20 at every rung and to ISIC 2019 at HSV.
- In REAL-2, a set whose C CI (T or R) includes 0 counts as p = 1. It did not apply to any set.
- Small sets (n/d ≈ 1.5) are reported as stress tests (RESEARCH_SCHEDULE v1, Phase 4 "Sets" line, 2026-10-30 10:40 IST). **This rule was not blind.** It was set knowing iteration 6's finding that the eraser overfits at n ≈ d (coordinate test/train SD ratio 2–3). 06 §4 and §6 add further pipeline-check failures on these sets (restoration, shams, matching). The small-set numbers are reported in full in 06, so the exclusion can be checked.

## 5. Equivalence and the fused cost L

- On the large ISIC/HAM sets the fused cost L has CIs that include 0 (06 §2; 08 §1). Absence of evidence is not evidence of absence. No equivalence margin was registered, so no claim of "L ≈ 0" is made.
- A retrospective illustration (not a test): the largest upper limit is 6.0 × 10⁻³ nats (HAM, ConvNeXt-T). That is 2.6% of R_Y(S) = 0.232 and about 3× age's own added value at t0 (2.0).
  - Measured against the image-only risk, the data are consistent with a small fused cost.
  - Measured against age's own contribution, they are not precise enough to call the cost negligible.
- A future confirmatory equivalence test would need a margin justified in clinical units, for example a change in sensitivity at fixed specificity. That margin is not available here.

## 6. Reproduction of surprising results

| Result | Why surprising | Reproduced? |
|---|---|---|
| η_T ≈ 1 with η_R ≈ 0 on ISIC 2019 | the strongest possible substitution | yes: 4 deep rungs and 3 fold assignments, all η_T between 0.92 and 1.06 |
| The class-conditional eraser's in-sample cost is negative while its outer cost is large | sign reversal | yes, on two independent sets (HIBA −54.5 / +285; MSK 2020 −4.5 / +40). The mechanism (labels of inner-validation rows inside the transform) is identified in code (06 §6) |
| η_R rises with age–label dependence | was predicted by SYN-1, so not unexpected | consistent on 7 sets; no trend test |
| D2 partial doses have lower η than full erasure (HAM, MILK10k) | not predicted | **not** reproduced or tested; reported descriptively only |
| LEACE η on MILK10k is 0.60 while the oracle positive control gives 1.01 | rules out an estimator explanation for the lower η | single fold assignment; MILK10k clin (shared lesions) gives η 0.65, but it is not independent |
| Attribute-only adversarial erasure has random-like η (0.06–0.17) | stronger than the registered prediction | 4 arms (2 adversaries × 2 λ), one set and one seed per fold; not replicated on another set |

## 7. Synthetic-study statistics (05)

- Population ground truth for linear transformations comes from Gauss–Hermite quadrature of the Gaussian-logistic Bayes risk (04 §7; notebook 01 §7; scripts/gauss_logit.py).
- Estimator bias is reported as the difference from ground truth, across 3–5 seeds per configuration.
- The SYN-3b metric amendment was registered after SYN-3's results had been seen. Its results are labelled post hoc, and SYN-3's original (registered) metric is reported alongside.

## 8. Remaining statistical limitations

1. **No held-out confirmatory dataset** (§3).
2. **Bootstrap conditional on fits** (§2). Full nested refitting per resample was computationally out of reach (about 2000× the E50 cost).
3. **MILK10k derm and clin share lesions.** They are not independent replications.
4. **Ratio estimator.** η is unstable when C is small. The registered decision rule prevents the worst cases, but CIs remain wide on ISIC 2019 and ISIC 2020.
5. **Leakage is measured by probes of limited capacity.** A probe failing to decode age is not proof of absence. REAL-4 extended the probe families to gradient-boosted trees (08 §3.2); on HIBA even those fail at t0.
6. **Restricted predictor family.** All conclusions are about penalised linear heads on frozen embeddings. E9 (fine-tuned encoders) was not run.

7. **Leakage audit values carry no uncertainty** (§9).

## 9. REAL-4 to REAL-8 (added 2026-11-02)

**Status.** All five registrations are exploratory, and no multiplicity correction is applied. Every p-value below is descriptive.

Comparisons computed:

| Run | Comparisons |
|---|---|
| REAL-5 | 4 η differences (NAE − LEACE) |
| REAL-6 | 24 η_T − η_R contrasts (12 raw, 12 PCA) |
| REAL-7 | 2 sex-arm η contrasts; 6 new specificity cells (the sex row; the age and site rows come from E50/E51); 8 joint-minus-sum interactions in M and 2 in C |
| REAL-8 | no η comparison: every transfer C has a CI including 0, so η is undefined by the registered rule |

**Pairing.** Every cross-run comparison was made only after verifying identical outer-test cases: identical y, groups and t0 per-image losses (real_new_analysis.check_same). This covers NAE vs E50 LEACE, joint vs the three single erasers, and transfer vs the target's E50 arms. All checks passed.

**The positive control as a coverage check.** The oracle's η CI covers the true value 1 on 4/4 large sets, and its L CI covers 0 on 4/4.
- This is consistent with nominal coverage, but it is not a calibration study (4 cases).
- On HIBA both fail. There the bootstrap, which conditions on the fitted heads, does not capture the estimator's finite-sample bias.

**A source of variance the bootstrap does not see: the subsample (REAL-6).**
- Across the three subsample seeds, raw η_T ranges over 0.13 at n/d = 10 (CI widths 0.76–1.12) and 0.22 at n/d = 2 (CI widths 0.66–1.10).
- For PCA, the range is 0.55 at n/d = 2 and 0.68 at n/d = 5, comparable to or larger than the CI widths.
- At small n/d, the bootstrap CI therefore understates the uncertainty of a fitted-pipeline η.

**Leakage values have no CIs.** real_new.py's audit() returns point estimates only (HGB and ridge probes fitted once per fold).
- Differences such as LEACE 0.253 vs matched random 0.275 (HAM within-class R²) are descriptive. Only large differences are interpreted: for example, ridge R² near 0 after LEACE against 0.22–0.26 after NAE.
- The probes' own seed and fit variability is not quantified.

**Direction of tests.**
- REAL-5's one-sided p for η_NAE > η_LEACE is 1.0 for all four arms; every reverse-direction CI excludes 0.
- REAL-7's specificity cells are the E50/E51 one-sided targeted − matched contrasts. The random-arm overshoot biases them towards 0 (06 §3).

**Deviations from the registry** (also in 08 and 09):
1. REAL-6: within-test leakage not computed.
2. REAL-8: source eraser fitted on all source rows instead of outer-training rows.

Neither affects target-test labels.

**Post-hoc analyses**, labelled as such:
- the per-fold breakdown of REAL-6 (q = 5, seed 0, PCA);
- the sex-arm summary (real_new_analysis.py `sex`);
- the site-probe leakage specificity table, read from existing E50/E51 JSONs (08 §6.2);
- the leakage-retention percentages (08 §3.2).

None was registered. All read saved outputs only.
