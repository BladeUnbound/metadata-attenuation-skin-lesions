# 06 — Real-data baselines: the three-objective frontier of existing erasers (REAL-1)

*Phase 4. Written 2026-11-01 from saved per-observation arrays; no model was refitted for this document.*
- *Scripts:* scripts/real_reanalysis.py (tables, Pareto figure) and scripts/real_figs.py (η figure).
- *Raw results:* results/real/REAL-1_three_objectives.{csv,json}.
- *Figures:* figures/REAL-1_pareto.{pdf,png} and figures/REAL-1_eta.{pdf,png}.
- *Registered as REAL-1 (exploratory)* in EXPERIMENT_REGISTRY.md, 2026-10-30 11:42 IST.
- *Not covered here:* the confirmatory test (REAL-2) and the representation ladder (REAL-3) are reported in 08. The ViT-L/16 and HSV/ViT-B runs those two needed were new computation (EXECUTION_LOG, 2026-10-30 11:42–14:05 IST).

Directive items covered: E1 (frontier of existing methods), E2 (matched-cost comparison), E3 (substitutability regimes), E10 (restoration and dose), E11 (missing metadata), and the site column of E5.

## 1. What was measured

**Representations and protocol.**
- Frozen ConvNeXt-T embeddings (d = 768); fold assignment 0 unless stated.
- Strict cross-fitted two-stage estimator (E40): 5 patient/lesion-grouped outer folds.
- Every eraser and head is fitted on the outer-training rows only. The outer test fold is used only for prediction.
- For each test image: the image-only log-loss l0 and the image + age log-loss l1.

**Arms.** Every eraser is fitted on the training fold, using the attribute and never diagnostic labels; E50c, below, is the exception.

| Arm | Definition |
|---|---|
| t0 | unchanged features S |
| T-D1-1.0 | LEACE. Rank 1 for scalar age |
| T-D1-t, t < 1 | partial D1 dose X − t·c·Bᵀ. An invertible map of S, so information-preserving in population (04) |
| T-D2-t | information dose: the erased coordinate is replaced by √(1−t)·c + √t·σε |
| T-LW-D1-1.0 | LEACE with Ledoit–Wolf whitening |
| R-D1-1.0-* | random whitened subspace, mean over 5 draws. Rank: the smallest grid rank (doubling, then one √2 refinement) whose inner-CV, training-fold image log-loss reaches LEACE's |
| SR-* | random whitened subspace of LEACE's rank (1), 5 draws |
| sham-* | LEACE fitted to permuted age, 5 draws |
| rest-T | LEACE output with the erased coordinate re-appended. [Z, c] is an invertible linear map of S, so this is a pipeline check |
| E50c (…_light_cond) | class-conditional LEACE: the age coordinate is residualised on training-fold class means before fitting. It uses training labels and removes pooled within-class age covariance (linear Z ⊥ A \| Y, the separation / equalised-odds analogue) |

**Quantities** (definitions in 04; all in 10⁻³ nats per image):
- C = mean(l0(Z) − l0(S)) is the image-only cost.
- L = mean(l1(Z) − l1(S)) is the fused, irreplaceable cost.
- M = C − L is the gain in age's added value.
- η = M / C is the substitution efficiency.

**Statistics.**
- CIs come from a grouped paired bootstrap (B = 2000), using the same resamples for every arm and for numerator and denominator.
- η is reported only when C's CI excludes 0.
- η_T − η_R is the paired bootstrap difference, with one-sided p = (1 + #{diff ≤ 0}) / (B + 1). The minimum attainable p is 0.0005.
- Leakage is the ridge R² for age: training-fold (nested) and within-test-fold.

## 2. Main table: LEACE and matched random erasure (ConvNeXt-T, fold assignment 0)

The sets are ordered by how well age alone predicts the label: age-only AUROC from research/results/E50_age_label_diag.json.

| Set (n) | age AUROC | Arm | C [95% CI] | M | L [95% CI] | η [95% CI] | η_T − η_R (p) |
|---|---|---|---|---|---|---|---|
| PAD-UFES-20 (1,089) | 0.516 | LEACE | 2.6 [−0.9, 6.4] | 0.2 | 2.4 | undefined (C CI includes 0) | — |
| | | random | 3.8 [−0.1, 7.6] | 0.0 | 3.8 | undefined | |
| ISIC 2019 non-HAM (15,316) | 0.599 | LEACE | 7.5 [4.2, 10.7] | 7.2 | 0.25 [−2.7, 3.2] | **0.97** [0.64, 1.56] | 0.94 (0.0005) |
| | | random | 11.5 [9.4, 13.7] | 0.3 | 11.2 | 0.03 [−0.01, 0.06] | |
| HAM10000 (10,015) | 0.671 | LEACE | 14.1 [10.3, 17.8] | 11.2 | 2.9 [−0.1, 6.0] | **0.79** [0.61, 1.01] | 0.71 (0.0005) |
| | | random | 25.5 [22.3, 28.7] | 2.1 | 23.4 | 0.08 [0.06, 0.11] | |
| ISIC 2020 (33,126) | 0.672 | LEACE | 1.70 [0.92, 2.49] | 1.21 | 0.49 [−0.18, 1.16] | **0.71** [0.50, 1.18] | 0.61 (0.0005) |
| | | random | 2.07 [1.57, 2.56] | 0.22 | 1.85 | 0.10 [0.06, 0.15] | |
| MILK10k derm (5,117) | 0.715 | LEACE | 94.4 [83.8, 104.8] | 56.6 | 37.8 [29.0, 46.5] | **0.60** [0.53, 0.68] | 0.25 (0.0005) |
| | | random | 133.1 [122.4, 144.2] | 47.1 | 86.0 | 0.35 [0.32, 0.39] | |
| MILK10k clin (5,117) | 0.715 | LEACE | 91.8 [81.5, 102.1] | 60.1 | 31.7 [24.1, 39.6] | **0.65** [0.59, 0.72] | 0.27 (0.0005) |
| | | random | 126.7 [115.3, 138.2] | 48.5 | 78.2 | 0.38 [0.34, 0.42] | |
| MSK 2020 (1,174)† | 0.72 | LEACE | 30.6 [17.5, 44.7] | 10.3 | 20.4 | 0.34 [0.11, 0.55] | −0.16 (0.93) |
| | | random | 69.0 [53.1, 86.1] | 34.1 | 34.9 | 0.49 [0.44, 0.56] | |
| HIBA (1,224)† | 0.84 | LEACE | 320 [277, 361] | 117 | 203 | 0.37 [0.15, 0.54] | −0.15 (0.99) |
| | | random | 446 [409, 483] | 234 | 212 | 0.52 [0.45, 0.59] | |

† n/d ≈ 1.5. These sets are a documented stress test, not evidence (§5 and §8: restoration fails on them).

Leakage (ridge R² for age, within-test):
- t0 lies between 0.23 (ISIC 2020) and 0.34 (MILK10k clin); HAM and ISIC 2019 are 0.32–0.33, MILK10k derm 0.28.
- LEACE brings it to |R²| ≤ 0.03 on every large set; the training-fold R² is ≤ 0.002 in absolute value.
- Matched random leaves 0.06–0.29.

figures/REAL-1_pareto.png shows, per set, the leakage–C plane, the leakage–M plane, and C against M with the M = C ceiling. figures/REAL-1_eta.png (left) plots η against age AUROC.

### Findings

1. **Identity 3 holds empirically.** No arm on any set has M significantly above C. All points in the bottom row of the Pareto figure lie on or below the dashed M = C line within their CIs.
2. **Most of LEACE's image-only cost is substitutable by recorded age.**
   - On the three large ISIC/HAM sets, LEACE's η is 0.71–0.97. The fused cost L has a CI that includes 0 on all three; this is a failure to reject, not a demonstration that L = 0.
   - On MILK10k, roughly 60–65% of the cost is substitutable, and L is clearly positive (32–38 × 10⁻³ nats).
3. **Random erasure at equal or larger image cost is barely substitutable** on the three ISIC/HAM sets (η_R 0.03–0.10). LEACE's η exceeds it by 0.61–0.94 (all p = 0.0005).
4. **The gap narrows as age becomes more predictive of the label.**

   | Quantity | ISIC 2019 | HAM | ISIC 2020 | MILK derm | MILK clin | MSK (stress) | HIBA (stress) |
   |---|---|---|---|---|---|---|---|
   | η_R | 0.03 | 0.08 | 0.10 | 0.35 | 0.38 | 0.49 | 0.52 |
   | η_T − η_R | 0.94 | 0.71 | 0.61 | 0.25 | 0.27 | −0.16 | −0.15 |

   This matches the SYN-1 regime prediction (05): when the label depends strongly on age, information destroyed *anywhere* in the representation tends to be age-correlated, so random erasure also becomes substitutable (SYN-1 strong-dependence regime: random η = 0.93). Five large sets with two tied AUROCs (HAM 0.671 and ISIC 2020 0.672) do not support a quantitative trend. The ordering is descriptive.
5. **PAD-UFES-20 sits in the "irrelevant metadata" regime.** Age carries almost no label information (AUROC 0.516), and neither LEACE nor random erasure has a detectable cost. η is undefined, as 04 predicts for this regime. With n = 1,089 the study also has little power.

## 3. Matched-cost comparison (E2) and its limitation

**How the matching works.** The random rank is chosen on *training-fold* inner-CV image log-loss: the smallest grid rank reaching LEACE's. By construction, then, the random arm costs at least as much as LEACE in training. On the outer test it overshoots on every large set (ConvNeXt-T, fold 0):

| Set | Matched ranks (5 folds) | Test image log-loss, LEACE − random [95% CI] | Matching check |
|---|---|---|---|
| ISIC 2020 | 32–128 | −0.37 [−1.19, 0.38] | passes |
| ISIC 2019 | 23–91 | −3.99 [−7.38, −0.40] | random costlier |
| HAM10000 | 91–181 | −11.4 [−15.8, −7.4] | random costlier |
| MILK10k derm / clin | 512 (every fold) | −38.7 / −34.9 | random costlier |

On MILK10k the grid stepped from 362, whose loss was below the target, to 512, which overshot it (rank_search_grid in the E50 JSON). The overshoot is therefore grid coarseness, not a search failure.

**Does the overshoot explain the η gap? No, on three independent checks.**
- **(a) Matching passes and the gap remains.** On ISIC 2020 (ConvNeXt-T) matching passes and η_T − η_R = 0.61 (p = 0.0005). It also passes on the REAL-2 ViT-L/16 runs for ISIC 2019 and ISIC 2020 (08), where the confirmatory η contrast is significant.
- **(b) Random η barely moves with rank.** The E50c runs match random erasure to a much cheaper target (HAM ranks 8–45 instead of 91–181), and there matching passes on every large set. Random η there is HAM 0.078 (against 0.083 at the larger rank), ISIC 2020 0.13 (against 0.10), MILK10k derm 0.25 (against 0.35) and MILK10k clin 0.31 (against 0.38).
- **(c) Direction of the bias.** Random η is roughly stable over a 4–5× change in rank, and where it moves (MILK10k) it is *higher* at the larger rank. The overshoot therefore, if anything, favours the random arm, which makes η_T − η_R conservative.

**At matched cost, LEACE also dominates on leakage.**
- On HAM and ISIC 2019, random erasure that costs as much or more leaves most of the linearly decodable age: within-test R² 0.25 against t0's 0.33 on HAM, and 0.29 against 0.32 on ISIC 2019.
- On MILK10k, rank-512 random erasure removes most of it (0.06–0.08 from 0.28–0.34), but at about 1.4× LEACE's image cost and without reaching zero.
- LEACE reaches |R²| ≤ 0.03 at a lower image cost on every large set. In the leakage–C plane (top row of the Pareto figure), the LEACE family lies below and to the left of the random arm on every large set.

**The matched-erasure direction of E2 was not measured.** E2 also asks for random erasure tuned to *match LEACE's leakage* (R² ≈ 0). That would need ranks close to d, and the arm was not run. Even random rank 512 of 768 leaves some age decodable on MILK10k (R² 0.06–0.08).

## 4. Doses and restoration (E10)

**Restoration.**
- Re-appending the erased coordinate (rest-T) restores image-only performance on every large set: C is 0.51 [−0.32, 1.29] on HAM, 0.96 [−0.36, 2.20] on ISIC 2019, −0.30 [−1.02, 0.44] on ISIC 2020, 0.98 [−2.82, 4.80] on MILK10k derm and 3.5 [−1.8, 9.0] on MILK10k clin. LEACE's cost is therefore entirely the removed one-dimensional coordinate, as expected.
- On HIBA (161 [122, 206]) and MSK 2020 (15.3 [6.1, 25.0]) restoration fails. [Z, c] carries exactly the information in S, so the failure is a fitting artefact: penalised heads are not invariant to non-orthogonal reparametrisation, and the effect is largest at n ≈ d. The training-fold age R² for rest-T (−1.5 on HIBA, −3.1 on MSK) shows the same instability. It is why §2 treats these two sets as stress tests only.

**D1 doses (t < 1).**
- In population, these are invertible maps of S (04). The measured costs (HAM t = 0.5: 2.2 [0.8, 3.5]; t = 0.75: 7.4) are therefore costs of the restricted, penalised linear head family, not information loss. Approximately: shrinking a direction by (1 − t) raises the L2 penalty for using it by (1 − t)⁻², so the penalised head uses it less.
- Age also stays linearly decodable after D1 doses (training-fold R² 0.26–0.38 at t ≤ 0.75). D1 doses are therefore not reported as erasure, and C, L and η along them are not interpreted.

**D2 information doses.** These destroy information. η by dose t:

| Set | t = 1 | t = 0.75 | t = 0.5 | t = 0.25 |
|---|---|---|---|---|
| HAM10000 | 0.82 | 0.76 | 0.61 | 0.45 |
| MILK10k derm | 0.59 | 0.49 | 0.48 | 0.39 |
| MILK10k clin | 0.66 | 0.54 | 0.49 | 0.42 |
| ISIC 2020 | 0.69 | 0.62 | 0.63 | 0.57 |
| ISIC 2019 | 0.92 | 0.86 | 0.80 | 0.92 |

- On HAM and MILK10k, partial information removal is *less* substitutable per unit of cost than full removal. On HAM, L stays roughly constant (2.5–2.7) while C falls from 14.3 to 4.5.
- On the two ISIC sets η is flat within its CIs.
- No prediction was registered for this pattern. It is reported descriptively and was not followed up.

**Ledoit–Wolf LEACE** matches LEACE on every large set: |ΔC| ≤ 0.2 × 10⁻³ (≤ 0.3% of C) and |Δη| ≤ 0.02. It differs on HIBA and MSK (η 0.47 against 0.37, and 0.55 against 0.34), the n ≈ d sets.

**Controls.**
- Same-rank random (SR) has no detectable cost on any set.
- Shams (LEACE on permuted age) cost little: 0.10 [0.00, 0.20] on ISIC 2020, 0.34 [0.17, 0.52] on ISIC 2019, 1.9 [0.4, 3.7] on HIBA, and CIs including 0 elsewhere. That is 2–5% of LEACE's cost on the large sets, an upper bound on the non-age-specific part of LEACE's cost.

## 5. Missing metadata at deployment (E11)

**Setup.** Age is missing completely at random (MCAR) at test with probability π. The deployment falls back to the image-only head when age is missing and uses the fused head otherwise. The expected added risk of erasing, relative to the same deployment on S, is then exactly

  cost(π) = π·C + (1 − π)·L = C·[1 − (1 − π)·η].

η is therefore the share of the image-only cost recovered *per unit of metadata availability*. E11 needs no new estimation beyond C and L.

Point estimates, × 10⁻³ nats:

| Set | Arm | π = 0 (all age present) | 0.25 | 0.5 | 0.75 | π = 1 (no age) |
|---|---|---|---|---|---|---|
| HAM10000 | LEACE | 2.9 | 5.7 | 8.5 | 11.3 | 14.1 |
| | random | 23.4 | 23.9 | 24.5 | 25.0 | 25.5 |
| ISIC 2019 | LEACE | 0.25 | 2.1 | 3.9 | 5.7 | 7.5 |
| | random | 11.2 | 11.2 | 11.3 | 11.4 | 11.5 |
| ISIC 2020 | LEACE | 0.49 | 0.79 | 1.09 | 1.40 | 1.70 |
| MILK10k derm | LEACE | 37.8 | 52.0 | 66.1 | 80.3 | 94.4 |
| | random | 86.0 | 97.8 | 109.6 | 121.3 | 133.1 |

- The endpoint CIs are those of L (π = 0) and C (π = 1) in §2.
- Interior CIs come from the grouped paired bootstrap of π·c_i + (1 − π)·l_i (B = 2000, seed 601), computed 2026-11-02 00:29 IST (scripts/real_e11.py; results/real/REAL-1_E11_missing_age.csv). The endpoints reproduce §2 exactly.

95% CIs, × 10⁻³ nats:

| Set | Arm | π = 0 | 0.25 | 0.5 | 0.75 | π = 1 |
|---|---|---|---|---|---|---|
| HAM10000 | LEACE | [−0.1, 6.0] | [2.7, 8.8] | [5.4, 11.6] | [7.9, 14.5] | [10.2, 17.8] |
| | random | [20.3, 26.4] | [20.8, 27.0] | [21.3, 27.5] | [21.8, 28.1] | [22.3, 28.7] |
| ISIC 2019 | LEACE | [−2.7, 3.2] | [−0.8, 4.8] | [1.0, 6.6] | [2.6, 8.6] | [4.2, 10.7] |
| | random | [9.1, 13.3] | [9.2, 13.4] | [9.2, 13.5] | [9.3, 13.6] | [9.4, 13.7] |
| ISIC 2020 | LEACE | [−0.2, 1.2] | [0.1, 1.5] | [0.4, 1.8] | [0.7, 2.1] | [0.9, 2.5] |
| | random | [1.4, 2.3] | [1.4, 2.4] | [1.5, 2.4] | [1.5, 2.5] | [1.6, 2.6] |
| MILK10k derm | LEACE | [29.0, 46.5] | [43.2, 60.8] | [57.2, 75.2] | [70.5, 89.9] | [83.8, 104.8] |
| | random | [76.8, 95.3] | [88.5, 107.3] | [99.9, 119.4] | [111.4, 131.7] | [122.4, 144.2] |
| MILK10k clin | LEACE | [24.1, 39.6] | [39.3, 54.5] | [53.7, 70.1] | [67.7, 86.0] | [81.5, 102.1] |
| | random | [68.7, 87.8] | [80.6, 100.0] | [92.4, 112.6] | [103.9, 125.2] | [115.3, 138.2] |

(ISIC 2020 random point estimates: 1.85, 1.90, 1.96, 2.01, 2.07. MILK10k clin point estimates: LEACE 31.7, 46.8, 61.8, 76.8, 91.8; random 78.2, 90.4, 102.5, 114.6, 126.7.)

**Reading.**
- LEACE's added risk rises linearly from L to C as age goes missing. The random arm's is nearly flat, because η_R ≈ 0.
- On HAM10000 and ISIC 2019, LEACE's cost becomes detectable once at least a quarter to a half of test cases lack age (lower limit > 0 from π = 0.25 on HAM and π = 0.5 on ISIC 2019).
- The two arms' levels are not cost-matched (the random arm overshoots, §3), so the comparison is the *slope*, not the gap.

**Caveats.**
- MCAR is a strong assumption: in these sets, missing age is concentrated in particular sources.
- A single fused model trained on complete data would also need its own handling of missing inputs.
- With MAR or MNAR missingness the formula does not hold, and the cost has to be estimated per missingness pattern.

## 6. Class-conditional erasure (E50c): the simplest task-aware linear eraser

E50c removes age's *within-class* linear covariance. It uses training labels and keeps the between-class age information.

| Set | C | L | η | within-test age R² | vs unconditional LEACE |
|---|---|---|---|---|---|
| HAM10000 | 5.1 [2.9, 7.3] | 0.62 | 0.88 [0.57, 1.38] | 0.005 | C 14.1, L 2.9 |
| ISIC 2020 | 1.45 | 0.40 | 0.72 [0.49, 1.26] | −0.03 | C 1.70 |
| ISIC 2019 | 1.72 [0.26, 3.15] | −1.0 | 1.59 [0.82, 5.84] | 0.02 | C 7.5 |
| MILK10k derm | 33.1 [25.5, 40.5] | 11.7 | 0.65 [0.53, 0.80] | 0.069 | C 94.4, L 37.8 |
| MILK10k clin | 38.3 [31.2, 45.4] | 13.0 | 0.66 [0.56, 0.79] | 0.069 | C 91.8, L 31.7 |

**What the comparison shows.**
- Using labels inside the eraser cuts the image-only cost by 58–77% on HAM, ISIC 2019 and MILK10k (15% on ISIC 2020), and the fused cost by 59–79% where L is measurable (HAM, MILK10k).
- It does *not* raise η beyond the unconditional eraser's CIs.
- It leaves some age decodable (R² 0.07 on MILK10k), since it does not target the between-class part.
- E50c and LEACE are therefore *not* at matched leakage. The comparison shows a different operating point, not a dominating method. This is the real-data counterpart of 07 §3: task awareness moves the operating point but does not improve substitution efficiency.

**Label optimism (directive Phase 5 item).** The rank-matching target is the eraser's inner-CV, training-fold image log-loss, with the eraser fitted on the whole training fold. A label-using eraser therefore sees the inner-validation labels. Mean over 5 folds, × 10⁻³ nats:

| Set | Unconditional (attribute-only): inner / outer C | Class-conditional (uses labels): inner / outer C |
|---|---|---|
| HAM10000 | 19.7 / 14.1 | 4.5 / 5.1 |
| ISIC 2019 | 8.9 / 7.5 | 1.2 / 1.7 |
| ISIC 2020 | 1.9 / 1.7 | 1.3 / 1.5 |
| MILK10k derm | 107.9 / 94.4 | 21.7 / 33.1 |
| MILK10k clin | 103.1 / 91.8 | 20.3 / 38.3 |
| MSK 2020 | 25.2 / 30.6 | **−4.5 / 40.5** |
| HIBA | 309.8 / 320.5 | **−54.5 / 284.8** |

- The attribute-only eraser's training-fold estimate is within about 40% of its outer-test cost everywhere. On the large sets it is the larger of the two, the conservative direction, partly because inner-CV heads train on fewer rows.
- The label-using eraser's training-fold estimate is optimistic. The optimism is mild on the large sets (outer/inner ratio 1.1–1.4) and 1.5–1.9× on MILK10k. On the n ≈ d sets it reverses sign: in-sample the eraser appears to *improve* diagnosis, while on the outer test it costs 40–285 × 10⁻³ nats. That makes its matched random rank 1 and its "matched" comparison meaningless (T − matched image log-loss +283 on HIBA, +40 on MSK).
- The E50c outer-test estimates themselves remain valid, because the eraser never saw the test fold.
- **Methodological consequence.** Any task-aware eraser must be fitted in a nested way, with its own inner split, before it is scored. Using the same rows to train a label-aware transformation and to evaluate it gives a sign-reversed answer at n ≈ d. That rules out the "fit MATE on the training set, report cross-validated cost" protocol the directive cautioned against.

## 7. Site erasure (E51; site column of E5)

| Set | C | η_T | η_R | η_T − η_R [CI] (p) |
|---|---|---|---|---|
| HAM10000 | 32.5 [27.3, 37.9] | 0.21 [0.10, 0.32] | 0.05 | 0.16 [0.07, 0.25] (0.0005) |
| MILK10k derm | 14.2 [8.8, 19.3] | 0.44 [0.19, 0.84] | 0.13 | 0.30 [0.13, 0.59] (0.001) |
| HIBA | 208 | 0.14 | 0.17 | −0.03 (0.69) |
| PAD-UFES-20 | 49 | 0.28 | 0.40 | −0.11 (0.72) |

**Site behaves differently from age.**
- On HAM10000, LEACE on the site indicator costs more than age erasure (C 32.5 against 14.1), and only 21% of that cost is recoverable from the recorded site.
- Most of what the site directions carry is diagnostic signal that the site label does not explain. This is consistent with site being associated with acquisition and with case mix, rather than a single label-relevant nuisance.
- On HIBA and PAD-UFES-20 (n ≈ d) the site shams are not inert (11.8 and 27.6 × 10⁻³ nats), and the contrast is uninterpretable.
- The leakage column in REAL-1_three_objectives.csv for site arms is *age* leakage. Site leakage is not in this table.
- *Correction, 2026-11-02:* the earlier text said REAL-7 adds site leakage. It does not: REAL-7's audit probes age only.
  - Linear site leakage was already in the E51 JSONs (probe_site), and 08 §6.2 reports it. Site erasure takes the held-out site-probe log-loss gain from 0.622 to 0.000 on HAM10000, and from 0.472 to 0.000 on MILK10k.
  - No non-linear site probe was run.

## 8. Stability across fold assignments (ConvNeXt-T; E50b)

| Set | η_T at fold assignments 0 / 1 / 2 | η_R at fold assignments 0 / 1 / 2 | η_T − η_R p (all) |
|---|---|---|---|
| ISIC 2020 | 0.71 / 0.71 / 0.68 | 0.10 / 0.10 / 0.10 | 0.0005 |
| HAM10000 | 0.79 / 0.76 / 0.75 | 0.08 / 0.09 / 0.08 | 0.0005 |
| ISIC 2019 | 0.97 / 1.06 / 1.02 | 0.03 / 0.04 / 0.06 | 0.0005 |

- Re-partitioning moves η by at most 0.09, well inside any single run's CI.
- The fold-assignment spread is small compared with the bootstrap CI width (about 0.4–0.9 for η_T). The bootstrap, not partitioning, dominates the uncertainty.

## 9. Summary for the research question

1. **The ceiling M ≤ C is never violated on real data**, and the decomposition C = M + L is informative.
2. **On large dermoscopy sets, the diagnostic cost of erasing age is mostly substitutable** when age is available as metadata (η 0.71–0.97). Random erasure at equal or larger cost is not (η_R ≤ 0.10). The gap narrows where age predicts the label more strongly (MILK10k), as the synthetic regimes predict.
3. **The fused cost L is not distinguishable from 0** on HAM10000, ISIC 2019 and ISIC 2020 at ConvNeXt-T. Equivalence was not tested, so "erasure is free given age" is *not* established. On MILK10k, L is clearly positive.
4. **Small sets (n ≈ 1.5d) fail the pipeline checks** (restoration, shams, matching). They are reported as a stress test only.
5. **Task-aware linear erasure (E50c) gives a cheaper operating point at slightly higher leakage, with no gain in η.** Its in-sample cost estimates are optimistic and, at n ≈ d, sign-reversed. Nested fitting is a requirement for any MATE-like method, not a refinement.
6. **Site is a different object from age.** Its erasure is mostly irreplaceable (η 0.21–0.44).

**Open items:**
- E11 interior CIs.
- The matched-leakage random arm (not run).
- Equivalence testing for L.
- REAL-4 (positive control; distributional leakage) shows whether "linear guardedness" here leaves non-linear age information. 05 (SYN-3b) suggests it does: |A| R² 0.895 after LEACE.
