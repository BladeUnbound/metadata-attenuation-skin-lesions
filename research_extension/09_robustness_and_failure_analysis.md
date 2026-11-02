# 09 — Robustness and failure analysis

*Phase 6/7. Drafted 2026-11-01; completed 2026-11-02 from the REAL-4 to REAL-8 runs (08 §3–7). Sources: 05, 06 and 08 (sections cited), EXECUTION_LOG.md.*

The main claim this document stress-tests:

> **(C1)** On large dermoscopy datasets, the diagnostic cost of LEACE age erasure is mostly substitutable by recorded age (η_T ≈ 0.7–1.0), whereas performance-matched random erasure is not (η_R ≤ 0.10).

## 1. Robustness of C1

| Perturbation | Result | Where |
|---|---|---|
| Fold assignment (3 partitions) | η_T moves by ≤ 0.09; η_R by ≤ 0.03; all p = 0.0005 | 06 §8 |
| Representation (5 rungs, HSV to ViT-L/16) | η_T 0.63–1.05 at every deep rung; η_R ≤ 0.10; gap significant at each | 08 §2 |
| Untouched representation, confirmatory | 3/3 Holm rejections (ViT-L/16) | 08 §1 |
| Whitening estimator (Ledoit–Wolf) | \|Δη\| ≤ 0.02 on every large set | 06 §4 |
| Matched-random overshoot | random η is roughly rank-invariant (HAM: 0.078 at ranks 8–45, 0.083 at 91–181); the gap holds where matching passes | 06 §3 |
| Sham (permuted-age LEACE) | costs ≤ 5% of LEACE's C; no age-gain change | 06 §4 |
| Positive control (oracle residual, population η = 1) | recovered within CI on all 4 large sets: HAM 0.99 [0.88, 1.12], ISIC 2019 1.09 [0.86, 1.47], ISIC 2020 0.88 [0.73, 1.08], MILK10k 1.01 [0.96, 1.07]; oracle L CIs include 0. Fails on HIBA (0.88 [0.82, 0.94], L 30.4 > 0) | 08 §3.1 |
| Subsampling to n_train/d = 1.25–10 (ISIC 2019) | η_T − η_R > 0 (p ≤ 0.009) in 11/12 raw configurations; η_T biased towards η_R at q ≤ 2 (0.47–0.69 vs 0.97) | 08 §5 |
| Label conditioning (E50c) | η_T unchanged within CIs (0.65–0.88; ISIC 2019 1.59, wide) | 06 §6 |
| Restoration (re-append the erased coordinate) | C ≈ 0 on every large set: the cost is exactly the removed coordinate | 06 §4 |

**Where C1 weakens.**
- **MILK10k** (age–label AUROC 0.715): η_T 0.60–0.65 with η_R 0.35–0.38, so the gap shrinks to 0.25. The positive control (1.01) shows this lower η is a property of the data, not of the estimator.
- **Age-dominated labels** (HIBA, MSK; SYN-1 finding 3): random erasure becomes substitutable too, and the design loses its contrast.
- **Small sets:** these fail the pipeline checks (§2).
- **Irrelevant age** (PAD-UFES-20): η is undefined.

**Where C1's scope is narrower than its wording suggests (REAL-4, -7, -8).**
- **Linear only.** Gradient-boosted probes recover 55–88% of the within-class age R² after LEACE (08 §3.2). C1 is about the linear age direction.
- **Not portable.** An eraser fitted on one dataset removes 4–15% of another's linear age information and has no detectable cost there (08 §7).
- **Age, not metadata in general.** Site erasure is mostly non-substitutable (η 0.21 on HAM), and joint removal of age, sex and site is 42% substitutable on HAM (08 §6.3).

C1 is therefore a claim about the linear age direction, fitted on the target distribution, on large datasets with modest age–label dependence (AUROC ≈ 0.6–0.7). That is the regime the synthetic study identifies as informative.

## 2. Failures of the measurement pipeline, and what they imply

| Failure | Evidence | Consequence |
|---|---|---|
| n/d ≈ 1.5: the eraser removes the wrong amount | SYN-2: direction cosine 0.79–0.83 and out-of-sample leakage 0.34–0.37 at n/d = 1.25; real coordinate SD ratio 2–3 (iteration 6) | small sets are stress tests only. REAL-6: subsampling ISIC 2019 reproduces the SD ratio (1.8–2.0 at n/d = 1.25) and pulls η_T to 0.47–0.69; PCA fixes the SD ratio but not η (08 §5) |
| n/d ≈ 1.5: probes fail | SYN-1 finding 6: ridge/MLP R² negative where exact leakage is 0.19. Real PAD-UFES-20: HGB decile gain negative even at t0 (−0.05) | "R² ≈ 0 after erasure" on a small set certifies nothing |
| n/d ≈ 1.5: restoration fails | HIBA rest-T C = 161, MSK 15 (06 §4) | penalised heads are not reparametrisation-invariant at n ≈ d; C, L and η are unreliable there |
| n/d ≈ 1.5: shams are not inert | site shams on HIBA and PAD-UFES-20: 11.8 and 27.6 (06 §7) | same |
| n/d ≈ 1.5: the positive control fails | HIBA oracle η 0.88 [0.82, 0.94], L 30.4 [14.0, 46.0], where population L = 0 (08 §3.1) | the two-stage estimator itself is biased at n ≈ d; registered check, independent of the earlier rule |
| PCA-then-LEACE | per-fold image-only cost 131 and −11 in one configuration; η across seeds 0.12–0.80 (08 §5) | PCA plus the heads' standardisation whitens the retained components (likely mechanism, not tested); not a remedy |
| Adversarial erasure judged by its own adversary | training adversary near chance (decile CE up to 2.09 of 2.30) while held-out ridge R² is 0.22–0.26 (08 §4) | adversary loss is not a leakage certificate |
| Label-using eraser scored on its own training rows | class-conditional eraser: inner −54.5 vs outer +285 (HIBA); −4.5 vs +40 (MSK) (06 §6) | any task-aware eraser must be fitted in a nested way. This is the strongest methodological lesson for MATE-type methods |
| D1 partial doses | invertible maps that still "cost" (HAM t = 0.75: 7.4) | a restricted-family artefact; D1 doses are not erasure and are not interpreted |
| Matched-random grid | the √2 refinement is coarse at large rank (362 → 512); random overshoots on the test fold | η comparison robust (06 §3); M contrasts at "matched cost" are conservative |
| Registered SYN-3 leakage metric | signed-age R² cannot see \|A\| information | SYN-3b amendment, labelled post hoc |

## 3. Leakage beyond the linear probe (REAL-4)

**Prediction (05 SYN-3b).** LEACE achieves linear guardedness, but non-linear probes still decode age.

**HAM10000 (ConvNeXt-T)**, HGB probes fitted on outer-training rows, scored on the outer test:

| Arm | ridge R² (within-test) | HGB within-class R² | HGB \|A − median\| R² | HGB decile gain (fraction of log 10) |
|---|---|---|---|---|
| t0 | 0.325 | 0.337 | 0.176 | 0.079 |
| LEACE | −0.004 | 0.253 | 0.161 | 0.045 |
| matched random (j = 0) | 0.248 | 0.275 | 0.145 | 0.059 |
| sham (j = 0) | 0.325 | 0.339 | 0.172 | 0.079 |

**What the HAM audit shows.**
- **(i) LEACE's effect is narrowly linear.** It removes the linearly decodable age completely, but a gradient-boosted probe still recovers 75% of the within-class age R² and 57% of the decile gain.
- **(ii) Against non-linear probes, LEACE and matched random erasure remove similar amounts.** Within-class R² is 0.253 against 0.275; for |A| R², random removes slightly more (0.145 against 0.161).
- **(iii) Consequence for interpretation.** η measures how substitutable the information removed *with the linear age direction* is. It does not describe a representation from which age has been removed. Claims of "age-free features" would be false here.

**Other sets (08 §3.2).** The HAM pattern repeats on ISIC 2019, ISIC 2020 and MILK10k derm:
- ridge R² after LEACE is −0.034 to 0.015;
- the share of t0 HGB leakage that remains is 55–88% (within-class), 66–91% (|A|) and 44–63% (decile gain);
- matched random removes similar amounts, except on MILK10k, where the overshooting random arm (rank 512) removes more;
- on HIBA the HGB probes are negative even at t0, so the non-linear audit is uninformative there.

## 4. Non-linear erasers on real data (REAL-5)
Registered prediction (SYN-3/3c): the non-linear erasers do not exceed LEACE's η, and leave linear leakage above LEACE's. **Both hold** (08 §4):
- η is 0.06–0.17 against 0.79, and every difference has a CI excluding 0;
- ridge R² is 0.22–0.26 against −0.004;
- non-linear leakage is about the same as LEACE's.

Attribute-only adversarial erasure behaves like random distortion on the substitutability axis.

## 5. n/d scaling and PCA (REAL-6)
The registered prediction (SYN-2) gets mixed support (08 §5):
- **Matching instability:** confirmed. Random costs 6–10× LEACE at n/d = 1.25.
- **η stays near the full-data value:** refuted at n/d ≤ 2 (0.47–0.69 vs 0.97).
- **PCA narrows the instability:** not supported.

The η contrast with random survives down to n/d = 1.25, with a conservative bias.

**Deviation:** within-test leakage was registered but not computed.

## 6. Attribute specificity and joint removal (REAL-7)
08 §6:
- **Specific for age and site.** Diagonal gains are 9.1/9.5 (age) and 4.3/3.8 (site); off-diagonals are ≤ 1.5 in magnitude.
- **No power for sex.** Erasing sex is cheap (C 2.7 and 13.0), and recorded sex recovers none of it detectably.
- **Overlapping attribute subspaces.** Site erasure removes 29% of linear age R² on HAM.
- **Joint removal is sub-additive** (18% below the sum of single costs), and only 42% (HAM) / 62% (MILK10k) substitutable by all three attributes.
- **No sex-decodability probe** was computed.

## 7. Cross-dataset transfer (REAL-8)
08 §7:
- Erasers do not transfer. A source-fitted eraser removes 4–15% of the target's linear age R², and its cost on the target is not detectable.
- Age-covariance directions have cosine 0.32–0.57 between datasets.
- **Deviation:** the source eraser was fitted on all source rows, not on outer-training rows. No labels are used and the datasets differ; patient overlap across datasets cannot be checked.

## 8. Threats to validity not addressed by any experiment

1. **Recorded age is coarsened.** On every ISIC-family set (HAM10000, ISIC 2019, ISIC 2020, MILK10k, HIBA, MSK 2020), age is recorded in 5-year bins (18 levels). PAD-UFES-20 records exact ages.
   - η measures substitutability by the *recorded* age. Image age information finer than the bins cannot be substituted, which biases η downward relative to exact age.
   - The within-bin variance (about 2 years²) is around 1% of the age variance, so the bias is likely small, but it is not quantified.
2. **Age missingness.** 0.2–2.5% of rows on the large sets lack age (ISIC 2019: 2.5%). The eraser is fitted on rows with age, and the second stage handles missingness as in the parent paper. Missingness is not MCAR in general (it clusters by source), and E11 (06 §5) assumes MCAR.
3. **Predictive, not causal.** η is a statement about conditional predictive information, I(Y; S | Z) − I(Y; S | Z, A) (04 Identity 3). It does not identify age as a cause of the image features or of the diagnosis. Age is entangled with site, sun exposure, lesion type and referral pathway, so "age-related image information" is not "biological ageing".
4. **Label quality.** Labels mix histopathology, follow-up and consensus across sources. Label noise correlated with age (for example, histopathology confirmation more frequent in older patients) would enter C and M identically and is not separated.
5. **Frozen encoders and linear heads only.** All conclusions are V-information statements for penalised linear heads (10 §1). Fine-tuned encoders (E9) were not run; cost estimate in 11.
6. **No external confirmation set** (10 §3).

## 9. Computational failures (complete list; EXECUTION_LOG.md)

| Phase | Failure | Fix | Results affected? |
|---|---|---|---|
| P3 | SYN-2 worker SIGBUS (8 workers), then SIGSEGV (4 workers) | n_jobs = 1 | no |
| P3 | SYN-2 "SVD did not converge" (rank-deficient view covariances, cond 10¹⁶–10¹⁸) | eigendecomposition-based pseudo-inverse; notebook and SYN-1 re-run with identical values | no |
| P3 | SYN-3b flag not reaching loky workers | passed explicitly | no (caught before use) |
| P3 | SYN-3c duplicate config key | renamed | no |
| P3 | notebook: positional MLPRegressor argument | keyword | no |
| P6 | real_new_analysis.py / real_e11.py import path under `python -I` | sys.path insert, as in real_new.py | no (analysis scripts only) |
| P6 | real_new_analysis.py: NAE arm names sliced with [:-4] instead of 3 characters (KeyError) | slice by len("\|l0"); re-run | no (analysis only; caught by the error) |

REAL-4 to REAL-8: all 15 queued tasks exited 0 (2026-11-01 21:25 → 2026-11-02 00:28 IST). The logs contain one PyTorch UserWarning (nonlin.py:81, scalar conversion when logging the adversary loss), which has no effect on results.
