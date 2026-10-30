# Iteration 6: final report — mechanism identification for "Do Patient Metadata Still Help Strong Image Models?"

*Final, 2026-10-30 03:19 IST. Drafting began 2026-10-29 (file timestamp 23:23 IST); all experiments finished 2026-10-30 03:18 IST. Every number comes from results/E50–E55 JSON records via experiments/e50_report.py (iteration6/E50_tables.md) and experiments/iteration6_summary.py (iteration6/ITER6_TABLES.md). Pre-registrations: iteration6/REGISTRY.md, written 2026-10-29 13:49 IST before any E50–E55 code existed; amendments 1–3 at 14:55, 14:58 and 18:44 IST. Run log and deviations: iteration6/STATUS.md.*

Units: added value Δ in 10⁻³ nats of per-observation log-loss (positive = metadata help); brackets are paired grouped-bootstrap 95% CIs (B = 2000); p is one-sided.

## 1. Summary for decision-making

- **The redundancy mechanism for age passed its pre-registered confirmatory test on one untouched dataset and replicated on every large dataset. Under this design it cannot be tested on small, strongly age-associated datasets.**
  - **ISIC 2020** (confirmatory, never used in an erasure experiment):
    - Removing the linear trace of age from ConvNeXt-T features raises age's added value more than removing random subspaces of equal performance cost: +0.99 [+0.69, +1.29] ×10⁻³ nats, Holm p = 0.002.
    - It does the same against shams: +1.21 [+0.86, +1.55].
    - The effect is stable across three fold assignments (+0.85 to +1.08).
    - It rises monotonically with an information dose.
    - It holds under full-feature fusion (+1.49 [+0.82, +2.20]).
  - **HAM10000 and ISIC 2019** (replication) pass every check above.
  - **MILK10k** passes the main contrast, the sham and full-feature fusion. Its dose curve is significant only at t ≥ 0.75, its matched control hits the rank cap, and its fold assignments were not repeated.
  - All six earlier E45b claims survive the stricter, training-only control (§3.6).
  - **HIBA and MSK 2020 (age) and HIBA (site)** fail the registered test (p = 1.0). With about 1,100–1,200 images against 768 features, the closed-form eraser overfits: its coordinate is 1.8–3.1× more variable on test than on training rows. On HIBA and MSK 2020 the label also depends strongly on age, so erasing age removes diagnostic signal (§4).
- **The effect is attribute-specific where it can be tested.** On HAM10000 and MILK10k derm:
  - erasing age restores age's value but not site's;
  - erasing site restores site's value but not age's;
  - all four registered selectivity checks pass.
  - These are replication or exploratory sets (§5).
- **What is removed is the linear trace only.** After erasure an MLP still decodes age (R² 0.05–0.23) and site on the large sets. Targeted erasure lowers nonlinear decodability no more than random erasure (§6). The manuscript must say "linearly decodable age information".
- **The mechanism is quantitatively coherent on the large sets.**
  - Per unit of image log-loss lost, targeted erasure moves 0.71–0.97 of it into age's added value; matched random erasure moves 0.03–0.10.
  - A cross-fitted image-derived age estimate recovers exactly the rise.
  - Re-appending the erased coordinate restores the image model fully and returns 74–93% of the rise.
- **Class-conditional erasure (exploratory).** It removes only the age trace not explained by the label. It gives the same positive, specific, well-matched effect on all five large sets at a smaller image cost (§4.1). Part of what marginal erasure removes is label-induced covariance.
- **Estimator robustness.**
  - The paper's main result (H1) holds under score, full-feature and boosted fusion on all 8 sets.
  - On 5 of 8 sets, whether metadata add anything at ConvNeXt-T depends on the estimator.
  - Boosted fusion finds +4 to +8×10⁻³ nats more on 5 sets, so the paper's estimator is conservative (§7).
- **Dataset shift.** With rules fitted on source datasets only, nothing beat "always use metadata" on 7 held-out datasets. The existing collection rule equalled "always use". Metadata hurt on PAD-UFES-20 and no rule anticipated it (§9).
- **Fine-tuning seeds (E54).** Two extra training seeds on HIBA and PAD-UFES-20 (ResNet-18 and ConvNeXt-T) reproduce E14b. In all 12 seed × backbone × dataset runs, fine-tuned scores leave significantly less metadata value than frozen ones. The seed range of added value is at most 6.7×10⁻³ nats (§8).
- **Process.** Every registered experiment ran. Two jobs failed on code errors (E53 at start, E55 after its evaluations) and were fixed and rerun. E50b's scope was reduced for compute. All three are logged (§10).

## 2. Audit before the new round

See AUDIT_PLAN.md (A1–A11). The audit items that changed the design:
- **A1.** E45 chose the matched random rank with outer-test labels. E50 chooses it per outer fold from training-fold inner CV only.
- **A2.** E45's dose (D1, shrinking the erased coordinate) is invertible, so it preserves information for t < 1. E50 adds an information dose (D2: the erased coordinate is mixed with independent noise).
- **A3–A4.** E45 had no sham and no restoration. E50 adds 5 shams (LEACE fitted to permuted age), re-append restoration, a restoration fraction and an image-derived age block.
- **A9–A11.** Standalone metadata value (E30 lower bound ≥ 2×10⁻³) made age eligible in six sets and site in two. Three sets never touched by an erasure experiment (HIBA, MSK 2020, ISIC 2020) formed the confirmatory family. Seven datasets are independent at most (the two MILK10k sets share lesions).

## 3. Priority 1: dose–response, controls and restoration for age (E50a)

ConvNeXt-T, fold assignment 0, strict cross-fitting (E40), every eraser and probe fitted on outer-training rows only.

### 3.1 Registered confirmatory family (Holm, m = 4)

| Test | T − matched | p | Holm p | Reject |
|---|---|---|---|---|
| ISIC 2020, age | **+0.99 [+0.69, +1.29]** | 0.0005 | 0.002 | **yes** |
| HIBA, age | −116.56 [−160.01, −78.81] | 1.0 | 1.0 | no |
| MSK 2020, age | −23.84 [−30.07, −17.86] | 1.0 | 1.0 | no |
| HIBA, site (E51) | −54.11 [−81.37, −27.49] | 1.0 | 1.0 | no |

Gate G1 = **PROCEED**: ISIC 2020 passes Holm's first step (p ≤ 0.0125), and its targeted − sham CI is above 0 (+1.21 [+0.86, +1.55]).

### 3.2 Every set

| Set | Role | n | Matched rank per fold | Image LL t0 → T | **T − matched** | T − sham | Matching check (image LL T − matched) |
|---|---|---|---|---|---|---|---|
| ISIC 2020 | confirmatory | 33,126 | 91, 128, 45, 128, 32 | 0.0713 → 0.0730 | **+0.99 [+0.69, +1.29]** | +1.21 [+0.86, +1.55] | −0.37 [−1.19, +0.38] (passes) |
| HIBA | confirmatory | 1,224 | 512 ×5 (cap) | 0.3392 → 0.6597 | −116.56 [−160.01, −78.81] | +116.93 [+49.55, +173.49] | −125.62 (matched overshoots) |
| MSK 2020 | confirmatory | 1,174 | 256–512 | 0.3444 → 0.3751 | −23.84 [−30.07, −17.86] | +10.25 [+1.81, +18.05] | −38.39 (matched overshoots) |
| HAM10000 | replication | 10,015 | 91–181 | 0.2321 → 0.2462 | **+9.05 [+6.48, +11.70]** | +11.16 [+8.17, +14.21] | −11.43 [−15.76, −7.36] (matched more damaging: conservative) |
| ISIC 2019 non-HAM | replication | 15,316 | 23–91 | 0.3929 → 0.4004 | **+6.92 [+4.76, +9.14]** | +7.23 [+4.79, +9.68] | −3.99 [−7.38, −0.40] (conservative) |
| MILK10k derm | replication | 5,117 | 512 ×5 (cap) | 0.4236 → 0.5180 | **+9.53 [+4.50, +14.20]** | +56.54 [+48.13, +64.57] | −38.67 (conservative) |
| PAD-UFES-20 cc | replication | 1,089 | 1–11 | 0.5202 → 0.5229 | +0.22 [−2.63, +3.11] | +0.33 [−2.48, +3.23] | −1.21 [−6.53, +4.01] |
| MILK10k clin | correlated | 5,117 | 512 ×5 (cap) | 0.4291 → 0.5209 | +11.63 [+6.67, +16.59] | +60.02 [+51.56, +68.31] | −34.88 (conservative) |

- **Shams are near-inert everywhere.** T − sham is within 0.25 of T − t0 on every set. Sham image log-loss differs from t0 by at most 1.9×10⁻³ nats (HIBA +1.89 [+0.19, +3.72]; elsewhere −0.87 to +0.34).
- **Same-rank random controls** are equally inert. T − same-rank is within 0.17 of T − t0.

### 3.3 Dose–response (S3)

Under the information dose D2, targeted − matched increases with t and its CI lies above 0 at every dose on all three large sets:

| Set | t = 0.25 | 0.5 | 0.75 | 1.0 |
|---|---|---|---|---|
| ISIC 2020 | +0.23 [+0.11, +0.35] | +0.50 [+0.35, +0.65] | +0.80 [+0.57, +1.02] | +0.92 [+0.64, +1.19] |
| HAM10000 | +1.56 [+0.90, +2.20] | +3.42 [+2.20, +4.66] | +6.08 [+4.28, +7.90] | +9.02 [+6.59, +11.39] |
| ISIC 2019 | +1.32 [+0.79, +1.85] | +2.92 [+1.74, +4.07] | +4.75 [+3.10, +6.39] | +6.91 [+4.84, +9.01] |

- MILK10k derm and clin rise too, but their CIs include 0 until t = 0.75–1.
- PAD-UFES-20 is flat.
- HIBA and MSK 2020 decline (matched overshoots at every dose).

### 3.4 Restoration and the image-derived age (S6)

On the three large sets, where the eraser's test/train SD ratio is 0.99–1.01, inside amendment 1's [0.8, 1.25] window:

| Set | Rise in Δ_age (T − t0) | Re-append [S_erased, c]: Δ − Δt0 | Re-append: image LL − t0 | RF targeted | RF matched | Δ_â(T) − rise |
|---|---|---|---|---|---|---|
| ISIC 2020 | +1.21 | +0.15 [+0.03, +0.27] | −0.29 [−1.00, +0.44] | 0.71 [0.50, 1.16] | 0.10 [0.06, 0.15] | +0.25 [−0.28, +0.77] |
| HAM10000 | +11.18 | +0.82 [+0.50, +1.13] | +0.51 [−0.30, +1.27] | 0.79 [0.62, 1.02] | 0.08 [0.06, 0.11] | −0.04 [−3.37, +3.16] |
| ISIC 2019 | +7.23 | +1.91 [+1.08, +2.75] | +0.96 [−0.33, +2.37] | 0.97 [0.64, 1.57] | 0.03 [−0.01, 0.06] | +0.07 [−2.79, +3.33] |

- **Re-appending the erased coordinate** fully restores the image model (log-loss CI includes 0). It returns 74–93% of the rise in age's value, but not all of it (CIs exclude 0). Registered prediction (i) therefore holds for image log-loss and fails, narrowly, for Δ.
- **Restoration fraction.** Per unit of image log-loss lost, targeted erasure moves 0.71–0.97 of it into age's added value. Matched random erasure moves 0.03–0.10.
- **Image-derived age** â (cross-fitted ridge prediction of age from the original features) recovers the rise exactly: the S6-iii prediction under redundancy holds within the CI on all three sets.
- On MILK10k the same quantity exceeds the rise: +36.0 [+27.0, +45.4] for derm and +25.3 for clin. So the erased direction there carries label information beyond age (§4).

### 3.5 Residual decodability (S5; amendment 2)

- The training-fold ridge probe returns R² ≈ 0 after every exact LEACE eraser, by construction (amendment 2).
- The within-test-fold probe on ISIC 2020, HAM10000 and ISIC 2019 falls from 0.23–0.33 to between −0.03 and 0.01 after targeted erasure. After matched random erasure it stays at 0.19–0.29.
- On MILK10k, targeted erasure brings it from 0.28–0.34 to 0.01–0.02. The 512-dimensional matched control also lowers it, to 0.06–0.08.
- Nonlinear decodability is in §6.

### 3.6 Continuity with E45b

ConvNeXt-T:

| Set | E45b | E50 | Claim |
|---|---|---|---|
| HAM10000 | +10.11 [+7.22, +12.90] | +9.05 [+6.48, +11.70] | holds |
| ISIC 2019 | +6.86 [+4.61, +9.37] | +6.92 [+4.76, +9.14] | holds |
| MILK10k derm | +34.69 [+28.41, +41.04] | +9.53 [+4.50, +14.20] | holds, magnitude 3.6× smaller |
| PAD-UFES-20 | no claim | no claim | — |

- On MILK10k derm, E45b chose a single global matched rank of 256 using test labels. E50's nested search reaches the 512 cap in every fold, and that control removes more image information. The change is driven by the control choice (A1).

ResNet-18 (E50b, `--minimal`):

| Set | E45b | E50b | Claim |
|---|---|---|---|
| HAM10000 | +10.45 [+7.64, +13.25] | +8.89 [+6.37, +11.31] | holds |
| ISIC 2019 | +6.45 [+4.30, +8.82] | +6.22 [+4.08, +8.37] | holds |
| MILK10k derm | +21.55 [+16.71, +27.09] | +22.19 [+17.02, +27.22] | holds |
| PAD-UFES-20 | no claim | +1.89 [−0.16, +4.11] | — |

On MILK10k derm the nested search selected rank 256 in every fold, the matching check passed (+6.87 [−0.65, +14.16]), and the estimate agrees with E45b.

**All six E45b claims survive the nested, training-only control. No manuscript claim is dropped under the continuity rule.** The MILK10k ConvNeXt-T magnitude changes (34.7 → 9.5) and should be reported with E50's value.

### 3.7 Robustness: fold assignments 1–2 and ResNet-18 (E50b)

`--minimal` arms: t0, targeted, nested matched search, 5 matched random and 5 shams. ConvNeXt-T unless stated.

| Set | Fold assignment 0 (E50a) | 1 | 2 |
|---|---|---|---|
| ISIC 2020 | +0.99 [+0.69, +1.29] | +1.08 [+0.76, +1.38] | +0.85 [+0.53, +1.15] |
| HAM10000 | +9.05 [+6.48, +11.70] | +8.74 [+6.11, +11.41] | +8.65 [+6.22, +11.22] |
| ISIC 2019 | +6.92 [+4.76, +9.14] | +6.74 [+4.54, +8.97] | +6.42 [+4.21, +8.61] |

- Every one of the nine set × fold-assignment contrasts has a CI above 0 and p = 0.0005.
- T − sham CIs are above 0 in every run.
- The matching check passes or is conservative (matched more damaging) in every run. On ISIC 2020 it passes for fold assignments 0 and 1 and is conservative for 2 (−1.73 [−2.61, −0.85]).
- The spread across fold assignments (ISIC 2020 0.85–1.08; HAM10000 8.65–9.05; ISIC 2019 6.42–6.92) is small relative to each CI.
- The confirmatory ISIC 2020 result is not a fold artefact.

## 4. Why HIBA, MSK 2020 and PAD-UFES-20 cannot test the mechanism

Two distinct problems, both visible in diagnostics fixed before the confirmatory runs (amendment 1).

**1. The eraser overfits when n_train ≈ d.** These sets have 870–980 training rows against 768 features.
- The LEACE coordinate is 1.8–3.1× more variable on test than on training rows. The large sets show 0.99–1.08.
- Every targeted eraser on these sets shows this, whether age or site, marginal or class-conditional.
- The matched rank is chosen from training-fold inner CV, so it cannot see the test-time damage. The matching check then fails in one direction or the other:
  - matched overshoots (marginal age, HIBA and MSK; site, HIBA and PAD);
  - matched undershoots (class-conditional age, HIBA and MSK).
- Ledoit–Wolf whitening brings the ratio to 1.10–1.25 but does not fix the matching.

**2. On HIBA and MSK 2020 the label depends strongly on age.**

| Set | Age-only AUROC for the label | corr(ConvNeXt-T score, age) |
|---|---|---|
| HIBA | 0.84 | 0.66 |
| MSK 2020 | 0.72 | 0.39 |
| ISIC 2020 | 0.67 | 0.19 |

- Marginal LEACE removes all training covariance with age, including the part induced by the label. A rank-1 erasure therefore removes most of the diagnostic signal: HIBA image AUROC falls from 0.93 to 0.65, already in inner CV.
- This is the failure mode Kumar, Tan and Sharma (NeurIPS 2022) describe for concept removal when the concept is correlated with the task.

The registered results for these sets stand as failures (§3.1). The explanation above is a post-hoc reading of pre-specified diagnostics, not an exclusion rule.

### 4.1 Class-conditional erasure (E50c, exploratory, amendment 3)

The concept is the training-fold attribute minus its training-fold class mean (training labels only). The eraser is then applied without labels.

| Set | Image LL t0 → T (marginal T) | **T − matched** | T − sham | Matching check | Site gain T − matched |
|---|---|---|---|---|---|
| ISIC 2020 | 0.0713 → 0.0728 (0.0730) | **+0.86 [+0.56, +1.16]** | +1.04 [+0.73, +1.36] | +0.08 [−0.74, +0.89] | +0.03 [−0.03, +0.10] |
| HAM10000 | 0.2321 → 0.2372 (0.2462) | **+4.05 [+2.48, +5.65]** | +4.48 [+2.78, +6.22] | −0.43 [−2.90, +1.96] | +0.41 [−0.33, +1.20] |
| ISIC 2019 | 0.3929 → 0.3946 (0.4004) | **+2.53 [+1.43, +3.69]** | +2.74 [+1.59, +3.91] | −0.34 [−1.86, +1.23] | +0.39 [−0.17, +0.94] |
| MILK10k derm | 0.4236 → 0.4567 (0.5180) | **+13.62 [+9.66, +17.36]** | +21.28 [+16.60, +25.66] | +2.32 [−4.09, +9.23] | +0.32 [−0.61, +1.19] |
| MILK10k clin | 0.4291 → 0.4674 (0.5209) | **+14.05 [+9.88, +18.27]** | +25.24 [+20.53, +30.03] | +1.50 [−6.28, +9.61] | −2.15 [−6.20, +0.73] |
| PAD-UFES-20 | 0.5202 → 0.5211 | −0.07 [−1.33, +1.23] | −0.04 [−1.22, +1.19] | −12.31 (fails) | — |
| MSK 2020 | 0.3444 → 0.3849 (0.3751) | −3.43 [−12.02, +4.19] | −3.01 [−11.77, +4.94] | +40.32 (fails) | +4.00 [+1.37, +6.67] |
| HIBA | 0.3392 → 0.6240 (0.6597) | +61.34 [+37.14, +85.10] | +61.42 [+37.22, +85.05] | +283.45 (fails) | +16.66 [+9.28, +24.14] |

- **On the five large sets:** class-conditional erasure gives a positive, specific, well-matched effect.
  - The matching check passes everywhere.
  - Site and sex gains do not move on any of the five; every targeted − matched CI includes 0.
  - The image model's log-loss rises less than under marginal erasure: 15% less on ISIC 2020, 58–77% less on the other four (computed from unrounded log-losses).
- **HIBA's +61 is not attribute-specific.** The site gain also rises (+16.7), the image model's log-loss nearly doubles, and the eraser's SD ratio is 2.19.
- **Prediction (a)** (much smaller image cost on HIBA, MSK 2020 and MILK10k) holds on MILK10k only. It fails on HIBA and MSK 2020 because of problem 1.

## 5. Priority 2: attribute specificity with anatomical site (E51)

| Set | Role | **T − matched (site)** | T − sham | Matching check | SD ratio |
|---|---|---|---|---|---|
| HIBA | confirmatory | −54.11 [−81.37, −27.49], p = 1.0 | +28.11 [−2.08, +57.64] | −278.89 (matched overshoots) | 1.97 |
| HAM10000 | replication | **+4.26 [+1.75, +6.86]**, p = 0.0015 | +6.68 [+3.06, +10.41] | −16.51 (conservative) | 1.03 |
| MILK10k derm | exploratory | **+3.83 [+1.23, +6.31]**, p = 0.0025 | +6.11 [+2.25, +10.33] | −4.53 [−9.89, +0.77] (passes) | 1.08 |
| PAD-UFES-20 cc | exploratory | −87.07 [−104.42, −69.79], p = 1.0 | +5.32 [−15.36, +24.98] | −202.52 (matched overshoots) | 3.13 |

**Registered specificity matrix** M(E, V): targeted − matched for eraser E and gain variable V, with bootstrap seed 511.

| Set | M(age, age) | M(age, site) | M(age,age) − M(age,site) | M(site, site) | M(site, age) | M(site,site) − M(site,age) | Selective |
|---|---|---|---|---|---|---|---|
| HAM10000 | +9.05 [+6.49, +11.50] | +0.39 [−0.97, +1.69] | +8.66 [+5.84, +11.32] | +4.27 [+1.64, +6.82] | +0.55 [−0.49, +1.61] | +3.71 [+1.01, +6.38] | **age yes, site yes** |
| MILK10k derm | +9.53 [+4.42, +14.54] | −1.47 [−2.65, −0.25] | +10.99 [+5.77, +15.91] | +3.83 [+1.37, +6.31] | −1.15 [−2.48, +0.19] | +4.98 [+2.44, +7.55] | **age yes, site yes** |
| HIBA | −116.56 | −38.07 | −78.49 [−126.23, −39.28] | −54.11 | −128.01 | +73.90 [+39.91, +106.69] | no, no |
| PAD-UFES-20 | +0.22 [−2.53, +2.93] | −1.91 [−4.05, −0.08] | +2.12 [−0.74, +5.20] | −87.07 | −0.93 | −86.14 | no, no |

On the two large sets with both variables, erasing age restores age's value but not site's, and erasing site restores site's value but not age's: a double dissociation. Both sets are replication or exploratory. The single confirmatory site test (HIBA) fails for the reasons in §4.

## 6. Priority 3: residual nonlinear information (E52)

The registered MLP probe (one hidden layer of 256 units, early stopping) is fitted on outer-training rows of each arm's features and evaluated on the outer test folds.

| Set | Age MLP R²: t0 | After T-D1(1) | After matched random | After sham | Linear within-test R² after T |
|---|---|---|---|---|---|
| ISIC 2020 | 0.228 [0.195, 0.257] | **0.203 [0.169, 0.233]** | 0.201 | 0.227 | −0.034 |
| HAM10000 | 0.277 [0.246, 0.306] | **0.212 [0.181, 0.242]** | 0.195 | 0.268 | −0.004 |
| ISIC 2019 | 0.259 [0.227, 0.288] | **0.229 [0.197, 0.258]** | 0.223 | 0.254 | 0.013 |
| MILK10k derm | 0.177 [0.138, 0.219] | **0.050 [0.007, 0.093]** | −0.152 | 0.174 | 0.015 |
| MILK10k clin | 0.236 [0.195, 0.277] | **0.129 [0.083, 0.172]** | −0.119 | 0.231 | 0.013 |
| HIBA / MSK 2020 / PAD-UFES-20 | 0.107 / 0.024 / −0.180 (CIs include or are below 0) | negative | negative | ≈ t0 | ≈ 0 |

- **Wording rule triggered.**
  - Age: on all five large sets the MLP still decodes age after targeted linear erasure (CI lower bound > 0).
  - Site: on HAM10000 and MILK10k derm the MLP still decodes site after site erasure. Macro AUROC is 0.734 and 0.746 against 0.819 and 0.818 at t0. Log-loss gain CI lower bounds are 0.027 and 0.112.
  - The manuscript must describe the intervention as removing *linearly decodable* age (or site) information, never the attribute.
- **Linear vs nonlinear.** On ISIC 2020, HAM10000 and ISIC 2019:
  - Targeted erasure removes the linear trace; within-test R² goes to ≈ 0.
  - It reduces nonlinear decodability no more than matched random erasure: MLP R² 0.203 vs 0.201, 0.212 vs 0.195, and 0.229 vs 0.223.
  - The image head and second stage are linear, so the redundancy E50 identifies is redundancy with the information a linear read-out of the representation uses.
- **Small sets.** The MLP probe is weaker than the linear ridge probe at t0 on every set, and fails outright on the small sets (R² ≤ 0.11 with CIs including 0). No conclusion is drawn there.

## 7. Priority 4: estimator robustness (E53, E53b)

All four estimators use the same outer folds, grouping and test cases as E40 fold assignment 0. All tuning is inside training folds.
- **(a) Score fusion.** The E40 corrected estimator. Over all 24 set × rung pairs it reproduces E40's per-observation Δ to ≤ 4.3×10⁻³ nats, with mean differences ≤ 3.1×10⁻⁶ nats. The registered 10⁻⁸ equality check is below solver tolerance (STATUS.md).
- **(b) Full-feature fusion.** L2 logistic on [features, metadata] with grouped inner-CV penalty.
- **(c) Boosted fusion.** HistGradientBoosting on [score, raw metadata], fixed hyperparameters.
- **(d) Missingness only.** Score plus missingness indicators.

### 7.1 H1 under each estimator (Q1)

Δ(HSV) − Δ(ConvNeXt-T) has a CI above 0 under (a), (b) and (c) on **all 8 sets** (all p ≤ 0.006). The paper's main result does not depend on the fusion estimator.

### 7.2 How much the estimator changes Δ (Q2)

| Set | ConvNeXt-T Δ: (a) score | (b) full-feature | (c) boosted | (d) missingness only |
|---|---|---|---|---|
| ISIC 2020 | +0.92 [+0.11, +1.68] | +0.87 [+0.14, +1.54] | +1.36 [+0.49, +2.23] | −0.03 |
| HAM10000 | +2.38 [−2.13, +6.19] | **+4.96 [+2.12, +7.90]** | **+6.83 [+2.46, +11.21]** | −0.69 |
| ISIC 2019 | +9.80 [+3.93, +16.08] | +8.72 [+5.36, +12.26] | +16.01 [+9.02, +23.36] | +2.49 [+0.94, +3.89] |
| MILK10k derm | +8.29 [−0.74, +15.53] | **+11.99 [+6.62, +17.21]** | **+15.69 [+10.50, +20.84]** | −0.22 |
| MILK10k clin | +6.30 [−2.25, +13.03] | **+7.56 [+1.89, +12.97]** | **+12.32 [+7.69, +16.98]** | +0.11 |
| HIBA | +3.80 [−12.63, +20.13] | **+7.03 [+2.32, +11.88]** | +3.64 [−14.05, +20.38] | −12.82 |
| MSK 2020 | **+17.40 [+6.89, +27.94]** | +1.66 [−5.89, +8.86] | **+15.17 [+1.27, +29.02]** | +1.32 [+0.60, +2.19] |
| PAD-UFES-20 | +206.87 | +181.47 | +224.75 | +158.31 |

- **Boosted fusion recovers more metadata value than score fusion on 5 of 8 sets**, at both ResNet-18 and ConvNeXt-T. (c) − (a) has a CI above 0 on HAM10000, ISIC 2019, MILK10k derm and clin, and PAD-UFES-20: +4 to +8, and +15 to +18 on PAD-UFES-20. It is ≈ 0 on ISIC 2020, HIBA and MSK 2020. The linear second stage misses score × metadata interactions, so score fusion is a conservative estimate of added value.
- **Whether metadata add anything at ConvNeXt-T depends on the estimator on 5 of 8 sets.**
  - HAM10000, MILK10k derm and clin, and HIBA: zero under (a), positive under (b) and/or (c).
  - MSK 2020: positive under (a) and (c), zero under (b). (b) − (a) is −15.74 [−26.67, −5.23].
  - Dataset-level statements of the form "metadata no longer help ConvNeXt-T on X" should not be made from one estimator.
- **Missingness share (Q3)** at ConvNeXt-T:
  - PAD-UFES-20: 77% of (a), the known label-informative missingness;
  - ISIC 2019: 25%;
  - MSK 2020: 8%;
  - ≈ 0 elsewhere.
  - On HIBA, missingness indicators *hurt* (−12.8), an overfit second stage on 1,224 rows.

### 7.3 E53b: the E50 primary contrast under full-feature fusion (registered, gated on G1)

The rebuilt erasers reproduce E50's first-stage scores exactly (max difference 0).

| Set | (b) T − matched | (b) matching check | (a) T − matched (E50) | (b) − (a) |
|---|---|---|---|---|
| ISIC 2020 | **+1.49 [+0.82, +2.20]**, p = 0.0005 | −0.30 [−1.11, +0.57] (passes) | +0.99 [+0.70, +1.32] | +0.49 [−0.08, +1.06] |
| HAM10000 | **+8.12 [+5.53, +10.75]** | −12.88 (conservative) | +9.05 | −0.93 [−2.55, +0.78] |
| ISIC 2019 | **+5.13 [+3.66, +6.60]** | −3.77 (conservative) | +6.92 | −1.79 [−3.08, −0.50] |
| MILK10k derm | **+23.49 [+17.20, +29.94]** | −39.64 (conservative) | +9.53 | +13.96 [+9.56, +18.27] |
| MILK10k clin | **+25.85 [+19.79, +31.80]** | −36.70 (conservative) | +11.63 | +14.21 [+10.06, +18.61] |

- The targeted-erasure effect survives a change of fusion estimator on every set where E50 found it. Magnitudes agree within about 2 on the three large sets.
- On MILK10k, full-feature fusion roughly doubles the contrast. Matched random erasure (512 dimensions) also raises age's value under (b) by +31.7, so MILK10k's estimates depend strongly on the control and the estimator. They support the sign of the effect, not its size.

## 8. Priority 5: representation conditions (E54)

As registered, existing results are reused:
- E14 and E14b: fine-tuning on five datasets;
- E31c and E31d: fine-tuning on fractions of HAM10000;
- E31a and E31b: frozen family size and data size.

No new backbones were added.

The only new run repeated E14b on HIBA and PAD-UFES-20 with training seeds 1 and 2. It used the same schedule, folds and E14b second stage; fold k, seed s uses seed 1000·s + k. Seed 0 is E14b and reproduces its stored values exactly.

| Set | Backbone | Frozen Δ | Fine-tuned Δ, seeds 0 / 1 / 2 | Fine-tuned AUROC | Frozen − fine-tuned CI, per seed |
|---|---|---|---|---|---|
| HIBA | ResNet-18 | +18.86 | +1.37 / −2.05 / +1.21 | 0.940–0.946 | all above 0 (lowest bound +6.27) |
| HIBA | ConvNeXt-T | +2.60 | −7.24 / −8.71 / −8.97 | 0.951–0.956 | all above 0 (lowest bound +2.61) |
| PAD-UFES-20 | ResNet-18 | +234.17 | +172.71 / +171.48 / +176.21 | 0.897–0.899 | all above 0 (lowest bound +39.68) |
| PAD-UFES-20 | ConvNeXt-T | +206.44 | +123.94 / +121.04 / +127.77 | 0.925–0.927 | all above 0 (lowest bound +61.33) |

- **Seed variation is small** relative to the frozen–fine-tuned gap. The largest range is 6.73 (PAD-UFES-20 ConvNeXt-T), against a gap of about 82.
- **The E14b conclusion on these two datasets is not a training-seed artefact.**
- **Still open:**
  - seeds on ISIC 2019 and MILK10k, where E14b's drop was not significant;
  - strict cross-fitting of the fine-tuned second stage (review item R7-11);
  - a ConvNeXt-T fine-tune on ISIC 2019 and MILK10k.

## 9. Priority 6: dataset shift and decision rules (E55)

*The rerun (30 Oct 03:13–03:18) reproduced all 12 evaluations of the first run exactly, and both summaries (results/E55_shift_rules.json).*

**Setup.**
- Frozen ConvNeXt-T, harmonised age/sex/coarse site, complete cases.
- Everything is fitted on the source datasets: image head, second stage, imputation and rule inputs.
- The target's labels are used only to score.

**Dataset level** (leave one dataset out within a label family; 7 targets, the most this collection supports):

| Target | Trained on | n | Δ fused − image on target | Oracle | R2 (source-validated) | R3 (collection rule; predicted Δ) |
|---|---|---|---|---|---|---|
| HAM10000 | ISIC 2019, ISIC 2020 | 9,761 | +5.43 [+2.89, +8.10] | use | use ✓ | use ✓ (6.3) |
| ISIC 2019 | HAM, ISIC 2020 | 14,156 | +8.43 [+3.43, +13.47] | use | use ✓ | use ✓ (1.7) |
| ISIC 2020 | HAM, ISIC 2019 | 32,531 | +2.12 [+1.45, +2.76] | use | **don't ✗** | use ✓ (5.8) |
| PAD-UFES-20 | HIBA, MSK, MILK derm | 1,089 | **−19.61 [−36.42, −2.41]** | don't | **use ✗** | **use ✗** (15.7) |
| HIBA | PAD, MSK, MILK derm | 1,085 | +32.00 [+21.03, +42.63] | use | use ✓ | use ✓ (14.6) |
| MSK 2020 | PAD, HIBA, MILK derm | 1,157 | −5.65 [−13.69, +3.41] | don't | **use ✗** | **use ✗** (12.7) |
| MILK10k derm | PAD, HIBA, MSK | 5,086 | +18.79 [+14.51, +23.18] | use | use ✓ | use ✓ (22.7) |

Mean regret against the better choice on each target:

| Rule | Mean regret (×10⁻³ nats) |
|---|---|
| R0, never | 9.54 |
| R1, always | **3.61** |
| R2, source-validated | 3.91 |
| R3, collection rule | **3.61** |

- **R3 recommended "use" on every target, so it equals R1 here.** Neither source-fitted rule detected the two targets where metadata hurt.
  - On PAD-UFES-20, the harm is significant. Possible contributors include differences in site coding and in age distribution across collections; neither was tested.
  - On MSK 2020 the CI includes 0.
- **Within-dataset sources** are not independent and are reported separately:
  - HAM10000: rosendahl, vidir_modern and vienna_dias held out.
  - ISIC 2019: BCN ↔ MSK.
  - Metadata helped on 4 of 5 targets. Mean regret: R1 1.03, R2 3.81, R3 5.38, R0 16.94.
  - R3 said "don't use" on two HAM sources where metadata helped. Their source retained fractions were negative.
- **Reading.** With 7 targets the comparison is descriptive. No source-only rule beat "always use" here. The collection rule's added complexity bought nothing under shift. Metadata can hurt under dataset shift (PAD-UFES-20), and source data did not predict when.

## 10. Deviations, failures and compute

- E50a took 5.6 h of wall time against the registered ~2 h estimate. ISIC 2020 took 3 h 16 min, about 1 h 50 min of it with the Mac asleep (lid closed on battery, 15:54; pmset log). Results were unaffected.
- The PAD-UFES-20 shake-down run was superseded by amendments 1–2. Its log and JSON are kept.
- E53's first run failed (§7). It was rerun with an encoding-only fix.
- E50b scope narrowed for compute (STATUS.md, 29 Oct 23:06 IST):
  - ResNet-18 on the four E45b sets only;
  - fold assignments 1–2 on ISIC 2020, HAM10000 and ISIC 2019 only;
  - `--minimal` arms only: t0, targeted, nested matched search, 5 matched random and 5 sham, with no same-rank, re-append or LW arms. A unit test confirmed the default path reproduces stored contrasts exactly.
- E54 is registered as "if time allows" and was run last. It completed in full, both seeds on both datasets (30 Oct 01:22–03:13).
- **Code versions.** e50_dose_restoration.py has three logged versions (sha256, first 12 characters):

  | Hash | Used by | How it was made |
  |---|---|---|
  | 9915166e192a | E50a | original; copy in logs/e50_dose_restoration_E50a_version.py |
  | 4631999e6079 | E51, E50c | adds `--conditional` |
  | ca82ac760e14 | E52, E53b, E50b | adds `--minimal`; file modified 29 Oct 23:04 |

  Each edit left the default path unchanged. For `--minimal` this was tested by reproducing stored contrasts exactly (STATUS.md).

## 11. Limitations

1. **One confirmatory success out of four.**
   - The mechanism passed its registered confirmatory test on one untouched dataset (ISIC 2020).
   - The other two age tests and the site test failed. All three failures fall on sets of about 1,100–1,200 images, where §4 shows the design cannot work.
   - The four large-set supports (HAM10000, ISIC 2019, MILK10k derm and clin) are replication or exploratory: E45 had already been run on them.
   - MILK10k derm and clin share lesions, so they count once.
2. **The method's applicability is bounded.**
   - Closed-form LEACE overfits when the training set is not much larger than the feature dimension. These sets have 870–980 training rows against 768 features.
   - The design is also not attribute-specific when the label depends strongly on the attribute (HIBA, MSK 2020).
   - Small, strongly age-associated datasets therefore remain untested, not negative.
3. **Linear information only.**
   - An MLP still decodes age (R² 0.05–0.23) and site after targeted erasure (§6).
   - The intervention removes the linear trace that the linear image head and second stage use. It is not evidence about nonlinear redundancy.
   - The MLP probe itself is weak at these sample sizes: below the linear ridge at t0 everywhere, and uninformative on the small sets.
4. **The matched control is approximate.**
   - On HAM10000, ISIC 2019 and MILK10k the performance-matched random control removes more image information than the targeted eraser. The matching check's CI is below 0, so these contrasts are conservative.
   - On MILK10k the matched rank hits the 512 cap at ConvNeXt-T, and the contrast's size depends on the control and the estimator (§3.6, §7.3). Only its sign is robust.
5. **Restoration is incomplete.** Re-appending the erased coordinate restores the image model fully but returns only 74–93% of the rise in age's added value. The residual is small but its CI excludes 0 and it is unexplained.
6. **Class-conditional erasure (E50c) is exploratory and post hoc.** It was motivated by the confirmatory failures and has no confirmatory status. It also inherits the small-sample problem.
7. **Estimator dependence.** Per-dataset magnitudes of added value at strong representations change with the fusion estimator, and on 5 of 8 sets whether any value remains does too (§7.2). Score fusion is conservative.
8. **Dataset shift.**
   - E55 has seven dataset-level targets in two label families, so the rule comparison is descriptive.
   - Within-dataset source splits are not independent.
   - Harmonisation reduces metadata to age, sex and coarse site.
9. **Fine-tuning seeds (E54)** only cover HIBA and PAD-UFES-20, with the original (not strictly cross-fitted) second stage. The datasets where fine-tuning's effect was not significant (ISIC 2019, MILK10k) have still not been re-seeded.
10. **Deviations.** E50b used a reduced arm set and a reduced set list, and two jobs needed code fixes and reruns. All are logged in §10 and STATUS.md.
11. **No causal or clinical claims.**
    - The interventions act on learned representations, not on patients or images.
    - "The image model uses age-related features" means its linear read-out depends on directions correlated with age in the training data.

## 12. Novelty assessment

Literature checked on 29 Oct 2026, before 23:23 IST, with standard and extended web searches. Verification level is given for each item. Absence from a search is weak evidence.

| Element | Closest prior work (verification) | Assessment |
|---|---|---|
| Linear concept erasure (LEACE) as the intervention | Belrose et al., NeurIPS 2023 (verified earlier, iteration 5) | Tool, not contribution. |
| Removing a property from a representation and comparing with removal of the same number of random directions | Elazar, Ravfogel, Jacovi and Goldberg, "Amnesic Probing", TACL 2021 (§2.3 read from the PDF: INLP plus a "Rand" control of equal rank) | Our same-rank control is their control. **Precedent.** |
| Outcome = change in the *added value of a separately observed copy of the concept* (patient metadata) after erasing it from the image representation, with a training-only performance-matched random control, permuted-concept shams, an information dose and re-append restoration | None found. Amnesic probing measures the model's own task behaviour. Multimodal fusion work handles redundancy through losses, not interventions. | **Appears new as an evaluation design** (moderate novelty). The components are standard; the combination and the question it answers are not, as far as found. |
| Concept removal fails or destroys task signal when the concept correlates with the label | Kumar, Tan and Sharma, NeurIPS 2022 (abstract and listing) | Our HIBA and MSK 2020 failure is an instance of a known failure mode. **Not new.** The diagnostics we used (age-only AUROC, score–age correlation) are simple and practical. |
| Linear erasure leaves nonlinearly decodable information | Ravfogel, Vargas, Goldberg and Cotterell, "Adversarial Concept Erasure in Kernel Space", EMNLP 2022 (abstract and listing); Belrose et al. (guarantee is linear only) | Expected. Our E52 is a confirmation in this domain. **Not new.** |
| Label-aware erasure variants | Belrose, "Least-squares concept erasure with oracle concept labels" (O-LEACE), EleutherAI blog, Dec 2023 (read). It uses the *concept* label at inference. Fairness work on conditional independence (not read). | E50c's training-label residualisation, applied label-free, is a simple variant. **No method novelty claimed.** |
| LEACE overfits out of sample when n_train ≈ d, so training-only matching cannot see test-time damage | Not found stated for LEACE. It follows from covariance-estimation theory. | A useful practical caution. **Low novelty.** |
| Attribute-specific double dissociation (age vs site) of restored metadata value | None found. | **New empirical evidence** (two datasets, replication/exploratory status). |
| Added value at strong representations depends on the fusion estimator | Not found in the dermatology fusion literature checked. | **Useful empirical caution.** Low-to-moderate novelty. |
| Source-only rules for whether to use metadata under shift | Not searched specifically. | Negative, descriptive result. **Low novelty.** |

**No new mathematical framework is warranted.** The experiments raised no question that Proposition 1 plus LEACE's guarantee cannot frame:
- The specificity failures are explained by the known concept–label correlation problem.
- The small-sample failure is an estimation problem.
- One open question is redundancy through nonlinear information, which a linear read-out cannot use. Answering it would require nonlinear fusion and nonlinear erasure, and Ravfogel et al. (2022) report that exhaustive nonlinear erasure is an open problem.

## 13. Recommended manuscript changes

These are recommendations only; the paper has not been edited. Section references are to main.tex v6.

1. **Erasure section (Sec. "Erasing what the image encodes about age"; Table erasure; Fig. erasure).**
   - Replace E45b's matched-control numbers with E50's nested, training-only control:
     - HAM10000 +9.05;
     - ISIC 2019 +6.92;
     - MILK10k derm +9.53 at ConvNeXt-T (state the 3.6× change and its cause);
     - ResNet-18 +8.89 / +6.22 / +22.19.
   - Add:
     - the pre-registered confirmatory result on the untouched ISIC 2020 (+0.99 [+0.69, +1.29], Holm p = 0.002), stable across three fold assignments;
     - the sham control;
     - the information-dose curve, monotone with CIs above 0 at every dose on the three large sets;
     - restoration fractions (0.71–0.97 targeted vs 0.03–0.10 matched);
     - the image-derived-age check.
   - Report the three failed confirmatory tests and the reason (§4) in the text, not only the supplement.
2. **Specificity.** Replace the single-dataset sentence ("HAM10000: age …, site …, sex …") with the registered double dissociation on HAM10000 and MILK10k derm (§5), stating its replication or exploratory status.
3. **Wording (E52 rule).** Every mention of erasing "age" or "the image's age information" should say *linearly decodable* age information. In the Discussion, change "the value age loses to strong representations returns when their age information is removed" to: age's added value rises when the linear trace of age is removed, by 0.71–0.97 nats per nat of image log-loss lost, against 0.03–0.10 for matched random removal (ISIC 2020, HAM10000, ISIC 2019).
4. **"So the image models use age-related features for diagnosis."** Qualify this sentence. Marginal erasure also removes covariance that comes from the label's dependence on age. Class-conditional erasure (E50c, exploratory) costs 58–77% less image log-loss on four of the five large sets while still raising age's value. Suggested wording: "the image model's linear read-out depends on directions that covary with age; part of this covariance is induced by the label's dependence on age."
5. **Limitation (ii) and the Discussion's per-dataset magnitudes** ("from little (HAM10000; HIBA …) to about half (PAD-UFES-20, MSK 2020, ISIC 2019)").
   - Add that these magnitudes depend on the fusion estimator, and that on 5 of 8 sets so does whether any value remains at ConvNeXt-T.
   - State that H1 holds under all three estimators on all 8 sets (E53).
   - Consider an appendix table with (a)–(d).
6. **Missingness.** Quantify PAD-UFES-20's missingness share (77% of added value at ConvNeXt-T) and ISIC 2019's (25%) from E53(d).
7. **Dataset shift paragraph.** Add E55:
   - with decision rules fitted on sources only, no rule beat "always use";
   - the collection rule recommended use on every target and so equalled "always use";
   - metadata hurt on PAD-UFES-20 and no source-fitted rule flagged it.
   - Cut or qualify any implication that the collection rule is useful under shift.
8. **Limitation (i), fine-tuning seeds.** Replace "one training run per fold" with: "three training seeds on HIBA and PAD-UFES-20 (both backbones). The fine-tuning drop held for every seed, with a seed range ≤ 6.7×10⁻³ nats. Other datasets used one seed."
9. **New limitation.** The erasure design needs n_train ≫ d and a weak attribute–label association. It is uninformative on HIBA, MSK 2020 and PAD-UFES-20.
10. **Related work.** Cite amnesic probing (Elazar et al., TACL 2021) as the closest methodological precedent and state the difference: the outcome is the value of a separately observed metadata variable. Cite Kumar et al. (NeurIPS 2022) for the concept–label correlation failure mode. Both need a full-text read before citation; amnesic probing §2.3 has been read.
11. **Do not add** a new mathematical framework (§12).
