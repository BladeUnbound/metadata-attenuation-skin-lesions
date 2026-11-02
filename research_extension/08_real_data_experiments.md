# 08 — Real-data experiments (REAL-2 to REAL-8)

*Phase 6. Sections 1–2 were written 2026-11-01; later sections are added as their runs finish. Registry entries: EXPERIMENT_REGISTRY.md, "Phase 4/6 — real data", registered 2026-10-30 11:42 IST before any of these runs. Protocol, arm definitions and units (10⁻³ nats per image) are as in 06 §1.*

## 1. REAL-2 — confirmatory test of substitution efficiency (ViT-L/16 SWAG, frozen)

**Registered hypothesis.** On large sets, LEACE age erasure is more substitutive than performance-matched random erasure: η_T − η_R > 0.
- Test: one-sided, with Holm correction over HAM10000, ISIC 2019 (non-HAM) and ISIC 2020, α = 0.05.
- Statistic: the paired grouped bootstrap of η_T − η_R (B = 2000).
- A set whose C CI includes 0 counts as p = 1.

**Why ViT-L/16.** Erasure had never been run on these features in this project. The runs used the unchanged e50_dose_restoration.py code path in minimal mode (EXECUTION_LOG, 2026-10-30 11:42–13:05 IST). The test was computed by scripts/real_reanalysis.py on 2026-11-01 21:25 IST.

| Set | C [95% CI] | M | L [95% CI] | η_T [95% CI] | η_R | η_T − η_R [95% CI] | p | Holm p | Decision |
|---|---|---|---|---|---|---|---|---|---|
| ISIC 2020 | 0.92 [0.40, 1.43] | 0.86 | 0.06 [−0.36, 0.50] | 0.93 [0.63, 1.82] | 0.10 | 0.83 [0.54, 1.75] | 0.0005 | 0.0015 | reject H₀ |
| HAM10000 | 9.48 [6.05, 12.93] | 7.89 | 1.60 [−1.47, 4.56] | 0.83 [0.60, 1.20] | 0.07 | 0.77 [0.54, 1.14] | 0.0005 | 0.0015 | reject H₀ |
| ISIC 2019 | 4.75 [1.81, 7.49] | 5.00 | −0.25 [−2.82, 2.27] | 1.05 [0.65, 2.30] | 0.03 | 1.02 [0.63, 2.26] | 0.0015 | 0.0015 | reject H₀ |

**Outcome.** All three registered tests reject at Holm-adjusted p = 0.0015. The registered hypothesis is supported on all three sets.

**Registered secondary diagnostics** (age-gain contrasts in M, 10⁻³ nats):

| Set | LEACE − matched random | LEACE − sham | Matching check (image log-loss, LEACE − random) | Sham image cost |
|---|---|---|---|---|
| ISIC 2020 | 0.74 [0.49, 0.99] | 0.86 [0.58, 1.14] | −0.32 [−0.89, 0.22]: passes | 0.04 [−0.00, 0.09] |
| HAM10000 | 6.40 [4.15, 8.63] | 7.89 [5.38, 10.28] | −13.2 [−18.0, −8.4]: random costlier | 0.25 [0.03, 0.47] |
| ISIC 2019 | 4.84 [3.15, 6.56] | 4.98 [3.25, 6.76] | −0.73 [−3.50, 2.24]: passes | 0.07 [−0.11, 0.24] |

**Reading.**
- On ISIC 2019 and ISIC 2020 the matching check passes, so the random arm costs the same image log-loss as LEACE on the test folds. The confirmatory difference is therefore not driven by unequal cost.
- On HAM10000 the random arm overshoots (as on ConvNeXt-T; 06 §3). 06 §3 also shows that random η changes little with rank.
- The sham costs 0.25 on HAM (CI excludes 0), about 3% of C there.

**What REAL-2 does and does not show.**
- **It shows** that, on a frozen foundation representation never used in exploratory erasure work, the diagnostic information removed together with linearly decodable age is largely recoverable from the recorded age, while equally costly random removal is not.
- **It does not show** that the fused cost L is zero: its CIs include 0 on all three sets, but equivalence was not tested.
- **It does not establish** anything about non-linear age information remaining after LEACE (REAL-4), or about clinical consequences.
- η's CIs are wide, with upper limits up to 2.3, because C is small relative to its sampling error on ISIC 2019 and ISIC 2020. The confirmatory statistic is the difference, whose lower limits (0.54–0.63) are well above 0.

## 2. REAL-3 — representation ladder (E6). Exploratory.

Rungs: HSV colour histograms, ResNet-18 (E50b), ConvNeXt-T (E50a), ViT-B/16 and ViT-L/16 SWAG (REAL-2), all frozen and all at fold assignment 0. The ViT-B/16 and HSV runs are new (2026-10-30 13:05–14:05 IST).

**Baseline quantities at t0.** Image-only risk R_Y(S) in nats; Δ_A(S), the added value of age, in 10⁻³ nats; age R² before erasure, within-test and training-fold.

| Set | Rung | R_Y(S) | Δ_A(S) | age R² within-test (training-fold) |
|---|---|---|---|---|
| HAM10000 | HSV | 0.306 | 14.7 | 0.03 (0.06) |
| | ResNet-18 | 0.256 | 2.6 | 0.22 (0.28) |
| | ConvNeXt-T | 0.232 | 2.0 | 0.33 (0.39) |
| | ViT-B/16 | 0.233 | 2.2 | 0.30 (0.37) |
| | ViT-L/16 | 0.226 | 2.4 | 0.27 (0.35) |
| ISIC 2019 | HSV | 0.493 | 11.4 | 0.08 (0.10) |
| | ResNet-18 | 0.410 | 3.7 | 0.26 (0.31) |
| | ConvNeXt-T | 0.393 | 3.2 | 0.32 (0.37) |
| | ViT-B/16 | 0.382 | 3.2 | 0.29 (0.36) |
| | ViT-L/16 | 0.391 | 3.5 | 0.30 (0.36) |
| ISIC 2020 | ConvNeXt-T | 0.071 | 1.1 | 0.23 (0.29) |
| | ViT-L/16 | 0.069 | 0.6 | 0.21 (0.29) |

**LEACE erasure along the ladder:**

| Set | Rung | C [95% CI] | η_T [95% CI] | η_R | η_T − η_R [95% CI] | M contrast, LEACE − matched |
|---|---|---|---|---|---|---|
| HAM10000 | HSV | 4.9 [2.8, 7.2] | 0.82 [0.57, 1.39] | 0.05 | 0.77 [0.52, 1.33] | 3.67 [2.92, 4.45] |
| | ResNet-18 | 18.5 [13.0, 24.0] | 0.63 [0.47, 0.85] | 0.10 | 0.53 [0.39, 0.75] | 8.89 [6.37, 11.31] |
| | ConvNeXt-T | 14.1 [10.3, 17.8] | 0.79 [0.61, 1.01] | 0.08 | 0.71 | 9.05 [6.48, 11.70] |
| | ViT-B/16 | 13.6 [10.0, 17.4] | 0.79 [0.61, 1.02] | 0.08 | 0.71 [0.54, 0.93] | 8.95 [6.21, 11.53] |
| | ViT-L/16 | 9.5 [6.0, 12.9] | 0.83 [0.60, 1.20] | 0.07 | 0.77 [0.54, 1.14] | 6.40 [4.15, 8.63] |
| ISIC 2019 | HSV | 0.2 [−0.5, 0.8] | undefined | — | — (p = 1) | 0.92 [0.56, 1.26] |
| | ResNet-18 | 7.6 [3.8, 11.2] | 0.92 [0.58, 1.62] | 0.05 | 0.87 [0.54, 1.57] | 6.22 [4.08, 8.37] |
| | ConvNeXt-T | 7.5 [4.2, 10.7] | 0.97 [0.64, 1.56] | 0.03 | 0.94 | 6.92 [4.76, 9.14] |
| | ViT-B/16 | 6.1 [2.8, 9.4] | 1.00 [0.64, 2.00] | 0.05 | 0.95 [0.60, 1.94] | 5.57 [3.64, 7.49] |
| | ViT-L/16 | 4.7 [1.8, 7.5] | 1.05 [0.65, 2.30] | 0.03 | 1.02 [0.63, 2.26] | 4.84 [3.15, 6.56] |
| ISIC 2020 | ConvNeXt-T | 1.70 [0.92, 2.49] | 0.71 [0.50, 1.18] | 0.10 | 0.61 | 0.99 [0.69, 1.29] |
| | ViT-L/16 | 0.92 [0.40, 1.43] | 0.93 [0.63, 1.82] | 0.10 | 0.83 [0.54, 1.75] | 0.74 [0.49, 0.99] |

Two more rungs: MILK10k derm ResNet-18 (C 86.9, η 0.60, η_R 0.37, difference 0.23 [0.17, 0.29]) and PAD-UFES-20 ResNet-18 (C not detectable).

figures/REAL-1_eta.png (right) plots η_T along the ladder with C annotated.

**Answers to the registered question.**
1. **How much age is encoded.** Every deep rung carries substantial linear age information (within-test R² 0.21–0.33; training-fold 0.28–0.39), against 0.03–0.08 for HSV. Beyond ResNet-18, stronger encoders do not encode *more* age: ConvNeXt-T is the highest on both sets, and ViT-L/16 slightly lower.
2. **The attenuation effect.** Age's added value Δ_A(S) falls from 11–15 at HSV to 2–3.5 at every deep rung. This reproduces the parent paper's metadata-attenuation result inside this protocol.
3. **Cost of erasure.** Within the deep rungs, the strongest encoder pays the *lowest* image-only cost for erasing age. HAM: ResNet-18 18.5, ConvNeXt-T 14.1, ViT-B 13.6, ViT-L 9.5. ISIC 2019: 7.6 / 7.5 / 6.1 / 4.7. ISIC 2020: 1.70 → 0.92. The CIs of adjacent rungs overlap, and no ladder trend test was registered.
4. **Substitution efficiency.** LEACE's η lies between 0.63 and 1.05 at every deep rung on HAM and ISIC 2019. Random η stays ≤ 0.10 throughout. The η gap is significant at every deep rung (p ≤ 0.0015), including HSV on HAM.
   - η is *not* materially changed by representation strength: overlapping CIs, a range of 0.79–0.83 above ResNet-18 on HAM, and 0.92–1.05 across all deep rungs on ISIC 2019.
   - ResNet-18 on HAM is the one low point (0.63 [0.47, 0.85]).
5. **Interpretation, descriptive only.** Stronger representations encode age about as strongly and lose less diagnostic signal when it is removed. What they lose remains largely substitutable by the recorded age. The ladder therefore does not support the idea that "foundation models entangle age with diagnosis more than weak models". If anything, the entangled share (1 − η)·C shrinks with representation strength on HAM: 6.9 at ResNet-18, 1.6 at ViT-L.

*Sections 3–7 were written 2026-11-02 from the queue that ran 2026-11-01 21:25 → 2026-11-02 00:28 IST (EXECUTION_LOG.md). All runs used real_new.py 5c5f8074d8a3 importing e50_dose_restoration.py ca82ac760e14; summaries come from real_new_analysis.py (results/real/REAL-new_*.json). All five registrations are exploratory. ConvNeXt-T throughout, fold assignment 0.*

## 3. REAL-4 — positive control and distributional leakage audit

### 3.1 Positive control: oracle residual (population η = 1)

Z = S − β̂·A, with β̂ fitted on outer-training rows, and A supplied to the second stage. (Z, A) contains S exactly, so the population η is 1 and L is 0. The estimate shows how far the strict two-stage estimator (an image head on Z, then the image score plus age) pulls η̂ away from 1.

| Set | LEACE C | LEACE η | Oracle C | Oracle L [CI] | Oracle η [CI] |
|---|---|---|---|---|---|
| HAM10000 | 14.1 | 0.79 [0.61, 1.01] | 13.0 | 0.08 [−1.37, 1.60] | 0.99 [0.88, 1.12] |
| ISIC 2019 | 7.5 | 0.97 [0.64, 1.55] | 7.8 | −0.72 [−2.68, 1.24] | 1.09 [0.86, 1.47] |
| ISIC 2020 | 1.70 | 0.71 [0.50, 1.18] | 1.68 | 0.20 [−0.10, 0.52] | 0.88 [0.73, 1.08] |
| MILK10k derm | 94.4 | 0.60 [0.53, 0.67] | 73.5 | −0.97 [−4.94, 3.25] | 1.01 [0.96, 1.07] |
| HIBA (stress test) | 320 | 0.37 [0.15, 0.54] | 262 | **30.4 [14.0, 46.0]** | **0.88 [0.82, 0.94]** |
| PAD-UFES-20 (smoke test) | 2.6 [−0.9, 6.4] | undefined | 1.9 [−0.6, 4.5] | — | undefined |

**Findings.**
1. **The large sets pass.** On all four, η̂ recovers 1 within its CI, and the oracle's L has a CI that includes 0. On these sets the estimator does not by itself push η below 1, so LEACE's η < 1 is not an estimator artefact.
2. **MILK10k: the lower η comes from what LEACE removes.** The oracle gives 1.01 [0.96, 1.07] where LEACE gives 0.60 [0.53, 0.67]. So the part of LEACE's cost on MILK10k that age cannot recover is a property of what LEACE removes, not of the measurement.
3. **ISIC 2020** passes, but its point estimate (0.88) sits below 1. With C ≈ 1.7, a small absolute L (0.20) is a large share of the cost.
4. **HIBA fails the positive control.** η̂ = 0.88 with a CI that excludes 1, and L = 30.4 > 0, although (Z, A) contains S.
   - This is a finite-sample shortfall of the two-stage estimator at n/d ≈ 1.5. The mechanism is not isolated: the first stage overfits, β̂ is noisy, or both.
   - It is new evidence for the small-set rule (10 §4). The check was registered before the run, but the rule itself was set earlier and not blind.
5. **PAD-UFES-20:** C is not detectable for either arm, so η is undefined, as in REAL-1.

### 3.2 Leakage beyond the linear probe

Probes were fitted on outer-training rows and scored on the outer test fold:
- ridge: within-test R² of signed age;
- HGB: gradient-boosted trees, giving within-diagnostic-class R², |A − median| R², and the age-decile log-loss gain as a fraction of log 10.

Random and sham arms use draw j = 0.

| Set | Arm | ridge R² | HGB within-class R² | HGB \|A − med\| R² | HGB decile gain |
|---|---|---|---|---|---|
| HAM10000 | t0 | 0.325 | 0.337 | 0.176 | 0.079 |
| | LEACE | −0.004 | 0.253 | 0.161 | 0.045 |
| | matched random | 0.248 | 0.275 | 0.145 | 0.059 |
| | sham | 0.325 | 0.339 | 0.172 | 0.079 |
| ISIC 2019 | t0 | 0.323 | 0.346 | 0.086 | 0.018 |
| | LEACE | 0.013 | 0.306 | 0.076 | 0.008 |
| | matched random | 0.289 | 0.333 | 0.081 | 0.014 |
| | sham | 0.322 | 0.349 | 0.084 | 0.017 |
| ISIC 2020 | t0 | 0.226 | 0.268 | 0.051 | 0.032 |
| | LEACE | −0.034 | 0.226 | 0.042 | 0.020 |
| | matched random | 0.185 | 0.242 | 0.042 | 0.030 |
| | sham | 0.227 | 0.267 | 0.049 | 0.032 |
| MILK10k derm | t0 | 0.284 | 0.195 | 0.245 | 0.052 |
| | LEACE | 0.015 | 0.107 | 0.161 | 0.024 |
| | matched random | 0.056 | 0.050 | 0.078 | 0.007 |
| | sham | 0.284 | 0.193 | 0.259 | 0.053 |
| HIBA | t0 | 0.355 | 0.126 | −0.024 | −0.025 |
| | LEACE | −0.061 | −0.252 | −0.075 | −0.126 |

**Share of t0 leakage that remains after LEACE.**

| | HAM | ISIC 2019 | ISIC 2020 | MILK10k |
|---|---|---|---|---|
| HGB within-class R² | 75% | 88% | 84% | 55% |
| \|A − median\| R² | 91% | 88% | 82% | 66% |
| decile gain | 57% | 44% | 63% | 46% |

**Findings.**
1. **LEACE delivers linear guardedness and nothing more.** Ridge R² after erasure is −0.034 to 0.015 on every large set. Gradient-boosted probes still recover 55–88% of the within-class age R².
2. **Against non-linear probes, LEACE and matched random erasure remove similar amounts.** On HAM10000, ISIC 2019 and ISIC 2020, within-class R² is 0.23–0.31 after LEACE and 0.24–0.33 after random, and |A| R² is within 0.016. On MILK10k the random arm (rank 512, 1.4× LEACE's cost) removes more of everything.
3. **Shams are inert on the five sets in the table:** each leakage measure is within 0.014 of t0. (On PAD-UFES-20, the smoke test, the sham decile gain differs by 0.024, inside the probe noise there: t0 is already negative.)
4. **HIBA:** HGB probes are negative even at t0. The non-linear audit is uninformative at n/d ≈ 1.5, as SYN-1 finding 6 predicted.
5. **Consequence.** η describes the information removed *with the linear age direction*. It does not describe a representation from which age has been removed (09 §3).

## 4. REAL-5 — nonlinear adversarial erasure on the development set

**Setup.**
- Arms: NAE-MSE and NAE-decile (scripts/nonlin.py) at λ_A ∈ {2, 8}.
- The erasers are attribute-only (λ_Y = 0) and fitted on each outer fold's training rows.
- A is standardised, so the MSE adversary's chance loss is 1.0. The decile adversary's chance loss is log 10 = 2.30.
- LEACE and matched random come from the identical E50 HAM10000 run. Pairing was checked: identical y, groups and t0 losses.

| Arm | C [CI] | L | M | η [CI] | η − η_LEACE [CI] | ridge R² | HGB within-class | \|A\| R² | decile gain | distortion |
|---|---|---|---|---|---|---|---|---|---|---|
| t0 | — | — | — | — | — | 0.325 | 0.337 | 0.176 | 0.079 | 0 |
| LEACE | 14.1 [10.2, 17.8] | 2.9 | 11.2 | 0.79 [0.61, 1.01] | — | −0.004 | 0.253 | 0.161 | 0.045 | — |
| matched random (E50) | 25.5 | 23.4 | 2.1 | 0.08 [0.06, 0.11] | — | 0.248 | 0.275 | 0.145 | 0.059 | — |
| NAE-MSE, λ 2 | 10.1 [6.7, 13.6] | 8.5 | 1.6 | 0.16 [0.09, 0.26] | −0.63 [−0.84, −0.47] | 0.234 | 0.246 | 0.151 | 0.054 | 0.036 |
| NAE-MSE, λ 8 | 7.1 [3.8, 10.4] | 5.9 | 1.2 | 0.17 [0.08, 0.32] | −0.62 [−0.84, −0.44] | 0.224 | 0.257 | 0.152 | 0.057 | 0.085 |
| NAE-decile, λ 2 | 10.0 [6.7, 13.3] | 9.3 | 0.6 | 0.06 [−0.00, 0.13] | −0.73 [−0.94, −0.57] | 0.257 | 0.268 | 0.143 | 0.057 | 0.048 |
| NAE-decile, λ 8 | 8.4 [5.6, 11.3] | 7.4 | 0.9 | 0.11 [0.05, 0.19] | −0.68 [−0.90, −0.51] | 0.245 | 0.266 | 0.141 | 0.054 | 0.086 |

Distortion is the mean squared change per dimension on test rows, in the raw feature scale.

**Training adversary at the end of fitting.** Capped loss on the last mini-batch, per fold:
- MSE adversary: 0.31–0.69 at λ 2 and 0.50–0.98 at λ 8 (chance 1.0);
- decile adversary: 1.69–1.88 at λ 2 and 1.89–2.09 at λ 8 (chance 2.30).

**Registered prediction (SYN-3/3c):** the nonlinear erasers do not exceed LEACE's η, and they leave more linear leakage than LEACE. **Both parts hold, by a wide margin:**
- η is 0.06–0.17 against LEACE's 0.79, and every difference has a CI excluding 0;
- ridge R² is 0.22–0.26 against LEACE's −0.004.

**Findings.**
1. **The adversarial erasers' cost is about as substitutable as random erasure's.** Their η (0.06–0.17) is in the range of matched random (0.08), not LEACE (0.79). They cost 50–72% of LEACE's image-only cost, and almost none of it is recoverable from age.
2. **They remove less age by most probes.** Linear R² falls only from 0.325 to 0.22–0.26. HGB within-class R² is 0.25–0.27, against LEACE's 0.253. Decile gain is higher than after LEACE. Only |A − median| R² is marginally lower (0.141–0.152 against 0.161).
3. **Fooling the training adversary is not removal.** At λ 8 the training adversaries end near chance in some folds (MSE up to 0.98, decile up to 2.09). Fresh held-out probes still decode age with ridge R² 0.22–0.26. This is the gap between adversarial and post-hoc probes that Elazar & Goldberg (2018) reported for text (01).
4. **The metadata-aware objective**, fused cost L against leakage:
   - LEACE dominates every NAE arm on L and linear leakage together (L 2.9 against 5.9–9.3).
   - On HGB within-class R², only NAE-MSE λ 2 is lower than LEACE (0.246 against 0.253), and its L is 8.5.

**Caveats.**
- One development set and one backbone.
- Two λ values per adversary, and no task term (as registered).
- No distortion-matched sham.
- Nonlinear training is seed-dependent, and one seed per fold was used.

## 5. REAL-6 — n/d scaling and the PCA remedy

**Setup.**
- ISIC 2019 (non-HAM), ConvNeXt-T (d = 768).
- Outer-training rows were subsampled by whole groups to n_train/d = q ∈ {1.25, 2, 5, 10}, with 3 subsample seeds. The test folds were kept whole.
- The full-data reference is q ≈ 16 (N = 15,316): η_T 0.97 [0.64, 1.55], η_R 0.03, C 7.5. There the random arm costs 1.5× LEACE.
- **PCA variant:** PCA(k) fitted on training rows, then LEACE, with k = max(8, q·d/10) = 96, 153, 384 and 768. At q = 10, k = d, so the PCA arm is an orthogonal rotation of the raw features.

| q | variant | coordinate SD ratio | η_T (3 seeds) | η_R | random/LEACE cost | η_T − η_R p (3 seeds) |
|---|---|---|---|---|---|---|
| 1.25 | raw | 1.81–2.04 | 0.56, undefined (C −0.7), 0.63 | 0.06–0.12 | 6.4–10× | 0.0005, 1, 0.0005 |
| 2 | raw | 1.22–1.26 | 0.59, 0.69, 0.47 | 0.07–0.11 | 3.2–6.6× | 0.008, 0.004, 0.009 |
| 5 | raw | 1.03–1.05 | 0.80, 1.02, 0.80 | 0.03–0.04 | 2.5–2.8× | ≤ 0.002 |
| 10 | raw | 1.01–1.02 | 1.01, 1.04, 0.91 | 0.03–0.04 | 1.8–2.0× | 0.0005 |
| 1.25 | PCA (k 96) | 0.91–0.93 | 0.69, 0.37, 0.61 | 0.05–0.11 | 1.3–1.8× | 0.0005, 0.17, 0.0005 |
| 2 | PCA (k 153) | 0.94–0.97 | 0.24, 0.35, 0.80 | −0.01 to 0.12 | 0.8–3.0× | ≤ 0.009 |
| 5 | PCA (k 384) | 0.98–0.99 | 0.12, 0.80, undefined | −0.03 to 0.06 | 0.8–4.7× | 0.026, 0.004, 1 |
| 10 | PCA (k 768) | 1.01–1.02 | 0.77, 0.93, 0.96 | 0.01–0.05 | 1.7–2.4× | ≤ 0.001 |

η_T CI widths: raw 0.51–0.60 at q = 1.25, 0.66–1.10 at q = 2, 0.77–2.12 at q = 5, 0.76–1.12 at q = 10. They do not shrink monotonically with n, because C is larger at small q.

**Registered prediction (SYN-2):** at small n/d, η estimates and matching become unstable while point estimates of η stay near the full-data value, and PCA narrows the instability.

1. **Matching instability: holds.**
   - Per-fold matched ranks range from 1 to 512 at q ≤ 2.
   - The random arm costs 6–10× LEACE at q = 1.25 and 3–7× at q = 2, falling to about 2× at q = 10 (1.5× at full data).
   - The matched-cost contrast is uninterpretable at q ≤ 2.
2. **η point estimates stay near the full-data value: refuted at q ≤ 2.**
   - Raw η_T is 0.47–0.69 at q ≤ 2 (5 defined values) against 0.97 at full data, and 0.80–1.04 at q ≥ 5.
   - SYN-2 found the *exact* η of the fitted eraser near its large-sample value in the redundant regime (0.98–1.00 at n/d = 1.25). On real data there is no ground truth, so it cannot be separated whether the drop is in the fitted eraser or in the penalised-head estimate.
   - The direction matches HIBA's positive-control shortfall (§3.1).
3. **PCA narrows the instability: not supported.**
   - PCA fixes the coordinate SD ratio (0.91–1.02 at every q), but η_T across seeds ranges 0.24–0.80 at q = 2 and 0.12–0.80 at q = 5.
   - At q = 5, seed 0, the per-fold image-only costs are 7.5, 130.9, −11.4, 39.6 and 3.1, with L tracking C. The instability is in the heads, not the eraser.
   - A likely mechanism, **not tested**: the heads standardise each feature, so PCA followed by standardisation whitens the k retained components, which changes the penalised head's effective prior.
   - PCA also changes the estimand: substitutability inside a k-dimensional projection, against a different baseline.
   - Even at k = d, a pure rotation, PCA-arm η differs from raw by up to 0.24 (0.77 against 1.01). That is inside the CIs, but it is a measure of preprocessing sensitivity.
4. **The qualitative contrast survives.**
   - η_T − η_R > 0 with p ≤ 0.009 (uncorrected, exploratory) in 11 of 12 raw configurations; the 12th has undetectable C.
   - Small n/d pulls η_T towards η_R, so the bias is conservative for the contrast.
   - Small-set η values (HIBA 0.37, MSK 2020, PAD-UFES-20) are therefore likely biased downward and are not interpreted.

**Deviation.** Within-test leakage, a registered metric, was not computed by real_new.py's ndscale mode. Only the coordinate SD ratio was recorded.

## 6. REAL-7 — attribute specificity with sex, and joint removal

### 6.1 Sex erasure

LEACE on the sex indicator, minimal mode. Sex adds little to diagnosis at t0: Δ_sex is 0.41 on HAM and −0.01 on MILK10k (× 10⁻³ nats).

| Set | C [CI] | η_sex [CI] | η_R | η_T − η_R [CI] (p) | matching check | sham C |
|---|---|---|---|---|---|---|
| HAM10000 | 2.65 [1.02, 4.28] | 0.07 [−0.40, 0.31] | 0.005 | 0.05 [−0.39, 0.30] (0.35) | −0.87 [−2.63, 1.03] | 0.22 [−0.04, 0.47] |
| MILK10k derm | 13.0 [7.8, 18.2] | 0.13 [−0.07, 0.32] | 0.003 | 0.12 [−0.06, 0.31] (0.086) | −0.59 [−4.74, 3.46] | −2.0 [−3.9, −0.1] |

**What this shows.**
- Erasing the linear sex direction costs little, and recorded sex recovers none of that cost detectably.
- Matching passes on both sets.
- The MILK10k sham is slightly favourable, with a CI excluding 0.
- **No sex-decodability probe was computed** (the E51 audit probes age and site only). Out-of-sample removal of sex information is therefore not verified.

### 6.2 Specificity matrix

Each cell is the targeted − matched contrast in the age, site or sex gain, × 10⁻³ nats, with 95% CIs. Sources: E50 (age), E51 (site) and the REAL-7 sex runs.

| Eraser ↓ / gain → | HAM age | HAM site | HAM sex | MILK age | MILK site | MILK sex |
|---|---|---|---|---|---|---|
| age | **9.05 [6.48, 11.70]** | 0.39 [−0.92, 1.69] | 0.53 [−0.20, 1.25] | **9.53 [4.5, 14.2]** | −1.47 [−2.68, −0.36] | 0.23 [−0.48, 0.91] |
| site | 0.55 [−0.41, 1.60] | **4.27 [1.75, 6.86]** | 0.03 [−0.22, 0.30] | −1.15 [−2.59, 0.28] | **3.83 [1.23, 6.31]** | −0.02 [−0.18, 0.13] |
| sex | 0.38 [0.08, 0.70] | −0.31 [−0.68, 0.04] | 0.16 [−0.56, 0.88] | 0.80 [−0.27, 1.88] | 0.71 [−0.10, 1.61] | 1.60 [−0.79, 3.78] |

Matching checks (targeted − matched image log-loss):
- HAM: age −11.4, site −16.5, sex −0.9;
- MILK10k: age −38.7, site −4.5 [−9.9, 0.8], sex −0.6.

The random arms overshoot for age (both sets) and for site (HAM), which biases M contrasts towards 0.

**Leakage specificity**, linear probes from the same runs: ridge within-test R² for age, and site-probe log-loss gain.

| Eraser | HAM age R² (t0 0.325) | HAM site gain (t0 0.622) | MILK age R² (t0 0.284) | MILK site gain (t0 0.472) |
|---|---|---|---|---|
| age | −0.004 | 0.590 | 0.015 | 0.462 |
| site | 0.231 | 0.000 (AUROC 0.44) | 0.268 | 0.000 (AUROC 0.45) |
| sex | 0.275 | 0.613 | 0.254 | 0.460 |

**Findings.**
1. **Substitution is attribute-specific for age and site.**
   - Each eraser raises the gain of its own attribute: age 9.1 and 9.5, site 4.3 and 3.8.
   - Off-diagonal entries are at most 1.5 in magnitude. Only two have CIs excluding 0:
     - HAM, sex eraser → age gain, +0.38. Consistent with the sex eraser removing 15% of linear age R².
     - MILK10k, age eraser → site gain, −1.47. There the random arm overshoots LEACE's cost by 39, so a larger random-arm site gain is expected.
2. **Sex has no power in this design.** Sex is weakly predictive at t0, and erasing it is cheap, so there is little cost to substitute. The diagonal sex entries are not significant, and no specificity claim is made for sex.
3. **The attribute directions overlap, asymmetrically.**
   - Site erasure removes 29% of linear age R² on HAM and 6% on MILK10k.
   - Sex erasure removes 15% and 11%.
   - Age erasure removes about 5% and 2% of site decodability.
   - The below-chance site AUROC after site erasure is the usual linear-probe signature of an oblique projection on held-out rows.

### 6.3 Joint removal of age, sex and site

LEACE on [age, sex, site one-hot] jointly, against single-attribute erasers on identical cases (pairing checked).

| Set | joint rank | joint C [CI] | Σ single C | joint − Σ [CI] | η_all, joint [CI] | η_all, matched random | η by block (age / site / sex) |
|---|---|---|---|---|---|---|---|
| HAM10000 | 15 | 40.4 [34.1, 46.5] | 49.2 (14.1 + 32.5 + 2.7) | −8.8 [−11.7, −5.8] | 0.42 [0.32, 0.53] | 0.15 [0.11, 0.18] | 0.30 / 0.19 / 0.03 |
| MILK10k derm | 6 | 99.9 [88.8, 110.2] | 121.6 (94.4 + 14.2 + 13.0) | −21.8 [−30.1, −13.0] | 0.62 [0.55, 0.70] | 0.40 [0.35, 0.44] | 0.58 / 0.09 / 0.04 |

**Interaction in M** (joint M minus the sum of the single erasers' M, for the same gain block):
- age block: −4.4 [−6.1, −2.6] (HAM), −5.2 [−7.5, −2.6] (MILK10k);
- all metadata: −4.4 [−6.6, −2.2] and −6.1 [−9.8, −2.0];
- site: −0.8 [−2.7, 1.1] and −7.1 [−15.1, −1.2];
- sex: CIs include 0.

The joint matched random arm overshoots (C 67.4 on HAM and 133.1 on MILK10k).

**Age leakage after joint erasure:**
- HAM: ridge −0.001; HGB within-class 0.203 (60% of t0; age-only LEACE 0.253); |A| 0.127; decile 0.037.
- MILK10k: ridge 0.019; within-class 0.090; |A| 0.138; decile 0.021.

**Findings.**
1. **Sub-additive.** Joint removal costs 18% less than the sum of single removals on both sets, and the substitutable parts are sub-additive too. The attribute subspaces overlap (§6.2), so single-attribute analyses double-count shared information.
2. **Removing all recorded metadata is much less substitutable than removing age alone, when site matters.** On HAM, η_all is 0.42 against age-only 0.79. Site dominates the joint cost there (single site C 32.5, η 0.21; 06 §7). On MILK10k age dominates (94 of 100), and η_all (0.62) is close to age-only (0.60).
3. **The joint-versus-random gap remains** (0.42 vs 0.15; 0.62 vs 0.40), with the overshoot bias in the conservative direction (06 §3).
4. **Joint erasure (rank 15) removes more non-linear age information than rank-1 LEACE**, but still leaves 60% of HAM's within-class HGB R².

## 7. REAL-8 — cross-dataset transfer

**Setup.**
- A LEACE age eraser is fitted on a source set and applied to the target set's ConvNeXt-T features.
- Target heads are fitted on target outer-training rows only.
- T-own is the target's own fold-wise eraser (identical to E50 LEACE; pairing checked).
- **Cosine** compares the age-covariance directions Cov(S, A) of source and target (per target training fold). These are the directions along which LEACE subtracts.

| Source → target | cos | target C, transfer [CI] | M, transfer [CI] | ridge R² after transfer (t0) | HGB within-class after (t0) | own C / η / ridge R² |
|---|---|---|---|---|---|---|
| HAM10000 → ISIC 2019 | 0.33–0.35 | 1.0 [−0.06, 2.0] | 0.60 [0.34, 0.85] | 0.306 (0.323) | 0.328 (0.346) | 7.5 / 0.97 / 0.013 |
| ISIC 2019 → HAM10000 | 0.32–0.34 | −0.04 [−0.73, 0.68] | 0.14 [0.04, 0.25] | 0.312 (0.325) | 0.341 (0.337) | 14.1 / 0.79 / −0.004 |
| HAM10000 → ISIC 2020 | 0.50–0.57 | −0.20 [−0.84, 0.49] | 0.26 [0.13, 0.38] | 0.192 (0.226) | 0.240 (0.268) | 1.70 / 0.71 / −0.034 |

**Findings.**
1. **A LEACE age eraser does not transfer between dermoscopy datasets in the same frozen space.**
   - On the target it removes 4–15% of the linearly decodable age (ridge R² falls by 0.013–0.034). The target's own eraser removes all of it.
   - Its image-only cost is not detectable (every C CI includes 0), so η is undefined.
   - The small positive M (0.14–0.60) shows that it removes a sliver of age-related signal, which age then substitutes.
2. **The datasets' age directions share structure but are far from identical.** The cosine is 0.32–0.57. Two random directions in 768 dimensions have an expected |cos| of about √(2/(π·768)) ≈ 0.03.
3. **Consequence.** Linear guardedness, and any η measured with it, belongs to an eraser *and the distribution it was fitted on*. "The age direction of the encoder" is not a dataset-free object, and an eraser for deployment would have to be fitted on the deployment distribution.

**Deviation.**
- The registry specified fitting the source eraser on the source's outer-training rows. The code fitted it on all source rows with age.
- No diagnostic labels enter the eraser, and source and target are different datasets, so no target-test label information can leak.
- Patient overlap between datasets, which would let target-test images inform the eraser, cannot be checked: there are no shared patient identifiers. ISIC 2019 (non-HAM) excludes HAM images by construction.

## 8. What REAL-4 to REAL-8 change about the main claim

- **Strengthened.** The estimator recovers η = 1 on every large set (positive control). So LEACE's η values of 0.6–0.97 there are measurements of the removed information. Notably, MILK10k's 0.60 is a real property of the data.
- **Narrowed.** η is a statement about the *linear* age direction:
  - non-linear age information largely survives LEACE (55–88% of within-class HGB R²);
  - the eraser does not transfer across datasets;
  - adding site to the erasure target drops substitutability sharply (HAM η_all 0.42).
- **Method question (07).** Attribute-only adversarial erasure gives random-like η (0.06–0.17) with more leakage than LEACE, which confirms the decision not to propose MATE.
- **Small n/d.** η is biased towards the random arm (0.47–0.69 against 0.97), and PCA does not fix it. Small-set values stay stress tests only.
