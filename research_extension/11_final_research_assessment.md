# 11 — Final research assessment

*Phase 7, written 2026-11-02 after all scheduled runs finished (2026-11-02 00:28 IST). It synthesises 00–10; every number cites the section that reports it with its CI. Together, 00–11 are the technical report the directive asks for (§14). No manuscript draft has been written.*

## 1. Bottom line

**Is there a scientifically and mathematically defensible contribution? Yes, but a modest one, and it is not a method.** It sits at the directive's Level 3: characterising when recorded metadata can substitute for erased representation information. Its four parts:

1. **A framing identity, not a theorem.** For any transformation Z = T(S) that does not use A at inference, the gain in age's incremental value is M = C − L ≤ C (04 Identity 3). Age can become more useful only by as much image information as is lost. What varies is the *substitution efficiency* η = M/C. The algebra is the chain rule, and it is not claimed as new.
2. **A measurement design** that estimates η for an attribute-free eraser against:
   - performance-matched random subspaces;
   - permuted-attribute shams;
   - re-append restoration;
   - an oracle positive control with known η = 1.

   Everything is strictly cross-fitted, with grouped paired bootstraps (06 §1, 10).
3. **An empirical result, confirmed on a representation never used for erasure before** (REAL-2, ViT-L/16, registered in advance). On large dermoscopy datasets, the diagnostic cost of removing the linear age direction is mostly recoverable from recorded age (η 0.83–1.05), while the cost of matched random erasure is not (η ≤ 0.10). Holm p = 0.0015 on 3/3 sets (08 §1). It holds:
   - across 4 deep encoders and 3 fold assignments;
   - with Ledoit–Wolf whitening;
   - against a positive control that recovers η = 1 on every large set (06 §8, 08 §2, 08 §3.1).
4. **Mapped boundaries**, each measured rather than assumed:
   - **linear only:** 55–88% of non-linear within-class age information survives (08 §3.2);
   - **dataset-specific:** erasers do not transfer (08 §7);
   - **age, not site:** site erasure is 21% substitutable, and joint metadata erasure 42% on HAM (06 §7, 08 §6.3);
   - **large n/d only:** η is biased towards the random arm at n/d ≤ 2 (08 §5);
   - **weaker when age strongly predicts the label:** MILK10k gap 0.25 (06 §2).

**What it is not:**
- not a Level 2 contribution: no new eraser improves the three-objective frontier, and the one family tested on real data was worse (08 §4);
- not causal;
- not evidence of an "age-free" representation;
- not yet independently replicated on new patients.

## 2. The directive's eight questions (§12)

### 2.1 What is genuinely novel (conservatively)

Novelty is judged against the searches in 03 §5. Absence from searches is weak evidence.

| Item | Why it counts | Strength |
|---|---|---|
| Using the substitution efficiency η of an **attribute-free** eraser, against matched random erasure, sham, restoration and an oracle control, as an estimate of how redundant a representation's attribute trace is with the recorded attribute | Not found as a measured outcome in the erasure, fairness or metadata-fusion literature (03 §3, "partially explored") | Design novelty. The parts are standard; the combination and the estimand are not |
| The side-information form of the minimal-leakage result: least leakage compatible with L = 0 is I(A; 𝔐(S)), with metadata enlarging the feasible erasure set (04 Proposition 3, Corollary 5) | PEF and Simai et al. treat the task-agnostic H(S\|Z,A) = 0. The task-specific, metadata-aware form was not found stated | Elementary (minimal sufficiency); a framing result |
| The empirical fact, confirmed in advance on ViT-L/16, that frozen dermoscopy encoders' linear age direction is ~80–100% substitutable by recorded age while matched random erasure is ≤ 10% | New measurement | Confirmatory on a new representation; **not** on new patients (10 §3) |
| An empirical map of where the design fails, with diagnostics: n/d (SD ratio, positive-control failure, matched-rank instability), label optimism of label-using erasers, sham non-inertness on small sets | The failure modes are known in general; their effect on η and the diagnostics are not documented elsewhere (03 §3) | Methodological observation |

### 2.2 What is already known (cite; do not claim)

- **Erasure itself:**
  - linear guardedness and the closed-form eraser (LEACE), and RLACE's closed form for regression;
  - linear erasure leaves non-linear information; adversarial training does not certify removal (Ravfogel et al. 2022; Elazar & Goldberg 2018). REAL-4 and REAL-5 reproduce both on dermoscopy features rather than discovering them;
  - nonlinear and task-preserving erasers (Obliviator, LEOPARD, FaRM, conditional autoencoders).
- **Recoverability and residualisation:**
  - maximal retention requires recoverability given A (PEF Lemma 3.3; Simai et al. 2026; Moyer et al. 2018);
  - attribute-conditional residualisation gives η = 1 trivially (Frisch–Waugh–Lovell; Metadata Normalization; Weber et al.);
  - the independence-versus-utility trade-off (Zhao & Gordon 2019).
- **Medical images and log-loss:**
  - medical images encode age, sex and race (Poplin 2018; Gichoya 2022; Brown et al. 2023);
  - log-loss reduction equals conditional mutual information for Bayes predictors.

Full table: 03 §3.

### 2.3 What is only an empirical observation (exploratory unless stated)

| Observation | Where | Status |
|---|---|---|
| η_LEACE 0.6–1.05 vs η_random ≤ 0.10 on large sets; the gap shrinks as age–label dependence rises (0.94 → 0.25) | 06 §2 | ViT-L/16 confirmatory; rest exploratory |
| Stronger encoders encode about as much linear age, pay a lower erasure cost, and keep η; the entangled share (1 − η)·C on HAM falls from 6.9 (ResNet-18) to 1.6 (ViT-L) | 08 §2 | Exploratory; no trend test |
| Erasing one linear direction returns age's added value close to its colour-histogram level (HAM ConvNeXt-T: 2.0 + 11.2 ≈ 13.2 against 14.7 at HSV; ISIC 2019: 3.2 + 7.2 ≈ 10.4 against 11.4) | derived from 08 §2 and 06 §2 | **Post hoc**, point estimates; noticed while writing this assessment |
| MILK10k's lower η (0.60) is a property of the data: the oracle recovers 1.01 | 08 §3.1 | Exploratory |
| Site erasure is costly and mostly non-substitutable (HAM: C 32.5, η 0.21); joint age + sex + site erasure is sub-additive (−18%) and 42% substitutable on HAM | 06 §7, 08 §6.3 | Exploratory |
| Attribute-only adversarial erasure has random-like η (0.06–0.17), with more linear leakage than LEACE | 08 §4 | Exploratory; registered prediction confirmed |
| LEACE erasers do not transfer across datasets (they remove 4–15% of the target's linear age R²) | 08 §7 | Exploratory; deviation noted |
| A label-using eraser scored on its own training rows looks cheap, even beneficial, but is costly out of sample (HIBA −54.5 vs +285) | 06 §6 | Exploratory; mechanism identified in code; seen on two sets |
| η is biased towards η_R at n/d ≤ 2; PCA does not fix it | 08 §5 | Exploratory; registered prediction partly refuted |

### 2.4 What is mathematically established (all elementary; 04, notebook 01)

- **Identity 3.** M = C − L, so M ≤ C, for transformations that do not use A at inference (Bayes predictors, log-loss).
- **Proposition 1.** The ceiling holds for restricted predictor families closed under the transformation. Penalised linear heads are not closed under non-orthogonal maps, which explains the D1-dose costs.
- **Propositions 2–3 and Corollary 5.** These give minimum leakage under image-only versus fused preservation. Image-only preservation forces leakage ≥ I(A; P(Y|S)).
- **Proposition 4.** Linear erasers with the same kernel are information-equivalent (verified numerically to 10⁻¹³).
- **The E11 formula.** Under MCAR missingness at rate π, added risk = π·C + (1 − π)·L = C·[1 − (1 − π)·η] (06 §5). This is linearity, nothing more.

### 2.5 What is not supported

| Claim | Status | Evidence |
|---|---|---|
| A transformation can make age more useful without image-only cost | **Impossible** (population) | Identity 3 |
| A MATE-type method improves the three-objective frontier | No evidence; the real-data audit points the other way | 07 §3–5; 08 §4 |
| LEACE produces age-free features | **False** for non-linear probes | 08 §3.2 |
| The fused cost L is negligible ("free given age") | Not tested: no equivalence margin; largest upper limit 6.0 (HAM) is 3× age's own value | 10 §5 |
| Foundation models entangle age with diagnosis more than weak models | Not supported; descriptively the opposite | 08 §2 |
| An eraser fitted on one dataset protects another | Not supported | 08 §7 |
| PCA rescues small datasets | Not supported | 08 §5 |
| Specificity for sex | No power | 08 §6.1 |
| 1 − η measures photo-ageing or any biological process | Not tested; age is entangled with site, exposure and referral | 09 §8 |
| Generalisation beyond dermoscopy, beyond frozen encoders and linear heads, or to new patients | Not tested | 09 §8; 10 §8 |

### 2.6 Experiments essential for publication

Ordered by importance. Costs are on this machine (Apple M5, 24 GB), from observed run times where available.

| # | Experiment | Why essential | Cost |
|---|---|---|---|
| 1 | **External confirmatory replication.** A dataset untouched by this project, with age, diagnosis and patient IDs, and n_train/d ≥ 5. Register a REAL-2-style test, plus one-sided η_T > 0.5, before extracting its features. Candidate: ISIC 2024 (SLICE-3D); its metadata fields, label balance and licence must be checked first | REAL-2 replicates on a new representation, not on new patients (10 §3). This is the largest gap | Feature extraction dominates (time it on a pilot). Erasure runs take about 20–40 min per set (REAL-2: 19–37 min) |
| 2 | **Leakage audit with uncertainty.** Bootstrap or repeated-seed HGB probes for t0 / LEACE / random on the 4 large sets | REAL-4 leakage values have no CIs (10 §9). The "linear only" boundary rests on them | CPU, about 2–4 h (REAL-4 took 4–22 min per set per single fit) |
| 3 | **Refit variance.** 5–10 fold assignments on HAM10000 and ISIC 2019 (ConvNeXt-T, minimal mode) | The bootstrap conditions on one fit. Only 3 assignments exist (10 §2) | CPU, about 3–6 h |
| 4 | **A pre-specified equivalence margin for L**, tested on the external set | Turns "L's CI includes 0" into a claim, or refutes it | No extra compute; needs a justified margin (10 §5) |
| 5 | **One fine-tuned encoder (E9)**, split-cross-fitted, on HAM10000 or ISIC 2019 | Reviewers will ask whether frozen-encoder results survive fine-tuning | GPU about 3–5 h per dataset and backbone (registry), plus about 30 min of erasure runs |

### 2.7 Optional

- A registered trend test along the representation ladder.
- A second development set for the NAE audit, or task-preserving variants (TP-img, TP-fused) on real data. Low expected value (07 §3, §5).
- Further attributes with real diagnostic signal, such as Fitzpatrick type where recorded.
- Quantifying the bias from 5-year age bins, using a large exactly-aged set (none available here).
- MAR/MNAR missingness for E11.
- A non-linear site probe.
- A sex-decodability probe.

### 2.8 Remove from the research scope

- **MATE method development (Level 2).** No mechanism, no synthetic advantage, and negative real-data evidence (07).
- **D1 partial doses.** They are invertible maps whose cost is a restricted-family artefact (06 §4).
- **PCA as a small-sample remedy** (08 §5).
- **Small sets (HIBA, MSK 2020, PAD-UFES-20) as evidence.** Keep them as a stress-test appendix only (09 §2).
- **Sex as a specificity test** (no power).
- **The matched-cost M contrast as a headline.** Random overshoots, so report η instead (06 §3).
- **Any wording implying age-free features, portable erasers, causal mechanisms or clinical utility.**

## 3. Contribution levels (directive §12)

| Level | Assessment |
|---|---|
| **1.** Apply existing erasers and measure metadata's incremental value | Done, and done carefully, with controls, but a pure Level 1 paper would be incremental on its own |
| **2.** Task-preserving transformation with a better three-objective frontier | **Investigated and not achieved.** The ceiling (Identity 3) and Proposition 3 fix the frontier. LEACE sits on it in the linear-Gaussian case (SYN-1). Non-linear and task-aware variants did not beat it in synthetic data (SYN-3c), and on real data attribute-only adversarial erasure was worse (REAL-5) |
| **3.** Characterise when metadata can substitute for erased information | **This is where the contribution is.** The identity and sufficiency conditions (04) are elementary but correct and correctly attributed. The measurement design is validated by synthetic ground truth (05) and a real positive control (08 §3.1). The empirical characterisation is confirmed on an untouched representation and bounded by measured limits (08, 09) |

## 4. Relation to the parent paper

The parent paper ("Do Patient Metadata Still Help Strong Image Models?") reports that age's incremental value attenuates as image models strengthen. This extension adds a mechanism-level description, still predictive rather than causal:
- the strong encoders carry the age-substitutable information in a linearly removable direction;
- removing that one direction gives age back most of its value (η ≈ 0.8–1.0);
- age's added value then returns close to the colour-histogram level (§2.3, post hoc).

Within this protocol, the extension also reproduces the attenuation (08 §2, Δ_A from 11–15 at HSV to 2–3.5 at deep rungs).

**Recommendation.** The most defensible near-term use is a mechanism section, or a companion analysis, in the parent paper, with the framing identity, the design and the boundaries stated as above. A standalone paper needs essential experiments 1–3 first.

## 5. Publication potential

- **As a methods paper at a major ML venue:** low. There is no new method, the identity is the chain rule, and the boundary findings (linear-only erasure, adversary ≠ removal) are known in the erasure literature.
- **As a measurement and characterisation paper in medical image analysis:** moderate, conditional on the external replication (§2.6 item 1). Its strengths are rare in this area:
  - a registered confirmatory test;
  - matched controls and a positive control;
  - nested evaluation;
  - documented failures.

  Its weaknesses are one modality, essentially one attribute, frozen encoders only, and so far no new patients.
- **Today, without further experiments:** suitable as an extension of the parent paper or as a workshop-length contribution. The exploratory/confirmatory distinction must be stated as in 10 §3.

## 6. Coverage of the directive's final-report items (§14)

| # | Item | Where |
|---|---|---|
| 1 | Full literature review | 01 |
| 2 | Paper-by-paper mathematical comparison | 02_literature_comparison.csv; 01 |
| 3 | Novelty audit | 03; §2.1–2.2 here |
| 4 | Mathematical derivations | 04; notebooks/01_information_theory_verification.ipynb |
| 5 | Synthetic experiment results | 05 |
| 6 | Baseline comparisons | 06 |
| 7 | MATE method, if justified | 07 (not justified; reasons and real-data audit) |
| 8 | Three-objective Pareto plots | figures/SYN-1_pareto_exact, figures/REAL-1_pareto |
| 9 | Attribute-specific experiments | 06 §7; 08 §6 |
| 10 | Non-linear leakage analysis | 05 SYN-3/3b/3c; 08 §3.2, §4 |
| 11 | Representation-strength analysis | 08 §2; figures/REAL-1_eta |
| 12 | Robustness and failure analysis | 09 |
| 13 | Statistical significance and uncertainty | 10 |
| 14 | Reproducibility details | RESULTS_INDEX.md (config, script hash, raw results per item); configs/; EXECUTION_LOG.md; EXPERIMENT_REGISTRY.md |
| 15 | Limitations | 09 §8; 10 §8–9 |
| 16 | Publication potential | §5 here |

## 7. Reproducibility summary

- **Code.** Every real-data result names its code version (RESULTS_INDEX.md): e50_dose_restoration.py 9915166e192a → 4631999e6079 → ca82ac760e14, and real_new.py 5c5f8074d8a3. real_new.py reproduced E50a's per-image losses exactly (maximum difference 0.0).
- **Seeds.**
  - folds: StratifiedGroupKFold random_state 0, 1, 2;
  - bootstrap: 601 for every re-analysis (real_reanalysis.py, real_new_analysis.py, real_e11.py). The targeted − matched contrasts read directly from the E50/E51 JSONs, including the 08 §6.2 specificity matrix, come from e50_dose_restoration.py's own bootstrap (seeds 50 and 501 in that script);
  - random subspaces: 51000 + 100k + j (E50 and real_new.py);
  - shams: 53000 + 100k + j (E50; real_new.py uses j = 0);
  - REAL-6 subsamples: 70000 + 100·4q + rep.
- **Execution.** Jobs ran sequentially through a logged runner. Every start and end time comes from the system clock (EXECUTION_LOG.md); one estimated timestamp was corrected openly.
- **Registration.** Every experiment's hypothesis and protocol were written in EXPERIMENT_REGISTRY.md before the run. Post-hoc analyses are labelled post hoc (10 §9).

## 8. Open work, in order

1. External replication: choose the dataset, register, then extract features.
2. Leakage CIs.
3. Refit variance.
4. Equivalence margin.
5. E9 on one set.
6. Then decide between a parent-paper section and a standalone manuscript.
