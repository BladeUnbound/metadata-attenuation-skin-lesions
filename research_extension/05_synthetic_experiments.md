# 05 — Synthetic experiments with known ground truth

*Phase 3. Registry entries SYN-1, SYN-2 and SYN-3 were written before each run (EXPERIMENT_REGISTRY.md).*

*Code:*
- *scripts/synth.py (SYN-1/2);*
- *scripts/synth3.py and scripts/nonlin.py (SYN-3);*
- *scripts/gauss_logit.py (exact risks);*
- *scripts/analyse_syn.py (tables, figures).*

*Configs: configs/SYN-*.json (seeds, grids, code hashes). Raw results: results/synthetic/SYN-*.csv, with summaries alongside. All values in nats; tables show ×10⁻³ where stated.*

## Design

**Latent-Gaussian, logistic-label model.**
- Image channels: c_age = A + 0.5U (U is image-specific "age-like" variation), c_diag = G, and d/4 nuisance factors.
- These are mixed into S ∈ ℝ^d by a random orthonormal map, plus isotropic noise.
- The label is Y ~ Bernoulli(σ(−1 + β_A A + β_U U + β_G G)).

For any *linear* transformation fitted on a finite sample, the exact population values of C_Y, M_A, L_YA, η and the leakage R²(A | Z) are computed by quadrature (04 §7). This separates three things:
- what the fitted transformation removes;
- what the estimators report;
- sampling noise.

Estimates use held-out log-loss of logistic heads, either near-unpenalised (C = 10³) or E40-style penalised (C chosen from {10⁻⁴, …, 1} by 3-fold CV).

**Regimes.** The directive's regimes 1–3 and two that 04 §6 identified:
- R1 redundant (β_U = 0);
- R2 entangled (β_U = 0.8);
- R3 irrelevant (β_A = β_U = 0);
- R6 strong age–label (β_A = 2.5, β_G = 0.5);
- R7 shared component (corr(A, G) = 0.6, β_A = 0).

The directive's regime 4 (nonlinear encoding) is SYN-3.

## SYN-1 — does the pipeline recover the truth, and do the metrics separate the regimes?

Exact population quantities of the *fitted* erasers, at n_train = 10,000, d = 128 (mean of 5 seeds; ×10⁻³ nats except η and R²):

| Regime | Method | C_Y | M_A | L_YA | η | leakage R²(A\|Z) |
|---|---|---|---|---|---|---|
| R1 redundant | LEACE | 45.3 | 45.3 | 0.0 | **1.000** | 0.012 |
| | matched random | 74.0 | 30.9 | 43.2 | 0.42 | 0.17 |
| | oracle residual (uses A) | 45.3 | 45.3 | 0.0 | 1.000 | — |
| R2 entangled | LEACE | 82.5 | 60.5 | 21.9 | **0.734** | 0.013 |
| | matched random | 126.6 | 54.2 | 72.4 | 0.43 | 0.015 |
| | oracle residual | 64.4 | 64.4 | 0.0 | 1.000 | — |
| R3 irrelevant | LEACE | 0.0 | 0.0 | 0.0 | undefined (C = 0) | 0.012 |
| R6 strong age–label | LEACE | 157.1 | 157.1 | 0.0 | 1.000 | 0.013 |
| | matched random | 164.1 | 152.3 | 11.9 | **0.928** | 0.009 |
| R7 shared component | LEACE | 34.7 | 28.1 | 6.5 | 0.811 | 0.013 |
| | matched random | 64.6 | 27.0 | 37.6 | 0.42 | 0.004 |

Shams (LEACE on permuted A) and rank-1 random subspaces have C_Y and M_A below 2 × 10⁻³ in every regime. Their η is undefined (it is a ratio of noise).

### Findings
1. **Registered predictions (a)–(d) hold.**
   - (a) LEACE's η is 1.00 in the redundant regime, 0.73 in the entangled regime (where the image's age direction also carries the diagnostic U), and 0.81 when age shares a component with the diagnosis. C = M = 0 when age is irrelevant.
   - (b) LEACE, Ledoit–Wolf LEACE and orthogonal removal have identical exact quantities in every configuration (Proposition 4).
   - (c) Matched random erasure has η ≈ 0.42.
   - (d) The oracle (attribute-conditional) residual has η = 1 everywhere.
2. **The metrics separate redundancy removal from destruction.**
   - Destruction (random) produces C_Y > M_A with η ≈ 0.4.
   - Redundancy removal (LEACE in R1) produces C_Y = M_A.
   - In the entangled and shared-component regimes the same eraser shows intermediate η, measuring exactly the non-substitutable share.
   - Irrelevant age gives C_Y = M_A = 0: leakage falls and nothing else moves.
3. **Unexpected: in R6 (age dominates the label), random erasure has η = 0.93.** When most of the image's diagnostic information is age-related, almost anything erased is replaceable by age. The *contrast* between targeted and random erasure then nearly vanishes (M_A 157 vs 152), even though targeted erasure is perfectly substitutive. The design is uninformative on datasets where age dominates the label, and this gives a principled account of why HIBA cannot separate targeted from random erasure (E50 §4). This explanation was not in the registry.
4. **Directive Regime 1 restated.** In R1 the expectation "leakage falls, diagnostic performance preserved, age more useful" is satisfied by *fused* diagnostic performance (L_YA = 0), never by image-only performance (C_Y = M_A > 0). This is exactly Corollary 3 of 04.
5. **Estimator accuracy (prediction (e)).** For LEACE, matched random and oracle together, the mean absolute error of the penalised-head estimate of M_A against the exact value is:

   | n_train | d = 32 | d = 128 |
   |---|---|---|
   | 500 | 9.3 × 10⁻³ | 11.4 × 10⁻³ |
   | 2,500 | 4.6 × 10⁻³ | 4.3 × 10⁻³ |
   | 10,000 | 1.9 × 10⁻³ | 1.5 × 10⁻³ |

   Mean bias is at most 7.8 × 10⁻³. Near-unpenalised heads are as good at n ≥ 2,500, and fail at n = 500, d = 128 (MAE 106 × 10⁻³).
6. **Small samples: what the probes miss.** At n_train = 500, d = 128, fitted LEACE still has exact η ≈ 1.00, but 19% of linear age information survives in the population (exact R² = 0.186), because the estimated covariance direction is noisy. The within-test ridge and MLP probes report *negative* R² there (−0.21 and −0.46): they fail at that sample size. **A probe R² ≈ 0 after erasure on a small dataset is not evidence that erasure succeeded.** This bears directly on E50's small-set probe values (HIBA −0.06).

Figure: figures/SYN-1_pareto_exact.{pdf,png}. Rows show leakage vs C_Y, leakage vs M_A, and C_Y vs M_A with the ceiling M = C. LEACE's markers are hidden under the orthogonal eraser's because their values are identical.

## SYN-2 — n/d instability and the head–geometry hypothesis

Regimes R1 and R2, d ∈ {64, 256, 768}, n_train/d ∈ {1.25, …, 25}, 5 seeds, test n = 10,000.

**The registered hypothesis H-S2 is not supported.**
- Exact quantities of LEACE, LW-LEACE and orthogonal removal coincide (maximum gap 3 × 10⁻¹⁶).
- Under penalised heads the measured cost differs by a mean of 0.08 × 10⁻³ nats (largest single configuration 17 × 10⁻³).
- That gap is uncorrelated with LEACE's coordinate SD ratio (r = 0.085).
- **LEACE's oblique geometry is therefore not what made the E50 small-set contrasts fail.**

What does change as n/d falls (mean over seeds; values nearly identical across d):

| n_train / d | 1.25 | 1.5 | 2 | 3 | 5 | 10 | 25 |
|---|---|---|---|---|---|---|---|
| cos(estimated, true Cov(S,A) direction) | 0.79–0.83 | 0.83–0.85 | 0.85–0.88 | 0.90 | 0.93 | 0.97 | 0.99 |
| exact leakage R²(A\|Z) after LEACE (before: 0.67) | 0.34–0.37 | 0.30–0.34 | 0.27–0.29 | 0.22 | 0.15–0.16 | 0.08–0.09 | 0.04 |
| LEACE coordinate SD ratio | 1.44–1.55 | 1.21–1.27 | 1.08–1.15 | 1.03–1.05 | 1.01–1.05 | 0.98–1.01 | 0.99–1.01 |
| exact η of fitted LEACE, R1 | 0.98–1.00 | 0.98–1.00 | 0.97–1.00 | 0.99–1.00 | 0.99–1.00 | 1.00 | 1.00 |
| exact η of fitted LEACE, R2 | 0.60–0.61 | 0.61–0.63 | 0.63–0.65 | 0.67 | 0.69 | 0.72 | 0.73 |

### Findings
1. **At small n/d the fitted eraser removes the wrong amount, not the wrong kind of information.**
   - The estimated direction is noisy. Out of sample, LEACE removes only about 45% of linear age information at n/d = 1.25, against about 94% at n/d = 25.
   - What it does remove remains substitutable: η stays near its large-sample value (1.00 in R1, 0.60–0.73 in R2).
   - The coordinate SD ratio (1.4–1.5 at n/d = 1.25, as on the small real sets) is a symptom of the noisy fit, not of a geometric distortion.
2. **The measurements fail before the eraser does.**
   - Penalised-head estimates of η scatter widely at n/d ≤ 2, from −25.7 to 0.6 where the exact value is about 1.0.
   - The matched-random search, run on training rows only, often selects a control whose true cost differs substantially from LEACE's, in either direction.
   - Both mirror the E50 small-set pattern: matching checks failed both ways, with large, unstable contrasts.
3. **Limitation of the synthetic matched search.** It uses the doubling grid {1, 2, 4, …} without E50's √2 refinement, so even at large n/d it typically overshoots LEACE's cost (for example R1, d = 64, n/d = 10: exact cost 96 vs 44 × 10⁻³). η comparisons, being ratios, are unaffected in kind. Contrasts of M_A at "matched cost" in SYN-1/2 are not exactly matched, and the real-data phase keeps E50's refined search.

**Implication for the real data.** On HIBA, MSK 2020 and PAD-UFES-20 (n_train/d ≈ 1.1–1.3), the E50 failures are best read as measurement failures. Three factors contribute:
- noisy matching;
- noisy Δ estimates;
- incomplete erasure, invisible to small-sample probes;
- plus, for HIBA and MSK 2020, age dominating the label (SYN-1 finding 3).

They are not evidence for or against substitution. Remedies to test in Phase 4:
- (i) reduce d before erasure (PCA fitted on training rows), raising n/d;
- (ii) evaluate η rather than the matched contrast;
- (iii) report the exact-truth caveat that leakage probes cannot certify erasure at these sizes.

Figure: figures/SYN-2_n_over_d.{pdf,png}.

## SYN-3, SYN-3b, SYN-3c — nonlinear encoding, nonlinear and task-aware erasers, adaptive probes

**Design.**
- d = 32. Separate splits: fit 8k, head/probe 8k, test 20k. 3 seeds.
- Regimes:
  - L1/L2: linear redundant / entangled;
  - N1/N2: the same plus two channels A² − 1 and cos 2A, which have zero linear covariance with A.
- Transformations, all attribute-free at inference: residual-MLP erasers trained against an MLP adversary.
  - **NAE:** adversarial only.
  - **TP-img:** plus a diagnosis head on z.
  - **TP-fused:** plus a diagnosis head on [z, A]. This is the MATE candidate 04 §9 supports, targeting L_YA.
- Adversaries:
  - SYN-3/3b: an MSE regressor for signed A.
  - SYN-3c: an age-decile classifier with cross-entropy.
- Heads and probes are fitted on the head split after the transformation is frozen. η and the three quantities use gradient-boosted heads; tuned-logistic heads give the same picture (results/synthetic/SYN-3*.csv).

Values are means of 3 seeds (C, M, L in 10⁻³ nats; leakage R² or the decile log-loss gain as a fraction of log 10). "Before" leakage: ridge 0.66, |A| R² 0.90, decile gain 0.46 (N regimes).

| Regime | Method (λ_A) | ridge R²(A) | \|A\| R² | decile gain | C_Y | M_A | L_YA | η |
|---|---|---|---|---|---|---|---|---|
| L1 linear redundant | LEACE | 0.00 | 0.00 | −0.01 | 42.3 | 42.1 | 0.2 | **0.995** |
| | NAE (8) | 0.13 | 0.00 | 0.00 | 38.5 | 36.0 | 2.5 | 0.94 |
| | TP-fused (8) | 0.14 | 0.00 | 0.01 | 35.6 | 34.0 | 1.5 | 0.96 |
| L2 linear entangled | LEACE | 0.00 | 0.00 | −0.01 | 77.3 | 56.8 | 20.4 | **0.735** |
| | NAE (8) | 0.13 | 0.00 | 0.00 | 67.2 | 45.8 | 21.4 | 0.68 |
| | TP-img (32) | 0.08 | 0.00 | 0.00 | 70.9 | 48.3 | 22.6 | 0.68 |
| | TP-fused (8) | 0.12 | 0.00 | 0.00 | 68.0 | 45.1 | 22.9 | 0.66 |
| N1 nonlinear redundant | LEACE | 0.00 | **0.895** | 0.31 | 46.8 | 47.7 | −1.0 | 1.02 |
| | NAE-MSE (8) | 0.10 | 0.75 | 0.25 | 40.7 | 40.1 | 0.6 | 0.99 |
| | NAE-decile (8) | 0.11 | **0.25** | **0.04** | 43.3 | 41.5 | 1.7 | 0.96 |
| | TP-fused-decile (8) | 0.10 | 0.29 | 0.06 | 44.3 | 42.2 | 2.1 | 0.95 |
| N2 nonlinear entangled | LEACE | 0.00 | 0.895 | 0.31 | 75.0 | 55.3 | 19.7 | **0.738** |
| | NAE-MSE (8) | 0.10 | 0.75 | 0.25 | 61.4 | 44.8 | 16.6 | 0.73 |
| | NAE-decile (8) | 0.11 | 0.25 | 0.04 | 64.8 | 46.3 | 18.4 | 0.72 |
| | TP-fused (2) | 0.17 | 0.69 | 0.26 | 43.3 | 34.3 | 9.0 | 0.79 |
| | TP-fused-decile (8) | 0.12 | 0.32 | 0.06 | 61.1 | 43.6 | 17.5 | 0.71 |
| all | sham NAE (permuted A, λ 8) | 0.60–0.63 | 0.22–0.88 | 0.13–0.43 | 11–17 | 2.7–4.1 | 7–14 | 0.16–0.36 |

### Findings
1. **Linear erasure can look complete when it is not (prediction (a); SYN-3b metric).**
   - After LEACE in N1/N2, every probe of *signed* age is at chance (ridge, MLP, HGB: R² ≤ 0).
   - |A| remains decodable with R² 0.895, essentially unchanged from 0.898.
   - The age-decile log-loss gain is still 0.31 of its maximum (0.46 before).
   - Within-class leakage also flags it (0.16 vs 0.75).
   - **Point-regression R² for age is not a sufficient leakage measure.** Distributional measures (age bins, the probabilistic predictive distribution) or non-monotone targets are required. The registered SYN-3 metric could not see this; the SYN-3b amendment was added after it was noticed and is labelled as such.
2. **Nonlinear adversarial erasure removes what its adversary measures, not "the information" (prediction (b); SYN-3c).**
   - With an MSE adversary for signed A, |A| information survives (R² 0.75–0.88) at every strength up to λ_A = 32.
   - With an age-decile adversary it falls to 0.25–0.46, and decile gain to 0.04–0.11.
   - No nonlinear eraser reached exact linear guardedness: ridge R² stays 0.06–0.25, where LEACE gives 0.
   - Erasure claims therefore must name the adversary family, and leakage must be audited by probes from other families.
3. **Task-aware erasure does not improve substitution (prediction (c) not supported).**
   - At comparable leakage, TP-fused does not have lower L_YA than TP-img or NAE in any regime.
   - No nonlinear method exceeds LEACE's η in the linear regimes (prediction (d) holds).
   - The single higher η (N2 TP-fused at λ_A = 2: 0.79) comes with much higher leakage (|A| 0.69, ridge 0.17) and 3 seeds.
   - **Theory accounts for this.** By Proposition 3, an attribute-free eraser can preserve the fused posterior only if 𝔐(S) carries no age information. In the entangled regimes the diagnostic image-specific signal U lives in the same channel as A (A + 0.5U). Without A at inference, no transformation can keep U while removing A. The irreplaceable loss is fixed by the data, not by the eraser, and LEACE already sits on that frontier in the linear-Gaussian case.
4. **The nonlinear sham is not inert (prediction (e) not supported).** An adversary trained against permuted age still distorts the representation (distortion 0.27–0.31). It destroys 11–17 × 10⁻³ nats of diagnostic information with η 0.16–0.36, behaving like random erasure. Nonlinear erasers need distortion-matched shams, not just permuted-label shams.

**Implications for Phase 4–5.**
- A task-preserving (MATE-type) transformation shows no advantage in any synthetic regime where ground truth is known. Theory explains why the attribute-free frontier is data-determined.
- Per the directive ("if a result invalidates the proposed method, do not force the method to work"), MATE is **not** taken to a confirmatory real-data comparison.
- Real-data work keeps LEACE as the principal intervention, η and L_YA as outcomes, and nonlinear erasers (MSE and decile adversaries) only as an E4 leakage audit on the development set.

## Failures during Phase 3 (EXECUTION_LOG.md)
- SYN-3c: first launch failed at import with a duplicate keyword in the config dictionary; fixed and re-run (2 min). No results affected.
- **SYN-2, first run:** a worker was killed (SIGBUS) under memory pressure with 8 workers at d = 768.
- **SYN-2, second run:** a worker crashed (SIGSEGV) with 4 workers.
- **SYN-2, third run:** "SVD did not converge" in the exact-risk pseudo-inverse. The view covariances after erasure are rank-deficient by construction, with condition numbers of 10¹⁶–10¹⁸, and the failure was intermittent.
- **Fix:** an eigendecomposition-based pseudo-inverse for symmetric PSD matrices. The notebook and SYN-1 were re-run on the fixed engine with identical values, and SYN-2 then completed. No output from the failed runs was used.
