# 03 — Novelty audit

*Phase 1, written 2026-10-30. It rests on 01_literature_review.md and 02_literature_comparison.csv; the identities quoted are derived in full in 04_mathematical_foundations.md. Standard of evidence: a claim is "not found" only after the targeted searches listed in §5, and absence from searches is treated as weak evidence.*

## 1. The question, stated precisely

For a transformation Z = T(S) of a frozen image representation, with metadata A observed separately at inference:

M_A(T) = Δ_A(Z) − Δ_A(S), where Δ_A(·) = R_Y(·) − R_Y(·, A).

M_A is the change in the incremental diagnostic value of the separately observed attribute. The question is how M_A behaves jointly with leakage R_A(Z) and image-only cost C_Y = R_Y(Z) − R_Y(S), and whether a transformation can raise M_A without unacceptable diagnostic cost.

This is distinct from five neighbouring questions:
- **Ordinary concept erasure** asks only about R_A(Z) and a utility of Z alone (I(Z;S) or task accuracy from Z).
- **Attribute classification / leakage** asks whether A is decodable from S.
- **Privacy** asks about I(Z;A) or adversarial accuracy.
- **Fairness** asks about group gaps in a predictor.
- **Task preservation** asks about R_Y(Z).

None of them makes R_Y(Z, A) − R_Y(Z), the value of A *given* the transformed image, the object of study.

## 2. A result that reshapes the question (derived in 04 §3)

For any transformation that uses only S (possibly with independent randomness), with Bayes predictors and log-loss, the chain rule gives:

M_A = I(Y;A|Z) − I(Y;A|S) = C_Y − L_YA, with L_YA := R_Y(Z,A) − R_Y(S,A) = I(Y;S|Z,A) ≥ 0.

Hence **M_A ≤ C_Y**:
- Separately observed age can gain incremental value only up to the diagnostic information the image representation has lost.
- "Age becomes more useful without image-only diagnostic degradation" is impossible at the population level.
- What can vary is the **substitution efficiency** η = M_A / C_Y ≤ 1: the share of the lost image information that the metadata replaces.
- Equivalently, what can vary is the **irreplaceable loss** L_YA, the degradation of the *fused* (image + age) predictor.

Two further consequences matter for novelty:
1. **If the transformation may use A at inference**, perfect substitution (L_YA = 0, η = 1) is achieved trivially by any map that is invertible given A. Examples:
   - the OLS residual on A (oracle LEACE; the Frisch–Waugh–Lovell theorem in the linear case);
   - Metadata Normalization;
   - test-time orthogonalisation (Weber et al.);
   - a conditional transport.

   In that regime M_A = C_Y identically, and measuring M_A is uninformative.
2. **The informative regime is the A-free eraser** (fitted with A on training data, applied without A, as LEACE in E50 or the encoder of a conditional autoencoder). There η < 1 is possible, and 1 − η measures how much of the erased, attribute-aligned image information is *not* replaceable by the recorded attribute.

For age in skin images, 1 − η is a direct estimate of the diagnostic signal in the image's "age direction" beyond chronological age, for example photo-ageing or lesion-specific changes. This is a scientifically meaningful quantity.

The identity itself is elementary: chain rule plus non-negativity of conditional MI. It is not claimed as new theory.

## 3. Classification of findings

### Already established (cite; do not claim)
| Item | Where |
|---|---|
| Linear guardedness ⇔ zero cross-covariance; minimal-distortion closed-form eraser; continuous attributes under OLS | LEACE (NeurIPS 2023) |
| Linear adversarial erasure in closed form for regression (orthogonal removal of the covariance direction) | RLACE (ICML 2022), Prop. 3.1 |
| Linear erasure leaves nonlinear information; kernel/adversarial protection does not transfer to other adversaries; post-hoc probes beat training-time adversaries | Ravfogel et al. EMNLP 2022; Elazar & Goldberg EMNLP 2018; RLACE fn. 11; LEACE limitations |
| Nonlinear erasure via HSIC, MMD, rate-distortion or conditional autoencoders, including supervised (task-preserving) variants | Obliviator, LEOPARD, FaRM/KRaM, Simai et al. 2026 |
| Making a representation independent of A costs image-only utility when base rates differ | Zhao & Gordon (NeurIPS 2019); Kumar et al. (NeurIPS 2022) |
| Maximal retention under erasure requires recoverability given A: H(S\|Z,A) = 0 | PEF (AISTATS 2025) Lemma 3.3; Simai et al. 2026 Thm 1; Moyer et al. 2018 |
| Residualising image features on metadata, and modelling metadata explicitly beside an orthogonalised image part | FWL (1933/1963); Metadata Normalization (CVPR 2021); semi-structured regression (TAS 2023); Weber et al. (CXR, 2023) |
| Re-fusing removed demographic attributes into a fair image representation for clinical utility | FairREAD (arXiv 2024/25) |
| Medical images encode age, sex and race; age encoding can be manipulated (in dermatology too) and affects fairness; age removed as a confounder from imaging features | Poplin 2018; Gichoya 2022; Brown et al. 2023 (ShorT); Zhao, Adeli & Pohl 2020 |
| Log-loss reduction of a Bayes predictor equals conditional MI; chain-rule decompositions; redundancy/synergy (PID) | Cover & Thomas; Gadgil et al. 2024; Williams & Beer 2010 |
| Task-preserving constrained (Lagrangian) formulations of erasure | Song et al. (AISTATS 2019); Obliviator |

### Partially explored
| Item | What exists | What is missing |
|---|---|---|
| Change in the attribute's incremental value after removing its trace from the image | Implied by FWL / semi-structured identifiability (re-attribution of shared signal to the covariate); motivates FairREAD | Never the measured outcome; no matched controls; never used to quantify redundancy |
| The bound M_A ≤ C_Y and the substitution efficiency η | Follows directly from the chain rule (this project's Proposition 1 is the same algebra with X in place of S) | Not stated in the erasure literature found; elementary, so a framing device rather than a theorem |
| Intervening on attribute encoding in dermatology | ShorT varies age encoding and reads off fairness | Effect on metadata's incremental value not studied |

### Potentially novel (conservatively)
1. **A measurement design.** The substitution efficiency of an **attribute-free eraser**, η = M_A / C_Y, used as an empirical estimate of how redundant an image representation's attribute trace is with the recorded attribute. It is tested against:
   - performance-matched random subspaces chosen on training data only;
   - permuted-attribute shams;
   - an information dose;
   - re-append restoration.

   This project's E50 already estimates η = 0.71–0.97 for LEACE against 0.03–0.10 for matched random erasure on three large dermoscopy sets.
2. **An attribute-specific double dissociation** of restored metadata value (age vs site) on two datasets (E51; replication/exploratory status).
3. **An empirical account of when the design fails**: n_train ≈ d eraser overfitting, and strong attribute–label dependence. The failure modes themselves are known; the diagnostics and their effect on M_A are not documented elsewhere.

Not novel, and not to be claimed:
- a "task-preserving adversarial transformation" (the MATE minimax as written is the Xie/LAFTR/CF-Net family);
- HSIC- or MMD-based nonlinear erasure;
- conditional-autoencoder erasure;
- re-fusion of demographics.

## 4. Recommended research question

The directive's original question ("selectively remove metadata-predictive information while preserving diagnostic utility, and make age more useful") is either impossible (if utility means image-only utility, by M_A ≤ C_Y) or trivial (if the transformation may use A at inference, by FWL / oracle residualisation).

The better-posed question is:

> **How much of the attribute-predictive information in a pretrained medical image representation is substitutable by the recorded attribute, and what is in the remainder?**
> Operationally: estimate η = M_A / C_Y and L_YA for attribute-free erasers at matched leakage and matched cost. Determine how η depends on the attribute (age, sex, site), the dataset (age–label dependence), the representation (HSV histogram → ViT-L) and the sample regime (n/d). Then test whether nonlinear or task-aware erasers raise η (lower L_YA) relative to LEACE at the same leakage. That last comparison is the only place a methodological contribution could arise.

A "MATE" method is worth developing only if:
- (i) it targets L_YA (fused-risk preservation) rather than image-only risk;
- (ii) it is A-free at inference;
- (iii) it is distinct from Obliviator (supervised HSIC), FaRM and the conditional autoencoder of Simai et al.

Its distinctive element would be a training signal that preserves the **fused** predictor R_Y(Z, A) instead of R_Y(Z). A conditional-reconstruction objective (Simai et al.) already does this in the task-agnostic limit, so any novelty would be narrow: the task-specific, metadata-conditioned variant, and an empirical comparison. Phase 5 decides whether even that is warranted after the synthetic and baseline results.

## 5. Searches performed (2026-10-30)
- Mandatory titles.
- Linear, nonlinear and adversarial erasure.
- Fair, invariant and controllable representations.
- Privacy funnel and principal inertia components.
- Sufficiency, information bottleneck and kernel dimension reduction.
- Disentanglement.
- MI/CMI estimation.
- PID and modality competition.
- Medical demographic encoding, shortcuts, confounder removal, metadata normalisation and orthogonalisation.
- Causal and shift-robust representations.
- Three targeted novelty searches:
  - "remove demographic information from image embedding then add demographics as separate input";
  - "radiomics residualised on clinical variables, added value";
  - "concept erasure with side information available at inference".

  These surfaced FairREAD and Weber et al. (closest) and no study of M_A as an outcome.

Residual risk: works using other vocabulary may exist (econometrics "partialling-out" with incremental R², or radiomics "added value after orthogonalisation"). These are classical in spirit, and the FWL entry acknowledges them.
