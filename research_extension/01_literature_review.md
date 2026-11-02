# 01 — Literature review: concept erasure, task preservation and the incremental value of separately observed metadata

*Phase 1, written 2026-10-30. Reading notes with page-level detail for the mandatory papers are in literature/paper_notes.md, and the machine-readable matrix is in 02_literature_comparison.csv. Verification levels:*
- **[F]** *full text read for the stated parts, from the arXiv PDF;*
- **[A]** *abstract or official listing read;*
- **[S]** *secondary description only.*

*No work is described beyond its verification level.*

Notation follows the directive: X is the image, S = f(X) ∈ ℝ^d its representation, A the metadata (age), Y the diagnosis, and Z = T_φ(S) the transformed representation. Papers use other letters; they are translated here.

---

## 1. The six mandatory papers

### 1.1 Linear Adversarial Concept Erasure — Ravfogel, Twiton, Goldberg, Cotterell, ICML 2022 [F]
- **Objective.** A maximin game, max over P ∈ P_K, min over θ, of Σ_n ℓ(a_n, g⁻¹(θᵀ P s_n)). Here P_K is the set of rank-(d−K) orthogonal projections.
- **Results.**
  - Closed forms exist for linear regression and partial least squares. For K = 1 with squared loss, P* = I − Sᵀaaᵀ S / (aᵀSSᵀa) (Prop. 3.1): the orthogonal projection removing the cross-covariance direction.
  - For logistic loss there is no closed form. The relaxation RLACE optimises over the Fantope and then rounds to a projection.
- **Guarantees.** Exact optimum only for the closed-form cases; linear adversaries only.
- **Task preservation.** Only the minimal-rank, orthogonal-projection constraint.
- **Evaluation.**
  - GloVe gender (rank 1 suffices) and Bias in Bios fairness (BERT, frozen and fine-tuned), against INLP and gradient-reversal adversarial training.
  - Qualitative CelebA pixel-space erasure.
- **A finding the directive must heed (footnote 11).** Adversaries trained jointly with the eraser reach chance accuracy, but fresh adversaries at test time still recover the attribute.
- **For this project.** Age is a scalar attribute with a squared-loss adversary. Here RLACE coincides with an orthogonal projection removing Cov(S, A), and LEACE also zeroes that covariance with a minimal-distortion oblique projection. Both reach the same linear erasure level and differ only in distortion geometry, so a separate "linear adversarial" arm for age is a distortion control, not a stronger eraser. Their difference matters for categorical attributes (site, age bins) under logistic adversaries.

### 1.2 LEACE: Perfect Linear Concept Erasure in Closed Form — Belrose et al., NeurIPS 2023 [F]
- **Theory.** For one-hot concepts with positive class probabilities, five conditions are equivalent:
  1. linear guardedness against every convex loss;
  2. optimality of the trivial predictor;
  3. equal class-conditional means;
  4. Σ_SA = 0;
  5. statistical parity of every linear classifier.

  The equivalence extends to continuous A under OLS loss (§4.3).
- **Erasers.**
  - An affine map guards A iff colsp(Σ_SA) ⊆ ker P (Thm 4.1).
  - The minimiser of E‖PS − S‖²_M subject to Cov(PS, A) = 0 is P* = I − W⁺ P_{WΣ_SA} W for **every** p.s.d. M (Thm 4.2), with the bias term in Thm 4.3.
  - If A is available at inference, the minimal-distortion guarded representation is the OLS residual S − Σ_SA Σ_AA⁺(A − EA) ("oracle LEACE", App. I).
- **Evaluation.** Bias in Bios; part-of-speech "concept scrubbing" across all layers of LLMs.
- **Limitations (authors).** Linear only. They conjecture that nondestructive erasure against arbitrary nonlinear adversaries is intractable without knowledge of the data-generating process. Kernel erasers do not transfer across kernels.
- **For this project.**
  - LEACE is the eraser behind E45–E55.
  - Its guarantee is in-sample. Out of sample it fails when n_train ≈ d (E50 §4: the test/train SD ratio of the coordinate is 1.8–3.1).
  - Oracle LEACE is admissible here because age *is* observed at inference, and §2.4 below shows it trivially achieves perfect metadata substitution.

### 1.3 Fundamental Limits of Perfect Concept Erasure — Basu Roy Chowdhury, Dubey, Beirami, Kidambi, Monath, Ahmed, Chaturvedi, AISTATS 2025 (PMLR 258:901–909) [F §§1–4]
- **Objective.** max_f I(Z;S) s.t. I(Z;A) ≤ ε, a privacy-funnel / information-bottleneck form after Calmon et al. (2017). Utility is task-agnostic.
- **Assumptions.** Finite supports, disjoint supports across concept groups, |S| > |A|.
- **Results.**
  - Perfect erasure is achievable if the smallest principal inertia component is 0 or |S| > |A|.
  - Under perfect erasure, I(Z;S) = I(Z;S|A) ≤ H(S|A), **with equality iff H(S|Z,A) = 0** (Lemma 3.3).
  - Funnel bounds: max{0, u − H(S|A)} ≤ ε(u) ≤ u·I(A;S)/H(S).
  - If the group-conditional distributions are permutations of one another, a piecewise bijection to a common law attains the bound (Def. 4.1, Method 4.2).
- **Evaluation.** Synthetic data, GloVe gender, Jigsaw religion with GPT-4 embeddings, with KDE-estimated MI. Baselines include INLP, RLACE, LEACE, FaRM and KRaM.
- **Limitations.** Needs estimable finite-support distributions; hard with little data.
- **For this project.** The equality condition H(S|Z,A) = 0 is the task-agnostic form of "everything erased is recoverable from the separately observed attribute". §2.4 turns it into a statement about Y.

### 1.4 Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure — Akbari, Afshari, Boddeti, NeurIPS 2025 [F §§3–5]
- **Objective.** inf_θ HSIC(Z_θ, A) − HSIC(Z_θ, Y), a bi-objective with an unsupervised variant that preserves S instead of Y. It is solved iteratively (an HSIC-trained encoder, then an RKHS "disentanglement" eigenproblem) so the utility–erasure path can be traced.
- **Guarantee.** Characteristic-kernel HSIC = 0 iff independence. There is no optimisation or finite-sample guarantee; the authors list theoretical guarantees as open.
- **Evaluation.**
  - Dial-Mention, Dial-Sentiment and Bias in Bios (BERT, GPT-2, LLaMA, DeepSeek; frozen and fine-tuned), against MLP adversaries until attribute accuracy reaches the majority rate.
  - Baselines: INLP, FaRM, KRaM and others.
- **Findings.** Erasure costs less utility on better-disentangled representations from more capable models. Skewed sampling worsens the trade-off.
- **For this project.**
  - **Supervised, task-preserving nonlinear erasure already exists.**
  - Its observation that capability eases erasure is the counterpart of the directive's E6 question.

### 1.5 Nonlinear Concept Erasure: a Density Matching Approach (LEOPARD) — Saillenfest, Lemberger, ECAI 2025 per the authors' repository (arXiv 2507.12341; one listing names a co-located workshop) [F §§1–5]
- **Objective.** Learn a rank-r orthogonal projection P = UUᵀ minimising γ·(orthogonality penalty) plus MMD² between the class-conditional laws of the projected embeddings (mixture of Gaussian kernels). It is rounded to the nearest rank-r projection afterwards; a cascaded variant uses nested projections.
- **Guarantee.** None beyond "MMD = 0 iff equal laws" for characteristic kernels.
- **Scope.** Discrete concepts only.
- **Evaluation.** GloVe ternary gender, Bias in Bios and DIAL race, by probe accuracy, downstream accuracy and fairness.
- **For this project.** A nonlinear, projection-constrained baseline. Continuous age needs bins or a continuous dependence measure.

### 1.6 Principled Nonlinear Concept Erasure and Steering — Simai, Ravfogel, Shaham, arXiv 2610.32516 (23 Oct 2026; preprint) [F §§1–5, App. A.2]
- **Objective.** A conditional autoencoder: encoder Z = e(S), which does **not** use A at inference; decoder d(Z, A); reconstruction plus independence (Z ⊥ A). No task labels are used.
- **Guarantees** (discrete, idealised).
  - If d(Z, A) = S a.s. and Z ⊥ A, then I(Z;S) = H(S|A), the maximum among A-independent deterministic representations (Thm 1, Cors 1–2).
  - Under Z ⊥ A, I(S;Z) = H(S|A) − H(S|Z,A) (Prop. 2).
  - Small reconstruction MSE implies approximate preservation under a Lipschitz left-inverse assumption (Prop. 1).
  - MI is not monotone in HSIC (Prop. 3).
  - Steering: decoding with a different attribute value.
- **Evaluation.** Several text datasets and language models. The authors claim better trade-offs than supervised competitors.
- **For this project.** This is the most explicit formalisation of "remove A from the representation but keep everything recoverable once A is supplied". Its idealised regime implies perfect metadata substitution for every downstream task (§2.4), although the paper never measures the downstream incremental value of A.

### 1.7 Related 2026 papers surfaced by the mandatory searches
- **MANCE** (Avitan, Goldberg, Elazar; arXiv 2607.03973) [A + method skim]. Classifier-guided iterative erasure constrained to an estimated data manifold. Variants are stacked on LEACE; it includes vision (CLIP on CelebA) settings.
- **MUtE** (Saillenfest; arXiv 2609.11253) [A + skim]. Optimal-bound piecewise erasure with dual counterfactual maps, for discrete concepts in continuous spaces.
- **Closed-Form Concept Erasure via Double Projections** (arXiv 2604.10032) [A]. Diffusion-model weight editing; peripheral.

---

## 2. The wider literature, by area

### 2.1 Linear erasure and guardedness
- **INLP** (Ravfogel, Elazar, Gonen, Twiton, Goldberg; ACL 2020) [A]. Iteratively projects onto the nullspace of linear attribute classifiers. Shown suboptimal by RLACE (Props 4.1–4.2).
- **SAL** (Shao, Ziser, Cohen; EACL 2023) [A]. Spectral removal of the directions of maximal covariance with the attribute, with a kernel extension. Covered by LEACE's Thm 4.1 as guaranteeing linear guardedness.
- **Log-linear guardedness** (Ravfogel, Goldberg, Cotterell; ACL 2023) [A]. Binary log-linear guardedness can be undone by a multiclass log-linear downstream model, a limit of guardedness for downstream bias.
- **Mean projection and Fair PCA** are discussed in LEACE App. D [F].

### 2.2 Nonlinear erasure
- **Kernelised concept erasure** (Ravfogel, Vargas, Goldberg, Cotterell; EMNLP 2022) [A]. Protection does not transfer to other nonlinear adversaries; exhaustive nonlinear erasure is open.
- **FaRM** (Basu Roy Chowdhury, Chaturvedi; TACL 2022) [A]. Rate-distortion-based debiasing, with or without a task. It leaks less to nonlinear probes.
- **KRaM** (Basu Roy Chowdhury et al.; NeurIPS 2023) [A]. Kernelised rate-distortion erasure for categorical, **continuous** and vector concepts, plus an alignment score.
- Obliviator, LEOPARD, MANCE and the 2026 conditional autoencoder are described in §1.

### 2.3 Adversarial representation learning and its leakage
- **Gradient reversal** (Ganin & Lempitsky, ICML 2015) [S, via RLACE].
- **Censoring representations with an adversary** (Edwards & Storkey; ICLR 2016) [A]. Includes removal of private information from images.
- **Controllable invariance** (Xie et al.; NeurIPS 2017) [A]. Encoder, discriminator and predictor in a minimax game.
- **LAFTR** (Madras, Creager, Pitassi, Zemel; ICML 2018) [A]. Adversarial objectives matched to parity, equalised odds and equal opportunity, with reconstruction and a task head.
- **Elazar & Goldberg (EMNLP 2018)** [A]. The adversary reaches chance in training, but **post-hoc probes still recover demographics**, so adversarial training is not a reliable route to invariance. This is the empirical basis for the directive's E4 "adaptive adversary" safeguard.

### 2.4 Fair and invariant representations, task-utility trade-offs and recoverability given the attribute
- **Learning Fair Representations** (Zemel et al.; ICML 2013) [A]. The two-goal representation problem.
- **Inherent trade-offs** (Zhao & Gordon; NeurIPS 2019) [A]. When base rates differ across groups, any representation satisfying statistical parity incurs a lower-bounded joint error.
  - This bounds the **image-only** cost C_Y of making Z independent of A when Y depends on A.
  - It does **not** bound a fused predictor that receives A separately. That distinction is the hinge of this investigation.
- **Learning controllable fair representations** (Song et al.; AISTATS 2019) [A]. A constrained information-theoretic objective solved through Lagrangian duality. This is the closest precedent for the directive's constrained (τ_A, ε_Y) formulation.
- **Invariant representations without adversarial training** (Moyer et al.; NeurIPS 2018) [A]. Penalises I(z;c) with a decoder conditioned on (z, c); c is needed only in training. Recoverability given the nuisance is used as the preservation principle.
- **Adversarially learned representations for information obfuscation and inference** (Bertran et al.; ICML 2019) [A]. A privacy–utility adversarial game with stochastic transformations.
- **Privacy funnel** (Makhdoumi, Salamatian, Fawaz, Médard; ITW 2014) [A]; **principal inertia components** (Calmon et al.; IEEE TIT 2017) [A].
- **Frisch–Waugh–Lovell theorem** (Frisch & Waugh, Econometrica 1933; Lovell, JASA 1963) [A].
  - In a linear model on [image features residualised on A, A], the fit is identical to the model on [original features, A], and A's coefficient equals its marginal (total) effect.
  - With test-time residualisation, the redundant signal is therefore re-attributed to A exactly. This is the linear, finite-sample form of perfect substitution.
- **Metadata Normalization** (Lu, Zhao, Zhang, Pohl, Fei-Fei, Niebles, Adeli; CVPR 2021) [A]. A batch-level GLM layer regresses metadata effects out of features during training.
- **Semi-structured distributional regression** (Rügamer, Kolb, Klein; The American Statistician 2023) [A]. Deep (e.g. image) predictors are orthogonalised against structured covariates to keep the covariate effects identifiable.
- **Post-hoc orthogonalisation of chest X-ray embeddings** (Weber, Ingrisch, Bischl, Rügamer; arXiv 2311.01349; republished at PAKDD 2025 per the authors' institute pages) [F §2 and discussion].
  - QR-based residualisation of embeddings on age, sex and race.
  - The authors argue that clinically useful protected information "should be modeled explicitly by including protected information as a predictor", citing semi-structured models.
- **Generalizing orthogonalization for models with non-linearities** (Rügamer et al.; arXiv 2405.02475) [A]. Corrections for ReLU-type non-linearities.
- **FairREAD** (Gao, Hao, Zhou; arXiv 2412.16373) [F §§1–3].
  - Chest X-ray (CheXpert): an adversarial and orthogonality-constrained fair image encoder, then **re-fusion of demographic attributes as a separate input**, motivated explicitly by the clinical relevance of the removed information.
  - Evaluated by AUC, accuracy and fairness gaps; ablations on adversarial weight and thresholds.
  - It does **not** report the re-fused attributes' incremental predictive value as an outcome.

### 2.5 Sufficiency, information bottleneck and dimension reduction
- **Information bottleneck** (Tishby, Pereira, Bialek 2000) [S]; **Deep VIB** (Alemi, Fischer, Dillon, Murphy; ICLR 2017) [A].
- **Achille & Soatto (JMLR 2018)** [A]. For sufficient representations, invariance to nuisances is equivalent to minimality.
- **Kernel dimension reduction** (Fukumizu, Bach, Jordan; JMLR 2004) [A]. Finds a subspace B with Y ⊥ S | BᵀS via RKHS covariance operators. This is the population notion of a diagnostic-sufficient subspace that an erasure must avoid cutting into.
- **Locatello et al. (ICML 2019)** [A]. Unsupervised disentanglement is impossible without inductive biases. This is relevant to any claim that "age" and "diagnosis" occupy separable components of S.

### 2.6 Information estimation and decomposition
- **Conditional MI equals the expected log-loss reduction for Bayes predictors** (Gadgil, Covert, Lee; ICLR 2024) [F, iteration 5].
- **CCMI** (Mukherjee, Asnani, Kannan; UAI 2019) [A]. A classifier-based CMI estimator.
- **Formal limits on MI measurement** (McAllester & Stratos; AISTATS 2020) [A]. Distribution-free high-confidence lower bounds are O(ln N). They suggest difference-of-cross-entropies estimates, which carry no bound guarantee; this is exactly the form of the project's Δ.
- **Partial information decomposition** (Williams & Beer 2010) [A]: redundancy, unique information and synergy. Interaction information conflates redundancy and synergy. **PID for multimodal data** (Liang et al.; NeurIPS 2023) [A, iteration 5].

### 2.7 Multimodal fusion dynamics
- **Modality competition** (Huang, Lin, Zhou, Yang, Huang; ICML 2022) [A]. Jointly trained late-fusion networks learn only a subset of modalities.
- **Metadata fusion in dermatology**: Höhn et al. 2021 review, Pacheco & Krohling 2020, Ahammed et al. CVPR-W 2025 [F/A, earlier iterations].

### 2.8 Demographic and attribute information in medical images, and shortcuts
- **Age and sex from retinal photographs** (Poplin et al.; Nature Biomedical Engineering 2018) [A]. Age MAE about 3.3 years; sex AUC 0.97.
- **Race from radiographs** (Gichoya et al.; Lancet Digital Health 2022) [A]. AUC 0.81–0.99, robust to degradation.
- **Algorithmic encoding of protected characteristics** in chest X-ray models (Glocker, Jones, Bernhardt, Winzeck; eBioMedicine 2023) [A, citation only].
- **ShorT** (Brown et al.; Nature Communications 2023) [A].
  - **Age** as the sensitive attribute in chest X-ray and **dermatology**. Age encoding is varied by scaling an age head's gradients.
  - The amount encoded bears little relation to fairness across datasets; what matters is whether changing the encoding changes fairness. This is the closest precedent for "intervene on attribute encoding and read off a downstream consequence"; the consequence there is fairness, not metadata's value.
- **The limits of fair medical imaging AI** (Yang, Zhang, Gichoya, Katabi, Ghassemi; Nature Medicine 2024) [A/S]. Stronger encoding goes with larger fairness gaps, and fairness fails to transfer across sites.
- **Confounder-free networks** (Zhao, Adeli, Pohl; Nature Communications 2020) [A]. Adversarial removal of **age** (a confounder) from MRI features, with a task predictor.
- **Adversarial debiasing of 3D CT foundation embeddings** (arXiv 2502.04386) [S].
- **Dermatology shortcuts**: Bissoto, Valle, Avila (CVPR-W 2020) [A]. Artefact biases; debiasing "not ready".

### 2.9 Dataset shift and causal views
- **Causality matters in medical imaging** (Castro, Walker, Glocker; Nature Communications 2020) [A].
- **NuRD** (Puli, Zhang, Oermann, Ranganath; ICLR 2022) [A]. Representations optimal under nuisance-randomised distributions; includes chest X-ray.
- **Causally motivated shortcut removal using auxiliary labels** (Makar et al.; AISTATS 2022) [A]. Conditional-independence regularisers from auxiliary (nuisance) labels.

---

## 3. Synthesis for the three-objective problem

1. **Attribute leakage R_A(Z).**
   - Linear: solved in closed form (LEACE; RLACE closed forms).
   - Nonlinear: heuristic and adversary-specific. Kernel and HSIC/MMD methods offer independence criteria without optimisation or finite-sample guarantees, and post-hoc adversaries routinely beat training-time adversaries (Elazar & Goldberg 2018; RLACE fn. 11).
   - Leakage must therefore be measured by adaptive probes fitted after the transformation is fixed.
2. **Diagnostic cost C_Y.**
   - When Y depends on A, making Z independent of A costs image-only utility in a provable way (Zhao & Gordon 2019).
   - Task-preserving erasers (Obliviator supervised, FaRM, LAFTR, CF-Net) trade this off empirically.
3. **Metadata substitution M_A.**
   - Never the measured outcome in the erasure literature found here.
   - The literature does contain its ingredients:
     - (a) the condition that makes it perfect: S recoverable from (Z, A) (PEF Lemma 3.3 equality; Moyer et al.; Simai et al.);
     - (b) the classical linear identity that makes it exact under test-time residualisation (FWL; Metadata Normalization; semi-structured orthogonalisation; Weber et al.);
     - (c) an architecture that re-fuses removed demographics for clinical utility (FairREAD).
   - The algebra linking M_A to C_Y is the chain rule (04_mathematical_foundations.md derives M_A = C_Y − [R_Y(Z,A) − R_Y(S,A)] ≤ C_Y). It is an elementary consequence of standard identities and is not claimed as new.
4. **Medical imaging.**
   - Images encode age, sex and race (retina, radiographs), and **age** encoding has been manipulated in dermatology (ShorT).
   - Age has been adversarially removed as a confounder (CF-Net), and protected features orthogonalised out of chest X-ray embeddings with an explicit recommendation to model them separately.
   - No study found measures how the separately observed attribute's incremental diagnostic value changes when its trace is removed, with performance-matched controls.

Search coverage: 6 mandatory papers [F] and 3 further 2026 erasure papers; about 45 additional works verified at [A] or [F] across the 16 requested areas, from primary sources (arXiv, PMLR, NeurIPS, ACL Anthology, CVF, PMC). Absence of evidence from searches is weak evidence; 03_novelty_audit.md states the claims accordingly.
