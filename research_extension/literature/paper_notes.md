# Primary-source reading notes (Phase 1)

Each entry was read from the arXiv PDF (text extracted with pdftotext on 2026-10-30) unless marked otherwise. Verification levels:
- **[F]** full text read for the stated parts;
- **[A]** abstract/listing only;
- **[S]** secondary source only.

Notation: X is the representation, Z_c the concept (the papers' own letters vary), Y the task.

## M1. Linear Adversarial Concept Erasure (RLACE) [F]
- **Bibliography.** Ravfogel, Twiton, Goldberg, Cotterell. ICML 2022, PMLR 162:18400–18421. arXiv 2201.12091 (v8, 17 Dec 2024). Code: github.com/shauli-ravfogel/rlace-icml.
- **Formulation.** A sequential maximin game. The eraser chooses a rank-(D−K) orthogonal projection P ∈ P_K (P = I − WᵀW, WWᵀ = I_K). The predictor then fits a GLM θ on PX. The game is max_P min_θ Σ ℓ(y_n, g⁻¹(θᵀPx_n)).
- **Closed forms.**
  - Linear regression, K = 1 (Prop. 3.1): P* = I − XᵀyyᵀX / (yᵀXXᵀy), the orthogonal projection removing the cross-covariance direction Xᵀy.
  - Partial least squares (Prop. 3.3, via Lemma 3.2): remove the top-K eigenvectors of XᵀyyᵀX.
  - INLP is shown suboptimal for these objectives (Props. 4.1, 4.2).
- **RLACE.** For classification losses, the projection set is relaxed to its convex hull, the Fantope F_K = {A : I ≽ A ≽ 0, tr A = K}. The method alternates SGD on θ, gradient ascent on P and Fantope projection, then rounds to the nearest rank-K projection.
- **Guarantee.** Exact optimum for the closed-form cases; none for the relaxation's convergence. Linear adversaries only.
- **Task preservation.** Only implicit: minimal-rank orthogonal projection, no task term.
- **Experiments.**
  - GloVe binary gender: rank 1 suffices against logistic probes (later proven by LEACE).
  - Bias in Bios profession classification with BERT, frozen and fine-tuned: TPR-GAP fairness, compared with INLP and adversarial fine-tuning (gradient reversal, linear and MLP adversaries).
  - CelebA pixel-space erasure (qualitative).
- **Notable.** Footnote 11: adversaries trained jointly converged to chance accuracy, but this "did not generalize to new adversaries at test time". This is the reason adaptive post-hoc probes are needed.
- **Limitation (authors).** Only linear classifiers are guarded; nonlinear erasure is an open challenge.
- **Relevance.** For a scalar attribute (age) with a squared-loss linear adversary, linear adversarial erasure equals the orthogonal projection removing Cov(X, A). It differs from LEACE only in using an orthogonal rather than an oblique (whitened, minimal-distortion) projection, so both achieve Cov(PX, A) = 0. An "RLACE for age" baseline is therefore a distortion variant of LEACE, not a different erasure level.

## M2. LEACE: Perfect linear concept erasure in closed form [F]
- **Bibliography.** Belrose, Schneider-Joseph, Ravfogel, Cotterell, Raff, Biderman. NeurIPS 2023. arXiv 2306.03819. Code: github.com/EleutherAI/concept-erasure.
- **Definitions.**
  - Guardedness (Def. 2.1): X (V, L)-guards Z if its conditional law is among the worst for predictors in V under every loss in L.
  - Linear guardedness: V = affine predictors, L = all losses convex in the prediction.
- **Main equivalences (§3).** For one-hot Z with positive class probabilities, the following are equivalent:
  1. linear guardedness;
  2. the trivial loss is optimal for every convex loss;
  3. the class-conditional means equal the unconditional mean;
  4. Σ_XZ = 0;
  5. statistical parity of every linear classifier (App. C).
- **Erasers.**
  - Theorem 4.1: r(x) = Px + b guards Z iff colsp(Σ_XZ) ⊆ ker P.
  - Theorem 4.2: argmin_P E‖PX − X‖²_M s.t. Cov(PX, Z) = 0 is P* = I − W⁺ P_{WΣ_XZ} W, with W = (Σ_XX^{1/2})⁺, for **every** p.s.d. M.
  - Theorem 4.3: the bias term b* = E X − P* E X.
- **Continuous Z (§4.3).** The same eraser applies when Z ∈ ℝ^k under OLS loss, because guardedness is equivalent to Σ_XZ = 0. This is how this project already applies LEACE to age.
- **Oracle LEACE (App. I, Theorem I.1).** If Z is available at inference, the minimal-distortion X′ with Cov(X′, Z) = 0 is the OLS residual X − Σ_XZ Σ_ZZ⁺ (Z − E Z).
- **Experiments.**
  - Bias in Bios (BERT, 2 orders of magnitude faster than RLACE and INLP).
  - Concept scrubbing of part-of-speech in every layer of LLMs (Pythia, LLaMA).
- **Limitations (authors).**
  - Linear only.
  - They conjecture that nondestructive editing against a general nonlinear adversary is intractable unless the data-generating process is known.
  - Erasers fitted with one kernel do not generalise to other kernels.
- **Relevance.**
  - It is the eraser already used in E45–E55.
  - Its guarantee holds in sample. Out of sample it fails when n ≈ d (this project's E50, §4).
  - Oracle LEACE is admissible in the metadata setting because A *is* observed at inference.

## M3. Fundamental Limits of Perfect Concept Erasure (PEF) [F, §§1–4]
- **Bibliography.** Basu Roy Chowdhury, Dubey, Beirami, Kidambi, Monath, Ahmed, Chaturvedi. AISTATS 2025, PMLR 258:901–909. arXiv 2503.20098. Code: github.com/brcsomnath/PEF.
- **Formulation.** max_f I(Z; X) s.t. I(Z; A) ≤ ε, Z = f(X). Task-agnostic utility; information-bottleneck / privacy-funnel form (Calmon et al. 2017).
- **Assumptions.** (A1)–(A5):
  - finite supports;
  - disjoint supports across concept groups (A4);
  - |X| > |A|.
- **Results.**
  - Lemma 3.2: perfect erasure is achievable if the smallest principal inertia component is 0 or |X| > |A|.
  - Lemma 3.3: under perfect erasure, I(Z; X) = I(Z; X | A) ≤ H(X | A), with equality iff H(X | Z, A) = 0.
  - Erasure-funnel bounds: max{0, u − H(X|A)} ≤ ε(u) ≤ u I(A;X)/H(X).
  - Def. 4.1: if the group distributions are equal up to permutation, a piecewise bijection to a common distribution Q attains the bound.
  - For unequal distributions, a minimum-entropy-coupling construction is used.
- **Experiments.** Synthetic data; GloVe gender; Jigsaw religion/toxicity with GPT-4 embeddings. KDE estimates of MI. Baselines include INLP, RLACE, LEACE, FaRM, KRaM.
- **Limitations (authors).** Requires an estimable finite-support distribution; difficult in low-resource settings.
- **Relevance.** The equality condition H(X|Z,A) = 0 is the task-agnostic version of perfect metadata substitution. If (Z, A) determines X, then for any task I(Y; Z, A) = I(Y; X, A).

## M4. Obliviator Reveals the Cost of Nonlinear Guardedness in Concept Erasure [F, §§3–5, limitations]
- **Bibliography.** Akbari, Afshari, Boddeti. NeurIPS 2025 (poster 118956). arXiv 2603.07529 (v2). Code: github.com/ramin-akbari/Obliviator.
- **Formulation.**
  - Nonlinear guardedness as statistical independence, Z ⊥ S, made tractable with HSIC (characteristic kernels).
  - Bi-objective inf_θ HSIC(Z_θ, S) − HSIC(Z_θ, Y) (eq. 7), solved iteratively: an encoder trained with an HSIC objective (eq. 8, with τ-weighted terms preserving X, Y and the current iterate), then an "RKHS disentanglement" eigenvalue problem (eq. 11).
- **Variants.** Supervised, using task labels Y, and unsupervised, preserving X.
- **Guarantee.** None for optimal nonlinear erasure; the authors list a closed form or theoretical guarantee as open.
- **Experiments.**
  - Datasets: Dial-Mention, Dial-Sentiment (DeepMoji) and Bias in Bios (BERT, GPT-2, LLaMA, DeepSeek), frozen and fine-tuned.
  - Utility–erasure curves against MLP adversaries, run until attribute accuracy reaches the majority rate.
  - Baselines: INLP, FaRM, KRaM and others.
- **Findings.** Erasure is more utility-preserving on better-disentangled representations from more capable models. Skewed sampling worsens the trade-off (HSIC bias).
- **Limitations (authors).** Needs attribute (and, for the supervised variant, task) labels; the optimisation is non-convex. MANCE notes that the output dimensionality is not guaranteed to be preserved.
- **Relevance.** Task-preserving nonlinear erasure, explicitly trading attribute dependence against task dependence, is **already established** here. The finding "stronger representations erase more gracefully" parallels this project's E6 question.

## M5. Nonlinear Concept Erasure: a Density Matching Approach (LEOPARD) [F, §§1–5]
- **Bibliography.** Saillenfest, Lemberger. arXiv 2507.12341 (v1 16 Jul 2025, v2 15 Aug 2025). The repository and later citations give ECAI 2025, FAIA vol. 413; one arXiv listing names a co-located workshop (AIQxQIA 2025). The venue is recorded as "ECAI 2025 per repository; listing inconsistent". Code: github.com/ToineSayan/non-LEOPARD.
- **Formulation.** Learn a rank-r orthogonal projection P = UUᵀ minimising γ·(orthogonality penalty) + MMD² between class-conditional distributions of the projected embeddings, using a mixture of Gaussian kernels and a median-heuristic bandwidth. It is rounded to the nearest rank-r projection after training. A cascaded variant uses nested projections.
- **Guarantee.** MMD with a characteristic kernel vanishes iff the distributions match; no finite-sample or optimisation guarantee.
- **Scope.** Discrete concepts only. The authors suggest summing losses for multiple concepts.
- **Experiments.** GloVe (ternary gender), Bias in Bios, DIAL (race). Evaluated by probe accuracy, downstream accuracy, WS-353 similarity and fairness metrics. Competitive but not better on DIAL race.
- **Relevance.** A nonlinear-erasure baseline that keeps an orthogonal-projection form. Applying it to continuous age needs binning or a continuous dependence measure.

## M6. Principled Nonlinear Concept Erasure and Steering [F, §§1–5, App. A.2]
- **Bibliography.** Simai, Ravfogel, Shaham. arXiv 2610.32516, submitted 23 Oct 2026. Preprint, no venue.
- **Formulation.** A conditional autoencoder: encoder Z = e(X), which does not use the concept at inference, and decoder d(Z, C). Objective: reconstruction + independence (Z ⊥ C). Steering decodes with a different c′. No target-task labels are used.
- **Guarantees (discrete setting, idealised).**
  - Theorem 1: if d(Z, C) = X a.s. and Z ⊥ C, then I(X;Z) = H(X|C) = H(U). This is maximal among C-independent deterministic representations (Cor. 1) and gives "perfect nonlinear guardedness" (Def. 1, Cor. 2).
  - Prop. 2: under Z ⊥ C, I(X;Z) = H(X|C) − H(X|Z,C).
  - Prop. 1 (imperfect regime): small reconstruction MSE implies approximate preservation, under a Lipschitz left-inverse assumption.
  - Prop. 3: MI is not monotone in HSIC.
- **Experiments.** Multiple datasets, sensitive attributes and language models. The authors claim better utility–erasure trade-offs than supervised competitors and faster convergence.
- **Relevance.** This is the closest existing formalisation of "remove C from the representation but keep everything recoverable once C is supplied". The condition H(X|Z,C) = 0 implies that the separately observed concept fully substitutes for what was erased, for any task. It does not analyse the downstream incremental value of the concept.

## Additional 2026 papers found by the mandatory searches
- **MANCE: Manifold Aware Concept Erasure** [A + method skim]. Avitan, Goldberg, Elazar. arXiv 2607.03973 (2026).
  - Iterative classifier-guided updates projected onto an estimated data manifold. MANCE+ and MANCE++ prepend LEACE or other closed-form stages.
  - 119 settings, including 13 LMs and 40 CelebA-CLIP attributes (vision). Code: github.com/MatanAvitan/mance.
  - Relevance: shows erasure applied to vision (CLIP) representations; leakage–surgicality trade-offs.
- **MUtE: A Dual Framework for Concept Erasure and Counterfactual Interventions** [A + skim]. Saillenfest. arXiv 2609.11253 (2026). Code: github.com/toinesayan/MUtE.
  - Optimal-bound piecewise erasure for discrete concepts in continuous spaces, with dual counterfactual maps.
- **Closed-Form Concept Erasure via Double Projections** [A]. Zhang, Cheng, Wang, Liu et al. arXiv 2604.10032 (2026).
  - Weight editing in diffusion and flow models. **Low relevance**: it erases generative concepts, not a representation used for prediction.
