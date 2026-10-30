# Stage 12 (first pass): literature and novelty audit for the graph extension

*2026-10-28. Each entry says what was verified and how. Nothing here is cited in the paper until its primary source has been read.*

## Closest prior work

| Work | Verified how | What it already does | Consequence for us |
|---|---|---|---|
| Gadgil, Covert, Lee, "Estimating Conditional Mutual Information for Dynamic Feature Selection", ICLR 2024 (arXiv 2306.03301) | Method section read from the PDF | A value network v_i(x_S) is trained by regressing the **per-sample incremental loss improvement** Δ(x_S, x_i, y) = ℓ(f(x_S), y) − ℓ(f(x_{S∪i}), y). Lemma 1: with the Bayes predictor and cross-entropy, E[Δ \| x_S] = I(y; x_i \| x_S = x_S). Theorem 2 adds "prior information" z, with patient demographics as the example; used to choose and stop feature acquisition. | Predicting per-observation metadata benefit from the image representation **is** this problem, with our Δ_i as their regression target. A graph statistic that predicts Δ_i is a nonparametric (kNN) value network. **No method-novelty claim is tenable**; any graph contribution must be framed as an empirical analysis, and H4 must compare against a parametric estimate of the same quantity. |
| Covert, Qiu, Lu, Kim, White, Lee, "Learning to Maximize Mutual Information for Dynamic Feature Selection", ICML 2023 (PMLR 202) | Abstract and listing | Greedy per-instance selection by conditional mutual information, amortised. | Same as above. |
| Yang, Wang, Hu, "Efficient Quantification of Multimodal Interaction at Sample Level", arXiv 2506.17248 (comments field: accepted to ICML 2025) | Abstract only | Sample-level redundancy / uniqueness / synergy from pointwise information (LSMI estimator). | Any per-lesion redundancy analysis (6.1) must cite it and compare; full text still to read. |
| Liang et al., "Quantifying & Modeling Multimodal Interactions: An Information Decomposition Framework", NeurIPS 2023 (arXiv 2302.12247) | Abstract and proceedings listing | PID (redundancy, uniqueness, synergy) for multimodal data and models; exact convex estimator for discrete support, approximate for continuous; synthetic validation. | Candidate framework for 6.1; dataset-level, not per-sample. |
| Frenzel & Pompe (2007); Vejmelka & Paluš (2008): kNN estimators of conditional mutual information | Secondary sources only (Runge, arXiv 1709.01447; Molavipour et al., arXiv 2006.07225) | Local, per-point kNN estimation of CMI. | Graph-local estimation of I(Y;A \| S = s) is an application of established estimators; primary papers still to read. |
| Ahammed et al., CVPR-W 2025 (already cited in the paper) | Read earlier | Metadata gain smaller for pretrained, fine-tuned backbones. | Dataset-level only. |

## No direct precedent found (two extended and several standard searches)

- Per-lesion (per-observation) metadata benefit in skin-lesion classification, related to position in an image-representation graph.
- Smoothness (Dirichlet energy) of a fusion-gain signal on a kNN graph of image embeddings. Dirichlet energy of signals on kNN graphs is itself standard (e.g. Xu et al., NeurIPS 2023, for feature selection; arXiv 2006.06885 as a smoothness index over embeddings).

Absence of evidence from a search is weak; this list will be re-checked before any claim is written.

## Methods references needed by the plan (verified at listing/abstract level)

- Bates, Hastie, Tibshirani, "Cross-validation: what does it estimate and how well does it do it?", JASA 119(546):1434–1445, 2024: naive CV intervals can undercover badly; nested CV variance estimate. Relevant to R4.
- Ravfogel et al., "Null It Out" (INLP), ACL 2020; Belrose et al., "LEACE: Perfect linear concept erasure in closed form", NeurIPS 2023. LEACE gives a closed-form erasure that blocks all linear probes with minimal change; preferred for Stage 7 ablation–restoration. Both guarantee only linear erasure.

## Consequence for the brief's Stage 6

Two mathematical facts frame what can be claimed (both standard, neither new):
1. E[Δ_i | s_i, a_i] = KL(q1 ‖ q0) for a calibrated fusion model; E[Δ_i | s_i] = I(Y;A | S = s_i) for Bayes predictors (Gadgil et al., Lemma 1, in our notation).
2. I(Y;A | S = s) ≤ H(Y | S = s): metadata can help at a point only as much as the image leaves uncertain there. So "metadata help most at ambiguous lesions" is partly forced by the mathematics; the empirical content lies in how tight the bound is and in whether ambiguity measured on a graph adds anything beyond the model's own predictive entropy and K_i.

A defensible contribution is therefore **empirical**: where in representation space the remaining metadata information sits, and whether representation strength removes it by resolving cases (shrinking H(Y \| S)) or by redundancy (shrinking the share I(Y;A \| S)/H(Y \| S) that metadata resolve). That decomposition uses an established normalised quantity; the question it answers about this literature appears to be open.
