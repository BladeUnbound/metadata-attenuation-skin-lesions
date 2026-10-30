# Stages 3–5 report: representation graphs, local ambiguity, predicting metadata benefit (E42)

*Written after queue 6 finished (22:38 IST, 2026-10-28). Registered in REGISTRY.md at 21:59 IST, before implementation. Sources: results/E42_*.json, data/<ds>/e40/e42_convnext_tiny.npz. E44 (smoothness) is running and is reported separately.*

## Registered decisions (ConvNeXt-T, fold assignment 0, corrected E40 estimator, 7 counting units, Holm)

| Hypothesis | Rule | Units holding | Verdict |
|---|---|---|---|
| H1: local label entropy (training labels, inductive graph) correlates positively with realised benefit Δ_i | ≥ 5 of 7 | 4 (ISIC 2020, ISIC 2019, PAD-UFES-20, MILK10k); not HAM10000 (Holm p 0.14), HIBA (0.46), MSK 2020 (0.53) | **Fails** |
| H2: image predictive entropy h(q0) correlates positively with Δ_i (manipulation check) | ≥ 5 of 7 | 7 | **Holds**, as the bound u(s) ≤ H(Y \| S=s) requires |
| H4: graph statistics (LE, NC, MS, G) reduce held-out squared error of Δ_i beyond K_i, h(q0), margin, norm, density | ≥ 4 of 7 | 0 after Holm (nominal p ≈ 0.01 on MSK 2020 and ISIC 2020; MILK10k derm 0.003 but clin 0.37) | **Fails** |

Spearman(LE, Δ) at ConvNeXt-T ranges from −0.002 (MSK 2020) to 0.26 (PAD-UFES-20). The k = 10, k = 50 and binary-weight sensitivities give the same pattern (e.g. HAM10000 0.020/0.027/0.021; ISIC 2019 0.090/0.116/0.100).

## H3 (exploratory): does the ambiguity–benefit relation depend on the representation?

Spearman(LE, Δ), HSV minus ConvNeXt-T, paired grouped bootstrap:

| HAM | ISIC 2020 | ISIC 2019 | PAD | HIBA | MSK | MILK derm | MILK clin |
|---|---|---|---|---|---|---|---|
| +0.12 [0.09, 0.15] | +0.07 [0.05, 0.10] | +0.01 (ns) | **−0.20** [−0.26, −0.14] | +0.16 [0.06, 0.25] | −0.01 (ns) | +0.05 [0.02, 0.08] | +0.05 [0.02, 0.08] |

With a weak representation, local ambiguity in colour-histogram space tracks where metadata help more closely on 5 of 8 sets. PAD-UFES-20 is the exception, and its benefit is dominated by label-informative missingness rather than image ambiguity. Between ResNet-18 and ConvNeXt-T, differences are small and mostly not significant (HAM10000 +0.04, HIBA +0.08).

## Why H4 failed, and what predicts benefit

1. **The registered ridge predictors are weak.** Held-out R² is negative on 6 of 8 sets for both B1 and B1+G. They are trained on inner Δ_j from models fitted on 64% of the data, whose level differs from the outer Δ_i, and the selected penalty often sits at the grid edge. The registered comparison (paired squared-error difference) remains valid, but it compares two poor predictors.
2. **Post hoc, labelled as such: rank correlation with realised Δ_i,** which is insensitive to that level offset:

   | | K | h(q0) | B1 | B1+G | G | LE |
   |---|---|---|---|---|---|---|
   | HAM10000 | **0.128** | 0.095 | −0.017 | 0.032 | 0.110 | 0.025 |
   | ISIC 2020 | 0.149 | 0.164 | 0.142 | 0.131 | **0.190** | 0.108 |
   | ISIC 2019 | **0.214** | 0.167 | 0.154 | 0.173 | 0.155 | 0.102 |
   | PAD-UFES-20 | **0.687** | 0.533 | 0.675 | 0.658 | 0.133 | 0.259 |
   | HIBA | **0.154** | 0.130 | −0.024 | −0.018 | −0.018 | 0.022 |
   | MSK 2020 | **0.178** | 0.089 | 0.099 | 0.022 | 0.145 | −0.002 |
   | MILK10k derm | **0.182** | 0.171 | 0.089 | 0.092 | 0.116 | 0.120 |
   | MILK10k clin | 0.163 | **0.167** | 0.125 | 0.142 | 0.120 | 0.156 |

   K_i, the fusion model's own label-free expected gain, is the best or near-best ranker on 7 of 8 sets. Graph-local benefit G is best only on ISIC 2020. Learned combinations rank worse than K alone.
3. **Concentration (registered secondary).** Ranking lesions by K_i and using metadata only for the top 25% captures 76–105% of the total realised benefit on six sets (HAM 0.90, ISIC 2020 0.97, ISIC 2019 1.05, MSK 0.86, MILK 0.82/0.76), and 70% on PAD-UFES-20. Random selection captures 25%. HIBA's total benefit at ConvNeXt-T is near zero (3.8 ×10⁻³ nats), so its capture ratios are unstable and not interpreted.

## Interpretation

- Metadata benefit is concentrated: a minority of lesions carries most of it, and those lesions are where the image model is uncertain. This is required by u(s) ≤ H(Y | S = s), and H2 confirms it.
- A kNN graph over the representation does not locate those lesions better than the fitted models already do. This matches the theoretical expectation in AUDIT.md §5: under calibration E[Δ_i | s, a] = K_i, so label-free statistics can add only through local misspecification, and no consistent misspecification was detectable.
- Together with the literature result (dynamic feature selection already predicts per-sample incremental value), **the graph statistic should not be pursued as the paper's mathematical contribution.** The registered gate for Stage 6 ("only if graph statistics add to K_i and margin on held-out groups") is not met.

## What survives as a contribution candidate (for Stage 6, empirical)

- **Concentration of metadata utility:** most of the benefit sits in a minority of lesions, identifiable without labels by K_i or by image uncertainty. This is a practical, testable statement ("collect metadata for the uncertain 25%"). Its clinical value is not established here.
- **Representation dependence of where the benefit sits (H3):** with weak features, benefit follows local label ambiguity; with strong features it follows it less, on 5 of 8 sets.
- **The entropy/efficiency decomposition** (STAGE1_REPORT.md finding 4): attenuation is mostly a fall in the share of remaining uncertainty that metadata resolve, not a fall in the uncertainty itself.

**Caveat on K_i:** it over-predicts the level of benefit by 1.1–7× (STAGE1_REPORT.md), so it is useful for ranking but not as a calibrated estimate of the gain.

## Recommendation

**Stop** the graph-statistic line as a methods contribution. **Revise** the extension toward an empirical analysis of concentration and the decomposition, with uncertainty intervals and across fold assignments, before anything is written into the paper. E44 is reported for completeness under its pre-fixed interpretation.
