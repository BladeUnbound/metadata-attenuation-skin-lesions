# Stage 1 report: estimator audit and corrected baseline (E40)

*2026-10-28 22:28 IST (file creation time). Sources: results/E40_*.json, data/<ds>/e40/*.npz. Registered in REGISTRY.md (21:48 IST) before implementation.*

## Verdict

- **Reproduction.** The original estimator, called unchanged, reproduced every stored pre-registered Δ̂: 120 of 120 cells (8 image sets × 5 rungs × 3 fold assignments) to 5 decimals.
- **The leak is real but small.** The strictly cross-fitted estimator changes Δ̂ by at most 0.42 of a CI half-width (HAM10000, ViT-B, fold assignment 2). No cell exceeds the registered threshold of half a CI width. At deep rungs the change is mostly upward: the original estimator slightly *overstated* attenuation (HAM10000 ConvNeXt-T: 1.74 → 2.38 ×10⁻³ nats at fold assignment 0). HSV is essentially unchanged.
- **Pre-registered conclusions.** H1 holds in every image set under both estimators. One H2 conclusion changes: ISIC 2020 (wave 1) goes from not holding (stored p = 0.046, Holm 0.092) to holding (p = 0.014, Holm 0.028). The underlying difference is 0.36 ×10⁻³ nats. My bootstrap re-run of the original estimator gave p = 0.058 against the stored 0.046, so this test sits at the edge of Monte Carlo resolution. Under the corrected estimator, H2 would hold in 4 of 6 independent datasets instead of 3 of 6.
- **Registered decision.** No H1 flip, and no cell beyond half a CI width. The corrected estimator becomes primary for all new work. The paper keeps its pre-registered results and gets a methods note reporting the corrected estimator and the ISIC 2020 H2 change.
- **Duplicate-merged ISIC 2019 groups** (10 merges): ConvNeXt-T Δ̂ 10.45 [4.30, 16.00] vs 9.80 [3.6, 15.9] (different folds). Immaterial.

## Corrected baseline, fold assignment 0 (×10⁻³ nats; HSV → ConvNeXt-T)

| Image set | Δ̂ mean | median Δ_i | Δ̂ positives | Δ̂ negatives | group-weighted | mean K_i | image → fusion AUROC (CNX-T) |
|---|---|---|---|---|---|---|---|
| HAM10000 | 18.55 → 2.38 | 11.18 → 0.48 | 138.3 → 15.2 | 3.6 → 0.8 | 15.8 → 2.2 | 23.9 → 7.1 | 0.887 → 0.890 |
| ISIC 2020 | 3.57 → 0.92 | 1.10 → 0.28 | 194.2 → 48.4 | 0.15 → 0.06 | 10.9 → 4.1 | 4.0 → 1.5 | 0.859 → 0.860 |
| ISIC 2019 (non-HAM) | 19.34 → 9.80 | 25.17 → 6.62 | 56.1 → 25.6 | 8.8 → 5.3 | 25.9 → 13.8 | 22.9 → 13.5 | 0.823 → 0.832 |
| PAD-UFES-20 | 386.4 → 206.9 | 517.4 → 135.5 | 468.7 → 256.5 | 312.4 → 162.2 | 423.2 → 236.4 | 415.1 → 230.5 | 0.874 → 0.960 |
| HIBA | 72.1 → 3.80 | 84.5 → 7.64 | 83.7 → 4.0 | 63.3 → 3.6 | 49.2 → **−0.84** | 100.6 → 26.9 | 0.928 → 0.930 |
| MSK 2020 | 31.9 → 17.4 | 42.7 → 10.1 | 224.2 → 117.4 | 4.1 → 3.0 | 31.9 → 17.4 | 37.1 → 20.8 | 0.727 → 0.769 |
| MILK10k derm. | 52.1 → 8.29 | 56.4 → 8.04 | 31.8 → 4.5 | 102.0 → 17.5 | = mean | 55.6 → 15.4 | 0.845 → 0.851 |
| MILK10k clin. | 58.4 → 6.30 | 64.7 → 5.07 | 36.9 → 3.0 | 111.2 → 14.5 | = mean | 62.4 → 12.5 | 0.842 → 0.848 |

Calibration, Brier score and per-seed values are in the JSON records. Fusion calibration slopes range from 0.87 to 1.00.

## Findings beyond the registered questions

1. **The estimand choice matters (R5).** Image-weighted and group-weighted Δ̂ differ materially where groups hold many images. HIBA at ConvNeXt-T: +3.8 vs −0.84 (the sign changes). ISIC 2020: 0.92 vs 4.09. The paper reports image-weighted Δ̂ without saying so.
2. **Training variability is material (R4).** Across the three fold assignments, Δ̂ at ConvNeXt-T spans 22.2 ×10⁻³ on PAD-UFES-20 (seed-0 CI half-width 27.1). MILK10k spans 4.6–4.8 (half-width 7.2–7.7) and HIBA 6.2 (half-width 16.5). The grouped bootstrap holds the fitted models fixed, so its CIs omit this source of variation (Bates et al., JASA 2024). Claims that rest on one fold assignment are affected; the HIBA floor is an example (fold assignments 0/1/2 give 3.8/10.0/9.1 corrected).
3. **K_i over-predicts realised benefit.** Mean K_i (the fusion model's own expected gain) exceeds mean Δ_i in every cell, by 1.1–7× (largest for HIBA ConvNeXt-T, about 7×). The fusion model shifts predictions further than the realised gains justify; it is over-confident about metadata. For H4, K_i is therefore a biased predictor of benefit, although its ranking may still be informative.
4. **Exploratory decomposition** (THEORY_AUDIT.md §4; proposed after HAM10000's E40 aggregates were seen). Δ = H(Y|S) · η, with H estimated by image-only held-out log-loss. Share of the HSV → ConvNeXt-T log-attenuation due to falling image uncertainty:

   | HAM | ISIC 2020 | ISIC 2019 | PAD | HIBA | MSK | MILK derm | MILK clin |
   |---|---|---|---|---|---|---|---|
   | 13% | 9% | 33% | 61% | 13% | 12% | 13% | 11% |

   On 7 of 8 image sets, most of the attenuation comes from metadata resolving a smaller share of the uncertainty the image leaves (η falls). It does not come from the image resolving more cases. Redundancy (the representation encodes what the metadata carry) is one explanation. Another is that a strong representation leaves its residual uncertainty in cases where metadata are uninformative. These need the local analyses (E42) and concept erasure (E45) to separate them. No uncertainty intervals yet; descriptive only.

## Risk-register updates (AUDIT.md §4)

| # | New status |
|---|---|
| R1 | Quantified: max 0.42 CI half-widths; one H2 conclusion (ISIC 2020) changes. Corrected estimator primary from here. |
| R2 | Fixed in E40; folded into R1's numbers. |
| R4 | Confirmed material (finding 2). Report fold-assignment spread with every headline Δ̂. |
| R5 | Confirmed material (finding 1). Report both weightings; HIBA's floor is weighting-dependent. |
| R7 | Checked: immaterial. |
| R18 | Fine-tuned scores share R1. Given its size on frozen features, a full re-run (~10–15 h GPU) is not justified now; disclose instead. |

## Recommendation

Proceed. Stage 2 (E41: alternative label definitions, and a replication matrix with both weightings and the fold-assignment spread) and Stage 3–5 (E42–E44, pre-registered) can both run on the saved per-observation arrays. Run E42 first, since it is the brief's main question, then E41.
