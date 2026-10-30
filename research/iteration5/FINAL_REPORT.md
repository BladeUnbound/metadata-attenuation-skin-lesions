# Iteration 5: final report — graph-conditional metadata utility and audit of the existing paper

*Compiled 2026-10-29 from results/E40–E48 JSON records. The pre-registrations (REGISTRY.md) were written 21:48–22:30 IST on 2026-10-28, before each experiment was implemented; times were verified from file metadata. Stage reports: STAGE1_REPORT.md, STAGE3_5_REPORT.md. Theory: THEORY_AUDIT.md. Literature: LITERATURE.md.*

## 1. Summary for decision-making

- **The paper's main result survives a stricter audit.**
  - H1 (metadata add more to colour histograms than to ConvNeXt-T) holds on every image set under the leak-free estimator, under every fold assignment and both weightings, and on complete cases.
  - It also holds under the alternative label definition on 5 of 6 testable datasets, the exception being MILK10k melanoma-vs-rest, where metadata carry almost no information to begin with.
- **H2** (ResNet-18 vs ConvNeXt-T) is fragile: it depends on the estimator (ISIC 2020), the label (ISIC 2019), the fold assignment (MSK 2020) and the weighting (ISIC 2019).
- **The graph extension fails its pre-registered tests.** Graph statistics do not predict per-lesion metadata benefit beyond the fitted models (H4: 0 of 7). Per-sample prediction of incremental feature value is already established (Gadgil, Covert and Lee, ICLR 2024). The graph should not become the paper's mathematical contribution.
- **The strongest new result is mechanistic** (E45, exploratory). Erasing the one linear direction that encodes age from the image representation raises age's added value 3–7× on the three datasets where age is decodable, far more than performance-matched random erasure. The dataset where age is not decodable (PAD-UFES-20) shows no effect. This is interventional evidence for the redundancy term in Proposition 1, for age.
- **The interpretability test has to change.** A fixed, centred Gaussian blur passes the published shift-null test on 88% of images. The mask-swap null passes it on exactly 5%. Real activation maps pass the swap null on 10–19% of images, against 37–57% under the shift null.
- **Smaller corrections to the paper:**
  - a leak in the second stage (small; corrected estimator now primary);
  - an unstated image-vs-group weighting and a missing-value encoding that each move HIBA's floor (3.8 → 12.7–13.0 ×10⁻³ nats without missing indicators);
  - fold-assignment spread omitted from the CIs;
  - a retained-fraction range that depends on the label definition;
  - the E30 recoverability claim, which holds only among variables with standalone value;
  - Proposition 1's interpretation overstated.

## 2. Estimator audit and corrected baseline (Stage 1; E40)

See STAGE1_REPORT.md.
- All 120 stored results were reproduced exactly.
- **The leak:** the second stage was trained on scores from first-stage models that had seen the test fold's labels. Strict cross-fitting changes Δ̂ by at most 0.42 CI half-widths, mostly upward at deep rungs; the original slightly overstated attenuation.
- **One H2 conclusion changes:** ISIC 2020, Holm 0.092 → 0.028, a 0.36 ×10⁻³-nat difference at the edge of bootstrap resolution.
- **Registered decision:** corrected estimator primary; the paper keeps its pre-registered results and adds a methods note.
- **Assumptions linking Δ̂ to information** (THEORY_AUDIT.md §3):
  - E[Δ̂] = I(Y;A | s) + KL(p(y|s) ‖ q0) − KL(p(y|s,a) ‖ q1). So Δ̂ estimates conditional mutual information only under well-specified second stages and negligible estimation error.
  - Even then the target is conditional on the one-dimensional score s, not on the representation S. They coincide only under score sufficiency.

## 3. Replication matrix (Stage 2; E40, E41)

✓ holds, ✗ fails, ~ direction holds but not significant, n/a not derivable. Tests are one-sided paired grouped bootstrap, uncorrected unless stated; fold assignment 0 unless stated.

**H1: Δ(HSV) > Δ(ConvNeXt-T)**

| Condition | HAM | ISIC 2020 | ISIC 2019 | PAD | HIBA | MSK | MILK derm | MILK clin |
|---|---|---|---|---|---|---|---|---|
| Original estimator (pre-registered, Holm) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Corrected estimator (Holm) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fold assignments 0/1/2 (point estimate > 0) | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ |
| Group-weighted (point estimate) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Alternative label | ✓ (MAL; + akiec ✓) | n/a | ✓ (MAL; + AK ✓) | ✓ (MEL, 52 positives) | ✓ (MEL, p 0.046) | ✓ (MEL) | ✗ (MEL, p 0.17) | ✗ (MEL, p 0.52) |
| Complete cases, corrected | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**H2: Δ(ResNet-18) > Δ(ConvNeXt-T)**

| Condition | HAM | ISIC 2020 | ISIC 2019 | PAD | HIBA | MSK | MILK derm | MILK clin |
|---|---|---|---|---|---|---|---|---|
| Original (Holm; HAM unadjusted) | ✓ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| Corrected (Holm; HAM unadjusted) | ✓ | **✓** | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ |
| Fold assignments 0/1/2 (point > 0) | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓✓ | ✓✓**✗** | ✓✓✓ | ✓✓✓ |
| Group-weighted (point) | ✓ | ✓ | **✗** (−0.05) | ✓ | ✓ | ✓ | ✓ | ✓ |
| Alternative label | ✓ | n/a | **✓** (MAL) | ✓ (p 0.044) | ✗ | ✗ | ✗ | ✗ |
| Complete cases, corrected | ✓ | ✓ | ✗ | ✗ | ✓ | ✓ | ✓ | ✓ |

The MILK10k alternative-label failures arise because melanoma-vs-rest metadata information is 1.8 ×10⁻³ nats there (flagged small); there is nothing to attenuate.

**Other claims**

| Claim | Status |
|---|---|
| "Almost none remains" at ConvNeXt-T for HIBA and HAM10000 | **Narrow.** HIBA: ρ 0.017 / 0.046 / 0.042 across fold assignments (corrected), group-weighted Δ̂ < 0, but Δ̂ 12.7–13.0 without missing indicators (E47). HAM10000: ρ ≈ 0.08–0.09, so "small" rather than "almost none". |
| "About half remains" for PAD-UFES-20, MSK 2020, ISIC 2019 | Survives for the published labels (ρ 0.39–0.52 across fold assignments). **Label-dependent:** ISIC 2019 malignant-vs-benign gives ρ = 0.20 [0.14, 0.25]. |
| ρ range "a property of the dataset, its label definition and its metadata" | Supported; the label component is now demonstrated directly. |
| P1 (clinical images retain more than dermoscopic) failed | Unchanged under the corrected estimator (Δ̂ at ConvNeXt-T: clin 6.30 < derm 8.29 ×10⁻³ nats, same lesions). |
| P2 (lesion diameter retained < 0.5 on MSK 2020) | Not re-tested in this iteration. |
| Fine-tuning results (E14, E14b, E31c/d) | Share the R1 leak pattern. Not re-run: the frozen-feature leak was small, and a re-run costs ~10–15 h GPU. Disclose. |

**Dependence on a few datasets:** H1 holds without exception across conditions. H2's support comes mainly from PAD-UFES-20, HIBA, MILK10k and HAM10000, and three of those four have the largest ρ changes or label-informative missingness.

## 4. Graphs, local ambiguity and prediction of benefit (Stages 3–5; E42, E44)

See STAGE3_5_REPORT.md. Registered decisions at ConvNeXt-T, 7 counting units:
- **H1 (local label entropy ↔ benefit) fails:** 4 of 7.
- **H2 (image uncertainty ↔ benefit) holds:** 7 of 7, a check implied by I(Y;A | S=s) ≤ H(Y | S=s).
- **H4 (graph statistics add predictive value) fails:** 0 of 7.
- **H3 (exploratory):** with colour histograms, ambiguity tracks benefit more closely on 5 of 8 sets.
- **Post hoc:** the fusion model's own label-free expected gain K_i is the best or near-best ranker on 7 of 8 sets. Its top 25% of lesions carries 76–105% of total benefit on six sets.

**E44 (smoothness, exploratory, interpretation fixed in advance).** Normalised Dirichlet energy of Δ_i relative to the label × uncertainty-stratified null:

| | HAM | ISIC 2020 | ISIC 2019 | PAD | HIBA | MSK | MILK derm | MILK clin |
|---|---|---|---|---|---|---|---|---|
| HSV | 0.986* | 0.986* | 0.976* | 1.003 | 0.979* | 1.002 | 0.989* | 0.973* |
| ConvNeXt-T | 0.997 | 0.999 | 0.989* | 1.002 | 1.005 | 0.994 | 0.995* | 0.997 |

\* stratified permutation p < 0.05. Reference: the label itself has ratio 0.49–0.96 and image uncertainty 0.07–0.75 under the free null.

On strong representations, metadata benefit carries essentially no local organisation beyond label and image uncertainty: at most 1.1% below the null, on 2 of 8 sets. On colour histograms there is a little (1–3%, 6 of 8). Energy measures how much benefit differs between graph neighbours; it does not say why.

## 5. Mechanism (Stage 7; E45, exploratory)

Linear concept erasure (LEACE) fitted per outer training fold; dose t ∈ {0, 0.25, 0.5, 0.75, 1}; strictly cross-fitted estimator. Values are the added value of the erased variable, ×10⁻³ nats, ConvNeXt-T (ResNet-18 in brackets):

| Dataset | Age decodable at t = 0 (R²) | Δ_age t = 0 → 1 | Same-rank random | Performance-matched random (rank) | Manipulation check |
|---|---|---|---|---|---|
| HAM10000 | 0.37 | 2.01 → **13.19** [7.90, 18.26] (2.56 → 14.17) | 2.02 | 2.89 (64) | R² → 0.00 |
| ISIC 2019 | 0.35 | 3.22 → **10.45** [5.60, 15.60] (3.67 → 10.66) | 3.22 | 3.53 (32) | R² → 0.00 |
| MILK10k derm | 0.31 | 12.44 → **69.02** [57.90, 79.26] (17.48 → 69.50) | 12.44 | 35.06 (256)† | R² → 0.00 |
| PAD-UFES-20 (complete cases) | **0.02** | 0.64 → 0.88 (ns) | 0.62 | 0.59 (1) | nothing to erase |

† Under-matched: the random erasure raised image log-loss to 0.497 against the concept erasure's 0.518. For ResNet-18 it over-matched (0.545 vs 0.538), and concept erasure still exceeded it (69.5 vs 52.8).

- **Dose–response is monotone** in every decodable case (HAM10000 ConvNeXt-T: 2.0, 2.5, 3.7, 7.3, 13.2).
- **Specificity.** Erasing age raises age's value far more than the other variables' (HAM10000: age +11.2; site +1.5; sex +0.6).
- **Restoration.** t → 0 returns exactly to baseline (sanity check).
- **Side effect.** Removing the single age direction also costs image performance: HAM10000 image log-loss 0.232 → 0.246; MILK10k 0.424 → 0.518. The image models use age-correlated features for diagnosis.
- **Sex** is decodable (AUROC 0.74–0.78) but has almost no added value. Erasing it changes little, consistent with there being nothing to restore.
- **Site** erasure raises site's value on HAM10000 (1.5 → 8.4, against 4.1 for an over-matched random control). It is less specific (age +4.4) and noisy elsewhere (PAD-UFES-20: random control larger).

**Reading, under the stated design assumptions.**
- For age, removing linearly decodable age information from a fixed representation restores the metadata's added value. A matched loss of image performance does not.
- This is the substitution (R(S) > 0) pathway of Proposition 1, observed by intervention rather than correlation, for one variable on three datasets.

**What it does not establish:**
- That training learns age *in order to* diagnose.
- That nonlinear age information is absent after erasure (LEACE removes linear information only).
- That other variables behave the same way.
- Paired CIs of concept-vs-control differences were not computed; per-observation arrays were not saved. The CI of Δ_age at t = 1 excludes the matched control's point value in all three decodable datasets.

## 6. Recoverability re-analysis (Stage 8; E46)

| Variant | Rows | Partial Spearman (controls: I_var, image AUROC) | Jackknife sign stable | Mixed-model coefficient (dataset random intercept, variable fixed effects, ranks) |
|---|---|---|---|---|
| A: as run (I_var CI > 0; Δ_var / I_var) | 95 | −0.60 | yes (−0.52 to −0.63) | −0.49 [−0.67, −0.31] |
| D: filtered, absolute Δ_var | 95 | −0.51 | yes | −0.38 [−0.57, −0.19] |
| B: all rows, absolute Δ_var | 160 | −0.28 | yes | −0.10 [−0.25, 0.05] |
| C: all rows, Δ_var / (I_var + median) | 160 | −0.16 | yes | −0.03 [−0.21, 0.15] |

The association is robust among variables with demonstrable standalone value and weak otherwise. The filter is defensible, since a retained share is undefined when there is nothing to retain. But it was not in the dated registry, and the 95 rows include each (dataset, variable) five times, once per rung, so rows are not independent beyond the dataset intercept. **Recommendation:** keep the claim, restricted explicitly to "variables with standalone value", state that the filter was set at implementation, and call it descriptive.

## 7. Missingness and source dependence (Stage 9; E47)

Source: results/E47_missingness_source.json. Corrected estimator, fold assignment 0. The first run crashed on a read-only-array bug before producing results; the fix is recorded in overnight/STATUS7.md.

**Missingness is label-informative**, with counts:
- PAD-UFES-20: all 804 images missing sex and skin type are benign (72.9% malignant among observed). Unknown growth or change histories are 88% malignant.
- HAM10000: 0 of 57 images with missing sex are melanoma (11.2% among observed).
- ISIC 2020: 0 of 65 images with missing sex are melanoma.
- ISIC 2019: images without a lesion ID (older archive images) have some metadata missing in 44% of cases, against 2% for BCN and MSK.

**Encoding matters, in both directions** (×10⁻³ nats; pre-registered indicators / no indicators / multiple imputation, m = 5):

| | HSV | ConvNeXt-T |
|---|---|---|
| PAD-UFES-20 | 386 / 255 / 228 | **207 / 112 / 91** |
| ISIC 2019 | 19.3 / 16.4 / 16.2 | 9.8 / 8.0 / 7.5 |
| HIBA | 72 / 84 / 85 | **3.8 / 12.7 / 13.0** |
| HAM10000 | 18.6 / 18.9 / 18.8 | 2.4 / 3.7 / 3.3 |
| MILK10k derm / clin | 52 / 52 / 51; 58 / 58 / 58 | 8.3 / 11.4 / 11.2; 6.3 / 8.5 / 8.5 |
| ISIC 2020, MSK 2020 | unchanged | unchanged |

- **Indicators inflate Δ̂ where missingness carries the label** (PAD-UFES-20, ISIC 2019).
- **The indicator and "unknown" parameters depress Δ̂ at strong representations** on HIBA, HAM10000 and MILK10k, most likely through extra parameters that do not pay off. HIBA's "almost none remains" floor is therefore partly an encoding artefact.
- **H1's direction holds under every encoding.**
- **Complete-case analysis is not a neutral fix.** PAD-UFES-20's complete cases are 1,089 of 2,298 images, mostly biopsied, so they are a different population.

**Source dependence:**
- HAM10000 leave-one-source-out, ConvNeXt-T: Δ̂ is negative when vidir_modern (−14.4 [−32.1, 0.9]) or rosendahl (−6.5 [−16.8, 3.6]) is held out, and positive for vidir_molemax (7.6 [6.1, 9.1]) and vienna_dias (18.0 [3.2, 30.6]).
- The four sources differ sharply in melanoma prevalence (molemax 0.6%; the others 15–20%), so held-out-source results mix prevalence shift with metadata transfer.
- ISIC 2019: BCN → MSK −10.5 [−31.8, 9.1]; MSK → BCN +29.2 [11.5, 49.6].

**Interpretation.** Within-dataset added value does not transfer reliably across acquisition sources even inside one dataset. This agrees with the paper's E33 cross-dataset result, which it already describes as mixed and confounded. Provenance and prevalence associations are part of what the second stage learns. These analyses cannot separate a genuine diagnostic signal from a provenance shortcut; the paper's existing hedge is appropriate and should cite this within-dataset evidence.

## 8. Shift-null audit (Stage 10; E48)

| Backbone | Images | Shift null: share p ≤ 0.05 (published E15) | Swap null | Centred Gaussian: shift | Centred Gaussian: swap |
|---|---|---|---|---|---|
| ResNet-18 | 2,225 | 0.55 (0.53) | **0.16** | 0.88 | 0.05 |
| ResNet-50 | 2,225 | 0.37 (0.36) | **0.10** | 0.88 | 0.05 |
| ConvNeXt-T | 2,225 | 0.57 (0.56) | **0.19** | 0.88 | 0.05 |

- **The positive control confirms the analytic argument** (THEORY_AUDIT.md §5). A map that carries no image information passes the shift null on 88% of images because lesions and maps are both centred.
- **Some lesion-specific evidence remains.** Under the swap null (masks of other images with similar lesion area), 10–19% of real maps are significant, above the nominal 5%.
- **Recommendation:**
  - Replace the shift null with the swap null in the paper.
  - Report 10–19% instead of 36–56%.
  - Keep "pixel permutation inflates significance" as a relative statement.
  - Move the CAM section to an appendix: it does not measure metadata utility. Its only link is that missed melanomas are smaller lesions.

## 9. Literature and novelty (Stage 12)

See LITERATURE.md.
- **Per-sample incremental feature value is established** (Gadgil, Covert, Lee, ICLR 2024, Lemma 1; Covert et al., ICML 2023), with patient demographics as an example.
- **Sample-level multimodal interaction estimation exists** (Yang, Wang, Hu, arXiv 2506.17248, listed as ICML 2025).
- **kNN conditional-MI estimators are standard** (Frenzel–Pompe, via secondary sources).
- **No precedent found for:**
  - per-lesion metadata benefit related to an image-representation graph;
  - smoothness of a fusion-gain signal;
  - concept erasure used to test whether metadata value is restored.

  Absence from a search is weak evidence. The erasure design should be re-searched specifically (terms: "concept erasure" with "multimodal", "auxiliary features" or "tabular") before any novelty claim.

## 10. Formal proposal (Stage 6), stated as an empirical framework

No new theorem is proposed. The framework combines standard facts with falsifiable predictions; THEORY_AUDIT.md §4 gives each fact.

1. **Local utility.** u(s) = I(Y;A | S = s), with Δ(S) = E u(S) and u(s) ≤ H(Y | S = s).
   - *Prediction:* benefit concentrates where the image is uncertain.
   - *Status:* confirmed (H2, 7 of 7). Concentration was shown post hoc via K_i.
2. **Entropy–efficiency decomposition.** Δ(S) = H(Y | S) · η(S).
   - *Prediction (redundancy):* attenuation is driven mainly by falling η.
   - *Status:* exploratory, proposed after seeing HAM10000 aggregates. η accounts for 67–91% of log-attenuation on 7 of 8 sets (PAD-UFES-20: 39%). Needs uncertainty intervals.
3. **Substitution, tested by intervention.** If A's value at S is reduced because S encodes A (R(S) > 0), then erasing A's trace from S must raise Δ_A more than equal-cost random erasure.
   - *Status:* confirmed for age on 3 of 3 datasets where age is decodable; null where it is not. Not shown for sex (no value to restore) or, specifically, for site.
4. **What would falsify the framework:**
   - attenuation driven by H rather than η across datasets;
   - erasure of a decodable, valuable variable failing to restore its value;
   - restoration that random erasure matches.

## 11. Revised contribution statement (verified results only)

1. **Pre-registered, multi-dataset evidence** that patient metadata add markedly less to frozen deep image representations than to colour histograms. This survives a strictly cross-fitted estimator, three fold assignments, both weightings, complete cases, and an alternative label definition on 5 of 6 testable datasets. The difference between deep representations (ResNet-18 vs ConvNeXt-T) is small and condition-dependent.
2. **An exact accounting identity** (Proposition 1, with its scope corrected), plus **interventional evidence for its substitution term.** Erasing the linear trace of age from the image representation restores age's added value several-fold, specifically and dose-dependently. Performance-matched random erasure does not.
3. **A characterisation of where metadata help.** Benefit is concentrated in lesions the image model finds uncertain, and can be ranked without labels. The image-representation graph adds nothing beyond the fitted models (a pre-registered negative result).
4. **Methodological findings for this literature:**
   - stacking with shared folds leaks test labels;
   - image- vs group-weighting can flip conclusions;
   - missing-value encoding can raise or lower added value by a factor of two to three;
   - retained fractions depend on the label definition;
   - circular-shift nulls for saliency overlap are invalid for centred lesions (positive-control demonstration).

Withdrawn or narrowed:
- the graph statistic as a method;
- HIBA's floor as a dataset property (it depends on weighting and missing-value encoding);
- "almost none remains" for HAM10000;
- the 36–56% "significant" saliency maps;
- the E30 claim beyond variables with standalone value;
- "added value attenuates as S improves" as a consequence of Proposition 1.

## 12. Remaining work, costs and publication risks

| Priority | Item | Cost | Why |
|---|---|---|---|
| 1 | Revise the paper (methods note on E40; H2 table with corrected column; weighting statement; swap-null interpretability in an appendix; Proposition 1 wording; E45 as a new results subsection; E30 scope; label dependence of ρ) | ~0.5 day | Bring the paper in line with verified results |
| 2 | E45 paired CIs (concept vs matched control) with saved per-observation arrays; more seeds for the random controls; exact matching by interpolating rank | ~1–2 h CPU | The mechanistic result is the paper's strongest new claim and needs paired inference |
| 3 | E45 extended to fine-tuned representations (does fine-tuning re-encode age after erasure?) and to nonlinear probes after erasure | ~2–4 h GPU | Linear-only erasure is the main caveat |
| 4 | Uncertainty intervals for the η decomposition; repeat across fold assignments | ~0.5 h CPU | Currently descriptive |
| 5 | Nested-CV variance (Bates et al. 2024) for headline CIs | ~2 h CPU | CIs omit training variability (R4) |
| 6 | Strictly cross-fitted fine-tuning (E14/E14b/E31) | ~10–15 h GPU | Only if a reviewer insists; the frozen leak was small |
| 7 | P2 under the corrected estimator | minutes | Completes the replication matrix |
| — | PID analysis (6.1) | — | Not recommended: no distinct testable prediction beyond the η decomposition and E45; estimator instability on continuous S |

**Publication risks:**
1. Novelty. The attenuation result is partly anticipated (Ahammed et al. 2025) and per-sample utility is established (Gadgil et al. 2024). The paper rests on rigour, the interventional mechanism, and the methodological findings.
2. H2's fragility invites "your ladder is just image AUROC".
3. Most datasets come from the ISIC archive. Patient overlap across collections cannot be tested.
4. Length: 13 pages before these additions. A journal (e.g. IEEE JBHI) remains the realistic venue.
5. The author block is still needed.

**Update 2026-10-29 10:05 IST.** Priority 1 (paper revision; draft v6, paper/main.pdf) and priority 2 (E45b paired CIs: erasure beats the matched control in 6 of 8 pairs, every pair where age is decodable) are done. The remaining items are unchanged.
