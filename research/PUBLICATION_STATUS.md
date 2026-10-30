# Publication status

*Last updated: 26 Oct 2026, iteration 1.*

## Manuscript state

The current draft (`alzheimers_paper.pdf`) **cannot be submitted**. Every quantitative claim either rests on a leaky split or was never measured. The planned paper (an ADNI study of genetics vs MRI for progression) has no real-data results yet.

## Claims in the current draft

| Claim | Status | Evidence |
|---|---|---|
| 86.39% validation accuracy | **Wrong number, and inflated.** The saved model is 86.16% (epoch 29). At least 67% of validation images have a near-copy of the same slice in training. Near-copy leakage alone adds at least 5.6 pp for a comparable classifier. The exact split can't be recovered, so the 86.16% can't be re-measured either | Notebook logs; E02c, E08 |
| +11.3 pp after the first LR drop | **Wrong.** The real step is +7.8 pp | Notebook logs |
| SNN keeps 83.0%, 1.0 GMAC, 6.5% energy | **Never measured.** The numbers were hardcoded, and the weight transfer was broken | E01 |
| Central crop drops accuracy to 26.88% | Measured, but doesn't test the hippocampus | Review |
| "Radiogenomics" | **No genetic data used** | n/a |

## Verified so far (synthetic or engineering only)

- A correct Keras→PyTorch port of the trained model (E01).
- Validated tests for "does genetics add value": the likelihood-ratio test at a correct 5% rate, which requires out-of-fold MRI scores (E03).
- A clarified estimand: the added-value test can't separate genetics acting through atrophy from direct effects (E03). Double machine learning does not fix this at realistic sample sizes (E07b), so the paper should claim only "added value over the MRI model".
- Measured leakage in the original dataset (E02c, E08) and measured M5 training throughput (E06).
- A valid significance test for saliency-anatomy overlap (E05).
- Calibration metrics (E04) and a patient-level cohort interface with leakage checks (`pipeline/cohort.py`).

## Evidence still missing for the planned paper

1. ADNI MCI cohort: baseline 3D T1, APOE, genotypes for the polygenic risk score, and 36-month outcomes. **Needs a human data-access application.**
2. An external cohort (AIBL or OASIS-3), with the same need.
3. A choice of GWAS summary statistics for the polygenic risk score, made and recorded before any results are seen.
4. Real out-of-fold MRI scores, then the prespecified tests.
5. A systematic literature search on the mediated-vs-direct framing before claiming any novelty.

## Next actions that need you

- Apply for ADNI access, including genetic data.
- Apply for AIBL or OASIS-3 access.
- Decide whether I may download OASIS-1 (public, but under a data-use agreement) for 3D pipeline development.

---

## Iteration 2 update (27 Oct 2026)

**Candidate paper (Track A):** *Does patient metadata still help strong image models? Attenuation of auxiliary-information value with representation capacity in dermoscopic lesion classification.*

### Evidence in hand (public data, preliminary)
- A pre-registered analysis on HAM10000 with lesion-grouped CV: metadata adds 0.0186 log-loss for a colour-histogram model and about 0.002 for frozen deep backbones (CIs include 0). This is consistent across 3 fold assignments (E10).
- Robust to a source-shortcut control; paired attenuation from ResNet-18 to ConvNeXt is significant (E10b, exploratory).
- Supporting maths: the slope's sign separates mediated from direct auxiliary information (E09, MATH_NOTES).
- A methods component: calibrated spatial testing of saliency vs expert lesion masks (E05, E12, E12b).

### Missing before submission
1. Independent replication on a second dataset (confirmatory, specified in advance).
2. Fine-tuned (not only frozen) image models.
3. Full literature verification of novelty, including one possible counter-example.
4. Multiplicity-corrected tests.

### Venue thinking (provisional)
A methods or evaluation paper in medical image analysis or machine learning for health, rather than a clinical journal: the contribution is an analysis framework plus evidence, not a diagnostic tool. Reassess after replication.

### Claims that must not be made
- Any clinical efficacy claim.
- That metadata is useless in general.
- Anything about the original Alzheimer's/Kaggle results as generalisation.

---

## Iteration 4 update (28 Oct 2026)

**Draft v4** (`paper/main.tex`, 8 pages, compiles cleanly): *Do Patient Metadata Still Help Strong Image Models? Representation-Dependent Attenuation of Auxiliary Information in Skin Lesion Classification.*

### What the paper can claim
- Pre-registered: weak-vs-strong attenuation in HAM10000 and 6/6 independent replication datasets (Holm within registered families).
- Pre-registered, negative: within-deep attenuation only 3/6; imaging modality does not explain the retained share (P1 failed).
- Exploratory: fine-tuning and more training data shrink metadata value (HAM dose–response; HIBA; PAD-UFES-20 partly); bigger frozen backbones do not.
- Leakage: missingness carries the label (PAD-UFES-20 severely); complete-case analyses reproduce the main pattern.

### Claims that must not be made
- Clinical benefit or harm; causal effects of metadata; that metadata are useless; that modality is isolated anywhere except the MILK10k paired comparison; anything from E33 beyond "results under joint shift are mixed".

### Before submission
1. E31d and the extra fine-tuning runs (queued: `overnight/queue3.sh`).
2. Author block and affiliations (human).
3. Venue choice: 8 pages exceeds most IEEE conference limits; a journal (e.g. IEEE JBHI) or a conference with a supplement would fit.
4. A fresh literature search for 2026 work on metadata fusion vs backbone strength.

## Update 2026-10-29: draft v6 (iteration 5 folded in)
- **Added:**
  - strictly cross-fitted estimator (methods note, robustness appendix);
  - Proposition 1 wording corrected;
  - new sections on concept erasure (E45/E45b) and on where metadata help (E42/E44/E49);
  - interpretability moved to an appendix with the mask-swap null (E48);
  - narrowed claims (HIBA floor, E30 scope, label-dependent ρ).
- **Claims added:**
  - erasing age's linear trace restores its value beyond matched random erasure (exploratory, paired CIs);
  - benefit concentrates in uncertain lesions (post hoc);
  - pre-registered negative graph result.
- **Must not claim:**
  - a new per-sample utility estimator (Gadgil et al. 2024);
  - that training learns age in order to diagnose;
  - nonlinear erasure;
  - the published shift-null significance shares.
- **Before submission:** author block; venue (18 pages including appendices; consider a main paper plus supplement); optionally strictly cross-fitted fine-tuning (R7-11).
