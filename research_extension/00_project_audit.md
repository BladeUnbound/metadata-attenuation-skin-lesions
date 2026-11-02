# 00 — Project audit before the MATE investigation

*Written 2026-10-30 10:28–10:40 IST, before any research_extension code or experiment existed. Sources: the repository at commit d35ce35 (private GitHub BladeUnbound/metadata-attenuation-skin-lesions), research/EXPERIMENTS.md, research/iteration5/FINAL_REPORT.md and research/iteration6/FINAL_REPORT.md.*

## 1. Where this investigation starts

The directive asks: when can separately observed patient metadata substitute for information removed from an image representation, and can metadata-predictive information be removed selectively while diagnostic utility is preserved? The three quantities it names are already partly measured in this project. The audit below separates what exists from what is new.

| Directive quantity | Existing counterpart | Where |
|---|---|---|
| R_A(Z), attribute leakage | Linear ridge probe on training folds (trivially ≈ 0 after exact LEACE); within-test-fold ridge probe; one registered MLP probe | E45, E50 (amendment 2), E52 |
| C_Y(φ), diagnostic cost | Image-only held-out log-loss, AUROC and Brier after each transformation | E50 arms (`image_logloss`, `image_auroc`) |
| Δ_A(Z), incremental value of age | Strictly cross-fitted two-stage estimator, per-observation Δ_i on identical outer test cases | E40 estimator, used in E45–E55 |
| M_A(φ) = Δ_A(Z) − Δ_A(S) | `targeted_minus_t0` contrast; the registered primary contrast is targeted − matched random | E50 |
| Matched-cost comparison | Random subspaces whose rank is chosen by inner CV on training data to match the eraser's image-loss cost | E50 (nested; E45 used test labels, audit A1) |

So "Level 1" of the directive (apply an existing eraser, then measure metadata's incremental value) has already been carried out for one eraser (closed-form LEACE), one attribute in depth (age) and one more in part (site), on eight image sets. What does not exist yet: a literature audit of this specific question, a formal treatment of when M_A can be positive without diagnostic cost, synthetic ground truth, adversarial or nonlinear erasers, a three-objective frontier, adaptive probes, and any task-preserving transformation.

## 2. Assets

**Data** (research/data, 29 GB, not in git). Eight image sets with frozen embeddings for a representation ladder. Analysis populations as used by E50 (PAD-UFES-20 on complete cases):

| Set | n images | n groups | Prevalence | Grouping | Metadata available |
|---|---|---|---|---|---|
| HAM10000 | 10,015 | 7,470 | 0.111 | lesion | age, sex, site (+ source) |
| ISIC 2020 | 33,126 | 2,056 | 0.018 | patient | age, sex, site |
| ISIC 2019 (non-HAM) | 15,316 | 6,461 | 0.223 | lesion | age, sex, site |
| PAD-UFES-20 (cc) | 1,089 | 541 | 0.675 | patient | age, sex, site, Fitzpatrick, 6 symptoms |
| HIBA | 1,224 | 563 | 0.435 | patient | age, sex, site, Fitzpatrick |
| MSK 2020 | 1,174 | 1,174 | 0.126 | image | age, sex, site |
| MILK10k derm | 5,117 | 5,117 | 0.710 | lesion | age, sex, site |
| MILK10k clin | 5,117 | 5,117 | 0.710 | lesion (same lesions as derm) | age, sex, site |

Independent datasets: 7 at most (MILK10k derm and clin share lesions). Embedding dimensions: HSV 48, ResNet-18 512, ConvNeXt-T 768, ViT-B/16 768, ViT-L/16 SWAG 1024 (all eight sets); HAM10000 also has ResNet-34/50/101 and ConvNeXt-S/B. Fine-tuned *embeddings* do not exist: E14/E14b/E54 saved only fine-tuned scores.

**Code** (research/experiments): strict cross-fitting (`e40_crossfit_baseline`, `e10_ham_attenuation.fit_tuned/outer_fold`), closed-form LEACE with clipped or Ledoit–Wolf whitening, random whitened subspaces, information dose D2, nested matched-rank search, shams, re-append restoration and cross-fitted image-derived attribute (`e50_dose_restoration`), MLP probe (`e52`), four fusion estimators (`e53`), grouped bootstrap (`boot_idx`, `boot_mean`).

**Compute**: Apple M5, 10 cores, 24 GB unified memory, MPS GPU; 276 GB free disk. Measured costs: full E50 run (≈52 arms) 4 min on 1.1k images, 35 min on 10k, 85 min on 33k; light E50 (24 arms) about 60% of that; fine-tuning a ConvNeXt-T fold on 1.1k images ≈ 3–6 min.

**Environment gaps**: no nbclient/ipykernel (notebooks will be executed by a small runner that stores outputs with nbformat), no LightGBM/XGBoost (scikit-learn histogram gradient boosting instead), no `concept_erasure` package (own LEACE implementation, already validated against E45 reproduction; RLACE to be implemented from the paper).

## 3. Established findings that constrain the design

1. **Age erasure raises age's incremental value beyond performance-matched random erasure** on ISIC 2020 (registered confirmatory, Holm p = 0.002), HAM10000, ISIC 2019 and MILK10k, stable over three fold assignments, two backbones and two fusion estimators (E50/E50b/E53b).
2. **It is attribute-specific** on HAM10000 and MILK10k derm: age erasure raises age's but not site's value, and vice versa (E51; replication/exploratory status).
3. **Only linear information is removed**: an MLP probe still decodes age after LEACE (R² 0.05–0.23), and decodes it about as well as after matched random erasure (E52).
4. **Closed-form LEACE fails out of sample when n_train ≈ d** (HIBA, MSK 2020, PAD-UFES-20: 871–980 training rows, d = 768): the erased coordinate is 1.8–3.1× more variable on test than on training rows, and training-only matching cannot see test-time damage.
5. **Marginal erasure is not task-preserving when the label depends on the attribute** (HIBA age-only AUROC 0.84): rank-1 age erasure took HIBA's image AUROC from 0.93 to 0.65. A class-conditional eraser (exploratory) cut the image cost by 15–77% on the large sets while keeping a positive, well-matched effect (E50c).
6. **Restoration is incomplete**: re-appending the erased coordinate restores image log-loss fully but returns 74–93% of the rise in age's value.
7. **Added value depends on the fusion estimator** at strong representations (E53); score fusion is conservative.
8. **Metadata use under dataset shift** cannot be decided from source datasets with the rules tried (E55).

Items 4 and 5 are exactly the two failure modes a "task-preserving" transformation would have to address, and item 3 is the reason adaptive nonlinear probes are mandatory.

## 4. Methodological lessons carried forward

- Never choose a rank, strength or matched control with outer-test labels (E45's flaw, audit A1).
- A probe trained on erased training rows is trivially defeated by an exact linear eraser; leakage must be measured by probes fitted after the transformation is fixed, on data the eraser did not see (amendment 2).
- Report eraser out-of-sample diagnostics (coordinate SD ratio) and matching checks; treat n/d as a first-class variable.
- A rise in the metadata's measured value is uninterpretable without the image-only cost: the matched-cost comparison is the essential control.
- Grouped (patient/lesion) bootstrap on identical held-out observations; Holm within declared confirmatory families.
- Fine-tuned features of training rows are not exchangeable with test rows; erasure on fine-tuned features needs cross-fitted fine-tuning.

## 5. Confirmatory status available for new claims

Every image set has now been used in an age-erasure experiment (E50 on all eight), so no dataset is untouched for claims about *age erasure as such*. A new transformation (MATE) has not touched any dataset. The plan therefore designates one development set for method design (HAM10000) and keeps the other large sets for a registered confirmatory comparison of MATE against LEACE and matched controls at matched diagnostic cost. Synthetic data carry the ground-truth checks.

## 6. Rules that remain in force

No fabricated results or citations; pre-registration (hypothesis, protocol, compute estimate) written before each experiment, with times from `date`; never label anything pre-registered after the fact; sequential jobs; reuse caches; record failures; no causal or clinical claims; split integrity and leakage checks. New files in research_extension are untracked in git until the user asks for a commit.
