# Iteration 6: audit and staged run plan (mechanism identification)

Written 2026-10-29 13:49 IST, before any iteration-6 code was written or run. Existing results, registries and decision criteria (iteration5/REGISTRY.md, E45b claim rule) are left untouched; nothing below edits them.

## 1. What already exists

| Piece | Where | State |
|---|---|---|
| Training-fold-only attribute subspace | `experiments/e45_erasure.py` `leace()` | LEACE (closed form) fitted per outer fold on training rows with the attribute observed; uses metadata only, never diagnostic labels; applied to all rows |
| Doses | E45 | t ∈ {0.25, 0.5, 0.75, 1} for the targeted eraser only, as S_t = S − t(S − S_erased) |
| Controls | E45, E45b | same-rank random whitened subspace; "performance-matched" random subspace (rank grid r·2^j); 5 subspaces of each in E45b (t = 1, age only) |
| Estimator | `e45_erasure.corrected_foldwise` | E40 strict cross-fitting: inner grouped OOF scores train the second stage; the outer test fold is only predicted |
| Linear probes | E45 `probe()` | age: Ridge(α = 100) held-out R²; site: logistic accuracy; t = 0 and t = 1 (and same-rank random) only |
| Paired inference | E45b | grouped bootstrap (B = 2000) of per-observation paired differences at t = 1, age only |
| Specificity | E45 | Δ for age, sex, site and the full block under each eraser (point estimates and marginal CIs only) |
| Folds | `StratifiedGroupKFold(5, shuffle, random_state = seed)` | fold assignment 0 reproduces E40's saved fold vector on all 8 image sets; no grouping unit spans two folds (checked 29 Oct, 13:4x) |
| Embeddings | `data/<ds>/emb_{convnext_tiny,resnet18}.npy` | cached for all 8 image sets (ConvNeXt-T 768-d); no image recomputation needed |
| Fine-tuning, data size | E14, E14b, E31c, E31d | one training seed per fold; original (leaky) second stage; review item R7-11 open |
| Shift | E33, E47, E34 | E33 cross-dataset transfer within label families; E47 HAM10000 leave-one-source-out and ISIC 2019 BCN↔MSK; E34 collection rule (uses target-dataset I_var, i.e. target labels) |

## 2. Audit findings

| # | Finding | Severity | Action in this round |
|---|---|---|---|
| A1 | **The performance-matched random rank in E45 was selected with outer test labels.** `outer_only_logloss()` compares image-only log-loss computed on the outer test folds (`y[te]`) across candidate ranks. E45b reused those ranks (HAM 64, ISIC 2019 32, MILK10k 256, PAD 1 for ConvNeXt-T). The selection targets image-only loss, not Δ, so the direction of any bias is unclear, but it breaks the rule that test labels never enter hyperparameter selection. | Medium (control validity) | E50 selects the matched rank separately in each outer training fold from inner-CV, training-only image log-loss. If any E45b claim fails under the nested control, the paper drops it. |
| A2 | **The registered dose is not an information dose.** For t < 1, I − tM (M a projection) is invertible, so S_t carries exactly the information in S; a refitted linear first stage loses nothing except through its L2 penalty and standardisation. E45's convex curves (most of the rise at t = 1) fit this reading. | Interpretation | Keep the projection-scaling dose (D1) for continuity; add an information dose (D2): the erased coordinates are replaced by √(1−t)·c + √t·σ·ε with independent Gaussian ε, so linear decodability falls in proportion to 1 − t. D2 is the primary dose–response family. |
| A3 | No sham control (same pipeline, attribute carrying no signal). | Gap | Sham: LEACE fitted on the attribute permuted within the training fold (5 permutations). |
| A4 | "Restoration" in E45 was only the identity at t = 0 and was never run as a separate check. | Gap | (i) Re-append the erased coordinates to the erased features (information-equivalent to the original; should reproduce t = 0 up to regularisation); (ii) metadata restoration fraction; (iii) image-trace restoration: Δ from adding the erased coordinate itself in the second stage. |
| A5 | Probes: only at t = 0/1; site probe scored by accuracy (uninformative under class imbalance); age α fixed. | Gap | Nested-α ridge for age (plus fixed α = 100 for continuity); site probe scored as held-out log-loss reduction over the training prior and macro one-vs-rest AUROC; every arm probed. |
| A6 | The random-control centring used the LEACE mean (rows with the attribute observed) rather than the control's own training mean. | Negligible | E50 centres each eraser on its own training rows. |
| A7 | First-stage penalties chosen per fold were discarded. | Logging | E50 logs every chosen C, every per-fold rank, the random seeds, the folds and the code hashes. |
| A8 | Metadata dummy columns are built from the category set of all rows (no labels involved); a category seen only in a test fold has an all-zero training column. | Not a leak | Recorded; unchanged for comparability. |
| A9 | Standalone metadata value (E30, metadata-only, computed before any erasure): age's lower 95% bound is ≥ 2×10⁻³ nats in HAM10000, ISIC 2019, ISIC 2020, HIBA, MSK 2020 and MILK10k; site's only in HAM10000 (8.0×10⁻³) and HIBA (4.3×10⁻²). PAD-UFES-20 (1.7×10⁻³) and MILK10k (1.8×10⁻³) are borderline; ISIC 2019, MSK 2020 and ISIC 2020 site are not supported. | Scoping | Site is the second attribute; it is tested where supported. |
| A10 | Datasets never used in any erasure experiment: ISIC 2020, HIBA, MSK 2020 (and MILK10k clinical, which shares lesions with MILK10k dermoscopic). | Opportunity | These three form the confirmatory family; the four E45 sets become replications with corrected controls. |
| A11 | Independent datasets: 7 (MILK10k derm + clin counted once). ISIC 2019 non-HAM has a BCN and an MSK source; HAM10000 has 4 sources, but vidir_molemax has 24 melanomas. | Limitation | P6 counts datasets, not image sets; within-dataset sources are reported separately and never pooled as independent replications. |

## 3. Staged plan

Every stage writes a JSON record to `research/results/`, per-observation arrays to `data/<ds>/e5x/`, and a status line to `iteration6/STATUS.md`. Jobs run one at a time.

| Stage | Experiment | Content | Compute (estimate) | Gate after the stage |
|---|---|---|---|---|
| 0 | — | This audit; REGISTRY.md with all primary contrasts fixed | — | — |
| 1 | **E50a** (P1) | Age; ConvNeXt-T; fold assignment 0; 8 image sets. Arms: targeted D1 and D2 at 25/50/75/100%; 5 nested performance-matched random subspaces at every dose; 5 shams; 5 same-rank random; 2 restoration arms; probes in every arm | ~50 corrected fits per set; 20–35 s each on the large sets → ~2 h CPU | **G1** (REGISTRY.md): proceed, revise or stop |
| 1b | E50b | Robustness: ResNet-18 (100% arms) and fold assignments 1, 2 (ConvNeXt-T, 100% arms) | ~2 h CPU | — |
| 2 | **E51** (P2) | Site erasure: HIBA (confirmatory), HAM10000 (replication), PAD-UFES-20 and MILK10k derm (borderline, exploratory); 2×2 specificity matrix with paired contrasts | ~1 h CPU | — |
| 3 | E52 (P3) | Pre-specified MLP probe before/after erasure | ~0.5 h CPU | — |
| 4 | E53 (P4) | Score fusion vs full-feature regularised fusion vs small boosted fusion vs missingness-only; same folds; E53b repeats the erasure contrast under full-feature fusion if G1 = proceed | ~1.5 h CPU | — |
| 5 | E54 (P5) | Reuse E14/E14b/E31; extra fine-tuning training seeds on the small sets (HIBA, PAD-UFES-20) | ~2 h GPU (MPS) | Skip if earlier stages overrun |
| 6 | E55 (P6) | Source-held-out evaluation with decision rules fitted on sources only | ~0.5 h CPU | — |
| 7 | — | Final report: verified results, limitations, novelty, manuscript changes | — | — |

The confirmatory family's Holm adjustment is completed only after stage 2 (it contains HIBA site). The gate after stage 1 uses the first Holm step (p ≤ 0.05/4), which rejects regardless of the site p-value.
