# Review log

Each round: a strict IEEE-style review of the current draft, then concrete actions. An item is closed only with evidence (experiment ID, compiled page, or verified citation).

## Round 1: draft v1 (27 Oct 2026, 4 pages, replication pending)

### Major
| # | Criticism | Action | Status |
|---|---|---|---|
| M1 | Central claim rests on one dataset; replication shows [PENDING] | Run E13 on ISIC 2020, ISIC 2019 (non-HAM), PAD-UFES-20 exactly as pre-registered | In progress |
| M2 | "Capacity" covers only frozen ImageNet features with a linear head. A reviewer will say a fine-tuned model is what practitioners use, and that frozen-feature quality confounds capacity with pretraining domain | **E14:** fine-tune ResNet-18 and ConvNeXt-T end to end on HAM10000 (lesion-grouped 5-fold), measure added value on their out-of-fold scores | Open |
| M3 | The second stage uses only a 1-D image score, which may discard image information that interacts with metadata | Report the existing control (full features + metadata vs score + metadata, E10) in the paper; add it to the table | **Closed**: Table `ham_robust` shows two-stage vs joint log-loss within 0.003 at every rung |
| M4 | Nats are hard to interpret clinically | Add ΔAUROC and ΔBrier with grouped bootstrap CIs alongside Δ log-loss | **Closed**: E10d, Table `ham_robust`; ΔAUROC 0.043 → 0.003 |
| M5 | Proposition 1 is a direct consequence of the chain rule; claiming it as a contribution overstates novelty | Reframe as "an elementary decomposition that fixes the interpretation"; credit the chain rule explicitly | **Closed**: contribution 1 reworded; the proof cites the chain rule; identity verified exactly (`pipeline/test_info_identity.py`) |
| M6 | The recoverability-mechanism claim is not tested statistically, and E10c (inconclusive) isn't reported | Report E10c honestly; let PAD-UFES-20 M1–M3 carry the mechanism test | **Closed with a negative result**: E10c reported; PAD M1–M3 all failed; E16 shows the premise (invisible symptoms) was false. The paper states the mechanism is not isolated |
| M7 | Interpretability evaluated for one backbone only | **E15:** exact CAM for ResNet-18, ResNet-50, ConvNeXt-T (derivation through ConvNeXt's post-pool LayerNorm), with the same overlap tests | Open |

### Minor
| # | Criticism | Action | Status |
|---|---|---|---|
| m1 | Tables I and III overflow the column | Narrower layouts / `\resizebox` | Open |
| m2 | No figure of the central result | Small-multiples figure of Δ along the ladder per dataset, with CIs | **Closed**: Fig. 1 generated from JSON |
| m3 | Author block is a placeholder | Needs the authors' decision; flag in the hand-off | Open (human) |
| m4 | 9 references are [M] (from memory) | Verify each by search; fix or drop | **Closed**: all core references verified; the unverifiable CVPR-W 2025 citation was dropped; BCN20000 now cites its Sci. Data 2024 version |
| m5 | Text says "0.019 to 0.002 nats" with 3-decimal rounding, while tables use 10⁻³ nats | One unit everywhere (10⁻³ nats) | **Closed** |
| m6 | Abstract length versus IEEE's ≤ 250 words | Count and trim | Open |
| m7 | HAM10000 row of the dataset table is hand-typed | Generate it from E10a JSON | **Closed** |

## Round 2: draft v2 (6 pages; ISIC 2020 and PAD-UFES-20 replicated; ISIC 2019, E14, E15 pending)

| # | Criticism | Action | Status |
|---|---|---|---|
| R2-1 | "H2 held in 2 of 2 after Holm" uses a 4-test family; the pre-registered family is 6 tests (3 datasets), and ISIC 2020's H2 (raw p = 0.046) may not survive | Final Holm over all six tests once ISIC 2019 is in; abstract and conclusion generated from the final result | Open (automatic) |
| R2-2 | Contribution 3 promised a mechanism test "with symptoms that photographs cannot show", a premise E16 showed to be false | Reworded: the mechanism test failed informatively | **Closed** |
| R2-3 | Bootstrap p-values are bounded below by 1/(B+1) ≈ 0.0005; reporting "0.000" or exact small values overstates precision | Report "≤0.001" | **Closed** |
| R2-4 | Fig. 1's panels have very different y-scales without saying so | Caption states it; Table `retained` gives the normalised comparison (ρ) | **Closed** |
| R2-5 | Abstract omits the dataset-dependent floor, the key qualification of the main claim | Rewrite the abstract after the final replication results | Open |
| R2-6 | Interpretability rests on one backbone | Replace with the E15 multi-backbone table | Open (E15 running) |

## Round 3: draft v3 (replication complete; E14 and E18 pending)

| # | Criticism | Action | Status |
|---|---|---|---|
| R3-1 | The conclusion reported only the H1 success; it omitted the H2 failure and the dataset-dependent floor | Conclusion rewritten | **Closed** |
| R3-2 | Table VIII floated after the references | Moved into the results | **Closed** |
| R3-3 | **Strongest objection:** H1 contrasts deep features with a colour histogram, a deliberately weak baseline, so "metadata help it more" is unsurprising. The more interesting within-deep claim (H2) replicated in only 1 of 3 datasets | Reframe the contribution around (a) the size of attenuation in practically relevant terms, (b) the retained fraction ρ and its 7-fold variation, (c) the methodological warning against weak baselines. Strengthen the high-capacity end: E14 (fine-tuned) and E18 (304M-parameter foundation model) | Open (E14 running, E18 queued) |
| R3-4 | My own earlier framing "dermoscopy keeps little, smartphone keeps half" was contradicted by ISIC 2019 (ρ = 0.44) | Abstract and discussion now attribute the floor to the dataset, not the modality | **Closed** |
| R3-5 | Abstract was 264 words, over IEEE's 250 | Rewritten to 239 words | **Closed** |

## Round 4: integrity audit (prompted by E20)
| # | Finding | Action | Status |
|---|---|---|---|
| R4-1 | **Missingness is label-informative in all four datasets**, severely in PAD-UFES-20: 804 images missing gender, Fitzpatrick and history are all benign and unbiopsied, a proxy for the excluded `biopsed` field. The pre-registered encoding kept missing-value indicators | E21 complete-case analysis on all datasets. Methods, results, discussion, limitations and abstract updated. Pre-registered results kept and reported as such | **Closed**: H1 holds 4/4, H2 2/4 complete-case; PAD I_A 0.416 → 0.155 nats |
| R4-2 | My first missingness audit had a bug (`astype(str)` keeps NaN in this pandas version), which hid gender and ISIC missingness | Fixed (`isna()` first); audit re-run | **Closed** |
| R4-3 | E20's PAD per-variable result (sex, Fitzpatrick) was an artefact of R4-1 | Marked invalid in the registry; not used in the paper | **Closed** |
| R4-4 | Discussion quoted ρ only from the leak-inclusive encoding | Complete-case ρ added next to it | **Closed** |

## Round 5: draft v4 (28 Oct 2026; 8 pages; second wave, strength manipulations and E30–E34 integrated)

Strict-reviewer pass over the rendered PDF, plus a fresh literature search (2 extended queries).

### Major
| # | Criticism | Action | Status |
|---|---|---|---|
| R5-1 | Fig. 3 (dose–response) mixes populations: 10/25/50% on complete cases with one subsample each, 100% on all images (E14). The figure's strongest point is not comparable to the others | **E31d** registered before running (REGISTRY_E30_E34.md addendum): 100% on complete cases + a second subsample at 10/25/50%. Running in `overnight/queue3.sh`; figure and text to be regenerated from E31d | **Open (running)** |
| R5-2 | Closest prior work missing. Ahammed et al. (CVPR-W 2025), dropped in round 1 as unverifiable, is now on CVF Open Access. Its table shows metadata gains of 0.023 AUROC from scratch vs 0.001 for pretrained, fine-tuned models | Read in full; cited in Related Work with that calculation and its limits (single runs, no CIs, not framed as attenuation). Contribution 4 is stated as a controlled, inferential version of a pattern visible there | **Closed** |
| R5-3 | "7 of 7 datasets" (in the iteration-3 report) double-counts MILK10k, whose two image sets share lesions | Paper counts independent datasets (6) and states the rule: MILK10k holds only if both image sets hold. H1 6/6, H2 3/6. Report corrected | **Closed** |
| R5-4 | Representation-strength comparisons (Table VIII, E31, E18) are many and uncorrected | Stated in the text; all labelled exploratory; abstract says "exploratory manipulations" | **Closed** |
| R5-5 | Fine-tuning: one run per fold, fixed schedule, three datasets; PAD-UFES-20 fine-tuning includes the missingness leak | Limitation (i) and (vi); Table VIII caption. E14b on ISIC 2019 and MILK10k queued | Partly closed |
| R5-6 | Title said "capacity", but E31a/E18 show parameter count does not drive the effect | Title and framing changed to "representation-dependent"; size-ladder null reported | **Closed** |

### Minor
| # | Criticism | Action | Status |
|---|---|---|---|
| r5-1 | Negative numbers typeset with a hyphen ("-13.5") in macros and tables (also in earlier drafts) | Generator converts to a true minus everywhere | **Closed** |
| r5-2 | ViT-L SWAG weights uncited | Singh et al. CVPR 2022 verified and cited | **Closed** |
| r5-3 | "No image appears in two datasets" overstated: PAD-UFES-20 has no ISIC identifiers | Reworded | **Closed** |
| r5-4 | Abstract 253 words (> 250) | Trimmed to 245 (pdftotext count) | **Closed** |
| r5-5 | E30's row exclusion (I_var CI must exclude 0) was fixed in code before the run but not in the registry | Disclosed in the text ("the rest had no significant standalone value") and in EXPERIMENTS.md | **Closed** |
| r5-6 | Eight pages exceeds most IEEE conference limits | Venue decision needed (journal such as IEEE JBHI, or conference + supplement) | Open (human) |
| r5-7 | Author block placeholder | Needs the authors | Open (human) |

### Verified this round
- New references (6): HIBA (Sci Data 2023), MILK10k (JID 2026), Müller et al. (JAAD 2026), MSK 2020 ISIC collection DOI, SWAG (CVPR 2022), Ahammed et al. (CVPR-W 2025). Author names given only as initials where the source gave initials.
- Every number in the paper is generated from `research/results/*.json` by `scripts/make_numbers.py`; no new hand-typed numbers except the Ahammed table calculation (computed from its published Table 5).
- Holm families match the pre-registrations: 6 tests (wave 1), 8 tests (wave 2).

## Round 6: draft v5 (28 Oct 2026; 13 pages = 8 main + appendices; all experiments added)

The user asked for every experiment to be in the paper. Added: Appendix A (index of all experiments E09–E34, with E01–E08 and E11 excluded as unrelated tracks), and Appendices B–E with generated tables/figures for E09, E18, E19, E20, E30, E31a, E31b, E32, E33, E34. Then a strict pass over the whole draft.

| # | Criticism | Action | Status |
|---|---|---|---|
| R5-1 | Fig. 3 mixed populations | **E31d** run: one population, two subsamples, 100% on complete cases. Mean Δ 8.9 → 5.4 → 3.2 → 1.2 ×10⁻³; Spearman −0.97 over 7 runs. Figure, text and abstract regenerated | **Closed** |
| R6-1 | Appendix D said "every variable attenuates" on ISIC 2020, where sex and site have no detectable value to attenuate | Reworded to variables with detectable HSV value | **Closed** |
| R6-2 | Appendix E said the recoverability pattern holds "within most datasets" (unquantified) | Computed from E30_table: deep-only within-dataset Spearman negative in 8/8 (−0.40 to −0.95), with as few as 4 points; generated macros | **Closed** |
| R6-3 | E34 discussion reported only the failure; the simple baseline's decision accuracy (0.80, precision 0.81, recall 0.83) was omitted | Added to Discussion and Table XIX | **Closed** |
| R6-4 | E19 silently lacked ISIC 2020 (never embedded) | Embedded and analysed 2026-10-28 (queue4, exit 0); Table res now covers all 7 image sets. ISIC 2020: Spearman −0.6, prediction does not hold; overall 3 of 7 (was 3 of 6). Text is macro-driven and unchanged in substance | Closed (2026-10-28) |
| R6-5 | E19's PAD-UFES-20 result uses the leak-inclusive encoding | Stated in text and caption | **Closed** |
| R6-6 | Fraction AUROCs at 2 decimals hid that AUROC barely moves from 50% to 100% while Δ keeps falling | 3 decimals; one cautious sentence (adjacent CIs overlap) | **Closed** |
| R6-7 | HAM10000 source names in the LOSO text were capitalised by guess | Printed verbatim from the dataset field | **Closed** |
| R6-8 | Index table overflowed | Column widths fixed; build has no overfull boxes | **Closed** |
| R6-9 | Fine-tuning evidence beyond HAM10000/HIBA/PAD-UFES-20 | E14b ISIC 2019 and MILK10k (ResNet-18) added to Table ft; AUROC rose on both, added value fell slightly but neither difference excludes zero; Sec. ft, discussion, conclusion, limitation (i) and index row qualified accordingly | Closed (2026-10-28) |
| r5-6, r5-7 | Venue (13 pages incl. appendices) and author block | Need the authors | Open (human) |

**Status 2026-10-28 11:45 (draft v5 final regeneration).** All experiments queued in iteration 4 have finished and are in the paper. Earlier-round items still marked "Open" in their own tables (m1, m6, R2-5, R2-6) were superseded: no overfull boxes, abstract 245 words, abstract states the dataset-dependent floor, E15 multi-backbone interpretability is in the paper. Only human decisions remain open: author block and venue (r5-6, r5-7).

## Round 7 (2026-10-29): iteration-5 audit folded into draft v6

Source of every item: research/iteration5/ (AUDIT.md risk register, FINAL_REPORT.md). All numbers come from make_numbers.py → scripts/iter5_numbers.py.

| ID | Issue | Action in draft v6 | Status |
|---|---|---|---|
| R7-1 | Second stage trained on scores from first-stage models that saw the test fold (AUDIT R1) | Setup: audit paragraph (reproduction 120/120, max change 0.42 CI half-widths). Theory: estimator text and KL identity. Results: corrected H1/H2 counts and ISIC 2020 H2 change. Appendix: robustness table | Closed |
| R7-2 | Proposition 1 interpretation implied monotone attenuation | Rewritten with nested counterexamples; measurability of the inverse stated | Closed |
| R7-3 | Δ̂ target conditions on a one-dimensional score; image weighting unstated | Stated in Theory; limitation (ii) updated; HIBA per-patient value reported | Closed |
| R7-4 | CIs omit training variability | Limitation (viii), citing Bates et al. (2024), with the fold-assignment spread | Closed (disclosed) |
| R7-5 | Retained fraction depends on the label definition | ISIC 2019 melanoma vs malignant ρ in "How much information remains?" | Closed |
| R7-6 | Missing-value encoding changes added value (E47) | Complete-case section; abstract | Closed |
| R7-7 | E30 recoverability claim depends on an unregistered filter | Discussion restricts it to variables with standalone value, with mixed-model results for both row sets | Closed |
| R7-8 | Shift null invalid for centred lesions (E48) | Interpretability moved to an appendix with the swap null and the Gaussian positive control; contribution 5 replaced | Closed |
| R7-9 | Mechanism not isolated (limitation iv) | New Section "Erasing what the image encodes about age" (E45/E45b, with paired CIs); limitation (iv) narrowed to age and linear information | Closed |
| R7-10 | Graph extension (user brief) | New Section "Where do metadata help?": pre-registered negative result, concentration (post hoc), Hη decomposition (exploratory), with Gadgil et al. (2024) and Covert et al. (2023) cited | Closed |
| R7-11 | Fine-tuned scores share the leak | Disclosed in limitation (i); not re-run (~10–15 h GPU) | Open (by choice) |
| R7-12 | Length: 18 pages including appendices and references | Venue decision; candidate split into main paper plus supplement | Open (human) |
| r5-6, r5-7 | Venue and author block | Need the authors | Open (human) |

## Round 8 (2026-10-30 09:50 IST): iteration-6 results folded into draft v7

Source: research/iteration6/FINAL_REPORT.md §13 (11 recommendations). All numbers come from make_numbers.py → scripts/iter6_numbers.py; the figure from make_figures.py (E50). Backups: main_v6.tex.bak, main_v6.pdf, refs_v6.bib.bak, scripts/make_numbers_v6.py.bak, scripts/make_figures_v6.py.bak.

| ID | Issue | Action in draft v7 | Status |
|---|---|---|---|
| R8-1 | Erasure section rested on E45b's matched control, whose rank was chosen with test labels (iteration-6 audit A1) | Section rewritten around E50: registered confirmatory family (ISIC 2020 passes, Holm p 0.002; HIBA, MSK 2020 and HIBA site fail, reported with the reason), shams, information dose, restoration fraction, image-derived age, fold assignments, ResNet-18, full-feature fusion. New Table erasure, Fig. erasure (all 8 sets), appendix Tables apperasure and apperasurerob | Closed |
| R8-2 | Specificity shown on one dataset only | Registered age/site double dissociation (HAM10000, MILK10k derm), appendix Table specificity | Closed |
| R8-3 | Wording implied erasing age itself | "Linearly decodable"; E52 MLP result in text; Discussion sentence replaced by the restoration-fraction statement | Closed |
| R8-4 | "So the image models use age-related features for diagnosis" | Replaced: the linear read-out depends on directions that covary with age, part label-induced; class-conditional erasure (exploratory) reported | Closed |
| R8-5 | Per-dataset levels depend on the fusion estimator (E53) | "How much information remains?", Discussion, limitation (ii); new appendix Table estimators; H1 under (b) and (c) in the replication paragraph; abstract | Closed |
| R8-6 | Missingness share unquantified | Complete-case section: PAD-UFES-20 77%, ISIC 2019 25% (E53 d) | Closed |
| R8-7 | Collection rule implied useful under shift | E55 in the shift paragraph and appendix Table rules; appendix collection-rule sentence qualified | Closed |
| R8-8 | One fine-tuning seed (R7 limitation i) | E54: 3 seeds on HIBA and PAD-UFES-20, drop held for 12 of 12 runs; Sec. ft and limitation (i) | Closed (other datasets still one seed) |
| R8-9 | Erasure design's preconditions not stated | Limitation (iv) and Sec. erase: needs n ≫ d and a weak attribute–label association | Closed |
| R8-10 | Closest methodological precedent not cited | Related work: Elazar et al. (TACL 2021; §2.3 read from the PDF) and Kumar et al. (NeurIPS 2022; abstract and listing read) | Closed |
| R8-11 | Age–label diagnostic existed only in a registry note | Recomputed reproducibly (research/experiments/e50_age_label_diag.py → results/E50_age_label_diag.json; descriptive, post hoc) and cited by macro | Closed |
| R8-12 | Appendix floats queued until \clearpage (23 pages, near-empty float pages) | All floats [tp], float-page fraction 0.75: 21 pages, no overfull boxes, no undefined references | Closed |
| R7-11, R7-12, r5-6, r5-7 | Strict cross-fitting of fine-tuning; length (now 21 pages); venue; author block | Unchanged | Open (human / by choice) |
