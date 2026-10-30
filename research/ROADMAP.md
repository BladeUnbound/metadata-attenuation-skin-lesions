# Research roadmap

*Last updated: 26 Oct 2026, iteration 1.*

## Current question

Does genetic risk (APOE, polygenic risk score) add predictive value to 3D MRI for MCI-to-Alzheimer's progression? Does any added value reflect a **direct** genetic contribution, or only atrophy the MRI model failed to capture?

The second half of the question came out of experiment E03. It is now the most promising direction (Tracks A + B).

## Blockers

| Blocker | Status | Workaround |
|---|---|---|
| ADNI access (MRI, APOE, GWAS genotypes) | **Not started.** It needs a human application at adni.loni.usc.edu; I can't apply for you. | Statistical framework and pipeline validated on synthetic cohorts (E03, E04, E05, E07) |
| External cohort (AIBL or OASIS-3) | Not started | Same as above |
| Public 3D MRI for pipeline development | OASIS-1 (via nilearn) is a candidate. It has a data-use agreement, so it needs your OK before download. | None yet |
| GWAS summary statistics for the polygenic risk score | Not chosen | Interface in place: the score enters as a column |

## Tracks

| Track | Status | Evidence so far | Next step |
|---|---|---|---|
| A. Does genetics add value? | Framework validated on synthetic data | E03: out-of-fold scores required; in-sample scores lose about half the power | Real ADNI data |
| B. Separating direct vs mediated genetic value | **Most promising** | E03 shows the added-value test cannot tell them apart; E07 tests double machine learning | Deeper literature check; mediation framing; ADNI |
| C. Robustness to scanner and cohort shift | Not started | None | Needs multi-site 3D data (ADNI or OASIS) |
| D. Checking saliency against anatomy | Method validated | E05: shuffling voxels gives 34.5% false positives; shifting the mask gives 5.0% | Brain-constrained null; real saliency maps |
| E. Hardware-efficient 3D models | Benchmarked | E06 | Compare region-volume features with a 3D CNN once real data exist |
| F. Pivot | Not needed yet | None | Revisit if ADNI access fails |
| SNN (secondary) | Weight transfer repaired (E01) | E01: corrected port matches Keras on 100% of slices | Only worth pursuing with neuromorphic hardware |

## Candidate improvements (Phase 4), ranked

| # | Candidate | Weakness it fixes | Falsifiable prediction | Data | Cost | Novelty potential |
|---|---|---|---|---|---|---|
| 1 | Double machine learning test for a direct genetic effect, alongside the added-value test | The added-value test can't separate mediated from direct genetic value (E03: 85.6% rejection with no direct effect) | False-positive rate near 5% when APOE acts only through atrophy, with power kept for direct effects (E07) | Synthetic now; ADNI later | Low | Medium: no AD application found; debiased ML for multimodal neuroimaging exists (arXiv 2103.07088) |
| 2 | Attenuation curve: genetics' added value across MRI models of increasing capacity (hippocampal volume, then regional volumes, then 3D CNN) | Published studies disagree, possibly because their MRI models differ | Added value shrinks as the MRI model improves, if APOE acts mostly through atrophy | ADNI | Low once data exist | Medium: framing not found |
| 3 | Spatially calibrated saliency-anatomy overlap test | Overlap claims are usually visual, or use invalid voxel-shuffle nulls | Shuffle null false-positive rate well above 5%; shift null about 5% (confirmed in E05) | Synthetic now; OASIS-1 or ADNI later | Low | Low–medium: spatial nulls are known for brain maps, but not found applied to AI saliency maps |
| 4 | Region-volume features vs 3D CNN on the M5 | Is a 3D CNN worth its compute for progression prediction? | AUC difference under 0.02 at a fraction of memory and time | ADNI or OASIS-1 | Medium | Low |
| 5 | Leakage-corrected benchmark of the original 2D model | The 86% is likely inflated | Large share of validation images have near-copies in training (E02) | Kaggle data (have it) | Low | Low: leakage is already documented |
| 6 | Correctly converted SNN | Old SNN numbers were hardcoded and the transfer was broken | Converted SNN matches the ANN on the same inputs | Kaggle data | Medium | Low without hardware |

## Decision after iteration 1

The decision rule said: pursue candidate 1 if double machine learning held its false-positive rate at realistic n, otherwise fall back to candidate 2. **It did not hold:** with tuned models it flagged 44% of mediated-only cases at n = 400 and 28% at n = 1,600 (E07b). So:

1. **Candidate 2 (attenuation curve) is now the lead.** Next experiment (E09): in simulation, a ladder of MRI models of increasing capacity (hippocampal volume only → 10 regional volumes → all features), testing whether genetics' added value falls as capacity rises under mediation but not under a direct effect. This is the signature a real ADNI analysis would look for, and it needs no causal assumptions.
2. **Candidate 1 is demoted** to a sensitivity analysis.
3. **Candidate 5 is settled:** near-copy leakage is confirmed (E02c, E08), so the paper's 86% must not be reported as generalisation.
4. **Candidate 4 is strengthened by E06:** 3D training is compute-bound at about 3.5 volumes/s, so compact representations have measurable value.

## Iteration log
- **Iteration 1** (26 Oct 2026): E01–E08 run. Two hypotheses falsified (DML validity at realistic n; large GPU and fp16 gains). One method failure diagnosed and fixed (E02 embedding → E02c visual check). One self-introduced bug fixed (E07b).

---

# Iteration 2 (26–27 Oct 2026): pivot to accessible data

**ADNI is no longer a dependency.** The research now runs on HAM10000 (primary) and VisA (secondary). The rationale is in `iteration2/DATASETS.md` and `iteration2/DECISION.md`.

## Current lead question
**Does patient metadata still add predictive value once the image model is strong, and does any loss of value reflect the image already encoding the metadata?** This carries over the original project's "auxiliary information vs imaging" idea to data we can actually use.

## Status by track

| Track | Status | Key evidence | Next step |
|---|---|---|---|
| **A. Attenuation curve** (HAM10000) | **Lead; preliminary support** | E10 (pre-registered): metadata added value 0.0186 → about 0.002 across the ladder; E10b: holds with source adjustment; ResNet-18 vs ConvNeXt paired difference significant | Replicate on a second dataset; fine-tuned models; read the counter-example paper |
| D. Saliency vs anatomy (HAM10000 masks) | Strong secondary | E12/E12b: pixel shuffling over-declares 3×; sign-convention pitfall; misses are small lesions | Repeat across backbones; targeted XAI literature search |
| B. Anisotropy for anomaly detection (VisA) | Reproduction (whitening already published) | E11 running; candle: no gain from removing PCs, whitening slightly worse | Finish E11; likely close the track |
| Mechanism (per variable) | Inconclusive | E10c: all variables attenuate and become recoverable together | Find a variable or dataset where recoverability doesn't rise |
| Original Alzheimer's question | Parked | n/a | Only if data access appears |

## Next iteration (3), in priority order
1. **Replication dataset for Track A:** verify access to PAD-UFES-20 (smartphone lesion images with rich metadata) or ISIC 2020 (patient IDs, age, sex, site). Run E10's pre-registered protocol unchanged, as a confirmatory test with multiplicity correction.
2. **Fine-tuning ladder:** end-to-end fine-tuned ResNet-18 → ConvNeXt-T on HAM10000 (lesion-grouped). Does attenuation hold when capacity comes from training, not only from a better frozen representation? This is the first compute-bound workload on the M5.
3. **Literature:** full reads of the CVPR-W 2025 metadata paper (Ahammed et al.) and SLIMP (arXiv 2505.23709); a targeted epidemiology search on incremental value vs baseline model strength.
4. Close Track B after E11, unless all-12-category results contradict the candle result.
