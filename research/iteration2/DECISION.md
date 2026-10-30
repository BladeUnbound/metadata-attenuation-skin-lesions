# Research decision report: iteration 2

*27 Oct 2026. Final for iteration 2 (E11 complete).*

## Candidates compared

| Direction | Evidence so far | Access and feasibility | Distinctiveness | Verdict |
|---|---|---|---|---|
| **A. Attenuation of auxiliary-information value with image-model capacity** (HAM10000) | E09 derivation + simulation; **E10 pre-registered, 3 of 4 predictions confirmed or directionally confirmed**; E10b shows it survives a source-shortcut control, and the paired ResNet-18 vs ConvNeXt difference is significant | Public data, lesion IDs, ran in minutes on the M5 | No controlled backbone-ladder analysis found (4 searches); one possible counter-example to read | **Lead** |
| **D. Calibrated spatial testing of saliency vs anatomy** (HAM10000 masks) | E05 (synthetic) + E12/E12b (real): pixel shuffling over-declares significance 3×; exact FFT shift null derived and tested; one pitfall (sign convention) and one finding (misses are small lesions) | Same data; cheap | The spatial-null idea is known from brain mapping; applying it to XAI may be new (unverified) | **Strong secondary**; could become a methods section of the same paper or a separate short paper |
| **B. Feature anisotropy / whitening for kNN anomaly detection** (VisA) | E11: whitening raises mean image AUROC 0.900 → 0.925 (8/12 categories, Wilcoxon p = 0.052) | Done; 2.4 min per category | **Low**: reproduces Mahalanobis PatchCore (2026). Only open angle: predicting *when* whitening helps | **Closed as a reproduction**; the predictability question is parked as a possible small follow-up |
| Original Alzheimer's MRI + genetics | Iteration 1: statistics validated on synthetic data only | **Blocked** (ADNI, OASIS DUA) | n/a | Parked |
| SNN | Iteration 1: no new method | Needs hardware | n/a | Dropped |

## Why A is the strongest direction
1. **It answers a real question with real data:** do patient metadata still help once the image model is good? The answer here is "much less, and indistinguishable from zero for strong backbones", for melanoma detection on HAM10000. Practically, that tells people when building metadata fusion is worth the effort.
2. **It has a candidate mechanism, not yet a confirmed one.** The derivation predicts that falling added value means the image already contains the metadata's information. Recoverability rising as added value falls is consistent with that, and the source-adjusted control rules out the most obvious shortcut explanation. The per-variable test (E10c) was **inconclusive**: all three variables attenuated and became recoverable together, so it couldn't single out the mechanism.
3. **The design is defensible:** pre-registered predictions, lesion-grouped CV, tuning confined to training folds, lesion-bootstrap CIs, paired comparisons, overfitting and source controls.
4. **It carries over the original project's idea** ("does auxiliary information add value beyond imaging?") to a dataset we can actually use. The framework transfers back to Alzheimer's/genetics if data access ever appears.

## What's still missing before this is a paper
1. **Independent replication** on a second image + metadata dataset, ideally a different population: PAD-UFES-20 (smartphone images, rich metadata) or ISIC 2019/2020 (metadata, patient IDs in 2020). Access needs to be verified.
2. **Fine-tuned models, not only frozen features plus linear heads.** Capacity here means "a better frozen representation". Whether attenuation holds when the image model is trained end to end is untested.
3. **Per-variable attenuation** (age vs sex vs site), with recoverability for each, to test the mechanism directly.
4. **Multiplicity-corrected confirmatory tests** on the second dataset, specified in advance from E10b.
5. **Full reads** of the closest papers, especially the CVPR-W 2025 counter-example and SLIMP.

## Not claimed
- No clinical claim. HAM10000 is a retrospective research collection from two sites.
- No claim that metadata is useless in general. Only that its incremental value shrinks for these frozen backbones on this dataset.
- No novelty claim yet.
