# Pre-registration: E10, metadata added value vs image-model capacity on HAM10000

*Written 26 Oct 2026, before any HAM10000 model was fitted. Not to be edited after E10 runs; deviations go in EXPERIMENTS.md.*

## Question
In dermoscopic lesion classification, does patient metadata (age, sex, anatomical site) add less predictive value as the image representation improves? If so, does that attenuation track how well each image representation encodes the metadata?

## Data
- HAM10000 (Tschandl et al., *Sci Data* 2018; Harvard Dataverse doi:10.7910/DVN/DBW86T; CC BY-NC 4.0), 10,015 images.
- **Grouping:** `lesion_id`. Every split keeps all images of a lesion together (5-fold StratifiedGroupKFold).
- **Excluded from the auxiliary variables:** `dx_type` (how the diagnosis was confirmed). It depends on the diagnosis itself (e.g. follow-up is used for nevi), so it would leak the label. E10 reports how predictive it is, as a leakage check.
- **Auxiliary variables used:** age (with a missing-value indicator; unknown values imputed with the training-fold median), sex (one-hot, including unknown), localization (one-hot).

## Tasks
- **Primary:** melanoma vs all other lesions (binary). Metric: out-of-fold log-loss.
- **Secondary:** 7-class diagnosis, out-of-fold multiclass log-loss; melanoma AUROC.

## Capacity ladder (frozen features + logistic head)
1. Colour-histogram baseline (HSV histograms, 48-d)
2. ResNet-18, ImageNet weights, global-average-pooled features
3. ResNet-50, ImageNet weights
4. A larger or newer torchvision backbone (ConvNeXt-Tiny or ViT-B/16), if weights download.

Capacity is measured by image-only out-of-fold performance, **not** parameter count. Heads: L2 logistic regression with the penalty chosen by inner CV within each training fold. Metadata is added at the head (image score + metadata, and also full features + metadata).

## Added value and uncertainty
- Added value = log-loss(image only) − log-loss(image + metadata), out of fold.
- 95% CI by bootstrap over lesions (2,000 resamples, grouped by `lesion_id`).
- Repeated over 3 different fold assignments.

## Recoverability (mechanism)
Out-of-fold R² for predicting age, and AUROC for predicting sex and the most common sites, from each image representation.

## Predictions
1. Added value is positive for the colour-histogram baseline.
2. **Attenuation:** added value decreases from rung 2 to the strongest rung.
3. It does not reach zero: sex in particular is weakly encoded in dermoscopic images, so a floor remains.
4. Recoverability of age and site increases with capacity, and the rungs with higher recoverability show less added value.

## What would falsify it
- Added value flat or rising with capacity (E09's "direct" signature), with CIs that exclude a decrease.
- Or attenuation present but unrelated to recoverability, which would point to a different mechanism, e.g. optimisation or overfitting differences.

## Alternative explanations to check
- **Overfitting of richer heads:** compare image-score + metadata against full-features + metadata.
- **Baseline difficulty:** report added value both as an absolute and as a relative log-loss reduction.
- **Metadata leakage:** show that `dx_type` alone strongly predicts the label (why it's excluded), and check that no remaining field is a near-proxy for the label.
