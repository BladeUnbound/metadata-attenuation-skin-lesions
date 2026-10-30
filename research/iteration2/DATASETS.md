# Dataset feasibility matrix (iteration 2, 26 Oct 2026)

Access was verified by actual requests on this date, not taken from documentation.

| Dataset | Access verified | Licence | Size | Grouping IDs | Labels / extras | Fits | Blockers | Decision |
|---|---|---|---|---|---|---|---|---|
| **HAM10000** (Harvard Dataverse doi:10.7910/DVN/DBW86T) | Yes: API lists files; metadata downloaded | CC BY-NC 4.0 (terms text on Dataverse) | 2.8 GB images + 421 MB ISIC 2018 test set | `lesion_id` (7,470 lesions / 10,015 images; 45% of images in multi-image lesions) | 7 diagnoses; age, sex, site; image source; **expert lesion masks**; `dx_type` (leaks the label) | Track 1 (metadata vs capacity); Track D (saliency vs lesion masks) | Slow download (~0.5 MB/s per connection); no patient ID beyond lesion | **Primary** |
| ISIC 2018 Task 3 test set (same Dataverse record) | Yes: listed | Same | 421 MB, 1,512 images | None found yet | Diagnosis ground truth; metadata to be checked | Held-out check for Track 1 | Same source institutions as HAM10000, so not truly external | Secondary |
| **VisA** (Amazon) | Yes: HTTP 200, 1.93 GB | CC BY 4.0 (README verified) | 1.9 GB | Not needed (official normal-only training split) | 12 categories, image labels, **pixel masks**, official `split_csv` | Track 2 (anomaly detection) | Whitening idea already published (Mahalanobis PatchCore) | **Primary for Track 2** |
| MVTec AD | Official site: download page sits behind a form. Kaggle mirror `ipythonx/mvtec-ad` (5.3 GB) is listed | CC BY-NC-SA 4.0 | 5.3 GB | n/a | 15 categories, masks | Track 2 independent benchmark | Form on the official site; mirror provenance not verified | Fallback benchmark |
| MedMNIST+ (Zenodo 10519652) | Yes: API lists files | CC BY 4.0 | 15 MB – 12 GB per subset | **None** | Labels only | Fast pipeline checks | No metadata, no grouping; DermaMNIST comes from HAM10000 and may inherit lesion leakage (unverified) | Not used this iteration |
| OASIS-1 / OASIS-3 | Not downloaded | Data-use agreement | n/a | Subject IDs | MRI + clinical; OASIS-3 has APOE | Original Alzheimer's question | Needs your agreement to the DUA (OASIS-1) or an application (OASIS-3) | Parked |
| ADNI | No access | DUA | n/a | n/a | n/a | n/a | No access | Not pursued (per the brief) |

## Selection rationale
- **HAM10000** is the only accessible dataset that has images, clinically meaningful metadata, a grouping identifier **and** ground-truth masks. That lets one dataset test the attenuation question (Track 1) and the spatial saliency test from iteration 1 (E05) on real masks.
- **VisA** has a clean official protocol and a permissive licence for an independent CV question (Track 2).
