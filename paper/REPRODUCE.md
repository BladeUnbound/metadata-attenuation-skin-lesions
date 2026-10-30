# Reproducing the paper

Every number, table and figure in `main.pdf` is generated from `research/results/*.json` by two scripts. No result is typed by hand.

```sh
cd "~/Desktop/CV EL"
PY=novelty/.venv/bin/python   # Python 3.14, torch 2.14, torchvision 0.29, scikit-learn 1.9, statsmodels 0.15

# 1. Data (public; licences in research/iteration2/DATASETS.md and EXPERIMENTS.md, iteration 3)
#    HAM10000: Harvard Dataverse doi:10.7910/DVN/DBW86T -> research/data/ham10000/{raw,images,masks}
#    ISIC 2020: Kaggle cdeotte/jpeg-melanoma-256x256 -> research/data/isic2020/x
#    ISIC 2019: official ISIC metadata + Kaggle mnowak061/isic2019-256x256-jpeg -> research/data/isic2019
#    PAD-UFES-20: official Mendeley metadata + Kaggle mahdavi1202/skin-cancer -> research/data/pad_ufes

# 2. Embeddings (Apple M5, MPS)
$PY -I research/experiments/e10_ham_embed.py
for d in isic2020 isic2019_nonham pad_ufes; do $PY -I research/experiments/e13_replication.py $d embed; done

# 3. Analyses (each writes research/results/<ID>.json)
$PY -I research/experiments/e10a_ham_metadata_audit.py      # leakage audit
$PY -I research/experiments/e10_ham_attenuation.py          # pre-registered HAM10000 (Table II, Fig. 1)
$PY -I research/experiments/e10b_source_control.py          # source control + paired tests (Table IV)
$PY -I research/experiments/e10c_per_variable.py
$PY -I research/experiments/e10d_auroc_brier.py             # Table III
for d in isic2020 isic2019_nonham pad_ufes; do $PY -I research/experiments/e13_replication.py $d analyse; done  # Table V
$PY -I research/experiments/e16_within_class_recoverability.py
$PY -I research/experiments/e17_retained_fraction.py ham10000 isic2020 isic2019_nonham pad_ufes  # Table VIII, Fig. 2
$PY -I research/experiments/e21_complete_case.py            # complete-case table
$PY -I research/experiments/e12_ham_cam_lesion.py           # Table VI (predicted-class maps)
E12_MODE=fixed $PY -I research/experiments/e12_ham_cam_lesion.py
$PY -I research/experiments/e15_cam_backbones.py            # Table VII
$PY -I research/experiments/e14_finetune.py                 # fine-tuning section
# exploratory, reported in EXPERIMENTS.md: e09, e18, e19, e20

# 4. Tests of the maths and statistics
(cd novelty && .venv/bin/python -m pytest -q)               # SNN maths + added-value statistics
(cd research/pipeline && ../../novelty/.venv/bin/python -m pytest -q)  # calibration, saliency nulls, cohort, info identity

# 5. Paper
$PY -I paper/scripts/make_numbers.py && $PY -I paper/scripts/make_figures.py
(cd paper && tectonic main.tex)
```

## Pre-registrations
- `research/iteration2/PREREGISTRATION_E10.md` (HAM10000)
- `research/iteration3/PREREGISTRATION_REPLICATION.md` (ISIC 2020, ISIC 2019 non-HAM, PAD-UFES-20)

Deviations are listed in the paper (Section IV) and in `research/EXPERIMENTS.md`.

## Known issues
- Fold assignment and bootstrap use fixed seeds; GPU embeddings may differ in the last float digits between machines.
- `paper/REVIEW_LOG.md` records every review round and the status of each criticism.

## Iteration 6 (E50–E55; research/iteration6/REGISTRY.md)

```sh
cd "~/Desktop/CV EL/research/experiments"; PY=../../novelty/.venv/bin/python
for d in pad_ufes hiba msk2020 isic2020 ham10000 isic2019_nonham milk10k_derm milk10k_clin; do $PY -I e50_dose_restoration.py $d; done   # E50a
for d in hiba ham10000 pad_ufes milk10k_derm; do $PY -I e50_dose_restoration.py $d --concept site --light; done                           # E51
for d in hiba msk2020 milk10k_derm pad_ufes milk10k_clin ham10000 isic2019_nonham isic2020; do $PY -I e50_dose_restoration.py $d --light --conditional; done  # E50c
$PY -I e52_nonlinear_probe.py hiba msk2020 pad_ufes milk10k_derm milk10k_clin ham10000 isic2019_nonham isic2020 --concept age
$PY -I e52_nonlinear_probe.py hiba ham10000 pad_ufes milk10k_derm --concept site
$PY -I e53_estimators.py hiba msk2020 pad_ufes milk10k_derm milk10k_clin ham10000 isic2019_nonham isic2020
for d in milk10k_derm milk10k_clin ham10000 isic2019_nonham isic2020; do $PY -I e53b_erasure_fullfusion.py $d; done
$PY -I e50_dose_restoration.py isic2020 --fold-seed 1 --minimal   # E50b; likewise fold seed 2, HAM10000, ISIC 2019, and --rung resnet18
$PY -I e54_finetune_seeds.py hiba 1 2; $PY -I e54_finetune_seeds.py pad_ufes 1 2
$PY -I e55_shift_rules.py
$PY -I e50_age_label_diag.py
$PY -I e50_report.py; $PY -I iteration6_summary.py
```
