"""Descriptive, post hoc (REGISTRY amendment 3, 2026-10-29 18:44 IST): how strongly the label depends on age in each
E50 image set. On the E50 analysis rows with age recorded: AUROC of raw age for the label, and the Pearson correlation
between age and the strictly cross-fitted ConvNeXt-T image score of E50's t = 0 arm (fold assignment 0). No model is
fitted here; values are read from data/<ds>/e50/age_convnext_tiny_f0.npz.

Usage: python e50_age_label_diag.py
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e45_erasure import concept, prepare

SETS = ["isic2020", "hiba", "msk2020", "ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes", "milk10k_clin"]


def main():
    out = {}
    for ds in SETS:
        idx, df, y, g, _ = prepare(ds)
        z, obs = concept(df, ds, "age")
        z = z[:, 0]
        f = np.load(RESEARCH / f"data/{ds}/e50/age_convnext_tiny_f0.npz")
        assert np.array_equal(f["y"], y), ds
        s = f["t0|score"]
        out[ds] = {"n_age_observed": int(obs.sum()), "age_auroc_for_label": round(float(roc_auc_score(y[obs], z[obs])), 3),
                   "corr_score_age": round(float(np.corrcoef(s[obs], z[obs])[0, 1]), 3)}
        print(ds, out[ds], flush=True)
    return out


if __name__ == "__main__":
    exp = Experiment("E50_age_label_diag", "Descriptive (post hoc, amendment 3): age-label dependence per E50 set",
                     {"sets": SETS}, {"score": "E50 t0 arm, ConvNeXt-T, fold assignment 0"}, 0)
    exp.finish(main(), interpretation="Descriptive only; see iteration6/FINAL_REPORT.md section 4.")
