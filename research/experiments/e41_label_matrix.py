"""E41 (exploratory; registered in research/iteration5/REGISTRY.md, 22:30 IST): alternative label definitions and
complete cases under the strictly cross-fitted (E40) estimator, fold assignment 0.

Usage: python e41_label_matrix.py labels | cc
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import oof_scores, per_image_ll
from e13_replication import block_matrix
from e40_crossfit_baseline import boot_idx, boot_mean, ci, corrected, fit_predict, load
from x_common import VARS, subset

RUNGS = ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
MELD2 = "Malignant melanocytic proliferations (Melanoma)"
LABELS = [
    ("ham10000", "MAL", lambda df: df["dx"].isin(["mel", "bcc"])),
    ("ham10000", "MAL_akiec", lambda df: df["dx"].isin(["mel", "bcc", "akiec"])),
    ("isic2019_nonham", "MAL", lambda df: (df["MEL"] + df["BCC"] + df["SCC"]) > 0),
    ("isic2019_nonham", "MAL_ak", lambda df: (df["MEL"] + df["BCC"] + df["SCC"] + df["AK"]) > 0),
    ("pad_ufes", "MEL", lambda df: df["diagnostic"] == "MEL"),
    ("hiba", "MEL", lambda df: df["diagnosis_2"] == MELD2),
    ("msk2020", "MEL", lambda df: df["diagnosis_2"] == MELD2),
    ("milk10k_derm", "MEL", lambda df: df["diagnosis_2"] == MELD2),
    ("milk10k_clin", "MEL", lambda df: df["diagnosis_2"] == MELD2),
]
B_RHO = "convnext_tiny"


def analyse(X_of, y, groups, df, enc):
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    inv, idx = boot_idx(groups, np.random.default_rng(41))
    # metadata-only information I_A: training-fold prevalence vs metadata-only logistic, strictly per fold
    p_prev, p_meta = np.zeros(len(y)), np.zeros(len(y))
    for tr, te in folds:
        M = enc(tr)
        p_prev[te] = y[tr].mean()
        p_meta[te] = fit_predict(M[tr], y[tr], M[te])
    ia_i = per_image_ll(y, p_prev) - per_image_ll(y, p_meta)
    ia_bs = boot_mean(ia_i, inv, idx)
    out = {"n": int(len(y)), "positives": int(y.sum()), "n_groups": int(len(np.unique(groups))),
           "I_A": round(float(ia_i.mean()), 5), "I_A_ci95": ci(ia_bs), "rungs": {}}
    d_by = {}
    for rung in RUNGS:
        X = X_of(rung)
        score, _ = oof_scores(X, y, groups, folds)
        q0, q1 = corrected(X, y, groups, folds, enc, score)
        d = per_image_ll(y, q0) - per_image_ll(y, q1)
        d_by[rung] = d
        out["rungs"][rung] = {"image_auroc": round(float(roc_auc_score(y, score)), 4), "delta": round(float(d.mean()), 5),
                              "delta_ci95": ci(boot_mean(d, inv, idx)),
                              "delta_group_weighted": round(float(pd.Series(d).groupby(groups).mean().mean()), 5)}
        print("   ", rung, out["rungs"][rung], flush=True)
    d_bs = boot_mean(d_by[B_RHO], inv, idx)
    out["rho_cnx"] = round(float(d_by[B_RHO].mean() / ia_i.mean()), 4) if ia_i.mean() > 0 else None
    out["rho_cnx_ci95"] = ci(d_bs / ia_bs)
    out["I_A_small"] = bool(out["I_A_ci95"][0] < 2e-3)
    for hyp, (hi, lo) in {"H1": ("hsv", "convnext_tiny"), "H2": ("resnet18", "convnext_tiny")}.items():
        diff = d_by[hi] - d_by[lo]
        bs = boot_mean(diff, inv, idx)
        out[hyp] = {"diff": round(float(diff.mean()), 5), "ci95": ci(bs), "p_one_sided": float((1 + np.sum(bs <= 0)) / (len(bs) + 1))}
    return out


def run_labels():
    res = {}
    for ds, name, rule in LABELS:
        a = load(ds)
        y = rule(a["df"]).to_numpy(int)
        print(ds, name, "positives", int(y.sum()), flush=True)
        X_of = lambda r, ds=ds: np.load(RESEARCH / f"data/{ds}/emb_{r}.npy")
        res[f"{ds}|{name}"] = analyse(X_of, y, np.asarray(a["groups"]), a["df"], a["enc"])
    return res


def run_cc():
    res = {}
    for ds in ["ham10000", "isic2020", "isic2019_nonham", "pad_ufes", "hiba", "msk2020", "milk10k_derm", "milk10k_clin"]:
        s = subset(ds, complete_case=True)
        a = load(ds)
        assert np.array_equal(a["y"][s["idx"]], s["y"]), ds  # same row order as the embeddings
        num = [c for _, k, c in VARS[ds] if k == "num"]
        cat = [c for _, k, c in VARS[ds] if k == "cat"]
        df = s["df"]
        enc = lambda tr, df=df, num=num, cat=cat: block_matrix(df, num, cat, tr)
        X_of = lambda r, ds=ds, idx=s["idx"]: np.load(RESEARCH / f"data/{ds}/emb_{r}.npy")[idx]
        print(ds, "complete cases", len(s["idx"]), flush=True)
        res[f"{ds}|cc"] = analyse(X_of, s["y"], np.asarray(s["g"]), df, enc)
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    exp = Experiment(f"E41_{mode}", "REGISTRY.md E41 (exploratory): replication under alternative labels / complete cases",
                     {"labels": [f"{d}|{n}" for d, n, _ in LABELS]} if mode == "labels" else {"subset": "complete cases (x_common.VARS)"},
                     {"estimator": "E40 corrected", "fold_assignment": 0, "B": 2000}, 0)
    exp.finish(run_labels() if mode == "labels" else run_cc(), interpretation="See research/EXPERIMENTS.md.")
