"""E53 (registered in research/iteration6/REGISTRY.md, 2026-10-29 13:49 IST): estimator robustness of metadata
added value. Same outer folds, grouping and test cases as E40 fold assignment 0; full metadata block; every penalty,
scaler, imputation median and category map is fitted on the outer training fold.

(a) score fusion: E40 corrected (logistic on [inner OOF score] vs [score, metadata]); reproduces E40's saved arrays.
(b) full-feature regularised fusion: L2 logistic on [standardised features, metadata encoding], C by grouped inner CV
    (E10 grid); counterpart: the same procedure on the features alone (the tuned first stage).
(c) small nonlinear fusion: HistGradientBoosting on [score, raw metadata with native missing/categorical handling],
    fixed hyperparameters (no tuning); counterpart: the same model on [score].
(d) missingness-only: logistic on [score, missingness indicators of the registered variables] vs [score].

Usage: python e53_estimators.py <dataset> [...]   |   python e53_estimators.py summary
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, RESULTS, Experiment
from e10_ham_attenuation import fit_tuned, outer_fold, per_image_ll
from e40_crossfit_baseline import boot_idx, boot_mean, ci, fit_predict, load
from e50_dose_restoration import sha
from x_common import VARS, missing

RUNGS = ["hsv", "resnet18", "convnext_tiny"]
DATASETS = ["ham10000", "isic2020", "isic2019_nonham", "pad_ufes", "hiba", "msk2020", "milk10k_derm", "milk10k_clin"]
HGB = dict(max_iter=200, learning_rate=0.05, max_depth=3, min_samples_leaf=20, l2_regularization=1.0,
           early_stopping=False, random_state=0)


def raw_meta(df, ds, tr):
    """Raw metadata for HGB: numeric as float (NaN kept); categorical as codes of the training fold's categories
    (missing or unseen -> NaN)."""
    cols, is_cat = [], []
    for _, kind, c in VARS[ds]:
        v = df[c]
        m = missing(v)
        if kind == "num":
            x = pd.to_numeric(v, errors="coerce").to_numpy(float, copy=True)
            x[m] = np.nan
            cols.append(x)
            is_cat.append(False)
        else:
            s = v.astype(str).to_numpy()
            cats = sorted(set(s[tr][~m[tr]]))
            code = {c_: i for i, c_ in enumerate(cats)}
            x = np.array([code.get(t, np.nan) for t in s], float)
            x[m] = np.nan
            cols.append(x)
            is_cat.append(True)
    return np.column_stack(cols), np.array(is_cat)


def miss_matrix(df, ds):
    return np.column_stack([missing(df[c]).astype(float) for _, _, c in VARS[ds]])


def full_fit(X, M, y, g, tr, te):
    sx = StandardScaler().fit(X[tr])
    sm = StandardScaler().fit(M[tr])
    Z = np.column_stack([sx.transform(X), sm.transform(M)])
    m, C = fit_tuned(Z[tr], y[tr], g[tr])
    return te, m.predict_proba(Z[te])[:, 1], C


def run_rung(ds, a, rung, folds, inner):
    y, g, df = a["y"], np.asarray(a["groups"]), a["df"]
    X = np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy").astype(np.float64)
    jobs, tags = [], []
    for k, (tr, te) in enumerate(folds):
        jobs.append(delayed(outer_fold)(X, y, g, tr, te))
        tags.append((k, -1))
        for j, (i1, i2) in enumerate(inner[k]):
            jobs.append(delayed(outer_fold)(X, y, g, tr[i1], tr[i2]))
            tags.append((k, j))
    full_jobs = [delayed(full_fit)(X, a["enc"](tr), y, g, tr, te) for tr, te in folds]
    outs = Parallel(n_jobs=8)(jobs + full_jobs)
    outs, fulls = outs[:len(jobs)], outs[len(jobs):]
    n = len(y)
    P = {k_: np.zeros(n) for k_ in ["a0", "a1", "b0", "b1", "c0", "c1", "d1"]}
    s_out = np.zeros(n)
    C_full = []
    Mmiss = miss_matrix(df, ds)
    for k, (tr, te) in enumerate(folds):
        s_in = np.full(n, np.nan)
        for (idx, sc, _), (kk, j) in zip(outs, tags):
            if kk == k:
                (s_out if j < 0 else s_in)[idx] = sc
        M = a["enc"](tr)
        P["a0"][te] = fit_predict(s_in[tr, None], y[tr], s_out[te, None])
        P["a1"][te] = fit_predict(np.column_stack([s_in[tr], M[tr]]), y[tr], np.column_stack([s_out[te], M[te]]))
        P["d1"][te] = fit_predict(np.column_stack([s_in[tr], Mmiss[tr]]), y[tr], np.column_stack([s_out[te], Mmiss[te]]))
        P["b0"][te] = 1 / (1 + np.exp(-s_out[te]))
        R, is_cat = raw_meta(df, ds, tr)
        h0 = HistGradientBoostingClassifier(**HGB).fit(s_in[tr, None], y[tr])
        P["c0"][te] = h0.predict_proba(s_out[te, None])[:, 1]
        h1 = HistGradientBoostingClassifier(categorical_features=np.r_[False, is_cat], **HGB).fit(
            np.column_stack([s_in[tr], R[tr]]), y[tr])
        P["c1"][te] = h1.predict_proba(np.column_stack([s_out[te], R[te]]))[:, 1]
    for te, p, C in fulls:
        P["b1"][te] = p
        C_full.append(float(C))
    L = {k_: per_image_ll(y, v) for k_, v in P.items()}
    return L, C_full


def run(ds):
    a = load(ds)
    y, g = a["y"], np.asarray(a["groups"])
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(a["df"], y, g))
    inner = [list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], g[tr])) for tr, _ in folds]
    inv, bidx = boot_idx(g, np.random.default_rng(53))
    out_dir = RESEARCH / f"data/{ds}/e53"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    f = lambda v: round(1e3 * float(v), 3)
    for rung in RUNGS:
        L, C_full = run_rung(ds, a, rung, folds, inner)
        np.savez_compressed(out_dir / f"{rung}.npz", y=y, groups=g.astype(str), **L)
        e40 = np.load(RESEARCH / f"data/{ds}/e40/{rung}_s0.npz")
        D = {"a_score_fusion": L["a0"] - L["a1"], "b_full_feature": L["b0"] - L["b1"], "c_hgb": L["c0"] - L["c1"],
             "d_missingness_only": L["a0"] - L["d1"]}
        r = {"reproduces_E40_corrected": bool(np.allclose(D["a_score_fusion"], e40["delta"], atol=1e-8)),
             "C_full_feature": C_full,
             "image_logloss": {k_: round(float(L[k_].mean()), 5) for k_ in ["a0", "b0", "c0"]},
             "fusion_logloss": {k_: round(float(L[k_].mean()), 5) for k_ in ["a1", "b1", "c1", "d1"]}}
        for name, d in D.items():
            r[name] = {"delta_x1e3": f(d.mean()), "ci95_x1e3": [f(v) for v in ci(boot_mean(d, inv, bidx))]}
        for name in ["b_full_feature", "c_hgb", "d_missingness_only"]:
            diff = D[name] - D["a_score_fusion"]
            r[name]["minus_a_x1e3"] = {"diff": f(diff.mean()), "ci95": [f(v) for v in ci(boot_mean(diff, inv, bidx))]}
        res[rung] = r
        print(ds, rung, json.dumps(r), flush=True)
    # H1-type contrast under each estimator: delta(HSV) - delta(ConvNeXt-T), paired
    Lh, Lc = (np.load(out_dir / f"{r_}.npz") for r_ in ["hsv", "convnext_tiny"])
    pairs = {"a_score_fusion": ("a0", "a1"), "b_full_feature": ("b0", "b1"), "c_hgb": ("c0", "c1"),
             "d_missingness_only": ("a0", "d1")}
    res["H1_hsv_minus_convnext"] = {}
    for name, (i0, i1) in pairs.items():
        d = (Lh[i0] - Lh[i1]) - (Lc[i0] - Lc[i1])
        bs = boot_mean(d, inv, bidx)
        res["H1_hsv_minus_convnext"][name] = {"diff_x1e3": f(d.mean()), "ci95_x1e3": [f(v) for v in ci(bs)],
                                              "p_one_sided": round(float((1 + np.sum(bs <= 0)) / (len(bs) + 1)), 5)}
    return res


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        exp = Experiment(f"E53_{ds}", "iteration6/REGISTRY.md E53: estimator robustness (score fusion, full-feature "
                         "fusion, boosted fusion, missingness-only)", {"dataset": ds},
                         {"rungs": RUNGS, "hgb": HGB, "fold_seed": 0, "inner_fold_seed": 0, "B": 2000,
                          "code_sha256_12": {p.name: sha(p) for p in [Path(__file__)]}}, 0)
        exp.finish(run(ds), interpretation="See iteration6 reports.")
