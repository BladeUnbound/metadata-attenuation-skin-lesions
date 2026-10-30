"""E10 step 2: metadata added value vs image-model capacity on HAM10000.

Follows research/iteration2/PREREGISTRATION_E10.md. Deviations are flagged in
the output under "exploratory".
For each fold assignment (3 seeds) and each ladder rung:
  1. Out-of-fold image score: L2 logistic on standardised features, penalty
     chosen by lesion-grouped inner 3-fold CV inside the training fold only.
  2. Second stage (same outer folds): logistic on [score] vs [score + metadata];
     per-image out-of-fold log-loss.
  3. Added value = mean per-image log-loss difference, 95% CI by bootstrap over lesions.
Seed 0 also runs: full features + metadata (overfitting control), recoverability
of metadata and of image source from features, and the 7-class secondary task.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import log_loss, r2_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment

DATA = RESEARCH / "data/ham10000"
DEV_RUN = os.environ.get("HAM_DEV") == "1"  # pipeline check only; results are not reported
TAG = "dev_" if DEV_RUN else ""
RUNGS = ["hsv", "resnet18", "resnet50"] if DEV_RUN else ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
CS = np.logspace(-4, 0, 5)
SEEDS = [0] if DEV_RUN else [0, 1, 2]
EPS = 1e-12


def meta_matrix(df, train_idx):
    age = df["age"].copy()
    med = age.iloc[train_idx].median()
    cols = [age.fillna(med).to_numpy()[:, None] / 100, age.isna().to_numpy(float)[:, None],
            pd.get_dummies(df["sex"].fillna("unknown"), dtype=float).to_numpy(),
            pd.get_dummies(df["localization"].fillna("unknown"), dtype=float).to_numpy()]
    return np.hstack(cols)


def fit_tuned(X, y, groups, multiclass=False):
    """Choose C by lesion-grouped inner CV (log-loss), then refit on all of X."""
    inner = GroupKFold(3)
    best, best_ll = None, np.inf
    for C in CS:
        lls = []
        for tr, va in inner.split(X, y, groups):
            m = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr])
            lls.append(log_loss(y[va], m.predict_proba(X[va]), labels=np.unique(y)))
        if np.mean(lls) < best_ll:
            best, best_ll = C, np.mean(lls)
    return LogisticRegression(C=best, max_iter=3000).fit(X, y), best


def outer_fold(X, y, groups, tr, te, multiclass=False):
    sc = StandardScaler().fit(X[tr])
    m, C = fit_tuned(sc.transform(X[tr]), y[tr], groups[tr], multiclass)
    Xt = sc.transform(X[te])
    return te, (m.predict_log_proba(Xt) if multiclass else m.decision_function(Xt)), C


def oof_scores(X, y, groups, folds, multiclass=False):
    outs = Parallel(n_jobs=5)(delayed(outer_fold)(X, y, groups, tr, te, multiclass) for tr, te in folds)
    k = len(np.unique(y)) if multiclass else 1
    S = np.zeros((len(y), k)) if multiclass else np.zeros(len(y))
    for te, s, _ in outs:
        S[te] = s
    return S, [c for *_, c in outs]


def per_image_ll(y, P):
    P = np.clip(P, EPS, 1 - EPS)
    if P.ndim == 1:
        return -(y * np.log(P) + (1 - y) * np.log(1 - P))
    return -np.log(P[np.arange(len(y)), y])


def second_stage(Z_img, Z_both, y, folds, multiclass=False):
    lls = []
    for Z in (Z_img, Z_both):
        P = np.zeros((len(y), len(np.unique(y)))) if multiclass else np.zeros(len(y))
        for tr, te in folds:
            sc = StandardScaler().fit(Z[tr])
            m = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Z[tr]), y[tr])
            pr = m.predict_proba(sc.transform(Z[te]))
            P[te] = pr if multiclass else pr[:, 1]
        lls.append(per_image_ll(y, P))
    return lls


def lesion_bootstrap(d, groups, rng, B=2000):
    codes, inv = np.unique(groups, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    counts = np.bincount(inv)
    stats = []
    for _ in range(B):
        idx = rng.integers(0, len(codes), len(codes))
        stats.append(sums[idx].sum() / counts[idx].sum())
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def recoverability(X, df, groups, folds):
    out = {}
    age = df["age"].to_numpy()
    ok = ~np.isnan(age)
    pred = np.zeros(len(age))
    for tr, te in folds:
        tr, te = tr[ok[tr]], te[ok[te]]
        sc = StandardScaler().fit(X[tr])
        pred[te] = RidgeCV(alphas=np.logspace(-1, 4, 6)).fit(sc.transform(X[tr]), age[tr]).predict(sc.transform(X[te]))
    out["age_r2"] = round(float(r2_score(age[ok], pred[ok])), 3)

    def bin_auc(target, mask):
        p = np.zeros(len(target))
        for tr, te in folds:
            tr, te = tr[mask[tr]], te[mask[te]]
            sc = StandardScaler().fit(X[tr])
            p[te] = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(X[tr]), target[tr]).decision_function(sc.transform(X[te]))
        return round(float(roc_auc_score(target[mask], p[mask])), 3)

    sex_mask = df["sex"].isin(["male", "female"]).to_numpy()
    out["sex_auroc"] = bin_auc((df["sex"] == "male").to_numpy(int), sex_mask)
    top_sites = df["localization"].value_counts().index[:5]
    out["site_auroc_top5_mean"] = round(float(np.mean([bin_auc((df["localization"] == s).to_numpy(int),
                                                                np.ones(len(df), bool)) for s in top_sites])), 3)
    out["exploratory_source_auroc_mean"] = round(float(np.mean([bin_auc((df["dataset"] == s).to_numpy(int),
                                                                         np.ones(len(df), bool))
                                                                 for s in df["dataset"].unique()])), 3)
    return out


def main():
    df = pd.read_csv(DATA / ("meta_dev.csv" if DEV_RUN else "raw/HAM10000_metadata.csv"))
    y = (df["dx"] == "mel").to_numpy(int)
    y7 = pd.factorize(df["dx"], sort=True)[0]
    groups = df["lesion_id"].to_numpy()
    rng = np.random.default_rng(0)
    exp = Experiment("E10_ham_attenuation" + ("_DEV" if DEV_RUN else ""), "See research/iteration2/PREREGISTRATION_E10.md",
                     {"source": "HAM10000 (Dataverse, CC BY-NC 4.0)", "n_images": len(df),
                      "n_lesions": int(df["lesion_id"].nunique())},
                     {"rungs": RUNGS, "Cs": CS.tolist(), "seeds": SEEDS, "outer": "5-fold StratifiedGroupKFold(lesion)",
                      "inner": "3-fold GroupKFold(lesion)", "bootstrap": "2000 lesion resamples"}, SEEDS)
    res = {"primary": {}, "seed0_extra": {}}
    for seed in SEEDS:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(df, y, groups))
        res["primary"][f"seed{seed}"] = {}
        for rung in RUNGS:
            X = np.load(DATA / f"emb_{TAG}{rung}.npy")
            score, Cs = oof_scores(X, y, groups, folds)
            M = np.zeros((len(df), meta_matrix(df, folds[0][0]).shape[1]))
            for tr, te in folds:
                M[te] = meta_matrix(df, tr)[te]
            ll_img, ll_both = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
            d = ll_img - ll_both
            r = {"image_auroc": round(float(roc_auc_score(y, score)), 4),
                 "image_logloss": round(float(ll_img.mean()), 4),
                 "both_logloss": round(float(ll_both.mean()), 4),
                 "added_value": round(float(d.mean()), 5),
                 "added_value_ci95": [round(v, 5) for v in lesion_bootstrap(d, groups, rng)],
                 "relative_added_value": round(float(d.mean() / ll_img.mean()), 4),
                 "chosen_C": Cs}
            res["primary"][f"seed{seed}"][rung] = r
            print(seed, rung, {k: v for k, v in r.items() if k != "chosen_C"}, flush=True)
            if seed == 0:
                extra = {"recoverability": recoverability(X, df, groups, folds)}
                Xm = np.hstack([X, M])
                s_full, _ = oof_scores(Xm, y, groups, folds)
                ll_full = per_image_ll(y, 1 / (1 + np.exp(-s_full)))
                extra["full_features_plus_meta_logloss"] = round(float(ll_full.mean()), 4)
                extra["full_features_plus_meta_auroc"] = round(float(roc_auc_score(y, s_full)), 4)
                S7, _ = oof_scores(X, y7, groups, folds, multiclass=True)
                l7_img, l7_both = second_stage(S7, np.column_stack([S7, M]), y7, folds, multiclass=True)
                extra["7class_image_logloss"] = round(float(l7_img.mean()), 4)
                extra["7class_added_value"] = round(float((l7_img - l7_both).mean()), 5)
                res["seed0_extra"][rung] = extra
                print("   extra", extra, flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
