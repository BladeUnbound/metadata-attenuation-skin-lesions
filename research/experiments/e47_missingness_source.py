"""E47 (exploratory; registered in research/iteration5/REGISTRY.md, 22:03 IST): missingness and source dependence.

(a) Missingness by label (all datasets) and by source (HAM10000 `dataset`; ISIC 2019 lesion-ID prefix), with counts.
(b) Metadata encodings under the strictly cross-fitted estimator (fold assignment 0; HSV and ConvNeXt-T):
    'indicators'  pre-registered (training-fold median + missing indicator; "unknown" category);
    'no_ind'      training-fold median / mode imputation, no indicator, no "unknown" level;
    'mi5'         multiple imputation without indicators: IterativeImputer (BayesianRidge, sample_posterior) fitted on the
                  training fold over age + one-hot categories (missing rows NaN), m = 5, fusion probabilities averaged.
(c) Source-aware evaluation: HAM10000 leave-one-source-out; ISIC 2019 BCN -> MSK and MSK -> BCN. Training sources give
    inner grouped out-of-fold scores for the second stage; the held-out source is scored by a first stage fitted on all
    training sources and is used only for prediction.

Usage: python e47_missingness_source.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import oof_scores, outer_fold, per_image_ll
from e40_crossfit_baseline import boot_idx, boot_mean, ci, corrected, fit_predict, load
from x_common import VARS, missing

DATASETS = ["ham10000", "isic2020", "isic2019_nonham", "pad_ufes", "hiba", "msk2020", "milk10k_derm", "milk10k_clin"]
RUNGS = ["hsv", "convnext_tiny"]


def cols(ds):
    return [c for _, k, c in VARS[ds] if k == "num"], [c for _, k, c in VARS[ds] if k == "cat"]


def enc_no_ind(df, num, cat):
    def enc(tr):
        out = []
        for c in num:
            v = pd.to_numeric(df[c], errors="coerce")
            out.append(v.fillna(v.iloc[tr].median()).to_numpy()[:, None] / 100)
        for c in cat:
            v = df[c].astype(str).where(~missing(df[c]))
            out.append(pd.get_dummies(v.fillna(v.iloc[tr].mode().iloc[0]), dtype=float).to_numpy())
        return np.hstack(out)
    return enc


def design_nan(df, num, cat):
    parts = [pd.to_numeric(df[c], errors="coerce").to_numpy()[:, None] / 100 for c in num]
    for c in cat:
        miss = missing(df[c])
        D = np.array(pd.get_dummies(df[c].astype(str).where(~miss), dtype=float).to_numpy(), copy=True)
        D[miss] = np.nan
        parts.append(D)
    return np.hstack(parts)


def mi_fusion(score_in, score_out, y, folds, Dnan, m=5):
    """Strictly cross-fitted fusion with multiple imputation; returns averaged q1."""
    q1 = np.zeros(len(y))
    for k, (tr, te) in enumerate(folds):
        acc = np.zeros(len(te))
        for r in range(m):
            imp = IterativeImputer(estimator=BayesianRidge(), sample_posterior=True, random_state=r, max_iter=10)
            imp.fit(Dnan[tr])
            Dt = imp.transform(Dnan)
            acc += fit_predict(np.column_stack([score_in[k][tr], Dt[tr]]), y[tr], np.column_stack([score_out[te], Dt[te]]))
        q1[te] = acc / m
    return q1


def inner_scores(X, y, g, folds):
    """Inner out-of-fold scores within each training fold (as in E40), returned per outer fold."""
    out = []
    for tr, te in folds:
        s = np.full(len(y), np.nan)
        res = Parallel(n_jobs=8)(delayed(outer_fold)(X, y, g, tr[a], tr[b])
                                 for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], g[tr]))
        for idx, sc, _ in res:
            s[idx] = sc
        out.append(s)
    return out


def encodings(ds):
    a = load(ds)
    df, y, g = a["df"], a["y"], np.asarray(a["groups"])
    num, cat = cols(ds)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, g))
    inv, idx = boot_idx(g, np.random.default_rng(47))
    res = {}
    for rung in RUNGS:
        X = np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy")
        s_out, _ = oof_scores(X, y, g, folds)
        s_in = inner_scores(X, y, g, folds)
        q0 = np.zeros(len(y))
        for k, (tr, te) in enumerate(folds):
            q0[te] = fit_predict(s_in[k][tr, None], y[tr], s_out[te, None])
        l0 = per_image_ll(y, q0)
        r = {}
        for name, enc in [("indicators", a["enc"]), ("no_ind", enc_no_ind(df, num, cat))]:
            q1 = np.zeros(len(y))
            for k, (tr, te) in enumerate(folds):
                M = enc(tr)
                q1[te] = fit_predict(np.column_stack([s_in[k][tr], M[tr]]), y[tr], np.column_stack([s_out[te], M[te]]))
            d = l0 - per_image_ll(y, q1)
            r[name] = {"delta": round(float(d.mean()), 5), "ci95": ci(boot_mean(d, inv, idx))}
        d = l0 - per_image_ll(y, mi_fusion(s_in, s_out, y, folds, design_nan(df, num, cat)))
        r["mi5"] = {"delta": round(float(d.mean()), 5), "ci95": ci(boot_mean(d, inv, idx))}
        res[rung] = r
        print(ds, rung, r, flush=True)
    return res


def missing_tables(ds):
    a = load(ds)
    df, y = a["df"], a["y"]
    out = {}
    for n, _, c in VARS[ds]:
        m = missing(df[c])
        out[n] = {"n_missing": int(m.sum()), "pos_rate_missing": round(float(y[m].mean()), 3) if m.any() else None,
                  "pos_rate_observed": round(float(y[~m].mean()), 3)}
    src = None
    if ds == "ham10000":
        src = df["dataset"].astype(str)
    elif ds == "isic2019_nonham":
        src = df["lesion_id"].fillna("none").str.extract(r"^([A-Za-z]+)")[0].fillna("none")
    if src is not None:
        anym = np.any([missing(df[c]) for _, _, c in VARS[ds]], axis=0)
        out["by_source"] = {s: {"n": int((src == s).sum()), "any_missing": round(float(anym[(src == s).to_numpy()].mean()), 3),
                                "pos_rate": round(float(y[(src == s).to_numpy()].mean()), 3)} for s in src.unique()}
    return out


def source_split(ds, train_mask, test_mask):
    a = load(ds)
    y, g = a["y"], np.asarray(a["groups"])
    tr, te = np.flatnonzero(train_mask), np.flatnonzero(test_mask)
    res = {"n_train": int(len(tr)), "n_test": int(len(te)), "pos_test": int(y[te].sum())}
    if y[te].sum() < 5 or y[te].sum() == len(te):
        res["skipped"] = "too few positives or negatives in held-out source"
        return res
    for rung in RUNGS:
        X = np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy")
        q0, q1 = corrected(X, y, g, [(tr, te)], a["enc"], oof_scores(X, y, g, [(tr, te)])[0])
        d = (per_image_ll(y, q0) - per_image_ll(y, q1))[te]
        inv, idx = boot_idx(g[te], np.random.default_rng(470))
        res[rung] = {"delta": round(float(d.mean()), 5), "ci95": ci(boot_mean(d, inv, idx))}
    return res


def main():
    out = {"missingness": {ds: missing_tables(ds) for ds in DATASETS}}
    out["encodings"] = {ds: encodings(ds) for ds in DATASETS}
    ham = load("ham10000")["df"]["dataset"].astype(str).to_numpy()
    out["source_ham_loso"] = {s: source_split("ham10000", ham != s, ham == s) for s in np.unique(ham)}
    src = load("isic2019_nonham")["df"]["lesion_id"].fillna("none").str.extract(r"^([A-Za-z]+)")[0].fillna("none").to_numpy()
    out["source_isic2019"] = {"BCN_to_MSK": source_split("isic2019_nonham", src == "BCN", src == "MSK"),
                              "MSK_to_BCN": source_split("isic2019_nonham", src == "MSK", src == "BCN")}
    for k in ["source_ham_loso", "source_isic2019"]:
        print(k, out[k], flush=True)
    return out


if __name__ == "__main__":
    exp = Experiment("E47_missingness_source", "REGISTRY.md E47 (exploratory): missingness handling and source dependence",
                     {"datasets": DATASETS}, {"rungs": RUNGS, "mi": "IterativeImputer(BayesianRidge, sample_posterior), m=5",
                                              "estimator": "E40 corrected"}, 0)
    exp.finish(main(), interpretation="See research/EXPERIMENTS.md.")
