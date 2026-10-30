"""E16: recoverability of metadata from images, conditional on the label.

Diagnosis of the failed PAD-UFES-20 mechanism test (E13 M3): an image model
that predicts malignancy will "recover" any variable correlated with
malignancy, so marginal recoverability is confounded by the label. Here a
variable's recoverability is measured within each label class separately
(Y=0 and Y=1), out of fold with the same grouping, then averaged weighted by
class size. Proposition 1 concerns I(Y;A|X); label-conditional recoverability
is the quantity that speaks to whether the image carries A's information
beyond what it says about Y.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e13_replication import adapter

LADDER = ["hsv", "resnet18", "convnext_tiny"]


def oof_binary(X, t, mask, groups, seed=0):
    p = np.full(len(t), np.nan)
    idx = np.flatnonzero(mask)
    if t[idx].min() == t[idx].max():
        return None
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, t[idx], groups[idx]):
        tr, te = idx[tr], idx[te]
        sc = StandardScaler().fit(X[tr])
        p[te] = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(X[tr]), t[tr]).decision_function(sc.transform(X[te]))
    return float(roc_auc_score(t[idx], p[idx]))


def oof_r2(X, v, mask, groups, seed=0):
    idx = np.flatnonzero(mask)
    pred = np.full(len(v), np.nan)
    bins = pd.qcut(v[idx], 5, labels=False, duplicates="drop")
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, bins, groups[idx]):
        tr, te = idx[tr], idx[te]
        sc = StandardScaler().fit(X[tr])
        pred[te] = RidgeCV(alphas=np.logspace(-1, 4, 6)).fit(sc.transform(X[tr]), v[tr]).predict(sc.transform(X[te]))
    return float(r2_score(v[idx], pred[idx]))


def variables(name, df):
    if name == "pad_ufes":
        out = {}
        for c in ["itch", "grew", "hurt", "changed", "bleed", "elevation"]:
            v = df[c].astype(str)
            out[c] = ("bin", (v == "True").to_numpy(int), v.isin(["True", "False"]).to_numpy())
        out["age"] = ("num", pd.to_numeric(df["age"], errors="coerce").to_numpy(), None)
        return out
    out = {"sex": ("bin", (df["sex"] == "male").to_numpy(int), df["sex"].isin(["male", "female"]).to_numpy()),
           "age": ("num", pd.to_numeric(df["age"], errors="coerce").to_numpy(), None)}
    site = df["localization"]
    for s in site.value_counts().index[:3]:
        out[f"site={s}"] = ("bin", (site == s).to_numpy(int), np.ones(len(df), bool))
    return out


def run(name):
    if name == "ham10000":
        from e10_ham_embed import DATA
        df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
        y = (df["dx"] == "mel").to_numpy(int)
        groups = df["lesion_id"].to_numpy()
        emb = lambda k: np.load(DATA / f"emb_{k}.npy")
    else:
        a = adapter(name)
        df, y, groups = a["df"], a["y"], a["groups"]
        emb = lambda k: np.load(RESEARCH / f"data/{name}/emb_{k}.npy")
    out = {}
    for var, (kind, v, base_mask) in variables(name, df).items():
        ok = ~np.isnan(v) if kind == "num" else base_mask
        out[var] = {}
        for k in LADDER:
            X = emb(k)
            f = oof_binary if kind == "bin" else oof_r2
            marginal = f(X, v, ok, groups)
            within, weights = [], []
            for c in (0, 1):
                m = ok & (y == c)
                val = f(X, v, m, groups)
                if val is not None:
                    within.append(val)
                    weights.append(m.sum())
            out[var][k] = {"marginal": round(marginal, 3) if marginal is not None else None,
                           "within_class": round(float(np.average(within, weights=weights)), 3) if within else None}
        print(name, var, out[var], flush=True)
    return out


def main(datasets):
    exp = Experiment("E16_within_class_recoverability",
                     "Marginal recoverability of metadata overstates what images carry beyond the label.",
                     {"datasets": datasets}, {"ladder": LADDER, "metric": "AUROC (binary) / R^2 (age), out of fold"}, 0)
    res = {d: run(d) for d in datasets}
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["pad_ufes", "ham10000"])
