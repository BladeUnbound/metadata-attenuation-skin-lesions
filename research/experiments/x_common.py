"""Shared helpers for E30–E34 (research/iteration4/REGISTRY_E30_E34.md).

Everything here reuses the earlier, tested pieces (two-stage added value,
grouped folds, OOB blocks) and adds: a variable registry, complete-case masks,
cached out-of-fold image scores, metrics, recoverability probes, site harmonisation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))
from calibration import brier, calibration_intercept_slope  # noqa: E402
from common import RESEARCH  # noqa: E402
from e10_ham_attenuation import EPS, oof_scores, second_stage  # noqa: E402
from e13_replication import oof_block  # noqa: E402
from e17_retained_fraction import prevalence_ll  # noqa: E402
from e18_foundation_rung import load_dataset  # noqa: E402

DATASETS = ["ham10000", "isic2020", "isic2019_nonham", "pad_ufes", "hiba", "msk2020", "milk10k_derm", "milk10k_clin"]
RUNGS = ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16", "vitl16_swag"]

# (variable name, kind, column)
VARS = {
    "ham10000": [("age", "num", "age"), ("sex", "cat", "sex"), ("site", "cat", "localization")],
    "isic2020": [("age", "num", "age_approx"), ("sex", "cat", "sex"), ("site", "cat", "anatom_site_general_challenge")],
    "isic2019_nonham": [("age", "num", "age_approx"), ("sex", "cat", "sex"), ("site", "cat", "anatom_site_general")],
    "pad_ufes": [("age", "num", "age"), ("sex", "cat", "gender"), ("site", "cat", "region"),
                 ("fitzpatrick", "cat", "fitspatrick")] + [(c, "cat", c) for c in
                                                           ["itch", "grew", "hurt", "changed", "bleed", "elevation"]],
    "hiba": [("age", "num", "age_approx"), ("sex", "cat", "sex"), ("site", "cat", "anatom_site_1"),
             ("fitzpatrick", "cat", "fitzpatrick_skin_type")],
    "msk2020": [("age", "num", "age_approx"), ("sex", "cat", "sex"), ("site", "cat", "anatom_site_1")],
    "milk10k_derm": [("age", "num", "age_approx"), ("sex", "cat", "sex"), ("site", "cat", "anatom_site_1")],
    "milk10k_clin": [("age", "num", "age_approx"), ("sex", "cat", "sex"), ("site", "cat", "anatom_site_1")],
}
SOURCE_COL = {"ham10000": "dataset"}  # isic2019 source derived below


def missing(s):
    return (s.isna() | s.astype(str).str.strip().str.lower().isin(["nan", "", "unknown", "unk", "none"])).to_numpy()


def available(ds):
    try:
        d = load_dataset(ds)
    except Exception:
        return False
    return (d["dir"] / "emb_convnext_tiny.npy").exists() and len(d["y"]) > 0


def subset(ds, complete_case=True):
    """Dataset restricted to complete cases (all registered variables recorded) if requested."""
    d = load_dataset(ds)
    df, y, g = d["df"], d["y"], d["groups"]
    cols = [c for _, _, c in VARS[ds]]
    keep = ~np.any([missing(df[c]) for c in cols], axis=0) if complete_case else np.ones(len(df), bool)
    idx = np.flatnonzero(keep)
    return dict(d=d, idx=idx, df=df.iloc[idx].reset_index(drop=True), y=y[idx], g=g[idx],
                miss_rate={n: float(missing(df[c]).mean()) for n, _, c in VARS[ds]})


def folds_for(s, seed=0):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(s["df"], s["y"], s["g"]))


def emb(s, rung):
    p = s["d"]["dir"] / f"emb_{rung}.npy"
    return np.load(p)[s["idx"]] if p.exists() else None


def cached_score(ds, s, rung, tag, folds):
    """Out-of-fold image score, cached so E30/E32/E33 use identical scores."""
    out = s["d"]["dir"] / "oof" / f"score_{rung}_{tag}_s0.npy"
    out.parent.mkdir(exist_ok=True)
    if out.exists():
        return np.load(out)
    X = emb(s, rung)
    if X is None:
        return None
    score, _ = oof_scores(X, s["y"], s["g"], folds)
    np.save(out, score)
    return score


def block(s, items, folds):
    num = [c for _, k, c in items if k == "num"]
    cat = [c for _, k, c in items if k == "cat"]
    return oof_block(s["df"], num, cat, folds)


def per_image_ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def oof_probs(Z, y, folds):
    P = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(Z[tr])
        P[te] = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Z[tr]), y[tr]).predict_proba(sc.transform(Z[te]))[:, 1]
    return P


def group_boot_mean(values, groups, rng, B=2000):
    codes, inv = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(inv, weights=values), np.bincount(inv)
    idx = rng.integers(0, len(codes), (B, len(codes)))
    return sums[idx].sum(1) / counts[idx].sum(1)


def ci(samples):
    return [round(float(np.percentile(samples, 2.5)), 6), round(float(np.percentile(samples, 97.5)), 6)]


def full_metrics(y, p):
    out = {"auroc": float(roc_auc_score(y, p)), "logloss": float(per_image_ll(y, p).mean()), "brier": float(brier(y, p))}
    try:
        c = calibration_intercept_slope(y, p)
        out.update(calib_intercept=float(c["intercept"]), calib_slope=float(c["slope"]))
    except Exception:
        out.update(calib_intercept=None, calib_slope=None)
    return out


# ---------------------------------------------------------------- recoverability probes
def _probe_fold(X, t, tr, te, kind):
    sc = StandardScaler().fit(X[tr])
    if kind == "num":
        return te, RidgeCV(alphas=np.logspace(-1, 4, 6)).fit(sc.transform(X[tr]), t[tr]).predict(sc.transform(X[te]))
    return te, LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(X[tr]), t[tr]).decision_function(sc.transform(X[te]))


def _probe(X, t, groups, mask, kind, seed=0):
    idx = np.flatnonzero(mask)
    if kind != "num" and (t[idx].min() == t[idx].max() or min(np.bincount(t[idx].astype(int))) < 10):
        return None
    strat = pd.qcut(t[idx], 5, labels=False, duplicates="drop") if kind == "num" else t[idx]
    try:
        splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(idx, strat, groups[idx]))
    except ValueError:
        return None
    pred = np.full(len(t), np.nan)
    for te, p in Parallel(n_jobs=5)(delayed(_probe_fold)(X, t, idx[a], idx[b], kind) for a, b in splits):
        pred[te] = p
    return float(r2_score(t[idx], pred[idx])) if kind == "num" else float(roc_auc_score(t[idx], pred[idx]))


def recoverability(X, s, col, kind, min_level=30):
    """Marginal and label-conditional recoverability of one variable from features X.
    num: R^2; cat: macro one-vs-rest AUROC over levels with >= min_level images (binary vars: plain AUROC)."""
    v = s["df"][col]
    y, g = s["y"], s["g"]
    if kind == "num":
        t = pd.to_numeric(v, errors="coerce").to_numpy(float)
        ok = ~np.isnan(t)
        targets = [(t, ok)]
    else:
        lv = v.astype(str)
        levels = [l for l, n in lv.value_counts().items() if n >= min_level]
        if len(levels) < 2:
            return None
        if len(levels) == 2:
            levels = levels[:1]
        targets = [((lv == l).to_numpy(int), np.ones(len(lv), bool)) for l in levels]

    def score(mask_extra):
        vals = []
        for t, ok in targets:
            r = _probe(X, t, g, ok & mask_extra, kind)
            if r is not None:
                vals.append(r)
        return float(np.mean(vals)) if vals else None

    marginal = score(np.ones(len(y), bool))
    within, w = [], []
    for c in (0, 1):
        r = score(y == c)
        if r is not None:
            within.append(r)
            w.append((y == c).sum())
    return {"marginal": marginal, "within": float(np.average(within, weights=w)) if within else None}


def normalise_recov(x, kind):
    if x is None:
        return None
    return max(0.0, x) if kind == "num" else max(0.0, 2 * x - 1)


# ---------------------------------------------------------------- harmonised metadata (E33)
def site_coarse(v):
    s = str(v).lower()
    rules = [("palms_soles", ["palm", "sole", "acral"]), ("oral_genital", ["oral", "genital", "lip", "mucos"]),
             ("head_neck", ["head", "neck", "face", "scalp", "ear", "nose"]),
             ("upper_ext", ["upper extremity", "arm", "forearm", "hand"]),
             ("lower_ext", ["lower extremity", "thigh", "leg", "foot"]),
             ("trunk", ["trunk", "torso", "back", "chest", "abdomen"])]
    for name, keys in rules:
        if any(k in s for k in keys):
            return name
    return "unknown"


def harmonised(ds, s):
    cols = {n: c for n, _, c in VARS[ds]}
    df = s["df"]
    age = pd.to_numeric(df[cols["age"]], errors="coerce")
    sex = df[cols["sex"]].astype(str).str.lower().map(lambda x: x if x in ("male", "female") else "unknown")
    site = df[cols["site"]].map(site_coarse)
    return pd.DataFrame({"age": age.to_numpy(), "sex": sex.to_numpy(), "site": site.to_numpy()})


def isic_ids(ds, s):
    df = s["df"]
    for c in ("image_id", "image_name", "image", "isic_id"):
        if c in df:
            return set(df[c].astype(str).str.replace("_downsampled", "", regex=False))
    return set()
