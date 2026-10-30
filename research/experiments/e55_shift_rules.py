"""E55 (registered in research/iteration6/REGISTRY.md, 2026-10-29 13:49 IST): source-held-out evaluation of
metadata use with decision rules fitted on sources only.

Units: complete cases, frozen ConvNeXt-T features, harmonised age/sex/coarse site (E33 pipeline). Dataset level:
leave-one-dataset-out within a label family (melanoma: HAM10000, ISIC 2019 non-HAM, ISIC 2020; malignant: PAD-UFES-20,
HIBA, MSK 2020, MILK10k derm; MILK10k clin is not used because it shares lesions with derm). Within-dataset sources
(reported separately, never counted as independent): HAM10000 rosendahl / vidir_modern / vienna_dias held out
(vidir_molemax only as a training source: 24 melanomas); ISIC 2019 BCN <-> MSK.

Everything is fitted on the training units: the image head (grouped inner CV penalty), the second stage (on source
out-of-fold scores), metadata imputation and categories. The target's labels are used only to score.
Rules: R0 never use metadata; R1 always; R2 use iff the mean leave-one-source-out added value across the training units
is > 0 (needs >= 2 training units); R3 E34 B1 from sources only: predicted added value = pooled-source metadata-only
value x median source retained fraction; use iff >= 1e-3 nats.

Usage: python e55_shift_rules.py
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import Experiment
from e10_ham_attenuation import fit_tuned, oof_scores
from e33_shift import fit_lr, meta_design
from e40_crossfit_baseline import boot_idx, boot_mean, ci
from e50_dose_restoration import sha
from x_common import emb, harmonised, per_image_ll, subset

FAMILIES = {"melanoma": ["ham10000", "isic2019_nonham", "isic2020"],
            "malignant": ["pad_ufes", "hiba", "msk2020", "milk10k_derm"]}
THRESH = 1e-3
_cache = {}


def unit(name, s, mask=None, ds=None):
    m = np.ones(len(s["y"]), bool) if mask is None else mask
    H = harmonised(ds or name, s).iloc[np.flatnonzero(m)].reset_index(drop=True)
    return {"name": name, "X": emb(s, "convnext_tiny")[m].astype(np.float64), "y": s["y"][m],
            "g": np.array([f"{name}|{x}" for x in np.asarray(s["g"])[m]]), "H": H}


def pool(us):
    import pandas as pd
    return {"X": np.vstack([u["X"] for u in us]), "y": np.concatenate([u["y"] for u in us]),
            "g": np.concatenate([u["g"] for u in us]), "H": pd.concat([u["H"] for u in us], ignore_index=True)}


def folds_of(p):
    return list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(p["X"], p["y"], p["g"]))


def fit_apply(train, test):
    p = pool(train)
    oof, _ = oof_scores(p["X"], p["y"], p["g"], folds_of(p))
    sc = StandardScaler().fit(p["X"])
    head, C = fit_tuned(sc.transform(p["X"]), p["y"], p["g"])
    st = head.decision_function(sc.transform(test["X"]))
    Hs, Ht = meta_design(p["H"], test["H"])
    s0, m0 = fit_lr(oof[:, None], p["y"])
    s1, m1 = fit_lr(np.column_stack([oof, Hs]), p["y"])
    p0 = m0.predict_proba(s0.transform(st[:, None]))[:, 1]
    p1 = m1.predict_proba(s1.transform(np.column_stack([st, Ht])))[:, 1]
    return per_image_ll(test["y"], p0), per_image_ll(test["y"], p1), p0, p1, float(C)


def meta_only_value(u):
    """Grouped-CV metadata-only log-loss reduction over the training-fold prevalence (nats)."""
    key = ("I", u["name"], len(u["y"]))
    if key in _cache:
        return _cache[key]
    d = np.zeros(len(u["y"]))
    for tr, te in folds_of(u):
        a, b = meta_design(u["H"].iloc[tr].reset_index(drop=True), u["H"].iloc[te].reset_index(drop=True))
        sc = StandardScaler().fit(a)
        p = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(a), u["y"][tr]).predict_proba(sc.transform(b))[:, 1]
        prev = np.full(len(te), u["y"][tr].mean())
        d[te] = per_image_ll(u["y"][te], prev) - per_image_ll(u["y"][te], p)
    _cache[key] = float(d.mean())
    return _cache[key]


def in_dist_delta(u):
    key = ("D", u["name"], len(u["y"]))
    if key in _cache:
        return _cache[key]
    folds = folds_of(u)
    s, _ = oof_scores(u["X"], u["y"], u["g"], folds)
    d = np.zeros(len(u["y"]))
    for tr, te in folds:  # second stage on out-of-fold scores (as E33's in-distribution reference)
        a, b = meta_design(u["H"].iloc[tr].reset_index(drop=True), u["H"].iloc[te].reset_index(drop=True))
        s0, m0 = fit_lr(s[tr, None], u["y"][tr])
        s1, m1 = fit_lr(np.column_stack([s[tr], a]), u["y"][tr])
        d[te] = per_image_ll(u["y"][te], m0.predict_proba(s0.transform(s[te, None]))[:, 1]) - \
            per_image_ll(u["y"][te], m1.predict_proba(s1.transform(np.column_stack([s[te], b])))[:, 1])
    _cache[key] = float(d.mean())
    return _cache[key]


def evaluate(train, test):
    l0, l1, p0, p1, C = fit_apply(train, test)
    inv, idx = boot_idx(test["g"], np.random.default_rng(55))
    d = l0 - l1
    out = {"target": test["name"], "train": [u["name"] for u in train], "n_target": int(len(d)),
           "prevalence_target": round(float(test["y"].mean()), 4), "head_C": C,
           "LL_image": round(float(l0.mean()), 5), "LL_fused": round(float(l1.mean()), 5),
           "delta_x1e3": round(1e3 * float(d.mean()), 3), "delta_ci95_x1e3": [round(1e3 * v, 3) for v in ci(boot_mean(d, inv, idx))],
           "auroc_image": round(float(roc_auc_score(test["y"], p0)), 4), "auroc_fused": round(float(roc_auc_score(test["y"], p1)), 4)}
    # R2: leave-one-source-out among the training units
    if len(train) >= 2:
        loso = {}
        for j, u in enumerate(train):
            a0, a1, *_ = fit_apply([v for i, v in enumerate(train) if i != j], u)
            loso[u["name"]] = round(1e3 * float((a0 - a1).mean()), 3)
        out["R2_loso_x1e3"] = loso
        out["R2_use"] = bool(np.mean(list(loso.values())) > 0)
    else:
        out["R2_use"] = None
    # R3: E34 B1 from sources only
    I_src = meta_only_value(pool_unit(train))
    ratios = {u["name"]: (in_dist_delta(u), meta_only_value(u)) for u in train}
    ret = [dd / ii for dd, ii in ratios.values() if ii > 1e-4]
    pred = I_src * float(np.median(ret)) if ret else None
    out["R3_inputs"] = {"I_src": round(I_src, 5), "per_source_delta_I": {k: [round(v[0], 5), round(v[1], 5)] for k, v in ratios.items()},
                        "median_retained": round(float(np.median(ret)), 3) if ret else None,
                        "predicted_delta": round(pred, 5) if pred is not None else None}
    out["R3_use"] = bool(pred is not None and pred >= THRESH)
    best = min(l0.mean(), l1.mean())
    reg = lambda use: None if use is None else round(1e3 * float((l1.mean() if use else l0.mean()) - best), 3)
    out["regret_x1e3"] = {"R0_never": reg(False), "R1_always": reg(True), "R2_source_validated": reg(out["R2_use"]),
                          "R3_collection_rule": reg(out["R3_use"])}
    out["oracle_use"] = bool(l1.mean() < l0.mean())
    print(json.dumps(out), flush=True)
    return out


def pool_unit(us):
    p = pool(us)
    p["name"] = "+".join(u["name"] for u in us)
    return p


def main():
    subs = {d: subset(d, True) for f in FAMILIES.values() for d in f}
    res = {"dataset_level": {}, "within_dataset": {}}
    for fam, dss in FAMILIES.items():
        units = {d: unit(d, subs[d]) for d in dss}
        for t in dss:
            res["dataset_level"][f"{fam}:{t}"] = evaluate([units[d] for d in dss if d != t], units[t])
    s = subs["ham10000"]
    src = s["df"]["dataset"].astype(str).to_numpy()
    ham = {v: unit(f"ham:{v}", s, src == v, "ham10000") for v in np.unique(src)}
    for t in ["rosendahl", "vidir_modern", "vienna_dias"]:
        res["within_dataset"][f"ham10000:{t}"] = evaluate([u for k, u in ham.items() if k != t], ham[t])
    s = subs["isic2019_nonham"]
    src = s["df"]["lesion_id"].fillna("none").str.extract(r"^([A-Za-z]+)")[0].fillna("none").to_numpy()
    isic = {v: unit(f"isic2019:{v}", s, src == v, "isic2019_nonham") for v in ["BCN", "MSK"]}
    res["within_dataset"]["isic2019:BCN"] = evaluate([isic["MSK"]], isic["BCN"])
    res["within_dataset"]["isic2019:MSK"] = evaluate([isic["BCN"]], isic["MSK"])
    for level in ["dataset_level", "within_dataset"]:
        rows = [r for r in res[level].values()]
        res[f"{level}_summary"] = {
            "n_targets": len(rows),
            "mean_regret_x1e3": {k: (round(float(np.mean([r["regret_x1e3"][k] for r in rows if r["regret_x1e3"][k] is not None])), 3)
                                     if any(r["regret_x1e3"][k] is not None for r in rows) else None)
                                 for k in ["R0_never", "R1_always", "R2_source_validated", "R3_collection_rule"]},
            "n_oracle_use": int(sum(r["oracle_use"] for r in rows))}
    res["independence_note"] = ("7 dataset-level targets (MILK10k counted once); within-dataset sources share "
                                "acquisition context with their parent dataset and are not independent replications")
    return res


if __name__ == "__main__":
    exp = Experiment("E55_shift_rules", "iteration6/REGISTRY.md E55: metadata-use decision rules fitted on sources only, "
                     "evaluated on held-out datasets and sources", {"families": FAMILIES},
                     {"features": "frozen ConvNeXt-T", "metadata": "harmonised age/sex/coarse site, complete cases",
                      "threshold_nats": THRESH, "code_sha256_12": {p.name: sha(p) for p in [Path(__file__)]}}, 0)
    exp.finish(main(), interpretation="See iteration6 reports.")
