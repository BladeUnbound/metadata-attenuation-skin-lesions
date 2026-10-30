"""E52 (registered in research/iteration6/REGISTRY.md, 2026-10-29 13:49 IST): residual nonlinear decodability.

Pre-specified probe: MLP, one hidden layer of 256 ReLU units, Adam (lr 1e-3), L2 alpha 1e-3, batch 256, max_iter 200,
early stopping on a 10% random split of the training fold, random_state 0; inputs standardised on the training fold;
age target standardised by training mean/SD; site as a classifier. Fitted on outer-training rows of each arm's
features, evaluated on the outer test fold. Arms (rebuilt exactly from E50/E51 seeds and logged ranks): t = 0,
T-D1(1), T-D2(0.5), matched random subspace 0 at D1(1), sham 0. Diagnostic labels are never used.

Usage: python e52_nonlinear_probe.py <dataset> --concept age|site
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, RESULTS, Experiment
from e40_crossfit_baseline import boot_idx, ci
from e45_erasure import concept, prepare, whitening
from e50_dose_restoration import CODE, DOSES, d1, d2, leace_parts, r2_boot, random_parts, sha

MLP = dict(hidden_layer_sizes=(256,), activation="relu", solver="adam", learning_rate_init=1e-3, alpha=1e-3,
           batch_size=256, max_iter=200, early_stopping=True, validation_fraction=0.1, random_state=0)


def mlp_age(Xk, z, obs, folds):
    p = np.full(len(z), np.nan)
    for k, (tr, te) in enumerate(folds):
        trc, tec = tr[obs[tr]], te[obs[te]]
        sc = StandardScaler().fit(Xk[k][trc])
        m, s = z[trc].mean(), z[trc].std()
        net = MLPRegressor(**MLP).fit(sc.transform(Xk[k][trc]), (z[trc] - m) / s)
        p[tec] = net.predict(sc.transform(Xk[k][tec])) * s + m
    return p


def mlp_site(Xk, lab, folds):
    K = int(lab.max()) + 1
    P, gain, used = np.zeros((len(lab), K)), np.full(len(lab), np.nan), np.zeros(len(lab), bool)
    for k, (tr, te) in enumerate(folds):
        trc, tec = tr[lab[tr] >= 0], te[lab[te] >= 0]
        cls = np.unique(lab[trc])
        tec = tec[np.isin(lab[tec], cls)]
        sc = StandardScaler().fit(Xk[k][trc])
        net = MLPClassifier(**MLP).fit(sc.transform(Xk[k][trc]), lab[trc])
        pr = np.clip(net.predict_proba(sc.transform(Xk[k][tec])), 1e-12, 1)
        prior = np.array([np.mean(lab[trc] == c) for c in cls])
        pos = np.searchsorted(cls, lab[tec])
        gain[tec] = np.log(pr[np.arange(len(tec)), pos]) - np.log(prior[pos])
        P[np.ix_(tec, cls)] = pr
        used[tec] = True
    aucs = [roc_auc_score(lab[used] == c, P[used, c]) for c in range(K) if 5 <= np.sum(lab[used] == c) < used.sum()]
    return gain, used, float(np.mean(aucs))


def run(ds, conc):
    e50 = json.loads((RESULTS / (("E50" if conc == "age" else "E51") + f"_{ds}_{conc}_convnext_tiny_f0"
                                 + ("" if conc == "age" else "_light") + ".json")).read_text())["results"]
    ranks = e50["meta"]["rank_matched"]
    idx, df, y, g, _ = prepare(ds)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, g))
    X = np.load(RESEARCH / f"data/{ds}/emb_convnext_tiny.npy")[idx].astype(np.float64)
    Zc, obs = concept(df, ds, conc)
    wh = [whitening(X[tr]) for tr, _ in folds]
    T = [leace_parts(X[tr[obs[tr]]], Zc[tr[obs[tr]]]) for tr, _ in folds]
    R0 = [random_parts(wh[k], ranks[k], np.random.default_rng(51000 + 100 * k)) for k in range(5)]
    SH = []
    for k, (tr, _) in enumerate(folds):
        trc = tr[obs[tr]]
        SH.append(leace_parts(X[trc], Zc[trc][np.random.default_rng(53000 + 100 * k).permutation(len(trc))]))
    arms = {"t0": [X] * 5, "T-D1-1.0": [d1(X, e, 1.0) for e in T],
            "T-D2-0.5": [d2(X, e, 0.5, tr, np.random.default_rng(54000 + 100 * k + DOSES.index(0.5)))
                         for k, (e, (tr, _)) in enumerate(zip(T, folds))],
            "R-D1-1.0-0": [d1(X, e, 1.0) for e in R0], "sham-0": [d1(X, e, 1.0) for e in SH]}
    inv, bidx = boot_idx(g, np.random.default_rng(52))
    out = {}
    for name, Xk in arms.items():
        if conc == "age":
            z = Zc[:, 0]
            p = mlp_age(Xk, z, obs, folds)
            r2 = float(1 - np.sum((z[obs] - p[obs]) ** 2) / np.sum((z[obs] - z[obs].mean()) ** 2))
            out[name] = {"mlp_r2": round(r2, 4), "mlp_r2_ci95": ci(r2_boot(z, p, obs, inv, bidx))}
        else:
            lab = np.where(obs, Zc.argmax(1), -1)
            gain, used, auc = mlp_site(Xk, lab, folds)
            bs = np.bincount(inv, weights=np.where(used, gain, 0))[bidx].sum(1) / np.bincount(inv, weights=used.astype(float))[bidx].sum(1)
            out[name] = {"mlp_logloss_gain": round(float(np.nanmean(gain[used])), 4), "mlp_logloss_gain_ci95": ci(bs),
                         "mlp_macro_auroc": round(auc, 4)}
        lin = e50["arms"].get(name, {})
        out[name]["linear_probe_from_E50"] = lin.get("probe_age") if conc == "age" else lin.get("probe_site")
        print(ds, conc, name, out[name], flush=True)
    return {"ranks_matched": ranks, "arms": out}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--concept", default="age")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for ds in a.datasets:
        exp = Experiment(f"E52_{ds}_{a.concept}", "iteration6/REGISTRY.md E52: residual nonlinear decodability after "
                         "linear erasure", {"dataset": ds}, {"concept": a.concept, "mlp": MLP, "rung": "convnext_tiny",
                                                             "fold_seed": 0, "code_sha256_12": {p.name: sha(p) for p in
                                                                                                [Path(__file__), *CODE]}}, 0)
        exp.finish(run(ds, a.concept), interpretation="See iteration6 reports.")
