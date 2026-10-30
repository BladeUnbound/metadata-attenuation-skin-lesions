"""E34 (exploratory; request E24): can cheap indicators predict whether collecting a
metadata variable is worth it? Runs on E30's table (results/E30_table.csv).
Registered in research/iteration4/REGISTRY_E30_E34.md before implementation.

Target: delta_var (added value of the variable over the image score, nats).
Features: image_auroc, recov_within_norm, recov_marginal_norm, I_var, missing_rate,
source-shift indicator (1 if the dataset has a source field the image can recover).
Leave-one-dataset-out (MILK10k derm and clin held out together: they share lesions).
Models: B0 predict 0; B1 predict I_var * median retained fraction of training datasets;
ridge on standardised features with the penalty chosen by inner leave-one-dataset-out.
Decision rule, fixed before results: "collect" if predicted delta_var >= 1e-3 nats;
scored against the exhaustive evaluation (true delta_var >= 1e-3).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment

FEATS = ["image_auroc", "recov_within_norm", "recov_marginal_norm", "I_var", "missing_rate", "source_shift"]
THRESH = 1e-3
ALPHAS = np.logspace(-3, 3, 13)


def cluster(d):
    return "milk10k" if d.startswith("milk10k") else d


def fit_ridge(tr, alpha):
    mu, sd = tr[FEATS].mean(), tr[FEATS].std().replace(0, 1)
    m = Ridge(alpha=alpha).fit((tr[FEATS] - mu) / sd, tr.delta_var)
    return lambda df: m.predict((df[FEATS] - mu) / sd)


def inner_alpha(tr):
    best, best_err = ALPHAS[0], np.inf
    cl = tr.cl.unique()
    if len(cl) < 3:
        return 1.0
    for a in ALPHAS:
        errs = [np.abs(fit_ridge(tr[tr.cl != c], a)(tr[tr.cl == c]) - tr[tr.cl == c].delta_var).mean() for c in cl]
        if np.mean(errs) < best_err:
            best, best_err = a, np.mean(errs)
    return best


def decision(y_true, y_pred):
    t, p = y_true >= THRESH, y_pred >= THRESH
    tp, fp, fn = int((t & p).sum()), int((~t & p).sum()), int((t & ~p).sum())
    return {"precision": tp / (tp + fp) if tp + fp else None, "recall": tp / (tp + fn) if tp + fn else None,
            "accuracy": float((t == p).mean()), "n_collect_true": int(t.sum()), "n_collect_pred": int(p.sum())}


def main():
    tab = pd.read_csv(RESEARCH / "results/E30_table.csv")
    tab["source_shift"] = tab.dataset.isin(["ham10000", "isic2019_nonham"]).astype(float)
    tab = tab.dropna(subset=FEATS + ["delta_var"]).copy()
    tab["cl"] = tab.dataset.map(cluster)
    exp = Experiment("E34_collection_rule", "Exploratory: a transparent rule predicts delta_var better than baselines "
                     "under leave-one-dataset-out evaluation.", {"source": "results/E30_table.csv",
                                                                 "clusters": sorted(tab.cl.unique())},
                     {"features": FEATS, "threshold_nats": THRESH, "alphas": ALPHAS.tolist()}, 0)
    preds = []
    for c in tab.cl.unique():
        tr, te = tab[tab.cl != c], tab[tab.cl == c]
        if tr.cl.nunique() < 2:
            continue
        ret = (tr.delta_var / tr.I_var.where(tr.I_var > 0)).groupby(tr.cl).median().median()
        alpha = inner_alpha(tr)
        te = te.assign(pred_b0=0.0, pred_b1=te.I_var * ret, pred_ridge=fit_ridge(tr, alpha)(te), alpha=alpha)
        preds.append(te)
    if not preds:
        exp.finish({"note": "fewer than 2 dataset clusters; rule cannot be evaluated"}, interpretation="blocked")
        return
    P = pd.concat(preds)
    P.to_csv(RESEARCH / "results/E34_predictions.csv", index=False)
    res = {"n_rows": int(len(P)), "n_clusters": int(P.cl.nunique()),
           "limitation": "few independent datasets; generalisation evidence is weak by construction"}
    for m in ("pred_b0", "pred_b1", "pred_ridge"):
        res[m] = {"mae": float(np.abs(P[m] - P.delta_var).mean()),
                  "spearman": float(spearmanr(P[m], P.delta_var)[0]) if P[m].nunique() > 1 else None,
                  "decision": decision(P.delta_var.to_numpy(), P[m].to_numpy())}
    worst = P.assign(err=np.abs(P.pred_ridge - P.delta_var)).nlargest(5, "err")
    res["largest_ridge_errors"] = worst[["dataset", "rung", "variable", "delta_var", "pred_ridge"]].round(5).to_dict("records")
    print(res, flush=True)
    exp.finish(res, interpretation="See research/iteration4/SUMMARY.md.")


if __name__ == "__main__":
    main()
