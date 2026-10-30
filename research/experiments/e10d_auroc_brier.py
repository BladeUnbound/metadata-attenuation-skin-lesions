"""E10d: metadata added value in clinically familiar units (closes review item M4).

HAM10000, E10 seed-0 lesion-grouped folds. For each rung: out-of-fold
probabilities from [score] and [score + metadata]; ΔAUROC and ΔBrier with
2,000 lesion-bootstrap resamples (AUROC recomputed on each resample).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import Experiment
from e10_ham_attenuation import DATA, RUNGS, meta_matrix, oof_scores

B = 2000


def oof_probs(Z, y, folds):
    P = np.zeros(len(y))
    for tr, te in folds:
        sc = StandardScaler().fit(Z[tr])
        P[te] = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Z[tr]), y[tr]).predict_proba(sc.transform(Z[te]))[:, 1]
    return P


def main():
    df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
    y = (df["dx"] == "mel").to_numpy(int)
    groups = df["lesion_id"].to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    M = np.zeros((len(df), meta_matrix(df, folds[0][0]).shape[1]))
    for tr, te in folds:
        M[te] = meta_matrix(df, tr)[te]
    codes, inv = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inv == c) for c in range(len(codes))]
    rng = np.random.default_rng(10)
    boots = [np.concatenate([members[c] for c in rng.integers(0, len(codes), len(codes))]) for _ in range(B)]
    exp = Experiment("E10d_auroc_brier", "Metadata ΔAUROC and ΔBrier attenuate along the ladder like Δ log-loss.",
                     {"source": "HAM10000"}, {"folds": "E10 seed 0", "bootstrap": B}, 0)
    res = {}
    for rung in RUNGS:
        score, _ = oof_scores(np.load(DATA / f"emb_{rung}.npy"), y, groups, folds)
        p0, p1 = oof_probs(score[:, None], y, folds), oof_probs(np.column_stack([score, M]), y, folds)
        dauc = roc_auc_score(y, p1) - roc_auc_score(y, p0)
        dbrier = np.mean((p0 - y) ** 2) - np.mean((p1 - y) ** 2)
        bs_auc = [roc_auc_score(y[b], p1[b]) - roc_auc_score(y[b], p0[b]) for b in boots]
        bs_br = [np.mean((p0[b] - y[b]) ** 2) - np.mean((p1[b] - y[b]) ** 2) for b in boots]
        res[rung] = {"auroc_image": round(float(roc_auc_score(y, p0)), 4), "auroc_with_meta": round(float(roc_auc_score(y, p1)), 4),
                     "delta_auroc": round(float(dauc), 4),
                     "delta_auroc_ci95": [round(float(np.percentile(bs_auc, q)), 4) for q in (2.5, 97.5)],
                     "delta_brier": round(float(dbrier), 5),
                     "delta_brier_ci95": [round(float(np.percentile(bs_br, q)), 5) for q in (2.5, 97.5)]}
        print(rung, res[rung], flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
