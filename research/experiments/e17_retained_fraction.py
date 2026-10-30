"""E17 (exploratory): comparable summary across datasets.

Standalone metadata information:  I_A  ≈ logloss(prevalence) - logloss(metadata only), out of fold.
Retained fraction at rung k:      rho_k = Delta_k / I_A, with Delta_k from E10/E13 (fold assignment 0).
rho_k is the share of the metadata's standalone information that is still not
redundant with image representation k. CIs: grouped bootstrap of the ratio
of means, using per-image loss differences. Note: Delta_k can exceed I_A
(synergy), so rho_k is not bounded by 1.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import EPS, meta_matrix, oof_scores, second_stage
from e13_replication import adapter, oof_block

RUNGS = ["hsv", "resnet18", "convnext_tiny", "vit_b_16"]
B = 2000


def per_image_ll(y, p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def prevalence_ll(y, folds):
    p = np.zeros(len(y))
    for tr, te in folds:
        p[te] = y[tr].mean()
    return per_image_ll(y, p)


def run(name):
    if name == "ham10000":
        from e10_ham_embed import DATA
        df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
        y = (df["dx"] == "mel").to_numpy(int)
        groups = df["lesion_id"].to_numpy()
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
        M = np.zeros((len(df), meta_matrix(df, folds[0][0]).shape[1]))
        for tr, te in folds:
            M[te] = meta_matrix(df, tr)[te]
        emb = lambda k: np.load(DATA / f"emb_{k}.npy")
    else:
        a = adapter(name)
        df, y, groups = a["df"], a["y"], a["groups"]
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
        M = oof_block(df, *a["blocks"]["all"], folds)
        emb = lambda k: np.load(RESEARCH / f"data/{name}/emb_{k}.npy")
    l_prev = prevalence_ll(y, folds)
    _, l_meta = second_stage(np.zeros((len(y), 1)), M, y, folds)  # metadata-only model
    d_meta = l_prev - l_meta
    codes, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(17)
    idx = rng.integers(0, len(codes), (B, len(codes)))
    cnt = np.bincount(inv)
    def gsum(d):
        return np.bincount(inv, weights=d)
    out = {"I_A_nats": round(float(d_meta.mean()), 5)}
    sm = gsum(d_meta)
    for k in RUNGS:
        score, _ = oof_scores(emb(k), y, groups, folds)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
        d = l0 - l1
        sd = gsum(d)
        ratio_bs = sd[idx].sum(1) / sm[idx].sum(1)
        out[k] = {"delta": round(float(d.mean()), 5), "rho": round(float(d.mean() / d_meta.mean()), 3),
                  "rho_ci95": [round(float(np.percentile(ratio_bs, q)), 3) for q in (2.5, 97.5)]}
    print(name, out, flush=True)
    return out


def main(datasets):
    exp = Experiment("E17_retained_fraction", "The share of metadata information that remains non-redundant with "
                     "strong image models differs between dermoscopy and smartphone/clinical-history data.",
                     {"datasets": datasets}, {"rungs": RUNGS, "bootstrap": B, "folds": "assignment 0"}, 0)
    exp.finish({d: run(d) for d in datasets}, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["ham10000", "isic2020", "pad_ufes"])
