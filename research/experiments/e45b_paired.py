"""E45b (exploratory; registered in research/iteration5/REGISTRY.md, 2026-10-29 09:39 IST): paired inference for the
age-erasure result of E45. Arms: t = 0; LEACE erasure of age at t = 1; same-rank random erasure (5 subspaces);
performance-matched random erasure at the rank E45 selected (5 subspaces). Per-observation Δ_age is saved per arm and
paired differences are bootstrapped over grouping units.

Usage: python e45b_paired.py <dataset> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, RESULTS, Experiment
from e10_ham_attenuation import per_image_ll
from e40_crossfit_baseline import boot_idx, boot_mean, ci
from e45_erasure import concept, corrected_foldwise, leace, prepare, random_eraser

RUNGS = ["convnext_tiny", "resnet18"]
N_RAND = 5


def arm(Xk, y, g, folds, blocks):
    _, q0, q1 = corrected_foldwise(Xk, y, g, folds, {"age": blocks["age"]})
    l0 = per_image_ll(y, q0)
    return l0 - per_image_ll(y, q1["age"]), float(l0.mean())


def run(ds):
    idx, df, y, g, blocks = prepare(ds)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, g))
    inv, bidx = boot_idx(g, np.random.default_rng(451))
    e45 = json.loads((RESULTS / f"E45_{ds}.json").read_text())["results"]
    Zc, obs = concept(df, ds, "age")
    out_dir = RESEARCH / f"data/{ds}/e45b"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {}
    for rung in RUNGS:
        X = np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy")[idx].astype(np.float64)
        er = [leace(X[tr[obs[tr]]], Zc[tr[obs[tr]]]) for tr, _ in folds]
        d0, ll0 = arm([X] * 5, y, g, folds, blocks)
        d1, ll1 = arm([X - (X - mu) @ M for mu, M, _ in er], y, g, folds, blocks)
        rank, mrank = er[0][2], e45[rung]["age"]["random_matched"]["rank"]
        same, matched, ll_same, ll_matched = [], [], [], []
        for r in range(N_RAND):
            rng = np.random.default_rng(45100 + r)
            d, ll = arm([X - (X - mu) @ random_eraser(X[tr], rank, rng)[1] for (mu, _, _), (tr, _) in zip(er, folds)], y, g, folds, blocks)
            same.append(d); ll_same.append(ll)
            d, ll = arm([X - (X - mu) @ random_eraser(X[tr], mrank, rng)[1] for (mu, _, _), (tr, _) in zip(er, folds)], y, g, folds, blocks)
            matched.append(d); ll_matched.append(ll)
        same_m, matched_m = np.mean(same, 0), np.mean(matched, 0)
        np.savez_compressed(out_dir / f"{rung}.npz", d0=d0, d1=d1, same=np.array(same), matched=np.array(matched))
        f = lambda v: round(1e3 * float(v), 3)
        res[rung] = {
            "rank_concept": int(rank), "rank_matched": int(mrank),
            "image_logloss": {"t0": round(ll0, 5), "concept": round(ll1, 5), "same_rank": [round(v, 5) for v in ll_same],
                              "matched": [round(v, 5) for v in ll_matched]},
            "delta_age_x1e3": {"t0": f(d0.mean()), "concept": f(d1.mean()), "same_rank_each": [f(d.mean()) for d in same],
                               "matched_each": [f(d.mean()) for d in matched]},
            "concept_minus_matched": {"diff": f((d1 - matched_m).mean()), "ci95": [f(v) for v in ci(boot_mean(d1 - matched_m, inv, bidx))]},
            "concept_minus_same_rank": {"diff": f((d1 - same_m).mean()), "ci95": [f(v) for v in ci(boot_mean(d1 - same_m, inv, bidx))]},
            "concept_minus_t0": {"diff": f((d1 - d0).mean()), "ci95": [f(v) for v in ci(boot_mean(d1 - d0, inv, bidx))]},
        }
        res[rung]["claim_holds"] = bool(res[rung]["concept_minus_matched"]["ci95"][0] > 0)
        print(ds, rung, res[rung], flush=True)
    return res


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        exp = Experiment(f"E45b_{ds}", "REGISTRY.md E45b: paired inference, age erasure vs matched random erasure",
                         {"dataset": ds}, {"rungs": RUNGS, "n_random": N_RAND, "B": 2000}, 0)
        exp.finish(run(ds), interpretation="See research/EXPERIMENTS.md.")
