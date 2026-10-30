"""E03: Does the likelihood-ratio test for genetics stay valid when the MRI score
comes from a learned model, and what does a rejection actually mean?

Synthetic cohorts only (code validation, not clinical evidence). Each cohort:
  X     (n, p) MRI-derived features; the first 10 carry atrophy signal
  APOE  e4 allele count ~ Binomial(2, 0.25); PRS ~ N(0, 1)
  atrophy = signal(X) + mediation * APOE    (APOE -> atrophy pathway)
  y ~ Bernoulli(sigmoid(-0.7 + 1.2 * atrophy_std + direct * (0.6*APOE + 0.4*PRS)))
The MRI score is an L2 logistic regression on X, used three ways:
  in_sample   fitted and scored on the same patients (the leakage-prone way)
  out_of_fold 5-fold cross-fitted predictions (the planned way)
  oracle      the true atrophy value (the best any MRI model could do)
Scenarios:
  strict_null    no mediation, no direct effect  -> every valid test rejects ~5%
  mediated_null  APOE raises atrophy, no direct effect
  direct_effect  direct genetic effect -> power
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[2] / "novelty"))
from added_value import lrt_added_predictors
from common import Experiment

SEED, N, P, N_SIMS = 0, 400, 200, 500
SCENARIOS = {"strict_null": (0.0, 0.0), "mediated_null": (0.8, 0.0), "direct_effect": (0.0, 0.6)}


def simulate(rng, mediation, direct):
    X = rng.normal(size=(N, P))
    apoe = rng.binomial(2, 0.25, N)
    prs = rng.normal(size=N)
    atrophy = X[:, :10].sum(1) / np.sqrt(10) + mediation * apoe
    atrophy = (atrophy - atrophy.mean()) / atrophy.std()
    logit = -0.7 + 1.2 * atrophy + direct * (0.6 * apoe + 0.4 * prs)
    y = rng.random(N) < 1 / (1 + np.exp(-logit))
    # Imaging features also reflect APOE-driven atrophy, as real MRI would
    X[:, :10] += (mediation * apoe / np.sqrt(10))[:, None]
    return X, np.column_stack([apoe, prs]), y.astype(int), atrophy


def mri_scores(X, y, seed):
    clf = LogisticRegression(C=0.1, max_iter=2000)
    in_sample = clf.fit(X, y).decision_function(X)
    oof = np.empty(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        oof[te] = LogisticRegression(C=0.1, max_iter=2000).fit(X[tr], y[tr]).decision_function(X[te])
    return {"in_sample": in_sample, "out_of_fold": oof}


def main():
    rng = np.random.default_rng(SEED)
    exp = Experiment(
        "E03_crossfit_lrt",
        "Cross-fitted MRI scores keep the LRT valid; in-sample scores cost power; "
        "APOE effects mediated by atrophy make the LRT reject even with no direct genetic effect.",
        {"source": "synthetic cohorts (no patient data)"},
        {"n": N, "p": P, "n_sims": N_SIMS, "scenarios": SCENARIOS, "mri_model": "L2 logistic, C=0.1"},
        SEED,
    )
    results = {}
    for name, (mediation, direct) in SCENARIOS.items():
        rej = {"in_sample": [], "out_of_fold": [], "oracle": []}
        for i in range(N_SIMS):
            X, G, y, atrophy = simulate(rng, mediation, direct)
            scores = mri_scores(X, y, seed=i)
            scores["oracle"] = atrophy
            for k, s in scores.items():
                rej[k].append(lrt_added_predictors(y, s, G)["p"] < 0.05)
        results[name] = {k: round(float(np.mean(v)), 3) for k, v in rej.items()}
        print(name, results[name], flush=True)
    se = np.sqrt(0.05 * 0.95 / N_SIMS)
    exp.finish(
        {"rejection_rate_at_0.05": results, "mc_se_at_5pct": round(float(se), 4)},
        interpretation="See research/EXPERIMENTS.md for the written interpretation.",
        next_step="Pre-specify the estimand: 'added value over this MRI model', not 'direct genetic effect'.",
    )


if __name__ == "__main__":
    main()
