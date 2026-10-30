"""E07b: Why did DML over-reject in E07 (20.7% at n=400 under the strict null)?

Hypothesis: the final-stage maths is correct, and the over-rejection comes from
poorly estimated nuisance models (ridge with a fixed penalty fitting noise in
200 features), which breaks DML's rate condition.

Check 1 (implementation): oracle nuisances under the strict null, where
  E[G | X] = (0.5, 0) and E[y | X] = sigmoid(-0.7 + 1.2 * atrophy).
  If the code is right, rejection should be ~5%.
Check 2 (estimation): cross-validated penalties (RidgeCV, LogisticRegressionCV)
  in all three scenarios at n = 400 and 1600.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegressionCV, RidgeCV

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))
import e03_crossfit_lrt as e03
from common import Experiment
from dml import dml_from_residuals, dml_plr

SEED, N_SIMS = 2, 300


def oracle(n, rng):
    e03.N = n
    rej = []
    for _ in range(N_SIMS):
        X, G, y, atrophy = e03.simulate(rng, 0.0, 0.0)
        ly = 1 / (1 + np.exp(-(-0.7 + 1.2 * atrophy)))
        rG = G - np.array([0.5, 0.0])
        rej.append(dml_from_residuals(y - ly, rG)["p"] < 0.05)
    return round(float(np.mean(rej)), 3)


def tuned(n, scenario, rng):
    e03.N = n
    mediation, direct = e03.SCENARIOS[scenario]
    rej = []
    for i in range(N_SIMS):
        X, G, y, _ = e03.simulate(rng, mediation, direct)
        r = dml_plr(y, G, X,
                    LogisticRegressionCV(Cs=np.logspace(-3, 1, 9), cv=3, max_iter=2000),
                    RidgeCV(alphas=np.logspace(-1, 5, 13)), seed=i)
        rej.append(r["p"] < 0.05)
    return round(float(np.mean(rej)), 3)


def main():
    rng = np.random.default_rng(SEED)
    exp = Experiment(
        "E07b_dml_diagnosis",
        "DML's final stage is correct (oracle ~5%); E07's over-rejection is a nuisance-estimation "
        "problem that tuning reduces.",
        {"source": "synthetic cohorts (no patient data), generator from E03"},
        {"n_sims": N_SIMS, "p": e03.P, "outcome_model": "LogisticRegressionCV", "treatment_model": "RidgeCV"},
        SEED,
    )
    res = {"oracle_strict_null": {f"n{n}": oracle(n, rng) for n in (400, 1600)}}
    print(res, flush=True)
    res["tuned"] = {}
    for n in (400, 1600):
        res["tuned"][f"n{n}"] = {s: tuned(n, s, rng) for s in e03.SCENARIOS}
        print(n, res["tuned"][f"n{n}"], flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
