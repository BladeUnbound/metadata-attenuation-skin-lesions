"""E07: Can double machine learning separate a direct genetic effect from
genetics that merely predicts atrophy the MRI model missed?

Same synthetic scenarios as E03. Target: under mediated_null (APOE acts only
through atrophy) a test of "direct effect" should reject ~5%, while the E03
added-value LRT rejected 85.6%. Under direct_effect it should keep power.
Also runs at n = 1600 to see whether any size distortion shrinks with n,
as DML theory predicts when nuisance errors shrink.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))
import e03_crossfit_lrt as e03
from common import Experiment
from dml import dml_plr

SEED, N_SIMS = 1, 300


def run(n, scenario, rng):
    e03.N = n
    mediation, direct = e03.SCENARIOS[scenario]
    rej = []
    for i in range(N_SIMS):
        X, G, y, _ = e03.simulate(rng, mediation, direct)
        r = dml_plr(y, G, X, LogisticRegression(C=0.1, max_iter=2000), Ridge(alpha=50.0), seed=i)
        rej.append(r["p"] < 0.05)
    return round(float(np.mean(rej)), 3)


def main():
    rng = np.random.default_rng(SEED)
    exp = Experiment(
        "E07_dml_direct_effect",
        "DML partialling-out keeps the false-positive rate near 5% when APOE acts only through atrophy, "
        "unlike the added-value LRT, while retaining power for a direct effect.",
        {"source": "synthetic cohorts (no patient data), generator from E03"},
        {"n_sims": N_SIMS, "p": e03.P, "outcome_model": "L2 logistic C=0.1",
         "treatment_model": "ridge alpha=50", "folds": 5},
        SEED,
    )
    results = {}
    for n in (400, 1600):
        results[f"n{n}"] = {s: run(n, s, rng) for s in e03.SCENARIOS}
        print(n, results[f"n{n}"], flush=True)
    exp.finish({"rejection_rate_at_0.05": results,
                "e03_lrt_out_of_fold_n400_for_comparison": {"strict_null": 0.056, "mediated_null": 0.856,
                                                            "direct_effect": 0.586}},
               interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
