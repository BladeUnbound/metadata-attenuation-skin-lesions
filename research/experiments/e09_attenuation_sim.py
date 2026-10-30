"""E09: Does an attenuation curve distinguish mediated from direct auxiliary information?

Synthetic only. Each sample has a latent image content vector Z (40-d) and an
auxiliary variable A (think: patient age). The outcome depends on Z, and A may
act through Z (mediated), directly (direct), or both.
  Z = Z0 + mediation * A * u      (u: fixed unit direction in latent space)
  logit P(y) = -0.5 + w.Z + direct * A
A capacity ladder of image models sees progressively less noisy views of Z:
  view_k = Z + noise_k * eps,  noise_k from 3.0 (weak model) to 0.1 (strong).
Added value of A at each rung: out-of-fold log-loss of [image score] minus
out-of-fold log-loss of [image score, A] (positive = A helps).
Recoverability: out-of-fold R^2 of predicting A from the view.
Predictions (written before running):
  mediated: added value falls toward 0 as capacity rises; recoverability rises
  direct:   added value stays roughly flat
  mixed:    added value falls to a non-zero floor
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import log_loss, r2_score

sys.path.insert(0, str(Path(__file__).parent))
from common import Experiment

SEED, N, D, REPS = 0, 3000, 40, 10
NOISE = [3.0, 2.0, 1.2, 0.7, 0.35, 0.1]
SCENARIOS = {"mediated": (1.2, 0.0), "direct": (0.0, 0.8), "mixed": (1.2, 0.4), "none": (0.0, 0.0)}


def oof_logloss(X, y, seed):
    p = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), X, y,
                          cv=KFold(5, shuffle=True, random_state=seed), method="predict_proba")[:, 1]
    return log_loss(y, p)


def run(scenario, rng, rep):
    mediation, direct = SCENARIOS[scenario]
    u = np.ones(D) / np.sqrt(D)
    w = rng.normal(size=D) / np.sqrt(D) * 2.0
    A = rng.normal(size=N)
    Z = rng.normal(size=(N, D)) + mediation * A[:, None] * u[None, :]
    y = (rng.random(N) < 1 / (1 + np.exp(-(-0.5 + Z @ w + direct * A)))).astype(int)
    rows = []
    for noise in NOISE:
        view = Z + noise * rng.normal(size=Z.shape)
        cv = KFold(5, shuffle=True, random_state=rep)
        # First stage: out-of-fold image score (logit) from the view
        score = cross_val_predict(LogisticRegression(C=1.0, max_iter=2000), view, y, cv=cv,
                                  method="decision_function")[:, None]
        ll_img = oof_logloss(score, y, rep)
        ll_both = oof_logloss(np.column_stack([score, A]), y, rep)
        r2 = r2_score(A, cross_val_predict(LinearRegression(), view, A, cv=cv))
        rows.append({"noise": noise, "image_logloss": ll_img, "added_value": ll_img - ll_both,
                     "recoverability_r2": r2})
    return rows


def main():
    rng = np.random.default_rng(SEED)
    exp = Experiment(
        "E09_attenuation_sim",
        "Attenuation of auxiliary added value with image-model capacity separates mediated from direct "
        "auxiliary information, and tracks recoverability of the auxiliary variable from the image.",
        {"source": "synthetic"}, {"n": N, "latent_dim": D, "reps": REPS, "noise_ladder": NOISE,
                                  "scenarios": SCENARIOS}, SEED)
    out = {}
    for s in SCENARIOS:
        allrows = [run(s, rng, r) for r in range(REPS)]
        agg = []
        for i, noise in enumerate(NOISE):
            av = np.array([rows[i]["added_value"] for rows in allrows])
            agg.append({"noise": noise,
                        "image_logloss": round(float(np.mean([r[i]["image_logloss"] for r in allrows])), 4),
                        "added_value_mean": round(float(av.mean()), 4),
                        "added_value_sd": round(float(av.std(ddof=1)), 4),
                        "recoverability_r2": round(float(np.mean([r[i]["recoverability_r2"] for r in allrows])), 3)})
        out[s] = agg
        print(s, [(a["noise"], a["added_value_mean"], a["recoverability_r2"]) for a in agg], flush=True)
    exp.finish(out, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
