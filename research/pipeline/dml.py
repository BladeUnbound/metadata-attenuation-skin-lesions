"""Double/debiased machine learning (Chernozhukov et al., Econometrics Journal 2018)
for the partially linear model  y = G @ theta + g(X) + e,  E[e | G, X] = 0.

Here y is progression (0/1), G the genetic predictors and X the MRI features.
The hypothesis H0: theta = 0 is "no direct genetic association once the image
is accounted for". Under H0, E[y | G, X] = g(X) exactly, whatever the link
function, so the test is valid for binary y even though the model is linear.

Procedure (cross-fitted, Neyman-orthogonal "partialling out"):
  1. On K folds, fit l(X) ~ E[y | X] and m(X) ~ E[G | X] on the other folds.
  2. Residualise: ry = y - l(X), rG = G - m(X), out of fold.
  3. theta = (rG' rG)^-1 rG' ry, with a heteroskedasticity-robust covariance.
  4. Wald chi-square test with dim(G) degrees of freedom.
Validity needs both nuisance models to be estimated well enough (product of
their errors o(n^-1/2)); E07 checks whether that holds at realistic n and p.
"""
import numpy as np
from scipy import stats
from sklearn.base import clone
from sklearn.model_selection import KFold


def dml_plr(y, G, X, outcome_model, treatment_model, n_folds=5, seed=0):
    y = np.asarray(y, float)
    G = np.asarray(G, float).reshape(len(y), -1)
    ry, rG = np.empty_like(y), np.empty_like(G)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        lm = clone(outcome_model).fit(X[tr], y[tr])
        ry[te] = y[te] - (lm.predict_proba(X[te])[:, 1] if hasattr(lm, "predict_proba") else lm.predict(X[te]))
        for j in range(G.shape[1]):
            rG[te, j] = G[te, j] - clone(treatment_model).fit(X[tr], G[tr, j]).predict(X[te])
    return dml_from_residuals(ry, rG)


def dml_from_residuals(ry, rG):
    """Final DML stage given out-of-fold residuals (also used with oracle nuisances)."""
    y = ry
    A = rG.T @ rG / len(y)
    theta = np.linalg.solve(A, rG.T @ ry / len(y))
    psi = rG * (ry - rG @ theta)[:, None]
    A_inv = np.linalg.inv(A)
    cov = A_inv @ (psi.T @ psi / len(y)) @ A_inv / len(y)
    wald = float(theta @ np.linalg.solve(cov, theta))
    return {"theta": theta, "se": np.sqrt(np.diag(cov)), "wald": wald,
            "p": float(stats.chi2.sf(wald, rG.shape[1]))}
