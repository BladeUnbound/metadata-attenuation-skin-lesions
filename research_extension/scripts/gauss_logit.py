"""Exact Bayes risks for linear-Gaussian representations with a logistic label (ground truth for P2/P3).

Model: latent F ~ N(0, Sigma_F) in R^m; any representation is a linear view V = K F; the label is
Y ~ Bernoulli(sigmoid(b + w^T F)). For a linear view, the logit given V is Gaussian with
    mean  mu(V) = b + w^T Sigma K^T (K Sigma K^T)^+ V   (Var mu = v_mu)
    var   s2    = w^T Sigma w - v_mu                      (constant in V),
so P(Y=1|V) = E_xi sigmoid(mu + sqrt(s2) xi) and H(Y|V) = E_mu h(P(Y=1|mu)), a 2-D Gauss-Hermite integral.
Bayes risks therefore depend only on (b, v_mu, s2). Everything is in nats.
"""
import numpy as np

_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)  # probabilists' Hermite: weight exp(-x^2/2)
_GH_W = _GH_W / _GH_W.sum()


def _sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def explained(Sigma, w, K):
    """v_mu = Var(E[w^T F | K F]) for the linear view K F."""
    if K is None or K.shape[0] == 0:
        return 0.0
    C = K @ Sigma @ K.T
    c = K @ Sigma @ w
    # symmetric PSD pseudo-inverse by eigendecomposition: views after erasure are rank-deficient by construction
    # (condition numbers ~1e16-1e18), where SVD-based pinv intermittently failed to converge (SYN-2, 2026-10-30)
    lam, V = np.linalg.eigh((C + C.T) / 2)
    keep = lam > lam.max() * 1e-12
    proj = V[:, keep].T @ c
    return float((proj ** 2 / lam[keep]).sum())


def entropy_given(b, v_mu, s2):
    """H(Y | V) for a view with explained logit variance v_mu and residual variance s2."""
    s2 = max(s2, 0.0)
    mu = b + np.sqrt(max(v_mu, 0.0)) * _GH_X                     # outer: mu ~ N(b, v_mu)
    p = (_sig(mu[:, None] + np.sqrt(s2) * _GH_X[None, :]) * _GH_W[None, :]).sum(1)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    h = -(p * np.log(p) + (1 - p) * np.log(1 - p))
    return float((h * _GH_W).sum())


def risk(Sigma, w, b, K):
    """Bayes log-loss R_Y(KF) = H(Y | K F)."""
    tot = float(w @ Sigma @ w)
    v = explained(Sigma, w, K)
    return entropy_given(b, v, tot - v)


def mi_gauss(Sigma, a, K):
    """I(A; K F) for A = a^T F (Gaussian, so leakage is fully linear)."""
    if K is None or K.shape[0] == 0:
        return 0.0
    va = float(a @ Sigma @ a)
    r2 = explained(Sigma, a, K) / va
    return float(-0.5 * np.log(max(1 - r2, 1e-300)))


def three_quantities(Sigma, w, b, KS, KZ, a):
    """Exact (C_Y, M_A, L_YA, eta, leakage) for image view S = KS F, transformed view Z = KZ F and A = a^T F."""
    RS, RZ = risk(Sigma, w, b, KS), risk(Sigma, w, b, KZ)
    RSA, RZA = risk(Sigma, w, b, np.vstack([KS, a])), risk(Sigma, w, b, np.vstack([KZ, a]))
    C, L = RZ - RS, RZA - RSA
    M = (RZ - RZA) - (RS - RSA)
    return {"R_S": RS, "R_Z": RZ, "R_SA": RSA, "R_ZA": RZA, "C_Y": C, "L_YA": L, "M_A": M,
            "eta": (M / C) if C > 1e-12 else float("nan"), "Delta_S": RS - RSA, "Delta_Z": RZ - RZA,
            "I_A_S": mi_gauss(Sigma, a, KS), "I_A_Z": mi_gauss(Sigma, a, KZ)}


def sample(Sigma, w, b, n, rng):
    """Draw (F, Y) from the model."""
    F = rng.multivariate_normal(np.zeros(len(Sigma)), Sigma, size=n, method="cholesky")
    y = (rng.random(n) < _sig(b + F @ w)).astype(int)
    return F, y
