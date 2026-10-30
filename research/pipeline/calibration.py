"""Calibration metrics for binary risk predictions.

calibration_intercept_slope fits logit P(y=1) = a + b * logit(p) (Cox 1958).
Perfect calibration gives a = 0, b = 1. b < 1 means predictions are too
extreme (overconfident); b > 1 means too timid. The intercept here is
estimated jointly with the slope; calibration-in-the-large (slope fixed at 1)
is reported separately because it answers a different question.
"""
import numpy as np
import statsmodels.api as sm

EPS = 1e-12


def _logit(p):
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def calibration_intercept_slope(y, p):
    lp = _logit(p)
    fit = sm.Logit(np.asarray(y, float), sm.add_constant(lp)).fit(disp=0)
    (a, b), ci = fit.params, fit.conf_int()
    return {"intercept": a, "slope": b, "slope_ci": tuple(ci[1])}


def calibration_in_the_large(y, p):
    """Intercept with the slope fixed at 1 (offset = logit(p))."""
    fit = sm.GLM(np.asarray(y, float), np.ones((len(y), 1)), family=sm.families.Binomial(),
                 offset=_logit(p)).fit()
    return fit.params[0]


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, n_bins=10):
    """Expected calibration error with equal-frequency bins."""
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.array_split(np.argsort(p), n_bins)
    return float(sum(len(b) * abs(y[b].mean() - p[b].mean()) for b in bins) / len(y))
