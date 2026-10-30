"""Known-answer tests for the evaluation pipeline."""
import numpy as np

from calibration import brier, calibration_in_the_large, calibration_intercept_slope, ece
from saliency_overlap import overlap_metrics


def _cohort(rng, n=20000):
    true_logit = rng.normal(scale=1.5, size=n) - 0.5
    y = rng.random(n) < 1 / (1 + np.exp(-true_logit))
    return y, true_logit


def _p(logit):
    return 1 / (1 + np.exp(-logit))


def test_perfect_calibration():
    y, L = _cohort(np.random.default_rng(0))
    r = calibration_intercept_slope(y, _p(L))
    assert abs(r["slope"] - 1) < 0.05 and abs(r["intercept"]) < 0.05
    assert ece(y, _p(L)) < 0.02


def test_overconfident_predictions_give_slope_half():
    """Predicted logit = 2 x true logit, so the recalibration slope is exactly 1/2."""
    y, L = _cohort(np.random.default_rng(1))
    r = calibration_intercept_slope(y, _p(2 * L))
    assert abs(r["slope"] - 0.5) < 0.03
    assert ece(y, _p(2 * L)) > ece(y, _p(L))


def test_shifted_predictions_recover_the_shift():
    """Predicted logit = true logit + 0.8, so calibration-in-the-large is -0.8."""
    y, L = _cohort(np.random.default_rng(2))
    assert abs(calibration_in_the_large(y, _p(L + 0.8)) + 0.8) < 0.05


def test_brier_known_value():
    assert np.isclose(brier([0, 1, 1, 0], [0.5, 0.5, 0.5, 0.5]), 0.25)


def test_shift_null_works_in_2d():
    from saliency_overlap import shift_null_pvalue
    rng = np.random.default_rng(0)
    mask = np.zeros((32, 32), bool)
    mask[10:16, 10:16] = True
    assert shift_null_pvalue(mask.astype(float), mask, 99, rng) <= 0.02
    p_noise = shift_null_pvalue(rng.random((32, 32)), mask, 99, rng)
    assert 0 < p_noise <= 1


def test_exact_shift_null_matches_explicit_shifts():
    """FFT cross-correlation must equal mass_ratio computed shift by shift."""
    from saliency_overlap import shift_null_exact
    rng = np.random.default_rng(1)
    s = rng.random((12, 10)) ** 3
    mask = np.zeros((12, 10), bool)
    mask[3:7, 2:5] = True
    obs, p = shift_null_exact(s, mask)
    explicit = np.array([overlap_metrics(s, np.roll(mask, (-i, -j), axis=(0, 1)))["mass_ratio"]
                         for i in range(12) for j in range(10)])
    assert np.isclose(obs, overlap_metrics(s, mask)["mass_ratio"])
    assert np.isclose(p, np.mean(explicit >= obs - 1e-12))


def test_overlap_metrics_known_cases():
    mask = np.zeros((10, 10, 10), bool)
    mask[2:4, 2:4, 2:4] = True  # 8 of 1000 voxels
    inside = mask.astype(float)
    r = overlap_metrics(inside, mask)
    assert r["mass_ratio"] == 1 and r["pointing"] == 1 and r["topk_dice"] == 1
    assert np.isclose(r["enrichment"], 1000 / 8)
    uniform = overlap_metrics(np.ones((10, 10, 10)), mask)
    assert np.isclose(uniform["enrichment"], 1.0)
