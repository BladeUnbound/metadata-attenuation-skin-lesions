"""Numerical checks for every claim in DERIVATION.md, part B (statistics).

The simulations check error rates directly: under a true null, a valid 5%
test should reject in about 5% of datasets.
"""
import numpy as np
import pytest
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

from added_value import delong_paired, holm, interaction_test, lrt_added_predictors

N_SIMS = 1000
# 95% binomial band around 0.05 for 1000 simulations is about [0.037, 0.064]
NOMINAL_BAND = (0.035, 0.066)


def simulate_cohort(rng, n, genetics_effect=0.0):
    """Synthetic MCI cohort: latent MRI atrophy, APOE e4 copies, a PRS and progression."""
    atrophy = rng.normal(size=n)
    apoe = rng.binomial(2, 0.25, size=n)
    prs = rng.normal(size=n)
    logit = -0.7 + 1.2 * atrophy + genetics_effect * (0.6 * apoe + 0.4 * prs)
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    mri_score = atrophy + rng.normal(scale=0.8, size=n)  # noisy model output
    return y, mri_score, np.column_stack([apoe, prs])


def test_auc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    y = rng.random(300) < 0.4
    a = np.round(rng.normal(size=300) + y, 1)  # rounding forces ties
    b = np.round(rng.normal(size=300) + 0.5 * y, 1)
    r = delong_paired(y, a, b)
    assert np.isclose(r["auc_a"], roc_auc_score(y, a))
    assert np.isclose(r["auc_b"], roc_auc_score(y, b))


def test_delong_variance_matches_bootstrap():
    rng = np.random.default_rng(1)
    n = 400
    y = rng.random(n) < 0.35
    latent = rng.normal(size=n) + y
    a = latent + rng.normal(scale=0.5, size=n)
    b = latent + rng.normal(scale=1.0, size=n)
    se = delong_paired(y, a, b)["se"]
    diffs = []
    for _ in range(2000):
        i = rng.integers(0, n, n)
        diffs.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
    assert abs(np.std(diffs) / se - 1) < 0.1


def test_delong_type1_error_for_non_nested_scores():
    """Two different models with equal true AUC: DeLong should reject about 5%."""
    rng = np.random.default_rng(2)
    rejections = 0
    for _ in range(N_SIMS):
        n = 300
        y = rng.random(n) < 0.35
        latent = rng.normal(size=n) + y
        a = latent + rng.normal(size=n)
        b = latent + rng.normal(size=n)
        rejections += delong_paired(y, a, b)["p"] < 0.05
    assert NOMINAL_BAND[0] <= rejections / N_SIMS <= NOMINAL_BAND[1]


@pytest.fixture(scope="module")
def null_genetics_sims():
    """Datasets where genetics has no effect given MRI: the null of interest."""
    rng = np.random.default_rng(3)
    lrt_p, delong_p = [], []
    for _ in range(N_SIMS):
        y, mri, gen = simulate_cohort(rng, n=400, genetics_effect=0.0)
        r = lrt_added_predictors(y, mri, gen)
        lrt_p.append(r["p"])
        delong_p.append(delong_paired(y, r["pred_full"], r["pred_base"])["p"])
    return np.array(lrt_p), np.array(delong_p)


def test_lrt_type1_error(null_genetics_sims):
    lrt_p, _ = null_genetics_sims
    assert NOMINAL_BAND[0] <= (lrt_p < 0.05).mean() <= NOMINAL_BAND[1]


def test_delong_is_invalid_for_nested_models(null_genetics_sims):
    """Demler et al. (2012): DeLong on nested models rejects far less than 5%."""
    _, delong_p = null_genetics_sims
    assert (delong_p < 0.05).mean() < 0.02


def test_lrt_power_when_genetics_matters():
    rng = np.random.default_rng(4)
    hits = [
        lrt_added_predictors(*simulate_cohort(rng, n=400, genetics_effect=1.0))["p"] < 0.05
        for _ in range(200)
    ]
    assert np.mean(hits) > 0.8


def test_interaction_type1_error():
    """No true APOE x MRI interaction: the Wald test should reject about 5%."""
    rng = np.random.default_rng(5)
    rejections = 0
    for _ in range(N_SIMS):
        y, mri, gen = simulate_cohort(rng, n=400, genetics_effect=1.0)
        carrier = (gen[:, 0] > 0).astype(float)
        rejections += interaction_test(y, mri, carrier)["p"] < 0.05
    assert NOMINAL_BAND[0] <= rejections / N_SIMS <= NOMINAL_BAND[1]


def test_holm_matches_statsmodels():
    rng = np.random.default_rng(6)
    for _ in range(50):
        p = rng.random(rng.integers(2, 12)) ** 2
        assert np.allclose(holm(p), multipletests(p, method="holm")[1])
