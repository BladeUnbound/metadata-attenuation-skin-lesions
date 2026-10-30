"""Exact checks of the added-value decomposition on random discrete distributions."""
import numpy as np

from info_identity import cond_mi, decomposition


def random_joint(rng, nx=8, na=3, ny=2):
    p = rng.random((nx, na, ny)) ** 3
    return p / p.sum()


def test_identity_holds_exactly_for_random_distributions_and_coarsenings():
    rng = np.random.default_rng(0)
    for _ in range(200):
        p = random_joint(rng)
        n_s = int(rng.integers(1, 8))
        g = rng.integers(0, n_s, size=p.shape[0])
        lhs, direct, residual = decomposition(p, g, n_s)
        assert np.isclose(lhs, direct + residual, atol=1e-12)
        assert lhs >= -1e-12 and direct >= -1e-12


def test_full_representation_leaves_only_the_direct_term():
    rng = np.random.default_rng(1)
    for _ in range(50):
        p = random_joint(rng)
        g = np.arange(p.shape[0])  # injective: S carries all of X
        lhs, direct, residual = decomposition(p, g, p.shape[0])
        assert np.isclose(residual, 0, atol=1e-12) and np.isclose(lhs, direct, atol=1e-12)


def test_fully_mediated_auxiliary_vanishes_at_full_capacity_but_not_before():
    """A -> X -> Y with no direct path: I(Y;A|X) = 0, yet a coarse S leaves A informative."""
    rng = np.random.default_rng(2)
    p_a = np.array([0.5, 0.5])
    p_x_a = rng.dirichlet(np.ones(8) * 0.3, size=2)        # A changes the image
    p_y_x = rng.random(8)                                    # Y depends on the image only
    p = np.zeros((8, 2, 2))
    for a in range(2):
        for x in range(8):
            p[x, a, 1] = p_a[a] * p_x_a[a, x] * p_y_x[x]
            p[x, a, 0] = p_a[a] * p_x_a[a, x] * (1 - p_y_x[x])
    full = decomposition(p, np.arange(8), 8)
    coarse = decomposition(p, np.zeros(8, int), 1)          # S constant: sees nothing
    assert np.isclose(full[1], 0, atol=1e-12) and np.isclose(full[0], 0, atol=1e-12)
    assert coarse[0] > 1e-4                                   # A adds value to a blind model


def test_cond_mi_matches_hand_computation():
    # Y = A XOR X with uniform bits: I(Y;A) = 0 but I(Y;A|X) = log 2
    p = np.zeros((2, 2, 2))
    for x in range(2):
        for a in range(2):
            p[x, a, x ^ a] = 0.25
    assert np.isclose(cond_mi(p, [2], [1], []), 0, atol=1e-12)
    assert np.isclose(cond_mi(p, [2], [1], [0]), np.log(2), atol=1e-12)
