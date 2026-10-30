"""Exact information-theoretic quantities for the added-value decomposition.

For a discrete joint p(x, a, y) and a deterministic representation s = g(x):
  Delta(S) = I(Y; A | S)                       (Bayes log-loss reduction from adding A to S)
           = I(Y; A | X) + [ I(Y; X | S) - I(Y; X | S, A) ]
The bracket is zero when g is injective (S carries all of X).
"""
import numpy as np


def _H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def cond_mi(joint, a_axes, b_axes, c_axes):
    """I(A; B | C) for a joint array over named axes (all axes not listed are marginalised)."""
    keep = sorted(set(a_axes) | set(b_axes) | set(c_axes))
    drop = tuple(i for i in range(joint.ndim) if i not in keep)
    p = joint.sum(axis=drop) if drop else joint
    pos = {ax: i for i, ax in enumerate(keep)}

    def H_of(axes):
        idx = tuple(i for ax, i in pos.items() if ax not in axes)
        return _H((p.sum(axis=idx) if idx else p).ravel())

    A, B, C = set(a_axes), set(b_axes), set(c_axes)
    return H_of(A | C) + H_of(B | C) - H_of(A | B | C) - H_of(C)


def with_representation(p_xay, g, n_s):
    """Joint over (x, a, y, s) with s = g(x) deterministic."""
    nx, na, ny = p_xay.shape
    out = np.zeros((nx, na, ny, n_s))
    for x in range(nx):
        out[x, :, :, g[x]] = p_xay[x]
    return out  # axes: 0=x, 1=a, 2=y, 3=s


def decomposition(p_xay, g, n_s):
    j = with_representation(p_xay, g, n_s)
    X, A, Y, S = 0, 1, 2, 3
    lhs = cond_mi(j, [Y], [A], [S])
    direct = cond_mi(j, [Y], [A], [X])
    residual = cond_mi(j, [Y], [X], [S]) - cond_mi(j, [Y], [X], [S, A])
    return lhs, direct, residual
