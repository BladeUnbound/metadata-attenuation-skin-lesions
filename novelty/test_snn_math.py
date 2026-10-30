"""Numerical checks for every claim in DERIVATION.md, part A (spiking networks)."""
import pytest
import torch

from snn_math import (
    if_static_count,
    lif_bptt_manual,
    lif_forward,
    surrogate_count_gradient,
)

torch.set_default_dtype(torch.float64)
THETA = 1.0


@pytest.mark.parametrize("beta", [1.0, 0.8])
@pytest.mark.parametrize("detach_reset", [False, True])
def test_step2_manual_bptt_matches_autograd(beta, detach_reset):
    """Step 2: the hand-derived gradient recursion equals PyTorch autograd."""
    gen = torch.Generator().manual_seed(0)
    T, n = 12, 500
    I = (torch.rand(T, n, generator=gen) * 1.6 - 0.3).requires_grad_()
    v0 = (torch.rand(n, generator=gen) * THETA).requires_grad_()
    g = torch.randn(T, n, generator=gen)

    s, _, _ = lif_forward(I, THETA, beta, v0, detach_reset)
    (g * s).sum().backward()

    eps, dv0 = lif_bptt_manual(I.detach(), g, THETA, beta, v0.detach(), detach_reset)
    assert torch.allclose(eps, I.grad, atol=1e-12)
    assert torch.allclose(dv0, v0.grad, atol=1e-12)
    assert I.grad.abs().sum() > 0  # the check is not vacuous


@pytest.mark.parametrize("T", [1, 4, 16, 64])
def test_step4_static_if_count_closed_form(T):
    """Step 4: under constant input, the IF spike count has an exact closed form."""
    gen = torch.Generator().manual_seed(T)
    n = 20000
    I = torch.rand(n, generator=gen) * 3 * THETA - THETA  # spans I < 0, [0, theta), I >= theta
    v0 = torch.rand(n, generator=gen) * THETA
    arg = (v0 + T * I) / THETA
    clear = (arg - arg.round()).abs() > 1e-9  # skip exact ties, where float rounding decides
    s, _, _ = lif_forward(I.expand(T, n), THETA, 1.0, v0)
    assert torch.equal(s.sum(0)[clear], if_static_count(I, THETA, T, v0)[clear])
    assert clear.float().mean() > 0.999


def test_step3_rate_identity_holds_for_any_input():
    """Step 3: theta * N_T = v0 + sum(I_t) - v_T, for arbitrary input sequences."""
    gen = torch.Generator().manual_seed(1)
    T, n = 32, 5000
    I = torch.randn(T, n, generator=gen) * 2.0  # large, negative, non-constant
    v0 = torch.rand(n, generator=gen) * THETA
    s, _, vT = lif_forward(I, THETA, 1.0, v0)
    assert torch.allclose(THETA * s.sum(0), v0 + I.sum(0) - vT, atol=1e-9)


@pytest.mark.parametrize("detach_reset", [False, True])
@pytest.mark.parametrize("T", [4, 16, 64])
def test_step5_surrogate_count_gradient_closed_form(detach_reset, T):
    """Step 5: surrogate BPTT of N_T w.r.t. a constant input has a closed form."""
    gen = torch.Generator().manual_seed(T)
    n = 5000
    I = (torch.rand(n, generator=gen) * 1.4 * THETA - 0.2 * THETA).requires_grad_()
    v0 = torch.full((n,), THETA / 2)
    s, u, _ = lif_forward(I.expand(T, n), THETA, 1.0, v0, detach_reset)
    s.sum().backward()
    expected = surrogate_count_gradient(u.detach(), THETA, detach_reset)
    assert torch.allclose(I.grad, expected, atol=1e-9)


def test_step5_corollary_scaling_with_T():
    """Step 5 corollary: with the reset kept, the gradient never exceeds T/theta.
    With the reset detached it can grow like T^2."""
    T, n = 64, 2000
    I = torch.linspace(0.05, 0.95, n) * THETA
    v0 = torch.full((n,), THETA / 2)
    _, u, _ = lif_forward(I.expand(T, n), THETA, 1.0, v0)
    kept = surrogate_count_gradient(u, THETA, detach_reset=False)
    detached = surrogate_count_gradient(u, THETA, detach_reset=True)
    assert kept.max() <= T / THETA
    assert detached.max() > 10 * T / THETA


@pytest.mark.parametrize("beta", [1.0, 0.9])
def test_step6_threshold_scaling_equals_input_gain(beta):
    """Step 6: scaling a threshold by g > 0 gives exactly the same spikes as
    dividing the input by g, so gene-conditioned thresholds add no expressivity
    beyond multiplicative (FiLM-scale) conditioning."""
    gen = torch.Generator().manual_seed(2)
    T, n = 20, 4000
    I = torch.rand(T, n, generator=gen) * 1.5
    g = 2.0 ** torch.randint(-3, 4, (n,), generator=gen).to(I.dtype)  # powers of 2: exact in floating point
    v0 = torch.rand(n, generator=gen) * THETA
    s_threshold, _, _ = lif_forward(I, THETA * g, beta, v0 * g, width=THETA * g)
    s_gain, _, _ = lif_forward(I / g, THETA, beta, v0)
    assert torch.equal(s_threshold, s_gain)

    g_any = torch.rand(n, generator=gen) * 1.5 + 0.5  # arbitrary gains: equal up to rounding at ties
    s_threshold, _, _ = lif_forward(I, THETA * g_any, beta, v0 * g_any, width=THETA * g_any)
    s_gain, _, _ = lif_forward(I / g_any, THETA, beta, v0)
    assert (s_threshold != s_gain).float().mean() < 1e-4
