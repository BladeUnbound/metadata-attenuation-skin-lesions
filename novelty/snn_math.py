"""Spiking-neuron maths for the CV EL project, written to be checked rather than trusted.

Each function backs one numbered step in DERIVATION.md, and test_snn_math.py
checks every step numerically against PyTorch autograd or exact simulation.

Neuron model (discrete time t = 1..T, soft reset):
    u_t = beta * v_{t-1} + I_t          membrane before spiking
    s_t = H(u_t - theta)                 spike (Heaviside)
    v_t = u_t - theta * s_t              membrane after reset-by-subtraction
beta = 1 is the integrate-and-fire (IF) neuron; beta < 1 is leaky (LIF).
"""
import torch


def rect_surrogate(x, width):
    """Rectangular surrogate derivative of H: 1/width on |x| < width/2, else 0."""
    return (x.abs() < width / 2).to(x.dtype) / width


class RectSpike(torch.autograd.Function):
    """Heaviside forward, rectangular surrogate backward."""

    @staticmethod
    def forward(ctx, x, width):
        ctx.save_for_backward(x)
        ctx.width = width
        return (x >= 0).to(x.dtype)

    @staticmethod
    def backward(ctx, grad):
        (x,) = ctx.saved_tensors
        return grad * rect_surrogate(x, ctx.width), None


def lif_forward(I, theta, beta=1.0, v0=None, detach_reset=False, width=None):
    """Simulate one layer of neurons.

    I: input currents, shape (T, ...). Returns spikes (T, ...), pre-reset
    membranes u (T, ...) and the final membrane v_T (...).
    """
    width = theta if width is None else width
    v = torch.zeros_like(I[0]) if v0 is None else v0
    spikes, us = [], []
    for t in range(I.shape[0]):
        u = beta * v + I[t]
        s = RectSpike.apply(u - theta, width)
        v = u - theta * (s.detach() if detach_reset else s)
        spikes.append(s)
        us.append(u)
    return torch.stack(spikes), torch.stack(us), v


def lif_bptt_manual(I, g, theta, beta=1.0, v0=None, detach_reset=False, width=None):
    """Hand-derived backward pass (DERIVATION.md step 2).

    g: dL/ds_t arriving from downstream, shape (T, ...).
    Returns eps_t = dL/du_t = dL/dI_t (T, ...) and dL/dv0 (...).

    Recursion, with h_t = surrogate(u_t - theta) and eps_{T+1} = 0:
        eps_t = h_t * g_t + beta * eps_{t+1} * (1 - theta * h_t)   (reset kept)
        eps_t = h_t * g_t + beta * eps_{t+1}                        (reset detached)
    """
    width = theta if width is None else width
    with torch.no_grad():
        _, u, _ = lif_forward(I, theta, beta, v0, detach_reset, width)
    h = rect_surrogate(u - theta, width)
    eps = torch.zeros_like(I)
    nxt = torch.zeros_like(I[0])
    for t in reversed(range(I.shape[0])):
        keep = 1.0 if detach_reset else 1.0 - theta * h[t]
        eps[t] = h[t] * g[t] + beta * nxt * keep
        nxt = eps[t]
    return eps, beta * eps[0]


def if_static_count(I, theta, T, v0):
    """Closed-form spike count of an IF neuron under constant input (step 4).

    Valid for v0 in [0, theta): N_T = clip(floor((v0 + T*I) / theta), 0, T).
    """
    return torch.clamp(torch.floor((v0 + T * I) / theta), 0, T)


def surrogate_count_gradient(u, theta, detach_reset):
    """Closed form of surrogate-BPTT dN_T/dI for constant input I (step 5).

    Assumes beta = 1 and a rectangular surrogate of width theta, so that
    theta * h_t is exactly 0 or 1. u: pre-reset membranes, shape (T, ...).
      reset kept:     dN_T/dI = t_last / theta   (t_last = last step in window, 0 if none)
      reset detached: dN_T/dI = sum of t over in-window steps / theta
    """
    T = u.shape[0]
    in_window = (u - theta).abs() < theta / 2
    steps = torch.arange(1, T + 1, dtype=u.dtype).reshape(-1, *[1] * (u.dim() - 1))
    if detach_reset:
        return (in_window * steps).sum(0) / theta
    return (in_window * steps).amax(0) / theta
