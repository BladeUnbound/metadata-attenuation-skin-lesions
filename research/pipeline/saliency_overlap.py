"""Quantitative overlap between a 3D saliency map and an anatomical mask.

Metrics (computed on the positive part of the saliency map):
  mass_ratio   fraction of saliency mass inside the mask
  enrichment   mass_ratio / (mask volume / brain volume); 1 = no better than uniform
  pointing     1 if the peak saliency voxel lies in the mask
  topk_dice    Dice between the mask and the top-|mask| saliency voxels

Significance needs a null that respects spatial smoothness. Shuffling voxels
destroys the smoothness of real saliency maps and makes chance overlap look
significant; shifting the mask rigidly keeps it (the idea behind "spin tests"
for cortical maps, Alexander-Bloch et al. 2018). E05 measures the difference.
"""
import numpy as np


def overlap_metrics(saliency, mask, brain=None):
    s = np.clip(np.asarray(saliency, float), 0, None)
    m = np.asarray(mask, bool)
    brain = np.ones_like(m) if brain is None else np.asarray(brain, bool)
    s = s * brain
    total = s.sum()
    mass = s[m].sum() / total if total > 0 else 0.0
    k = int(m.sum())
    top = np.zeros(s.size, bool)
    top[np.argpartition(s.ravel(), -k)[-k:]] = True
    top = top.reshape(s.shape)
    return {
        "mass_ratio": mass,
        "enrichment": mass / (m.sum() / brain.sum()),
        "pointing": float(m.ravel()[np.argmax(s)]),
        "topk_dice": 2 * (top & m).sum() / (top.sum() + m.sum()),
    }


def shift_null_pvalue(saliency, mask, n_null, rng, metric="mass_ratio"):
    """p-value against random circular shifts of the mask (preserves its shape and
    the map's spatial autocorrelation). Valid on a box; within a brain, restrict
    shifts to placements that keep the mask inside the brain."""
    observed = overlap_metrics(saliency, mask)[metric]
    null = []
    axes = tuple(range(mask.ndim))  # works for 2D images and 3D volumes
    for _ in range(n_null):
        shift = tuple(int(rng.integers(0, n)) for n in mask.shape)
        null.append(overlap_metrics(saliency, np.roll(mask, shift, axis=axes))[metric])
    return (1 + np.sum(np.array(null) >= observed)) / (n_null + 1)


def shift_null_exact(saliency, mask):
    """Exact mask-shift null for mass_ratio, over every circular shift at once.

    mass_inside(shift) = sum_x s(x) m(x - shift) is a circular cross-correlation,
    so one FFT gives it for all prod(shape) shifts. Returns (observed mass_ratio,
    p-value = share of shifts whose mass_ratio >= observed, the identity shift included).
    Cost O(V log V) instead of O(n_null * V).
    """
    s = np.clip(np.asarray(saliency, float), 0, None)
    m = np.asarray(mask, float)
    total = s.sum()
    if total == 0:
        return 0.0, 1.0
    corr = np.real(np.fft.ifftn(np.fft.fftn(s) * np.conj(np.fft.fftn(m)))) / total
    observed = corr.flat[0]
    return float(observed), float(np.mean(corr >= observed - 1e-12))


def voxel_permutation_pvalue(saliency, mask, n_null, rng, metric="mass_ratio"):
    """p-value against voxel shuffling. Included to show why it is invalid for
    smooth maps; do not use it for reported results."""
    observed = overlap_metrics(saliency, mask)[metric]
    flat = np.asarray(saliency, float).ravel()
    null = [overlap_metrics(rng.permutation(flat).reshape(mask.shape), mask)[metric] for _ in range(n_null)]
    return (1 + np.sum(np.array(null) >= observed)) / (n_null + 1)
