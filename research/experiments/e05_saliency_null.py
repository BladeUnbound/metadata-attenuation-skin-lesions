"""E05: Which null distribution gives valid p-values for saliency-mask overlap?

Synthetic 3D volumes only. Saliency maps are smoothed Gaussian noise that is
independent of the mask by construction, so any test should reject ~5%.
Compared nulls: voxel permutation vs rigid circular shift of the mask.
A positive control adds a true signal inside the mask to check power.
"""
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))
from common import Experiment
from saliency_overlap import overlap_metrics, shift_null_pvalue, voxel_permutation_pvalue

SEED, SHAPE, N_SIMS, N_NULL, SIGMA = 0, (40, 48, 40), 200, 199, 4.0


def hippocampus_like_mask(shape):
    z, y, x = np.indices(shape)
    c = np.array(shape) * np.array([0.45, 0.55, 0.35])
    r = np.array([5, 9, 4])  # elongated ellipsoid, about 750 voxels
    return (((z - c[0]) / r[0]) ** 2 + ((y - c[1]) / r[1]) ** 2 + ((x - c[2]) / r[2]) ** 2) <= 1


def smooth_noise(rng):
    s = gaussian_filter(rng.normal(size=SHAPE), SIGMA, mode="wrap")
    return (s - s.mean()) / s.std()


def main():
    rng = np.random.default_rng(SEED)
    mask = hippocampus_like_mask(SHAPE)
    exp = Experiment(
        "E05_saliency_null",
        "Voxel-permutation p-values are invalid for smooth saliency maps; mask-shift p-values are valid.",
        {"source": "synthetic 3D volumes"},
        {"shape": SHAPE, "smoothing_sigma_vox": SIGMA, "n_sims": N_SIMS, "n_null": N_NULL,
         "mask_voxels": int(mask.sum())},
        SEED,
    )
    out = {}
    for condition, signal in (("null", 0.0), ("signal_in_mask", 0.6)):
        p_perm, p_shift, enrich = [], [], []
        for _ in range(N_SIMS):
            s = smooth_noise(rng) + signal * gaussian_filter(mask.astype(float), 1.5)
            p_perm.append(voxel_permutation_pvalue(s, mask, N_NULL, rng))
            p_shift.append(shift_null_pvalue(s, mask, N_NULL, rng))
            enrich.append(overlap_metrics(s, mask)["enrichment"])
        out[condition] = {
            "voxel_permutation_rejection": float(np.mean(np.array(p_perm) <= 0.05)),
            "mask_shift_rejection": float(np.mean(np.array(p_shift) <= 0.05)),
            "median_enrichment": float(np.median(enrich)),
        }
        print(condition, out[condition], flush=True)
    exp.finish(out, interpretation="See research/EXPERIMENTS.md.",
               next_step="Use mask-shift (or brain-constrained placement) nulls for any reported overlap p-value.")


if __name__ == "__main__":
    main()
