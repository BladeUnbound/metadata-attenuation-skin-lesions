"""E11 (REAL-1): added risk of erasure when age is missing completely at random at test with probability pi.
cost(pi) = mean(pi * c_i + (1 - pi) * l_i), with c_i, l_i as in real_reanalysis.py; grouped paired bootstrap CI.
Reads saved arrays only. Usage: python real_e11.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_reanalysis import OUT, arrays, boot_idx, boot_mean, load  # noqa: E402

PIS = [0.0, 0.25, 0.5, 0.75, 1.0]
SETS = ["ham10000", "isic2019_nonham", "isic2020", "milk10k_derm", "milk10k_clin"]


def main():
    rows = []
    for ds in SETS:
        f = load(ds, "age_convnext_tiny_f0")
        inv, idx = boot_idx(f["groups"], np.random.default_rng(601))
        L0, L1 = arrays(f, "t0")
        for arm in ["T-D1-1.0", "R-D1-1.0-*"]:
            l0, l1 = arrays(f, arm)
            c, l = l0 - L0, l1 - L1
            for pi in PIS:
                v = pi * c + (1 - pi) * l
                b = boot_mean(v, inv, idx)
                rows.append(dict(set=ds, arm=arm, pi=pi, cost_x1e3=1e3 * v.mean(),
                                 ci_lo_x1e3=1e3 * np.percentile(b, 2.5), ci_hi_x1e3=1e3 * np.percentile(b, 97.5)))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "REAL-1_E11_missing_age.csv", index=False)
    print(df.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
