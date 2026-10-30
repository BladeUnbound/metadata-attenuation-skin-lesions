"""E46 (exploratory; registered in research/iteration5/REGISTRY.md, 22:03 IST): sensitivity re-analysis of E30
(recoverability vs incremental utility) on results/E30_table.csv.

Variants: (A) E30 as run (rows with I_var CI > 0; retained = delta_var / I_var); (B) all rows, absolute delta_var;
(C) all rows, delta_var / (I_var + c), c = median I_var; (D) filtered rows, absolute delta_var.
For each: Spearman(recov_within_norm, outcome); partial Spearman controlling I_var and image AUROC;
leave-one-dataset-out jackknife of the partial correlation; mixed model (dataset random intercept, variable fixed effects)
on rank-transformed variables.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from common import RESULTS, Experiment
from e30_recoverability_utility import partial_spearman


def analyse(t, outcome):
    t = t.copy()
    t["out"] = outcome
    r = {"n_rows": int(len(t)), "n_datasets": int(t.dataset.nunique()),
         "spearman": round(float(spearmanr(t.recov_within_norm, t.out)[0]), 3),
         "partial": round(float(partial_spearman(t.recov_within_norm, t.out, [t.I_var, t.image_auroc])), 3)}
    jk = {}
    for d in t.dataset.unique():
        s = t[t.dataset != d]
        jk[d] = round(float(partial_spearman(s.recov_within_norm, s.out, [s.I_var, s.image_auroc])), 3)
    r["jackknife_partial"] = jk
    r["jackknife_sign_stable"] = bool(all(np.sign(v) == np.sign(r["partial"]) for v in jk.values()))
    m = t.assign(**{c: rankdata(t[c]) / len(t) for c in ["out", "recov_within_norm", "I_var", "image_auroc"]})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smf.mixedlm("out ~ recov_within_norm + I_var + image_auroc + C(variable)", m, groups=m["dataset"]).fit()
    r["mixed_coef_recov"] = round(float(fit.params["recov_within_norm"]), 3)
    r["mixed_ci95"] = [round(float(v), 3) for v in fit.conf_int().loc["recov_within_norm"]]
    r["mixed_p"] = float(fit.pvalues["recov_within_norm"])
    return r


def main():
    t = pd.read_csv(RESULTS / "E30_table.csv")
    t = t[t.recov_within_norm.notna()].reset_index(drop=True)
    f = t[t.I_var_lo > 0].reset_index(drop=True)
    c = float(t.I_var.median())
    res = {"A_registered_as_run": analyse(f, f.delta_var / f.I_var),
           "B_all_rows_absolute": analyse(t, t.delta_var),
           "C_all_rows_shrunk_ratio": analyse(t, t.delta_var / (t.I_var.clip(lower=0) + c)),
           "D_filtered_absolute": analyse(f, f.delta_var),
           "c_median_I_var": c}
    for k, v in res.items():
        print(k, v, flush=True)
    return res


if __name__ == "__main__":
    exp = Experiment("E46_recov_reanalysis", "REGISTRY.md E46 (exploratory): is E30's negative association robust to the "
                     "unregistered filter, the ratio outcome and dataset clustering?", {"source": "results/E30_table.csv"}, {}, 0)
    exp.finish(main(), interpretation="See research/EXPERIMENTS.md.")
