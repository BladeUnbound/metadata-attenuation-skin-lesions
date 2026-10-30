"""E30 (exploratory; request E20): recoverability of metadata vs its incremental utility.
Registered in research/iteration4/REGISTRY_E30_E34.md before implementation.

Per dataset (complete cases, seed-0 grouped folds) x representation x variable:
  I_var        metadata-only utility (log-loss reduction over prevalence), with CI
  delta_var    added value of the variable over the image score, with CI
  delta_all    added value of the full block
  recov        marginal and within-class recoverability (AUROC / R^2) and Gini-normalised versions
  image_auroc, missing rate before filtering, source recoverability (HAM10000 only)
Then: Spearman and partial correlations between within-class recoverability and the
retained fraction delta_var / I_var (rows where I_var's CI excludes 0), with a
dataset-clustered bootstrap, and leave-one-dataset-out ridge prediction of retained fraction.
Outputs: results/E30_table.csv, results/E30_recoverability_utility.json, results/E30_scatter.pdf
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from x_common import (DATASETS, RUNGS, VARS, available, block, cached_score, ci, emb, folds_for, group_boot_mean,
                      normalise_recov, per_image_ll, prevalence_ll, recoverability, second_stage, subset)

B = 1000


def dataset_rows(ds, rng):
    s = subset(ds, complete_case=True)
    folds = folds_for(s)
    y, g = s["y"], s["g"]
    prev = prevalence_ll(y, folds)
    M_all = block(s, VARS[ds], folds)
    var_blocks, I = {}, {}
    for name, kind, col in VARS[ds]:
        var_blocks[name] = block(s, [(name, kind, col)], folds)
        _, l1 = second_stage(np.zeros((len(y), 1)), var_blocks[name], y, folds)
        dd = prev - l1
        I[name] = (float(dd.mean()), ci(group_boot_mean(dd, g, rng, B)))
    rows = []
    for rung in RUNGS:
        score = cached_score(ds, s, rung, "cc", folds)
        if score is None:
            continue
        X = emb(s, rung)
        auroc = float(roc_auc_score(y, score))
        a0, a1 = second_stage(score[:, None], np.column_stack([score, M_all]), y, folds)
        d_all = float((a0 - a1).mean())
        for name, kind, col in VARS[ds]:
            l0, l1 = second_stage(score[:, None], np.column_stack([score, var_blocks[name]]), y, folds)
            dv = l0 - l1
            rec = recoverability(X, s, col, kind) or {"marginal": None, "within": None}
            rows.append(dict(dataset=ds, rung=rung, variable=name, kind=kind, n=len(y), image_auroc=auroc,
                             delta_all=d_all, I_var=I[name][0], I_var_lo=I[name][1][0], I_var_hi=I[name][1][1],
                             delta_var=float(dv.mean()), **dict(zip(["delta_var_lo", "delta_var_hi"],
                                                                     ci(group_boot_mean(dv, g, rng, B)))),
                             recov_marginal=rec["marginal"], recov_within=rec["within"],
                             recov_within_norm=normalise_recov(rec["within"], kind),
                             recov_marginal_norm=normalise_recov(rec["marginal"], kind),
                             missing_rate=s["miss_rate"][name]))
        print(ds, rung, f"AUROC {auroc:.3f}", f"delta_all {1000 * d_all:.2f}e-3", flush=True)
    return rows


def partial_spearman(a, b, controls):
    """Spearman correlation of a and b after removing linear effects of ranked controls."""
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    C = np.column_stack([np.ones(len(a))] + [pd.Series(c).rank().to_numpy() for c in controls])
    res = lambda v: v - C @ np.linalg.lstsq(C, v, rcond=None)[0]
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def analyse(tab, rng):
    t = tab[(tab.I_var_lo > 0) & tab.recov_within_norm.notna()].copy()
    t["retained"] = t.delta_var / t.I_var
    out = {"n_rows_used": int(len(t)), "n_rows_total": int(len(tab)),
           "datasets_used": sorted(t.dataset.unique().tolist())}
    if len(t) < 8:
        out["note"] = "too few rows with informative variables"
        return out, t
    out["spearman_recov_within_vs_retained"] = float(spearmanr(t.recov_within_norm, t.retained)[0])
    out["spearman_recov_marginal_vs_retained"] = float(spearmanr(t.recov_marginal_norm, t.retained)[0])
    out["partial_spearman_within_given_Ivar_auroc"] = partial_spearman(t.recov_within_norm, t.retained,
                                                                      [t.I_var, t.image_auroc])
    ds = t.dataset.unique()
    boots = []
    for _ in range(B):
        pick = rng.choice(ds, len(ds), replace=True)
        bt = pd.concat([t[t.dataset == d] for d in pick])
        if bt.recov_within_norm.nunique() > 2:
            boots.append(partial_spearman(bt.recov_within_norm, bt.retained, [bt.I_var, bt.image_auroc]))
    out["partial_spearman_dataset_cluster_ci95"] = ci(np.array(boots)) if boots else None
    # per-representation correlations (generalisation across representations)
    out["spearman_by_rung"] = {r: float(spearmanr(g.recov_within_norm, g.retained)[0])
                               for r, g in t.groupby("rung") if len(g) >= 5}
    # leave-one-dataset-out ridge prediction of retained fraction
    feats = ["recov_within_norm", "recov_marginal_norm", "image_auroc", "I_var", "missing_rate"]
    tt = t.dropna(subset=feats)
    errs, base_errs = [], []
    for d in tt.dataset.unique():
        tr, te = tt[tt.dataset != d], tt[tt.dataset == d]
        if len(tr) < 8 or len(te) == 0:
            continue
        mu, sd = tr[feats].mean(), tr[feats].std().replace(0, 1)
        m = RidgeCV(alphas=np.logspace(-2, 3, 11)).fit((tr[feats] - mu) / sd, tr.retained)
        errs.append(np.abs(m.predict((te[feats] - mu) / sd) - te.retained).mean())
        base_errs.append(np.abs(tr.retained.median() - te.retained).mean())
    out["lodo_mae_ridge"] = float(np.mean(errs)) if errs else None
    out["lodo_mae_median_baseline"] = float(np.mean(base_errs)) if base_errs else None
    out["lodo_n_folds"] = len(errs)
    return out, t


def plot(t):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dss = sorted(t.dataset.unique())
    fig, axes = plt.subplots(1, len(dss), figsize=(2.1 * len(dss), 2.2), sharey=True, squeeze=False)
    for ax, d in zip(axes[0], dss):
        g = t[t.dataset == d]
        ax.scatter(g.recov_within_norm, g.retained, s=12, color="#2a78d6", edgecolor="white", linewidth=0.4)
        ax.axhline(0, color="#5b6b75", lw=0.7)
        ax.set_title(d, fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("within-class recoverability", fontsize=6)
    axes[0][0].set_ylabel("retained fraction", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESEARCH / "results/E30_scatter.pdf")


def main():
    rng = np.random.default_rng(30)
    dss = [d for d in DATASETS if available(d)]
    exp = Experiment("E30_recoverability_utility", "H30 (exploratory): higher within-class recoverability goes with "
                     "a lower retained fraction, controlling for I_var and image AUROC.",
                     {"datasets": dss, "rows": "complete cases"}, {"rungs": RUNGS, "bootstrap": B, "folds": "seed 0"}, 0)
    rows = []
    for d in dss:
        rows += dataset_rows(d, rng)
        pd.DataFrame(rows).to_csv(RESEARCH / "results/E30_table.csv", index=False)  # checkpoint after each dataset
    tab = pd.DataFrame(rows)
    summary, t = analyse(tab, rng)
    if len(t):
        plot(t)
    print(summary, flush=True)
    exp.finish(summary, interpretation="See research/iteration4/SUMMARY.md.")


if __name__ == "__main__":
    main()
