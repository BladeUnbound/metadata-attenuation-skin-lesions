"""E32 (exploratory; request E22): correspondence, permutation and shortcut controls.
Registered in research/iteration4/REGISTRY_E30_E34.md before implementation.

Per dataset, on original data and on complete cases, rungs HSV and ConvNeXt-T:
  matched     added value of correctly matched metadata
  permuted    R = 50 permutations: each grouping unit's metadata record (taken from
              one representative image) is reassigned to another random unit, which
              preserves the joint metadata distribution and missingness patterns while
              breaking the image-metadata-label correspondence; empirical p-value
  loo         leave-one-variable-out added value (complete cases, ConvNeXt-T)
  source      source-adjusted image-only and fused models (HAM10000: `dataset`;
              ISIC 2019: BCN vs other from the lesion_id prefix)
  implausible frequency of (age < 30, positive label) in real vs permuted records
Fused and image-only models also report AUROC, Brier and calibration.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from x_common import (DATASETS, VARS, available, block, cached_score, ci, folds_for, full_metrics, group_boot_mean,
                      oof_probs, second_stage, subset)

R, B = 50, 1000


def source_codes(ds, s):
    if ds == "ham10000":
        return pd.get_dummies(s["df"]["dataset"], dtype=float).to_numpy()
    if ds == "isic2019_nonham":
        return s["df"]["lesion_id"].fillna("").str.startswith("BCN").to_numpy(float)[:, None]
    return None


def permuted_df(s, rng):
    df, g = s["df"], s["g"]
    units = pd.Series(np.arange(len(g))).groupby(g).first()       # representative row per unit
    perm = dict(zip(units.index, rng.permutation(units.to_numpy())))
    src = np.array([perm[x] for x in g])
    cols = [c for c in df.columns if c != "path"]
    out = df.copy()
    meta_cols = sorted({c for _, _, c in VARS_DS})
    out[meta_cols] = df[meta_cols].iloc[src].to_numpy()
    return out


def run(ds, cc, rng):
    global VARS_DS
    VARS_DS = VARS[ds]
    s = subset(ds, complete_case=cc)
    folds = folds_for(s)
    y, g = s["y"], s["g"]
    tag = "cc" if cc else "all"
    M = block(s, VARS[ds], folds)
    res = {"n": int(len(y))}
    age_col = [c for n, k, c in VARS[ds] if n == "age"][0]
    real_age = pd.to_numeric(s["df"][age_col], errors="coerce")
    res["implausible_real"] = float(((real_age < 30) & (y == 1)).mean())
    for rung in ["hsv", "convnext_tiny"]:
        score = cached_score(ds, s, rung, tag, folds)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
        matched = float((l0 - l1).mean())
        nulls, impl = [], []
        for _ in range(R):
            sp = dict(s)
            sp["df"] = permuted_df(s, rng)
            Mp = block(sp, VARS[ds], folds)
            p0, p1 = second_stage(score[:, None], np.column_stack([score, Mp]), y, folds)
            nulls.append(float((p0 - p1).mean()))
            pa = pd.to_numeric(sp["df"][age_col], errors="coerce")
            impl.append(float(((pa < 30) & (y == 1)).mean()))
        nulls = np.array(nulls)
        p_img, p_fus = oof_probs(score[:, None], y, folds), oof_probs(np.column_stack([score, M]), y, folds)
        r = {"matched": round(matched, 6), "matched_ci95": ci(group_boot_mean(l0 - l1, g, rng, B)),
             "perm_mean": round(float(nulls.mean()), 6), "perm_sd": round(float(nulls.std()), 6),
             "perm_max": round(float(nulls.max()), 6), "p_empirical": float((1 + np.sum(nulls >= matched)) / (R + 1)),
             "image_only": full_metrics(y, p_img), "fused": full_metrics(y, p_fus)}
        res["implausible_permuted_mean"] = float(np.mean(impl))
        if cc and rung == "convnext_tiny":
            r["leave_one_out"] = {}
            for name, _, _ in VARS[ds]:
                Mi = block(s, [v for v in VARS[ds] if v[0] != name], folds)
                q0, q1 = second_stage(score[:, None], np.column_stack([score, Mi]), y, folds)
                r["leave_one_out"][name] = round(float((q0 - q1).mean()), 6)
        S = source_codes(ds, s)
        if S is not None:
            a0, a1 = second_stage(np.column_stack([score, S]), np.column_stack([score, S, M]), y, folds)
            r["source_adjusted_delta"] = round(float((a0 - a1).mean()), 6)
            r["source_adjusted_image_only"] = full_metrics(y, oof_probs(np.column_stack([score, S]), y, folds))
        res[rung] = r
        print(ds, tag, rung, {k: v for k, v in r.items() if k in ("matched", "perm_mean", "perm_max", "p_empirical")},
              flush=True)
    return res


def main():
    rng = np.random.default_rng(32)
    dss = [d for d in DATASETS if available(d)]
    exp = Experiment("E32_permutation_controls", "H32 (exploratory): matched metadata beats every permutation where "
                     "matched added value > 0; permuted added value is centred at or below 0.",
                     {"datasets": dss}, {"permutations": R, "rungs": ["hsv", "convnext_tiny"], "folds": "seed 0"}, 0)
    out = {}
    for d in dss:
        out[d] = {"original": run(d, False, rng), "complete_case": run(d, True, rng)}
    exp.finish(out, interpretation="See research/iteration4/SUMMARY.md.")


if __name__ == "__main__":
    main()
