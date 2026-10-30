"""E22 P1/P2 (pre-registered in iteration3/PREREGISTRATION_E22.md).

P1: MILK10k, lesions that have both a clinical and a dermoscopic image. For each
modality separately (one image per lesion per modality, same lesions, same folds):
rho = Delta_ConvNeXt / I_A. Paired lesion bootstrap of rho_clin - rho_derm.
Prediction: > 0 (clinical images leave more metadata information non-redundant).

P2: MSKCC 2020, measured lesion diameter as its own block: added value over HSV
and over ConvNeXt-T, and its retained fraction (prediction: < 0.5).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import lesion_bootstrap, oof_scores, second_stage
from e13_replication import adapter, oof_block
from e17_retained_fraction import prevalence_ll

B = 2000


def paired_modality(rng):
    der, cli = adapter("milk10k_derm"), adapter("milk10k_clin")
    Xd = np.load(RESEARCH / "data/milk10k_derm/emb_convnext_tiny.npy")
    Xc = np.load(RESEARCH / "data/milk10k_clin/emb_convnext_tiny.npy")
    # one image per lesion per modality, lesions present in both
    di = der["df"].reset_index().drop_duplicates("lesion_id").set_index("lesion_id")["index"]
    ci = cli["df"].reset_index().drop_duplicates("lesion_id").set_index("lesion_id")["index"]
    common = di.index.intersection(ci.index)
    di, ci = di[common].to_numpy(), ci[common].to_numpy()
    y = der["y"][di]
    assert (y == cli["y"][ci]).all(), "labels must agree across modalities for the same lesion"
    dfm = der["df"].iloc[di].reset_index(drop=True)
    groups = np.asarray(common)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(dfm, y, groups))
    M = oof_block(dfm, *der["blocks"]["all"], folds)
    _, l_meta = second_stage(np.zeros((len(y), 1)), M, y, folds)
    d_meta = prevalence_ll(y, folds) - l_meta
    out, per = {"n_lesions": int(len(common)), "positives": int(y.sum()), "I_A_nats": round(float(d_meta.mean()), 5)}, {}
    for mod, X in (("dermoscopic", Xd[di]), ("clinical", Xc[ci])):
        score, _ = oof_scores(X, y, groups, folds)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
        per[mod] = l0 - l1
        out[mod] = {"av": round(float(per[mod].mean()), 5), "rho": round(float(per[mod].mean() / d_meta.mean()), 3)}
    idx = rng.integers(0, len(y), (B, len(y)))  # one image per lesion, so lesion bootstrap = row bootstrap
    rho_c = per["clinical"][idx].mean(1) / d_meta[idx].mean(1)
    rho_d = per["dermoscopic"][idx].mean(1) / d_meta[idx].mean(1)
    diff = rho_c - rho_d
    out["rho_clin_minus_derm"] = round(float(out["clinical"]["rho"] - out["dermoscopic"]["rho"]), 3)
    out["ci95"] = [round(float(np.percentile(diff, q)), 3) for q in (2.5, 97.5)]
    out["P1_supported"] = bool(out["ci95"][0] > 0)
    return out


def size_block(rng):
    a = adapter("msk2020")
    df, y, groups = a["df"], a["y"], a["groups"]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    S = oof_block(df, ["clin_size_long_diam_mm"], [], folds)
    out = {}
    for k in ["hsv", "convnext_tiny"]:
        score, _ = oof_scores(np.load(RESEARCH / f"data/msk2020/emb_{k}.npy"), y, groups, folds)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, S]), y, folds)
        dd = l0 - l1
        out[k] = {"av": round(float(dd.mean()), 5), "ci95": [round(v, 5) for v in lesion_bootstrap(dd, groups, rng)]}
    out["retained"] = round(out["convnext_tiny"]["av"] / out["hsv"]["av"], 3) if out["hsv"]["av"] > 0 else None
    out["P2_supported"] = bool(out["retained"] is not None and out["retained"] < 0.5)
    return out


def main():
    rng = np.random.default_rng(22)
    exp = Experiment("E22_paired_modality", "P1: rho higher for clinical than dermoscopic images of the same lesions; "
                     "P2: measured diameter attenuates with capacity.", {"datasets": ["milk10k", "msk2020"]},
                     {"bootstrap": B, "folds": "assignment 0"}, 0)
    res = {"P1_milk10k": paired_modality(rng), "P2_msk2020": size_block(rng)}
    print(res, flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
