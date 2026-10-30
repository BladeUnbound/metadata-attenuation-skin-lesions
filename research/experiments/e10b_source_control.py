"""E10b (exploratory, not pre-registered): is the E10 attenuation driven by
image-source detection?

E10a showed the image source alone predicts melanoma (AUROC 0.73), and the dev
run showed deep features identify the source almost perfectly. If deep models
gain by detecting the source, metadata's shrinking value could reflect that
shortcut rather than better lesion reading.
Controls, on E10's seed-0 lesion-grouped folds:
  1. Source-adjusted added value: second stage [score, source] vs
     [score, source, metadata], so both sides already know the source.
  2. Within-source melanoma AUROC of the out-of-fold image score, for the two
     sources with enough melanomas (vidir_modern, rosendahl).
  3. Paired lesion-bootstrap tests of attenuation between rungs (same images,
     same folds): AV(a) - AV(b) for hsv vs each deep rung and for
     resnet18 vs the two strongest rungs, with and without source adjustment.
Expectation: if attenuation is real lesion information, (1) still falls with
capacity and (2) still rises with capacity inside each source.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import DATA, RUNGS, lesion_bootstrap, meta_matrix, oof_scores, second_stage

SOURCES_FOR_AUROC = ["vidir_modern", "rosendahl"]


def main():
    df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
    y = (df["dx"] == "mel").to_numpy(int)
    groups = df["lesion_id"].to_numpy()
    src = pd.get_dummies(df["dataset"], dtype=float).to_numpy()
    rng = np.random.default_rng(1)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    M = np.zeros((len(df), meta_matrix(df, folds[0][0]).shape[1]))
    for tr, te in folds:
        M[te] = meta_matrix(df, tr)[te]
    exp = Experiment("E10b_source_control",
                     "Attenuation persists after adjusting for image source; within-source AUROC rises with capacity.",
                     {"source": "HAM10000", "note": "exploratory; deviation from E10 pre-registration"},
                     {"folds": "E10 seed 0", "rungs": RUNGS}, 0)
    res, per_image = {}, {}
    for rung in RUNGS:
        X = np.load(DATA / f"emb_{rung}.npy")
        score, _ = oof_scores(X, y, groups, folds)
        ll_s, ll_sm = second_stage(np.column_stack([score, src]), np.column_stack([score, src, M]), y, folds)
        ll_i, ll_im = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
        d = ll_s - ll_sm
        per_image[rung] = {"unadjusted": ll_i - ll_im, "source_adjusted": d}
        res[rung] = {
            "source_adjusted_added_value": round(float(d.mean()), 5),
            "ci95": [round(v, 5) for v in lesion_bootstrap(d, groups, rng)],
            "within_source_auroc": {s: round(float(roc_auc_score(y[df["dataset"] == s], score[df["dataset"] == s])), 4)
                                    for s in SOURCES_FOR_AUROC},
        }
        print(rung, res[rung], flush=True)
    pairs = [("hsv", r) for r in RUNGS[1:]] + [("resnet18", "convnext_tiny"), ("resnet18", "vit_b_16"),
                                               ("resnet50", "convnext_tiny")]
    paired = {}
    for kind in ("unadjusted", "source_adjusted"):
        for a, b in pairs:
            diff = per_image[a][kind] - per_image[b][kind]
            paired[f"{kind}:{a}-{b}"] = {"av_diff": round(float(diff.mean()), 5),
                                         "ci95": [round(v, 5) for v in lesion_bootstrap(diff, groups, rng)]}
    res["paired_attenuation"] = paired
    for k, v in paired.items():
        print(k, v, flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
