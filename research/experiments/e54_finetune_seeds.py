"""E54 (registered in research/iteration6/REGISTRY.md, 2026-10-29 13:49 IST): repeated training seeds for the E14b
fine-tuning runs on the small sets. Same schedule, folds (assignment 0), image loading and second stage as E14b
(original estimator, for comparability; review item R7-11 on strict cross-fitting stays open). Training seed s uses
torch/numpy seed 1000*s + k for fold k (E14b used k). Per-fold logits are checkpointed in data/<ds>/e54/.

Usage: python e54_finetune_seeds.py <dataset> <seed> [<seed> ...]
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import lesion_bootstrap, oof_scores, second_stage
from e13_replication import oof_block
from e14_finetune import load_256, train_fold
from e18_foundation_rung import load_dataset

BACKBONES = ["resnet18", "convnext_tiny"]


def main(name, seeds):
    d = load_dataset(name)
    df, y, groups = d["df"], d["y"], d["groups"]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    M = oof_block(df, *d["blocks"]["all"], folds)
    ck = RESEARCH / f"data/{name}/e54"
    ck.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    imgs = [load_256(p) for p in d["paths"]]
    print(f"{name}: loaded {len(imgs)} images in {time.perf_counter() - t0:.0f}s", flush=True)
    exp = Experiment(f"E54_{name}", "REGISTRY.md E54: seed spread of fine-tuned added value (E14b schedule)",
                     {"dataset": name, "n": int(len(y))}, {"backbones": BACKBONES, "seeds": [0] + seeds,
                                                          "seed_rule": "fold k, seed s: 1000*s + k (seed 0 = E14b)",
                                                          "estimator": "E14b original second stage"}, 0)
    rng = np.random.default_rng(154)
    res = {}
    for bb in BACKBONES:
        frozen, _ = oof_scores(np.load(d["dir"] / f"emb_{bb}.npy"), y, groups, folds)
        f0, f1 = second_stage(frozen[:, None], np.column_stack([frozen, M]), y, folds)
        res[bb] = {"frozen_added_value": round(float((f0 - f1).mean()), 5)}
        per = {}
        for s in [0] + seeds:
            score = np.zeros(len(y))
            for k, (tr, te) in enumerate(folds):
                path = (RESEARCH / f"data/{name}/e14b/{bb}_fold{k}.npy") if s == 0 else ck / f"{bb}_seed{s}_fold{k}.npy"
                if path.exists():
                    score[te] = np.load(path)
                    continue
                score[te] = train_fold(bb, imgs, y, tr, te, seed=1000 * s + k)
                np.save(path, score[te])
                print(f"{name} {bb} seed {s} fold {k}: AUROC {roc_auc_score(y[te], score[te]):.3f}", flush=True)
            l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
            dd = l0 - l1
            per[s] = {"auroc": round(float(roc_auc_score(y, score)), 4), "added_value": round(float(dd.mean()), 5),
                      "added_value_ci95": [round(v, 5) for v in lesion_bootstrap(dd, groups, rng)],
                      "frozen_minus_finetuned_ci95": [round(v, 5) for v in lesion_bootstrap((f0 - f1) - dd, groups, rng)]}
            print(name, bb, s, per[s], flush=True)
        av = [v["added_value"] for v in per.values()]
        res[bb].update(per_seed=per, added_value_mean=round(float(np.mean(av)), 5),
                       added_value_range=[min(av), max(av)])
    exp.finish(res, interpretation="See iteration6 reports.")


if __name__ == "__main__":
    main(sys.argv[1], [int(s) for s in sys.argv[2:]])
