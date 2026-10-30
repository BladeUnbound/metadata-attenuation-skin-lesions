"""E14b (exploratory): fine-tuned models on further datasets, same fixed schedule as E14.

Usage: python e14b_finetune_multi.py <dataset> <backbone> [<backbone> ...]
Fold assignment 0 (StratifiedGroupKFold by the dataset's grouping unit).
Per-fold out-of-fold logits are checkpointed, so the run resumes after interruption.
Reports fine-tuned vs frozen image AUROC, added value of metadata on each
(E13 encoding), lesion/patient-bootstrap CIs, and the paired difference.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import lesion_bootstrap, meta_matrix, oof_scores, second_stage
from e13_replication import oof_block
from e14_finetune import load_256, train_fold
from e18_foundation_rung import load_dataset


def main(name, backbones):
    d = load_dataset(name)
    df, y, groups = d["df"], d["y"], d["groups"]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    if d["ham"]:
        M = np.zeros((len(df), meta_matrix(df, folds[0][0]).shape[1]))
        for tr, te in folds:
            M[te] = meta_matrix(df, tr)[te]
    else:
        M = oof_block(df, *d["blocks"]["all"], folds)
    ck_dir = RESEARCH / f"data/{name}/e14b"
    ck_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    imgs = [load_256(p) for p in d["paths"]]
    print(f"{name}: loaded {len(imgs)} images in {time.perf_counter() - t0:.0f}s", flush=True)
    exp = Experiment(f"E14b_{name}", "Fine-tuned scores leave no more metadata added value than frozen ones.",
                     {"dataset": name, "n": int(len(y))}, {"backbones": backbones, "schedule": "as E14"}, 0)
    rng = np.random.default_rng(141)
    res = {}
    for bb in backbones:
        score = np.zeros(len(y))
        for k, (tr, te) in enumerate(folds):
            ck = ck_dir / f"{bb}_fold{k}.npy"
            if ck.exists():
                score[te] = np.load(ck)
                continue
            score[te] = train_fold(bb, imgs, y, tr, te, seed=k)
            np.save(ck, score[te])
            print(f"{name} {bb} fold {k}: AUROC {roc_auc_score(y[te], score[te]):.3f}", flush=True)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
        frozen, _ = oof_scores(np.load(d["dir"] / f"emb_{bb}.npy"), y, groups, folds)
        f0, f1 = second_stage(frozen[:, None], np.column_stack([frozen, M]), y, folds)
        d_ft, d_fr = l0 - l1, f0 - f1
        res[bb] = {"finetuned_auroc": round(float(roc_auc_score(y, score)), 4),
                   "frozen_auroc": round(float(roc_auc_score(y, frozen)), 4),
                   "added_value_finetuned": round(float(d_ft.mean()), 5),
                   "added_value_finetuned_ci95": [round(v, 5) for v in lesion_bootstrap(d_ft, groups, rng)],
                   "added_value_frozen": round(float(d_fr.mean()), 5),
                   "frozen_minus_finetuned_ci95": [round(v, 5) for v in lesion_bootstrap(d_fr - d_ft, groups, rng)]}
        print(name, bb, res[bb], flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
