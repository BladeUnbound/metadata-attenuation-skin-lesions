"""E18 (exploratory extension, not pre-registered): does attenuation continue
above the ImageNet-1k ladder, at a large weakly supervised foundation model?

Rung 6: torchvision ViT-L/16, SWAG_LINEAR_V1 weights (weakly supervised
pretraining on Instagram hashtags, linear head on ImageNet; built into
torchvision, no external code). Same preprocessing as the other rungs
(shorter side 224, centre crop 224, ImageNet normalisation), pooled class-token
features. Then, per dataset (fold assignment 0): image AUROC, added value of
metadata with grouped bootstrap CI, retained fraction rho, and the paired
difference ConvNeXt-T minus ViT-L/16-SWAG.
Usage: python e18_foundation_rung.py embed <dataset> | analyse
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import lesion_bootstrap, meta_matrix, oof_scores, second_stage
from e10_ham_embed import DEV, MEAN, STD, image_paths, load_224
from e13_replication import adapter, oof_block

DATASETS = ["ham10000", "isic2020", "isic2019_nonham", "pad_ufes", "hiba", "msk2020", "milk10k_derm", "milk10k_clin"]
RUNG = "vitl16_swag"


def load_dataset(name):
    if name == "ham10000":
        from e10_ham_embed import DATA
        df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
        return dict(df=df, y=(df["dx"] == "mel").to_numpy(int), groups=df["lesion_id"].to_numpy(),
                    paths=image_paths(df), dir=DATA, ham=True)
    a = adapter(name)
    a.update(paths=list(a["df"]["path"]), dir=RESEARCH / f"data/{name}", ham=False)
    return a


@torch.no_grad()
def embed(name, chunk=512):
    d = load_dataset(name)
    m = tvm.vit_l_16(weights=tvm.ViT_L_16_Weights.IMAGENET1K_SWAG_LINEAR_V1)
    m.heads = torch.nn.Identity()
    m = m.eval().to(DEV)
    feats, t0 = [], time.perf_counter()
    for i in range(0, len(d["paths"]), chunk):
        x = torch.stack([torch.from_numpy(np.asarray(load_224(p), np.float32) / 255).permute(2, 0, 1)
                         for p in d["paths"][i:i + chunk]])
        x = (x - MEAN) / STD
        feats.append(torch.cat([m(x[j:j + 32].to(DEV)).float().cpu() for j in range(0, len(x), 32)]).numpy())
        print(f"{name}: {min(i + chunk, len(d['paths']))}/{len(d['paths'])} ({time.perf_counter() - t0:.0f}s)", flush=True)
    np.save(d["dir"] / f"emb_{RUNG}.npy", np.concatenate(feats).astype(np.float32))


def analyse_one(name, rng):
    d = load_dataset(name)
    y, groups, df = d["y"], d["groups"], d["df"]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    if d["ham"]:
        M = np.zeros((len(df), meta_matrix(df, folds[0][0]).shape[1]))
        for tr, te in folds:
            M[te] = meta_matrix(df, tr)[te]
    else:
        M = oof_block(df, *d["blocks"]["all"], folds)
    out, per = {}, {}
    for rung in ["convnext_tiny", RUNG]:
        score, _ = oof_scores(np.load(d["dir"] / f"emb_{rung}.npy"), y, groups, folds)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
        per[rung] = l0 - l1
        out[rung] = {"image_auroc": round(float(roc_auc_score(y, score)), 4),
                     "added_value": round(float(per[rung].mean()), 5),
                     "ci95": [round(v, 5) for v in lesion_bootstrap(per[rung], groups, rng)]}
    diff = per["convnext_tiny"] - per[RUNG]
    out["convnext_minus_vitl"] = {"diff": round(float(diff.mean()), 5),
                                  "ci95": [round(v, 5) for v in lesion_bootstrap(diff, groups, rng)]}
    print(name, out, flush=True)
    return out


def analyse():
    rng = np.random.default_rng(18)
    exp = Experiment("E18_foundation_rung", "Added value of metadata at a ViT-L/16 SWAG representation is no larger "
                     "than at ConvNeXt-T (attenuation continues or plateaus).",
                     {"datasets": DATASETS, "note": "exploratory extension; not pre-registered"},
                     {"rung": "torchvision vit_l_16 IMAGENET1K_SWAG_LINEAR_V1, class token", "folds": "assignment 0"}, 0)
    def ready(n):
        try:
            return (load_dataset(n)["dir"] / f"emb_{RUNG}.npy").exists()
        except Exception:  # dataset not downloaded yet
            return False
    exp.finish({n: analyse_one(n, rng) for n in DATASETS if ready(n)},
               interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2])
    else:
        analyse()
