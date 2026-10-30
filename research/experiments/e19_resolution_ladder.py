"""E19: a controlled capacity manipulation (addresses review R3-3 and limitation iii).

The architecture ladder confounds capacity with architecture. Here the model is
fixed (frozen ConvNeXt-T) and only the image information is varied: each 224 crop
is downsampled to r x r and upsampled back to 224 (bilinear), r in {16, 32, 64, 112, 224}.
Prediction (written before running): added value of metadata and the retained
fraction rho both decrease monotonically as r increases, on every dataset.
Falsified if, on a dataset, added value at r = 224 is not below r = 16, or the
sequence is clearly non-monotone (Spearman rho between r and added value > -0.8).
Fold assignment 0; grouped bootstrap CIs for added value.
Usage: python e19_resolution_ladder.py embed <dataset> | analyse
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import Experiment
from e10_ham_attenuation import lesion_bootstrap, oof_scores, second_stage
from e10_ham_embed import DEV, MEAN, STD, headless, load_224
from e17_retained_fraction import prevalence_ll
from e18_foundation_rung import load_dataset

RES = [16, 32, 64, 112, 224]
DATASETS = ["ham10000", "isic2019_nonham", "pad_ufes", "isic2020", "hiba", "milk10k_derm", "milk10k_clin"]


@torch.no_grad()
def embed(name, chunk=512):
    d = load_dataset(name)
    model = headless("convnext_tiny")
    feats = {r: [] for r in RES}
    t0 = time.perf_counter()
    for i in range(0, len(d["paths"]), chunk):
        x = torch.stack([torch.from_numpy(np.asarray(load_224(p), np.float32) / 255).permute(2, 0, 1)
                         for p in d["paths"][i:i + chunk]])
        for r in RES:
            xr = x if r == 224 else F.interpolate(F.interpolate(x, size=r, mode="bilinear", antialias=True,
                                                                align_corners=False),
                                                  size=224, mode="bilinear", align_corners=False)
            xr = (xr - MEAN) / STD
            feats[r].append(torch.cat([model(xr[j:j + 64].to(DEV)).float().cpu() for j in range(0, len(xr), 64)]).numpy())
        print(f"{name}: {min(i + chunk, len(d['paths']))}/{len(d['paths'])} ({time.perf_counter() - t0:.0f}s)", flush=True)
    for r in RES:
        np.save(d["dir"] / f"emb_cnx_res{r}.npy", np.concatenate(feats[r]).astype(np.float32))


def analyse_one(name, rng):
    from sklearn.model_selection import StratifiedGroupKFold
    from e10_ham_attenuation import meta_matrix
    from e13_replication import oof_block
    d = load_dataset(name)
    y, groups, df = d["y"], d["groups"], d["df"]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    if d["ham"]:
        M = np.zeros((len(df), meta_matrix(df, folds[0][0]).shape[1]))
        for tr, te in folds:
            M[te] = meta_matrix(df, tr)[te]
    else:
        M = oof_block(df, *d["blocks"]["all"], folds)
    _, l_meta = second_stage(np.zeros((len(y), 1)), M, y, folds)
    i_a = float((prevalence_ll(y, folds) - l_meta).mean())
    out = {"I_A_nats": round(i_a, 5)}
    avs = []
    for r in RES:
        score, _ = oof_scores(np.load(d["dir"] / f"emb_cnx_res{r}.npy"), y, groups, folds)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
        dlt = l0 - l1
        avs.append(dlt.mean())
        out[r] = {"image_auroc": round(float(roc_auc_score(y, score)), 4), "added_value": round(float(dlt.mean()), 5),
                  "ci95": [round(v, 5) for v in lesion_bootstrap(dlt, groups, rng)], "rho": round(float(dlt.mean() / i_a), 3)}
    out["spearman_res_vs_av"] = round(float(spearmanr(RES, avs)[0]), 3)
    out["prediction_holds"] = bool(avs[-1] < avs[0] and out["spearman_res_vs_av"] <= -0.8)
    print(name, out, flush=True)
    return out


def analyse():
    rng = np.random.default_rng(19)
    exp = Experiment("E19_resolution_ladder",
                     "With architecture fixed, metadata added value and rho decrease monotonically with input resolution.",
                     {"datasets": DATASETS}, {"model": "frozen ConvNeXt-T", "resolutions": RES, "folds": "assignment 0"}, 0)
    def ready(n):
        try:
            return (load_dataset(n)["dir"] / "emb_cnx_res224.npy").exists()
        except Exception:  # dataset not downloaded yet
            return False
    res = {n: analyse_one(n, rng) for n in DATASETS if ready(n)}
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    if sys.argv[1] == "embed":
        embed(sys.argv[2])
    else:
        analyse()
