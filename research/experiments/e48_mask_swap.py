"""E48 (exploratory; registered in research/iteration5/REGISTRY.md, 22:03 IST): mask-swap null for the CAM overlap test.

Same procedure as E15 (out-of-fold melanoma head per backbone, all melanomas of each test fold plus an equal number
of non-melanomas, exact melanoma-evidence CAM at 56x56), on a fresh random selection. For every image:
  p_shift  exact circular-shift null (as published),
  p_swap   own mask vs the masks of 199 other selected images from the same lesion-area decile (null H0'': the map does
           not depend on which lesion is in the image),
and the same two p-values for a fixed centred Gaussian "evidence" map (positive control for the argument in
THEORY_AUDIT.md section 5: centre bias alone should pass the shift null but not the swap null).

Usage: python e48_mask_swap.py [backbones...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from common import RESEARCH, Experiment
from e10_ham_attenuation import fit_tuned
from e10_ham_embed import DATA, DEV, image_paths
from e12_ham_cam_lesion import mask_224
from e15_cam_backbones import RES, cam_and_check, maps, spatial_model
from saliency_overlap import overlap_metrics, shift_null_exact

N_SWAP = 199


def gaussian_map(res=RES, sd=RES / 6):
    u = np.arange(res) - (res - 1) / 2
    return np.exp(-(u[:, None] ** 2 + u[None, :] ** 2) / (2 * sd ** 2))


def swap_p(e, own, pool, rng):
    obs = overlap_metrics(e, own)["mass_ratio"]
    js = rng.choice(len(pool), size=min(N_SWAP, len(pool)), replace=False)
    null = np.array([overlap_metrics(e, pool[j])["mass_ratio"] for j in js])
    return (1 + np.sum(null >= obs)) / (1 + len(null))


def main(backbones):
    df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
    y = (df["dx"] == "mel").to_numpy(int)
    groups = df["lesion_id"].to_numpy()
    paths = image_paths(df)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    exp = Experiment("E48_mask_swap", "REGISTRY.md E48 (exploratory): shift null vs mask-swap null; centred-Gaussian control",
                     {"source": "HAM10000 + lesion masks"}, {"backbones": backbones, "res": RES, "n_swap": N_SWAP}, 48)
    G = gaussian_map()
    out = {}
    for name in backbones:
        X = np.load(DATA / f"emb_{name}.npy")
        body, gamma, eps = spatial_model(name)
        rng = np.random.default_rng(48)
        cams, masks, ys, preds = [], [], [], []
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            head, _ = fit_tuned(sc.transform(X[tr]), y[tr], groups[tr])
            w_std = head.coef_[0] / sc.scale_
            logit = head.decision_function(sc.transform(X[te]))
            pos = {i: j for j, i in enumerate(te)}
            mel = te[y[te] == 1]
            sel = np.concatenate([mel, rng.choice(te[y[te] == 0], size=len(mel), replace=False)])
            cam, _ = cam_and_check(name, maps(body, [paths[i] for i in sel]), w_std, gamma, eps)
            cam = F.interpolate(torch.from_numpy(cam)[:, None].float(), size=RES, mode="bilinear", align_corners=False)[:, 0].numpy()
            for j, i in enumerate(sel):
                mk = np.asarray(Image.fromarray(mask_224(df["image_id"][i]).astype(np.uint8) * 255)
                                .resize((RES, RES), Image.NEAREST)) > 127
                if mk.sum() == 0 or mk.all():
                    continue
                cams.append(cam[j]); masks.append(mk); ys.append(int(y[i])); preds.append(int(logit[pos[i]] > 0))
        area = np.array([m.mean() for m in masks])
        dec = np.digitize(area, np.quantile(area, np.linspace(0.1, 0.9, 9)))
        rows = []
        for k in range(len(masks)):
            pool_idx = np.flatnonzero((dec == dec[k]) & (np.arange(len(masks)) != k))
            pool = [masks[j] for j in pool_idx]
            _, ps = shift_null_exact(cams[k], masks[k])
            _, pg = shift_null_exact(G, masks[k])
            rows.append({"y": ys[k], "pred": preds[k], "area": area[k], "p_shift": ps, "p_swap": swap_p(cams[k], masks[k], pool, rng),
                         "gauss_p_shift": pg, "gauss_p_swap": swap_p(G, masks[k], pool, rng)})
        r = pd.DataFrame(rows)
        r.to_csv(RESEARCH / f"results/E48_{name}_per_image.csv", index=False)
        out[name] = {"n_images": int(len(r)),
                     "frac_sig_shift": round(float((r.p_shift <= 0.05).mean()), 3),
                     "frac_sig_swap": round(float((r.p_swap <= 0.05).mean()), 3),
                     "gauss_frac_sig_shift": round(float((r.gauss_p_shift <= 0.05).mean()), 3),
                     "gauss_frac_sig_swap": round(float((r.gauss_p_swap <= 0.05).mean()), 3),
                     "frac_sig_swap_melanoma": round(float((r[r.y == 1].p_swap <= 0.05).mean()), 3),
                     "frac_sig_swap_missed_melanoma": round(float((r[(r.y == 1) & (r.pred == 0)].p_swap <= 0.05).mean()), 3)}
        print(name, out[name], flush=True)
        del body
        if DEV == "mps":
            torch.mps.empty_cache()
    exp.finish(out, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["resnet18", "resnet50", "convnext_tiny"])
