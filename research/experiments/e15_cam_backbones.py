"""E15: interpretability across backbones (closes review item M7).

Exact class activation maps for frozen ResNet-18, ResNet-50 and ConvNeXt-T with
the out-of-fold logistic head from E10's seed-0 lesion-grouped folds.
  ResNet:   logit = b + sum_j (w_j / s_j) * mean_u F_j(u)        -> c(u) = sum_j (w_j/s_j) F_j(u)
  ConvNeXt: pooled p -> LayerNorm (per-image mean mu, std sigma, affine gamma, beta)
            logit = const + (1/sigma) * mean_u sum_j w~_j (F_j(u) - Fbar(u)),  w~_j = w_j gamma_j / s_j
            -> c(u) = sum_j w~_j (F_j(u) - Fbar(u)), exact up to the positive per-image factor 1/sigma
Each backbone's reconstruction is checked against the real head logits (Pearson r).
Same images as E12 (all melanomas plus a matched sample per fold), melanoma-evidence
maps (+CAM) as in E12b, overlap at 56x56, exact FFT shift null and pixel permutation.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from scipy.stats import mannwhitneyu
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parents[1] / "pipeline"))
from common import RESEARCH, Experiment
from e10_ham_attenuation import fit_tuned
from e10_ham_embed import DATA, DEV, MEAN, STD, image_paths, load_224
from e12_ham_cam_lesion import mask_224
from saliency_overlap import overlap_metrics, shift_null_exact

SEED, RES, N_PERM = 0, 56, 99


def spatial_model(name):
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    elif name == "resnet50":
        m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2)
    else:
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        ln = m.classifier[0]
        return m.features.eval().to(DEV), ln.weight.detach().cpu().numpy(), ln.eps
    return torch.nn.Sequential(*list(m.children())[:-2]).eval().to(DEV), None, None


@torch.no_grad()
def maps(body, paths, batch=32):
    out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([torch.from_numpy(np.asarray(load_224(p), np.float32) / 255).permute(2, 0, 1)
                         for p in paths[i:i + batch]])
        out.append(body(((x - MEAN) / STD).to(DEV)).float().cpu())
    return torch.cat(out)


def cam_and_check(name, Fm, w_std, gamma, eps):
    """Return CAM maps and per-image logits reconstructed from them (up to an affine map)."""
    Fn = Fm.numpy()  # (N, C, h, w)
    if gamma is None:
        cam = np.einsum("c,nchw->nhw", w_std, Fn)
        recon = cam.mean((1, 2))
    else:
        wt = w_std * gamma
        Fbar = Fn.mean(1, keepdims=True)
        cam = np.einsum("c,nchw->nhw", wt, Fn - Fbar)
        p = Fn.mean((2, 3))
        sigma = np.sqrt(p.var(1) + eps)
        recon = cam.mean((1, 2)) / sigma
    return cam, recon


def main(backbones):
    df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
    y = (df["dx"] == "mel").to_numpy(int)
    groups = df["lesion_id"].to_numpy()
    paths = image_paths(df)
    exp = Experiment("E15_cam_backbones",
                     "Melanoma evidence concentrates on the lesion for every backbone; misses are smaller lesions; "
                     "pixel permutation over-declares significance relative to the exact shift null.",
                     {"source": "HAM10000 + Tschandl lesion masks"},
                     {"backbones": backbones, "map_resolution": RES, "maps": "melanoma evidence (+CAM)"}, SEED)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, y, groups))
    out = {}
    for name in backbones:
        X = np.load(DATA / f"emb_{name}.npy")
        body, gamma, eps = spatial_model(name)
        rng = np.random.default_rng(SEED)
        rows, recon_r = [], []
        for tr, te in folds:
            sc = StandardScaler().fit(X[tr])
            head, _ = fit_tuned(sc.transform(X[tr]), y[tr], groups[tr])
            w_std = head.coef_[0] / sc.scale_
            logit = head.decision_function(sc.transform(X[te]))
            mel = te[y[te] == 1]
            other = rng.choice(te[y[te] == 0], size=len(mel), replace=False)
            sel = np.concatenate([mel, other])
            Fm = maps(body, [paths[i] for i in sel])
            cam, recon = cam_and_check(name, Fm, w_std, gamma, eps)
            pos = {i: j for j, i in enumerate(te)}
            recon_r.append(float(np.corrcoef(recon, [logit[pos[i]] for i in sel])[0, 1]))
            cam = F.interpolate(torch.from_numpy(cam)[:, None].float(), size=RES, mode="bilinear",
                                align_corners=False)[:, 0].numpy()
            for j, i in enumerate(sel):
                mk = np.asarray(Image.fromarray(mask_224(df["image_id"][i]).astype(np.uint8) * 255)
                                .resize((RES, RES), Image.NEAREST)) > 127
                if mk.sum() == 0 or mk.all():
                    continue
                m = overlap_metrics(cam[j], mk)
                _, p_shift = shift_null_exact(cam[j], mk)
                flat = np.clip(cam[j], 0, None).ravel()
                perm = [overlap_metrics(rng.permutation(flat).reshape(RES, RES), mk)["mass_ratio"] for _ in range(N_PERM)]
                rows.append({"y": int(y[i]), "pred": int(logit[pos[i]] > 0), "area": float(mk.mean()),
                             "enrichment": m["enrichment"], "mass_ratio": m["mass_ratio"], "p_shift": p_shift,
                             "p_perm": (1 + np.sum(np.array(perm) >= m["mass_ratio"])) / (N_PERM + 1)})
        r = pd.DataFrame(rows)
        r.to_csv(RESEARCH / f"results/E15_{name}_per_image.csv", index=False)
        mel = r[r.y == 1]
        det, mis = mel[mel.pred == 1], mel[mel.pred == 0]
        out[name] = {
            "n_images": int(len(r)), "cam_logit_reconstruction_r_min": round(min(recon_r), 6),
            "frac_enrichment_gt1": round(float((r.enrichment > 1).mean()), 3),
            "median_enrichment": round(float(r.enrichment.median()), 3),
            "frac_sig_shift": round(float((r.p_shift <= 0.05).mean()), 3),
            "frac_sig_perm": round(float((r.p_perm <= 0.05).mean()), 3),
            "melanoma_sensitivity_at_0": round(float(len(det) / len(mel)), 3),
            "area_detected_vs_missed": [round(float(det.area.median()), 3), round(float(mis.area.median()), 3)],
            "area_mw_p": float(mannwhitneyu(det.area, mis.area).pvalue) if len(det) > 5 else None,
        }
        print(name, out[name], flush=True)
        del body
        if DEV == "mps":
            torch.mps.empty_cache()
    exp.finish(out, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["resnet18", "resnet50", "convnext_tiny"])
