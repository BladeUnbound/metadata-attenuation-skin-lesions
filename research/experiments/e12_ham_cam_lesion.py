"""E12: Does a frozen-backbone melanoma classifier look at the lesion, and do
its errors go with attention that leaves the lesion?

Model: ResNet-50 (ImageNet, frozen) + L2 logistic head on standardised pooled
features, trained out-of-fold on E10's seed-0 lesion-grouped folds (penalty from
grouped inner CV). Because the head is linear on global-average-pooled
features, the class activation map is exact:
    CAM(x) = sum_c (w_c / sigma_c) * A_c(x)      (A = layer4 maps, 7x7)
The predicted-class evidence map is +CAM if the prediction is melanoma, -CAM otherwise.
Maps and expert lesion masks are compared at 56x56 on the same 224 centre crop.
Overlap: mass_ratio and enrichment; p-value from the exact mask-shift null
(every circular shift, via FFT); pixel-permutation p-value shown for contrast.

Hypotheses (written before running):
  H1  median enrichment > 1: evidence concentrates on the lesion more than its area share
  H2  misclassified images have lower enrichment than correctly classified ones
  H3  the pixel-permutation null declares far more images "significant" than the shift null
"""
import os
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
from saliency_overlap import overlap_metrics, shift_null_exact

SEED, RES, N_PERM = 0, 56, 199
# E12b: "fixed" uses the melanoma-evidence map (+CAM) for every image, removing the
# sign confound found in E12 (predicted-negative maps are benign-evidence maps).
MODE = os.environ.get("E12_MODE", "predicted")
EXP_ID = "E12b_ham_cam_fixed_sign" if MODE == "fixed" else "E12_ham_cam_lesion"


def mask_224(image_id):
    p = next((DATA / "masks").rglob(f"{image_id}_segmentation.png"))
    im = Image.open(p).convert("L")
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.NEAREST)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return np.asarray(im.crop((l, t, l + 224, t + 224))) > 127


@torch.no_grad()
def layer4_maps(paths, batch=32):
    m = tvm.resnet50(weights=tvm.ResNet50_Weights.IMAGENET1K_V2).eval().to(DEV)
    body = torch.nn.Sequential(*list(m.children())[:-2])
    out = []
    for i in range(0, len(paths), batch):
        x = torch.stack([torch.from_numpy(np.asarray(load_224(p), np.float32) / 255).permute(2, 0, 1)
                         for p in paths[i:i + batch]])
        out.append(body(((x - MEAN) / STD).to(DEV)).float().cpu())
    return torch.cat(out)  # (N, 2048, 7, 7)


def main():
    df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
    y = (df["dx"] == "mel").to_numpy(int)
    groups = df["lesion_id"].to_numpy()
    X = np.load(DATA / "emb_resnet50.npy")
    paths = image_paths(df)
    rng = np.random.default_rng(SEED)
    hyp = ("H2b: among true melanomas, missed ones have lower melanoma-evidence lesion enrichment than "
           "detected ones; enrichment correlates with the melanoma logit." if MODE == "fixed" else
           "H1 enrichment > 1; H2 errors have lower enrichment; H3 pixel permutation over-declares significance.")
    exp = Experiment(EXP_ID, hyp,
                     {"source": "HAM10000 + Tschandl lesion masks", "n_images": len(df)},
                     {"backbone": "resnet50 frozen", "map_resolution": RES, "null": "exact circular shift (FFT)",
                      "folds": "E10 seed-0"}, SEED)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(df, y, groups))
    rows = []
    for k, (tr, te) in enumerate(folds):
        sc = StandardScaler().fit(X[tr])
        head, _ = fit_tuned(sc.transform(X[tr]), y[tr], groups[tr])
        w = head.coef_[0] / sc.scale_
        logit = head.decision_function(sc.transform(X[te]))
        # Melanomas plus an equal-sized random sample of other lesions from this fold, to bound cost
        mel = te[y[te] == 1]
        other = rng.choice(te[y[te] == 0], size=min(len(mel), (y[te] == 0).sum()), replace=False)
        sel = np.concatenate([mel, other])
        A = layer4_maps([paths[i] for i in sel])
        cam = torch.einsum("c,nchw->nhw", torch.from_numpy(w).float(), A)
        cam = F.interpolate(cam[:, None], size=RES, mode="bilinear", align_corners=False)[:, 0].numpy()
        pos = {i: j for j, i in enumerate(te)}
        for j, i in enumerate(sel):
            pred = int(logit[pos[i]] > 0)
            ev = cam[j] if (pred == 1 or MODE == "fixed") else -cam[j]
            mk = np.asarray(Image.fromarray(mask_224(df["image_id"][i]).astype(np.uint8) * 255)
                            .resize((RES, RES), Image.NEAREST)) > 127
            if mk.sum() == 0 or mk.all():
                continue
            m = overlap_metrics(ev, mk)
            _, p_shift = shift_null_exact(ev, mk)
            flat = np.clip(ev, 0, None).ravel()
            perm = [overlap_metrics(rng.permutation(flat).reshape(RES, RES), mk)["mass_ratio"] for _ in range(N_PERM)]
            p_perm = (1 + np.sum(np.array(perm) >= m["mass_ratio"])) / (N_PERM + 1)
            rows.append({"idx": int(i), "y": int(y[i]), "pred": pred, "correct": int(pred == y[i]),
                         "logit": float(logit[pos[i]]),
                         "lesion_area_frac": float(mk.mean()), "enrichment": m["enrichment"],
                         "mass_ratio": m["mass_ratio"], "p_shift": p_shift, "p_perm": p_perm})
        print(f"fold {k}: {len(rows)} images", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(RESEARCH / f"results/{EXP_ID.split('_')[0]}_per_image.csv", index=False)
    corr, wrong = r[r.correct == 1], r[r.correct == 0]
    mw = mannwhitneyu(corr.enrichment, wrong.enrichment, alternative="greater")
    res = {
        "n_images": int(len(r)), "n_correct": int(len(corr)), "n_wrong": int(len(wrong)),
        "H1_median_enrichment": round(float(r.enrichment.median()), 3),
        "H1_frac_enrichment_gt1": round(float((r.enrichment > 1).mean()), 3),
        "H2_median_enrichment_correct": round(float(corr.enrichment.median()), 3),
        "H2_median_enrichment_wrong": round(float(wrong.enrichment.median()), 3),
        "H2_mannwhitney_p_one_sided": float(mw.pvalue),
        "H3_frac_sig_shift_null": round(float((r.p_shift <= 0.05).mean()), 3),
        "H3_frac_sig_pixel_perm": round(float((r.p_perm <= 0.05).mean()), 3),
        "median_lesion_area_frac": round(float(r.lesion_area_frac.median()), 3),
        "by_true_class": {int(c): {"median_enrichment": round(float(g.enrichment.median()), 3),
                                   "accuracy": round(float(g.correct.mean()), 3)} for c, g in r.groupby("y")},
    }
    if MODE == "fixed":
        from scipy.stats import spearmanr
        mel = r[r.y == 1]
        det, missed = mel[mel.pred == 1], mel[mel.pred == 0]
        res["H2b"] = {"n_detected": int(len(det)), "n_missed": int(len(missed)),
                      "median_enrichment_detected": round(float(det.enrichment.median()), 3),
                      "median_enrichment_missed": round(float(missed.enrichment.median()), 3),
                      "mannwhitney_p_one_sided": float(mannwhitneyu(det.enrichment, missed.enrichment,
                                                                    alternative="greater").pvalue),
                      "spearman_enrichment_vs_logit_melanomas": round(float(spearmanr(mel.enrichment, mel.logit)[0]), 3),
                      "spearman_enrichment_vs_logit_nonmelanomas": round(float(spearmanr(r[r.y == 0].enrichment,
                                                                                         r[r.y == 0].logit)[0]), 3)}
    print(res, flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
