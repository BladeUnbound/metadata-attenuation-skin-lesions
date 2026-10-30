"""E11: Do dominant variance directions of normal patch features hurt
nearest-neighbour anomaly detection (PatchCore-style) on VisA?

Baseline (PatchCore-lite): Wide-ResNet-50 layer2 + layer3 features, 3x3 local
average pooling, layer3 upsampled to 28x28, concatenated (1536-d per patch).
Memory bank: random 10% of training patches (PatchCore uses greedy coreset;
this is a weaker but identical-for-all-variants bank). Patch score = distance
to nearest bank patch; image score = max patch score; pixel map upsampled to
224 and Gaussian-smoothed (sigma 4).
Variants (fitted on training normal patches only):
  raw         Euclidean kNN
  centred     subtract the mean (must equal raw: Euclidean distance is shift-invariant)
  abtt_r      "all-but-the-top": centre, then remove the top r principal components
  whitened    PCA whitening with shrinkage (eigenvalues + 0.1 * mean eigenvalue)
Hypothesis: removing/whitening dominant directions improves image AUROC.
Falsified if the variants are no better than raw across categories.
Protocol: VisA official split_csv/1cls.csv (train = normal only).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision.models as tvm
from PIL import Image
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment

ROOT = RESEARCH / "data/visa/extracted"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SEED = 0
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
VARIANTS = ["raw", "centred", "abtt_1", "abtt_4", "abtt_16", "whitened"]


class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        m = tvm.wide_resnet50_2(weights=tvm.Wide_ResNet50_2_Weights.IMAGENET1K_V1).eval()
        self.stem = torch.nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1)
        self.l2, self.l3 = m.layer2, m.layer3

    @torch.no_grad()
    def forward(self, x):
        f2 = self.l2(self.stem(x))
        f3 = self.l3(f2)
        f2 = F.avg_pool2d(f2, 3, 1, 1)
        f3 = F.interpolate(F.avg_pool2d(f3, 3, 1, 1), size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat([f2, f3], 1)  # (B, 1536, 28, 28)


def load(path, mask=False):
    im = Image.open(path)
    im = im.convert("L" if mask else "RGB").resize((256, 256), Image.NEAREST if mask else Image.BILINEAR)
    im = im.crop((16, 16, 240, 240))
    a = np.asarray(im, np.float32)
    return (a > 0).astype(np.uint8) if mask else a / 255.0


def features(model, paths, batch=16):
    out = []
    for i in range(0, len(paths), batch):
        x = torch.from_numpy(np.stack([load(p) for p in paths[i:i + batch]])).permute(0, 3, 1, 2)
        out.append(model(((x - MEAN) / STD).to(DEV)).cpu())
    f = torch.cat(out)  # (N, C, 28, 28)
    return f.permute(0, 2, 3, 1).reshape(-1, f.shape[1]).numpy(), f.shape[0]


def fit_transform(bank, variant):
    mu = bank.mean(0)
    if variant == "raw":
        return lambda z: z
    if variant == "centred":
        return lambda z: z - mu
    cov = np.cov((bank - mu).T)
    evals, evecs = np.linalg.eigh(cov)
    evals, evecs = evals[::-1], evecs[:, ::-1]
    if variant.startswith("abtt_"):
        r = int(variant.split("_")[1])
        top = evecs[:, :r]
        return lambda z: (z - mu) - ((z - mu) @ top) @ top.T
    w = evecs / np.sqrt(evals + 0.1 * evals.mean())
    return lambda z: (z - mu) @ w


def knn_dist(test, bank, chunk=4096):
    b = torch.from_numpy(bank).to(DEV)
    bn = (b * b).sum(1)
    out = []
    for i in range(0, len(test), chunk):
        t = torch.from_numpy(test[i:i + chunk]).to(DEV)
        d2 = (t * t).sum(1, keepdim=True) + bn[None] - 2 * t @ b.T
        out.append(d2.min(1).values.clamp_min(0).sqrt().cpu())
    return torch.cat(out).numpy()


def run_category(model, cat, split, rng):
    rows = split[split["object"] == cat]
    tr = rows[rows["split"] == "train"]
    te = rows[rows["split"] == "test"]
    t0 = time.perf_counter()
    ftr, _ = features(model, [ROOT / p for p in tr["image"]])
    fte, n_te = features(model, [ROOT / p for p in te["image"]])
    feat_s = time.perf_counter() - t0
    bank = ftr[rng.choice(len(ftr), len(ftr) // 10, replace=False)].astype(np.float32)
    y_img = (te["label"] == "anomaly").to_numpy(int)
    masks = np.stack([load(ROOT / m, mask=True) if isinstance(m, str) and m else np.zeros((224, 224), np.uint8)
                      for m in te["mask"].fillna("")])
    res = {"n_train": int(len(tr)), "n_test": int(n_te), "n_anomalous": int(y_img.sum()),
           "bank_patches": int(len(bank)), "feature_s": round(feat_s, 1)}
    for v in VARIANTS:
        tf = fit_transform(bank, v)
        tb = tf(bank).astype(np.float32)
        t1 = time.perf_counter()
        d = knn_dist(tf(fte).astype(np.float32), tb).reshape(n_te, 28, 28)
        maps = np.stack([gaussian_filter(F.interpolate(torch.from_numpy(m)[None, None], size=224,
                                                       mode="bilinear", align_corners=False)[0, 0].numpy(), 4)
                         for m in d])
        res[v] = {"image_auroc": round(float(roc_auc_score(y_img, maps.reshape(n_te, -1).max(1))), 4),
                  "pixel_auroc": round(float(roc_auc_score(masks.ravel(), maps.ravel())), 4),
                  "score_s": round(time.perf_counter() - t1, 1)}
    return res


def main(categories):
    rng = np.random.default_rng(SEED)
    split = pd.read_csv(ROOT / "split_csv/1cls.csv")
    exp = Experiment("E11_visa_anisotropy",
                     "Removing or whitening dominant variance directions of normal patch features improves "
                     "kNN anomaly detection.",
                     {"source": "VisA (Amazon, CC BY 4.0), official 1cls split", "categories": categories},
                     {"backbone": "wide_resnet50_2 layer2+3, 1536-d", "bank": "random 10%", "k": 1,
                      "variants": VARIANTS, "device": DEV}, SEED)
    model = Extractor().to(DEV)
    res = {}
    for cat in categories:
        res[cat] = run_category(model, cat, split, rng)
        print(cat, {v: res[cat][v]["image_auroc"] for v in VARIANTS}, flush=True)
    res["mean_image_auroc"] = {v: round(float(np.mean([res[c][v]["image_auroc"] for c in categories])), 4) for v in VARIANTS}
    res["mean_pixel_auroc"] = {v: round(float(np.mean([res[c][v]["pixel_auroc"] for c in categories])), 4) for v in VARIANTS}
    print(res["mean_image_auroc"], res["mean_pixel_auroc"], flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["candle", "capsules", "pcb1", "macaroni1"])
