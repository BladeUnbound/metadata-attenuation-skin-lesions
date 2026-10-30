"""E14: does attenuation hold when capacity comes from fine-tuning? (closes review item M2)

HAM10000, melanoma vs rest, E10's seed-0 lesion-grouped 5-fold split.
Models: ResNet-18 and ConvNeXt-T, ImageNet-initialised, all layers trained.
Fixed schedule, chosen before running and never tuned on test folds:
  AdamW, lr 1e-4, weight decay 0.05, 6 epochs, cosine decay, batch 32,
  plain BCE; augmentation: random resized crop (scale 0.7-1) from the 256-px
  image, horizontal and vertical flips. Evaluation: 224 centre crop.
Each fold saves its out-of-fold logits, so the run resumes after interruption.
Then: added value of metadata on the fine-tuned scores (same second stage as E10),
lesion-bootstrap CI, and paired differences vs the frozen versions of the same backbone.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as tvm
import torchvision.transforms.v2 as T
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import lesion_bootstrap, meta_matrix, oof_scores, second_stage
from e10_ham_embed import DATA, DEV, MEAN, STD, image_paths

OUT = RESEARCH / "data/ham10000/e14"
EPOCHS, LR, WD, BATCH = 6, 1e-4, 0.05, 32


def build(name):
    if name == "resnet18":
        m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(m.fc.in_features, 1)
    else:
        m = tvm.convnext_tiny(weights=tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        m.classifier[2] = nn.Linear(m.classifier[2].in_features, 1)
    return m.to(DEV)


def load_256(p):
    im = Image.open(p).convert("RGB")
    w, h = im.size
    s = 256 / min(w, h)
    return np.asarray(im.resize((round(w * s), round(h * s)), Image.BILINEAR))


TRAIN_TF = T.Compose([T.RandomResizedCrop(224, scale=(0.7, 1.0), antialias=True),
                      T.RandomHorizontalFlip(), T.RandomVerticalFlip()])


def to_batch(arrs, train):
    xs = []
    for a in arrs:
        x = torch.from_numpy(a).permute(2, 0, 1).float() / 255
        x = TRAIN_TF(x) if train else T.functional.center_crop(x, 224)
        xs.append(x)
    return ((torch.stack(xs) - MEAN) / STD).to(DEV)


def train_fold(name, imgs, y, tr, te, seed):
    torch.manual_seed(seed)
    model = build(name)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    steps = EPOCHS * int(np.ceil(len(tr) / BATCH))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(EPOCHS):
        model.train()
        order = rng.permutation(tr)
        t0, tot = time.perf_counter(), 0.0
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            x = to_batch([imgs[k] for k in idx], train=True)
            t = torch.from_numpy(y[idx]).float().to(DEV)[:, None]
            loss = lossf(model(x), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
        print(f"  {name} epoch {ep + 1}/{EPOCHS} loss {tot / len(tr):.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    model.eval()
    logits = []
    with torch.no_grad():
        for i in range(0, len(te), 64):
            logits.append(model(to_batch([imgs[k] for k in te[i:i + 64]], train=False)).float().cpu().numpy()[:, 0])
    return np.concatenate(logits)


def main(backbones):
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
    y = (df["dx"] == "mel").to_numpy(int)
    groups = df["lesion_id"].to_numpy()
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    exp = Experiment("E14_finetune", "Fine-tuned models extract more than frozen ones; metadata added value on "
                     "fine-tuned scores is no larger than on the frozen versions (attenuation holds).",
                     {"source": "HAM10000", "split": "E10 seed-0 lesion-grouped 5-fold"},
                     {"epochs": EPOCHS, "lr": LR, "weight_decay": WD, "batch": BATCH, "device": DEV,
                      "aug": "RandomResizedCrop(0.7-1) + H/V flips", "eval": "centre crop 224"}, 0)
    t0 = time.perf_counter()
    imgs = [load_256(p) for p in image_paths(df)]
    print(f"loaded {len(imgs)} images in {time.perf_counter() - t0:.0f}s", flush=True)
    M = np.zeros((len(df), meta_matrix(df, folds[0][0]).shape[1]))
    for tr, te in folds:
        M[te] = meta_matrix(df, tr)[te]
    rng = np.random.default_rng(14)
    res = {}
    for name in backbones:
        score = np.zeros(len(df))
        t1 = time.perf_counter()
        for k, (tr, te) in enumerate(folds):
            ck = OUT / f"{name}_fold{k}.npy"
            if ck.exists():
                score[te] = np.load(ck)
                print(f"{name} fold {k}: resumed from checkpoint", flush=True)
                continue
            score[te] = train_fold(name, imgs, y, tr, te, seed=k)
            np.save(ck, score[te])
            print(f"{name} fold {k} done; fold AUROC {roc_auc_score(y[te], score[te]):.3f}", flush=True)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), y, folds)
        d_ft = l0 - l1
        frozen, _ = oof_scores(np.load(DATA / f"emb_{name}.npy"), y, groups, folds)
        f0, f1 = second_stage(frozen[:, None], np.column_stack([frozen, M]), y, folds)
        d_fr = f0 - f1
        diff = d_fr - d_ft
        res[name] = {
            "finetuned_auroc": round(float(roc_auc_score(y, score)), 4),
            "frozen_auroc": round(float(roc_auc_score(y, frozen)), 4),
            "finetuned_image_logloss": round(float(l0.mean()), 4),
            "added_value_finetuned": round(float(d_ft.mean()), 5),
            "added_value_finetuned_ci95": [round(v, 5) for v in lesion_bootstrap(d_ft, groups, rng)],
            "added_value_frozen": round(float(d_fr.mean()), 5),
            "frozen_minus_finetuned_av": round(float(diff.mean()), 5),
            "frozen_minus_finetuned_ci95": [round(v, 5) for v in lesion_bootstrap(diff, groups, rng)],
            "train_minutes": round((time.perf_counter() - t1) / 60, 1),
        }
        print(name, res[name], flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main(sys.argv[1:] or ["resnet18", "convnext_tiny"])
