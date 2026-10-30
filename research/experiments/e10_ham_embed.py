"""E10 step 1: cache image representations for every rung of the capacity ladder.

Rungs: HSV colour histogram (48-d) and frozen ImageNet backbones (ResNet-18,
ResNet-50, ConvNeXt-Tiny, ViT-B/16). Resize shorter side to 224, centre-crop
224, ImageNet normalisation. Features are the backbone's pooled pre-classifier
output. Cached to research/data/ham10000/emb_<rung>.npy in metadata row order.
Also records throughput on the M5 (MPS) per backbone.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.models as tvm
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment

DATA = RESEARCH / "data/ham10000"
DEV_RUN = os.environ.get("HAM_DEV") == "1"  # pipeline check on partially downloaded images; not results
IMG_DIR = DATA / ("dev_partial" if DEV_RUN else "images")
TAG = "dev_" if DEV_RUN else ""
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BACKBONES = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2),
    "convnext_tiny": (tvm.convnext_tiny, tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1),
    "vit_b_16": (tvm.vit_b_16, tvm.ViT_B_16_Weights.IMAGENET1K_V1),
}


def image_paths(meta):
    index = {p.stem: p for p in IMG_DIR.rglob("*.jpg")}
    missing = [i for i in meta["image_id"] if i not in index]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing, e.g. {missing[:3]}")
    return [index[i] for i in meta["image_id"]]


def load_224(path):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((round(w * s), round(h * s)), Image.BILINEAR)
    w, h = im.size
    l, t = (w - 224) // 2, (h - 224) // 2
    return im.crop((l, t, l + 224, t + 224))


def hsv_hist(im):
    hsv = np.asarray(im.convert("HSV"))
    return np.concatenate([np.histogram(hsv[..., c], bins=16, range=(0, 256), density=True)[0] for c in range(3)])


def headless(name):
    fn, w = BACKBONES[name]
    model = fn(weights=w)
    if name.startswith("resnet"):
        model.fc = torch.nn.Identity()
    elif name.startswith("convnext"):
        model.classifier[2] = torch.nn.Identity()
    else:
        model.heads = torch.nn.Identity()
    return model.eval().to(DEV)


@torch.no_grad()
def embed_all(model, images, batch=64):
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([torch.from_numpy(np.asarray(im, np.float32) / 255).permute(2, 0, 1)
                         for im in images[i:i + batch]])
        x = ((x - MEAN) / STD).to(DEV)
        out.append(model(x).float().cpu())
    return torch.cat(out).numpy()


def main():
    meta = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
    if DEV_RUN:
        have = {p.stem for p in IMG_DIR.rglob("*.jpg")}
        meta = meta[meta["image_id"].isin(have)].reset_index(drop=True)
        ok = []
        for p in image_paths(meta):
            try:
                Image.open(p).load()
                ok.append(True)
            except Exception:
                ok.append(False)
        meta = meta[ok].reset_index(drop=True)
        meta.to_csv(DATA / "meta_dev.csv", index=False)
    exp = Experiment(f"E10_embed{'_DEV' if DEV_RUN else ''}", "Cache ladder representations; measure M5 throughput.",
                     {"source": "HAM10000", "n_images": len(meta)},
                     {"input": "shorter side 224, centre crop 224, ImageNet norm", "device": DEV}, 0)
    t0 = time.perf_counter()
    images = [load_224(p) for p in image_paths(meta)]
    res = {"decode_resize_s": round(time.perf_counter() - t0, 1)}
    np.save(DATA / f"emb_{TAG}hsv.npy", np.stack([hsv_hist(im) for im in images]).astype(np.float32))
    for name in (["resnet18", "resnet50"] if DEV_RUN else BACKBONES):
        model = headless(name)
        embed_all(model, images[:64])  # warm-up
        if DEV == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        E = embed_all(model, images)
        if DEV == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t
        np.save(DATA / f"emb_{TAG}{name}.npy", E.astype(np.float32))
        res[name] = {"dim": int(E.shape[1]), "images_per_s": round(len(images) / dt, 1),
                     "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1)}
        print(name, res[name], flush=True)
        del model
        if DEV == "mps":
            torch.mps.empty_cache()
    exp.finish(res, interpretation="Embeddings cached.")


if __name__ == "__main__":
    main()
