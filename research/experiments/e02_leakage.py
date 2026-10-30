"""E02: How much does the image-level split leak in the Kaggle dataset?

Question: in a random 80/20 image-level split (as in the notebook), does a
validation image typically have a near-copy of the same underlying scan in
the training set?

Method:
  1. Embed every image with the frozen ImageNet VGG16 convolutional features
     from the trained model (global average pooled, 512-d, L2-normalised).
  2. Split 80/20 stratified (seed 42), as the notebook did. The exact notebook
     split can't be reproduced because directory listing order differs between
     systems; any seed is statistically equivalent.
  3. For each validation image, find its nearest training image within the
     same class and within other classes. Images in different classes come
     from different patients, so the cross-class nearest neighbour gives a
     reference for "similar but not the same scan".
  4. Calibrate a "same source" similarity by augmenting a sample of images
     ourselves (flip, small rotation, brightness, contrast) and measuring how
     similar each image stays to its own augmented copy.
  5. Exact duplicates are counted separately by file hash.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageEnhance

sys.path.insert(0, str(Path(__file__).parent))
from common import PROJECT, RESEARCH, Experiment
from vgg_port import load_from_h5

SEED = 42
DATA = RESEARCH / "data/kaggle_alz/extracted"
CACHE = RESEARCH / "data/e02_embeddings.npz"
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def list_images():
    files = sorted(p for p in DATA.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = [p.parent.name for p in files]
    return files, labels


def to_tensor(img):
    a = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


@torch.no_grad()
def embed(model, images, batch=64):
    out = []
    for i in range(0, len(images), batch):
        x = torch.stack([to_tensor(im) for im in images[i:i + batch]]).to(DEV)
        f = model.features(x).mean((2, 3))
        out.append(torch.nn.functional.normalize(f, dim=1).cpu())
    return torch.cat(out).numpy()


def augment(img, rng):
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    img = img.rotate(float(rng.uniform(-10, 10)))
    img = ImageEnhance.Brightness(img).enhance(float(rng.uniform(0.9, 1.1)))
    return ImageEnhance.Contrast(img).enhance(float(rng.uniform(0.9, 1.1)))


def nearest(sim_rows, cols_mask):
    s = np.where(cols_mask, sim_rows, -np.inf)
    return s.max(1)


def main():
    rng = np.random.default_rng(SEED)
    files, labels = list_images()
    labels = np.array(labels)
    exp = Experiment(
        "E02_leakage",
        "Most validation images have a near-copy of the same scan in the training set under an image-level split.",
        {"source": "Kaggle aryansinghal10/alzheimers-multiclass-dataset-equal-and-augmented (Apache-2.0)",
         "n_images": len(files), "classes": {c: int((labels == c).sum()) for c in np.unique(labels)}},
        {"embedding": "frozen VGG16 conv features of alzheimer_cnn.h5, GAP 512-d", "device": DEV},
        SEED,
    )
    hashes = [hashlib.md5(p.read_bytes()).hexdigest() for p in files]
    _, counts = np.unique(hashes, return_counts=True)
    exact_dupe_images = int(counts[counts > 1].sum() - (counts > 1).sum())

    model = load_from_h5(PROJECT / "alzheimer_cnn.h5").to(DEV)
    if CACHE.exists():
        E = np.load(CACHE)["E"]
    else:
        E = np.concatenate([embed(model, [Image.open(p) for p in files[i:i + 2048]])
                            for i in range(0, len(files), 2048)])
        np.savez_compressed(CACHE, E=E)

    # Stratified 80/20 split
    val = np.zeros(len(files), bool)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        val[rng.choice(idx, int(round(0.2 * len(idx))), replace=False)] = True
    tr_idx, va_idx = np.flatnonzero(~val), np.flatnonzero(val)
    Et = torch.from_numpy(E[tr_idx]).to(DEV)
    same, other = [], []
    for i in range(0, len(va_idx), 1024):
        rows = va_idx[i:i + 1024]
        sim = (torch.from_numpy(E[rows]).to(DEV) @ Et.T).cpu().numpy()
        same_class = labels[rows][:, None] == labels[tr_idx][None, :]
        same.append(nearest(sim, same_class))
        other.append(nearest(sim, ~same_class))
    same, other = np.concatenate(same), np.concatenate(other)

    # Same-source calibration: similarity of an image to its own augmented copy
    sample = rng.choice(len(files), 500, replace=False)
    originals = [Image.open(files[i]).convert("RGB") for i in sample]
    E_orig = embed(model, originals)
    E_aug = embed(model, [augment(im, rng) for im in originals])
    self_sim = (E_orig * E_aug).sum(1)
    thr = float(np.percentile(self_sim, 5))  # 95% of true augmented copies are above this

    per_class = {}
    for c in np.unique(labels):
        m = labels[va_idx] == c
        per_class[c] = {
            "median_nn_sim_same_class": round(float(np.median(same[m])), 4),
            "median_nn_sim_other_class": round(float(np.median(other[m])), 4),
            "frac_val_with_near_copy_in_train": round(float((same[m] >= thr).mean()), 3),
            "frac_other_class_above_threshold": round(float((other[m] >= thr).mean()), 3),
        }
    results = {
        "exact_duplicate_images": exact_dupe_images,
        "self_augmentation_similarity_p5_p50": [round(thr, 4), round(float(np.median(self_sim)), 4)],
        "overall_frac_val_with_near_copy_in_train": round(float((same >= thr).mean()), 3),
        "overall_frac_other_class_above_threshold": round(float((other >= thr).mean()), 3),
        "per_class": per_class,
    }
    exp.finish(results, interpretation="See research/EXPERIMENTS.md.",
               next_step="If leakage is confirmed, the 86% cannot be reported as generalisation; "
                         "re-evaluate only with source-level grouping.")


if __name__ == "__main__":
    main()
