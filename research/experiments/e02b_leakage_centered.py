"""E02b: E02 failed because its embedding couldn't discriminate (cross-class
neighbours were as similar as augmented copies, cosine ~0.997). This rerun:
  1. Centres the cached VGG embeddings (subtract the dataset mean, renormalise),
     removing the component every brain slice shares.
  2. Adds an independent pixel-space embedding: 32x32 grayscale, z-scored per
     image, centred, L2-normalised.
  3. Checks discrimination first: an embedding is only used if self-augmented
     copies are clearly more similar than different-class nearest neighbours
     (reported as an AUC; must exceed 0.95).
  4. Counts how many exact-duplicate files fall on both sides of a random split.
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from common import PROJECT, RESEARCH, Experiment
from e02_leakage import DEV, augment, embed, list_images, nearest
from vgg_port import load_from_h5

SEED = 42
PIX_CACHE = RESEARCH / "data/e02b_pixels.npz"


def pixel_embed(images):
    a = np.stack([np.asarray(im.convert("L").resize((32, 32), Image.BILINEAR), np.float32).ravel()
                  for im in images])
    a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
    return a


def centre(E, mean):
    C = E - mean
    return C / np.linalg.norm(C, axis=1, keepdims=True)


def split(labels, rng):
    val = np.zeros(len(labels), bool)
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        val[rng.choice(idx, int(round(0.2 * len(idx))), replace=False)] = True
    return val


def nn_sims(E, labels, val):
    tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    Et = torch.from_numpy(E[tr]).to(DEV)
    same, other = [], []
    for i in range(0, len(va), 1024):
        rows = va[i:i + 1024]
        sim = (torch.from_numpy(E[rows]).to(DEV) @ Et.T).cpu().numpy()
        sc = labels[rows][:, None] == labels[tr][None, :]
        same.append(nearest(sim, sc))
        other.append(nearest(sim, ~sc))
    return np.concatenate(same), np.concatenate(other)


def main():
    rng = np.random.default_rng(SEED)
    files, labels = list_images()
    labels = np.array(labels)
    exp = Experiment(
        "E02b_leakage_centered",
        "With a discriminative embedding, most validation images have a near-copy in training "
        "under an image-level split.",
        {"source": "Kaggle dataset (as E02)", "n_images": len(files)},
        {"embeddings": ["centred VGG16 GAP (cached from E02)", "32x32 grayscale pixels, centred"],
         "discrimination_gate_auc": 0.95},
        SEED,
    )
    hashes = np.array([hashlib.md5(p.read_bytes()).hexdigest() for p in files])
    val = split(labels, rng)
    tr_hashes = set(hashes[~val])
    exact_cross = int(sum(h in tr_hashes for h in hashes[val]))

    E_raw = np.load(RESEARCH / "data/e02_embeddings.npz")["E"]
    if PIX_CACHE.exists():
        P_raw = np.load(PIX_CACHE)["P"]
    else:
        P_raw = np.concatenate([pixel_embed([Image.open(p) for p in files[i:i + 4096]])
                                for i in range(0, len(files), 4096)])
        np.savez_compressed(PIX_CACHE, P=P_raw)

    sample = rng.choice(len(files), 500, replace=False)
    originals = [Image.open(files[i]).convert("RGB") for i in sample]
    augmented = [augment(im, rng) for im in originals]
    model = load_from_h5(PROJECT / "alzheimer_cnn.h5").to(DEV)
    vgg_pair = (embed(model, originals), embed(model, augmented))
    pix_pair = (pixel_embed(originals), pixel_embed(augmented))

    results = {"exact_duplicate_val_images_with_copy_in_train": exact_cross,
               "val_size": int(val.sum())}
    for name, raw, (a, b) in (("vgg_centred", E_raw, vgg_pair), ("pixels_centred", P_raw, pix_pair)):
        mean = raw.mean(0)
        E = centre(raw, mean).astype(np.float32)
        self_sim = (centre(a, mean) * centre(b, mean)).sum(1)
        same, other = nn_sims(E, labels, val)
        disc_auc = roc_auc_score(np.r_[np.ones(len(self_sim)), np.zeros(len(other))], np.r_[self_sim, other])
        thr = float(np.percentile(self_sim, 5))
        results[name] = {
            "discrimination_auc_selfaug_vs_otherclass_nn": round(float(disc_auc), 4),
            "passes_gate": bool(disc_auc > 0.95),
            "threshold_selfaug_p5": round(thr, 4),
            "median_nn_same_class": round(float(np.median(same)), 4),
            "median_nn_other_class": round(float(np.median(other)), 4),
            "frac_val_near_copy_in_train": round(float((same >= thr).mean()), 4),
            "frac_other_class_above_thr": round(float((other >= thr).mean()), 4),
        }
        print(name, results[name], flush=True)
    exp.finish(results, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
