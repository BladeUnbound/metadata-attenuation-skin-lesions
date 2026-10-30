"""E02c: visual check of nearest neighbours, plus near-copy rates by threshold.

Run after e02_leakage.py and e02b_leakage_centered.py (uses their caches).
Saves two contact sheets to research/results/: random validation images with
their nearest same-class and different-class training images, and borderline
pairs (similarity 0.93-0.96). Inspect them by eye before trusting a threshold.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e02_leakage import list_images
from e02b_leakage_centered import centre, nn_sims, split

THRESHOLDS = (0.90, 0.93, 0.95, 0.97, 0.99)


def sheet(files, labels, E, tr, picks, path, with_other):
    W = 128
    cols = 3 if with_other else 2
    img = Image.new("L", (cols * (W + 10), len(picks) * (W + 14)), 255)
    d = ImageDraw.Draw(img)
    sim = E[picks] @ E[tr].T
    for r, i in enumerate(picks):
        same = labels[tr] == labels[i]
        s_same, s_other = np.where(same, sim[r], -9), np.where(~same, sim[r], -9)
        cells = [(i, f"val {labels[i][:8]}"), (tr[s_same.argmax()], f"same {s_same.max():.3f}")]
        if with_other:
            cells.append((tr[s_other.argmax()], f"other {s_other.max():.3f}"))
        for c, (idx, txt) in enumerate(cells):
            img.paste(Image.open(files[idx]).convert("L").resize((W, W)), (c * (W + 10), r * (W + 14) + 12))
            d.text((c * (W + 10) + 2, r * (W + 14)), txt, fill=0)
    img.save(path)


def main():
    files, labels = list_images()
    labels = np.array(labels)
    P = np.load(RESEARCH / "data/e02b_pixels.npz")["P"]
    E = centre(P, P.mean(0)).astype(np.float32)
    val = split(labels, np.random.default_rng(42))
    tr, va = np.flatnonzero(~val), np.flatnonzero(val)
    exp = Experiment("E02c_visual_check",
                     "Pixel similarity >= ~0.93 corresponds to visually identical slices.",
                     {"source": "Kaggle dataset (as E02)"}, {"embedding": "centred 32x32 pixels"}, 42)
    same, other = nn_sims(E, labels, val)
    sheet(files, labels, E, tr, np.random.default_rng(7).choice(va, 8, replace=False),
          RESEARCH / "results/E02c_random_pairs.png", True)
    band = np.flatnonzero((same >= 0.93) & (same < 0.96))
    sheet(files, labels, E, tr, va[np.random.default_rng(3).choice(band, 8, replace=False)],
          RESEARCH / "results/E02c_borderline_pairs.png", False)
    res = {f"thr_{t}": {"val_with_same_class_nn_above": round(float(np.mean(same >= t)), 4),
                        "other_class_nn_above": round(float(np.mean(other >= t)), 4)} for t in THRESHOLDS}
    res["per_class_at_0.95"] = {c: round(float(np.mean(same[labels[val] == c] >= 0.95)), 3)
                                for c in np.unique(labels)}
    exp.finish(res, interpretation="Visual inspection (26 Oct 2026): all 8 borderline pairs at 0.93-0.96 "
               "and 6/8 random pairs were the same slice; different-class neighbours were all different brains.")


if __name__ == "__main__":
    main()
