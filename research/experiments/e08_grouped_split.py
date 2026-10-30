"""E08: How much does the image-level split inflate accuracy?

Classifier features: E02's cached frozen VGG16 features (512-d, standardised).
Grouping features: E02b's centred 32x32 pixel embedding, where similarity
>= 0.93 was visually confirmed as "same slice, altered contrast/brightness"
(E02c: 16 pairs inspected; 0.0% of different-class neighbours reach 0.93).
Near-copies are linked within each class; connected components approximate "same source
scan" groups. The same linear classifier is evaluated with
  (a) a random stratified image-level 5-fold split (what the notebook did)
  (b) a group-level 5-fold split (no group on both sides)
This measures leakage for a linear head on frozen features, not for the exact
original model; it bounds how much of the original 86% could be leakage.
Two thresholds are reported because single-linkage grouping can chain.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e02_leakage import list_images

SEED = 0
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def components(E, labels, thr, block=2048):
    Et = torch.from_numpy(E).to(DEV)
    rows, cols = [], []
    for i in range(0, len(E), block):
        sim = Et[i:i + block] @ Et.T
        r, c = torch.nonzero(sim >= thr, as_tuple=True)
        r = r.cpu().numpy() + i
        c = c.cpu().numpy()
        keep = (r != c) & (labels[r] == labels[c])
        rows.append(r[keep])
        cols.append(c[keep])
    r, c = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(r)), (r, c)), shape=(len(E), len(E)))
    return connected_components(graph, directed=False)[1]


def cv_accuracy(E, y, splitter, groups=None):
    accs = []
    for tr, te in splitter.split(E, y, groups):
        clf = LogisticRegression(max_iter=3000, C=1.0).fit(E[tr], y[tr])
        accs.append(float((clf.predict(E[te]) == y[te]).mean()))
    return accs


def main():
    _, labels = list_images()
    labels = np.array(labels)
    classes, y = np.unique(labels, return_inverse=True)
    E = np.load(RESEARCH / "data/e02_embeddings.npz")["E"].astype(np.float32)
    E = (E - E.mean(0)) / E.std(0)
    P = np.load(RESEARCH / "data/e02b_pixels.npz")["P"]
    P = P - P.mean(0)
    P = (P / np.linalg.norm(P, axis=1, keepdims=True)).astype(np.float32)
    exp = Experiment(
        "E08_grouped_split",
        "Accuracy under an image-level split is substantially higher than under a split that keeps "
        "near-copies together.",
        {"source": "Kaggle dataset, E02 embeddings", "n_images": int(len(y))},
        {"classifier": "multinomial logistic regression on 512-d frozen VGG16 features", "folds": 5,
         "grouping": "centred 32x32 pixels, single linkage within class", "thresholds": [0.93, 0.97]},
        SEED,
    )
    res = {"image_level": cv_accuracy(E, y, StratifiedKFold(5, shuffle=True, random_state=SEED))}
    for name, thr in (("grouped_thr_0.93", 0.93), ("grouped_thr_0.97", 0.97)):
        g = components(P, y, thr)
        sizes = np.bincount(g)
        res[name] = {
            "threshold": thr,
            "n_groups": int(len(sizes)),
            "largest_group_frac": round(float(sizes.max() / len(y)), 4),
            "groups_per_class": {c: int(len(np.unique(g[y == k]))) for k, c in enumerate(classes)},
            "acc": cv_accuracy(E, y, StratifiedGroupKFold(5, shuffle=True, random_state=SEED), g),
        }
        print(name, {k: v for k, v in res[name].items() if k != "acc"}, np.mean(res[name]["acc"]), flush=True)
    summary = {k: round(float(np.mean(v if isinstance(v, list) else v["acc"])), 4) for k, v in res.items()}
    exp.finish({"mean_accuracy": summary, "detail": res}, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
