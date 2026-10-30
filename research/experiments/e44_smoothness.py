"""E44 (exploratory; registered in research/iteration5/REGISTRY.md, 21:59 IST): smoothness of per-observation
metadata benefit Δ_i on the image-representation graph.

Graph (transductive, descriptive): all images of one dataset, cosine kNN (k = 20) on standardised, L2-normalised
features, same-group neighbours excluded, symmetrised by union, Gaussian weights with sigma = median 20-NN distance.
Signal: E40 corrected Δ_i (fold assignment 0), mean-centred.
Energies: E = ΔᵀLΔ (L = D − W); normalised ΔᵀLΔ / ΔᵀDΔ (primary); symmetric-normalised Δᵀ(I − D^-1/2 W D^-1/2)Δ / ΔᵀΔ.
Nulls (R = 1,000): (i) free permutation of Δ over nodes; (ii) permutation within cells of label × decile of h(q0).
Lower energy than null (ii) is the only result read as local organisation beyond label and image uncertainty.

Usage: python e44_smoothness.py <dataset> [...]
"""
import sys
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e42_graph import h2, knn

RUNGS = ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
K, R = 20, 1000


def graph(X, groups):
    Z = StandardScaler().fit_transform(X).astype(np.float32)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
    gcode = np.unique(groups, return_inverse=True)[1]
    nb, d = knn(Z, Z, K, gcode, gcode)
    sigma = float(np.median(d.max(1)))
    n = len(Z)
    W = coo_matrix((np.exp(-d.ravel() ** 2 / (2 * sigma ** 2)), (np.repeat(np.arange(n), K), nb.ravel())), shape=(n, n)).tocsr()
    W = W.maximum(W.T).tocoo()  # union symmetrisation
    keep = W.row < W.col
    return W.row[keep], W.col[keep], W.data[keep], np.asarray(W.sum(1)).ravel(), connected_components(W)[0]


def energies(x, i, j, w, deg):
    x = x - x.mean()
    e = np.sum(w * (x[i] - x[j]) ** 2)  # = xᵀLx (each undirected edge once)
    xs = x / np.sqrt(deg)
    e_sym = np.sum(w * (xs[i] - xs[j]) ** 2) / np.sum(x ** 2)
    return np.array([e, e / np.sum(deg * x ** 2), e_sym])


def run(ds):
    out = {}
    for rung in RUNGS:
        f = np.load(RESEARCH / f"data/{ds}/e40/{rung}_s0.npz")
        y, groups, delta, q0 = f["y"], f["groups"], f["delta"], np.clip(f["q0"], 1e-9, 1 - 1e-9)
        i, j, w, deg, ncomp = graph(np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy"), groups)
        obs = energies(delta, i, j, w, deg)
        hq = h2(q0)
        dec = np.digitize(hq, np.quantile(hq, np.linspace(0.1, 0.9, 9)))
        cells = [np.where((y == c) & (dec == b))[0] for c in (0, 1) for b in range(10)]
        rng = np.random.default_rng(44)
        null_free, null_strat = [], []
        for _ in range(R):
            null_free.append(energies(rng.permutation(delta), i, j, w, deg))
            p = delta.copy()
            for c in cells:
                if len(c) > 1:
                    p[c] = delta[rng.permutation(c)]
            null_strat.append(energies(p, i, j, w, deg))
        nf, ns = np.array(null_free), np.array(null_strat)
        names = ["unnormalised", "normalised", "sym_normalised"]
        r = {"n_components": int(ncomp), "n_edges": int(len(w))}
        for k, nm in enumerate(names):
            r[nm] = {"observed": float(obs[k]),
                     "ratio_free": round(float(obs[k] / nf[:, k].mean()), 4),
                     "p_free": float((1 + np.sum(nf[:, k] <= obs[k])) / (R + 1)),
                     "ratio_stratified": round(float(obs[k] / ns[:, k].mean()), 4),
                     "p_stratified": float((1 + np.sum(ns[:, k] <= obs[k])) / (R + 1))}
        # reference signals under the free null (how smooth label and image uncertainty are on the same graph)
        ref = {}
        for nm, sig in {"label": y.astype(float), "h_q0": hq}.items():
            o = energies(sig, i, j, w, deg)[1]
            nul = np.mean([energies(rng.permutation(sig), i, j, w, deg)[1] for _ in range(100)])
            ref[nm] = round(float(o / nul), 4)
        r["reference_ratio_free_normalised"] = ref
        out[rung] = r
        print(ds, rung, {k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in ("ratio_free", "p_free", "ratio_stratified", "p_stratified")} if "ratio_free" in v else v) for k, v in r.items()}, flush=True)
    return out


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        exp = Experiment(f"E44_{ds}", "REGISTRY.md E44 (exploratory): smoothness of Δ_i on the representation graph",
                         {"dataset": ds}, {"k": K, "R": R, "graph": "transductive, same-group edges removed, union symmetrised",
                                           "signal": "E40 corrected Δ_i, fold assignment 0"}, 0)
        exp.finish(run(ds), interpretation="See research/EXPERIMENTS.md.")
