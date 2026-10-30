"""E49 (exploratory; registered in research/iteration5/REGISTRY.md, 2026-10-29 09:40 IST).

(a) Entropy-efficiency decomposition Δ = H(Y|S)·η with grouped-bootstrap uncertainty for the share of the
    HSV -> ConvNeXt-T log-attenuation due to image uncertainty, s_H = log(H_hsv/H_cnx) / log(Δ_hsv/Δ_cnx).
(b) Record of the E42 post hoc ranking diagnostics (first computed 2026-10-28; post hoc).
"""
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import per_image_ll

DATASETS = ["ham10000", "isic2020", "isic2019_nonham", "pad_ufes", "hiba", "msk2020", "milk10k_derm", "milk10k_clin"]
B = 2000


def h2(p):
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def decomposition(ds):
    out = {}
    for seed in [0, 1, 2]:
        a, b = (np.load(RESEARCH / f"data/{ds}/e40/{r}_s{seed}.npz") for r in ["hsv", "convnext_tiny"])
        y, g = a["y"], a["groups"]
        la, lb = per_image_ll(y, a["q0"]), per_image_ll(y, b["q0"])
        da, db = a["delta"], b["delta"]
        share = lambda Ha, Hb, Da, Db: np.log(Ha / Hb) / np.log(Da / Db)
        r = {"H_hsv": float(la.mean()), "H_cnx": float(lb.mean()), "D_hsv": float(da.mean()), "D_cnx": float(db.mean()),
             "eta_hsv": float(da.mean() / la.mean()), "eta_cnx": float(db.mean() / lb.mean())}
        r["share_H"] = float(share(la.mean(), lb.mean(), da.mean(), db.mean())) if db.mean() > 0 else None
        if seed == 0:
            codes, inv = np.unique(g, return_inverse=True)
            idx = np.random.default_rng(49).integers(0, len(codes), (B, len(codes)))
            cnt = np.bincount(inv)[idx].sum(1)
            m = lambda v: np.bincount(inv, weights=v)[idx].sum(1) / cnt
            Ha, Hb, Da, Db = m(la), m(lb), m(da), m(db)
            ok = (Db > 0) & (Da > Db)
            s = share(Ha[ok], Hb[ok], Da[ok], Db[ok])
            r["share_H_ci95"] = [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))]
            r["bootstrap_excluded_share"] = float(1 - ok.mean())
        out[seed] = r
    return out


def posthoc(ds):
    e = np.load(RESEARCH / f"data/{ds}/e40/convnext_tiny_s0.npz")
    g = np.load(RESEARCH / f"data/{ds}/e40/e42_convnext_tiny.npz")
    D = e["delta"]
    preds = {"K": e["K"], "h_q0": h2(e["q0"]), "B1": g["pred_B1"], "B1G": g["pred_B1G"], "G": g["G"], "LE": g["LE"]}
    r = {k: round(float(spearmanr(v, D)[0]), 4) for k, v in preds.items()}
    o = np.argsort(-e["K"])
    r["K_top25_capture"] = round(float(D[o[: len(o) // 4]].sum() / D.sum()), 3)
    r["best"] = max(["K", "h_q0", "B1", "B1G", "G", "LE"], key=lambda k: r[k])
    return r


if __name__ == "__main__":
    exp = Experiment("E49_decomposition_posthoc", "REGISTRY.md E49 (exploratory): uncertainty for the decomposition; "
                     "record of E42 post hoc ranking", {"datasets": DATASETS}, {"B": B}, 0)
    res = {"decomposition": {d: decomposition(d) for d in DATASETS}, "posthoc_ranking": {d: posthoc(d) for d in DATASETS}}
    for d in DATASETS:
        x = res["decomposition"][d][0]
        print(d, round(x["share_H"], 3) if x["share_H"] is not None else None, x.get("share_H_ci95"),
              x.get("bootstrap_excluded_share"), [res["decomposition"][d][s]["share_H"] for s in (1, 2)], res["posthoc_ranking"][d])
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")
