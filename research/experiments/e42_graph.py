"""E42/E43 (registered in research/iteration5/REGISTRY.md, 2026-10-28 22:25 IST): inductive representation
graphs, local ambiguity, and held-out prediction of per-observation metadata benefit.

Inputs: E40 corrected per-observation arrays (data/<ds>/e40/<rung>_s0.npz) and cached embeddings.
Per outer fold k (fold assignment 0): training images T_k are graph nodes; each test image attaches to its
K nearest training images (cosine on T_k-standardised, L2-normalised features); training nodes link to their
K nearest training images from other groups. Test labels are never used for any statistic or model.

H4 (ConvNeXt-T only): inner cross-fitting inside T_k gives inner out-of-fold Δ_j for training nodes; ridge
regressions on training nodes (B1 vs B1+G) predict Δ_i for the test fold.

Usage: python e42_graph.py <dataset> [...] | summarise
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, RESULTS, Experiment
from e10_ham_attenuation import outer_fold, per_image_ll
from e40_crossfit_baseline import fit_predict, kl_bern, load

RUNGS = ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
PRIMARY = "convnext_tiny"
K_MAIN, K_SENS = 20, [10, 50]
B = 2000
ALPHAS = np.logspace(-3, 3, 7)
# (age, sex, site) columns per dataset for the metadata-similarity statistic MS_i
META_COLS = {"ham10000": ("age", "sex", "localization"), "isic2020": ("age_approx", "sex", "anatom_site_general_challenge"),
             "isic2019_nonham": ("age_approx", "sex", "anatom_site_general"), "pad_ufes": ("age", "gender", "region"),
             "hiba": ("age_approx", "sex", "anatom_site_1"), "msk2020": ("age_approx", "sex", "anatom_site_1"),
             "milk10k_derm": ("age_approx", "sex", "anatom_site_1"), "milk10k_clin": ("age_approx", "sex", "anatom_site_1")}
UNITS = {"ham10000": ["ham10000"], "isic2020": ["isic2020"], "isic2019_nonham": ["isic2019_nonham"], "pad_ufes": ["pad_ufes"],
         "hiba": ["hiba"], "msk2020": ["msk2020"], "milk10k": ["milk10k_derm", "milk10k_clin"]}


def h2(p):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def knn(Q, R, k, q_groups=None, r_groups=None, chunk=2048):
    """k nearest rows of R for each row of Q by cosine (rows already L2-normalised); same-group pairs excluded
    when both group arrays are given. Returns indices (n_q, k) and cosine distances."""
    idx = np.empty((len(Q), k), int)
    dist = np.empty((len(Q), k))
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ R.T
        if q_groups is not None:
            S[q_groups[i:i + chunk, None] == r_groups[None, :]] = -np.inf
        part = np.argpartition(-S, k, axis=1)[:, :k]
        idx[i:i + chunk] = part
        dist[i:i + chunk] = 1 - np.take_along_axis(S, part, 1)
    return idx, np.clip(dist, 0, None)


def weights(dist, sigma, binary=False):
    return np.ones_like(dist) if binary else np.exp(-dist ** 2 / (2 * sigma ** 2))


def meta_vectors(ds, df):
    a, s, t = META_COLS[ds]
    age = pd.to_numeric(df[a], errors="coerce").to_numpy()
    sex = df[s].astype(str).replace({"nan": "", "unknown": ""}).to_numpy()
    site = df[t].astype(str).replace({"nan": "", "unknown": ""}).to_numpy()
    return age, sex, site


def stats_for(nb, w, y_ref, pred_cls, meta_q, meta_r, delta_ref=None):
    """Graph statistics for query rows; nb indexes the reference arrays (y_ref, meta_r, delta_ref).
    meta_q / meta_r are (age, sex, site) tuples for the query rows and for the reference rows."""
    ws = w.sum(1)
    p = (np.sum(w * y_ref[nb], 1) + 1) / (ws + 2)
    out = {"LE": h2(p), "NC": np.sum(w * (y_ref[nb] == pred_cls[:, None]), 1) / ws}
    (age_q, sex_q, site_q), (age_r, sex_r, site_r) = meta_q, meta_r
    a_q, a_r = age_q[:, None], age_r[nb]
    age_sim = np.where(np.isnan(a_q) | np.isnan(a_r), 0.0, 1 - np.abs(a_q - a_r) / 100)
    sex_m = (sex_q[:, None] == sex_r[nb]) & (sex_r[nb] != "")
    site_m = (site_q[:, None] == site_r[nb]) & (site_r[nb] != "")
    out["MS"] = np.sum(w * (age_sim + sex_m + site_m) / 3, 1) / ws
    if delta_ref is not None:
        out["G"] = np.sum(w * delta_ref[nb], 1) / ws
    return out


def inner_delta(X, y, groups, tr, enc):
    """Inner out-of-fold Δ_j, q0_j, q1_j for j in T_k, using only T_k (E40's inner folds and first stage)."""
    inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], groups[tr]))
    outs = Parallel(n_jobs=8)(delayed(outer_fold)(X, y, groups, tr[a], tr[b]) for a, b in inner)
    s = np.full(len(y), np.nan)
    for idx, sc, _ in outs:
        s[idx] = sc
    M = enc(tr)
    q0, q1 = np.full(len(y), np.nan), np.full(len(y), np.nan)
    for a, b in inner:
        ia, ib = tr[a], tr[b]
        q0[ib] = fit_predict(s[ia, None], y[ia], s[ib, None])
        q1[ib] = fit_predict(np.column_stack([s[ia], M[ia]]), y[ia], np.column_stack([s[ib], M[ib]]))
    d = per_image_ll(y, np.nan_to_num(q0, nan=0.5)) - per_image_ll(y, np.nan_to_num(q1, nan=0.5))
    return d, q0, q1


def ridge_fit_predict(F_tr, t_tr, g_tr, F_te):
    sc = StandardScaler().fit(F_tr)
    A, Bm = sc.transform(F_tr), sc.transform(F_te)
    best, best_err = None, np.inf
    for al in ALPHAS:
        err = 0.0
        for a, b in GroupKFold(3).split(A, t_tr, g_tr):
            err += np.sum((Ridge(alpha=al).fit(A[a], t_tr[a]).predict(A[b]) - t_tr[b]) ** 2)
        if err < best_err:
            best, best_err = al, err
    return Ridge(alpha=best).fit(A, t_tr).predict(Bm), best


def boot_spearman(a, b, groups, rng, extra=None):
    """Grouped-bootstrap Spearman(a, b); with extra=(a2, b2) also the paired difference."""
    codes, inv = np.unique(groups, return_inverse=True)
    members = pd.Series(np.arange(len(inv))).groupby(inv).apply(np.array).to_numpy()
    sp = lambda u, v: float(np.corrcoef(rankdata(u), rankdata(v))[0, 1])
    obs = sp(a, b)
    bs, bd = [], []
    for _ in range(B):
        ix = np.concatenate(members[rng.integers(0, len(codes), len(codes))])
        r1 = sp(a[ix], b[ix])
        bs.append(r1)
        if extra is not None:
            bd.append(r1 - sp(extra[0][ix], extra[1][ix]))
    return obs, np.array(bs), (np.array(bd) if extra is not None else None)


def ci(x):
    return [round(float(np.percentile(x, 2.5)), 4), round(float(np.percentile(x, 97.5)), 4)]


def run(ds):
    a = load(ds)
    df, y, groups = a["df"], a["y"], np.asarray(a["groups"]).astype(str)
    gcode = np.unique(groups, return_inverse=True)[1]
    meta = meta_vectors(ds, df)
    sub = lambda rows: tuple(m[rows] for m in meta)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    res = {"n": int(len(y)), "rungs": {}}
    per_obs = {}
    for rung in RUNGS:
        f = np.load(RESEARCH / f"data/{ds}/e40/{rung}_s0.npz")
        assert np.array_equal(f["y"], y) and np.array_equal(f["fold"][folds[0][1]], np.zeros(len(folds[0][1])))
        X = np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy")
        q0, q1, delta = np.clip(f["q0"], 1e-9, 1 - 1e-9), np.clip(f["q1"], 1e-9, 1 - 1e-9), f["delta"]
        feats = {kk: {} for kk in [K_MAIN] + K_SENS}
        feats_bin = {}
        h4 = {"pred_B1": np.zeros(len(y)), "pred_B1G": np.zeros(len(y)), "alpha_B1": [], "alpha_B1G": []}
        n_comp_issues = 0
        for k, (tr, te) in enumerate(folds):
            sc = StandardScaler().fit(X[tr])
            Z = sc.transform(X).astype(np.float32)
            Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
            nb_te, d_te = knn(Z[te], Z[tr], max([K_MAIN] + K_SENS))
            order = np.argsort(d_te, axis=1)
            nb_te, d_te = np.take_along_axis(nb_te, order, 1), np.take_along_axis(d_te, order, 1)
            nb_tr, d_tr = knn(Z[tr], Z[tr], K_MAIN, gcode[tr], gcode[tr])
            sigma = float(np.median(np.max(d_tr, 1)))  # median distance to the K-th training neighbour
            pred_te = (q0[te] > 0.5).astype(int)
            yt = y[tr]
            for kk in [K_MAIN] + K_SENS:
                st = stats_for(nb_te[:, :kk], weights(d_te[:, :kk], sigma), yt, pred_te, sub(te), sub(tr))
                st["density"] = np.log(d_te[:, :kk].mean(1) + 1e-12)
                for name, v in st.items():
                    feats[kk].setdefault(name, np.zeros(len(y)))[te] = v
            stb = stats_for(nb_te[:, :K_MAIN], weights(d_te[:, :K_MAIN], sigma, True), yt, pred_te, sub(te), sub(tr))
            for name, v in stb.items():
                feats_bin.setdefault(name, np.zeros(len(y)))[te] = v
            if rung == PRIMARY:
                d_in, q0_in, q1_in = inner_delta(X, y, groups, tr, a["enc"])
                q0_in = np.clip(q0_in, 1e-9, 1 - 1e-9)
                d_tr_ref = d_in[tr]
                w_te = weights(d_te[:, :K_MAIN], sigma)
                G_te = np.sum(w_te * d_tr_ref[nb_te[:, :K_MAIN]], 1) / w_te.sum(1)
                feats[K_MAIN].setdefault("G", np.zeros(len(y)))[te] = G_te
                # training-node features (leave-group-out neighbours), same definitions
                w_tr = weights(d_tr, sigma)
                pred_tr = (q0_in[tr] > 0.5).astype(int)
                st_tr = stats_for(nb_tr, w_tr, yt, pred_tr, sub(tr), sub(tr), d_tr_ref)
                st_tr["density"] = np.log(d_tr.mean(1) + 1e-12)
                base_tr = np.column_stack([kl_bern(q1_in[tr], q0_in[tr]), h2(q0_in[tr]),
                                           np.abs(np.log(q0_in[tr] / (1 - q0_in[tr]))),
                                           np.linalg.norm(X[tr], axis=1), st_tr["density"]])
                base_te = np.column_stack([kl_bern(q1[te], q0[te]), h2(q0[te]), np.abs(np.log(q0[te] / (1 - q0[te]))),
                                           np.linalg.norm(X[te], axis=1), feats[K_MAIN]["density"][te]])
                gx_tr = np.column_stack([st_tr["LE"], st_tr["NC"], st_tr["MS"], st_tr["G"]])
                gx_te = np.column_stack([feats[K_MAIN][n][te] for n in ["LE", "NC", "MS", "G"]])
                p1, al1 = ridge_fit_predict(base_tr, d_tr_ref, groups[tr], base_te)
                p2, al2 = ridge_fit_predict(np.hstack([base_tr, gx_tr]), d_tr_ref, groups[tr], np.hstack([base_te, gx_te]))
                h4["pred_B1"][te], h4["pred_B1G"][te] = p1, p2
                h4["alpha_B1"].append(al1)
                h4["alpha_B1G"].append(al2)
        rng = np.random.default_rng(42)
        hq0, K = h2(q0), kl_bern(q1, q0)
        r = {}
        obs, bs, _ = boot_spearman(feats[K_MAIN]["LE"], delta, groups, rng)
        r["H1_spearman_LE_delta"] = {"rho": round(obs, 4), "ci95": ci(bs), "p_one_sided": float((1 + np.sum(bs <= 0)) / (B + 1))}
        obs, bs, _ = boot_spearman(hq0, delta, groups, rng)
        r["H2_spearman_hq0_delta"] = {"rho": round(obs, 4), "ci95": ci(bs), "p_one_sided": float((1 + np.sum(bs <= 0)) / (B + 1))}
        sp = lambda u: round(float(np.corrcoef(rankdata(u), rankdata(delta))[0, 1]), 4)
        r["descriptive_spearman_with_delta"] = {"K": sp(K), "margin": sp(-np.abs(np.log(q0 / (1 - q0)))),
                                                "NC": sp(feats[K_MAIN]["NC"]), "MS": sp(feats[K_MAIN]["MS"]),
                                                "density": sp(feats[K_MAIN]["density"]),
                                                "norm": sp(np.linalg.norm(X, axis=1))}
        r["sensitivity_H1_rho"] = {f"k{kk}": sp(feats[kk]["LE"]) for kk in K_SENS}
        r["sensitivity_H1_rho"]["binary_k20"] = sp(feats_bin["LE"])
        r["delta_mean"], r["mean_hq0"], r["mean_K"] = float(delta.mean()), float(hq0.mean()), float(K.mean())
        per_obs[rung] = {"LE": feats[K_MAIN]["LE"], "delta": delta}
        if rung == PRIMARY:
            e1, e2 = (delta - h4["pred_B1"]) ** 2, (delta - h4["pred_B1G"]) ** 2
            codes, inv = np.unique(groups, return_inverse=True)
            idx = rng.integers(0, len(codes), (B, len(codes)))
            sums, cnt = np.bincount(inv, weights=e1 - e2), np.bincount(inv)
            bsd = sums[idx].sum(1) / cnt[idx].sum(1)
            var = np.var(delta)
            r["H4"] = {"mse_B1": float(e1.mean()), "mse_B1G": float(e2.mean()), "mse_mean_only": float(var),
                       "diff_B1_minus_B1G": float((e1 - e2).mean()), "ci95": [float(np.percentile(bsd, 2.5)), float(np.percentile(bsd, 97.5))],
                       "p_one_sided": float((1 + np.sum(bsd <= 0)) / (B + 1)),
                       "r2_B1": float(1 - e1.mean() / var), "r2_B1G": float(1 - e2.mean() / var),
                       "alpha_B1": h4["alpha_B1"], "alpha_B1G": h4["alpha_B1G"]}
            tot = delta.sum()
            cap = {}
            for name, score in {"K": K, "B1": h4["pred_B1"], "B1G": h4["pred_B1G"], "LE": feats[K_MAIN]["LE"],
                                "hq0": hq0, "G": feats[K_MAIN]["G"]}.items():
                o = np.argsort(-score)
                cap[name] = {f"top{q}": round(float(delta[o[:int(len(o) * q / 100)]].sum() / tot), 3) for q in (10, 25, 50)}
            cap["random"] = {f"top{q}": q / 100 for q in (10, 25, 50)}
            r["H4_capture"] = cap
            np.savez_compressed(RESEARCH / f"data/{ds}/e40/e42_{rung}.npz", pred_B1=h4["pred_B1"], pred_B1G=h4["pred_B1G"],
                                **{f"{n}": v for n, v in feats[K_MAIN].items()})
        res["rungs"][rung] = r
        print(ds, rung, {k_: v for k_, v in r.items() if k_ not in ("H4_capture",)}, flush=True)
    # H3: paired difference of the H1 correlation, HSV and ResNet-18 vs ConvNeXt-T
    rng = np.random.default_rng(43)
    res["H3"] = {}
    for other in ["hsv", "resnet18"]:
        _, _, bd = boot_spearman(per_obs[other]["LE"], per_obs[other]["delta"], groups, rng,
                                 extra=(per_obs[PRIMARY]["LE"], per_obs[PRIMARY]["delta"]))
        d_obs = res["rungs"][other]["H1_spearman_LE_delta"]["rho"] - res["rungs"][PRIMARY]["H1_spearman_LE_delta"]["rho"]
        res["H3"][f"{other}_minus_cnx"] = {"diff": round(d_obs, 4), "ci95": ci(bd),
                                           "p_two_sided": float(min(1, 2 * min((1 + np.sum(bd <= 0)), (1 + np.sum(bd >= 0))) / (B + 1)))}
    return res


def holm(pairs):
    order = sorted(pairs, key=lambda t: t[1])
    out, run_max, m = {}, 0.0, len(order)
    for rank, (u, p) in enumerate(order):
        run_max = max(run_max, min(1.0, (m - rank) * p))
        out[u] = run_max
    return out


def summarise():
    R = {ds: json.loads((RESULTS / f"E42_{ds}.json").read_text())["results"] for u in UNITS.values() for ds in u}
    out = {}
    for hyp, key, rule in [("H1", "H1_spearman_LE_delta", 5), ("H2", "H2_spearman_hq0_delta", 5), ("H4", "H4", 4)]:
        unit_p = {u: max(R[ds]["rungs"][PRIMARY][key]["p_one_sided"] for ds in dss) for u, dss in UNITS.items()}
        adj = holm(list(unit_p.items()))
        held = [u for u in UNITS if adj[u] <= 0.05]
        out[hyp] = {"p_unit": unit_p, "p_holm": adj, "held_units": held, "n_held": len(held), "rule": f">= {rule} of 7",
                    "holds": len(held) >= rule}
    return out


if __name__ == "__main__":
    if sys.argv[1:] == ["summarise"]:
        exp = Experiment("E42_summary", "REGISTRY.md E42–E44: H1, H2, H4 decisions over 7 counting units", {}, {}, 0)
        exp.finish(summarise(), interpretation="See research/EXPERIMENTS.md.")
    else:
        for ds in sys.argv[1:]:
            exp = Experiment(f"E42_{ds}", "REGISTRY.md E42–E44", {"dataset": ds},
                             {"k": K_MAIN, "k_sensitivity": K_SENS, "weights": "gaussian, sigma = median k-th NN distance",
                              "graph": "inductive; training nodes exclude same-group neighbours", "B": B,
                              "ridge_alphas": ALPHAS.tolist()}, 0)
            exp.finish(run(ds), interpretation="See research/EXPERIMENTS.md.")
