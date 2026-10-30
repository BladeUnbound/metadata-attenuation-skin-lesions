"""E40 (exploratory audit): strictly cross-fitted baseline. Registered in research/iteration5/REGISTRY.md.

Original estimator (unchanged code from E10/E13): out-of-fold image scores, second stage fitted on the
same outer folds. Its second-stage training scores were produced by first-stage models that saw the
outer test fold's labels (AUDIT.md R1), and training-row imputation used medians that include the test
fold (R2). Corrected estimator: per outer fold, inner grouped out-of-fold scores on the training fold
train the second stage; the test fold is scored by the first stage fitted on the whole training fold
(identical to the original test scores) and is used only for prediction.

Usage: python e40_crossfit_baseline.py <dataset> [...] | dedup_isic2019 | tests
Per-observation arrays go to data/<ds>/e40/<rung>_s<seed>.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, RESULTS, Experiment
from e10_ham_attenuation import EPS, meta_matrix, oof_scores, outer_fold, per_image_ll, second_stage
from e13_replication import adapter, block_matrix, oof_block

RUNGS = ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
SEEDS = [0, 1, 2]
B = 2000
WAVES = {"wave1": ["isic2020", "isic2019_nonham", "pad_ufes"], "wave2": ["hiba", "msk2020", "milk10k_derm", "milk10k_clin"]}


def load(ds):
    if ds == "ham10000":
        df = pd.read_csv(RESEARCH / "data/ham10000/raw/HAM10000_metadata.csv")
        return dict(df=df, y=(df["dx"] == "mel").to_numpy(int), groups=df["lesion_id"].to_numpy(),
                    enc=lambda tr: meta_matrix(df, tr))
    a = adapter(ds)
    num, cat = a["blocks"]["all"]
    return dict(df=a["df"], y=a["y"], groups=a["groups"], enc=lambda tr: block_matrix(a["df"], num, cat, tr))


def stored_original(ds, seed, rung):
    if ds == "ham10000":
        return json.loads((RESULTS / "E10_ham_attenuation.json").read_text())["results"]["primary"][f"seed{seed}"][rung]["added_value"]
    return json.loads((RESULTS / f"E13_{ds}.json").read_text())["results"]["seeds"][str(seed)][rung]["av_all"]


def fit_predict(Ztr, ytr, Zte):
    sc = StandardScaler().fit(Ztr)
    return LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Ztr), ytr).predict_proba(sc.transform(Zte))[:, 1]


def corrected(X, y, groups, folds, enc, s_outer):
    """Strict cross-fitting. s_outer: original OOF scores (each test fold scored by a model fitted on its training fold)."""
    jobs, meta = [], []
    for k, (tr, te) in enumerate(folds):
        inner = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], groups[tr]))
        for itr, ite in inner:
            jobs.append(delayed(outer_fold)(X, y, groups, tr[itr], tr[ite]))
            meta.append(k)
    outs = Parallel(n_jobs=8)(jobs)
    s_in = np.full(len(y), np.nan)
    p0, p1 = np.zeros(len(y)), np.zeros(len(y))
    for k, (tr, te) in enumerate(folds):
        s_in[:] = np.nan
        for (idx, s, _), kk in zip(outs, meta):
            if kk == k:
                s_in[idx] = s
        assert not np.isnan(s_in[tr]).any()
        M = enc(tr)  # every row encoded with this training fold's medians only
        p0[te] = fit_predict(s_in[tr, None], y[tr], s_outer[te, None])
        p1[te] = fit_predict(np.column_stack([s_in[tr], M[tr]]), y[tr], np.column_stack([s_outer[te], M[te]]))
    return p0, p1


def probs_original(score, M, y, folds):
    """Same computation as e10.second_stage but returning probabilities."""
    out = []
    for Z in (score[:, None], np.column_stack([score, M])):
        P = np.zeros(len(y))
        for tr, te in folds:
            P[te] = fit_predict(Z[tr], y[tr], Z[te])
        out.append(P)
    return out


def calib(y, p):
    lp = np.log(np.clip(p, EPS, 1 - EPS) / np.clip(1 - p, EPS, 1))
    m = LogisticRegression(C=1e6, max_iter=3000).fit(lp[:, None], y)
    return round(float(m.intercept_[0]), 3), round(float(m.coef_[0, 0]), 3)


def kl_bern(p1, p0):
    p1, p0 = np.clip(p1, EPS, 1 - EPS), np.clip(p0, EPS, 1 - EPS)
    return p1 * np.log(p1 / p0) + (1 - p1) * np.log((1 - p1) / (1 - p0))


def boot_idx(groups, rng):
    codes, inv = np.unique(groups, return_inverse=True)
    return inv, rng.integers(0, len(codes), (B, len(codes)))


def boot_mean(v, inv, idx):
    sums, counts = np.bincount(inv, weights=v), np.bincount(inv)
    return sums[idx].sum(1) / counts[idx].sum(1)


def ci(bs):
    return [round(float(np.percentile(bs, 2.5)), 5), round(float(np.percentile(bs, 97.5)), 5)]


def summarise(y, groups, p0, p1, score):
    l0, l1 = per_image_ll(y, p0), per_image_ll(y, p1)
    d = l0 - l1
    gm = pd.Series(d).groupby(groups).mean()
    return d, {
        "delta": round(float(d.mean()), 5), "delta_median": round(float(np.median(d)), 5),
        "delta_pos": round(float(d[y == 1].mean()), 5), "delta_neg": round(float(d[y == 0].mean()), 5),
        "delta_group_weighted": round(float(gm.mean()), 5),
        "image_logloss": round(float(l0.mean()), 4), "fusion_logloss": round(float(l1.mean()), 4),
        "image_auroc": round(float(roc_auc_score(y, score)), 4), "fusion_auroc": round(float(roc_auc_score(y, p1)), 4),
        "image_brier": round(float(brier_score_loss(y, p0)), 5), "fusion_brier": round(float(brier_score_loss(y, p1)), 5),
        "image_calib_int_slope": calib(y, p0), "fusion_calib_int_slope": calib(y, p1),
        "K_mean": round(float(kl_bern(p1, p0).mean()), 5),
    }


def run(ds, groups_override=None, seeds=SEEDS, tag=""):
    a = load(ds)
    y, df = a["y"], a["df"]
    groups = a["groups"] if groups_override is None else groups_override
    out_dir = RESEARCH / f"data/{ds}/e40{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(40)
    inv, idx = boot_idx(groups, rng)
    res, failures = {}, []
    for seed in seeds:
        folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(df, y, groups))
        M_orig = np.zeros((len(y), a["enc"](folds[0][0]).shape[1]))
        for tr, te in folds:
            M_orig[te] = a["enc"](tr)[te]
        res[seed] = {}
        for rung in RUNGS:
            X = np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy")
            score, _ = oof_scores(X, y, groups, folds)
            o0, o1 = probs_original(score, M_orig, y, folds)
            d_o, r_o = summarise(y, groups, o0, o1, score)
            if not tag:  # reproduction check against the stored pre-registered result
                l0, l1 = second_stage(score[:, None], np.column_stack([score, M_orig]), y, folds)
                stored = stored_original(ds, seed, rung)
                r_o["stored"], r_o["reproduced"] = stored, bool(abs(round(float((l0 - l1).mean()), 5) - stored) < 1e-5)
                if not r_o["reproduced"]:
                    failures.append(f"{ds} seed {seed} {rung}: original {r_o['delta']} vs stored {stored}")
            c0, c1 = corrected(X, y, groups, folds, a["enc"], score)
            d_c, r_c = summarise(y, groups, c0, c1, score)
            diff = boot_mean(d_c - d_o, inv, idx)
            r_c["delta_ci95"] = ci(boot_mean(d_c, inv, idx))
            r_o["delta_ci95"] = ci(boot_mean(d_o, inv, idx))
            r = {"original": r_o, "corrected": r_c, "corrected_minus_original": round(float((d_c - d_o).mean()), 5),
                 "corrected_minus_original_ci95": ci(diff)}
            res[seed][rung] = r
            fold = np.zeros(len(y), int)
            for k, (_, te) in enumerate(folds):
                fold[te] = k
            np.savez_compressed(out_dir / f"{rung}_s{seed}.npz", y=y, groups=np.asarray(groups).astype(str),
                                score=score, q0=c0, q1=c1, delta=d_c, K=kl_bern(c1, c0), q0_orig=o0, q1_orig=o1,
                                delta_orig=d_o, fold=fold)
            print(ds, seed, rung, "orig", r_o["delta"], "corr", r_c["delta"], "diff", r["corrected_minus_original"],
                  r["corrected_minus_original_ci95"], flush=True)
    return res, failures


def tests():
    """H1/H2 one-sided paired grouped-bootstrap tests, seed 0, both estimators, Holm within the registered waves."""
    out = {}
    for ds in ["ham10000"] + WAVES["wave1"] + WAVES["wave2"]:
        out[ds] = {}
        f = {r: np.load(RESEARCH / f"data/{ds}/e40/{r}_s0.npz") for r in ["hsv", "resnet18", "convnext_tiny"]}
        g = f["hsv"]["groups"]
        rng = np.random.default_rng(41)
        inv, idx = boot_idx(g, rng)
        for hyp, (hi, lo) in {"H1": ("hsv", "convnext_tiny"), "H2": ("resnet18", "convnext_tiny")}.items():
            for est, key in [("original", "delta_orig"), ("corrected", "delta")]:
                diff = f[hi][key] - f[lo][key]
                bs = boot_mean(diff, inv, idx)
                out[ds][f"{hyp}_{est}"] = {"diff": round(float(diff.mean()), 5), "ci95": ci(bs),
                                           "p_one_sided": float((1 + np.sum(bs <= 0)) / (B + 1))}
    for wave, dss in WAVES.items():
        for est in ["original", "corrected"]:
            ps = sorted(((d, h, out[d][f"{h}_{est}"]["p_one_sided"]) for d in dss for h in ["H1", "H2"]), key=lambda t: t[2])
            run_max, m = 0.0, len(ps)
            for rank, (d, h, p) in enumerate(ps):
                run_max = max(run_max, min(1.0, (m - rank) * p))
                out[d][f"{h}_{est}"]["p_holm_" + wave] = run_max
                out[d][f"{h}_{est}"]["holds"] = run_max <= 0.05
    return out


def dedup_groups(ds, thr=0.98):
    a = load(ds)
    X = np.load(RESEARCH / f"data/{ds}/emb_convnext_tiny.npy").astype(np.float32)
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    codes, g = np.unique(np.asarray(a["groups"]).astype(str), return_inverse=True)
    parent = np.arange(len(codes))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    merged = 0
    for i in range(0, len(X), 2048):
        S = X[i:i + 2048] @ X.T
        for r, j in zip(*np.where(S > thr)):
            a_, b_ = find(g[i + r]), find(g[j])
            if a_ != b_:
                parent[a_] = b_
                merged += 1
    return np.array([find(v) for v in g]).astype(str), merged


if __name__ == "__main__":
    arg = sys.argv[1:]
    if arg == ["tests"]:
        exp = Experiment("E40_tests", "Sensitivity: H1/H2 under the corrected estimator (REGISTRY.md E40)", {}, {"B": B}, 0)
        exp.finish(tests(), interpretation="See research/EXPERIMENTS.md.")
    elif arg == ["dedup_isic2019"]:
        g, merged = dedup_groups("isic2019_nonham")
        exp = Experiment("E40_dedup_isic2019", "Sensitivity: ISIC 2019 groups merged over near-duplicates (cos > 0.98)",
                         {"dataset": "isic2019_nonham", "group_merges": merged, "n_groups": int(len(np.unique(g)))},
                         {"seeds": [0]}, 0)
        res, fails = run("isic2019_nonham", groups_override=g, seeds=[0], tag="_dedup")
        exp.finish(res, interpretation="See research/EXPERIMENTS.md.", failures=fails)
    else:
        for ds in arg:
            exp = Experiment(f"E40_{ds}", "REGISTRY.md E40: corrected vs original estimator",
                             {"dataset": ds}, {"rungs": RUNGS, "seeds": SEEDS, "inner": "StratifiedGroupKFold(5, seed 0)",
                                               "second_stage": "logistic C=1, standardised", "B": B}, SEEDS)
            res, fails = run(ds)
            exp.finish(res, interpretation="See research/EXPERIMENTS.md.", failures=fails)
