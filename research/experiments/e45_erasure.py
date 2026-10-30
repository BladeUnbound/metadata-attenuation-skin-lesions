"""E45 (exploratory; registered in research/iteration5/REGISTRY.md, 22:03 IST): linear concept erasure (LEACE),
dose-response, restoration and matched controls.

Per outer fold k, an eraser is fitted on the training fold's features and metadata concept Z (no diagnostic
labels), then applied to all rows; S_t = S - t (S - S_erased). The strictly cross-fitted (E40) estimator is
run on S_t, so the test fold is used only for prediction. For each S_t, metadata added value is measured for
each single variable (age, sex, site) and for the full block: the cross-variable specificity matrix.

Controls at t = 1: (a) a random subspace of the same rank, removed in whitened coordinates; (b) a random
subspace whose rank makes image-only log-loss rise as much as concept erasure does (rank found on outer
first-stage scores only, grid r * 2^j).

Usage: python e45_erasure.py <dataset> [...]
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import outer_fold, per_image_ll
from e13_replication import block_matrix
from e40_crossfit_baseline import boot_idx, boot_mean, ci, fit_predict, load
from x_common import VARS, missing, subset

RUNGS = ["convnext_tiny", "resnet18"]
DOSES = [0.25, 0.5, 0.75, 1.0]
CONCEPTS = ["age", "sex", "site"]
COMPLETE_CASE = {"pad_ufes"}  # registered: PAD-UFES-20 on complete cases (label-informative missingness)


def concept(df, ds, name):
    col = {n: c for n, _, c in VARS[ds]}[name]
    v = df[col]
    obs = ~missing(v)
    if name == "age":
        z = pd.to_numeric(v, errors="coerce").to_numpy(float)[:, None]
        obs &= ~np.isnan(z[:, 0])
    else:
        z = pd.get_dummies(v.astype(str), dtype=float).to_numpy()
        if name == "sex":
            z = z[:, :1]
    return np.nan_to_num(z), obs


def whitening(X):
    mu = X.mean(0)
    lam, V = np.linalg.eigh(np.cov(X - mu, rowvar=False))
    lam = np.clip(lam, lam.max() * 1e-6, None)
    return mu, V @ np.diag(lam ** -0.5) @ V.T, V @ np.diag(lam ** 0.5) @ V.T


def leace(X, Z):
    """Closed-form LEACE eraser (Belrose et al. 2023): x' = x - W+ P W (x - mu), P projecting onto W Sigma_xz."""
    mu, W, Wp = whitening(X)
    Sxz = (X - mu).T @ (Z - Z.mean(0)) / (len(X) - 1)
    U, s, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, s > s.max() * 1e-8]
    P = U @ U.T
    return mu, W @ P @ Wp, U.shape[1]  # row form: X' = X - (X - mu) @ M


def random_eraser(X, rank, rng):
    mu, W, Wp = whitening(X)
    U, _ = np.linalg.qr(rng.standard_normal((X.shape[1], rank)))
    return mu, W @ (U @ U.T) @ Wp


def prepare(ds):
    a = load(ds)
    if ds in COMPLETE_CASE:
        s = subset(ds, complete_case=True)
        idx, df, y, g = s["idx"], s["df"], s["y"], np.asarray(s["g"])
    else:
        idx, df, y, g = np.arange(len(a["y"])), a["df"].reset_index(drop=True), a["y"], np.asarray(a["groups"])
    num = {n: c for n, k, c in VARS[ds] if k == "num"}
    cat = {n: c for n, k, c in VARS[ds] if k == "cat"}
    blocks = {n: (lambda tr, n=n: block_matrix(df, [num[n]] if n in num else [], [cat[n]] if n in cat else [], tr))
              for n in CONCEPTS}
    blocks["all"] = lambda tr: block_matrix(df, list(num.values()), list(cat.values()), tr)
    return idx, df, y, g, blocks


def corrected_foldwise(Xk, y, g, folds, blocks):
    """E40 strict cross-fitting with fold-specific features Xk[k]. Returns image-only q0 and fusion q1 per block,
    plus the outer test scores."""
    jobs, tags = [], []
    for k, (tr, te) in enumerate(folds):
        jobs.append(delayed(outer_fold)(Xk[k], y, g, tr, te))
        tags.append((k, "outer"))
        for a, b in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], g[tr]):
            jobs.append(delayed(outer_fold)(Xk[k], y, g, tr[a], tr[b]))
            tags.append((k, "inner"))
    outs = Parallel(n_jobs=8)(jobs)
    s_out = np.zeros(len(y))
    q0 = np.zeros(len(y))
    q1 = {b: np.zeros(len(y)) for b in blocks}
    for k, (tr, te) in enumerate(folds):
        s_in = np.full(len(y), np.nan)
        for (idx, sc, _), (kk, kind) in zip(outs, tags):
            if kk == k:
                (s_out if kind == "outer" else s_in)[idx] = sc
        q0[te] = fit_predict(s_in[tr, None], y[tr], s_out[te, None])
        for b, enc in blocks.items():
            M = enc(tr)
            q1[b][te] = fit_predict(np.column_stack([s_in[tr], M[tr]]), y[tr], np.column_stack([s_out[te], M[te]]))
    return s_out, q0, q1


def outer_only_logloss(Xk, y, g, folds):
    outs = Parallel(n_jobs=5)(delayed(outer_fold)(Xk[k], y, g, tr, te) for k, (tr, te) in enumerate(folds))
    s = np.zeros(len(y))
    for idx, sc, _ in outs:
        s[idx] = sc
    q = np.zeros(len(y))
    for tr, te in folds:  # recalibrated image-only log-loss, same second stage family
        q[te] = fit_predict(s[tr, None], y[tr], s[te, None])
    return float(per_image_ll(y, q).mean())


def probe(Xk, Zc, obs, folds, kind):
    p = np.full(len(obs), np.nan)
    for k, (tr, te) in enumerate(folds):
        tr, te = tr[obs[tr]], te[obs[te]]
        sc = StandardScaler().fit(Xk[k][tr])
        A, Bm = sc.transform(Xk[k][tr]), sc.transform(Xk[k][te])
        if kind == "age":
            p[te] = Ridge(alpha=100.0).fit(A, Zc[tr, 0]).predict(Bm)
        else:
            t = Zc[:, :1] if kind == "sex" else Zc
            tgt = t[:, 0] if kind == "sex" else t.argmax(1)
            m = LogisticRegression(C=0.01, max_iter=3000).fit(A, tgt[tr])
            p[te] = m.decision_function(Bm) if kind == "sex" else m.predict(Bm)
    if kind == "age":
        return round(float(r2_score(Zc[obs, 0], p[obs])), 3)
    if kind == "sex":
        return round(float(roc_auc_score(Zc[obs, 0], p[obs])), 3)
    return round(float(np.mean(p[obs] == Zc[obs].argmax(1))), 3)  # site: accuracy (majority share reported too)


def summarise_cell(y, g, q0, q1, inv, idx):
    l0 = per_image_ll(y, q0)
    r = {"image_logloss": round(float(l0.mean()), 5)}
    for b, q in q1.items():
        d = l0 - per_image_ll(y, q)
        r[f"delta_{b}"] = round(float(d.mean()), 5)
        r[f"delta_{b}_ci95"] = ci(boot_mean(d, inv, idx))
    return r


def run(ds):
    idx, df, y, g, blocks = prepare(ds)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, g))
    inv, bidx = boot_idx(g, np.random.default_rng(45))
    rng = np.random.default_rng(450)
    res = {"n": int(len(y)), "complete_case": ds in COMPLETE_CASE}
    for rung in RUNGS:
        X = np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy")[idx].astype(np.float64)
        base = [X] * 5
        _, q0, q1 = corrected_foldwise(base, y, g, folds, blocks)
        R = {"t0": summarise_cell(y, g, q0, q1, inv, bidx)}
        print(ds, rung, "t0", R["t0"], flush=True)
        for c in CONCEPTS:
            Zc, obs = concept(df, ds, c)
            maj = None if c != "site" else round(float(np.bincount(Zc[obs].argmax(1)).max() / obs.sum()), 3)
            er = []
            for tr, te in folds:
                trc = tr[obs[tr]]
                mu, M, rank = leace(X[trc], Zc[trc])
                er.append((mu, M, rank))
            rc = {"rank": int(er[0][2]), "probe_t0": probe(base, Zc, obs, folds, c), "site_majority_share": maj}
            for t in DOSES:
                Xk = [X - t * (X - mu) @ M for mu, M, _ in er]
                _, q0, q1 = corrected_foldwise(Xk, y, g, folds, blocks)
                rc[f"t{t}"] = summarise_cell(y, g, q0, q1, inv, bidx)
                if t == 1.0:
                    rc["probe_t1"] = probe(Xk, Zc, obs, folds, c)
                    ll_concept = rc["t1.0"]["image_logloss"]
                print(ds, rung, c, t, rc[f"t{t}"], flush=True)
            # (a) random subspace, same rank
            rk = rc["rank"]
            Xa = [X - (X - mu) @ random_eraser(X[tr], rk, rng)[1] for (mu, _, _), (tr, _) in zip(er, folds)]
            _, q0, q1 = corrected_foldwise(Xa, y, g, folds, blocks)
            rc["random_same_rank"] = summarise_cell(y, g, q0, q1, inv, bidx)
            rc["random_same_rank"]["probe"] = probe(Xa, Zc, obs, folds, c)
            # (b) random subspace matched on image-only log-loss rise (grid search on outer scores only)
            target = ll_concept
            grid, best = [], None
            r_try = rk
            while r_try < X.shape[1]:
                erb = [random_eraser(X[tr], r_try, np.random.default_rng(4500 + r_try)) for tr, _ in folds]
                Xb = [X - (X - mu) @ M for mu, M in erb]
                ll = outer_only_logloss(Xb, y, g, folds)
                grid.append((r_try, round(ll, 5)))
                if best is None or abs(ll - target) < abs(best[1] - target):
                    best = (r_try, ll, Xb)
                if ll >= target:
                    break
                r_try *= 2
            _, q0, q1 = corrected_foldwise(best[2], y, g, folds, blocks)
            rc["random_matched"] = summarise_cell(y, g, q0, q1, inv, bidx)
            rc["random_matched"].update({"rank": int(best[0]), "grid": grid, "target_image_logloss": round(target, 5)})
            print(ds, rung, c, "controls", rc["random_same_rank"], rc["random_matched"], flush=True)
            R[c] = rc
        res[rung] = R
    return res


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        exp = Experiment(f"E45_{ds}", "REGISTRY.md E45 (exploratory): erasing a metadata variable's linear trace from "
                         "the image representation raises that variable's added value more than matched random erasure",
                         {"dataset": ds}, {"rungs": RUNGS, "doses": DOSES, "concepts": CONCEPTS, "estimator": "E40 corrected",
                                           "eraser": "LEACE closed form, fitted per outer training fold"}, 0)
        exp.finish(run(ds), interpretation="See research/EXPERIMENTS.md.")
