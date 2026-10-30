"""E50/E51 (registered in research/iteration6/REGISTRY.md, 2026-10-29 13:49 IST): attribute erasure with an
information dose, nested performance-matched random controls, shams and restoration.

Per outer fold k, every eraser is fitted on that training fold's features (and, for targeted and sham erasers, the
attribute; never diagnostic labels) and applied to all rows. Eraser parts (mu, A, Bt): coordinates c = (X - mu) A,
erased features X - c Bt. Doses: D1(t) = X - t c Bt (E45 definition; information-preserving for t < 1);
D2(t) = X - (c - c_t) Bt with c_t = sqrt(1 - t) c + sqrt(t) sigma eps (information dose). The matched random rank is
chosen per outer fold from inner-CV, training-fold-only image log-loss (AUDIT A1). The strictly cross-fitted (E40)
estimator is run on every arm; the outer test fold is used only for prediction.

Usage: python e50_dose_restoration.py <dataset> [--concept age|site] [--rung convnext_tiny] [--fold-seed 0] [--light]
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import sklearn
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import outer_fold, per_image_ll
from e40_crossfit_baseline import boot_idx, boot_mean, ci, fit_predict
from e45_erasure import concept, prepare, whitening

DOSES = [0.25, 0.5, 0.75, 1.0]
N_CTRL = 5
ALPHAS = np.logspace(-1, 4, 6)
N_JOBS = 8
GAINS = ["age", "site", "sex", "all"]
CODE = [Path(__file__), *(Path(__file__).parent / f for f in
                          ["e45_erasure.py", "e40_crossfit_baseline.py", "e10_ham_attenuation.py", "e13_replication.py",
                           "x_common.py"])]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


# ---------------------------------------------------------------- erasers: (mu, A, Bt, rank)
def lw_whitening(X):
    mu = X.mean(0)
    lam, V = np.linalg.eigh(LedoitWolf().fit(X).covariance_)
    return mu, V @ np.diag(lam ** -0.5) @ V.T, V @ np.diag(lam ** 0.5) @ V.T


def leace_parts(X, Z, shrink=False):
    """Any invertible whitening keeps the exact zero training cross-covariance; shrink=True uses Ledoit-Wolf
    (REGISTRY amendment 1, variant T-LW)."""
    mu, W, Wp = (lw_whitening if shrink else whitening)(X)
    Sxz = (X - mu).T @ (Z - Z.mean(0)) / (len(X) - 1)
    U, s, _ = np.linalg.svd(W @ Sxz, full_matrices=False)
    U = U[:, s > s.max() * 1e-8]
    return mu, W @ U, U.T @ Wp, U.shape[1]


def random_parts(wh, rank, rng):
    mu, W, Wp = wh
    U, _ = np.linalg.qr(rng.standard_normal((W.shape[0], rank)))
    return mu, W @ U, U.T @ Wp, rank


def coords(X, e):
    return (X - e[0]) @ e[1]


def d1(X, e, t):
    return X - t * coords(X, e) @ e[2]


def d2(X, e, t, tr, rng):
    c = coords(X, e)
    eps = rng.standard_normal(c.shape)
    ct = np.sqrt(1 - t) * c + np.sqrt(t) * c[tr].std(0) * eps
    return X - (c - ct) @ e[2]


# ---------------------------------------------------------------- strict cross-fitting on fold-specific features
def corrected(Xk, y, g, folds, inner, blocks, fold_blocks=None):
    jobs, tags = [], []
    for k, (tr, te) in enumerate(folds):
        jobs.append(delayed(outer_fold)(Xk[k], y, g, tr, te))
        tags.append((k, -1))
        for j, (a, b) in enumerate(inner[k]):
            jobs.append(delayed(outer_fold)(Xk[k], y, g, tr[a], tr[b]))
            tags.append((k, j))
    outs = Parallel(n_jobs=N_JOBS)(jobs)
    s_out, q0 = np.zeros(len(y)), np.zeros(len(y))
    q1 = {b: np.zeros(len(y)) for b in list(blocks) + list(fold_blocks or {})}
    inner_ll, Cs = [], []
    for k, (tr, te) in enumerate(folds):
        s_in = np.full(len(y), np.nan)
        cs = {}
        for (idx, sc, C), (kk, j) in zip(outs, tags):
            if kk == k:
                (s_out if j < 0 else s_in)[idx] = sc
                cs[j] = float(C)
        assert not np.isnan(s_in[tr]).any()
        inner_ll.append(float(per_image_ll(y[tr], 1 / (1 + np.exp(-s_in[tr]))).mean()))
        Cs.append([cs[-1]] + [cs[j] for j in range(len(inner[k]))])
        q0[te] = fit_predict(s_in[tr, None], y[tr], s_out[te, None])
        mats = {b: enc(tr) for b, enc in blocks.items()}
        mats.update({b: F[k] for b, F in (fold_blocks or {}).items()})
        for b, M in mats.items():
            q1[b][te] = fit_predict(np.column_stack([s_in[tr], M[tr]]), y[tr], np.column_stack([s_out[te], M[te]]))
    return dict(s=s_out, q0=q0, q1=q1, inner_ll=inner_ll, C=Cs)


def inner_only_ll(Xk, y, g, folds, inner, ks):
    jobs, tags = [], []
    for k in ks:
        tr = folds[k][0]
        for a, b in inner[k]:
            jobs.append(delayed(outer_fold)(Xk[k], y, g, tr[a], tr[b]))
            tags.append(k)
    outs = Parallel(n_jobs=N_JOBS)(jobs)
    res = {}
    for k in ks:
        tr = folds[k][0]
        s = np.full(len(y), np.nan)
        for (idx, sc, _), kk in zip(outs, tags):
            if kk == k:
                s[idx] = sc
        res[k] = float(per_image_ll(y[tr], 1 / (1 + np.exp(-s[tr]))).mean())
    return res


def matched_ranks(X, y, g, folds, inner, wh, target, dmax):
    """Per fold: smallest rank on {1, 2, 4, ...} (refined at sqrt(2) steps) whose inner-CV training-only image
    log-loss under random erasure reaches the targeted eraser's. Uses training labels only."""
    state = {k: {"r": 1, "grid": [], "phase": "double"} for k in range(len(folds))}
    done = {}
    while len(done) < len(folds):
        ks = [k for k in state if k not in done]
        Xk = [None] * len(folds)
        for k in ks:
            e = random_parts(wh[k], state[k]["r"], np.random.default_rng(50000 + 1000 * k + state[k]["r"]))
            Xk[k] = d1(X, e, 1.0)
        ll = inner_only_ll(Xk, y, g, folds, inner, ks)
        for k in ks:
            st, r = state[k], state[k]["r"]
            st["grid"].append((r, round(ll[k], 5)))
            hit = ll[k] >= target[k]
            if st["phase"] == "refine":
                done[k] = r if hit else st["upper"]
            elif hit:
                rr = int(round(r / np.sqrt(2)))
                if r >= 4 and rr > r // 2:
                    st.update(phase="refine", upper=r, r=rr)
                else:
                    done[k] = r
            elif 2 * r >= dmax:
                done[k] = r  # cap: no random subspace below dmax reached the target
            else:
                st["r"] = 2 * r
    return [done[k] for k in range(len(folds))], {k: state[k]["grid"] for k in state}


# ---------------------------------------------------------------- probes (training-fold fitting only)
def _ridge(A, z, alphas):
    ma, mz = A.mean(0), z.mean()
    Ac = A - ma
    lam, V = np.linalg.eigh(Ac.T @ Ac)
    b = V.T @ (Ac.T @ (z - mz))
    return [(V @ (b / (lam + a)), ma, mz) for a in alphas]


def probe_age(Xk, z, obs, g, folds):
    """Held-out R^2 of a ridge probe with alpha chosen by grouped inner CV (and of fixed alpha = 100, as E45)."""
    p, p100, chosen = np.full(len(z), np.nan), np.full(len(z), np.nan), []
    for k, (tr, te) in enumerate(folds):
        trc, tec = tr[obs[tr]], te[obs[te]]
        sc = StandardScaler().fit(Xk[k][trc])
        A, Bm = sc.transform(Xk[k][trc]), sc.transform(Xk[k][tec])
        err = np.zeros(len(ALPHAS))
        for a, b in GroupKFold(3).split(A, z[trc], g[trc]):
            for i, (w, ma, mz) in enumerate(_ridge(A[a], z[trc][a], ALPHAS)):
                err[i] += np.sum(((A[b] - ma) @ w + mz - z[trc][b]) ** 2)
        best = ALPHAS[int(np.argmin(err))]
        chosen.append(float(best))
        (w, ma, mz), (w1, ma1, mz1) = _ridge(A, z[trc], [best, 100.0])
        p[tec], p100[tec] = (Bm - ma) @ w + mz, (Bm - ma1) @ w1 + mz1
    r2 = lambda q: float(1 - np.sum((z[obs] - q[obs]) ** 2) / np.sum((z[obs] - z[obs].mean()) ** 2))
    return {"r2_nested": round(r2(p), 4), "r2_alpha100": round(r2(p100), 4), "alpha": chosen}, p


def probe_age_within_test(Xk, z, obs, g, folds):
    """Amendment 2: age decodability that a training-fold eraser leaves in new images. Within each outer test fold
    (never used to fit its eraser), a ridge probe is cross-validated over grouped 5-fold splits of that fold, alpha by
    grouped 3-fold CV inside the probe-training part. Uses features and age only, never diagnostic labels."""
    p = np.full(len(z), np.nan)
    for k, (_, te) in enumerate(folds):
        tec = te[obs[te]]
        for a, b in GroupKFold(5).split(tec, groups=g[tec]):
            fa, fb = tec[a], tec[b]
            sc = StandardScaler().fit(Xk[k][fa])
            A, Bm = sc.transform(Xk[k][fa]), sc.transform(Xk[k][fb])
            err = np.zeros(len(ALPHAS))
            for a2, b2 in GroupKFold(3).split(A, groups=g[fa]):
                for i, (w, ma, mz) in enumerate(_ridge(A[a2], z[fa][a2], ALPHAS)):
                    err[i] += np.sum(((A[b2] - ma) @ w + mz - z[fa][b2]) ** 2)
            (w, ma, mz), = _ridge(A, z[fa], [ALPHAS[int(np.argmin(err))]])
            p[fb] = (Bm - ma) @ w + mz
    return p


def ahat_block(X, z, obs, g, folds, inner):
    """Cross-fitted image-derived age (amendment 1): per outer fold, test rows from a ridge fitted on the training
    rows; training rows from inner out-of-fold ridge fits on the same inner splits. Alpha by grouped inner CV."""
    F = []
    for k, (tr, te) in enumerate(folds):
        trc = tr[obs[tr]]
        sc = StandardScaler().fit(X[trc])
        A = sc.transform(X[trc])
        err = np.zeros(len(ALPHAS))
        for a, b in GroupKFold(3).split(A, z[trc], g[trc]):
            for i, (w, ma, mz) in enumerate(_ridge(A[a], z[trc][a], ALPHAS)):
                err[i] += np.sum(((A[b] - ma) @ w + mz - z[trc][b]) ** 2)
        alpha = ALPHAS[int(np.argmin(err))]
        f = np.zeros(len(z))
        (w, ma, mz), = _ridge(A, z[trc], [alpha])
        f[te] = (sc.transform(X[te]) - ma) @ w + mz
        for a, b in inner[k]:
            fit = tr[a][obs[tr[a]]]
            s2 = StandardScaler().fit(X[fit])
            (w, ma, mz), = _ridge(s2.transform(X[fit]), z[fit], [alpha])
            f[tr[b]] = (s2.transform(X[tr[b]]) - ma) @ w + mz
        F.append(f[:, None] / 100)
    return F


def diag(X, e, z, obs, tr, te):
    c = coords(X, e)
    a, b = tr[obs[tr]], te[obs[te]]
    r = {"sd_train": round(float(c[tr].std(0).mean()), 3), "sd_test": round(float(c[te].std(0).mean()), 3)}
    if c.shape[1] == 1 and z.ndim == 1:
        r.update(corr_train=round(float(np.corrcoef(c[a, 0], z[a])[0, 1]), 3),
                 corr_test=round(float(np.corrcoef(c[b, 0], z[b])[0, 1]), 3))
    return r


def r2_boot(z, p, obs, inv, idx):
    w = obs.astype(float)
    zz = np.where(obs, z, 0.0)
    e2 = np.where(obs, (z - np.nan_to_num(p)) ** 2, 0.0)
    S = lambda v: np.bincount(inv, weights=v)[idx].sum(1)
    n, sz, szz, se = S(w), S(zz), S(zz ** 2), S(e2)
    return 1 - se / (szz - sz ** 2 / n)


def probe_site(Xk, lab, folds):
    """Multinomial logistic (C = 0.01, as E45): held-out log-loss reduction over the training prior (nats) and
    macro one-vs-rest AUROC over classes with >= 5 held-out rows."""
    K = int(lab.max()) + 1
    P, gain, used = np.zeros((len(lab), K)), np.full(len(lab), np.nan), np.zeros(len(lab), bool)
    for k, (tr, te) in enumerate(folds):
        trc, tec = tr[lab[tr] >= 0], te[lab[te] >= 0]
        cls = np.unique(lab[trc])
        tec = tec[np.isin(lab[tec], cls)]
        sc = StandardScaler().fit(Xk[k][trc])
        m = LogisticRegression(C=0.01, max_iter=3000).fit(sc.transform(Xk[k][trc]), lab[trc])
        pr = np.clip(m.predict_proba(sc.transform(Xk[k][tec])), 1e-12, 1)
        prior = np.array([np.mean(lab[trc] == c) for c in cls])
        pos = np.searchsorted(cls, lab[tec])
        gain[tec] = np.log(pr[np.arange(len(tec)), pos]) - np.log(prior[pos])
        P[np.ix_(tec, cls)] = pr
        used[tec] = True
    aucs = [roc_auc_score(lab[used] == c, P[used, c]) for c in range(K) if 5 <= np.sum(lab[used] == c) < used.sum()]
    return {"logloss_gain": round(float(np.nanmean(gain[used])), 4), "macro_auroc": round(float(np.mean(aucs)), 4),
            "n_excluded_unseen_class": int(np.sum(lab >= 0) - used.sum())}


# ---------------------------------------------------------------- main
def run(ds, conc, rung, fold_seed, light, cond=False, minimal=False):
    idx, df, y, g, blocks = prepare(ds)
    blocks = {b: blocks[b] for b in GAINS}
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fold_seed).split(df, y, g))
    inner = [list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], g[tr])) for tr, _ in folds]
    for tr, te in folds:
        assert not set(g[tr]) & set(g[te])
    fold = np.zeros(len(y), int)
    for k, (_, te) in enumerate(folds):
        fold[te] = k
    X = np.load(RESEARCH / f"data/{ds}/emb_{rung}.npy")[idx].astype(np.float64)
    Zc, obs = concept(df, ds, conc)
    za, obs_a = concept(df, ds, "age")
    za = za[:, 0]
    zs, obs_s = concept(df, ds, "site")
    lab_site = np.where(obs_s, zs.argmax(1), -1)
    agecol = blocks["age"]
    wh = [whitening(X[tr]) for tr, _ in folds]

    def zfit(trc, perm_seed=None):
        """Concept rows for an eraser fitted on training rows trc. cond (REGISTRY amendment 3, E50c): residualise on
        the training-fold class means (training labels only); the sham permutes within class."""
        Z = Zc[trc]
        if cond:
            Z = Z - np.vstack([Z[y[trc] == c].mean(0) for c in (0, 1)])[y[trc]]
        if perm_seed is None:
            return Z
        rng = np.random.default_rng(perm_seed)
        if not cond:
            return Z[rng.permutation(len(trc))]
        P = np.arange(len(trc))
        for c in (0, 1):
            ii = np.flatnonzero(y[trc] == c)
            P[ii] = ii[rng.permutation(len(ii))]
        return Z[P]
    arms, meta = {}, {}
    site_probe_arms = {"t0", "T-D1-1.0", "T-D2-0.5", "R-D1-1.0-0", "sham-0"} | (
        {"SR-0", "rest-T"} | {f"T-D1-{t}" for t in DOSES} | {f"T-D2-{t}" for t in DOSES} if conc == "site" else set())

    AH = ahat_block(X, za, obs_a, g, folds, inner)
    AHA = [np.column_stack([f, agecol(tr)]) for f, (tr, _) in zip(AH, folds)]

    def arm(name, Xk, eraser=None):
        fb = {"ahat": AH, "ahat+age": AHA}
        if eraser is not None:
            C = [coords(X, e) for e in eraser]
            fb.update({"coord": C, "coord+age": [np.column_stack([c, agecol(tr)]) for c, (tr, _) in zip(C, folds)]})
            meta.setdefault("eraser_diag", {})[name] = [diag(X, e, Zc[:, 0] if Zc.shape[1] == 1 else Zc, obs, tr, te)
                                                        for e, (tr, te) in zip(eraser, folds)]
        out = corrected(Xk, y, g, folds, inner, blocks, fb)
        pa, pred = probe_age(Xk, za, obs_a, g, folds)
        pa["r2_nested_ci95"] = ci(r2_boot(za, pred, obs_a, inv, bidx))
        pw = probe_age_within_test(Xk, za, obs_a, g, folds)
        pa["r2_within_test"] = round(float(1 - np.sum((za[obs_a] - pw[obs_a]) ** 2) /
                                           np.sum((za[obs_a] - za[obs_a].mean()) ** 2)), 4)
        pa["r2_within_test_ci95"] = ci(r2_boot(za, pw, obs_a, inv, bidx))
        out["probe_age"] = pa
        if name in site_probe_arms:
            out["probe_site"] = probe_site(Xk, lab_site, folds)
        arms[name] = out
        l0 = per_image_ll(y, out["q0"])
        print(ds, conc, rung, fold_seed, name, "LL0 %.5f" % l0.mean(),
              {b: round(1e3 * float((l0 - per_image_ll(y, q)).mean()), 2) for b, q in out["q1"].items()},
              "R2 %.3f" % pa["r2_nested"], out.get("probe_site", ""), flush=True)

    inv, bidx = boot_idx(g, np.random.default_rng(50))
    arm("t0", [X] * 5)
    T = []
    for k, (tr, _) in enumerate(folds):
        trc = tr[obs[tr]]
        T.append(leace_parts(X[trc], zfit(trc)))
    meta["rank_targeted"] = [int(e[3]) for e in T]
    arm("T-D1-1.0", [d1(X, e, 1.0) for e in T], T)
    target = arms["T-D1-1.0"]["inner_ll"]
    ranks, grid = matched_ranks(X, y, g, folds, inner, wh, target, X.shape[1])
    meta.update(rank_matched=ranks, rank_search_grid=grid, rank_search_target=target)
    print(ds, conc, "matched ranks", ranks, flush=True)
    R = [[random_parts(wh[k], ranks[k], np.random.default_rng(51000 + 100 * k + j)) for k in range(5)] for j in range(N_CTRL)]
    for j in range(N_CTRL):
        arm(f"R-D1-1.0-{j}", [d1(X, e, 1.0) for e in R[j]], R[j])
    for j in range(N_CTRL if not minimal else 0):
        e = [random_parts(wh[k], meta["rank_targeted"][k], np.random.default_rng(52000 + 100 * k + j)) for k in range(5)]
        arm(f"SR-{j}", [d1(X, ee, 1.0) for ee in e], e)
    for j in range(N_CTRL):
        e = []
        for k, (tr, _) in enumerate(folds):
            trc = tr[obs[tr]]
            e.append(leace_parts(X[trc], zfit(trc, 53000 + 100 * k + j)))
        arm(f"sham-{j}", [d1(X, ee, 1.0) for ee in e], e)
    if not minimal:  # minimal (E50b robustness runs, STATUS 2026-10-29): primary contrast and sham only
        arm("rest-T", [np.column_stack([d1(X, e, 1.0), coords(X, e)]) for e in T])
        TL = [leace_parts(X[tr[obs[tr]]], zfit(tr[obs[tr]]), shrink=True) for tr, _ in folds]
        arm("T-LW-D1-1.0", [d1(X, e, 1.0) for e in TL], TL)
        for j in range(N_CTRL):
            e = []
            for k, (tr, _) in enumerate(folds):
                trc = tr[obs[tr]]
                e.append(leace_parts(X[trc], zfit(trc, 53000 + 100 * k + j), shrink=True))
            arm(f"shamLW-{j}", [d1(X, ee, 1.0) for ee in e], e)
    if not light:
        arm("rest-R", [np.column_stack([d1(X, e, 1.0), coords(X, e)]) for e in R[0]])
        for t in DOSES[:-1]:
            arm(f"T-D1-{t}", [d1(X, e, t) for e in T])
        for i, t in enumerate(DOSES):
            arm(f"T-D2-{t}", [d2(X, e, t, tr, np.random.default_rng(54000 + 100 * k + i))
                              for k, (e, (tr, _)) in enumerate(zip(T, folds))])
            for j in range(N_CTRL):
                arm(f"R-D2-{t}-{j}", [d2(X, e, t, tr, np.random.default_rng(55000 + 1000 * j + 100 * k + i))
                                      for k, (e, (tr, _)) in enumerate(zip(R[j], folds))])
    tag = f"{conc}_{rung}_f{fold_seed}" + ("_light" if light else "") + ("_cond" if cond else "") + ("_min" if minimal else "")
    out_dir = RESEARCH / f"data/{ds}/e50"
    out_dir.mkdir(parents=True, exist_ok=True)
    save = {"y": y, "groups": np.asarray(g).astype(str), "fold": fold}
    for n, a in arms.items():
        save[f"{n}|l0"] = per_image_ll(y, a["q0"])
        for b, q in a["q1"].items():
            save[f"{n}|l1_{b}"] = per_image_ll(y, q)
        save[f"{n}|score"] = a["s"]
    np.savez_compressed(out_dir / f"{tag}.npz", **save)
    per_arm = {n: {"image_logloss": round(float(per_image_ll(y, a["q0"]).mean()), 5),
                   "image_auroc": round(float(roc_auc_score(y, a["s"])), 4),
                   "image_brier": round(float(brier_score_loss(y, a["q0"])), 5),
                   "fusion_age_logloss": round(float(per_image_ll(y, a["q1"]["age"]).mean()), 5),
                   "delta_x1e3": {b: round(1e3 * float((per_image_ll(y, a["q0"]) - per_image_ll(y, q)).mean()), 3)
                                  for b, q in a["q1"].items()},
                   "probe_age": a["probe_age"], "probe_site": a.get("probe_site"),
                   "inner_image_logloss": [round(v, 5) for v in a["inner_ll"]], "C": a["C"]}
               for n, a in arms.items()}
    return {"n": int(len(y)), "n_groups": int(len(np.unique(g))), "tag": tag, "meta": meta, "arms": per_arm,
            "contrasts": summarise(out_dir / f"{tag}.npz", conc, light)}


# ---------------------------------------------------------------- paired contrasts from the saved per-observation arrays
def summarise(path, conc, light):
    f = np.load(path)
    y, g = f["y"], f["groups"]
    inv, bidx = boot_idx(g, np.random.default_rng(501))
    has = lambda a: f"{a}|l0" in f.files
    l0 = lambda a: f[f"{a}|l0"]
    D = lambda a, b: f[f"{a}|l0"] - f[f"{a}|l1_{b}"]
    mean_arms = lambda fn, arms: np.mean([fn(a) for a in arms], 0)
    R1 = [f"R-D1-1.0-{j}" for j in range(N_CTRL)]
    SH = [f"sham-{j}" for j in range(N_CTRL)]
    SR = [f"SR-{j}" for j in range(N_CTRL)]

    def pc(v):
        bs = boot_mean(v, inv, bidx)
        return {"diff_x1e3": round(1e3 * float(v.mean()), 3), "ci95_x1e3": [round(1e3 * x, 3) for x in ci(bs)],
                "p_one_sided": round(float((1 + np.sum(bs <= 0)) / (len(bs) + 1)), 5)}

    out = {}
    for b in ["age", "site", "sex", "all"]:
        T = D("T-D1-1.0", b)
        out[f"gain_{b}"] = {
            "targeted_minus_matched": pc(T - mean_arms(lambda a: D(a, b), R1)),
            "targeted_minus_sham": pc(T - mean_arms(lambda a: D(a, b), SH)),
            "targeted_minus_t0": pc(T - D("t0", b)),
            "matched_minus_t0": pc(mean_arms(lambda a: D(a, b), R1) - D("t0", b)),
            "matched_each_x1e3": [round(1e3 * float(D(a, b).mean()), 3) for a in R1],
            "sham_each_x1e3": [round(1e3 * float(D(a, b).mean()), 3) for a in SH],
        }
        if has(SR[0]):
            out[f"gain_{b}"]["targeted_minus_same_rank"] = pc(T - mean_arms(lambda a: D(a, b), SR))
    out["primary"] = out[f"gain_{conc}"]["targeted_minus_matched"]
    out["matching_check_image_ll_targeted_minus_matched"] = pc(l0("T-D1-1.0") - mean_arms(l0, R1))
    out["image_ll_targeted_minus_t0"] = pc(l0("T-D1-1.0") - l0("t0"))
    out["image_ll_sham_minus_t0"] = pc(mean_arms(l0, SH) - l0("t0"))
    # restoration
    if has("rest-T"):
        out["restoration_reappend_T"] = {"delta_gain_vs_t0": pc(D("rest-T", conc) - D("t0", conc)),
                                         "image_ll_vs_t0": pc(l0("rest-T") - l0("t0"))}
    trace = D("T-D1-1.0", "coord")
    rise = D("T-D1-1.0", conc) - D("t0", conc)
    out["image_trace"] = {"delta_coord_T": pc(trace), "rise_in_gain_T": pc(rise), "coord_minus_rise": pc(trace - rise),
                          "age_beyond_coord_T": pc(D("T-D1-1.0", "coord+age") - trace),
                          "delta_coord_matched": pc(mean_arms(lambda a: D(a, "coord"), R1))}
    SL = [f"shamLW-{j}" for j in range(N_CTRL)]
    if has("T-LW-D1-1.0"):
        out["secondary_LW"] = {"targeted_minus_matched": pc(D("T-LW-D1-1.0", conc) - mean_arms(lambda a: D(a, conc), R1)),
                               "targeted_minus_shamLW": pc(D("T-LW-D1-1.0", conc) - mean_arms(lambda a: D(a, conc), SL)),
                               "targeted_minus_t0": pc(D("T-LW-D1-1.0", conc) - D("t0", conc)),
                               "image_ll_targeted_minus_matched": pc(l0("T-LW-D1-1.0") - mean_arms(l0, R1))}
    ah_rise = D("T-D1-1.0", "age") - D("t0", "age")
    out["image_derived_age"] = {"delta_ahat_t0": pc(D("t0", "ahat")), "delta_ahat_T": pc(D("T-D1-1.0", "ahat")),
                                "ahat_targeted_minus_matched": pc(D("T-D1-1.0", "ahat") - mean_arms(lambda a: D(a, "ahat"), R1)),
                                "ahat_T_minus_age_rise": pc(D("T-D1-1.0", "ahat") - ah_rise),
                                "age_beyond_ahat_T": pc(D("T-D1-1.0", "ahat+age") - D("T-D1-1.0", "ahat")),
                                "age_beyond_ahat_t0": pc(D("t0", "ahat+age") - D("t0", "ahat"))}
    for name, arms_ in (("targeted", ["T-D1-1.0"]), ("matched", R1)):
        num = mean_arms(lambda a: D(a, conc), arms_) - D("t0", conc)
        den = mean_arms(l0, arms_) - l0("t0")
        bn, bd = boot_mean(num, inv, bidx), boot_mean(den, inv, bidx)
        cd = ci(bd)
        r = {"numerator": pc(num), "denominator": pc(den)}
        if cd[0] > 0:
            r["RF"] = round(float(num.mean() / den.mean()), 3)
            r["RF_ci95"] = [round(x, 3) for x in ci(bn / bd)]
        out[f"restoration_fraction_{name}"] = r
    if not light:
        out["restoration_reappend_R"] = {"delta_gain_vs_t0": pc(D("rest-R", conc) - D("t0", conc)),
                                         "image_ll_vs_t0": pc(l0("rest-R") - l0("t0"))}
        out["dose_D2"] = {str(t): {"targeted_minus_matched": pc(D(f"T-D2-{t}", conc) - mean_arms(
            lambda a: D(a, conc), [f"R-D2-{t}-{j}" for j in range(N_CTRL)])),
            "targeted_minus_t0": pc(D(f"T-D2-{t}", conc) - D("t0", conc)),
            "image_ll_targeted_minus_matched": pc(l0(f"T-D2-{t}") - mean_arms(l0, [f"R-D2-{t}-{j}" for j in range(N_CTRL)]))}
            for t in DOSES}
        out["dose_D1"] = {str(t): {"targeted_minus_t0": pc(D(f"T-D1-{t}", conc) - D("t0", conc)),
                                   "image_ll_minus_t0": pc(l0(f"T-D1-{t}") - l0("t0"))} for t in DOSES}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("datasets", nargs="+")
    ap.add_argument("--concept", default="age")
    ap.add_argument("--rung", default="convnext_tiny")
    ap.add_argument("--fold-seed", type=int, default=0)
    ap.add_argument("--light", action="store_true")
    ap.add_argument("--conditional", action="store_true")
    ap.add_argument("--minimal", action="store_true", help="implies --light; skips same-rank, re-append and LW arms")
    a = ap.parse_args()
    a.light = a.light or a.minimal
    for ds in a.datasets:
        eid = ("E50" if a.concept == "age" else "E51") + f"_{ds}_{a.concept}_{a.rung}_f{a.fold_seed}" + ("_light" if a.light else "") + ("_cond" if a.conditional else "") + ("_min" if a.minimal else "")
        exp = Experiment(eid, "iteration6/REGISTRY.md E50/E51: targeted erasure of an attribute's linear trace raises "
                         "that attribute's added value more than performance-matched random erasure",
                         {"dataset": ds}, {"concept": a.concept, "rung": a.rung, "fold_seed": a.fold_seed,
                                           "inner_fold_seed": 0, "light": a.light, "class_conditional": a.conditional, "minimal": a.minimal, "doses": DOSES, "n_controls": N_CTRL,
                                           "probe_alphas": ALPHAS.tolist(), "site_probe_C": 0.01, "bootstrap_B": 2000,
                                           "seeds": "rank search 50000+1000k+r; matched 51000+100k+j; same-rank "
                                                    "52000+100k+j; sham 53000+100k+j; D2 noise 54000/55000+...; "
                                                    "bootstrap 50 (probes), 501 (contrasts)",
                                           "code_sha256_12": {p.name: sha(p) for p in CODE},
                                           "versions": {"numpy": np.__version__, "sklearn": sklearn.__version__}},
                         a.fold_seed)
        exp.finish(run(ds, a.concept, a.rung, a.fold_seed, a.light, a.conditional, a.minimal),
                   interpretation="See iteration6 reports.")
