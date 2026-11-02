"""Synthetic experiments SYN-1 / SYN-2 (EXPERIMENT_REGISTRY.md). Linear-Gaussian latent model with a logistic label,
so exact population quantities of any fitted *linear* transformation are available (scripts/gauss_logit.py).

Latents F = [A, U, G, N_1..N_k, E_1..E_d]. Image channels: c_age = A + sig_u*U, c_diag = G, nuisances N_j.
S = Q diag(scale) [c_age, c_diag, N] + sig_e * E (Q: random d x (2+k) orthonormal). logit = b + bA*A + bU*U + bG*G.
Erasers reuse research/experiments/e50_dose_restoration.py (leace_parts, random_parts) so that the synthetic and the
real-data pipelines are the same code.

Usage: python synth.py SYN-1 | SYN-2
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "research/experiments"))
from gauss_logit import mi_gauss, risk, explained  # noqa: E402
from e45_erasure import whitening  # noqa: E402
from e50_dose_restoration import leace_parts, random_parts  # noqa: E402

REGIMES = {
    "R1_redundant": dict(bA=1.0, bU=0.0, bG=1.0, rAG=0.0),
    "R2_entangled": dict(bA=1.0, bU=0.8, bG=1.0, rAG=0.0),
    "R3_irrelevant": dict(bA=0.0, bU=0.0, bG=1.0, rAG=0.0),
    "R6_strong_age_label": dict(bA=2.5, bU=0.0, bG=0.5, rAG=0.0),
    "R7_shared_component": dict(bA=0.0, bU=0.0, bG=1.0, rAG=0.6),
}
B0, SIG_U, SIG_E = -1.0, 0.5, 0.5
CS = np.logspace(-4, 0, 5)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


# ------------------------------------------------------------------ model
def build(regime, d, rng):
    r = REGIMES[regime]
    k = max(2, d // 4)
    m = 3 + k + d
    Sig = np.eye(m)
    Sig[0, 2] = Sig[2, 0] = r["rAG"]
    Q = np.linalg.qr(rng.standard_normal((d, 2 + k)))[0]
    scale = np.r_[1.0, 1.0, rng.uniform(0.5, 2.0, k)]
    L = np.zeros((2 + k, m))
    L[0, 0], L[0, 1] = 1.0, SIG_U          # c_age = A + sig_u U
    L[1, 2] = 1.0                          # c_diag = G
    L[2:, 3:3 + k] = np.eye(k)             # nuisances
    KS = Q @ np.diag(scale) @ L
    KS[:, 3 + k:] = SIG_E * np.eye(d)      # isotropic noise
    a = np.zeros(m); a[0] = 1.0
    w = np.zeros(m); w[0], w[1], w[2] = r["bA"], r["bU"], r["bG"]
    return dict(Sig=Sig, KS=KS, a=a, w=w, b=B0)


def draw(mod, n, rng):
    m = len(mod["Sig"])
    F = rng.multivariate_normal(np.zeros(m), mod["Sig"], size=n, method="cholesky")
    S = F @ mod["KS"].T
    A = F @ mod["a"]
    y = (rng.random(n) < 1 / (1 + np.exp(-(mod["b"] + F @ mod["w"])))).astype(int)
    return S, A, y


# ------------------------------------------------------------------ transformations (row convention: Z = S @ P + const)
def P_from_parts(e, d):
    mu, Am, Bt, _ = e
    return np.eye(d) - Am @ Bt


def fit_transforms(S, A, y, rng, d, which):
    out = {}
    Ac = A[:, None]
    if "leace" in which:
        e = leace_parts(S, Ac); out["leace"] = (P_from_parts(e, d), e)
    if "leace_lw" in which:
        e = leace_parts(S, Ac, shrink=True); out["leace_lw"] = (P_from_parts(e, d), e)
    if "orth" in which:
        v = ((S - S.mean(0)).T @ (A - A.mean())); v /= np.linalg.norm(v)
        out["orth"] = (np.eye(d) - np.outer(v, v), None)
    if "random_rank1" in which:
        e = random_parts(whitening(S), 1, rng); out["random_rank1"] = (P_from_parts(e, d), e)
    if "sham" in which:
        e = leace_parts(S, Ac[rng.permutation(len(A))]); out["sham"] = (P_from_parts(e, d), e)
    return out


def inner_ll(X, y, seed=0):
    """3-fold inner CV log-loss of a C=1 logistic head (training rows only; used for the matched search)."""
    ll = []
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(X, y):
        sc = StandardScaler().fit(X[tr])
        p = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X[tr]), y[tr]).predict_proba(sc.transform(X[va]))[:, 1]
        p = np.clip(p, 1e-12, 1 - 1e-12)
        ll.append(-(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p)).mean())
    return float(np.mean(ll))


def matched_random(S, y, P_target, rng, d):
    """Smallest random whitened-subspace rank whose inner-CV image log-loss on training rows reaches the targeted one's."""
    target = inner_ll(S @ P_target, y)
    wh = whitening(S)
    grid = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for r in [g for g in grid if g < d] + [d - 1]:
        e = random_parts(wh, r, rng)
        P = P_from_parts(e, d)
        if inner_ll(S @ P, y) >= target:
            return P, e, r
    return P, e, d - 1


# ------------------------------------------------------------------ exact population quantities for Z = S @ P (+ oracle)
def exact(mod, P=None, oracle_beta=None):
    Sig, KS, a, w, b = mod["Sig"], mod["KS"], mod["a"], mod["w"], mod["b"]
    if oracle_beta is not None:
        KZ = KS - np.outer(oracle_beta, a)            # view of S - beta*A (uses A at inference)
    else:
        KZ = P.T @ KS
    RS, RZ = risk(Sig, w, b, KS), risk(Sig, w, b, KZ)
    RSA, RZA = risk(Sig, w, b, np.vstack([KS, a])), risk(Sig, w, b, np.vstack([KZ, a]))
    C, L = RZ - RS, RZA - RSA
    M = C - L
    out = dict(ex_C=C, ex_L=L, ex_M=M, ex_eta=(M / C if C > 1e-9 else np.nan), ex_DeltaS=RS - RSA, ex_DeltaZ=RZ - RZA)
    if oracle_beta is None:
        out.update(ex_IAZ=mi_gauss(Sig, a, KZ), ex_R2AZ=explained(Sig, a, KZ) / float(a @ Sig @ a))
    out.update(ex_IAS=mi_gauss(Sig, a, KS))
    return out


# ------------------------------------------------------------------ held-out estimates
def heldout_ll(Xtr, ytr, Xte, yte, tuned):
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    if tuned:  # E40/E10 fit_tuned analogue: C by 3-fold CV log-loss
        best, bll = None, np.inf
        for C in CS:
            ll = []
            for a, v in KFold(3, shuffle=True, random_state=0).split(Xtr):
                p = np.clip(LogisticRegression(C=C, max_iter=3000).fit(Xtr[a], ytr[a]).predict_proba(Xtr[v])[:, 1], 1e-12, 1 - 1e-12)
                ll.append(-(ytr[v] * np.log(p) + (1 - ytr[v]) * np.log(1 - p)).mean())
            if np.mean(ll) < bll:
                best, bll = C, np.mean(ll)
        C = best
    else:
        C = 1e3
    p = np.clip(LogisticRegression(C=C, max_iter=5000).fit(Xtr, ytr).predict_proba(Xte)[:, 1], 1e-12, 1 - 1e-12)
    return float(-(yte * np.log(p) + (1 - yte) * np.log(1 - p)).mean())


def estimates(Ztr, Atr, ytr, Zte, Ate, yte, base, tuned, key):
    RZ = heldout_ll(Ztr, ytr, Zte, yte, tuned)
    RZA = heldout_ll(np.c_[Ztr, Atr], ytr, np.c_[Zte, Ate], yte, tuned)
    RS, RSA = base[key]
    C, L = RZ - RS, RZA - RSA
    return {f"{key}_C": C, f"{key}_L": L, f"{key}_M": C - L, f"{key}_eta": ((C - L) / C if abs(C) > 1e-4 else np.nan)}


def probes(Ztr, Atr, Zte, Ate, mlp):
    r2 = lambda p, t: float(1 - ((t - p) ** 2).mean() / t.var())
    out = {"probe_ridge_train": r2(Ridge(1.0).fit(Ztr, Atr).predict(Zte), Ate)}
    pw = np.zeros(len(Ate))
    for a, v in KFold(5, shuffle=True, random_state=0).split(Zte):
        pw[v] = Ridge(1.0).fit(Zte[a], Ate[a]).predict(Zte[v])
    out["probe_ridge_within_test"] = r2(pw, Ate)
    if mlp:
        sc = StandardScaler().fit(Ztr)
        net = MLPRegressor(hidden_layer_sizes=(64,), max_iter=200, early_stopping=True, random_state=0)
        out["probe_mlp"] = r2(net.fit(sc.transform(Ztr), Atr).predict(sc.transform(Zte)), Ate)
    return out


def coord_sd_ratio(e, Str, Ste):
    if e is None:
        return np.nan
    mu, Am, _, _ = e
    return float(((Ste - mu) @ Am).std() / ((Str - mu) @ Am).std())


# ------------------------------------------------------------------ one configuration
def run_one(regime, n_train, n_test, d, seed, methods, mlp_probe, extras=False):
    rng = np.random.default_rng(seed)
    mod = build(regime, d, rng)
    Str, Atr, ytr = draw(mod, n_train, rng)
    Ste, Ate, yte = draw(mod, n_test, rng)
    base = {}
    for key, tuned in (("unpen", False), ("pen", True)):
        base[key] = (heldout_ll(Str, ytr, Ste, yte, tuned), heldout_ll(np.c_[Str, Atr], ytr, np.c_[Ste, Ate], yte, tuned))
    T = fit_transforms(Str, Atr, ytr, rng, d, methods)
    if "random_matched" in methods:
        P, e, r = matched_random(Str, ytr, T["leace"][0], rng, d)
        T["random_matched"] = (P, e)
    rows = []
    common = dict(regime=regime, n_train=n_train, n_test=n_test, d=d, seed=seed, n_over_d=n_train / d)
    if extras:
        lw = LedoitWolf().fit(Str)
        true_v = mod["KS"] @ mod["Sig"] @ mod["a"]; true_v /= np.linalg.norm(true_v)
        est_v = (Str - Str.mean(0)).T @ (Atr - Atr.mean()); est_v /= np.linalg.norm(est_v)
        common.update(cond_SS=float(np.linalg.cond(np.cov(Str, rowvar=False))), lw_shrinkage=float(lw.shrinkage_),
                      cos_est_true=float(abs(true_v @ est_v)))
    for name, (P, e) in T.items():
        row = dict(common, method=name)
        row.update(exact(mod, P=P))
        Ztr, Zte = Str @ P, Ste @ P
        for key, tuned in (("unpen", False), ("pen", True)):
            row.update(estimates(Ztr, Atr, ytr, Zte, Ate, yte, base, tuned, key))
        row.update(probes(Ztr, Atr, Zte, Ate, mlp_probe and name in ("leace", "orth", "random_matched", "sham")))
        row["coord_sd_ratio"] = coord_sd_ratio(e, Str, Ste)
        row["rank_removed"] = int(round(d - np.trace(P))) if e is not None or name == "orth" else 0
        rows.append(row)
    if "oracle" in methods:
        beta = (Str - Str.mean(0)).T @ (Atr - Atr.mean()) / ((Atr - Atr.mean()) ** 2).sum()
        row = dict(common, method="oracle_residual")
        row.update(exact(mod, oracle_beta=beta))
        Ztr, Zte = Str - np.outer(Atr, beta), Ste - np.outer(Ate, beta)
        for key, tuned in (("unpen", False), ("pen", True)):
            row.update(estimates(Ztr, Atr, ytr, Zte, Ate, yte, base, tuned, key))
        rows.append(row)
    none = dict(common, method="none")
    none.update(exact(mod, P=np.eye(d)))
    none.update(probes(Str, Atr, Ste, Ate, mlp_probe))
    rows.append(none)
    return rows


def main(exp):
    out_dir = ROOT / "results/synthetic"; out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = ROOT / "configs"; cfg_dir.mkdir(exist_ok=True)
    t0 = time.time()
    if exp == "SYN-1":
        methods = ["leace", "leace_lw", "orth", "random_rank1", "random_matched", "sham", "oracle"]
        jobs = []
        for ri, regime in enumerate(REGIMES):
            for di, d in enumerate([32, 128]):
                for ni, n in enumerate([1000, 5000, 20000]):
                    for rep in range(5):
                        seed = 1000 * ri + 100 * di + 10 * ni + rep
                        jobs.append(delayed(run_one)(regime, n // 2, n // 2, d, seed, methods, n <= 5000))
        cfg = dict(experiment=exp, regimes=REGIMES, b0=B0, sig_u=SIG_U, sig_e=SIG_E, d=[32, 128], n=[1000, 5000, 20000],
                   split="50/50", seeds="1000*regime_idx + 100*d_idx + 10*n_idx + rep, rep 0..4", methods=methods,
                   heads={"unpen": "logistic C=1e3", "pen": f"logistic C in {CS.tolist()} by 3-fold CV"},
                   mlp_probe="hidden (64,), early stopping, n <= 5000, methods leace/orth/random_matched/sham/none")
    else:
        methods = ["leace", "leace_lw", "orth", "random_matched"]
        jobs = []
        for ri, regime in enumerate(["R1_redundant", "R2_entangled"]):
            for di, d in enumerate([64, 256, 768]):
                for qi, q in enumerate([1.25, 1.5, 2, 3, 5, 10, 25]):
                    for rep in range(5):
                        seed = 50000 + 1000 * ri + 100 * di + 10 * qi + rep
                        jobs.append(delayed(run_one)(regime, int(q * d), 10000, d, seed, methods, False, True))
        cfg = dict(experiment=exp, regimes=["R1_redundant", "R2_entangled"], d=[64, 256, 768],
                   n_over_d=[1.25, 1.5, 2, 3, 5, 10, 25], n_test=10000,
                   seeds="50000 + 1000*regime_idx + 100*d_idx + 10*ratio_idx + rep", methods=methods,
                   heads={"unpen": "logistic C=1e3", "pen": f"logistic C in {CS.tolist()} by 3-fold CV"})
    cfg["code_sha256_12"] = {p.name: sha(p) for p in [Path(__file__), HERE / "gauss_logit.py",
                                                     ROOT.parent / "research/experiments/e50_dose_restoration.py"]}
    (cfg_dir / f"{exp}.json").write_text(json.dumps(cfg, indent=2))
    rows = [r for res in Parallel(n_jobs=8 if exp == "SYN-1" else 1, verbose=5)(jobs) for r in res]
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / f"{exp}.csv", index=False)
    print(exp, "rows", len(df), "minutes", round((time.time() - t0) / 60, 1))


if __name__ == "__main__":
    main(sys.argv[1])
