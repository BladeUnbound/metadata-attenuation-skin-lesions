"""SYN-3 (EXPERIMENT_REGISTRY.md): nonlinear age encoding, nonlinear and task-aware attribute-free erasers, and adaptive
post-hoc probes.

Three splits:
  - fit (8k): fit the transformation;
  - head (8k): fit diagnosis heads and leakage probes after the transformation is frozen;
  - test (20k): evaluate.

Regimes (d = 32, linear mixing Q of channels plus isotropic noise):
  - L1: linear redundant (c_age = A + 0.5 U, Y <- A, G);
  - L2: linear entangled (L1 with Y <- U as well);
  - N1: nonlinear redundant (L1 plus channels A^2 - 1 and cos 2A, which have no linear covariance with A);
  - N2: nonlinear entangled (N1 with Y <- U).

Methods:
  - none;
  - LEACE;
  - NAE (adversarial, lam_Y = 0);
  - TP-img (adversarial + image-only diagnosis head);
  - TP-fused (adversarial + head on [z, A]);
  - sham (NAE against permuted A).

Usage: python synth3.py
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "research/experiments"))
from e50_dose_restoration import leace_parts  # noqa: E402
from nonlin import fit_transform  # noqa: E402

REG = {"L1_lin_redundant": dict(bU=0.0, nonlin=False), "L2_lin_entangled": dict(bU=0.8, nonlin=False),
       "N1_nonlin_redundant": dict(bU=0.0, nonlin=True), "N2_nonlin_entangled": dict(bU=0.8, nonlin=True)}
LAMS = [0.5, 2.0, 8.0]
LAMS_B = [0.5, 2.0, 8.0, 32.0]
EXTENDED = len(sys.argv) > 1 and sys.argv[1] in ("SYN-3b", "SYN-3c")  # parent process only; passed to workers explicitly
DECILE = len(sys.argv) > 1 and sys.argv[1] == "SYN-3c"
CS = np.logspace(-4, 0, 5)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def make(regime, n, rng, Q, scale):
    r = REG[regime]
    A = rng.standard_normal(n); U = rng.standard_normal(n); G = rng.standard_normal(n)
    ch = [A + 0.5 * U, G]
    if r["nonlin"]:
        ch += [(A ** 2 - 1) / np.sqrt(2), np.cos(2 * A) / 0.5]
    k = Q.shape[1] - len(ch)
    N = rng.standard_normal((n, k))
    C = np.c_[np.column_stack(ch), N] * scale
    S = C @ Q.T + 0.5 * rng.standard_normal((n, Q.shape[0]))
    y = (rng.random(n) < 1 / (1 + np.exp(-(-1.0 + 1.0 * A + r["bU"] * U + 1.0 * G)))).astype(int)
    return S, A, y


def ll(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def head_risk(Xh, yh, Xt, yt, kind):
    if kind == "logit_tuned":
        sc = StandardScaler().fit(Xh); Xh, Xt = sc.transform(Xh), sc.transform(Xt)
        best, bll = None, np.inf
        for C in CS:
            l = []
            for a, v in KFold(3, shuffle=True, random_state=0).split(Xh):
                l.append(ll(LogisticRegression(C=C, max_iter=3000).fit(Xh[a], yh[a]).predict_proba(Xh[v])[:, 1], yh[v]))
            if np.mean(l) < bll:
                best, bll = C, np.mean(l)
        return ll(LogisticRegression(C=best, max_iter=3000).fit(Xh, yh).predict_proba(Xt)[:, 1], yt)
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
                                       l2_regularization=1.0, early_stopping=True, random_state=0)
    return ll(m.fit(Xh, yh).predict_proba(Xt)[:, 1], yt)


def leakage(Zh, Ah, yh, Zt, At, yt, ext=False):
    r2 = lambda p, t: float(1 - ((t - p) ** 2).mean() / t.var())
    sc = StandardScaler().fit(Zh); Zh_, Zt_ = sc.transform(Zh), sc.transform(Zt)
    out = {"leak_ridge": r2(Ridge(1.0).fit(Zh_, Ah).predict(Zt_), At)}
    mlp = MLPRegressor(hidden_layer_sizes=(128, 128), max_iter=300, early_stopping=True, random_state=0).fit(Zh_, Ah)
    pm = mlp.predict(Zt_)
    out["leak_mlp"] = r2(pm, At)
    hgb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0).fit(Zh, Ah)
    ph = hgb.predict(Zt)
    out["leak_hgb"] = r2(ph, At)
    # conditional leakage: within-class R^2 of a probe fitted within each class (complementary to total leakage)
    wc = []
    for c in (0, 1):
        mh, mt = yh == c, yt == c
        m2 = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0).fit(Zh[mh], Ah[mh])
        wc.append(r2(m2.predict(Zt[mt]), At[mt]) * mt.mean())
    out["leak_hgb_within_class"] = float(sum(wc))
    if ext:  # SYN-3b: non-monotone and distributional leakage
        habs = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0).fit(Zh, np.abs(Ah))
        out["leak_hgb_absA"] = r2(habs.predict(Zt), np.abs(At))
        edges = np.quantile(Ah, np.linspace(0, 1, 11)[1:-1])
        bh, bt = np.digitize(Ah, edges), np.digitize(At, edges)
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, early_stopping=True, random_state=0).fit(Zh, bh)
        pr = np.clip(clf.predict_proba(Zt), 1e-12, 1)
        prior = np.bincount(bh, minlength=10) / len(bh)
        ll_m = -np.log(pr[np.arange(len(bt)), np.searchsorted(clf.classes_, bt)]).mean()
        ll_0 = -np.log(prior[bt]).mean()
        out["leak_decile_gain_frac"] = float((ll_0 - ll_m) / np.log(10))
    return out


def run(regime, seed, ext=False, dec=False):
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):   # one BLAS/OpenMP thread per worker (SYN-2 crashed under oversubscription)
        return _run(regime, seed, ext, dec)


def _run(regime, seed, ext=False, dec=False):
    rng = np.random.default_rng(seed)
    d = 32
    nch = 2 + (2 if REG[regime]["nonlin"] else 0) + 6
    Q = np.linalg.qr(rng.standard_normal((d, nch)))[0]
    scale = np.r_[np.ones(nch - 6), rng.uniform(0.5, 2, 6)]
    Sf, Af, yf = make(regime, 8000, rng, Q, scale)
    Sh, Ah, yh = make(regime, 8000, rng, Q, scale)
    St, At, yt = make(regime, 20000, rng, Q, scale)
    T = {"none": lambda S: S}
    e = leace_parts(Sf, Af[:, None])
    T["leace"] = lambda S, e=e: S - ((S - e[0]) @ e[1]) @ e[2]
    hist = {}
    for lam in (LAMS_B if ext else LAMS):
        methods = (("NAEdec", 0.0, "image"), ("TPfuseddec", 1.0, "fused")) if dec else (("NAE", 0.0, "image"), ("TPimg", 1.0, "image"), ("TPfused", 1.0, "fused"))
        for name, lamY, task in methods:
            f, h = fit_transform(Sf, Af, yf, lam_A=lam, lam_Y=lamY, task=task, seed=seed, adversary="decile" if dec else "mse")
            T[f"{name}_{lam}"] = f; hist[f"{name}_{lam}"] = h[-1]
    if not dec:
        f, h = fit_transform(Sf, Af[rng.permutation(len(Af))], yf, lam_A=8.0, seed=seed)
        T["sham_NAE"] = f; hist["sham_NAE"] = h[-1]
    base = {k: (head_risk(Sh, yh, St, yt, k), head_risk(np.c_[Sh, Ah], yh, np.c_[St, At], yt, k)) for k in ("logit_tuned", "hgb")}
    rows = []
    for name, f in T.items():
        Zh, Zt = f(Sh), f(St)
        row = dict(regime=regime, seed=seed, method=name, train_adv_mse_capped=hist.get(name, np.nan),
                   distortion=float(((Zt - St) ** 2).sum(1).mean() / d))
        for k in ("logit_tuned", "hgb"):
            RZ = head_risk(Zh, yh, Zt, yt, k)
            RZA = head_risk(np.c_[Zh, Ah], yh, np.c_[Zt, At], yt, k)
            RS, RSA = base[k]
            C, L = RZ - RS, RZA - RSA
            row.update({f"{k}_RZ": RZ, f"{k}_RZA": RZA, f"{k}_C": C, f"{k}_L": L, f"{k}_M": C - L,
                        f"{k}_eta": (C - L) / C if abs(C) > 1e-3 else np.nan})
        row.update(leakage(Zh, Ah, yh, Zt, At, yt, ext))
        rows.append(row)
    return rows


if __name__ == "__main__":
    t0 = time.time()
    cfg = dict(experiment=sys.argv[1] if len(sys.argv) > 1 else "SYN-3", adversary_target="age deciles, CE" if DECILE else "signed age, MSE", regimes=REG, lams=(LAMS_B if EXTENDED else LAMS), lam_D=1.0, lam_Y=1.0, epochs=60, batch=256, lr=1e-3,
               adv_steps=3, adversary="MLP 2x128 (MSE, capped at 1 in the transform loss)", head_in_transform="MLP 2x64",
               splits={"fit": 8000, "head": 8000, "test": 20000}, d=32, seeds=[0, 1, 2],
               eval_heads=["logistic tuned (C grid, 3-fold)", "HGB"], probes=["ridge", "MLP 2x128", "HGB", "HGB within class"],
               code_sha256_12={p.name: sha(p) for p in [Path(__file__), HERE / "nonlin.py"]})
    (ROOT / f"configs/{cfg['experiment']}.json").write_text(json.dumps(cfg, indent=2))
    regs = ["N1_nonlin_redundant", "N2_nonlin_entangled"] if DECILE else list(REG)
    jobs = [delayed(run)(r, s, EXTENDED, DECILE) for r in regs for s in (0, 1, 2)]
    rows = [x for res in Parallel(n_jobs=6, verbose=5)(jobs) for x in res]
    import pandas as pd
    (ROOT / "results/synthetic").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ROOT / f"results/synthetic/{cfg['experiment']}.csv", index=False)
    print(cfg["experiment"], "rows", len(rows), "minutes", round((time.time() - t0) / 60, 1))
