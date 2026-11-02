"""New real-data arms (EXPERIMENT_REGISTRY.md REAL-4 .. REAL-8), all through the strict cross-fitted estimator of
research/experiments/e50_dose_restoration.py (corrected) on identical outer-test cases.

Modes:
  oracle <set>          REAL-4: attribute-conditional residual Z = S - beta*A (positive control) + leakage audit of
                        t0 / LEACE / matched random j=0 / sham j=0 / oracle
  nae <set>             REAL-5: nonlinear adversarial erasers (MSE and age-decile adversaries, lam_A in {2, 8})
  ndscale <set>         REAL-6: subsampled training rows (n_train/d in {1.25, 2, 5, 10}, 3 seeds), raw vs PCA-then-LEACE
  joint <set>           REAL-7: LEACE on [age, sex, site] jointly, with 5 matched random subspaces
  transfer <src> <tgt>  REAL-8: eraser fitted on the source set, applied to the target set
Per-observation arrays go to results/real/arrays/<mode>_<set>.npz with keys "<arm>|l0" and "<arm>|l1_<block>".
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import StratifiedGroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT.parent / "research"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(RES / "experiments"))
from e10_ham_attenuation import per_image_ll  # noqa: E402
from e40_crossfit_baseline import boot_idx, boot_mean  # noqa: E402
from e45_erasure import concept, prepare, whitening  # noqa: E402
from e50_dose_restoration import (corrected, d1, inner_only_ll, leace_parts, matched_ranks,  # noqa: E402
                                  probe_age_within_test, random_parts)
from real_reanalysis import eta_diff  # noqa: E402

ARR = ROOT / "results/real/arrays"; ARR.mkdir(parents=True, exist_ok=True)
N_CTRL = 5


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:12]


def setup(ds, rung="convnext_tiny", fold_seed=0):
    idx, df, y, g, blocks = prepare(ds)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=fold_seed).split(df, y, g))
    inner = [list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], g[tr])) for tr, _ in folds]
    for tr, te in folds:
        assert not set(g[tr]) & set(g[te])
    X = np.load(RES / f"data/{ds}/emb_{rung}.npy")[idx].astype(np.float64)
    Zc, obs = concept(df, ds, "age")
    return dict(ds=ds, df=df, y=y, g=np.asarray(g), blocks=blocks, folds=folds, inner=inner, X=X, a=Zc[:, 0], obs=obs)


def evaluate(st, Xk, blocks=("age",)):
    out = corrected(Xk, st["y"], st["g"], st["folds"], st["inner"], {b: st["blocks"][b] for b in blocks})
    res = {"l0": per_image_ll(st["y"], out["q0"])}
    for b in blocks:
        res[f"l1_{b}"] = per_image_ll(st["y"], out["q1"][b])
    return res


# ------------------------------------------------------------------ leakage audit (outer-training fits only)
def audit(st, Xk, rng_seed=0):
    a, obs, y, folds = st["a"], st["obs"], st["y"], st["folds"]
    n = len(a)
    p_abs, gain, wc_pred = np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)
    for k, (tr, te) in enumerate(folds):
        trc, tec = tr[obs[tr]], te[obs[te]]
        Ztr, Zte = Xk[k][trc], Xk[k][tec]
        med = np.median(a[trc])
        hp = dict(max_iter=200, learning_rate=0.05, early_stopping=True, random_state=rng_seed)
        p_abs[tec] = HistGradientBoostingRegressor(**hp).fit(Ztr, np.abs(a[trc] - med)).predict(Zte)
        edges = np.quantile(a[trc], np.linspace(0, 1, 11)[1:-1])
        btr, bte = np.digitize(a[trc], edges), np.digitize(a[tec], edges)
        clf = HistGradientBoostingClassifier(**hp).fit(Ztr, btr)
        pr = np.clip(clf.predict_proba(Zte), 1e-12, 1)
        prior = np.bincount(btr, minlength=10) / len(btr)
        col = np.searchsorted(clf.classes_, bte)
        ok = clf.classes_[np.clip(col, 0, len(clf.classes_) - 1)] == bte
        lm = np.where(ok, -np.log(pr[np.arange(len(bte)), np.clip(col, 0, pr.shape[1] - 1)]), -np.log(1e-12))
        gain[tec] = -np.log(prior[bte]) - lm
        for c in (0, 1):
            mt, mte = y[trc] == c, y[tec] == c
            wc_pred[tec[mte]] = HistGradientBoostingRegressor(**hp).fit(Ztr[mt], a[trc][mt]).predict(Zte[mte])
    m = obs
    absA = np.abs(a - np.nanmedian(a))
    r2 = lambda p, t: float(1 - np.sum((t - p) ** 2) / np.sum((t - t.mean()) ** 2))
    wc = sum(r2(wc_pred[m & (y == c)], a[m & (y == c)]) * np.mean(y[m] == c) for c in (0, 1))
    pw = probe_age_within_test(Xk, a, obs, st["g"], folds)
    return {"absA_r2": r2(p_abs[m], absA[m]), "decile_gain_frac": float(np.nanmean(gain[m]) / np.log(10)),
            "within_class_r2": float(wc), "ridge_within_test_r2": r2(pw[m], a[m])}


def leace_k(st, conc_matrix=None):
    X, obs, a = st["X"], st["obs"], st["a"]
    Z = a[:, None] if conc_matrix is None else conc_matrix
    return [leace_parts(X[tr[obs[tr]]], Z[tr[obs[tr]]]) for tr, _ in st["folds"]]


def save(mode, ds, arrays, meta):
    y, g = arrays.pop("__y"), arrays.pop("__g")
    np.savez_compressed(ARR / f"{mode}_{ds}.npz", y=y, groups=g, **arrays)
    (ARR / f"{mode}_{ds}.json").write_text(json.dumps(meta, indent=1, default=float))


def put(arrays, arm, res):
    for k, v in res.items():
        arrays[f"{arm}|{k}"] = v


# ------------------------------------------------------------------ modes
def mode_oracle(ds):
    st = setup(ds); X, a, obs, folds = st["X"], st["a"], st["obs"], st["folds"]
    arrays = {"__y": st["y"], "__g": st["g"].astype(str)}
    meta = {"set": ds, "mode": "oracle", "leakage": {}}
    put(arrays, "t0", evaluate(st, [X] * 5)); meta["leakage"]["t0"] = audit(st, [X] * 5)
    T = leace_k(st)
    XT = [d1(X, e, 1.0) for e in T]
    put(arrays, "T", evaluate(st, XT)); meta["leakage"]["T"] = audit(st, XT)
    e50 = json.loads((RES / f"results/E50_{ds}_age_convnext_tiny_f0.json").read_text())["results"]
    ranks = e50["meta"]["rank_matched"]
    wh = [whitening(X[tr]) for tr, _ in folds]
    XR = [d1(X, random_parts(wh[k], ranks[k], np.random.default_rng(51000 + 100 * k)), 1.0) for k in range(5)]
    meta["leakage"]["R0"] = audit(st, XR)
    XS = []
    for k, (tr, _) in enumerate(folds):
        trc = tr[obs[tr]]
        XS.append(d1(X, leace_parts(X[trc], a[trc][np.random.default_rng(53000 + 100 * k).permutation(len(trc))][:, None]), 1.0))
    meta["leakage"]["sham0"] = audit(st, XS)
    XO = []
    for tr, _ in folds:                           # oracle residual: uses A at inference (positive control)
        trc = tr[obs[tr]]
        ac = a[trc] - a[trc].mean()
        beta = (X[trc] - X[trc].mean(0)).T @ ac / (ac @ ac)
        a_imp = np.where(obs, a, np.median(a[trc]))
        XO.append(X - np.outer(a_imp - a[trc].mean(), beta))
    put(arrays, "oracle", evaluate(st, XO))
    meta["rank_matched_from_E50"] = ranks
    save("oracle", ds, arrays, meta)


def mode_nae(ds):
    from nonlin import fit_transform
    st = setup(ds); X, a, obs, y, folds = st["X"], st["a"], st["obs"], st["y"], st["folds"]
    arrays = {"__y": y, "__g": st["g"].astype(str)}
    meta = {"set": ds, "mode": "nae", "leakage": {}, "train_adv": {}}
    put(arrays, "t0", evaluate(st, [X] * 5))
    for adv in ("mse", "decile"):
        for lam in (2.0, 8.0):
            arm = f"NAE-{adv}-{lam}"
            Xk, hist = [], []
            for k, (tr, _) in enumerate(folds):
                trc = tr[obs[tr]]
                f, h = fit_transform(X[trc], a[trc], y[trc], lam_A=lam, lam_Y=0.0, seed=100 + k, adversary=adv)
                Xk.append(f(X)); hist.append(h[-1])
            put(arrays, arm, evaluate(st, Xk))
            meta["leakage"][arm] = audit(st, Xk)
            meta["train_adv"][arm] = hist
            meta.setdefault("distortion", {})[arm] = float(np.mean([((Xk[k][te] - X[te]) ** 2).sum(1).mean() / X.shape[1] for k, (_, te) in enumerate(folds)]))
            print(arm, meta["leakage"][arm], flush=True)
    save("nae", ds, arrays, meta)


def matched_rank_fold(Xb, k, y, g, folds, inner, wh_k, target_k, dmax):
    """Fold-k version of e50 matched_ranks (same doubling then sqrt(2)-refinement rule, same seeds), for fold-specific
    feature spaces (PCA variant)."""
    r, phase, upper = 1, "double", None
    while True:
        e = random_parts(wh_k, r, np.random.default_rng(50000 + 1000 * k + r))
        Xk = [None] * len(folds); Xk[k] = d1(Xb, e, 1.0)
        hit = inner_only_ll(Xk, y, g, folds, inner, [k])[k] >= target_k
        if phase == "refine":
            return r if hit else upper
        if hit:
            rr = int(round(r / np.sqrt(2)))
            if r >= 4 and rr > r // 2:
                phase, upper, r = "refine", r, rr
                continue
            return r
        if 2 * r >= dmax:
            return r
        r *= 2


def subsample(tr, g, n_target, rng):
    groups = np.unique(g[tr]); rng.shuffle(groups)
    pos = {gg: i for i, gg in enumerate(groups)}
    order = np.argsort([pos[x] for x in g[tr]], kind="stable")
    keep = tr[order]
    # take whole groups until n_target rows are reached
    gi = np.array([pos[x] for x in g[keep]])
    last_group = gi[min(n_target, len(keep)) - 1]
    return np.sort(keep[gi <= last_group])


def mode_ndscale(ds):
    st = setup(ds); X, a, obs, y, g = st["X"], st["a"], st["obs"], st["y"], st["g"]
    d = X.shape[1]
    rows = []
    for q in (1.25, 2, 5, 10):
        for rep in range(3):
            rng = np.random.default_rng(70000 + 100 * int(q * 4) + rep)
            folds = [(subsample(tr, g, int(q * d), rng), te) for tr, te in st["folds"]]
            inner = [list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(tr, y[tr], g[tr])) for tr, _ in folds]
            sub = dict(st, folds=folds, inner=inner)
            for variant in ("raw", "pca"):
                if variant == "pca":
                    pcs = [PCA(n_components=max(8, int(q * d / 10)), random_state=0).fit(X[tr]) for tr, _ in folds]
                    Xb = [p.transform(X) for p in pcs]
                else:
                    Xb = [X] * 5
                arrays = {"__y": y, "__g": g.astype(str)}
                put(arrays, "t0", evaluate(sub, Xb))
                T = [leace_parts(Xb[k][tr[obs[tr]]], a[tr[obs[tr]]][:, None]) for k, (tr, _) in enumerate(folds)]
                XT = [d1(Xb[k], T[k], 1.0) for k in range(5)]
                res_T = corrected(XT, y, g, folds, inner, {"age": st["blocks"]["age"]})
                put(arrays, "T", {"l0": per_image_ll(y, res_T["q0"]), "l1_age": per_image_ll(y, res_T["q1"]["age"])})
                wh = [whitening(Xb[k][tr]) for k, (tr, _) in enumerate(folds)]
                dmax = Xb[0].shape[1]
                ranks = [matched_rank_fold(Xb[k], k, y, g, folds, inner, wh[k], res_T["inner_ll"][k], dmax) for k in range(5)]
                for j in range(N_CTRL):
                    XR = [d1(Xb[k], random_parts(wh[k], ranks[k], np.random.default_rng(51000 + 100 * k + j)), 1.0) for k in range(5)]
                    put(arrays, f"R-{j}", evaluate(sub, XR))
                sd = [float(((X_te := (Xb[k][te] - T[k][0])) @ T[k][1]).std() / ((Xb[k][tr] - T[k][0]) @ T[k][1]).std()) for k, (tr, te) in enumerate(folds)]
                tag = f"q{q}_rep{rep}_{variant}"
                np.savez_compressed(ARR / f"ndscale_{ds}_{tag}.npz", y=y, groups=g.astype(str), **{k: v for k, v in arrays.items() if not k.startswith("__")})
                rows.append(dict(q=q, rep=rep, variant=variant, n_train=[len(tr) for tr, _ in folds], dim=dmax, ranks=ranks, sd_ratio=float(np.mean(sd))))
                print(tag, rows[-1], flush=True)
    (ARR / f"ndscale_{ds}.json").write_text(json.dumps(rows, indent=1))


def mode_joint(ds):
    st = setup(ds); X, obs, y, g, df, folds = st["X"], st["obs"], st["y"], st["g"], st["df"], st["folds"]
    zA, oA = concept(df, ds, "age"); zS, oS = concept(df, ds, "sex"); zT, oT = concept(df, ds, "site")
    Z = np.c_[zA, zS, zT]; ob = oA & oS & oT
    blocks = ("age", "sex", "site", "all")
    arrays = {"__y": y, "__g": g.astype(str)}
    put(arrays, "t0", evaluate(st, [X] * 5, blocks))
    T = [leace_parts(X[tr[ob[tr]]], Z[tr[ob[tr]]]) for tr, _ in folds]
    XT = [d1(X, e, 1.0) for e in T]
    res_T = corrected(XT, y, g, folds, st["inner"], {b: st["blocks"][b] for b in blocks})
    put(arrays, "T-joint", {"l0": per_image_ll(y, res_T["q0"]), **{f"l1_{b}": per_image_ll(y, res_T["q1"][b]) for b in blocks}})
    wh = [whitening(X[tr]) for tr, _ in folds]
    ranks, _ = matched_ranks(X, y, g, folds, st["inner"], wh, res_T["inner_ll"], X.shape[1])
    for j in range(N_CTRL):
        XR = [d1(X, random_parts(wh[k], ranks[k], np.random.default_rng(51000 + 100 * k + j)), 1.0) for k in range(5)]
        put(arrays, f"R-joint-{j}", evaluate(st, XR, blocks))
    meta = {"set": ds, "mode": "joint", "rank_joint": [int(e[3]) for e in T], "rank_matched": ranks,
            "leakage_T": audit(st, XT)}
    save("joint", ds, arrays, meta)


def mode_transfer(src, tgt):
    s, t = setup(src), setup(tgt)
    e = leace_parts(s["X"][s["obs"]], s["a"][s["obs"]][:, None])          # fitted on all source rows with age
    arrays = {"__y": t["y"], "__g": t["g"].astype(str)}
    put(arrays, "t0", evaluate(t, [t["X"]] * 5))
    XT = [d1(t["X"], e, 1.0)] * 5
    put(arrays, "T-transfer", evaluate(t, XT))
    Tt = leace_k(t)
    XTt = [d1(t["X"], ee, 1.0) for ee in Tt]
    put(arrays, "T-own", evaluate(t, XTt))
    xs, as_ = s["X"][s["obs"]], s["a"][s["obs"]]
    u_src = (xs - xs.mean(0)).T @ (as_ - as_.mean()); u_src /= np.linalg.norm(u_src)
    cos = []
    for tr, _ in t["folds"]:
        trc = tr[t["obs"][tr]]
        u = (t["X"][trc] - t["X"][trc].mean(0)).T @ (t["a"][trc] - t["a"][trc].mean()); cos.append(float(abs(u_src @ u) / np.linalg.norm(u)))
    meta = {"source": src, "target": tgt, "mode": "transfer", "cos_src_vs_target_cov_direction": cos,
            "leakage": {"t0": audit(t, [t["X"]] * 5), "T-transfer": audit(t, XT), "T-own": audit(t, XTt)}}
    save("transfer", f"{src}_to_{tgt}", arrays, meta)


if __name__ == "__main__":
    t0 = time.time()
    mode = sys.argv[1]
    cfg = {"mode": mode, "args": sys.argv[2:], "code_sha256_12": {p.name: sha(p) for p in [Path(__file__), HERE / "nonlin.py", RES / "experiments/e50_dose_restoration.py"]},
           "started": time.strftime("%Y-%m-%d %H:%M %Z")}
    (ROOT / "configs" / f"REAL_{mode}_{'_'.join(sys.argv[2:])}.json").write_text(json.dumps(cfg, indent=1))
    {"oracle": mode_oracle, "nae": mode_nae, "ndscale": mode_ndscale, "joint": mode_joint, "transfer": mode_transfer}[mode](*sys.argv[2:])
    print("done", mode, sys.argv[2:], "minutes", round((time.time() - t0) / 60, 1))
