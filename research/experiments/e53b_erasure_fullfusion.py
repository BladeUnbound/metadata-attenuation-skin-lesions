"""E53b (registered in research/iteration6/REGISTRY.md, 2026-10-29 13:49 IST; gated on G1 = PROCEED, recorded
2026-10-29 20:37 IST): the E50 primary contrast recomputed with E53 estimator (b), full-feature regularised fusion.

Arms rebuilt exactly from E50 (ConvNeXt-T, fold assignment 0): t = 0, targeted T-D1(1) and the 5 matched random
subspaces R-D1(1)-j at E50's logged per-fold ranks and seeds (51000 + 100 k + j). Per arm and outer fold k:
(b0) image only: tuned L2 logistic on the erased features (E10 grid, grouped inner CV), as E50's first stage;
(b1) image + age: the same procedure on [standardised erased features, standardised age block], everything fitted on
the outer training fold. Delta^age_i = l_i(b0) - l_i(b1); primary contrast mean_i[Delta_i(T) - mean_j Delta_i(R_j)].
The b0 scores must reproduce E50's saved first-stage scores (checked). Runs only on sets whose E50 contrast CI is
above 0.

Usage: python e53b_erasure_fullfusion.py <dataset> [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, RESULTS, Experiment
from e10_ham_attenuation import outer_fold, per_image_ll
from e40_crossfit_baseline import boot_idx, boot_mean, ci
from e45_erasure import concept, prepare, whitening
from e50_dose_restoration import CODE, N_CTRL, d1, leace_parts, random_parts, sha
from e53_estimators import full_fit

N_JOBS = 8


def run(ds):
    e50 = json.loads((RESULTS / f"E50_{ds}_age_convnext_tiny_f0.json").read_text())["results"]
    lo = e50["contrasts"]["gain_age"]["targeted_minus_matched"]["ci95_x1e3"][0]
    assert lo > 0, f"{ds}: E50 contrast CI lower bound {lo} is not above 0; E53b does not apply"
    ranks = e50["meta"]["rank_matched"]
    idx, df, y, g, blocks = prepare(ds)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, g))
    X = np.load(RESEARCH / f"data/{ds}/emb_convnext_tiny.npy")[idx].astype(np.float64)
    Zc, obs = concept(df, ds, "age")
    wh = [whitening(X[tr]) for tr, _ in folds]
    T = [leace_parts(X[tr[obs[tr]]], Zc[tr[obs[tr]]]) for tr, _ in folds]
    assert [int(e[3]) for e in T] == e50["meta"]["rank_targeted"]
    arms = {"t0": lambda: [X] * 5, "T-D1-1.0": lambda: [d1(X, e, 1.0) for e in T]}  # built lazily (memory)
    for j in range(N_CTRL):
        arms[f"R-D1-1.0-{j}"] = lambda j=j: [d1(X, random_parts(wh[k], ranks[k], np.random.default_rng(51000 + 100 * k + j)), 1.0)
                                             for k in range(5)]
    ref = np.load(RESEARCH / f"data/{ds}/e50/age_convnext_tiny_f0.npz")
    L, rep, Cs = {}, {}, {}
    for name, build in arms.items():
        Xk = build()
        jobs = [delayed(outer_fold)(Xk[k], y, g, tr, te) for k, (tr, te) in enumerate(folds)]
        jobs += [delayed(full_fit)(Xk[k], blocks["age"](tr), y, g, tr, te) for k, (tr, te) in enumerate(folds)]
        outs = Parallel(n_jobs=N_JOBS)(jobs)
        s, p1 = np.zeros(len(y)), np.zeros(len(y))
        for te, sc, _ in outs[:5]:
            s[te] = sc
        for te, p, _ in outs[5:]:
            p1[te] = p
        rep[name] = float(np.max(np.abs(s - ref[f"{name}|score"])))
        L[name] = (per_image_ll(y, 1 / (1 + np.exp(-s))), per_image_ll(y, p1))
        Cs[name] = {"b0": [float(o[2]) for o in outs[:5]], "b1": [float(o[2]) for o in outs[5:]]}
        print(ds, name, "LL b0 %.5f b1 %.5f" % (L[name][0].mean(), L[name][1].mean()),
              "delta x1e3 %.3f" % (1e3 * (L[name][0] - L[name][1]).mean()), "max |score - E50| %.2e" % rep[name], flush=True)
    assert max(rep.values()) < 1e-3, f"{ds}: b0 scores do not reproduce E50 ({rep})"
    np.savez_compressed(RESEARCH / f"data/{ds}/e50/age_convnext_tiny_f0_e53b.npz", y=y, groups=np.asarray(g).astype(str),
                        **{f"{n}|b0": v[0] for n, v in L.items()}, **{f"{n}|b1": v[1] for n, v in L.items()})
    inv, bidx = boot_idx(g, np.random.default_rng(531))
    D = {n: v[0] - v[1] for n, v in L.items()}
    Rm = np.mean([D[f"R-D1-1.0-{j}"] for j in range(N_CTRL)], 0)
    Rl = np.mean([L[f"R-D1-1.0-{j}"][0] for j in range(N_CTRL)], 0)
    Da = {n: ref[f"{n}|l0"] - ref[f"{n}|l1_age"] for n in arms}
    Ra = np.mean([Da[f"R-D1-1.0-{j}"] for j in range(N_CTRL)], 0)

    def pc(v, p=False):
        bs = boot_mean(v, inv, bidx)
        o = {"diff_x1e3": round(1e3 * float(v.mean()), 3), "ci95_x1e3": [round(1e3 * x, 3) for x in ci(bs)]}
        if p:
            o["p_one_sided"] = round(float((1 + np.sum(bs <= 0)) / (len(bs) + 1)), 5)
        return o
    return {"n": int(len(y)), "ranks_matched": ranks, "rank_targeted": e50["meta"]["rank_targeted"],
            "reproduces_E50_scores_max_abs": rep, "C": Cs,
            "arms": {n: {"image_logloss_b0": round(float(v[0].mean()), 5), "fusion_logloss_b1": round(float(v[1].mean()), 5),
                         "delta_x1e3": round(1e3 * float(D[n].mean()), 3)} for n, v in L.items()},
            "contrasts": {"b_targeted_minus_matched": pc(D["T-D1-1.0"] - Rm, p=True),
                          "b_targeted_minus_t0": pc(D["T-D1-1.0"] - D["t0"]),
                          "b_matched_minus_t0": pc(Rm - D["t0"]),
                          "b_matching_check_image_ll_targeted_minus_matched": pc(L["T-D1-1.0"][0] - Rl),
                          "a_targeted_minus_matched_E50": pc(Da["T-D1-1.0"] - Ra),
                          "b_minus_a_contrast": pc((D["T-D1-1.0"] - Rm) - (Da["T-D1-1.0"] - Ra))}}


if __name__ == "__main__":
    for ds in sys.argv[1:]:
        exp = Experiment(f"E53b_{ds}", "iteration6/REGISTRY.md E53b: E50 primary age-erasure contrast under "
                         "full-feature regularised fusion (estimator b)", {"dataset": ds},
                         {"rung": "convnext_tiny", "fold_seed": 0, "n_controls": N_CTRL, "B": 2000, "bootstrap_seed": 531,
                          "code_sha256_12": {p.name: sha(p) for p in [Path(__file__), *CODE,
                                                                      Path(__file__).parent / "e53_estimators.py"]}}, 0)
        exp.finish(run(ds), interpretation="See iteration6 reports.")
