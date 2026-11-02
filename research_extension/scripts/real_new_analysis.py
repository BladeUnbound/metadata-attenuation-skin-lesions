"""Summaries for REAL-4 .. REAL-8 (arrays written by real_new.py) and the REAL-7 3x3 specificity matrix (E50/E51 JSONs).
Same estimands and bootstrap as real_reanalysis.py (grouped paired bootstrap, B = 2000, seed 601). New arms are paired
with saved E50 arms where the outer-test cases are identical (checked: y, groups and t0 losses must match exactly).

Usage: python real_new_analysis.py [oracle|nae|ndscale|joint|spec|sex|transfer|all]
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_reanalysis import ARMS_MIN, OUT, RES, analyse_tag, boot_idx, boot_mean, eta_diff, load  # noqa: E402

ARR = OUT / "arrays"
CI = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]


def tri(L0, L1, l0, l1, inv, idx):
    """C, L, M, eta for one arm against its baseline; returns summary dict and (boot C, boot M)."""
    c, l = l0 - L0, l1 - L1
    m = c - l
    bc, bl, bm = (boot_mean(v, inv, idx) for v in (c, l, m))
    out = dict(C=1e3 * c.mean(), C_ci=[1e3 * x for x in CI(bc)], L=1e3 * l.mean(), L_ci=[1e3 * x for x in CI(bl)],
               M=1e3 * m.mean(), M_ci=[1e3 * x for x in CI(bm)], eta=np.nan, eta_ci=[np.nan, np.nan])
    if CI(bc)[0] > 0:
        out["eta"], out["eta_ci"] = m.mean() / c.mean(), CI(bm / bc)
    return out, (bc, bm)


def check_same(f, g):
    ok = np.array_equal(f["y"], g["y"]) and np.array_equal(f["groups"].astype(str), g["groups"].astype(str))
    ok = ok and np.allclose(f["t0|l0"], g["t0|l0"], atol=1e-12, rtol=0)
    if not ok:
        raise SystemExit("outer-test cases or baselines differ; paired comparison not allowed")


def mean_draws(f, stem, key, n=5):
    return np.mean([f[f"{stem}{j}|{key}"] for j in range(n)], 0)


def oracle():
    rows = []
    for ds in ["pad_ufes", "ham10000", "isic2019_nonham", "milk10k_derm", "isic2020", "hiba"]:
        p = ARR / f"oracle_{ds}.npz"
        if not p.exists():
            continue
        f = np.load(p); meta = json.loads((ARR / f"oracle_{ds}.json").read_text())
        inv, idx = boot_idx(f["groups"], np.random.default_rng(601))
        for arm in ["T", "oracle"]:
            q, _ = tri(f["t0|l0"], f["t0|l1_age"], f[f"{arm}|l0"], f[f"{arm}|l1_age"], inv, idx)
            rows.append(dict(set=ds, arm=arm, **q))
        for arm, lk in meta["leakage"].items():
            rows.append(dict(set=ds, arm=f"leak:{arm}", **lk))
    return rows


def nae():
    rows = []
    f = np.load(ARR / "nae_ham10000.npz"); meta = json.loads((ARR / "nae_ham10000.json").read_text())
    e = load("ham10000", "age_convnext_tiny_f0"); check_same(f, e)
    inv, idx = boot_idx(f["groups"], np.random.default_rng(601))
    qT, bT = tri(f["t0|l0"], f["t0|l1_age"], e["T-D1-1.0|l0"], e["T-D1-1.0|l1_age"], inv, idx)
    rows.append(dict(set="ham10000", arm="LEACE (E50)", **qT))
    for arm in [k[:-len("|l0")] for k in f.files if k.startswith("NAE") and k.endswith("|l0")]:
        q, b = tri(f["t0|l0"], f["t0|l1_age"], f[f"{arm}|l0"], f[f"{arm}|l1_age"], inv, idx)
        d = dict(eta_minus_LEACE=np.nan, eta_minus_LEACE_ci=[np.nan, np.nan], p_eta_gt_LEACE=np.nan)
        if not np.isnan(q["eta"]) and not np.isnan(qT["eta"]):
            med, ci, p = eta_diff(b, bT)           # one-sided p for eta_NAE - eta_LEACE > 0
            d = dict(eta_minus_LEACE=med, eta_minus_LEACE_ci=ci, p_eta_gt_LEACE=p)
        rows.append(dict(set="ham10000", arm=arm, **q, **d, **{f"leak_{k}": v for k, v in meta["leakage"][arm].items()},
                         distortion=meta["distortion"][arm], adv_hist=meta["train_adv"][arm]))
    return rows


def ndscale():
    rows = []
    meta = json.loads((ARR / "ndscale_isic2019_nonham.json").read_text())
    for r in meta:
        tag = f"q{r['q']}_rep{r['rep']}_{r['variant']}"
        f = np.load(ARR / f"ndscale_isic2019_nonham_{tag}.npz")
        inv, idx = boot_idx(f["groups"], np.random.default_rng(601))
        qT, bT = tri(f["t0|l0"], f["t0|l1_age"], f["T|l0"], f["T|l1_age"], inv, idx)
        qR, bR = tri(f["t0|l0"], f["t0|l1_age"], mean_draws(f, "R-", "l0"), mean_draws(f, "R-", "l1_age"), inv, idx)
        match = boot_mean(f["T|l0"] - mean_draws(f, "R-", "l0"), inv, idx)
        diff = eta_diff(bT, bR) if (qT["C_ci"][0] > 0 and qR["C_ci"][0] > 0) else (np.nan, [np.nan, np.nan], 1.0)
        rows.append(dict(q=r["q"], rep=r["rep"], variant=r["variant"], n_train_mean=float(np.mean(r["n_train"])), dim=r["dim"],
                         ranks=r["ranks"], sd_ratio=r["sd_ratio"], **{f"T_{k}": v for k, v in qT.items()},
                         **{f"R_{k}": v for k, v in qR.items()}, eta_diff=diff[0], eta_diff_ci=diff[1], eta_diff_p=diff[2],
                         match_T_minus_R=1e3 * float(np.mean(f["T|l0"] - mean_draws(f, "R-", "l0"))),
                         match_ci=[1e3 * x for x in CI(match)]))
    return rows


def single_arrays(ds):
    """Single-attribute LEACE erasers on identical cases: age (E50a), site (E51 light), sex (E51 sex minimal)."""
    out = {}
    for name, tag in [("age", "age_convnext_tiny_f0"), ("site", "site_convnext_tiny_f0_light"), ("sex", "sex_convnext_tiny_f0_light_min")]:
        f = load(ds, tag)
        if f is not None:
            out[name] = f
    return out


def joint():
    rows = []
    for ds in ["ham10000", "milk10k_derm"]:
        p = ARR / f"joint_{ds}.npz"
        if not p.exists():
            continue
        f = np.load(p); meta = json.loads((ARR / f"joint_{ds}.json").read_text())
        singles = single_arrays(ds)
        inv, idx = boot_idx(f["groups"], np.random.default_rng(601))
        for b in ["age", "sex", "site", "all"]:
            L0, L1 = f["t0|l0"], f[f"t0|l1_{b}"]
            qJ, _ = tri(L0, L1, f["T-joint|l0"], f[f"T-joint|l1_{b}"], inv, idx)
            qR, _ = tri(L0, L1, mean_draws(f, "R-joint-", "l0"), mean_draws(f, "R-joint-", f"l1_{b}"), inv, idx)
            rows.append(dict(set=ds, block=b, arm="T-joint", **qJ)); rows.append(dict(set=ds, block=b, arm="R-joint", **qR))
            # interaction: joint M_b minus the sum over single erasers of their M_b (identical cases required)
            ms, ok = [], True
            for name, g in singles.items():
                try:
                    check_same(f, g)
                except SystemExit:
                    ok = False; break
                ms.append((g["T-D1-1.0|l0"] - g["t0|l0"]) - (g[f"T-D1-1.0|l1_{b}"] - g[f"t0|l1_{b}"]))
            if ok and len(ms) == 3:
                mj = (f["T-joint|l0"] - L0) - (f[f"T-joint|l1_{b}"] - L1)
                inter = mj - np.sum(ms, 0)
                bi = boot_mean(inter, inv, idx)
                cj = f["T-joint|l0"] - L0
                cs = np.sum([g["T-D1-1.0|l0"] - g["t0|l0"] for g in singles.values()], 0)
                bc = boot_mean(cj - cs, inv, idx)
                rows.append(dict(set=ds, block=b, arm="interaction joint-minus-sum", M=1e3 * inter.mean(), M_ci=[1e3 * x for x in CI(bi)],
                                 C=1e3 * (cj - cs).mean(), C_ci=[1e3 * x for x in CI(bc)]))
        rows.append(dict(set=ds, block="meta", arm="T-joint", rank_joint=meta["rank_joint"], rank_matched=meta["rank_matched"],
                         **{f"leak_{k}": v for k, v in meta["leakage_T"].items()}))
    return rows


def spec():
    """3x3 attribute-specificity matrix: eraser (age, site, sex) x gain block (age, site, sex): targeted - matched contrast."""
    rows = []
    for ds in ["ham10000", "milk10k_derm"]:
        for er, eid in [("age", f"E50_{ds}_age_convnext_tiny_f0"), ("site", f"E51_{ds}_site_convnext_tiny_f0_light"),
                        ("sex", f"E51_{ds}_sex_convnext_tiny_f0_light_min")]:
            p = RES / f"results/{eid}.json"
            if not p.exists():
                continue
            c = json.loads(p.read_text())["results"]["contrasts"]
            for b in ["age", "site", "sex", "all"]:
                x = c.get(f"gain_{b}", {}).get("targeted_minus_matched")
                if x:
                    rows.append(dict(set=ds, eraser=er, gain=b, diff_x1e3=x["diff_x1e3"], ci=x["ci95_x1e3"], p=x.get("p_one_sided")))
            mc = c.get("matching_check_image_ll_targeted_minus_matched")
            rows.append(dict(set=ds, eraser=er, gain="matching_check", diff_x1e3=mc["diff_x1e3"], ci=mc["ci95_x1e3"], p=None))
    return rows


def sex():
    """REAL-7 sex eraser on its own: C, L, M, eta for the sex block, as in REAL-1 (separate file; REAL-1 CSV unchanged)."""
    rows = []
    for ds in ["ham10000", "milk10k_derm"]:
        rows += analyse_tag(ds, "sex_convnext_tiny_f0_light_min", ARMS_MIN, conc="sex")
    return rows


def transfer():
    rows = []
    for src, tgt in [("ham10000", "isic2019_nonham"), ("isic2019_nonham", "ham10000"), ("ham10000", "isic2020")]:
        p = ARR / f"transfer_{src}_to_{tgt}.npz"
        if not p.exists():
            continue
        f = np.load(p); meta = json.loads((ARR / f"transfer_{src}_to_{tgt}.json").read_text())
        e = load(tgt, "age_convnext_tiny_f0"); check_same(f, e)
        inv, idx = boot_idx(f["groups"], np.random.default_rng(601))
        bs = {}
        for arm in ["T-transfer", "T-own"]:
            q, bs[arm] = tri(f["t0|l0"], f["t0|l1_age"], f[f"{arm}|l0"], f[f"{arm}|l1_age"], inv, idx)
            rows.append(dict(source=src, target=tgt, arm=arm, **q, **{f"leak_{k}": v for k, v in meta["leakage"][arm].items()}))
        qR, bR = tri(f["t0|l0"], f["t0|l1_age"], np.mean([e[f"R-D1-1.0-{j}|l0"] for j in range(5)], 0),
                     np.mean([e[f"R-D1-1.0-{j}|l1_age"] for j in range(5)], 0), inv, idx)
        rows.append(dict(source=src, target=tgt, arm="R (target E50)", **qR))
        if all(r["C_ci"][0] > 0 for r in rows[-3:]):
            rows[-3].update(zip(["eta_tr_minus_R", "eta_tr_minus_R_ci", "eta_tr_minus_R_p"], eta_diff(bs["T-transfer"], bR)))
            rows[-3].update(zip(["eta_tr_minus_own", "eta_tr_minus_own_ci", "p_tr_gt_own"], eta_diff(bs["T-transfer"], bs["T-own"])))
        rows[-3]["cos_src_vs_target"] = meta["cos_src_vs_target_cov_direction"]
    return rows


FUN = dict(oracle=oracle, nae=nae, ndscale=ndscale, joint=joint, spec=spec, sex=sex, transfer=transfer)

if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40); pd.set_option("display.max_colwidth", 60)
    which = sys.argv[1:] or ["all"]
    for name in (FUN if which == ["all"] else which):
        try:
            rows = FUN[name]()
        except FileNotFoundError as ex:
            print(name, "missing input:", ex); continue
        (OUT / f"REAL-new_{name}.json").write_text(json.dumps(rows, indent=1, default=float))
        print(f"== {name}"); print(pd.DataFrame(rows).round(3).to_string(index=False))
