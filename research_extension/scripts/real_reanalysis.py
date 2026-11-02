"""REAL-1 (three-objective re-analysis of saved erasure runs), REAL-2 (confirmatory eta test on ViT-L/16) and REAL-3
(representation ladder tables). Reads per-observation arrays saved by research/experiments/e50_dose_restoration.py
(research/data/<set>/e50/<tag>.npz) and leakage probes from research/results/E50_*.json; never refits a model.

Per observation i and arm Z (t0 = unchanged features):
    c_i = l0_i(Z) - l0_i(t0)              image-only cost contribution
    l_i = l1age_i(Z) - l1age_i(t0)        fused (image + age) cost contribution
    m_i = c_i - l_i                       change in age's incremental value
eta = mean(m) / mean(c), with a paired grouped-bootstrap ratio CI (same resamples for numerator and denominator).
Random and sham arms are averaged over their 5 draws per observation.

Usage: python real_reanalysis.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = ROOT.parent / "research"
sys.path.insert(0, str(RES / "experiments"))
from e40_crossfit_baseline import boot_idx, boot_mean  # noqa: E402

OUT = ROOT / "results/real"; OUT.mkdir(parents=True, exist_ok=True)
FIG = ROOT / "figures"
SETS = ["isic2020", "hiba", "msk2020", "ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes", "milk10k_clin"]
LARGE = ["isic2020", "ham10000", "isic2019_nonham", "milk10k_derm", "milk10k_clin"]
DIAG = json.loads((RES / "results/E50_age_label_diag.json").read_text())["results"]


def load(ds, tag):
    p = RES / f"data/{ds}/e50/{tag}.npz"
    return np.load(p) if p.exists() else None


def eid(ds, tag):
    conc = tag.split("_")[0]
    return ("E50" if conc == "age" else "E51") + f"_{ds}_{tag}"


def leak(ds, tag, arm):
    p = RES / f"results/{eid(ds, tag)}.json"
    if not p.exists():
        return np.nan, np.nan
    a = json.loads(p.read_text())["results"]["arms"].get(arm)
    if a is None or a.get("probe_age") is None:
        return np.nan, np.nan
    return a["probe_age"].get("r2_within_test", np.nan), a["probe_age"].get("r2_nested", np.nan)


def arrays(f, arm, conc="age"):
    if arm.endswith("*"):  # mean over the 5 draws
        stem = arm[:-1]
        l0 = np.mean([f[f"{stem}{j}|l0"] for j in range(5)], 0)
        l1 = np.mean([f[f"{stem}{j}|l1_{conc}"] for j in range(5)], 0)
    else:
        l0, l1 = f[f"{arm}|l0"], f[f"{arm}|l1_{conc}"]
    return l0, l1


def three(f, arm, inv, idx, conc="age"):
    L0, L1 = arrays(f, "t0", conc)
    l0, l1 = arrays(f, arm, conc)
    c, l = l0 - L0, l1 - L1
    m = c - l
    bc, bl, bm = boot_mean(c, inv, idx), boot_mean(l, inv, idx), boot_mean(m, inv, idx)
    ci = lambda b: [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    out = {"C": c.mean(), "C_ci": ci(bc), "L": l.mean(), "L_ci": ci(bl), "M": m.mean(), "M_ci": ci(bm)}
    if out["C_ci"][0] > 0:
        out["eta"], out["eta_ci"] = m.mean() / c.mean(), ci(bm / bc)
    else:
        out["eta"], out["eta_ci"] = np.nan, [np.nan, np.nan]
    return out, (bc, bm)


def eta_diff(bt, br):
    """Paired bootstrap of eta_T - eta_R; one-sided p = (1 + #{diff <= 0}) / (B + 1)."""
    d = bt[1] / bt[0] - br[1] / br[0]
    return float(np.median(d)), [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], float((1 + np.sum(d <= 0)) / (len(d) + 1))


def analyse_tag(ds, tag, arms, conc="age"):
    f = load(ds, tag)
    if f is None:
        return []
    inv, idx = boot_idx(f["groups"], np.random.default_rng(601))
    rows, boots = [], {}
    for arm in arms:
        need = f"{arm[:-1]}0|l0" if arm.endswith("*") else f"{arm}|l0"
        if need not in f.files:
            continue
        q, b = three(f, arm, inv, idx, conc)
        boots[arm] = b
        lw, ln = leak(ds, tag, arm.replace("*", "0"))
        rows.append(dict(set=ds, tag=tag, arm=arm, n=len(f["y"]), leak_within_test=lw, leak_trainfold=ln, **{k: v for k, v in q.items()}))
    if "T-D1-1.0" in boots and "R-D1-1.0-*" in boots and not np.isnan(rows[0]["C"]):
        tr = [r for r in rows if r["arm"] == "T-D1-1.0"][0]; rr = [r for r in rows if r["arm"] == "R-D1-1.0-*"][0]
        if tr["C_ci"][0] > 0 and rr["C_ci"][0] > 0:
            med, ci, p = eta_diff(boots["T-D1-1.0"], boots["R-D1-1.0-*"])
        else:
            med, ci, p = np.nan, [np.nan, np.nan], 1.0
        for r in rows:
            if r["arm"] == "T-D1-1.0":
                r.update(eta_T_minus_R=med, eta_T_minus_R_ci=ci, eta_T_minus_R_p=p)
    return rows


ARMS_FULL = ["T-D1-1.0", "T-D1-0.75", "T-D1-0.5", "T-D1-0.25", "T-D2-1.0", "T-D2-0.75", "T-D2-0.5", "T-D2-0.25",
             "R-D1-1.0-*", "SR-*", "sham-*", "rest-T", "T-LW-D1-1.0"]
ARMS_MIN = ["T-D1-1.0", "R-D1-1.0-*", "sham-*"]


def holm(ps):
    order = np.argsort(ps); adj = np.empty(len(ps)); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); adj[i] = run
    return adj


def main():
    allrows = []
    for ds in SETS:
        allrows += analyse_tag(ds, "age_convnext_tiny_f0", ARMS_FULL)
        allrows += analyse_tag(ds, "age_convnext_tiny_f0_light_cond", ["T-D1-1.0", "R-D1-1.0-*", "sham-*", "T-LW-D1-1.0"])
        for tag in ["age_convnext_tiny_f1_light_min", "age_convnext_tiny_f2_light_min", "age_resnet18_f0_light_min",
                    "age_vitl16_swag_f0_light_min", "age_hsv_f0_light_min", "age_vit_b_16_f0_light_min"]:
            allrows += analyse_tag(ds, tag, ARMS_MIN)
    for ds in ["hiba", "ham10000", "pad_ufes", "milk10k_derm"]:
        allrows += analyse_tag(ds, "site_convnext_tiny_f0_light", ["T-D1-1.0", "R-D1-1.0-*", "sham-*"], conc="site")
    df = pd.DataFrame(allrows)
    df.to_json(OUT / "REAL-1_three_objectives.json", orient="records", indent=1)
    flat = df.copy()
    for c in [c for c in flat.columns if c.endswith("_ci")]:
        flat[c + "_lo"] = flat[c].map(lambda v: v[0] if isinstance(v, list) else np.nan)
        flat[c + "_hi"] = flat[c].map(lambda v: v[1] if isinstance(v, list) else np.nan)
        flat = flat.drop(columns=c)
    flat.to_csv(OUT / "REAL-1_three_objectives.csv", index=False)
    # ---- REAL-2 confirmatory family (ViT-L/16)
    conf = flat[(flat.tag == "age_vitl16_swag_f0_light_min") & (flat.arm == "T-D1-1.0") & flat.set.isin(["ham10000", "isic2019_nonham", "isic2020"])]
    if len(conf) == 3:
        ps = conf.eta_T_minus_R_p.fillna(1.0).to_numpy()
        conf = conf.assign(p_holm=holm(ps), reject=holm(ps) <= 0.05)
        conf.to_csv(OUT / "REAL-2_confirmatory.csv", index=False)
        print(conf[["set", "C", "M", "eta", "eta_T_minus_R", "eta_T_minus_R_ci_lo", "eta_T_minus_R_ci_hi", "eta_T_minus_R_p", "p_holm", "reject"]].to_string(index=False))
    else:
        print("REAL-2 incomplete:", len(conf), "of 3 sets available")
    pareto(flat)
    return flat


def pareto(flat):
    sub = flat[(flat.tag == "age_convnext_tiny_f0")]
    sets = [s for s in SETS if s in sub.set.unique()]
    fig, axs = plt.subplots(3, len(sets), figsize=(2.2 * len(sets), 6.4), squeeze=False)
    col = lambda a: "#2a78d6" if a.startswith("T-D2") else ("#1f6f7a" if a.startswith("T-D1") or a.startswith("T-LW") else
                                                          ("#eb6834" if a.startswith("R-") else ("#5b6b75" if a.startswith("sham") else "#9aa5ad")))
    for j, ds in enumerate(sets):
        s = sub[sub.set == ds]
        for _, r in s.iterrows():
            c = col(r.arm)
            axs[0, j].scatter(r.leak_within_test, 1e3 * r.C, s=12, c=c)
            axs[1, j].scatter(r.leak_within_test, 1e3 * r.M, s=12, c=c)
            axs[2, j].errorbar(1e3 * r.C, 1e3 * r.M, xerr=[[1e3 * (r.C - r.C_ci_lo)], [1e3 * (r.C_ci_hi - r.C)]],
                               yerr=[[1e3 * (r.M - r.M_ci_lo)], [1e3 * (r.M_ci_hi - r.M)]], fmt="o", ms=3, c=c, lw=0.6)
        lim = 1e3 * max(s.C.max(), s.M.max(), 1e-3)
        axs[2, j].plot([0, lim], [0, lim], "k--", lw=0.6)
        axs[0, j].set_title(f"{ds}\nage AUROC {DIAG[ds]['age_auroc_for_label']:.2f}", fontsize=7)
        for i in range(3):
            axs[i, j].tick_params(labelsize=5.5)
    axs[0, 0].set_ylabel("C_Y (1e-3)", fontsize=7); axs[1, 0].set_ylabel("M_A (1e-3)", fontsize=7); axs[2, 0].set_ylabel("M_A (1e-3)", fontsize=7)
    for j in range(len(sets)):
        axs[0, j].set_xlabel("within-test age R^2", fontsize=6); axs[1, j].set_xlabel("within-test age R^2", fontsize=6); axs[2, j].set_xlabel("C_Y (1e-3)", fontsize=6)
    fig.suptitle("ConvNeXt-T, fold assignment 0. Teal: LEACE (D1 doses, LW); blue: LEACE information doses (D2); orange: matched random; grey: sham / same-rank. Dashed: M = C.", fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(FIG / "REAL-1_pareto.pdf"); fig.savefig(FIG / "REAL-1_pareto.png", dpi=130)


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
    flat = main()
    show = flat[flat.arm.isin(["T-D1-1.0", "R-D1-1.0-*", "sham-*"])]
    cols = ["set", "tag", "arm", "C", "M", "L", "eta", "eta_ci_lo", "eta_ci_hi", "eta_T_minus_R", "eta_T_minus_R_p", "leak_within_test"]
    print(show[[c for c in cols if c in show]].round(4).to_string(index=False))
