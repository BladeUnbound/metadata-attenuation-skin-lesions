"""Tables and registered decisions for E50/E51 (iteration6/REGISTRY.md): per-set contrasts, gate G1, Holm over the
confirmatory family. Reads results/E50_*.json and E51_*.json only; writes iteration6/E50_tables.md and
results/E50_gate.json.

Usage: python e50_report.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, RESULTS

CONFIRMATORY = [("hiba", "age"), ("msk2020", "age"), ("isic2020", "age"), ("hiba", "site")]
REPLICATION = ["ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes", "milk10k_clin"]
ORDER = ["hiba", "msk2020", "isic2020"] + REPLICATION


def load(eid):
    p = RESULTS / f"{eid}.json"
    return json.loads(p.read_text())["results"] if p.exists() else None


def fmt(c):
    return f"{c['diff_x1e3']:+.2f} [{c['ci95_x1e3'][0]:+.2f}, {c['ci95_x1e3'][1]:+.2f}]"


def holm(ps, alpha=0.05):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [None] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj, [a <= alpha for a in adj]


def table(conc, rung="convnext_tiny", f=0, light=False, cond=False):
    rows = []
    head = ("| set | role | n | rank T / matched (per fold) | image LL t0 → T / matched | Δ t0 | Δ T | Δ matched (mean of 5) | "
            "**T − matched** (p) | T − sham | T − same-rank | T-LW − matched | image LL T − matched |")
    rows += [head, "|" + "---|" * (head.count("|") - 1)]
    for ds in ORDER:
        eid = ("E50" if conc == "age" else "E51") + f"_{ds}_{conc}_{rung}_f{f}" + ("_light" if light else "") + ("_cond" if cond else "")
        r = load(eid)
        if r is None:
            continue
        a, c, m = r["arms"], r["contrasts"], r["meta"]
        g = c[f"gain_{conc}"]
        role = "confirmatory" if (ds, conc) in CONFIRMATORY else ("correlated" if ds == "milk10k_clin" else "replication")
        dm = sum(g["matched_each_x1e3"]) / len(g["matched_each_x1e3"])
        llm = sum(a[f"R-D1-1.0-{j}"]["image_logloss"] for j in range(5)) / 5
        rows.append(f"| {ds} | {role} | {r['n']} | {m['rank_targeted'][0]} / {m['rank_matched']} | "
                    f"{a['t0']['image_logloss']:.4f} → {a['T-D1-1.0']['image_logloss']:.4f} / {llm:.4f} | "
                    f"{a['t0']['delta_x1e3'][conc]:.2f} | {a['T-D1-1.0']['delta_x1e3'][conc]:.2f} | {dm:.2f} | "
                    f"**{fmt(g['targeted_minus_matched'])}** ({g['targeted_minus_matched']['p_one_sided']:.4f}) | "
                    f"{fmt(g['targeted_minus_sham'])} | {fmt(g['targeted_minus_same_rank'])} | "
                    f"{fmt(c['secondary_LW']['targeted_minus_matched'])} | {fmt(c['matching_check_image_ll_targeted_minus_matched'])} |")
    return "\n".join(rows)


def probes(conc="age"):
    rows = ["| set | R² train-fold probe t0 → T | R² within-test-fold t0 → T → T-LW → matched → sham | c test/train SD (T; T-LW) |",
            "|---|---|---|---|"]
    for ds in ORDER:
        r = load(f"E50_{ds}_{conc}_convnext_tiny_f0")
        if r is None:
            continue
        a, dg = r["arms"], r["meta"]["eraser_diag"]
        w = lambda n: a[n]["probe_age"]["r2_within_test"]
        sd = lambda n: sum(x["sd_test"] / x["sd_train"] for x in dg[n]) / len(dg[n])
        rows.append(f"| {ds} | {a['t0']['probe_age']['r2_nested']:.3f} → {a['T-D1-1.0']['probe_age']['r2_nested']:.3f} | "
                    f"{w('t0'):.3f} → {w('T-D1-1.0'):.3f} → {w('T-LW-D1-1.0'):.3f} → {w('R-D1-1.0-0'):.3f} → {w('sham-0'):.3f} | "
                    f"{sd('T-D1-1.0'):.2f}; {sd('T-LW-D1-1.0'):.2f} |")
    return "\n".join(rows)


def doses():
    rows = ["| set | D2 t=0.25 | 0.5 | 0.75 | 1.0 (T − matched, ×10⁻³) | D1 T − t0 at 0.25 / 0.5 / 0.75 / 1 |", "|---|---|---|---|---|---|"]
    for ds in ORDER:
        r = load(f"E50_{ds}_age_convnext_tiny_f0")
        if r is None or "dose_D2" not in r["contrasts"]:
            continue
        c = r["contrasts"]
        d2 = [fmt(c["dose_D2"][t]["targeted_minus_matched"]) for t in ["0.25", "0.5", "0.75", "1.0"]]
        d1 = " / ".join(f"{c['dose_D1'][t]['targeted_minus_t0']['diff_x1e3']:+.2f}" for t in ["0.25", "0.5", "0.75", "1.0"])
        rows.append(f"| {ds} | " + " | ".join(d2) + f" | {d1} |")
    return "\n".join(rows)


def restoration():
    rows = ["| set | re-append: Δ − Δt0 | re-append: LL − LLt0 | RF targeted | RF matched | Δ_â t0 | Δ_â T | â: T − matched | "
            "â(T) − rise | age beyond â (T) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in ORDER:
        r = load(f"E50_{ds}_age_convnext_tiny_f0")
        if r is None:
            continue
        c = r["contrasts"]
        rf = lambda k: (f"{c[k]['RF']:.2f} [{c[k]['RF_ci95'][0]:.2f}, {c[k]['RF_ci95'][1]:.2f}]" if "RF" in c[k] else "n/a (denominator CI ∋ 0)")
        ia = c["image_derived_age"]
        rows.append(f"| {ds} | {fmt(c['restoration_reappend_T']['delta_gain_vs_t0'])} | {fmt(c['restoration_reappend_T']['image_ll_vs_t0'])} | "
                    f"{rf('restoration_fraction_targeted')} | {rf('restoration_fraction_matched')} | {fmt(ia['delta_ahat_t0'])} | "
                    f"{fmt(ia['delta_ahat_T'])} | {fmt(ia['ahat_targeted_minus_matched'])} | {fmt(ia['ahat_T_minus_age_rise'])} | "
                    f"{fmt(ia['age_beyond_ahat_T'])} |")
    return "\n".join(rows)


def specificity(ds):
    """Registered E51 matrix M(E, V) = targeted-minus-matched contrast for eraser E and gain variable V, with the
    paired per-observation difference M(E, E) - M(E, V != E)."""
    import numpy as np
    from e40_crossfit_baseline import B, boot_idx, boot_mean, ci
    files = {"age": RESEARCH / f"data/{ds}/e50/age_convnext_tiny_f0.npz",
             "site": RESEARCH / f"data/{ds}/e50/site_convnext_tiny_f0_light.npz"}
    if not all(f.exists() for f in files.values()):
        return None
    out = {}
    for E, path in files.items():
        f = np.load(path)
        inv, idx = boot_idx(f["groups"], np.random.default_rng(511))
        D = lambda a, v: f[f"{a}|l0"] - f[f"{a}|l1_{v}"]
        C = {v: D("T-D1-1.0", v) - np.mean([D(f"R-D1-1.0-{j}", v) for j in range(5)], 0) for v in ["age", "site"]}
        other = "site" if E == "age" else "age"
        for v in ["age", "site"]:
            bs = boot_mean(C[v], inv, idx)
            out[f"M({E},{v})"] = {"diff_x1e3": round(1e3 * float(C[v].mean()), 3), "ci95_x1e3": [round(1e3 * x, 3) for x in ci(bs)]}
        d = C[E] - C[other]
        bs = boot_mean(d, inv, idx)
        out[f"M({E},{E})-M({E},{other})"] = {"diff_x1e3": round(1e3 * float(d.mean()), 3), "ci95_x1e3": [round(1e3 * x, 3) for x in ci(bs)]}
        out[f"selective_{E}"] = bool(out[f"M({E},{E})"]["ci95_x1e3"][0] > 0 and out[f"M({E},{E})-M({E},{other})"]["ci95_x1e3"][0] > 0)
    return out


def gate():
    out = {}
    p = {}
    for ds, conc in CONFIRMATORY:
        eid = ("E50" if conc == "age" else "E51") + f"_{ds}_{conc}_convnext_tiny_f0" + ("" if conc == "age" else "_light")
        r = load(eid)
        if r is not None:
            c = r["contrasts"][f"gain_{conc}"]
            p[f"{ds}:{conc}"] = {"p": c["targeted_minus_matched"]["p_one_sided"],
                                 "sham_ci_low": c["targeted_minus_sham"]["ci95_x1e3"][0]}
    out["confirmatory_p"] = p
    first = [k for k, v in p.items() if k.endswith(":age") and v["p"] <= 0.05 / 4]
    sham_ok = [k for k in first if p[k]["sham_ci_low"] > 0]
    repl = {}
    for ds in REPLICATION:
        r = load(f"E50_{ds}_age_convnext_tiny_f0")
        if r is not None:
            repl[ds] = r["contrasts"]["gain_age"]["targeted_minus_matched"]["ci95_x1e3"][0] > 0
    out["replication_ci_above_0"] = repl
    if sham_ok:
        out["G1"] = "PROCEED"
    elif first or any(repl.values()):
        out["G1"] = "REVISE"
    else:
        out["G1"] = "STOP"
    out["G1_basis"] = {"confirmatory_age_first_holm_step": first, "with_sham_ci_above_0": sham_ok}
    if len(p) == 4:
        keys = list(p)
        adj, rej = holm([p[k]["p"] for k in keys])
        out["holm"] = {k: {"p": p[k]["p"], "p_holm": round(a, 5), "reject": r_} for k, a, r_ in zip(keys, adj, rej)}
    return out


if __name__ == "__main__":
    g = gate()
    (RESULTS / "E50_gate.json").write_text(json.dumps(g, indent=2))
    txt = ["# E50/E51 tables (generated by experiments/e50_report.py; values ×10⁻³ nats unless stated)", "",
           "## Age erasure, ConvNeXt-T, fold assignment 0", "", table("age"), "",
           "## Residual age decodability", "", probes(), "", "## Dose–response", "", doses(), "",
           "## Restoration and image-derived age", "", restoration(), "", "## Gate", "", "```", json.dumps(g, indent=2), "```"]
    if any((RESULTS / f"E51_{ds}_site_convnext_tiny_f0_light.json").exists() for ds in ORDER):
        txt += ["", "## Site erasure (E51), ConvNeXt-T, fold assignment 0", "", table("site", light=True)]
        sp = {ds: specificity(ds) for ds in ["hiba", "ham10000", "pad_ufes", "milk10k_derm"]}
        txt += ["", "## Specificity matrix (E51; targeted − matched, ×10⁻³ nats)", "", "```", json.dumps(sp, indent=1), "```"]
        g["specificity"] = sp
        (RESULTS / "E50_gate.json").write_text(json.dumps(g, indent=2))
    if any((RESULTS / f"E50_{ds}_age_convnext_tiny_f0_light_cond.json").exists() for ds in ORDER):
        txt += ["", "## E50c (exploratory): class-conditional age erasure, ConvNeXt-T, fold assignment 0", "",
                table("age", light=True, cond=True)]
    (RESEARCH / "iteration6/E50_tables.md").write_text("\n".join(txt))
    print("\n".join(txt))
