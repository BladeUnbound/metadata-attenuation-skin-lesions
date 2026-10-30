"""Tables for the iteration-6 experiments not covered by e50_report.py: E50b robustness runs, E52, E53, E53b, E54
and E55. Reads results/*.json only; skips anything missing; writes iteration6/ITER6_TABLES.md.

Usage: python iteration6_summary.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, RESULTS

SETS = ["hiba", "msk2020", "isic2020", "ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes", "milk10k_clin"]


def load(eid):
    p = RESULTS / f"{eid}.json"
    return json.loads(p.read_text())["results"] if p.exists() else None


def fx(c, k="diff_x1e3", kc="ci95_x1e3"):
    return f"{c[k]:+.2f} [{c[kc][0]:+.2f}, {c[kc][1]:+.2f}]"


def r2(v, c):
    return f"{v:.3f} [{c[0]:.3f}, {c[1]:.3f}]"


def e50b():
    rows = ["| set | rung | fold assignment | n | rank T / matched | **T − matched** (p) | T − sham | T − t0 | "
            "image LL T − matched | image LL t0 → T |", "|---|---|---|---|---|---|---|---|---|---|"]
    for ds in SETS:
        for rung, f in [("convnext_tiny", 0), ("convnext_tiny", 1), ("convnext_tiny", 2), ("resnet18", 0)]:
            r = load(f"E50_{ds}_age_{rung}_f{f}_light_min") or (load(f"E50_{ds}_age_{rung}_f{f}") if f == 0 and rung == "convnext_tiny" else None)
            if r is None:
                continue
            c, a, m = r["contrasts"], r["arms"], r["meta"]
            g = c["gain_age"]
            rows.append(f"| {ds} | {rung} | {f} | {r['n']} | {m['rank_targeted'][0]} / {m['rank_matched']} | "
                        f"**{fx(g['targeted_minus_matched'])}** ({g['targeted_minus_matched']['p_one_sided']:.4f}) | "
                        f"{fx(g['targeted_minus_sham'])} | {fx(g['targeted_minus_t0'])} | "
                        f"{fx(c['matching_check_image_ll_targeted_minus_matched'])} | "
                        f"{a['t0']['image_logloss']:.4f} → {a['T-D1-1.0']['image_logloss']:.4f} |")
    return "\n".join(rows)


def e52():
    out = ["### Age (R² on outer test folds; linear = E50 nested ridge trained on training folds / within-test-fold ridge)", "",
           "| set | arm | MLP R² [CI] | linear R² (train-fold probe) | linear R² (within-test) |", "|---|---|---|---|---|"]
    for ds in SETS:
        r = load(f"E52_{ds}_age")
        if r is None:
            continue
        for arm, v in r["arms"].items():
            lin = v.get("linear_probe_from_E50") or {}
            out.append(f"| {ds} | {arm} | {r2(v['mlp_r2'], v['mlp_r2_ci95'])} | {lin.get('r2_nested', float('nan')):.3f} | "
                       f"{lin.get('r2_within_test', float('nan')):.3f} |")
    out += ["", "### Site (log-loss gain over the training prior, nats; macro one-vs-rest AUROC)", "",
            "| set | arm | MLP gain [CI] | MLP macro AUROC | linear gain | linear macro AUROC |", "|---|---|---|---|---|---|"]
    for ds in SETS:
        r = load(f"E52_{ds}_site")
        if r is None:
            continue
        for arm, v in r["arms"].items():
            lin = v.get("linear_probe_from_E50") or {}
            out.append(f"| {ds} | {arm} | {v['mlp_logloss_gain']:.4f} [{v['mlp_logloss_gain_ci95'][0]:.4f}, "
                       f"{v['mlp_logloss_gain_ci95'][1]:.4f}] | {v['mlp_macro_auroc']:.3f} | "
                       f"{lin.get('logloss_gain', float('nan')):.4f} | {lin.get('macro_auroc', float('nan')):.3f} |")
    return "\n".join(out)


def e53():
    est = ["a_score_fusion", "b_full_feature", "c_hgb", "d_missingness_only"]
    import numpy as np

    def rep(ds, rung):  # (a) vs E40 corrected per-observation delta; the registered 1e-8 check is below solver tolerance
        p = RESEARCH / f"data/{ds}/e53/{rung}.npz"
        if not p.exists():
            return "n/a"
        f, e = np.load(p), np.load(RESEARCH / f"data/{ds}/e40/{rung}_s0.npz")
        return f"max {np.abs((f['a0'] - f['a1']) - e['delta']).max():.1e}"
    out = ["| set | rung | (a) vs E40, max abs Δ_i diff (nats) | (a) score fusion | (b) full-feature | (c) boosted | "
           "(d) missingness only | (b) − (a) | (c) − (a) |", "|---|---|---|---|---|---|---|---|---|"]
    h1 = ["| set | " + " | ".join(est) + " |", "|---|---|---|---|---|"]
    for ds in SETS:
        r = load(f"E53_{ds}")
        if r is None:
            continue
        for rung in ["hsv", "resnet18", "convnext_tiny"]:
            v = r[rung]
            out.append(f"| {ds} | {rung} | {rep(ds, rung)} | " + " | ".join(fx(v[e], "delta_x1e3", "ci95_x1e3") for e in est) + " | "
                       + " | ".join(fx(v[e]["minus_a_x1e3"], "diff", "ci95") for e in ["b_full_feature", "c_hgb"]) + " |")
        h = r["H1_hsv_minus_convnext"]
        h1.append(f"| {ds} | " + " | ".join(f"{fx(h[e])} (p {h[e]['p_one_sided']:.4f})" for e in est) + " |")
    return "\n".join(out + ["", "H1 contrast Δ(HSV) − Δ(ConvNeXt-T) under each estimator (paired, ×10⁻³ nats):", ""] + h1)


def e53b():
    out = ["| set | (b) T − matched (p) | (b) T − t0 | (b) matched − t0 | (b) image LL T − matched | (a) T − matched (E50) | "
           "(b) − (a) | max abs b0 score − E50 |", "|---|---|---|---|---|---|---|---|"]
    for ds in SETS:
        r = load(f"E53b_{ds}")
        if r is None:
            continue
        c = r["contrasts"]
        out.append(f"| {ds} | **{fx(c['b_targeted_minus_matched'])}** ({c['b_targeted_minus_matched']['p_one_sided']:.4f}) | "
                   f"{fx(c['b_targeted_minus_t0'])} | {fx(c['b_matched_minus_t0'])} | "
                   f"{fx(c['b_matching_check_image_ll_targeted_minus_matched'])} | {fx(c['a_targeted_minus_matched_E50'])} | "
                   f"{fx(c['b_minus_a_contrast'])} | {max(r['reproduces_E50_scores_max_abs'].values()):.1e} |")
    return "\n".join(out)


def e54():
    out = ["| set | backbone | frozen Δ | seed | fine-tuned AUROC | fine-tuned Δ [CI] | frozen − fine-tuned [CI] |",
           "|---|---|---|---|---|---|---|"]
    k = lambda v: f"{1e3 * v:+.2f}"
    for ds in ["hiba", "pad_ufes"]:
        r = load(f"E54_{ds}")
        if r is None:
            continue
        for bb, v in r.items():
            for s, p in v["per_seed"].items():
                out.append(f"| {ds} | {bb} | {k(v['frozen_added_value'])} | {s} | {p['auroc']:.3f} | {k(p['added_value'])} "
                           f"[{k(p['added_value_ci95'][0])}, {k(p['added_value_ci95'][1])}] | "
                           f"[{k(p['frozen_minus_finetuned_ci95'][0])}, {k(p['frozen_minus_finetuned_ci95'][1])}] |")
            out.append(f"| {ds} | {bb} | | mean (range) | | {k(v['added_value_mean'])} ({k(v['added_value_range'][0])} to "
                       f"{k(v['added_value_range'][1])}) | |")
    return "\n".join(out + ["", "(×10⁻³ nats; E14b original second stage, seed 0 = E14b.)"])


def e55():
    r = load("E55_shift_rules")
    if r is None:
        return ""
    rules = ["R0_never", "R1_always", "R2_source_validated", "R3_collection_rule"]
    out = []
    for level in ["dataset_level", "within_dataset"]:
        out += [f"### {level.replace('_', ' ')}", "",
                "| target | trained on | n | Δ fused vs image [CI] | AUROC image → fused | oracle uses metadata | R2 uses | "
                "R3 uses (predicted) | regret R0 / R1 / R2 / R3 |", "|---|---|---|---|---|---|---|---|---|"]
        for key, v in r[level].items():
            reg = " / ".join("—" if v["regret_x1e3"][q] is None else f"{v['regret_x1e3'][q]:.2f}" for q in rules)
            out.append(f"| {key} | {', '.join(v['train'])} | {v['n_target']} | {fx(v, 'delta_x1e3', 'delta_ci95_x1e3')} | "
                       f"{v['auroc_image']:.3f} → {v['auroc_fused']:.3f} | {v['oracle_use']} | {v['R2_use']} | "
                       f"{v['R3_use']} ({v['R3_inputs']['predicted_delta']}) | {reg} |")
        s = r[f"{level}_summary"]
        out += ["", f"Mean regret (×10⁻³ nats) over {s['n_targets']} targets: " +
                ", ".join(f"{q} {s['mean_regret_x1e3'][q]}" for q in rules) + f"; oracle uses metadata on {s['n_oracle_use']}.", ""]
    return "\n".join(out + [r["independence_note"]])


if __name__ == "__main__":
    parts = [("E50b: robustness of the primary age-erasure contrast (fold assignment 0 ConvNeXt-T rows are E50a)", e50b),
             ("E52: residual nonlinear decodability", e52), ("E53: estimator robustness (×10⁻³ nats)", e53),
             ("E53b: E50 primary contrast under full-feature fusion (×10⁻³ nats)", e53b),
             ("E54: fine-tuning seed spread", e54), ("E55: dataset shift and decision rules", e55)]
    txt = ["# Iteration 6 tables (generated by experiments/iteration6_summary.py)", ""]
    for title, fn in parts:
        try:
            body = fn()
        except Exception as ex:  # report, never hide
            body = f"(table failed: {type(ex).__name__}: {ex})"
        txt += [f"## {title}", "", body, ""]
    (RESEARCH / "iteration6/ITER6_TABLES.md").write_text("\n".join(txt))
    print("\n".join(txt))
