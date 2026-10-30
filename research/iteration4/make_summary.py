"""Consolidated overnight summary: regenerated after every queue job.
Lists each experiment record produced since iteration 4 began, with runtime and headline numbers.
Headline extraction is deliberately simple; the JSON in research/results/ is the source of truth.
"""
import json
from datetime import datetime
from pathlib import Path

RES = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parent / "SUMMARY.md"
WATCH = ["E13_hiba", "E13_msk2020", "E13_milk10k_derm", "E13_milk10k_clin", "E22_paired_modality",
         "E17_retained_fraction", "E21_complete_case_wave2", "E30_recoverability_utility", "E32_permutation_controls",
         "E33_shift", "E34_collection_rule", "E31a_family_size", "E31b_scaling_frozen", "E31c_ft_fraction",
         "E18_foundation_rung", "E14_finetune", "E14b_pad_ufes", "E14b_hiba", "E14b_isic2019_nonham",
         "E14b_milk10k_derm", "E19_resolution_ladder", "E31d_ft_fraction_ext"]


def headline(eid, r):
    if eid.startswith("E13_"):
        t = r.get("confirmatory_tests", {})
        s0 = r["seeds"].get("0", {})
        av = {k: round(1000 * v["av_all"], 2) for k, v in s0.items()}
        return f"added value ×10⁻³ by rung {av}; H1 p={t.get('H1', {}).get('p_one_sided')}; H2 p={t.get('H2', {}).get('p_one_sided')}"
    if eid == "E22_paired_modality":
        p1, p2 = r.get("P1_milk10k", {}), r.get("P2_msk2020", {})
        return (f"P1: rho derm {p1.get('dermoscopic', {}).get('rho')} vs clin {p1.get('clinical', {}).get('rho')}, "
                f"diff {p1.get('rho_clin_minus_derm')} CI {p1.get('ci95')} supported={p1.get('P1_supported')}; "
                f"P2 retained {p2.get('retained')} supported={p2.get('P2_supported')}")
    if eid == "E30_recoverability_utility":
        return (f"rows {r.get('n_rows_used')}/{r.get('n_rows_total')}; spearman(within,retained) "
                f"{r.get('spearman_recov_within_vs_retained')}; partial {r.get('partial_spearman_within_given_Ivar_auroc')} "
                f"CI {r.get('partial_spearman_dataset_cluster_ci95')}; LODO MAE ridge {r.get('lodo_mae_ridge')} vs median "
                f"{r.get('lodo_mae_median_baseline')}")
    if eid == "E34_collection_rule":
        return "; ".join(f"{m}: MAE {r[m]['mae']:.5f}, spearman {r[m]['spearman']}" for m in ("pred_b0", "pred_b1", "pred_ridge") if m in r)
    if eid == "E32_permutation_controls":
        parts = []
        for d, x in r.items():
            c = x.get("complete_case", {}).get("convnext_tiny", {})
            parts.append(f"{d}: matched {c.get('matched')} perm {c.get('perm_mean')} p {c.get('p_empirical')}")
        return "; ".join(parts)
    if eid == "E33_shift":
        tr = r.get("transfer", {})
        return "; ".join(f"{k}: Δ {v['added_value']} ΔAUROC {v['delta_auroc']}" for k, v in list(tr.items())[:12])
    return json.dumps(r)[:400]


def main():
    lines = [f"# Iteration 4 summary (auto-generated {datetime.now():%Y-%m-%d %H:%M})", "",
             "Generated from research/results/*.json after each overnight job; the JSON files are the source of truth.",
             "Experiments E30–E34 are exploratory (see REGISTRY_E30_E34.md); E22 and E13 second wave are pre-registered.", "",
             "| Experiment | Finished (UTC) | Runtime (min) | Headline |", "|---|---|---|---|"]
    for eid in WATCH:
        p = RES / f"{eid}.json"
        if not p.exists():
            lines.append(f"| {eid} | not run | | |")
            continue
        j = json.loads(p.read_text())
        try:
            h = headline(eid, j["results"])
        except Exception as e:
            h = f"(summary error: {e})"
        lines.append(f"| {eid} | {j.get('started', '')} | {round(j.get('runtime_s', 0) / 60, 1)} | {h.replace('|', '/')} |")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
