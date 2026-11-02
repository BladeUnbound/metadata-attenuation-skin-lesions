"""Figures for REAL-1/REAL-3: eta against age-label dependence; representation ladder. Reads results/real/REAL-1_three_objectives.csv."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
f = pd.read_csv(ROOT / "results/real/REAL-1_three_objectives.csv")
diag = json.loads((ROOT.parent / "research/results/E50_age_label_diag.json").read_text())["results"]
LAB = {"isic2020": "ISIC 2020", "hiba": "HIBA", "msk2020": "MSK 2020", "ham10000": "HAM10000", "isic2019_nonham": "ISIC 2019",
       "milk10k_derm": "MILK10k derm", "milk10k_clin": "MILK10k clin", "pad_ufes": "PAD-UFES-20"}
fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8))
s = f[(f.tag == "age_convnext_tiny_f0")]
for arm, c, lab in [("T-D1-1.0", "#2a78d6", "LEACE"), ("R-D1-1.0-*", "#eb6834", "matched random")]:
    q = s[(s.arm == arm) & s.eta.notna()]
    x = q.set.map(lambda d: diag[d]["age_auroc_for_label"])
    axs[0].errorbar(x, q.eta, yerr=[q.eta - q.eta_ci_lo, q.eta_ci_hi - q.eta], fmt="o", c=c, ms=4, lw=0.7, label=lab, capsize=2)
    if arm == "T-D1-1.0":
        for xi, yi, d in zip(x, q.eta, q.set):
            axs[0].annotate(LAB[d], (xi, yi), fontsize=5.5, xytext=(3, 3), textcoords="offset points")
axs[0].axhline(1, color="k", lw=0.6, ls="--"); axs[0].set_xlabel("age-only AUROC for the label", fontsize=7)
axs[0].set_ylabel("substitution efficiency eta = M_A / C_Y", fontsize=7); axs[0].legend(fontsize=6); axs[0].tick_params(labelsize=6)
axs[0].set_title("ConvNeXt-T, fold 0 (PAD-UFES-20: C_Y not detectable)", fontsize=7)
rungs = [("age_hsv_f0_light_min", "HSV"), ("age_resnet18_f0_light_min", "R18"), ("age_convnext_tiny_f0", "CNX-T"), ("age_vit_b_16_f0_light_min", "ViT-B"), ("age_vitl16_swag_f0_light_min", "ViT-L")]
for ds, c, dy in [("ham10000", "#2a78d6", -9), ("isic2019_nonham", "#1f6f7a", 5), ("isic2020", "#5b6b75", -2)]:
    xs, ys, lo, hi, cs = [], [], [], [], []
    for i, (tag, nm) in enumerate(rungs):
        q = f[(f.set == ds) & (f.tag == tag) & (f.arm == "T-D1-1.0")]
        if len(q) and q.eta.notna().all():
            xs.append(i); ys.append(q.eta.iloc[0]); lo.append(q.eta.iloc[0] - q.eta_ci_lo.iloc[0]); hi.append(q.eta_ci_hi.iloc[0] - q.eta.iloc[0]); cs.append(1e3 * q.C.iloc[0])
    axs[1].errorbar(xs, ys, yerr=[lo, hi], fmt="o-", c=c, ms=4, lw=0.8, capsize=2, label=LAB[ds])
    for x, y, cc in zip(xs, ys, cs):
        axs[1].annotate(f"C={cc:.1f}", (x, y), fontsize=5, xytext=(4, dy), textcoords="offset points", color=c)
axs[1].set_xticks(range(len(rungs)), [r[1] for r in rungs], fontsize=6); axs[1].axhline(1, color="k", lw=0.6, ls="--")
axs[1].set_ylabel("LEACE eta", fontsize=7); axs[1].legend(fontsize=6); axs[1].tick_params(labelsize=6)
axs[1].set_title("representation ladder (C_Y in 1e-3 nats)", fontsize=7)
fig.tight_layout(); fig.savefig(ROOT / "figures/REAL-1_eta.pdf"); fig.savefig(ROOT / "figures/REAL-1_eta.png", dpi=140)
print("ok")
