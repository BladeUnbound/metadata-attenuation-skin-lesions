"""Tables and figures for SYN-1/SYN-2 (and SYN-3 when present). Writes results/synthetic/*_summary.csv and figures/.

Usage: python analyse_syn.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RS, FG = ROOT / "results/synthetic", ROOT / "figures"
FG.mkdir(exist_ok=True)
COL = {"leace": "#2a78d6", "leace_lw": "#5b9bd5", "orth": "#1f6f7a", "random_rank1": "#9aa5ad", "random_matched": "#eb6834",
       "sham": "#5b6b75", "oracle_residual": "#2f9e44", "none": "#1d2b33"}


def syn1():
    df = pd.read_csv(RS / "SYN-1.csv")
    m = df[df.method != "none"]
    g = m.groupby(["regime", "method", "n_train", "d"])
    summ = g.agg(ex_C=("ex_C", "mean"), ex_M=("ex_M", "mean"), ex_L=("ex_L", "mean"), ex_eta=("ex_eta", "mean"),
                 ex_R2AZ=("ex_R2AZ", "mean"), pen_C=("pen_C", "mean"), pen_M=("pen_M", "mean"), pen_eta=("pen_eta", "mean"),
                 unpen_eta=("unpen_eta", "mean"), probe_wt=("probe_ridge_within_test", "mean"), probe_mlp=("probe_mlp", "mean"),
                 n=("seed", "count")).reset_index()
    summ.to_csv(RS / "SYN-1_summary.csv", index=False)
    # estimator error against exact truth (penalised heads), by n
    m = m.assign(err_M_pen=m.pen_M - m.ex_M, err_C_pen=m.pen_C - m.ex_C, err_M_unpen=m.unpen_M - m.ex_M)
    err = m[m.method.isin(["leace", "random_matched", "oracle_residual"])].groupby(["n_train", "d"]).agg(
        mae_M_pen=("err_M_pen", lambda x: np.abs(x).mean()), bias_M_pen=("err_M_pen", "mean"),
        mae_C_pen=("err_C_pen", lambda x: np.abs(x).mean()), mae_M_unpen=("err_M_unpen", lambda x: np.abs(x).mean())).reset_index()
    err.to_csv(RS / "SYN-1_estimator_error.csv", index=False)
    # three Pareto views at n_train = 10000, d = 128 (exact quantities; leakage = exact R^2 of A from Z)
    sub = m[(m.n_train == 10000) & (m.d == 128) & (m.method != "oracle_residual")]
    regs = sorted(sub.regime.unique())
    fig, axs = plt.subplots(3, len(regs), figsize=(2.3 * len(regs), 6.6), squeeze=False)
    for j, r in enumerate(regs):
        s = sub[sub.regime == r]
        for meth, gg in s.groupby("method"):
            c = COL.get(meth, "k")
            axs[0, j].scatter(gg.ex_R2AZ, 1e3 * gg.ex_C, s=12, c=c, label=meth)
            axs[1, j].scatter(gg.ex_R2AZ, 1e3 * gg.ex_M, s=12, c=c)
            axs[2, j].scatter(1e3 * gg.ex_C, 1e3 * gg.ex_M, s=12, c=c)
        lim = max(1e3 * s.ex_C.max(), 1e-3)
        axs[2, j].plot([0, lim], [0, lim], color="k", lw=0.7, ls="--")   # ceiling M = C
        axs[0, j].set_title(r.replace("_", " "), fontsize=7)
        for i in range(3):
            axs[i, j].tick_params(labelsize=6)
    axs[0, 0].set_ylabel("C_Y (1e-3 nats)", fontsize=7); axs[1, 0].set_ylabel("M_A (1e-3 nats)", fontsize=7)
    axs[2, 0].set_ylabel("M_A (1e-3 nats)", fontsize=7)
    for j in range(len(regs)):
        axs[0, j].set_xlabel("leakage R^2(A|Z)", fontsize=6); axs[1, j].set_xlabel("leakage R^2(A|Z)", fontsize=6)
        axs[2, j].set_xlabel("C_Y (1e-3 nats); dashed: M = C", fontsize=6)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=6, ncol=4, loc="upper center", frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94)); fig.savefig(FG / "SYN-1_pareto_exact.pdf"); fig.savefig(FG / "SYN-1_pareto_exact.png", dpi=130)
    return summ, err


def syn2():
    df = pd.read_csv(RS / "SYN-2.csv")
    m = df[df.method != "none"]
    summ = m.groupby(["regime", "method", "d", "n_over_d"]).agg(
        ex_C=("ex_C", "mean"), pen_C=("pen_C", "mean"), pen_M=("pen_M", "mean"), ex_M=("ex_M", "mean"), ex_eta=("ex_eta", "mean"),
        pen_eta=("pen_eta", "mean"), ex_R2AZ=("ex_R2AZ", "mean"), probe_wt=("probe_ridge_within_test", "mean"),
        sd_ratio=("coord_sd_ratio", "mean"), cos=("cos_est_true", "mean"), cond=("cond_SS", "mean"),
        lw=("lw_shrinkage", "mean")).reset_index()
    summ.to_csv(RS / "SYN-2_summary.csv", index=False)
    # head-geometry test: penalised-head cost LEACE minus orthogonal, per configuration, against SD ratio
    piv = m.pivot_table(index=["regime", "d", "n_over_d", "seed"], columns="method", values=["pen_C", "ex_C", "coord_sd_ratio"])
    geo = pd.DataFrame({"pen_gap": piv["pen_C"]["leace"] - piv["pen_C"]["orth"],
                        "ex_gap": piv["ex_C"]["leace"] - piv["ex_C"]["orth"],
                        "sd_ratio": piv["coord_sd_ratio"]["leace"]}).reset_index()
    geo.to_csv(RS / "SYN-2_head_geometry.csv", index=False)
    fig, axs = plt.subplots(1, 3, figsize=(8.5, 2.6))
    for d, gg in summ[summ.method == "leace"].groupby("d"):
        for reg, g2 in gg.groupby("regime"):
            ls = "-" if reg.startswith("R1") else "--"
            axs[0].plot(g2.n_over_d, g2.ex_R2AZ, ls, marker="o", ms=3, label=f"d={d} {reg[:2]}")
            axs[1].plot(g2.n_over_d, g2.sd_ratio, ls, marker="o", ms=3)
    for d, gg in summ.groupby("d"):
        lm = gg[gg.method == "leace"].groupby("n_over_d").pen_C.mean(); rm = gg[gg.method == "random_matched"].groupby("n_over_d").pen_C.mean()
        axs[2].plot(lm.index, 1e3 * (rm - lm), marker="o", ms=3, label=f"d={d}")
    for a in axs:
        a.set_xscale("log"); a.tick_params(labelsize=6); a.set_xlabel("n_train / d", fontsize=7)
        a.set_xticks([1.25, 2, 5, 10, 25]); a.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axs[0].set_ylabel("population leakage R^2(A|Z) after LEACE", fontsize=7)
    axs[1].set_ylabel("LEACE coordinate SD ratio (test/train)", fontsize=7)
    axs[2].set_ylabel("matched random - LEACE, penalised C (1e-3)", fontsize=7); axs[2].axhline(0, color="k", lw=0.7)
    axs[0].legend(fontsize=5); axs[2].legend(fontsize=5)
    fig.tight_layout(); fig.savefig(FG / "SYN-2_n_over_d.pdf"); fig.savefig(FG / "SYN-2_n_over_d.png", dpi=130)
    return summ, geo


if __name__ == "__main__":
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    if (RS / "SYN-1.csv").exists():
        s, e = syn1()
        print(s[(s.n_train == 10000) & (s.d == 128)].round(4).to_string(index=False))
        print(e.round(5).to_string(index=False))
    if (RS / "SYN-2.csv").exists():
        s, g = syn2()
        print(s[s.method.isin(["leace", "orth", "random_matched"])].round(4).to_string(index=False))
        print("head geometry: corr(pen_gap, sd_ratio) =", round(g[["pen_gap", "sd_ratio"]].corr().iloc[0, 1], 3),
              "| max |ex_gap| =", g.ex_gap.abs().max())
