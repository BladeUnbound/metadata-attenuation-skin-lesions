"""Iteration-5 macros and tables (E40-E49), executed by make_numbers.py before its final write.

Every value is read from research/results/*.json. Missing results render as [PENDING].
"""
import json

import numpy as np

# Executed inside make_numbers.py's namespace (ALLREP, HOLM, LABEL, NAMES, PENDING, RUNGS, load, macros, milli,
# ci_milli, pct, tables come from there).

DS8 = ["ham10000"] + ALLREP
LAB8 = {"ham10000": "HAM10000", **LABEL}
SHORT = {"ham10000": "HAM", "isic2020": "ISIC20", "isic2019_nonham": "ISIC19", "pad_ufes": "PAD", "hiba": "HIBA",
         "msk2020": "MSK", "milk10k_derm": "MILK-d", "milk10k_clin": "MILK-c"}
YES, NO, NA = r"\checkmark", r"$\times$", "--"
mark = lambda b: YES if b else NO

# ------------------------------------------------------------------ E40: strictly cross-fitted estimator
e40 = {d: load(f"E40_{d}") for d in DS8}
t40 = load("E40_tests")
ratios, repro, cells = [], 0, 0
for d in DS8:
    for s, x in e40[d].items():
        for r, v in x.items():
            o = v["original"]
            hw = (o["delta_ci95"][1] - o["delta_ci95"][0]) / 2
            ratios.append(abs(v["corrected_minus_original"]) / hw)
            repro += bool(o.get("reproduced"))
            cells += 1
c0 = lambda d, r: e40[d]["0"][r]["corrected"]
macros.update({
    "ESTcells": str(cells), "ESTrepro": str(repro), "ESTmaxHW": f"{max(ratios):.2f}",
    "ESThamOrig": milli(e40["ham10000"]["0"]["convnext_tiny"]["original"]["delta"]),
    "ESThamCorr": milli(c0("ham10000", "convnext_tiny")["delta"]),
    "ESTisicHtwoOrig": f"{HOLM[1][('isic2020', 'H2')]:.3f}",
    "ESTisicHtwoCorr": f"{t40['isic2020']['H2_corrected']['p_holm_wave1']:.3f}",
    "ESTisicHtwoDiff": f"{1e3 * t40['isic2020']['H2_corrected']['diff']:.2f}",
    "WGThibaImg": milli(c0("hiba", "convnext_tiny")["delta"]),
    "WGThibaGrp": milli(c0("hiba", "convnext_tiny")["delta_group_weighted"]),
    "WGTisicImg": milli(c0("isic2020", "convnext_tiny")["delta"]),
    "WGTisicGrp": milli(c0("isic2020", "convnext_tiny")["delta_group_weighted"]),
})
hold = lambda d, h: bool(t40[d][f"{h}_corrected"].get("holds"))
indep = [d for d in ALLREP if d != "milk10k_clin"]
for h, key in [("H1", "ESTindepHone"), ("H2", "ESTindepHtwo")]:
    macros[key] = str(sum(hold(d, h) and (d != "milk10k_derm" or hold("milk10k_clin", h)) for d in indep))
spread = {d: [1e3 * e40[d][s]["convnext_tiny"]["corrected"]["delta"] for s in "012"] for d in DS8}
hw0 = {d: 1e3 * (c0(d, "convnext_tiny")["delta_ci95"][1] - c0(d, "convnext_tiny")["delta_ci95"][0]) / 2 for d in DS8}
macros["SPREADpad"], macros["SPREADpadHW"] = f"{np.ptp(spread['pad_ufes']):.1f}", f"{hw0['pad_ufes']:.1f}"
macros["SPREADhiba"] = ", ".join(f"{v:.1f}" for v in spread["hiba"])
kd = [c0(d, r)["K_mean"] / c0(d, r)["delta"] for d in DS8 for r in RUNGS if c0(d, r)["delta"] > 0]
macros["KoverDmin"], macros["KoverDmax"] = f"{min(kd):.1f}", f"{max(kd):.0f}"

# ------------------------------------------------------------------ E41, E47: labels, complete cases, encodings
e41l, e41c, e47 = load("E41_labels"), load("E41_cc"), load("E47_missingness_source")
e17 = load("E17_retained_fraction")
alt = {"ham10000": "ham10000|MAL", "isic2019_nonham": "isic2019_nonham|MAL", "pad_ufes": "pad_ufes|MEL",
       "hiba": "hiba|MEL", "msk2020": "msk2020|MEL", "milk10k_derm": "milk10k_derm|MEL", "milk10k_clin": "milk10k_clin|MEL"}
x = e41l["isic2019_nonham|MAL"]
macros["LABisicMalRho"], macros["LABisicMalRhoCI"] = f"{x['rho_cnx']:.2f}", f"[{x['rho_cnx_ci95'][0]:.2f}, {x['rho_cnx_ci95'][1]:.2f}]"
macros["LABisicMelRho"] = f"{c0('isic2019_nonham', 'convnext_tiny')['delta'] / e17['isic2019_nonham']['I_A_nats']:.2f}"
macros["LABmilkIA"] = milli(e41l["milk10k_derm|MEL"]["I_A"])
macros["LABhOneHeld"] = str(sum(e41l[alt[d]]["H1"]["p_one_sided"] < 0.05 for d in alt if d != "milk10k_clin"
                                and (d != "milk10k_derm" or e41l[alt["milk10k_clin"]]["H1"]["p_one_sided"] < 0.05)))
macros["LABhOneN"] = str(len([d for d in alt if d != "milk10k_clin"]))
macros["CChOne"] = str(sum(v["H1"]["p_one_sided"] < 0.05 for v in e41c.values()))
macros["CChTwo"] = str(sum(v["H2"]["p_one_sided"] < 0.05 for v in e41c.values()))
macros["CCn"] = str(len(e41c))
enc = e47["encodings"]
for d, k in [("pad_ufes", "Pad"), ("hiba", "Hiba"), ("isic2019_nonham", "Isic")]:
    for e, kk in [("indicators", "Ind"), ("no_ind", "NoInd"), ("mi5", "MI")]:
        macros[f"ENC{k}{kk}"] = milli(enc[d]["convnext_tiny"][e]["delta"])
loso = e47["source_ham_loso"]
macros["SRChamText"] = "; ".join(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} ${milli(v['convnext_tiny']['delta'])}$ "
                                 f"{ci_milli(v['convnext_tiny']['ci95'])}" for k, v in loso.items() if "convnext_tiny" in v)
si = e47["source_isic2019"]
macros["SRCbcnMsk"], macros["SRCmskBcn"] = milli(si["BCN_to_MSK"]["convnext_tiny"]["delta"]), milli(si["MSK_to_BCN"]["convnext_tiny"]["delta"])

# robustness matrix (appendix)
def h_orig(d, h):
    if d == "ham10000":
        return t40[d][f"{h}_original"]["p_one_sided"] < 0.05
    return HOLM[1 if d in ALLREP[:3] else 2][(d, h)] < 0.05


def h_corr(d, h):
    return t40[d][f"{h}_corrected"]["p_one_sided"] < 0.05 if d == "ham10000" else hold(d, h)


lo = {"H1": "hsv", "H2": "resnet18"}
rows = []
for h in ["H1", "H2"]:
    cond = {
        "Original estimator (registered)": [mark(h_orig(d, h)) for d in DS8],
        "Strictly cross-fitted": [mark(h_corr(d, h)) for d in DS8],
        "Fold assignments 0/1/2 (point $>0$)": [
            f"{sum(e40[d][s][lo[h]]['corrected']['delta'] > e40[d][s]['convnext_tiny']['corrected']['delta'] for s in '012')}/3" for d in DS8],
        "Group-weighted (point $>0$)": [mark(c0(d, lo[h])["delta_group_weighted"] > c0(d, "convnext_tiny")["delta_group_weighted"]) for d in DS8],
        "Alternative label": [NA if d not in alt else mark(e41l[alt[d]][h]["p_one_sided"] < 0.05) for d in DS8],
        "Complete cases": [mark(e41c[f"{d}|cc"][h]["p_one_sided"] < 0.05) for d in DS8],
        "No missing indicators (point $>0$)": [
            mark(enc[d]["hsv"]["no_ind"]["delta"] > enc[d]["convnext_tiny"]["no_ind"]["delta"]) if h == "H1" else NA for d in DS8],
    }
    rows.append(f"\\multicolumn{{9}}{{l}}{{\\textbf{{{h}}}: $\\Delta_{{\\mathrm{{{'HSV' if h == 'H1' else 'ResNet\\text{-}18'}}}}}>\\Delta_{{\\mathrm{{ConvNeXt}}}}$}}\\\\")
    rows += [f"{k} & " + " & ".join(v) + "\\\\" for k, v in cond.items()]
tables["app_robust"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Robustness of H1 and H2 (\checkmark: holds; $\times$: fails; --: not derivable). Registered and strictly cross-fitted rows: Holm-adjusted within the registered waves (HAM10000 unadjusted). Alternative label: malignant vs.\ benign for HAM10000 and ISIC 2019, melanoma vs.\ rest elsewhere; ISIC 2020 has no distinct alternative. Alternative-label, complete-case and encoding rows use the strictly cross-fitted estimator and uncorrected one-sided tests ($p<0.05$) or point estimates as stated; fold assignment 0 unless stated.}\label{tab:robust}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccccc}\toprule
Condition & """ + " & ".join(SHORT[d] for d in DS8) + r"""\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table*}"""

# ------------------------------------------------------------------ E42, E44, E49: where metadata help
s42 = load("E42_summary")
macros["GRPhOne"], macros["GRPhTwo"], macros["GRPhFour"] = (str(s42[h]["n_held"]) for h in ["H1", "H2", "H4"])
e49 = load("E49_decomposition_posthoc")
ph = e49["posthoc_ranking"]
macros["GRPkBest"] = str(sum(ph[d]["best"] == "K" for d in DS8))
macros["GRPgBest"] = ", ".join(LAB8[d] for d in DS8 if ph[d]["best"] == "G") or "none"
cap = [ph[d]["K_top25_capture"] for d in DS8 if d != "hiba"]
macros["GRPcapMin"], macros["GRPcapMax"] = pct(min(cap)), pct(max(cap))
e42 = {d: load(f"E42_{d}") for d in DS8}
h3 = [e42[d]["H3"]["hsv_minus_cnx"] for d in DS8]
macros["GRPhThreePos"] = str(sum(v["ci95"][0] > 0 for v in h3))
macros["GRPhThreeNeg"] = str(sum(v["ci95"][1] < 0 for v in h3))
e44 = {d: load(f"E44_{d}") for d in DS8}
st = lambda d, r: e44[d][r]["normalised"]
macros["SMOOTHcnxSig"] = str(sum(st(d, "convnext_tiny")["p_stratified"] < 0.05 for d in DS8))
macros["SMOOTHhsvSig"] = str(sum(st(d, "hsv")["p_stratified"] < 0.05 for d in DS8))
macros["SMOOTHcnxMax"] = pct(max(1 - st(d, "convnext_tiny")["ratio_stratified"] for d in DS8))
dec = e49["decomposition"]
sh = {d: dec[d]["0"]["share_H"] for d in DS8}
nonpad = [d for d in DS8 if d != "pad_ufes"]
macros["DECshareMin"], macros["DECshareMax"] = f"{min(sh[d] for d in nonpad):.2f}", f"{max(sh[d] for d in nonpad):.2f}"
macros["DECshareUpper"] = f"{max(dec[d]['0']['share_H_ci95'][1] for d in nonpad):.2f}"
macros["DECpad"] = f"{sh['pad_ufes']:.2f}"
macros["DECpadCI"] = f"[{dec['pad_ufes']['0']['share_H_ci95'][0]:.2f}, {dec['pad_ufes']['0']['share_H_ci95'][1]:.2f}]"
grows = []
for d in DS8:
    r = e42[d]["rungs"]["convnext_tiny"]
    a, b = r["H1_spearman_LE_delta"], r["H2_spearman_hq0_delta"]
    grows.append(f"{LAB8[d]} & {a['rho']:.2f} [{a['ci95'][0]:.2f}, {a['ci95'][1]:.2f}] & {b['rho']:.2f} & "
                 f"{r['H4']['p_one_sided']:.3f} & {ph[d]['K']:.2f} & {ph[d]['G']:.2f} & "
                 f"{(pct(ph[d]['K_top25_capture']) if d != 'hiba' else '--')} & {st(d, 'hsv')['ratio_stratified']:.3f} & "
                 f"{st(d, 'convnext_tiny')['ratio_stratified']:.3f} & {sh[d]:.2f}\\\\")
tables["app_graph"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Where metadata help (ConvNeXt-T unless stated; strictly cross-fitted, fold assignment 0). $\rho_{\mathrm{LE}}$, $\rho_{h}$: Spearman correlation of realised per-image benefit with local label entropy on the inductive graph (pre-registered H1, 95\% grouped CI) and with the image model's predictive entropy (H2). H4 $p$: one-sided, unadjusted, graph statistics vs.\ model-based baselines. $\rho_K$, $\rho_G$: post hoc rank correlation of benefit with the fusion model's expected gain $K_i$ and with graph-local benefit. Top 25\%: share of total benefit captured by the quarter of images with the largest $K_i$ (HIBA omitted: total benefit near zero). $E$: normalised Dirichlet energy of benefit relative to the label-by-uncertainty stratified null (HSV, ConvNeXt-T). $s_H$: share of the HSV$\to$ConvNeXt-T log-attenuation due to image uncertainty (exploratory).}\label{tab:graph}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccccccc}\toprule
Image set & $\rho_{\mathrm{LE}}$ & $\rho_h$ & H4 $p$ & $\rho_K$ & $\rho_G$ & Top 25\% & $E_{\mathrm{HSV}}$ & $E_{\mathrm{CNX}}$ & $s_H$\\\midrule
""" + "\n".join(grows) + r"""
\bottomrule\end{tabular}\end{table*}"""

# ------------------------------------------------------------------ E45, E45b: concept erasure
e45 = {d: load(f"E45_{d}") for d in ["ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes"]}
e45b = {d: load(f"E45b_{d}") for d in e45}
erows, held45 = [], 0
EL = {"ham10000": "HAM10000", "isic2019_nonham": "ISIC 2019", "milk10k_derm": "MILK10k (derm.)", "pad_ufes": "PAD-UFES-20$^\\ddagger$"}
for d in e45:
    for r in ["convnext_tiny", "resnet18"]:
        a = e45[d][r]["age"]
        t0 = e45[d][r]["t0"]["delta_age"]
        b = e45b[d][r] if e45b[d] else None
        if b:
            pm = b["concept_minus_matched"]
            held45 += b["claim_holds"]
            tail = (f"{np.mean(b['delta_age_x1e3']['same_rank_each']):.1f} & {np.mean(b['delta_age_x1e3']['matched_each']):.1f} ({b['rank_matched']}) & "
                    f"{pm['diff']:.1f} [{pm['ci95'][0]:.1f}, {pm['ci95'][1]:.1f}]")
        else:
            tail = f"\\multicolumn{{3}}{{c}}{{{PENDING}}}"
        erows.append(f"{EL[d] if r == 'convnext_tiny' else ''} & {NAMES[r]} & {a['probe_t0']:.2f} & {milli(t0)} & "
                     + " / ".join(milli(a[f't{t}']['delta_age']) for t in [0.25, 0.5, 0.75, 1.0]) + f" & {tail}\\\\")
tables["erasure"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Exploratory concept erasure (E45/E45b): added value of age ($10^{-3}$ nats) after removing the linear trace of age from the image representation (LEACE, fitted per training fold) at dose $t$, compared with removing random subspaces. $R^2$: out-of-fold linear decodability of age before erasure (after erasure: 0.00 throughout). Same rank: mean over 5 random one-dimensional subspaces. Matched: mean over 5 random subspaces whose rank (in brackets) raises image-only log-loss about as much as erasing age. Last column: paired difference, erasure minus matched, with 95\% grouped-bootstrap CI. Strictly cross-fitted estimator, fold assignment 0. $^\ddagger$Complete cases.}\label{tab:erasure}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llcccccc}\toprule
Dataset & Backbone & $R^2_{\mathrm{age}}$ & $t=0$ & $t=0.25/0.5/0.75/1$ & Same rank & Matched (rank) & Erasure $-$ matched\\\midrule
""" + "\n".join(erows) + r"""
\bottomrule\end{tabular}\end{table*}"""
macros["ERASEheld"] = str(held45) if all(e45b.values()) else PENDING
macros["ERASEn"] = str(2 * len(e45))
macros["ERASEhamT"] = milli(e45["ham10000"]["convnext_tiny"]["t0"]["delta_age"])
macros["ERASEhamOne"] = milli(e45["ham10000"]["convnext_tiny"]["age"]["t1.0"]["delta_age"])
macros["ERASEmilkT"] = milli(e45["milk10k_derm"]["convnext_tiny"]["t0"]["delta_age"])
macros["ERASEmilkOne"] = milli(e45["milk10k_derm"]["convnext_tiny"]["age"]["t1.0"]["delta_age"])
macros["ERASEpadRsq"] = f"{e45['pad_ufes']['convnext_tiny']['age']['probe_t0']:.2f}"
macros["ERASEhamLLzero"] = f"{e45['ham10000']['convnext_tiny']['t0']['image_logloss']:.3f}"
macros["ERASEhamLLone"] = f"{e45['ham10000']['convnext_tiny']['age']['t1.0']['image_logloss']:.3f}"
ham_spec = e45["ham10000"]["convnext_tiny"]
macros["ERASEhamSite"] = milli(ham_spec["age"]["t1.0"]["delta_site"] - ham_spec["t0"]["delta_site"])
macros["ERASEhamSex"] = milli(ham_spec["age"]["t1.0"]["delta_sex"] - ham_spec["t0"]["delta_sex"])

# ------------------------------------------------------------------ E46 recoverability re-analysis
e46 = load("E46_recov_reanalysis")
b46, a46 = e46["B_all_rows_absolute"], e46["A_registered_as_run"]
macros["RECallRows"] = str(b46["n_rows"])
macros["RECallPartial"] = f"{b46['partial']:.2f}"
macros["RECallMixed"] = f"{b46['mixed_coef_recov']:.2f} [{b46['mixed_ci95'][0]:.2f}, {b46['mixed_ci95'][1]:.2f}]"
macros["RECfiltMixed"] = f"{a46['mixed_coef_recov']:.2f} [{a46['mixed_ci95'][0]:.2f}, {a46['mixed_ci95'][1]:.2f}]"

# ------------------------------------------------------------------ E48 mask-swap null
e48 = load("E48_mask_swap")
bb = [e48[k] for k in ["resnet18", "resnet50", "convnext_tiny"]]
macros["SWAPrange"] = f"{pct(min(x['frac_sig_swap'] for x in bb))}--{pct(max(x['frac_sig_swap'] for x in bb))}"
macros["SWAPshiftRange"] = f"{pct(min(x['frac_sig_shift'] for x in bb))}--{pct(max(x['frac_sig_shift'] for x in bb))}"
macros["GAUSSshift"], macros["GAUSSswap"] = pct(bb[0]["gauss_frac_sig_shift"]), pct(bb[0]["gauss_frac_sig_swap"])
srows = [f"{NAMES[k]} & {e48[k]['n_images']:,} & {pct(e48[k]['frac_sig_shift'])} & {pct(e48[k]['frac_sig_swap'])} & "
         f"{pct(e48[k]['gauss_frac_sig_shift'])} & {pct(e48[k]['gauss_frac_sig_swap'])}\\\\".replace(",", "{,}")
         for k in ["resnet18", "resnet50", "convnext_tiny"]]
tables["app_swap"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Share of images whose melanoma-evidence map overlaps its lesion mask significantly ($p\le0.05$) under the circular-shift null and under the mask-swap null (masks of 199 other images in the same lesion-area decile), for real class-activation maps and for a fixed centred Gaussian map that carries no image information (E48; HAM10000).}\label{tab:swap}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}\toprule
 & & \multicolumn{2}{c}{Real maps} & \multicolumn{2}{c}{Centred Gaussian}\\
Backbone & Images & Shift & Swap & Shift & Swap\\\midrule
""" + "\n".join(srows) + r"""
\bottomrule\end{tabular}\end{table}"""
macros["ERASEhamDelta"] = milli(ham_spec["age"]["t1.0"]["delta_age"] - ham_spec["t0"]["delta_age"])
if e45b["milk10k_derm"]:
    mb = e45b["milk10k_derm"]["convnext_tiny"]["image_logloss"]
    macros["ERASEmilkLLconcept"], macros["ERASEmilkLLmatched"] = f"{mb['concept']:.2f}", f"{np.mean(mb['matched']):.2f}"
