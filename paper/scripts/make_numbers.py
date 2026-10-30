"""Generate paper/numbers.tex and paper/tables/*.tex from research/results/*.json.

No number in the paper is typed by hand: every result comes from this script.
Results that don't exist yet are rendered as a visible [PENDING] marker.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
RES = PAPER.parent / "research/results"
RUNGS = ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
NAMES = {"hsv": "HSV histogram", "resnet18": "ResNet-18", "resnet50": "ResNet-50",
         "convnext_tiny": "ConvNeXt-T", "vit_b_16": "ViT-B/16"}
PENDING = r"\textbf{[PENDING]}"


def load(name):
    p = RES / f"{name}.json"
    return json.loads(p.read_text())["results"] if p.exists() else None


def f3(x):  # nats, 3 decimals
    return f"{x:.3f}"


def milli(x):  # value in 10^-3 nats
    return f"{1000 * x:.1f}"


def ci_milli(ci):
    return f"[{1000 * ci[0]:.1f}, {1000 * ci[1]:.1f}]"


def pct(x):
    return f"{100 * x:.0f}\\%"


macros = {}
tables = {}

# ------------------------------------------------------------------ HAM10000
e10, e10a, e10b = load("E10_ham_attenuation"), load("E10a_ham_metadata_audit"), load("E10b_source_control")
seeds = e10["primary"]
mean_av = {r: float(np.mean([seeds[s][r]["added_value"] for s in seeds])) for r in RUNGS}
mean_auc = {r: float(np.mean([seeds[s][r]["image_auroc"] for s in seeds])) for r in RUNGS}
extra = e10["seed0_extra"]
macros.update({
    "HAMavHSV": milli(mean_av["hsv"]), "HAMavResEighteen": milli(mean_av["resnet18"]),
    "HAMavConvNeXt": milli(mean_av["convnext_tiny"]),
    "HAMdxtypeAUC": f"{e10a['dx_type_only']['oof_auroc']:.2f}",
    "HAMsourceAUC": f"{e10a['source_only']['oof_auroc']:.2f}",
    "HAMsourceRecov": f"{max(extra[r]['recoverability']['exploratory_source_auroc_mean'] for r in RUNGS):.3f}",
    "HAMageRecovHSV": f"{extra['hsv']['recoverability']['age_r2']:.2f}",
    "HAMageRecovConvNeXt": f"{extra['convnext_tiny']['recoverability']['age_r2']:.2f}",
    "HAMadjHSV": milli(e10b["hsv"]["source_adjusted_added_value"]),
    "HAMadjConvNeXt": milli(e10b["convnext_tiny"]["source_adjusted_added_value"]),
})
pr = e10b["paired_attenuation"]["unadjusted:resnet18-convnext_tiny"]
macros["HAMpairedRConv"] = milli(pr["av_diff"])
macros["HAMpairedRConvCI"] = ci_milli(pr["ci95"])

rows = []
for r in RUNGS:
    s0 = seeds["seed0"][r]
    rec = extra[r]["recoverability"]
    rows.append(f"{NAMES[r]} & {mean_auc[r]:.3f} & {milli(mean_av[r])} & {ci_milli(s0['added_value_ci95'])} & "
                f"{rec['age_r2']:.2f} & {rec['site_auroc_top5_mean']:.2f} & {rec['exploratory_source_auroc_mean']:.3f} \\\\")
tables["ham_ladder"] = r"""\begin{table}[t]\centering\footnotesize
\caption{HAM10000, melanoma vs.\ rest, lesion-grouped 5-fold CV. AUROC and $\hat\Delta$ are means over three fold assignments; the CI is the 95\% lesion bootstrap for assignment 0. Recoverability: out-of-fold $R^2$ for age and mean AUROC for the five commonest sites and for image source, predicted from the frozen features.}\label{tab:ham}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccccc}\toprule
 & Image & $\hat\Delta$ & 95\% CI & \multicolumn{3}{c}{Recoverability}\\
Representation & AUROC & ($10^{-3}$ nats) & ($10^{-3}$) & age & site & source\\\midrule
""" + "\n".join(rows) + r"""
\bottomrule\end{tabular}\end{table}"""

pairs = [("hsv", "resnet18"), ("hsv", "convnext_tiny"), ("resnet18", "convnext_tiny"),
         ("resnet18", "vit_b_16"), ("resnet50", "convnext_tiny")]
prow = []
for a, b in pairs:
    u = e10b["paired_attenuation"][f"unadjusted:{a}-{b}"]
    s = e10b["paired_attenuation"][f"source_adjusted:{a}-{b}"]
    prow.append(f"{NAMES[a]} $-$ {NAMES[b]} & {milli(u['av_diff'])} {ci_milli(u['ci95'])} & {milli(s['av_diff'])} {ci_milli(s['ci95'])} \\\\")
tables["ham_paired"] = r"""\begin{table}[t]\centering\footnotesize
\caption{HAM10000: paired differences in added value between representations, $\hat\Delta_a-\hat\Delta_b$ ($10^{-3}$ nats, 95\% paired lesion-bootstrap CI), fold assignment 0. Exploratory: not corrected for multiplicity.}\label{tab:hampaired}
\begin{tabular}{lcc}\toprule
Comparison & Unadjusted & Source-adjusted\\\midrule
""" + "\n".join(prow) + r"""
\bottomrule\end{tabular}\end{table}"""

# ------------------------------------------------------------------ HAM robustness (E10c, E10d, full-feature control)
e10c, e10d = load("E10c_per_variable"), load("E10d_auroc_brier")
for key, r in [("HSV", "hsv"), ("ResEighteen", "resnet18"), ("ConvNeXt", "convnext_tiny")]:
    macros[f"HAMdauc{key}"] = f"{e10d[r]['delta_auroc']:.3f}"
    macros[f"HAMdauc{key}CI"] = f"[{e10d[r]['delta_auroc_ci95'][0]:.3f}, {e10d[r]['delta_auroc_ci95'][1]:.3f}]"
for v, name in [("age", "Age"), ("sex", "Sex"), ("localization", "Site")]:
    macros[f"HAMretained{name}"] = pct(e10c[f"retained_fraction_{v}"])
macros["HAMsexRecovConvNeXt"] = f"{extra['convnext_tiny']['recoverability']['sex_auroc']:.2f}"
rrow = []
for r in RUNGS:
    d = e10d[r]
    s0 = seeds["seed0"][r]
    rrow.append(f"{NAMES[r]} & {d['delta_auroc']:+.3f} [{d['delta_auroc_ci95'][0]:+.3f}, {d['delta_auroc_ci95'][1]:+.3f}] & "
                f"{1000*d['delta_brier']:.2f} [{1000*d['delta_brier_ci95'][0]:.2f}, {1000*d['delta_brier_ci95'][1]:.2f}] & "
                f"{s0['both_logloss']:.3f} & {extra[r]['full_features_plus_meta_logloss']:.3f}\\\\")
tables["ham_robust"] = r"""\begin{table}[t]\centering\footnotesize
\caption{HAM10000 robustness (fold assignment 0). $\Delta$AUROC and $\Delta$Brier ($10^{-3}$) from adding metadata, with 95\% lesion-bootstrap CIs. Last two columns: out-of-fold log-loss of score\,+\,metadata (two-stage, as in Table~\ref{tab:ham}) vs.\ all frozen features\,+\,metadata in one model; similar values mean the one-dimensional score loses little.}\label{tab:hamrobust}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{lcccc}\toprule
 & $\Delta$AUROC & $\Delta$Brier & \multicolumn{2}{c}{Log-loss with metadata}\\
Representation & [95\% CI] & [95\% CI] & two-stage & joint\\\midrule
""" + "\n".join(rrow) + r"""
\bottomrule\end{tabular}\end{table}"""

# ------------------------------------------------------------------ Interpretability
e12, e12b = load("E12_ham_cam_lesion"), load("E12b_ham_cam_fixed_sign")
per = pd.read_csv(RES / "E12b_per_image.csv")
mel = per[per.y == 1]
macros.update({
    "INTfracEnriched": pct(e12b["H1_frac_enrichment_gt1"]),
    "INTpermSig": pct(e12b["H3_frac_sig_pixel_perm"]), "INTshiftSig": pct(e12b["H3_frac_sig_shift_null"]),
    "INTareaDetected": f"{mel[mel.pred == 1].lesion_area_frac.median():.2f}",
    "INTareaMissed": f"{mel[mel.pred == 0].lesion_area_frac.median():.2f}",
})
tables["interp"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Class-activation evidence vs.\ expert lesion masks (HAM10000, ResNet-50, %d images). Enrichment: share of positive evidence inside the lesion divided by the lesion's area share. Significance at $p\le0.05$.}\label{tab:interp}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccc}\toprule
Evidence map & Enrichment $>1$ & Shift null & Pixel perm.\\\midrule
Predicted class & %s & %s & %s\\
Melanoma (all images) & %s & %s & %s\\
\bottomrule\end{tabular}\end{table}""" % (
    e12b["n_images"],
    pct(e12["H1_frac_enrichment_gt1"]), pct(e12["H3_frac_sig_shift_null"]), pct(e12["H3_frac_sig_pixel_perm"]),
    pct(e12b["H1_frac_enrichment_gt1"]), pct(e12b["H3_frac_sig_shift_null"]), pct(e12b["H3_frac_sig_pixel_perm"]))

# ------------------------------------------------------------------ Replication (E13), two pre-registered waves
WAVES = {1: ["isic2020", "isic2019_nonham", "pad_ufes"], 2: ["hiba", "msk2020", "milk10k_derm", "milk10k_clin"]}
ALLREP = WAVES[1] + WAVES[2]
rep = {d: load(f"E13_{d}") for d in ALLREP}
meta = {d: (json.loads((RES / f"E13_{d}.json").read_text())["data"] if (RES / f"E13_{d}.json").exists() else None)
        for d in rep}
LABEL = {"isic2020": "ISIC 2020", "isic2019_nonham": "ISIC 2019 (non-HAM)", "pad_ufes": "PAD-UFES-20",
         "hiba": "HIBA", "msk2020": "MSK 2020", "milk10k_derm": "MILK10k (derm.)", "milk10k_clin": "MILK10k (clin.)"}
MODALITY = {"isic2020": "dermoscopy", "isic2019_nonham": "dermoscopy", "pad_ufes": "smartphone", "hiba": "dermoscopy",
            "msk2020": "mixed", "milk10k_derm": "dermoscopy", "milk10k_clin": "clin.\\ close-up"}
METAVARS = {"pad_ufes": "demogr.\\,+\\,symptoms", "hiba": "age, sex, site, skin type"}


def num(n):
    return f"{n:,}".replace(",", "{,}")


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i][2])
    m, out, running = len(pvals), {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i][2]))
        out[(pvals[i][0], pvals[i][1])] = running
    return out


fmtp = lambda p: "$\\le$0.001" if p <= 0.001 else f"{p:.3f}"  # bootstrap resolution 1/(B+1)
UNIT = {"isic2020": "patients", "isic2019_nonham": "lesions$^\\ast$", "pad_ufes": "patients", "hiba": "patients",
        "msk2020": "images$^\\dagger$", "milk10k_derm": "lesions", "milk10k_clin": "lesions"}
drows = [f"HAM10000 & dermoscopy & {num(e10a['n_images'])} & {num(e10a['n_lesions'])} lesions & "
         f"{num(e10a['class_counts']['mel'])} & age, sex, site\\\\"]
for d in ALLREP:
    m = meta[d]
    if m:
        drows.append(f"{LABEL[d]} & {MODALITY[d]} & {num(m['n_images'])} & {num(m['n_groups'])} {UNIT[d]} & "
                     f"{num(m['positives'])} & {METAVARS.get(d, 'age, sex, site')}\\\\")
    else:
        drows.append(f"{LABEL[d]} & \\multicolumn{{5}}{{c}}{{{PENDING}}}\\\\")
tables["datasets"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Datasets. Positives: melanoma for HAM10000 and ISIC; malignant otherwise. Groups define the cross-validation and bootstrap units. $^\ast$Images without a lesion identifier form their own group. $^\dagger$No lesion identifier is available. The two MILK10k rows show the same lesions.}\label{tab:data}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llrrrl}\toprule
Dataset & Modality & Images & Groups & Pos. & Metadata\\\midrule
""" + "\n".join(drows) + r"""
\bottomrule\end{tabular}\end{table}"""

held = {1: [0, 0], 2: [0, 0]}
lines, HOLM = [], {}
for w, dss in WAVES.items():
    pv = []
    for d in dss:
        t = rep[d]["confirmatory_tests"]
        pv += [(d, "H1", t["H1"]["p_one_sided"]), (d, "H2", t["H2"]["p_one_sided"])]
    HOLM[w] = holm(pv)
    for d in dss:
        s0 = rep[d]["seeds"]["0"]
        h1, h2 = HOLM[w][(d, "H1")], HOLM[w][(d, "H2")]
        held[w][0] += h1 < 0.05
        held[w][1] += h2 < 0.05
        cells = " & ".join(milli(s0[r]["av_all"]) for r in RUNGS)
        lines.append(f"{w} & {LABEL[d]} & {cells} & {fmtp(h1)} & {fmtp(h2)}\\\\")
    if w == 1:
        lines.append("\\midrule")
tables["replication"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Pre-registered replication in two waves: added value of all metadata $\hat\Delta$ ($10^{-3}$ nats, fold assignment 0) along the ladder, and Holm-adjusted one-sided p-values for H1 ($\Delta_{\mathrm{HSV}}>\Delta_{\mathrm{ConvNeXt}}$) and H2 ($\Delta_{\mathrm{ResNet\text{-}18}}>\Delta_{\mathrm{ConvNeXt}}$), grouped paired bootstrap. Holm families as registered: 6 tests (wave 1) and 8 tests (wave 2). The two MILK10k rows show the same lesions and are not independent.}\label{tab:rep}
\begin{tabular}{clccccccc}\toprule
Wave & Dataset & HSV & ResNet-18 & ResNet-50 & ConvNeXt-T & ViT-B/16 & H1 $p_{\mathrm{Holm}}$ & H2 $p_{\mathrm{Holm}}$\\\midrule
""" + "\n".join(lines) + r"""
\bottomrule\end{tabular}\end{table*}"""
indep = [d for d in ALLREP if d != "milk10k_clin"]  # MILK10k counted once as a dataset
pass1 = lambda d, h: HOLM[1 if d in WAVES[1] else 2][(d, h)] < 0.05
macros.update({
    "REPwaveOneHone": str(held[1][0]), "REPwaveOneHtwo": str(held[1][1]),
    "REPwaveTwoHone": str(held[2][0]), "REPwaveTwoHtwo": str(held[2][1]),
    "REPindepN": str(len(indep)),
    "REPindepHone": str(sum(pass1(d, "H1") and (d != "milk10k_derm" or pass1("milk10k_clin", "H1")) for d in indep)),
    "REPindepHtwo": str(sum(pass1(d, "H2") and (d != "milk10k_derm" or pass1("milk10k_clin", "H2")) for d in indep)),
    "REPmskHtwoP": f"{HOLM[2][('msk2020', 'H2')]:.3f}",
})
floors = []
for d in ALLREP:
    c = rep[d]["seeds"]["0"]["convnext_tiny"]
    floors.append(f"{LABEL[d]} ${milli(c['av_all'])}$ {ci_milli(c['av_all_ci95'])}")
macros["REPfloors"] = "; ".join(floors)

e16 = load("E16_within_class_recoverability")
if e16 and "pad_ufes" in e16:
    sym = ["itch", "grew", "hurt", "changed", "bleed", "elevation"]
    wc = [e16["pad_ufes"][v]["convnext_tiny"]["within_class"] for v in sym]
    macros["PADsymWithinMin"], macros["PADsymWithinMax"] = f"{min(wc):.2f}", f"{max(wc):.2f}"
mm = rep["pad_ufes"].get("mechanism")
if mm:
    rf = mm["retained_fraction_hsv_to_convnext"]
    macros["PADmechD"], macros["PADmechS"] = f"{rf['D_demographic']}", f"{rf['S_symptoms']}"
    macros["PADmechSymAUC"] = f"{mm['M3_mean_symptom_auroc_convnext']}"

# ------------------------------------------------------------------ E22 paired modality and lesion size (pre-registered)
e22 = load("E22_paired_modality")
p1, p2 = e22["P1_milk10k"], e22["P2_msk2020"]
macros.update({
    "PoneN": num(p1["n_lesions"]), "PoneRhoDerm": f"{p1['dermoscopic']['rho']:.3f}", "PoneRhoClin": f"{p1['clinical']['rho']:.3f}",
    "PoneDiff": f"{p1['rho_clin_minus_derm']:.3f}", "PoneCI": f"[{p1['ci95'][0]:.3f}, {p1['ci95'][1]:.3f}]",
    "PtwoHSV": milli(p2["hsv"]["av"]), "PtwoHSVCI": ci_milli(p2["hsv"]["ci95"]),
    "PtwoCNX": milli(p2["convnext_tiny"]["av"]), "PtwoCNXCI": ci_milli(p2["convnext_tiny"]["ci95"]),
    "PtwoRetained": f"{p2['retained']:.2f}",
    "MILKaucDerm": f"{rep['milk10k_derm']['seeds']['0']['convnext_tiny']['image_auroc']:.3f}",
    "MILKaucClin": f"{rep['milk10k_clin']['seeds']['0']['convnext_tiny']['image_auroc']:.3f}",
})
assert p1["P1_supported"] is False and p2["P2_supported"] is True  # prose in main.tex states these outcomes

# ------------------------------------------------------------------ E17 retained fraction (all 8 datasets)
e17 = load("E17_retained_fraction")
lab = {"ham10000": "HAM10000", **LABEL}
ALLDS = ["ham10000"] + ALLREP
r17 = []
for d in ALLDS:
    x = e17[d]
    cells = " & ".join(f"{x[k]['rho']:.2f} [{x[k]['rho_ci95'][0]:.2f}, {x[k]['rho_ci95'][1]:.2f}]"
                       for k in ["hsv", "resnet18", "convnext_tiny", "vit_b_16"])
    r17.append(f"{lab[d]} & {1000 * x['I_A_nats']:.1f} & {cells}\\\\")
tables["retained"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Exploratory: retained fraction $\rho_k=\hat\Delta_k/\hat I_A$ of the metadata's standalone information that is not redundant with image representation $k$ (all images, pre-registered encoding). $\hat I_A$: out-of-fold log-loss reduction of a metadata-only model over prevalence ($10^{-3}$ nats). 95\% grouped-bootstrap CIs, fold assignment 0. $\rho_k$ can exceed 1 under synergy and fall below 0 in finite samples.}\label{tab:retained}
\begin{tabular}{lccccc}\toprule
Dataset & $\hat I_A$ & HSV & ResNet-18 & ConvNeXt-T & ViT-B/16\\\midrule
""" + "\n".join(r17) + r"""
\bottomrule\end{tabular}\end{table*}"""
RK = {"ham10000": "HAM", "isic2020": "ISICtwenty", "isic2019_nonham": "ISICnineteen", "pad_ufes": "PAD",
      "hiba": "HIBA", "msk2020": "MSK", "milk10k_derm": "MILKderm", "milk10k_clin": "MILKclin"}
for d in ALLDS:
    macros[f"RHO{RK[d]}CNX"] = f"{e17[d]['convnext_tiny']['rho']:.2f}"
cnx = {d: e17[d]["convnext_tiny"]["rho"] for d in ALLDS}
macros["RHOmin"], macros["RHOmax"] = f"{min(cnx.values()):.2f}", f"{max(cnx.values()):.2f}"
macros["RHOminDS"], macros["RHOmaxDS"] = lab[min(cnx, key=cnx.get)], lab[max(cnx, key=cnx.get)]

# ------------------------------------------------------------------ E15 interpretability across backbones
e15 = load("E15_cam_backbones")
if e15:
    rows15 = []
    for k in ["resnet18", "resnet50", "convnext_tiny"]:
        if k not in e15:
            continue
        x = e15[k]
        rows15.append(f"{NAMES[k]} & {x['n_images']:,} & {pct(x['frac_enrichment_gt1'])} & {pct(x['frac_sig_shift'])} & "
                      f"{pct(x['frac_sig_perm'])} & {x['area_detected_vs_missed'][0]:.2f} / {x['area_detected_vs_missed'][1]:.2f}\\\\".replace(",", "{,}"))
    bb = [e15[k] for k in ["resnet18", "resnet50", "convnext_tiny"] if k in e15]
    macros["INTbbEnrichRange"] = f"{pct(min(x['frac_enrichment_gt1'] for x in bb))}--{pct(max(x['frac_enrichment_gt1'] for x in bb))}"
    macros["INTbbShiftRange"] = f"{pct(min(x['frac_sig_shift'] for x in bb))}--{pct(max(x['frac_sig_shift'] for x in bb))}"
    macros["INTbbPermRange"] = f"{pct(min(x['frac_sig_perm'] for x in bb))}--{pct(max(x['frac_sig_perm'] for x in bb))}"
    macros["INTbbAreaPmax"] = f"{max(x['area_mw_p'] for x in bb):.0e}".replace("e-", r"\times10^{-") + "}"
    tables["interp_backbones"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Melanoma evidence (exact CAM) vs.\ expert lesion masks across backbones (HAM10000; all melanomas plus a matched sample of other lesions). Significance at $p\le0.05$: exact circular-shift null vs.\ pixel permutation. Last column: median lesion area fraction of detected / missed melanomas.}\label{tab:interpbb}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}\toprule
Backbone & Images & Enrich.$>1$ & Shift null & Pixel perm. & Area det./miss.\\\midrule
""" + "\n".join(rows15) + r"""
\bottomrule\end{tabular}\end{table}"""

# ------------------------------------------------------------------ E21 complete-case sensitivity (both waves)
e21 = {**load("E21_complete_case"), **(load("E21_complete_case_wave2") or {})}
rows21 = []
for d in ALLDS:
    x = e21[d]
    rows21.append(f"{lab[d]} & {num(x['n_complete'])} ({num(x['n_dropped'])}) & {1000 * x['I_A_nats']:.1f} & "
                  f"{milli(x['hsv']['av'])} & {milli(x['convnext_tiny']['av'])} & {fmtp(x['H1']['p_one_sided'])} & "
                  f"{fmtp(x['H2']['p_one_sided'])} & {x['convnext_tiny']['rho']:.2f}\\\\")
tables["complete_case"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Complete-case sensitivity analysis: only images with every metadata field recorded (fold assignment 0; one-sided paired grouped-bootstrap p-values, uncorrected). $\hat I_A$ and $\hat\Delta$ in $10^{-3}$ nats; $\rho$ at ConvNeXt-T.}\label{tab:cc}
\begin{tabular}{lccccccc}\toprule
Dataset & Kept (dropped) & $\hat I_A$ & $\hat\Delta_{\mathrm{HSV}}$ & $\hat\Delta_{\mathrm{ConvNeXt}}$ & H1 $p$ & H2 $p$ & $\rho$\\\midrule
""" + "\n".join(rows21) + r"""
\bottomrule\end{tabular}\end{table*}"""
macros["CCholdHone"] = str(sum(e21[d]["H1"]["p_one_sided"] <= 0.05 for d in ALLDS))
macros["CCholdHtwo"] = str(sum(e21[d]["H2"]["p_one_sided"] <= 0.05 for d in ALLDS))
macros["CCfailHtwo"] = ", ".join(lab[d] for d in ALLDS if e21[d]["H2"]["p_one_sided"] > 0.05)
macros["CCn"] = str(len(ALLDS))
macros["PADIAleak"] = f"{1000 * e17['pad_ufes']['I_A_nats']:.0f}"
macros["PADIAcc"] = f"{1000 * e21['pad_ufes']['I_A_nats']:.0f}"
macros["PADccKept"] = num(e21["pad_ufes"]["n_complete"])
for d in ALLDS:
    macros[f"CCRHO{RK[d]}"] = f"{e21[d]['convnext_tiny']['rho']:.2f}"
ccr = {d: e21[d]["convnext_tiny"]["rho"] for d in ALLDS}
macros["CCRHOmin"], macros["CCRHOmax"] = f"{min(ccr.values()):.2f}", f"{max(ccr.values()):.2f}"

# ------------------------------------------------------------------ E32 permutation control
e32 = load("E32_permutation_controls")
if e32:
    ok = [d for d in e32 if e32[d]["complete_case"]["convnext_tiny"]["matched"] > e32[d]["complete_case"]["convnext_tiny"]["perm_max"]]
    macros["PERMbeatAll"], macros["PERMn"] = str(len(ok)), str(len(e32))
    macros["PERMp"] = f"{min(e32[d]['complete_case']['convnext_tiny']['p_empirical'] for d in e32):.2f}"
    macros["PERMR"] = str(json.loads((RES / "E32_permutation_controls.json").read_text())["config"]["permutations"])

# ------------------------------------------------------------------ Representation strength: fine-tuning, data, size, ViT-L
e14 = load("E14_finetune")
e14b = {d: r for d in ["hiba", "pad_ufes", "isic2019_nonham", "milk10k_derm"] if (r := load(f"E14b_{d}"))}
e31a, e31b, e31c, e18 = load("E31a_family_size"), load("E31b_scaling_frozen"), load("E31c_ft_fraction"), load("E18_foundation_rung")
srows = []
for d, res in [("ham10000", e14)] + list(e14b.items()):
    for k in [b for b in ["resnet18", "convnext_tiny"] if b in res]:
        x = res[k]
        diff = x.get("frozen_minus_finetuned_av", x["added_value_frozen"] - x["added_value_finetuned"])
        srows.append(f"{lab[d]} & {NAMES[k]} & {x['frozen_auroc']:.3f} / {x['finetuned_auroc']:.3f} & "
                     f"{milli(x['added_value_frozen'])} / {milli(x['added_value_finetuned'])} & "
                     f"{milli(diff)} {ci_milli(x['frozen_minus_finetuned_ci95'])}\\\\")
tables["finetune"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Exploratory: frozen vs.\ end-to-end fine-tuned image models (same grouped folds, fold assignment 0, all images; one training run per fold). $\hat\Delta$ and the paired difference (frozen $-$ fine-tuned) in $10^{-3}$ nats with 95\% grouped-bootstrap CI. PAD-UFES-20 includes its label-informative missingness.}\label{tab:ft}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccc}\toprule
Dataset & Backbone & AUROC frz./FT & $\hat\Delta$ frz./FT & Difference\\\midrule
""" + "\n".join(srows) + r"""
\bottomrule\end{tabular}\end{table}"""
r18 = e14["resnet18"]
macros.update({
    "FTham": milli(r18["added_value_finetuned"]), "FThamCI": ci_milli(r18["added_value_finetuned_ci95"]),
    "FThamFrozen": milli(r18["added_value_frozen"]), "FThamDiff": milli(r18["frozen_minus_finetuned_av"]),
    "FThamDiffCI": ci_milli(r18["frozen_minus_finetuned_ci95"]),
    "FThamAUC": f"{r18['finetuned_auroc']:.2f}",
    "FTcnxDiffCI": ci_milli(e14["convnext_tiny"]["frozen_minus_finetuned_ci95"]),
    "FTpadCNXfrozen": milli(e14b["pad_ufes"]["convnext_tiny"]["added_value_frozen"]),
    "FTpadCNXft": milli(e14b["pad_ufes"]["convnext_tiny"]["added_value_finetuned"]),
})
for d, key in [("isic2019_nonham", "Isic"), ("milk10k_derm", "Milk")]:
    if d in e14b:
        x = e14b[d]["resnet18"]
        macros[f"FT{key}AUCfrz"], macros[f"FT{key}AUCft"] = f"{x['frozen_auroc']:.3f}", f"{x['finetuned_auroc']:.3f}"
        macros[f"FT{key}Frozen"], macros[f"FT{key}Ft"] = milli(x["added_value_frozen"]), milli(x["added_value_finetuned"])
        macros[f"FT{key}DiffCI"] = ci_milli(x["frozen_minus_finetuned_ci95"])
for f, key in [("0.1", "Ten"), ("0.25", "TwentyFive"), ("0.5", "Fifty")]:
    x = e31c[f]
    macros[f"FTfrac{key}"] = milli(x["delta"])
    macros[f"FTfrac{key}CI"] = ci_milli(x["delta_ci95"])
    macros[f"FTfrac{key}AUC"] = f"{x['image']['auroc']:.2f}"
e31d = load("E31d_ft_fraction_ext")
if e31d:  # one population (complete cases), two subsamples at 10-50%; overrides the E31c-only macros
    for f, key in [(0.1, "Ten"), (0.25, "TwentyFive"), (0.5, "Fifty"), (1.0, "Full")]:
        rr = [e31d[k] for k in e31d if "|" in k and float(k.split("|")[0]) == f]
        macros[f"FTfrac{key}"] = milli(np.mean([r["delta"] for r in rr]))
        macros[f"FTfrac{key}AUC"] = f"{np.mean([r['image']['auroc'] for r in rr]):.3f}"
        macros[f"FTfrac{key}Runs"] = ", ".join(milli(r["delta"]) for r in rr)
    full = [e31d[k] for k in e31d if k.startswith("1.0|")][0]
    macros["FTfracFullCI"] = ci_milli(full["delta_ci95"])
    macros["FTfracSpearman"] = f"{e31d['spearman_fraction_vs_delta']:.2f}"
    macros["FTfracNruns"] = str(sum(1 for k in e31d if "|" in k))
fr = lambda ds, rung, f: np.mean([v["delta"] for k, v in e31b[ds].items() if k.startswith(f"{rung}|{f}|")])
macros["FZhamTen"], macros["FZhamFull"] = milli(fr("ham10000", "resnet18", 0.1)), milli(fr("ham10000", "resnet18", 1.0))
macros["FZisicTen"], macros["FZisicFull"] = milli(fr("isic2019_nonham", "resnet18", 0.1)), milli(fr("isic2019_nonham", "resnet18", 1.0))
# within-family size ladders
comps = [(ds, k, v) for ds, x in e31a.items() for k, v in x.items() if ":" in k]
macros["SIZEn"] = str(len(comps))
macros["SIZEsig"] = str(sum(v["ci95"][0] > 0 or v["ci95"][1] < 0 for _, _, v in comps))
aucs = [(ds, x[a]["image"]["auroc"], x[b]["image"]["auroc"]) for ds, x in e31a.items()
        for a, b in [("resnet18", "resnet101"), ("convnext_tiny", "convnext_base")]]
macros["SIZEaucChangeMin"] = f"{min(b - a for _, a, b in aucs):+.3f}"
macros["SIZEaucChangeMax"] = f"{max(b - a for _, a, b in aucs):+.3f}"
macros["SIZEdatasets"] = str(len(e31a))
# ViT-L/16 SWAG (304M) vs ConvNeXt-T
gains = [e18[d]["vitl16_swag"]["image_auroc"] - e18[d]["convnext_tiny"]["image_auroc"] for d in e18]
macros["VITLaucMin"], macros["VITLaucMax"] = f"{min(gains):+.3f}", f"{max(gains):+.3f}"
macros["VITLlower"] = ", ".join(lab[d] for d in e18 if e18[d]["convnext_minus_vitl"]["ci95"][0] > 0)
macros["VITLhigher"] = str(sum(e18[d]["convnext_minus_vitl"]["ci95"][1] < 0 for d in e18))
macros["VITLn"] = str(len(e18))

# ------------------------------------------------------------------ Exploratory: recoverability (E30), shift (E33), rule (E34)
e30, e33, e34 = load("E30_recoverability_utility"), load("E33_shift"), load("E34_collection_rule")
macros.update({
    "RECrows": str(e30["n_rows_used"]), "RECrowsAll": str(e30["n_rows_total"]),
    "RECrho": f"{e30['spearman_recov_within_vs_retained']:.2f}",
    "RECpartial": f"{e30['partial_spearman_within_given_Ivar_auroc']:.2f}",
    "RECpartialCI": "[{:.2f}, {:.2f}]".format(*e30["partial_spearman_dataset_cluster_ci95"]),
    "RECrungMin": f"{max(e30['spearman_by_rung'].values()):.2f}", "RECrungMax": f"{min(e30['spearman_by_rung'].values()):.2f}",
})
_t = pd.read_csv(RES / "E30_table.csv")
_t = _t[(_t.I_var_lo > 0) & _t.recov_within_norm.notna() & (_t.rung != "hsv")].copy()
_t["ret"] = _t.delta_var / _t.I_var
from scipy.stats import spearmanr as _sp
_rs = {k: _sp(g.recov_within_norm, g.ret)[0] for k, g in _t.groupby("dataset") if len(g) > 2}
macros.update({"RECdeepNeg": str(sum(v < 0 for v in _rs.values())), "RECdeepN": str(len(_rs)),
               "RECdeepMin": f"{max(_rs.values()):.2f}", "RECdeepMax": f"{min(_rs.values()):.2f}",
               "RECdeepMinRows": str(min(len(g) for _, g in _t.groupby("dataset")))})
e19h = [d for d in e19 if e19[d].get("prediction_holds")] if (e19 := load("E19_resolution_ladder")) else []
macros["RESholdsN"] = str(len(e19h))
tr = e33["transfer"]
sig = lambda v: (v["added_value_ci95"][0] > 0) - (v["added_value_ci95"][1] < 0)
macros.update({"SHIFTn": str(len(tr)), "SHIFThelp": str(sum(sig(v) == 1 for v in tr.values())),
               "SHIFThurt": str(sum(sig(v) == -1 for v in tr.values())),
               "SHIFTmin": f"{min(v['added_value'] for v in tr.values()):.3f}",
               "SHIFTmax": f"{max(v['added_value'] for v in tr.values()):.3f}"})
macros.update({"RULEzero": f"{e34['pred_b0']['mae']:.4f}", "RULEbase": f"{e34['pred_b1']['mae']:.4f}",
               "RULEridge": f"{e34['pred_ridge']['mae']:.4f}"})

# ------------------------------------------------------------------ Appendix tables (all remaining experiments)
P3 = lambda x: f"{x:.3f}"
# E09 simulation
e09 = load("E09_attenuation_sim")
sim_rows = []
for sc, nm in [("mediated", "Through the image"), ("direct", "Direct"), ("mixed", "Both"), ("none", "None")]:
    a, b = e09[sc][0], e09[sc][-1]
    sim_rows.append(f"{nm} & {1000*a['added_value_mean']:.1f} & {1000*b['added_value_mean']:.1f} & {a['recoverability_r2']:.2f} & {b['recoverability_r2']:.2f}\\\\")
tables["app_sim"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Simulation (E09; $n=3{,}000$, 10 repetitions, 6 image-noise levels). Mean added value ($10^{-3}$ nats) and recoverability ($R^2$) of the auxiliary variable from the image at the weakest and strongest image model.}\label{tab:sim}
\begin{tabular}{lcccc}\toprule
 & \multicolumn{2}{c}{$\hat\Delta$} & \multicolumn{2}{c}{Recoverability}\\
Auxiliary effect & weakest & strongest & weakest & strongest\\\midrule
""" + "\n".join(sim_rows) + r"""
\bottomrule\end{tabular}\end{table}"""
# E19 resolution
e19 = load("E19_resolution_ladder")
RES_ = ["16", "32", "64", "112", "224"]
res_rows = []
for d in [d for d in ALLDS if d in e19]:
    x = e19[d]
    res_rows.append(f"{lab[d]} & {x['16']['image_auroc']:.3f} & {x['224']['image_auroc']:.3f} & " +
                    " & ".join(milli(x[r]["added_value"]) for r in RES_) + f" & {x['spearman_res_vs_av']:.1f}\\\\")
tables["app_res"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Resolution ladder (E19, exploratory): frozen ConvNeXt-T on images downsampled to $r\times r$ pixels and resized back. Image AUROC at the extremes; $\hat\Delta$ ($10^{-3}$ nats) at each resolution; Spearman correlation of resolution with $\hat\Delta$ (prediction: $-1$). The manipulation is weak: image AUROC barely depends on resolution. All images (PAD-UFES-20 includes its missingness leak).}\label{tab:res}
\begin{tabular}{lccccccccc}\toprule
 & \multicolumn{2}{c}{Image AUROC} & \multicolumn{5}{c}{$\hat\Delta$ at resolution} & \\
Dataset & 16 px & 224 px & 16 & 32 & 64 & 112 & 224 & Spearman\\\midrule
""" + "\n".join(res_rows) + r"""
\bottomrule\end{tabular}\end{table*}"""
macros["RESholds"] = ", ".join(lab[d] for d in e19 if e19[d].get("prediction_holds"))
macros["RESn"] = str(len(e19))
# E20 per-variable (PAD-UFES-20 excluded: invalidated by the missingness audit)
e20 = load("E20_per_variable_floor")
pv_rows = []
for d in ["ham10000", "isic2020", "isic2019_nonham"]:
    for v in ["age", "sex", "site"]:
        x = e20[d][v]
        ret = "--" if x["retained"] is None or x["hsv"]["ci95"][0] <= 0 else f"{x['retained']:.2f}"
        pv_rows.append(f"{lab[d] if v == 'age' else ''} & {v} & {milli(x['hsv']['av'])} {ci_milli(x['hsv']['ci95'])} & "
                       f"{milli(x['convnext_tiny']['av'])} {ci_milli(x['convnext_tiny']['ci95'])} & {ret}\\\\")
tables["app_pervar"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Single-variable added value (E20, exploratory; $10^{-3}$ nats, 95\% CI, all images). Retained: ConvNeXt-T / HSV, shown only where the HSV value is significant. PAD-UFES-20 is omitted: its per-variable results were driven by label-informative missingness (Section~\ref{sec:cc}).}\label{tab:pervar}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccc}\toprule
Dataset & Variable & HSV & ConvNeXt-T & Retained\\\midrule
""" + "\n".join(pv_rows) + r"""
\bottomrule\end{tabular}\end{table}"""
# E31a family size
fam_rows = []
for d in e31a:
    x = e31a[d]
    cells = " & ".join(f"{x[r]['image']['auroc']:.3f} / {milli(x[r]['delta'])}" for r in
                       ["resnet18", "resnet34", "resnet50", "resnet101", "convnext_tiny", "convnext_small", "convnext_base"])
    fam_rows.append(f"{lab[d]} & {cells}\\\\")
tables["app_family"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Backbone size within families (E31a, exploratory; frozen, complete cases, fold assignment 0). Each cell: image AUROC / $\hat\Delta$ ($10^{-3}$ nats). No smallest-vs-largest difference in $\hat\Delta$ was significant (paired grouped bootstrap).}\label{tab:family}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccccc}\toprule
Dataset & ResNet-18 & ResNet-34 & ResNet-50 & ResNet-101 & ConvNeXt-T & ConvNeXt-S & ConvNeXt-B\\\midrule
""" + "\n".join(fam_rows) + r"""
\bottomrule\end{tabular}\end{table*}"""
# E31b scaling (frozen)
sc_rows = []
for d in e31b:
    for rung in ["resnet18", "convnext_tiny"]:
        cells = []
        for f in [0.1, 0.25, 0.5, 1.0]:
            rr = [v for k, v in e31b[d].items() if k.startswith(f"{rung}|{f}|")]
            cells.append(f"{np.mean([r['image']['auroc'] for r in rr]):.3f} / {milli(np.mean([r['delta'] for r in rr]))}")
        sc_rows.append(f"{lab[d] if rung == 'resnet18' else ''} & {NAMES[rung]} & " + " & ".join(cells) + "\\\\")
tables["app_scaling"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Training-set size with frozen features (E31b, exploratory; complete cases). The image head sees 10--100\% of each training fold's groups; the metadata stage uses all folds. Each cell: image AUROC / $\hat\Delta$ ($10^{-3}$ nats), mean of two subsamples below 100\%.}\label{tab:scaling}
\begin{tabular}{llcccc}\toprule
Dataset & Features & 10\% & 25\% & 50\% & 100\%\\\midrule
""" + "\n".join(sc_rows) + r"""
\bottomrule\end{tabular}\end{table*}"""
# E18 ViT-L
v_rows = []
for d in [d for d in ALLDS if d in e18]:
    x = e18[d]
    v_rows.append(f"{lab[d]} & {x['convnext_tiny']['image_auroc']:.3f} & {x['vitl16_swag']['image_auroc']:.3f} & "
                  f"{milli(x['convnext_tiny']['added_value'])} & {milli(x['vitl16_swag']['added_value'])} & "
                  f"{milli(x['convnext_minus_vitl']['diff'])} {ci_milli(x['convnext_minus_vitl']['ci95'])}\\\\")
tables["app_vitl"] = r"""\begin{table}[t]\centering\footnotesize
\caption{A larger pretrained representation (E18, exploratory; frozen, all images, fold assignment 0): ConvNeXt-T vs.\ ViT-L/16 SWAG~\cite{singh2022swag}. $\hat\Delta$ and the paired difference (ConvNeXt $-$ ViT-L) in $10^{-3}$ nats.}\label{tab:vitl}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccccc}\toprule
 & \multicolumn{2}{c}{Image AUROC} & \multicolumn{2}{c}{$\hat\Delta$} & \\
Dataset & CNX-T & ViT-L & CNX-T & ViT-L & Difference\\\midrule
""" + "\n".join(v_rows) + r"""
\bottomrule\end{tabular}\end{table}"""
# E32 permutation control
p_rows = []
for d in ALLDS:
    c = e32[d]["complete_case"]
    x = c["convnext_tiny"]
    p_rows.append(f"{lab[d]} & {milli(x['matched'])} & {milli(x['perm_mean'])} & {milli(x['perm_max'])} & "
                  f"{100*e32[d]['original']['implausible_real']:.1f} / {100*e32[d]['original']['implausible_permuted_mean']:.1f}\\\\")
tables["app_perm"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Permutation control (E32, exploratory; ConvNeXt-T, complete cases): $\hat\Delta$ ($10^{-3}$ nats) with the real metadata and the mean and maximum over %s permutations of whole metadata records between grouping units. Last column: \%% of records with an implausible combination (e.g.\ sex-specific site), real / permuted, all images.}\label{tab:perm}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lcccc}\toprule
Dataset & Real & Perm.\ mean & Perm.\ max & Implausible (\%%)\\\midrule
""" % macros["PERMR"] + "\n".join(p_rows) + r"""
\bottomrule\end{tabular}\end{table}"""
# E33 shift matrices
def cell(v):
    m = milli(v["added_value"])
    lo, hi = v["added_value_ci95"]
    return m + ("$^{+}$" if lo > 0 else "$^{-}$" if hi < 0 else "")
blocks = []
for fam in (["ham10000", "isic2020", "isic2019_nonham"], ["pad_ufes", "hiba", "msk2020", "milk10k_derm", "milk10k_clin"]):
    head = " & ".join(lab[t] for t in fam)
    rows = []
    for a_ in fam:
        cs = []
        for b_ in fam:
            if a_ == b_:
                v = e33["in_distribution"][a_]
                cs.append("\\textit{" + milli(v["added_value"]) + "}")
            else:
                cs.append(cell(e33["transfer"][f"{a_}->{b_}"]))
        rows.append(f"{lab[a_]} & " + " & ".join(cs) + "\\\\")
    blocks.append((len(fam), head, rows))
mat = []
for n_, head, rows in blocks:
    mat.append("\\multicolumn{%d}{l}{\\textbf{Source} $\\downarrow$ / \\textbf{target} $\\rightarrow$}\\\\" % (n_ + 1))
    mat.append(f" & {head}\\\\\\midrule")
    mat += rows
    mat.append("\\midrule")
mat = mat[:-1]
tables["app_shift"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Metadata added value under dataset shift (E33, exploratory; $10^{-3}$ nats). Image head and fusion model fitted on the source only (frozen ConvNeXt-T; harmonised age, sex, coarse site; complete cases) and applied unchanged to the target. $^{+}$/$^{-}$: 95\% target-group bootstrap CI above/below 0. Diagonal (italic): the target's own cross-validated value with the same harmonised metadata. Melanoma and malignancy datasets are paired only within their label family. Differences in prevalence and label definition between datasets are not separated from metadata effects.}\label{tab:shift}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}\toprule
""" + "\n".join(mat) + r"""
\bottomrule\end{tabular}\end{table*}"""
loso = e33["ham_leave_one_source_out"]
macros["LOSOtext"] = "; ".join(f"\\texttt{{{k.replace('_', chr(92) + '_')}}} ${milli(v['added_value'])}$ {ci_milli(v['added_value_ci95'])}" for k, v in loso.items())
# E34 rule
r_rows = []
for m, nm in [("pred_b0", "Predict zero"), ("pred_b1", "$\\hat I_{\\mathrm{var}}\\times$ median retained"), ("pred_ridge", "Ridge (6 features)")]:
    x = e34[m]
    dd = x["decision"]
    f2 = lambda v: "--" if v is None else f"{v:.2f}"
    r_rows.append(f"{nm} & {1000*x['mae']:.1f} & {f2(x['spearman'])} & {f2(dd['precision'])} & {f2(dd['recall'])} & {dd['accuracy']:.2f}\\\\")
tables["app_rule"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Predicting a variable's added value in a held-out dataset (E34, exploratory; leave-one-dataset-out, MILK10k held out as one; %d dataset--representation--variable rows). MAE in $10^{-3}$ nats. Decision rule fixed in advance: collect if predicted $\hat\Delta_{\mathrm{var}}\ge10^{-3}$ nats.}\label{tab:rule}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}\toprule
Predictor & MAE & Spearman & Precision & Recall & Accuracy\\\midrule
""" % e34["n_rows"] + "\n".join(r_rows) + r"""
\bottomrule\end{tabular}\end{table}"""
macros["RULEbaseAcc"] = f"{e34['pred_b1']['decision']['accuracy']:.2f}"
macros["RULEbasePrec"] = f"{e34['pred_b1']['decision']['precision']:.2f}"
macros["RULEbaseRec"] = f"{e34['pred_b1']['decision']['recall']:.2f}"

# ------------------------------------------------------------------ Iteration 5 (E40-E49)
exec((Path(__file__).parent / "iter5_numbers.py").read_text())

# ------------------------------------------------------------------ Iteration 6 (E50-E55)
exec((Path(__file__).parent / "iter6_numbers.py").read_text())

import re
MINUS = re.compile(r"(?<![\w$\\{}\-])-(?=\d)")  # a hyphen directly before a digit, not inside words or braces
fixneg = lambda t: MINUS.sub(r"\\ensuremath{-}", t)
macros = {k: fixneg(v) for k, v in macros.items()}
tables = {k: fixneg(v) for k, v in tables.items()}
(PAPER / "tables").mkdir(exist_ok=True)
for k, v in tables.items():
    # [tp]: allow float-only pages, so appendix tables pack instead of queueing until \clearpage (draft v7)
    v = v.replace("\\begin{table}[t]", "\\begin{table}[tp]").replace("\\begin{table*}[t]", "\\begin{table*}[tp]")
    if "table*" not in v:  # single-column tables: scale to the column width (review item m1)
        v = v.replace("\\begin{tabular}", "\\resizebox{\\columnwidth}{!}{\\begin{tabular}").replace(
            "\\end{tabular}", "\\end{tabular}}")
    (PAPER / "tables" / f"{k}.tex").write_text(v + "\n")
(PAPER / "numbers.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in macros.items()) + "\n")
print(f"wrote {len(macros)} macros, {len(tables)} tables")
