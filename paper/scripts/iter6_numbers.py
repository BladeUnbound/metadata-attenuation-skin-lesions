"""Iteration-6 macros and tables (E50-E55), executed by make_numbers.py after iter5_numbers.py.

Every value is read from research/results/*.json, except training-set sizes, which come from the fold arrays that
E50 saved next to its results (research/data/<set>/e50/age_convnext_tiny_f0.npz).
"""
import json

import numpy as np

# Executed inside make_numbers.py's namespace (load, macros, tables, pct, PAPER, RES come from there).

SETS6 = ["isic2020", "hiba", "msk2020", "ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes", "milk10k_clin"]
LARGE6 = ["isic2020", "ham10000", "isic2019_nonham"]  # n >> d; eraser test/train SD ratio within [0.8, 1.25]
LARGE5 = LARGE6 + ["milk10k_derm", "milk10k_clin"]
SMALL6 = ["hiba", "msk2020", "pad_ufes"]
LAB6 = {"isic2020": "ISIC 2020", "hiba": "HIBA", "msk2020": "MSK 2020", "ham10000": "HAM10000",
        "isic2019_nonham": "ISIC 2019", "milk10k_derm": "MILK10k (derm.)", "pad_ufes": "PAD-UFES-20$^\\ddagger$",
        "milk10k_clin": "MILK10k (clin.)"}
ROLE6 = {d: "Confirm." for d in ["isic2020", "hiba", "msk2020"]} | {
    d: "Replic." for d in ["ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes"]} | {"milk10k_clin": "Correl."}
DAG = r"$^\dagger$"


def c2(c, k="diff_x1e3", kc="ci95_x1e3"):
    return f"{c[k]:.2f} [{c[kc][0]:.2f}, {c[kc][1]:.2f}]"


def rng(vals, fmt="{:.2f}"):
    return f"{fmt.format(min(vals))}--{fmt.format(max(vals))}"


def z2(x):  # two decimals without a spurious minus sign on zero
    return f"{0.0 if abs(x) < 0.005 else x:.2f}"


def rank_range(v):
    return f"{min(v)}" if min(v) == max(v) else f"{min(v)}--{max(v)}"


def sd_ratio(r, arm="T-D1-1.0"):
    dg = r["meta"]["eraser_diag"][arm]
    return float(np.mean([x["sd_test"] / x["sd_train"] for x in dg]))


e50 = {d: load(f"E50_{d}_age_convnext_tiny_f0") for d in SETS6}
e50c = {d: load(f"E50_{d}_age_convnext_tiny_f0_light_cond") for d in SETS6}
e51 = {d: load(f"E51_{d}_site_convnext_tiny_f0_light") for d in ["hiba", "ham10000", "pad_ufes", "milk10k_derm"]}
e50b = {(d, f): load(f"E50_{d}_age_convnext_tiny_f{f}_light_min") for d in LARGE6 for f in (1, 2)}
e50r = {d: load(f"E50_{d}_age_resnet18_f0_light_min") for d in ["ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes"]}
e52 = {d: load(f"E52_{d}_age") for d in SETS6}
e53 = {d: load(f"E53_{d}") for d in SETS6}
e53b = {d: load(f"E53b_{d}") for d in LARGE5}
e54 = {d: load(f"E54_{d}") for d in ["hiba", "pad_ufes"]}
e55 = load("E55_shift_rules")
diag = load("E50_age_label_diag")
gate = json.loads((RES / "E50_gate.json").read_text())
prim = lambda r: r["contrasts"]["gain_age"]["targeted_minus_matched"]

# ------------------------------------------------------------------ E50a: confirmatory family and main table
holm6 = gate["holm"]
macros["ERAconfN"] = str(len(holm6))
macros["ERAconfRej"] = str(sum(v["reject"] for v in holm6.values()))
macros["ERAisicHolm"] = f"{holm6['isic2020:age']['p_holm']:.3f}"
for d, k in [("isic2020", "Isic"), ("ham10000", "Ham"), ("isic2019_nonham", "IsicNt"), ("milk10k_derm", "Milk"),
             ("milk10k_clin", "MilkClin"), ("pad_ufes", "Pad")]:
    macros[f"ERA{k}"] = c2(prim(e50[d]))
macros["ERAIsicSham"] = c2(e50["isic2020"]["contrasts"]["gain_age"]["targeted_minus_sham"])
macros["ERAhiba"] = f"{prim(e50['hiba'])['diff_x1e3']:.1f}"
macros["ERAmsk"] = f"{prim(e50['msk2020'])['diff_x1e3']:.1f}"
macros["ERAhibaSite"] = f"{e51['hiba']['contrasts']['gain_site']['targeted_minus_matched']['diff_x1e3']:.1f}"
macros["ERAhibaAUCzero"] = f"{e50['hiba']['arms']['t0']['image_auroc']:.2f}"
macros["ERAhibaAUCone"] = f"{e50['hiba']['arms']['T-D1-1.0']['image_auroc']:.2f}"
for d, k in [("hiba", "Hiba"), ("msk2020", "Msk"), ("isic2020", "Isic")]:
    macros[f"AGEauc{k}"] = f"{diag[d]['age_auroc_for_label']:.2f}"
    macros[f"AGEcorr{k}"] = f"{diag[d]['corr_score_age']:.2f}"
isic_f = [prim(e50["isic2020"])["diff_x1e3"]] + [prim(e50b[("isic2020", f)])["diff_x1e3"] for f in (1, 2)]
macros["ERAisicFolds"] = rng(isic_f)
macros["ERAfoldsAllPos"] = str(sum(prim(e50b[(d, f)])["ci95_x1e3"][0] > 0 for d in LARGE6 for f in (1, 2))
                               + sum(prim(e50[d])["ci95_x1e3"][0] > 0 for d in LARGE6))
macros["ERAisicFull"] = c2(e53b["isic2020"]["contrasts"]["b_targeted_minus_matched"])

# eraser overfit: test/train SD ratio of the targeted coordinate, every targeted eraser (age marginal, age
# class-conditional, site) on the small and on the large sets; training-set sizes from the saved folds
sm = [sd_ratio(r) for d in SMALL6 for r in (e50[d], e50c[d], e51.get(d)) if r]
lg = [sd_ratio(r) for d in LARGE5 for r in (e50[d], e50c[d], e51.get(d)) if r]
macros["ERAsdSmall"], macros["ERAsdLarge"] = rng(sm, "{:.1f}"), rng(lg)
ntr = []
for d in SMALL6:
    fold = np.load(RES.parent / f"data/{d}/e50/age_convnext_tiny_f0.npz")["fold"]
    ntr += [int((fold != k).sum()) for k in range(5)]
macros["ERAntrain"] = f"{min(ntr)}--{max(ntr)}"

# mechanism quantities on the large sets
cc = {d: e50[d]["contrasts"] for d in SETS6}
macros["ERAdoseAllPos"] = str(sum(all(cc[d]["dose_D2"][t]["targeted_minus_matched"]["ci95_x1e3"][0] > 0
                                      for t in ["0.25", "0.5", "0.75", "1.0"]) for d in LARGE6))
macros["ERArfT"] = rng([cc[d]["restoration_fraction_targeted"]["RF"] for d in LARGE6])
macros["ERArfR"] = rng([cc[d]["restoration_fraction_matched"]["RF"] for d in LARGE6])
rise = {d: cc[d]["gain_age"]["targeted_minus_t0"]["diff_x1e3"] for d in SETS6}
back = [1 - cc[d]["restoration_reappend_T"]["delta_gain_vs_t0"]["diff_x1e3"] / rise[d] for d in LARGE6]
macros["ERAreapp"] = f"{100 * min(back):.0f}--{pct(max(back))}"
macros["ERAahatN"] = str(sum(cc[d]["image_derived_age"]["ahat_T_minus_age_rise"]["ci95_x1e3"][0] <= 0
                             <= cc[d]["image_derived_age"]["ahat_T_minus_age_rise"]["ci95_x1e3"][1] for d in LARGE6))
mlp_t = [e52[d]["arms"]["T-D1-1.0"]["mlp_r2"] for d in LARGE5]
macros["ERAmlp"] = rng(mlp_t)
macros["ERAmlpPos"] = str(sum(e52[d]["arms"]["T-D1-1.0"]["mlp_r2_ci95"][0] > 0 for d in LARGE5))
macros["ERAmlpIsicZero"] = f"{e52['isic2020']['arms']['t0']['mlp_r2']:.2f}"
macros["ERAmlpIsicOne"] = f"{e52['isic2020']['arms']['T-D1-1.0']['mlp_r2']:.2f}"
macros["ERAmlpIsicRand"] = f"{e52['isic2020']['arms']['R-D1-1.0-0']['mlp_r2']:.2f}"
macros["ERAlinIsicZero"] = f"{e50['isic2020']['arms']['t0']['probe_age']['r2_within_test']:.2f}"
macros["ERAlinIsicOne"] = f"{e50['isic2020']['arms']['T-D1-1.0']['probe_age']['r2_within_test']:.2f}"

# continuity with E45b: every (set, backbone) pair whose E45b claim held
e45b6 = {d: load(f"E45b_{d}") for d in ["ham10000", "isic2019_nonham", "milk10k_derm", "pad_ufes"]}
claims = [(d, r) for d, x in e45b6.items() for r in ("convnext_tiny", "resnet18") if x[r]["claim_holds"]]
new = lambda d, r: prim(e50[d] if r == "convnext_tiny" else e50r[d])
macros["ERAcontN"] = str(len(claims))
macros["ERAcontHeld"] = str(sum(new(d, r)["ci95_x1e3"][0] > 0 for d, r in claims))
macros["ERAmilkOld"] = f"{e45b6['milk10k_derm']['convnext_tiny']['concept_minus_matched']['diff']:.1f}"
for d, k in [("ham10000", "Ham"), ("isic2019_nonham", "IsicNt"), ("milk10k_derm", "Milk"), ("pad_ufes", "Pad")]:
    macros[f"ERArn{k}"] = c2(prim(e50r[d]))

# class-conditional erasure (E50c, exploratory)
cond_ok = [d for d in LARGE5 if prim(e50c[d])["ci95_x1e3"][0] > 0 and
           e50c[d]["contrasts"]["matching_check_image_ll_targeted_minus_matched"]["ci95_x1e3"][0] <= 0 <=
           e50c[d]["contrasts"]["matching_check_image_ll_targeted_minus_matched"]["ci95_x1e3"][1]]
macros["ERAcondN"] = str(len(cond_ok))
macros["ERAcond"] = rng([prim(e50c[d])["diff_x1e3"] for d in LARGE5])
ll = lambda r, a: r["arms"][a]["image_logloss"]
saving = [1 - (ll(e50c[d], "T-D1-1.0") - ll(e50c[d], "t0")) / (ll(e50[d], "T-D1-1.0") - ll(e50[d], "t0")) for d in LARGE5]
macros["ERAcondCost"] = f"{100 * min(saving):.0f}--{pct(max(saving))}"

# site erasure and the registered specificity matrix (E51)
for d, k in [("ham10000", "Ham"), ("milk10k_derm", "Milk")]:
    macros[f"SITE{k}"] = c2(e51[d]["contrasts"]["gain_site"]["targeted_minus_matched"])
    sp = gate["specificity"][d]
    for key, kk in [("M(age,site)", "AgeSite"), ("M(site,age)", "SiteAge")]:
        macros[f"SPEC{k}{kk}"] = c2(sp[key])
macros["SPECboth"] = str(sum(v["selective_age"] and v["selective_site"] for v in gate["specificity"].values() if v))

tables["erasure"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Erasing the linear trace of age (E50, ConvNeXt-T, fold assignment 0; $10^{-3}$ nats). $R^2$: age decodability in held-out images before and after erasure (within-test-fold ridge probe). $\hat\Delta_{\rm age}$: added value of age with unchanged features, after erasure, and after removing performance-matched random subspaces (mean of 5; per-fold ranks chosen by inner cross-validation on training data only). Contrasts are paired differences with 95\% grouped-bootstrap CIs. Sham: eraser fitted to permuted age (mean of 5); $^\dagger$95\% CI above 0. Matching check: image-only log-loss, erasure minus matched control (negative: the control removed more image information, so the contrast is conservative). SD ratio: test/training standard deviation of the erased coordinate (overfit eraser if far from 1). Confirm.: registered confirmatory family (Holm, @CONFN@ tests); Replic.: datasets used in earlier erasure experiments; Correl.: shares lesions with MILK10k (derm.). $^\ddagger$Complete cases.}\label{tab:erasure}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{llccccccccc}\toprule
 & & & & \multicolumn{3}{c}{$\hat\Delta_{\rm age}$} & & & & \\\cmidrule(lr){5-7}
Image set & Role & Matched rank & $R^2$ ($t{=}0\to1$) & $t{=}0$ & Erased & Matched & Erased $-$ matched & Erased $-$ sham & Matching check & SD ratio\\\midrule
""" + "\n".join(
    f"{LAB6[d]} & {ROLE6[d]} & {rank_range(e50[d]['meta']['rank_matched'])} & "
    f"{z2(e50[d]['arms']['t0']['probe_age']['r2_within_test'])} $\\to$ {z2(e50[d]['arms']['T-D1-1.0']['probe_age']['r2_within_test'])} & "
    f"{e50[d]['arms']['t0']['delta_x1e3']['age']:.2f} & {e50[d]['arms']['T-D1-1.0']['delta_x1e3']['age']:.2f} & "
    f"{np.mean(cc[d]['gain_age']['matched_each_x1e3']):.2f} & {c2(prim(e50[d]))} & {cc[d]['gain_age']['targeted_minus_sham']['diff_x1e3']:.2f}{DAG if cc[d]['gain_age']['targeted_minus_sham']['ci95_x1e3'][0] > 0 else ''} & "
    f"{c2(cc[d]['matching_check_image_ll_targeted_minus_matched'])} & {sd_ratio(e50[d]):.2f}\\\\" for d in SETS6) + r"""
\bottomrule\end{tabular}\end{table*}"""
tables["erasure"] = tables["erasure"].replace("@CONFN@", macros["ERAconfN"])
tables["erasure"] = tables["erasure"].replace("\\begin{tabular}", "\\resizebox{\\textwidth}{!}{\\begin{tabular}").replace("\\end{tabular}", "\\end{tabular}}")

# ------------------------------------------------------------------ appendix: mechanism details
mrows = []
for d in SETS6:
    dose = " & ".join(f"{cc[d]['dose_D2'][t]['targeted_minus_matched']['diff_x1e3']:.2f}"
                      + (DAG if cc[d]['dose_D2'][t]['targeted_minus_matched']['ci95_x1e3'][0] > 0 else "")
                      for t in ["0.25", "0.5", "0.75", "1.0"])
    rf = lambda k: f"{cc[d][k]['RF']:.2f}" if "RF" in cc[d][k] else "--"
    m = e52[d]["arms"]
    mrows.append(f"{LAB6[d]} & {dose} & {rf('restoration_fraction_targeted')} & {rf('restoration_fraction_matched')} & "
                 f"{c2(cc[d]['restoration_reappend_T']['delta_gain_vs_t0'])} & {c2(cc[d]['image_derived_age']['ahat_T_minus_age_rise'])} & "
                 f"{m['t0']['mlp_r2']:.2f} / {m['T-D1-1.0']['mlp_r2']:.2f} / {m['R-D1-1.0-0']['mlp_r2']:.2f}\\\\")
tables["app_erasure"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Mechanism checks for age erasure (E50, E52; ConvNeXt-T, fold assignment 0; $10^{-3}$ nats). Dose: erased coordinates mixed with independent noise, $c_t=\sqrt{1-t}\,c+\sqrt t\,\sigma\varepsilon$, so linear decodability falls with $t$; entries are erasure minus matched random subspaces at the same dose ($^\dagger$95\% CI above 0). RF: rise in age's added value per unit rise in image-only log-loss, for erasure and for matched random subspaces (shown only where the denominator's CI excludes 0). Re-append: added value with the erased coordinate appended back, minus $t{=}0$. $\hat a$: cross-fitted image-derived age entered like a metadata variable after erasure, its added value minus the rise in age's added value (0 under pure redundancy). MLP $R^2$: held-out age decodability by a registered one-hidden-layer probe at $t=0$ / after erasure / after one matched random subspace.}\label{tab:apperasure}
\setlength{\tabcolsep}{2.5pt}
\begin{tabular}{lccccccccc}\toprule
 & \multicolumn{4}{c}{Dose $t$ (erasure $-$ matched)} & \multicolumn{2}{c}{RF} & & & \\\cmidrule(lr){2-5}\cmidrule(lr){6-7}
Image set & 0.25 & 0.5 & 0.75 & 1 & Erased & Matched & Re-append & $\hat a$ $-$ rise & MLP $R^2$\\\midrule
""" + "\n".join(mrows) + r"""
\bottomrule\end{tabular}\end{table*}"""

rrows = []
for d in SETS6:
    conds = []
    if d in LARGE6:
        conds += [(f"Fold assignment {f}", e50b[(d, f)]["contrasts"], True) for f in (1, 2)]
    if d in e50r:
        conds.append(("ResNet-18", e50r[d]["contrasts"], True))
    if d in e53b:
        b = e53b[d]["contrasts"]
        conds.append(("Full-feature fusion", {"gain_age": {"targeted_minus_matched": b["b_targeted_minus_matched"]},
                                              "matching_check_image_ll_targeted_minus_matched": b["b_matching_check_image_ll_targeted_minus_matched"]}, False))
    conds.append(("Class-conditional", e50c[d]["contrasts"], True))
    for i, (name, c, has_sham) in enumerate(conds):
        sham = c2(c["gain_age"]["targeted_minus_sham"]) if has_sham else "--"
        rrows.append(f"{LAB6[d] if i == 0 else ''} & {name} & {c2(c['gain_age']['targeted_minus_matched'])} & {sham} & "
                     f"{c2(c['matching_check_image_ll_targeted_minus_matched'])}\\\\")
tables["app_erasure_rob"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Robustness of the age-erasure contrast (E50b, E50c, E53b; $10^{-3}$ nats, 95\% grouped-bootstrap CIs). Fold assignments 1 and 2 and ResNet-18 use the primary arms only (erasure, nested matched random subspaces, shams). Full-feature fusion: both stages replaced by an $L_2$ logistic model on [features, age] with the same erasers. Class-conditional (exploratory, defined after the confirmatory results): the eraser removes the image's covariance with age minus its training-fold class mean, i.e.\ the age trace not explained by the label; it is applied without labels. Matching check as in Table~\ref{tab:erasure}.}\label{tab:apperasurerob}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{llccc}\toprule
Image set & Condition & Erased $-$ matched & Erased $-$ sham & Matching check\\\midrule
""" + "\n".join(rrows) + r"""
\bottomrule\end{tabular}\end{table*}"""

srows = []
for d in ["ham10000", "milk10k_derm", "hiba", "pad_ufes"]:
    sp, c = gate["specificity"][d], e51[d]["contrasts"]
    for i, (E, o) in enumerate([("age", "site"), ("site", "age")]):
        sel = sp[f"selective_{E}"]
        srows.append(f"{LAB6[d] if i == 0 else ''} & {E.capitalize()} & {c2(sp[f'M({E},age)'])} & {c2(sp[f'M({E},site)'])} & "
                     f"{c2(sp[f'M({E},{E})-M({E},{o})'])} & {YES if sel else NO}\\\\")
tables["app_specificity"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Attribute specificity (E50, E51; ConvNeXt-T, fold assignment 0; $10^{-3}$ nats, 95\% grouped-bootstrap CIs). Each entry is erasure minus performance-matched random subspaces, in the added value of age or of site. Registered rule: an eraser is selective if its effect on its own variable and the paired difference between its effect on its own and on the other variable both have CIs above 0. HIBA (site) was the confirmatory test; HAM10000 a replication; MILK10k (derm.) and PAD-UFES-20 exploratory. Site erasure's matching check failed on HIBA and PAD-UFES-20 (matched control far more damaging) and passed or was conservative on HAM10000 and MILK10k.}\label{tab:specificity}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llcccc}\toprule
Image set & Eraser & Age's added value & Site's added value & Own $-$ other variable & Selective\\\midrule
""" + "\n".join(srows) + r"""
\bottomrule\end{tabular}\end{table*}"""

# ------------------------------------------------------------------ E53: fusion estimators
EST = ["a_score_fusion", "b_full_feature", "c_hgb", "d_missingness_only"]
ci_pos = lambda v: v["ci95_x1e3"][0] > 0
macros["ESTnSets"] = str(len(SETS6))
for e, k in [("b_full_feature", "B"), ("c_hgb", "C")]:
    macros[f"ESThOne{k}"] = str(sum(e53[d]["H1_hsv_minus_convnext"][e]["p_one_sided"] < 0.05 for d in SETS6))
cnx = {d: e53[d]["convnext_tiny"] for d in SETS6}
macros["ESTdep"] = str(sum(len({ci_pos(cnx[d][e]) for e in EST[:3]}) > 1 for d in SETS6))
boost = [d for d in SETS6 if cnx[d]["c_hgb"]["minus_a_x1e3"]["ci95"][0] > 0]
macros["ESTboostN"] = str(len(boost))
macros["ESTboost"] = rng([cnx[d]["c_hgb"]["minus_a_x1e3"]["diff"] for d in boost if d != "pad_ufes"], "{:.1f}")
share = lambda d: cnx[d]["d_missingness_only"]["delta_x1e3"] / cnx[d]["a_score_fusion"]["delta_x1e3"]
macros["ESTmissPad"], macros["ESTmissIsic"] = pct(share("pad_ufes")), pct(share("isic2019_nonham"))
erows6 = []
for d in SETS6:
    h = e53[d]["H1_hsv_minus_convnext"]
    erows6.append(f"{LAB6[d].replace('$^\\ddagger$', '')} & " + " & ".join(c2(cnx[d][e], 'delta_x1e3', 'ci95_x1e3') for e in EST)
                  + " & " + "\\,".join(YES if h[e]["p_one_sided"] < 0.05 else NO for e in EST[:3]) + "\\\\")
tables["app_estimators"] = r"""\begin{table*}[t]\centering\footnotesize
\caption{Added value of all metadata at ConvNeXt-T under four fusion estimators (E53; $10^{-3}$ nats, 95\% grouped-bootstrap CIs; strictly cross-fitted, same folds, groups and test images; all images). (a) Score fusion: logistic second stage on the image score, as in the rest of the paper. (b) Full-feature fusion: $L_2$ logistic model on [image features, metadata], penalty by grouped inner CV. (c) Boosted fusion: gradient-boosted trees on [image score, raw metadata], fixed hyperparameters. (d) Image score plus missingness indicators only. H1: $\Delta_{\rm HSV}>\Delta_{\rm ConvNeXt}$ (paired, one-sided $p<0.05$) under (a)\,(b)\,(c).}\label{tab:estimators}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}\toprule
Image set & (a) Score & (b) Full-feature & (c) Boosted & (d) Missingness only & H1 (a)(b)(c)\\\midrule
""" + "\n".join(erows6) + r"""
\bottomrule\end{tabular}\end{table*}"""

# ------------------------------------------------------------------ E54: fine-tuning seeds
runs = [(d, bb, s) for d in e54 for bb in e54[d] for s in e54[d][bb]["per_seed"]]
macros["SEEDn"] = str(len(runs))
macros["SEEDheld"] = str(sum(e54[d][bb]["per_seed"][s]["frozen_minus_finetuned_ci95"][0] > 0 for d, bb, s in runs))
macros["SEEDrange"] = f"{1e3 * max(np.ptp([v['added_value'] for v in e54[d][bb]['per_seed'].values()]) for d in e54 for bb in e54[d]):.1f}"

# ------------------------------------------------------------------ E55: decision rules under dataset shift
dl = e55["dataset_level"]
sm55 = e55["dataset_level_summary"]["mean_regret_x1e3"]
macros["RULESn"] = str(e55["dataset_level_summary"]["n_targets"])
for k, kk in [("R0_never", "Never"), ("R1_always", "Always"), ("R2_source_validated", "Src"), ("R3_collection_rule", "Coll")]:
    macros[f"RULES{kk}"] = f"{sm55[k]:.1f}"
macros["RULESthreeUse"] = str(sum(v["R3_use"] for v in dl.values()))
macros["RULEShurt"] = str(sum(v["delta_ci95_x1e3"][1] < 0 for v in dl.values()))
macros["RULESpad"] = c2(dl["malignant:pad_ufes"], "delta_x1e3", "delta_ci95_x1e3")
qrows = []
for k, v in dl.items():
    u = lambda b: "use" if b else "skip"
    qrows.append(f"{LAB6[v['target']].replace('$^\\ddagger$', '')} & {num(v['n_target'])} & {c2(v, 'delta_x1e3', 'delta_ci95_x1e3')} & "
                 f"{u(v['oracle_use'])} & {u(v['R2_use'])} & {u(v['R3_use'])}\\\\")
tables["app_rules"] = r"""\begin{table}[t]\centering\footnotesize
\caption{Whether to use metadata on a new dataset, decided from source datasets only (E55; leave one dataset out within a label family; frozen ConvNeXt-T, harmonised age, sex and site, complete cases; $10^{-3}$ nats). $\Delta$: realised added value on the held-out target. Oracle: the better choice on the target. Source-validated: use if the mean leave-one-source-out added value among the sources is positive. Collection rule: standalone value of the pooled sources times their median retained fraction, use if $\ge10^{-3}$ nats. Mean regret: never @R0@, always @R1@, source-validated @R2@, collection rule @R3@.}\label{tab:rules}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lccccc}\toprule
Target & $n$ & $\Delta$ & Oracle & Source-val. & Coll.\ rule\\\midrule
""" + "\n".join(qrows) + r"""
\bottomrule\end{tabular}\end{table}"""

for k, kk in [("@R0@", "RULESNever"), ("@R1@", "RULESAlways"), ("@R2@", "RULESSrc"), ("@R3@", "RULESColl")]:
    tables["app_rules"] = tables["app_rules"].replace(k, macros[kk])
