"""E33 (exploratory; request E23): external-source and modality-shift evaluation.
Registered in research/iteration4/REGISTRY_E30_E34.md before implementation.

1. Provenance matrix: modality, label, metadata, identifiers, missingness, prevalence,
   source, and pairwise ISIC-ID overlaps (overlapping images are removed from targets).
2. Directed source -> target pairs with compatible labels (complete cases, harmonised
   metadata: age, sex, coarse site). Fitted on the source only:
     image head   frozen ConvNeXt-T features, L2 logistic, penalty by grouped inner CV
     second stage logistic on [score] and [score + metadata], using source out-of-fold scores
   Target metrics (nothing tuned on the target): log-loss of image-only vs fused and their
   difference (added value), AUROC, sensitivity at 90% specificity (threshold fixed on the
   source; target-oracle threshold reported for description), Brier, calibration.
   Uncertainty: bootstrap over target grouping units.
   Reference: the target's own cross-validated added value with the same harmonised metadata.
3. Leave-one-source-out within HAM10000 (its `dataset` field).
"""
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import fit_tuned, oof_scores
from x_common import (DATASETS, VARS, available, ci, emb, folds_for, full_metrics, group_boot_mean, harmonised,
                      isic_ids, missing, oof_probs, per_image_ll, subset)

MEL = ["ham10000", "isic2019_nonham", "isic2020"]
MAL = ["pad_ufes", "hiba", "msk2020", "milk10k_derm", "milk10k_clin"]
MODALITY = {"ham10000": "dermoscopy", "isic2019_nonham": "dermoscopy", "isic2020": "dermoscopy",
            "pad_ufes": "smartphone clinical", "hiba": "dermoscopy", "msk2020": "mixed (ISIC)",
            "milk10k_derm": "dermoscopy", "milk10k_clin": "clinical close-up"}
B = 1000


def meta_design(h_train, h_test):
    """Harmonised metadata; imputation and categories fixed on the source."""
    med = h_train.age.median()
    cats = {c: sorted(h_train[c].unique()) for c in ("sex", "site")}

    def enc(h):
        cols = [h.age.fillna(med).to_numpy()[:, None] / 100, h.age.isna().to_numpy(float)[:, None]]
        for c, lv in cats.items():
            cols.append(np.column_stack([(h[c] == l).to_numpy(float) for l in lv]))
        return np.hstack(cols)
    return enc(h_train), enc(h_test)


def fit_lr(Z, y):
    sc = StandardScaler().fit(Z)
    return sc, LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(Z), y)


def sens_at_spec(y, p, thr):
    return float((p[y == 1] >= thr).mean())


def threshold_for_spec(y, p, spec=0.90):
    return float(np.quantile(p[y == 0], spec))


def transfer(src, tgt, s_src, s_tgt, rng, drop=None):
    keep = np.ones(len(s_tgt["y"]), bool) if drop is None else ~drop
    ys, gs = s_src["y"], s_src["g"]
    yt, gt = s_tgt["y"][keep], s_tgt["g"][keep]
    Xs, Xt = emb(s_src, "convnext_tiny"), emb(s_tgt, "convnext_tiny")[keep]
    folds = folds_for(s_src)
    oof, _ = oof_scores(Xs, ys, gs, folds)                       # source OOF scores for the second stage
    sc = StandardScaler().fit(Xs)
    head, _ = fit_tuned(sc.transform(Xs), ys, gs)               # tuned on the source only
    st = head.decision_function(sc.transform(Xt))
    Hs, Ht = meta_design(harmonised(src, s_src), harmonised(tgt, s_tgt).iloc[np.flatnonzero(keep)].reset_index(drop=True))
    sc0, m0 = fit_lr(oof[:, None], ys)
    sc1, m1 = fit_lr(np.column_stack([oof, Hs]), ys)
    p0 = m0.predict_proba(sc0.transform(st[:, None]))[:, 1]
    p1 = m1.predict_proba(sc1.transform(np.column_stack([st, Ht])))[:, 1]
    ps0 = m0.predict_proba(sc0.transform(oof[:, None]))[:, 1]
    ps1 = m1.predict_proba(sc1.transform(np.column_stack([oof, Hs])))[:, 1]
    d = per_image_ll(yt, p0) - per_image_ll(yt, p1)
    out = {"n_target": int(len(yt)), "prevalence_target": round(float(yt.mean()), 4),
           "added_value": round(float(d.mean()), 6), "added_value_ci95": ci(group_boot_mean(d, gt, rng, B)),
           "image_only": full_metrics(yt, p0), "fused": full_metrics(yt, p1)}
    for name, p, ps in (("image_only", p0, ps0), ("fused", p1, ps1)):
        out[name]["sens_at_90spec_source_threshold"] = sens_at_spec(yt, p, threshold_for_spec(ys, ps))
        out[name]["sens_at_90spec_target_oracle"] = sens_at_spec(yt, p, threshold_for_spec(yt, p))
    out["delta_auroc"] = round(out["fused"]["auroc"] - out["image_only"]["auroc"], 4)
    return out


def in_distribution(ds, s, rng):
    """Target's own CV added value with the same harmonised metadata (reference)."""
    folds = folds_for(s)
    score, _ = oof_scores(emb(s, "convnext_tiny"), s["y"], s["g"], folds)
    H = np.zeros((len(s["y"]), 0))
    blocks = []
    for tr, te in folds:
        a, b = meta_design(harmonised(ds, s).iloc[tr].reset_index(drop=True), harmonised(ds, s).iloc[te].reset_index(drop=True))
        blocks.append((te, b, tr, a))
    width = blocks[0][1].shape[1]
    H = np.zeros((len(s["y"]), max(b[1].shape[1] for b in blocks)))
    for te, b, _, _ in blocks:
        H[te, :b.shape[1]] = b
    p0, p1 = oof_probs(score[:, None], s["y"], folds), oof_probs(np.column_stack([score, H]), s["y"], folds)
    d = per_image_ll(s["y"], p0) - per_image_ll(s["y"], p1)
    return {"added_value": round(float(d.mean()), 6), "ci95": ci(group_boot_mean(d, s["g"], rng, B)),
            "image_auroc": round(float(roc_auc_score(s["y"], p0)), 4)}


def provenance(dss):
    rows, subs = [], {}
    for d in dss:
        s_all, s_cc = subset(d, False), subset(d, True)
        subs[d] = s_cc
        rows.append({"dataset": d, "modality": MODALITY[d], "label": "melanoma vs rest" if d in MEL else "malignant vs benign",
                     "n_all": int(len(s_all["y"])), "n_complete_case": int(len(s_cc["y"])),
                     "prevalence": round(float(s_all["y"].mean()), 4), "groups": int(len(np.unique(s_all["g"]))),
                     "metadata": ", ".join(n for n, _, _ in VARS[d]),
                     "max_missing_rate": round(max(s_all["miss_rate"].values()), 3),
                     "source_field": "yes" if d in ("ham10000", "isic2019_nonham") else "no"})
    ids = {d: isic_ids(d, subs[d]) for d in dss}
    overlaps = {f"{a}|{b}": len(ids[a] & ids[b]) for a in dss for b in dss if a < b and ids[a] and ids[b]}
    return pd.DataFrame(rows), overlaps, subs, ids


def main():
    rng = np.random.default_rng(33)
    dss = [d for d in DATASETS if available(d)]
    exp = Experiment("E33_shift", "H33 (exploratory): metadata added value under source/modality shift differs "
                     "from in-distribution added value (direction not pre-specified).",
                     {"datasets": dss}, {"features": "frozen ConvNeXt-T", "metadata": "harmonised age/sex/site",
                                         "bootstrap": B}, 0)
    prov, overlaps, subs, ids = provenance(dss)
    prov.to_csv(RESEARCH / "results/E33_provenance.csv", index=False)
    res = {"overlaps": overlaps, "in_distribution": {}, "transfer": {}, "ham_leave_one_source_out": {}}
    for d in dss:
        res["in_distribution"][d] = in_distribution(d, subs[d], rng)
        print("in-dist", d, res["in_distribution"][d], flush=True)
    for family in (MEL, MAL):
        fam = [d for d in family if d in dss]
        for a, b in permutations(fam, 2):
            col = {"ham10000": "image_id", "isic2020": "image_name", "isic2019_nonham": "image"}
            tdf = subs[b]["df"]
            key = next((c for c in ("image_id", "image_name", "image", "isic_id") if c in tdf), None)
            drop = tdf[key].astype(str).str.replace("_downsampled", "", regex=False).isin(ids[a]).to_numpy() if key and ids[a] else None
            r = transfer(a, b, subs[a], subs[b], rng, drop)
            r["removed_overlap"] = int(drop.sum()) if drop is not None else 0
            res["transfer"][f"{a}->{b}"] = r
            print("transfer", a, b, {k: r[k] for k in ("n_target", "added_value", "delta_auroc")}, flush=True)
    if "ham10000" in dss:
        s = subs["ham10000"]
        src = s["df"]["dataset"].to_numpy()
        for held in np.unique(src):
            tr_mask, te_mask = src != held, src == held
            if s["y"][te_mask].sum() < 10:
                continue
            sa = {**s, "idx": s["idx"][tr_mask], "df": s["df"][tr_mask].reset_index(drop=True), "y": s["y"][tr_mask], "g": s["g"][tr_mask]}
            sb = {**s, "idx": s["idx"][te_mask], "df": s["df"][te_mask].reset_index(drop=True), "y": s["y"][te_mask], "g": s["g"][te_mask]}
            r = transfer("ham10000", "ham10000", sa, sb, rng)
            res["ham_leave_one_source_out"][held] = r
            print("LOSO", held, {k: r[k] for k in ("n_target", "added_value", "delta_auroc")}, flush=True)
    exp.finish(res, interpretation="See research/iteration4/SUMMARY.md.")


if __name__ == "__main__":
    main()
