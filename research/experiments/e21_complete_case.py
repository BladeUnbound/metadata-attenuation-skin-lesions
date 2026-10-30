"""E21: complete-case re-analysis after the missingness-leakage audit (deviation from
the E13 pre-registration, justified by its own leakage rule).

Audit (all datasets; see EXPERIMENTS.md): in PAD-UFES-20 every image with missing gender/Fitzpatrick/history
(804) is benign and not biopsied, so missingness is a proxy for the excluded
`biopsed` field; "grew"/"changed" = UNK is 88% malignant. In HAM10000, unknown sex
(57 images) is never melanoma. Missingness indicators and "unknown" levels can
therefore carry the label.
Here only images with every metadata field recorded are kept; the E13 protocol is
otherwise identical (fold assignment 0): ladder added value, H1/H2 paired tests,
and the retained fraction rho.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import Experiment
from e10_ham_attenuation import lesion_bootstrap, oof_scores, second_stage
from e13_replication import B, boot_samples, oof_block
from e17_retained_fraction import prevalence_ll
from e18_foundation_rung import load_dataset

RUNGS = ["hsv", "resnet18", "resnet50", "convnext_tiny", "vit_b_16"]
SPEC = {
    "pad_ufes": {"fields": ["age", "gender", "fitspatrick", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"],
                 "num": ["age"], "cat": ["gender", "fitspatrick", "region", "itch", "grew", "hurt", "changed", "bleed", "elevation"]},
    "ham10000": {"fields": ["age", "sex", "localization"], "num": ["age"], "cat": ["sex", "localization"]},
    "isic2020": {"fields": ["age_approx", "sex", "anatom_site_general_challenge"], "num": ["age_approx"],
                 "cat": ["sex", "anatom_site_general_challenge"]},
    "isic2019_nonham": {"fields": ["age_approx", "sex", "anatom_site_general"], "num": ["age_approx"],
                        "cat": ["sex", "anatom_site_general"]},
    "hiba": {"fields": ["age_approx", "sex", "anatom_site_1", "fitzpatrick_skin_type"], "num": ["age_approx"],
             "cat": ["sex", "anatom_site_1", "fitzpatrick_skin_type"]},
    "msk2020": {"fields": ["age_approx", "sex", "anatom_site_1"], "num": ["age_approx"], "cat": ["sex", "anatom_site_1"]},
    "milk10k_derm": {"fields": ["age_approx", "sex", "anatom_site_1"], "num": ["age_approx"], "cat": ["sex", "anatom_site_1"]},
    "milk10k_clin": {"fields": ["age_approx", "sex", "anatom_site_1"], "num": ["age_approx"], "cat": ["sex", "anatom_site_1"]},
}


def missing(s):
    # isna() first: in this pandas version astype(str) keeps NaN as NaN, so a text test alone misses it
    return (s.isna() | s.astype(str).str.strip().str.lower().isin(["nan", "", "unknown", "unk", "none"])).to_numpy()


def run(name, rng):
    d = load_dataset(name)
    df, y, groups = d["df"], d["y"], d["groups"]
    keep = ~np.any([missing(df[c]) for c in SPEC[name]["fields"]], axis=0)
    idx = np.flatnonzero(keep)
    dfc, yc, gc = df.iloc[idx].reset_index(drop=True), y[idx], groups[idx]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(dfc, yc, gc))
    M = oof_block(dfc, SPEC[name]["num"], SPEC[name]["cat"], folds)
    _, l_meta = second_stage(np.zeros((len(yc), 1)), M, yc, folds)
    i_a = float((prevalence_ll(yc, folds) - l_meta).mean())
    out = {"n_complete": int(len(idx)), "n_dropped": int((~keep).sum()), "positive_rate": round(float(yc.mean()), 3),
           "I_A_nats": round(i_a, 5)}
    per = {}
    for k in RUNGS:
        score, _ = oof_scores(np.load(d["dir"] / f"emb_{k}.npy")[idx], yc, gc, folds)
        l0, l1 = second_stage(score[:, None], np.column_stack([score, M]), yc, folds)
        per[k] = l0 - l1
        out[k] = {"image_auroc": round(float(roc_auc_score(yc, score)), 4), "av": round(float(per[k].mean()), 5),
                  "ci95": [round(v, 5) for v in lesion_bootstrap(per[k], gc, rng)], "rho": round(float(per[k].mean() / i_a), 3)}
    for hyp, (hi, lo) in {"H1": ("hsv", "convnext_tiny"), "H2": ("resnet18", "convnext_tiny")}.items():
        diff = per[hi] - per[lo]
        bs = boot_samples(diff, gc, rng)
        out[hyp] = {"diff": round(float(diff.mean()), 5),
                    "ci95": [round(float(np.percentile(bs, q)), 5) for q in (2.5, 97.5)],
                    "p_one_sided": float((1 + np.sum(bs <= 0)) / (B + 1))}
    print(name, out, flush=True)
    return out


def main():
    rng = np.random.default_rng(21)
    tag = "" if len(sys.argv) < 2 else "_wave2"
    exp = Experiment("E21_complete_case" + tag, "After removing rows whose missingness can carry the label, attenuation (H1) "
                     "persists; PAD-UFES-20's floor shrinks.",
                     {"datasets": list(SPEC), "note": "deviation from pre-registration; leakage correction"},
                     {"folds": "assignment 0", "rungs": RUNGS}, 0)
    names = sys.argv[1:] or list(SPEC)
    exp.finish({n: run(n, rng) for n in names}, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
