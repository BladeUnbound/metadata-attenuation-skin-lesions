"""E20 (exploratory): which metadata variables carry the non-redundant floor?

For each dataset and each single variable (age, sex, site; for PAD-UFES-20 also each
symptom), the added value over the HSV score and over the ConvNeXt-T score
(fold assignment 0, same two-stage design, grouped bootstrap CI), and the
variable's retained fraction (ConvNeXt / HSV). This explains *why* the dataset
floors in Table VIII differ, without changing any pre-registered result.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import Experiment
from e10_ham_attenuation import lesion_bootstrap, oof_scores, second_stage
from e13_replication import oof_block
from e18_foundation_rung import load_dataset

VARS = {
    "ham10000": {"age": (["age"], []), "sex": ([], ["sex"]), "site": ([], ["localization"])},
    "isic2020": {"age": (["age_approx"], []), "sex": ([], ["sex"]), "site": ([], ["anatom_site_general_challenge"])},
    "isic2019_nonham": {"age": (["age_approx"], []), "sex": ([], ["sex"]), "site": ([], ["anatom_site_general"])},
    "pad_ufes": {"age": (["age"], []), "sex": ([], ["gender"]), "site": ([], ["region"]),
                 "fitzpatrick": ([], ["fitspatrick"]),
                 **{c: ([], [c]) for c in ["itch", "grew", "hurt", "changed", "bleed", "elevation"]}},
}


def run(name, rng):
    d = load_dataset(name)
    y, groups, df = d["y"], d["groups"], d["df"]
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    scores = {k: oof_scores(np.load(d["dir"] / f"emb_{k}.npy"), y, groups, folds)[0] for k in ["hsv", "convnext_tiny"]}
    out = {}
    for var, (num, cat) in VARS[name].items():
        M = oof_block(df, num, cat, folds)
        row = {}
        for k, sc in scores.items():
            l0, l1 = second_stage(sc[:, None], np.column_stack([sc, M]), y, folds)
            dd = l0 - l1
            row[k] = {"av": round(float(dd.mean()), 5), "ci95": [round(v, 5) for v in lesion_bootstrap(dd, groups, rng, B=1000)]}
        row["retained"] = round(row["convnext_tiny"]["av"] / row["hsv"]["av"], 3) if row["hsv"]["av"] > 0 else None
        out[var] = row
        print(name, var, row, flush=True)
    return out


def main():
    rng = np.random.default_rng(20)
    exp = Experiment("E20_per_variable_floor", "Exploratory: identify which variables carry the non-redundant floor.",
                     {"datasets": list(VARS)}, {"rungs": ["hsv", "convnext_tiny"], "folds": "assignment 0"}, 0)
    exp.finish({n: run(n, rng) for n in VARS}, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
