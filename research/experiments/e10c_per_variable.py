"""E10c: per-variable attenuation. Does each metadata variable lose added value
in step with how well the image representation recovers that variable?

Prediction (written before running): age and site attenuate strongly (their
recoverability rises a lot with capacity); sex attenuates least (weakly
visible in dermoscopy). Seed-0 lesion-grouped folds from E10; same two-stage design.
Recoverability per variable comes from E10's seed0_extra results.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment
from e10_ham_attenuation import DATA, RUNGS, lesion_bootstrap, oof_scores, second_stage


def variable_block(df, name, train_idx):
    if name == "age":
        age = df["age"]
        med = age.iloc[train_idx].median()
        return np.column_stack([age.fillna(med).to_numpy() / 100, age.isna().to_numpy(float)])
    return pd.get_dummies(df[name].fillna("unknown"), dtype=float).to_numpy()


def main():
    df = pd.read_csv(DATA / "raw/HAM10000_metadata.csv")
    y = (df["dx"] == "mel").to_numpy(int)
    groups = df["lesion_id"].to_numpy()
    rng = np.random.default_rng(2)
    folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(df, y, groups))
    e10 = json.loads((RESEARCH / "results/E10_ham_attenuation.json").read_text())["results"]["seed0_extra"]
    exp = Experiment("E10c_per_variable", "Age and site attenuate strongly with capacity; sex least.",
                     {"source": "HAM10000"}, {"folds": "E10 seed 0", "variables": ["age", "sex", "localization"]}, 0)
    blocks = {}
    for v in ["age", "sex", "localization"]:
        B = np.zeros((len(df), variable_block(df, v, folds[0][0]).shape[1]))
        for tr, te in folds:
            B[te] = variable_block(df, v, tr)[te]
        blocks[v] = B
    res = {}
    for rung in RUNGS:
        score, _ = oof_scores(np.load(DATA / f"emb_{rung}.npy"), y, groups, folds)
        rec = e10[rung]["recoverability"]
        res[rung] = {}
        for v, B in blocks.items():
            a, b = second_stage(score[:, None], np.column_stack([score, B]), y, folds)
            d = a - b
            res[rung][v] = {"added_value": round(float(d.mean()), 5),
                            "ci95": [round(x, 5) for x in lesion_bootstrap(d, groups, rng, B=1000)]}
        res[rung]["recoverability"] = {"age_r2": rec["age_r2"], "sex_auroc": rec["sex_auroc"],
                                       "site_auroc_top5": rec["site_auroc_top5_mean"]}
        print(rung, {v: res[rung][v]["added_value"] for v in blocks}, res[rung]["recoverability"], flush=True)
    for v in blocks:
        first, last = res[RUNGS[0]][v]["added_value"], np.mean([res[r][v]["added_value"] for r in RUNGS[-2:]])
        res[f"retained_fraction_{v}"] = round(float(last / first), 3) if first > 0 else None
    print({k: v for k, v in res.items() if k.startswith("retained")}, flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
