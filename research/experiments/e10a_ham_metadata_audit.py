"""E10a: HAM10000 metadata audit before any image model is fitted.

Checks: images per lesion (why grouping matters), class balance, missing values,
label leakage through dx_type and the image-source column, and a metadata-only
baseline with lesion-grouped out-of-fold predictions.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict

sys.path.insert(0, str(Path(__file__).parent))
from common import RESEARCH, Experiment

META = RESEARCH / "data/ham10000/raw/HAM10000_metadata.csv"
SEED = 0


def design(df, cols):
    parts = []
    for c in cols:
        if c == "age":
            age = df["age"]
            parts += [age.fillna(age.median()).to_frame("age") / 100, age.isna().astype(float).to_frame("age_na")]
        else:
            parts.append(pd.get_dummies(df[c].fillna("missing"), prefix=c, dtype=float))
    return pd.concat(parts, axis=1).to_numpy()


def oof(X, y, groups):
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(LogisticRegression(C=1.0, max_iter=3000), X, y, cv=cv, groups=groups,
                             method="predict_proba")[:, 1]


def main():
    df = pd.read_csv(META)
    y = (df["dx"] == "mel").astype(int).to_numpy()
    g = df["lesion_id"].to_numpy()
    exp = Experiment("E10a_ham_metadata_audit",
                     "dx_type leaks the label; images per lesion > 1 makes image-level splits leak; "
                     "clinical metadata alone carries some melanoma signal.",
                     {"source": "HAM10000 metadata (Dataverse doi:10.7910/DVN/DBW86T, CC BY-NC 4.0)"},
                     {"task": "melanoma vs rest", "cv": "5-fold StratifiedGroupKFold by lesion_id"}, SEED)
    per_lesion = df.groupby("lesion_id").size()
    prevalence = y.mean()
    base_ll = log_loss(y, np.full(len(y), prevalence))
    res = {
        "n_images": int(len(df)), "n_lesions": int(per_lesion.size),
        "images_per_lesion": per_lesion.value_counts().sort_index().to_dict(),
        "frac_images_in_multi_image_lesions": round(float((df["lesion_id"].map(per_lesion) > 1).mean()), 3),
        "class_counts": df["dx"].value_counts().to_dict(),
        "melanoma_prevalence": round(float(prevalence), 4),
        "missing": {c: int(df[c].isna().sum() + (df[c] == "unknown").sum()) for c in ["age", "sex", "localization"]},
        "dx_type_by_dx": pd.crosstab(df["dx"], df["dx_type"]).to_dict(),
        "source_by_dx": pd.crosstab(df["dx"], df["dataset"]).to_dict(),
        "prevalence_logloss": round(float(base_ll), 4),
    }
    for name, cols in {"clinical_meta": ["age", "sex", "localization"], "dx_type_only": ["dx_type"],
                       "source_only": ["dataset"]}.items():
        p = oof(design(df, cols), y, g)
        res[name] = {"oof_logloss": round(float(log_loss(y, p)), 4), "oof_auroc": round(float(roc_auc_score(y, p)), 4)}
        print(name, res[name], flush=True)
    exp.finish(res, interpretation="See research/EXPERIMENTS.md.")


if __name__ == "__main__":
    main()
