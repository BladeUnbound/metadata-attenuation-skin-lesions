"""Integrity tests for the cohort interface, on a synthetic manifest with repeat visits."""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from cohort import Cohort, check_no_subject_overlap, out_of_fold_scores


def synthetic_manifest(rng, n_subjects=300):
    rows = []
    for s in range(n_subjects):
        progressed = int(rng.random() < 0.35)
        follow = rng.choice([12, 24, 36, 48])
        for visit in ["bl", "m06", "m12"]:
            rows.append({"subject_id": f"S{s:04d}", "visit": visit, "scan_path": "", "site": s % 5,
                         "age": 70 + rng.normal(), "sex": s % 2, "apoe_e4": rng.binomial(2, 0.25),
                         "prs": rng.normal(), "progressed": progressed, "followup_months": follow})
    return pd.DataFrame(rows)


def test_one_baseline_per_subject_and_followup_rule():
    df = synthetic_manifest(np.random.default_rng(0))
    c = Cohort.from_frame(df, horizon_months=36)
    assert c.table["subject_id"].is_unique
    stable_short = (c.table["progressed"] == 0) & (c.table["followup_months"] < 36)
    assert not stable_short.any()


def test_duplicate_baselines_are_rejected():
    df = synthetic_manifest(np.random.default_rng(1))
    df = pd.concat([df, df[df["visit"] == "bl"].head(3)])
    with pytest.raises(ValueError):
        Cohort.from_frame(df)


def test_folds_are_patient_level():
    c = Cohort.from_frame(synthetic_manifest(np.random.default_rng(2)))
    fold = c.folds(5)
    assert set(np.unique(fold)) == set(range(5))
    check_no_subject_overlap(c.table["subject_id"].to_numpy(), fold)


def test_overlap_check_catches_leakage():
    ids = np.array(["A", "A", "B", "C"])
    with pytest.raises(AssertionError):
        check_no_subject_overlap(ids, np.array([0, 1, 0, 1]))


def test_out_of_fold_scores_never_see_their_own_fold():
    """Labels that are pure noise: in-sample fitting of a flexible model scores AUC near 1,
    out-of-fold scoring must stay near 0.5."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(3)
    X = rng.normal(size=(300, 200))
    y = (rng.random(300) < 0.4).astype(int)
    fold = np.arange(300) % 5
    make = lambda: LogisticRegression(C=100, max_iter=5000)
    oof = out_of_fold_scores(X, y, fold, make)
    in_sample = make().fit(X, y).decision_function(X)
    assert roc_auc_score(y, in_sample) > 0.95
    assert abs(roc_auc_score(y, oof) - 0.5) < 0.1
