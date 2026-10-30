"""Cohort interface: the single entry point for real (ADNI / AIBL / OASIS-3) data later.

A cohort is a CSV manifest with one row per scan (visit):

  column              meaning
  subject_id          pseudonymous patient ID; never a name or anything identifying
  visit               visit code (e.g. bl, m06)
  scan_path           path to the preprocessed 3D T1 volume (NIfTI), may be empty for tabular-only use
  site                acquisition site / scanner (for robustness analyses)
  age, sex            baseline covariates
  apoe_e4             number of APOE e4 alleles (0, 1, 2), may be missing
  prs                 polygenic risk score (standardised), may be missing
  progressed          1 if MCI -> AD within the prespecified horizon, 0 if stable through it
  followup_months     follow-up length, used to exclude stable patients with too-short follow-up

Analyses use one baseline scan per subject. Folds are assigned per subject so that
no patient, and so no repeat scan or augmented copy, appears on both sides of a split.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REQUIRED = ["subject_id", "visit", "scan_path", "site", "age", "sex", "apoe_e4", "prs",
            "progressed", "followup_months"]


@dataclass
class Cohort:
    table: pd.DataFrame

    @classmethod
    def from_csv(cls, path, horizon_months=36, baseline_visit="bl"):
        df = pd.read_csv(path)
        missing = set(REQUIRED) - set(df.columns)
        if missing:
            raise ValueError(f"manifest is missing columns: {sorted(missing)}")
        return cls.from_frame(df, horizon_months, baseline_visit)

    @classmethod
    def from_frame(cls, df, horizon_months=36, baseline_visit="bl"):
        base = df[df["visit"] == baseline_visit].copy()
        dupes = base["subject_id"].duplicated()
        if dupes.any():
            raise ValueError(f"{dupes.sum()} subjects have more than one baseline scan")
        # Stable patients need follow-up through the whole horizon to count as stable
        eligible = (base["progressed"] == 1) | (base["followup_months"] >= horizon_months)
        return cls(base[eligible].reset_index(drop=True))

    def folds(self, n_folds=5, seed=0):
        """Patient-level stratified folds; returns an array of fold numbers per row."""
        fold = np.empty(len(self.table), int)
        splitter = StratifiedGroupKFold(n_folds, shuffle=True, random_state=seed)
        for k, (_, te) in enumerate(splitter.split(self.table, self.table["progressed"],
                                                   self.table["subject_id"])):
            fold[te] = k
        check_no_subject_overlap(self.table["subject_id"].to_numpy(), fold)
        return fold


def check_no_subject_overlap(subject_ids, fold):
    """Raise if any subject appears in more than one fold."""
    per_subject = pd.Series(fold).groupby(np.asarray(subject_ids)).nunique()
    leaked = per_subject[per_subject > 1]
    if len(leaked):
        raise AssertionError(f"{len(leaked)} subjects appear in more than one fold")


def out_of_fold_scores(features, y, fold, make_model):
    """Out-of-fold scores: each row is scored by a model that never saw its fold.

    make_model() must return a fresh, unfitted estimator. Any tuning must happen
    inside the estimator (e.g. a CV-wrapped model) so it never sees the test fold.
    """
    scores = np.empty(len(y))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        m = make_model().fit(features[tr], y[tr])
        scores[te] = m.decision_function(features[te]) if hasattr(m, "decision_function") \
            else m.predict_proba(features[te])[:, 1]
    return scores
