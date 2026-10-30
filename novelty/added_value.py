"""Statistics for the paper's central claim: does genetic risk add predictive
value to MRI for MCI-to-Alzheimer's progression? (DERIVATION.md, part B)

Recommended order, following Demler, Pencina & D'Agostino (Stat Med 2012):
  1. lrt_added_predictors   - test whether APOE / PRS are associated with
                              progression given the MRI score (primary test).
  2. delong_paired          - only if step 1 is significant, report the AUC
                              gain with a DeLong confidence interval. Do not
                              use DeLong's p-value for nested models: it is
                              invalid there (conservative; see tests).
  3. interaction_test + holm - subgroup claims (e.g. APOE e4 carriers) need a
                              formal interaction test with multiplicity control.

The MRI score must come from subjects the network was not trained on: an
external cohort, or out-of-fold predictions from cross-validation.
"""
import numpy as np
import statsmodels.api as sm
from scipy import stats


def _midrank(x):
    """Ranks starting at 1, with tied values sharing their average rank."""
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks_sorted = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks_sorted[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    ranks = np.empty(n)
    ranks[order] = ranks_sorted
    return ranks


def delong_paired(y, score_a, score_b, alpha=0.05):
    """DeLong test for two correlated AUCs on the same subjects.

    Uses the structural-component form of the U-statistic variance (DeLong et
    al. 1988), computed with midranks as in Sun & Xu (2014).
    Returns a dict with both AUCs, their difference, its standard error,
    a (1 - alpha) confidence interval, z and the two-sided p-value.
    """
    y = np.asarray(y).astype(bool)
    pos, neg = y, ~y
    m, n = pos.sum(), neg.sum()
    aucs, v10, v01 = [], [], []
    for s in (np.asarray(score_a, float), np.asarray(score_b, float)):
        tx, ty, tz = _midrank(s[pos]), _midrank(s[neg]), _midrank(s)
        aucs.append((tz[pos].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[pos] - tx) / n)       # per-positive placement values
        v01.append(1 - (tz[neg] - ty) / m)   # per-negative placement values
    cov = np.cov(np.array(v10)) / m + np.cov(np.array(v01)) / n
    diff = aucs[0] - aucs[1]
    se = np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = diff / se
    half = stats.norm.ppf(1 - alpha / 2) * se
    return {
        "auc_a": aucs[0], "auc_b": aucs[1], "diff": diff, "se": se,
        "ci": (diff - half, diff + half), "z": z, "p": 2 * stats.norm.sf(abs(z)),
    }


def _logit(y, X):
    return sm.Logit(y, sm.add_constant(X, has_constant="add")).fit(disp=0)


def lrt_added_predictors(y, base, added):
    """Likelihood-ratio test of H0: the added predictors have zero coefficients.

    base: (n, p) e.g. the MRI score; added: (n, k) e.g. APOE e4 count and PRS.
    Under H0, 2 * (llf_full - llf_base) is asymptotically chi-square with k
    degrees of freedom (Wilks 1938).
    """
    base = np.asarray(base, float).reshape(len(y), -1)
    added = np.asarray(added, float).reshape(len(y), -1)
    m0 = _logit(y, base)
    m1 = _logit(y, np.column_stack([base, added]))
    stat = 2 * (m1.llf - m0.llf)
    k = added.shape[1]
    return {
        "stat": stat, "df": k, "p": stats.chi2.sf(stat, k),
        "pred_base": m0.predict(), "pred_full": m1.predict(),
    }


def interaction_test(y, score, group):
    """Wald test of the score x group interaction in a logistic model.

    Tests whether the score predicts differently in two subgroups (e.g. APOE
    e4 carriers vs non-carriers), which is the correct test for a subgroup
    claim. Comparing two separate subgroup AUCs is not.
    """
    score = np.asarray(score, float)
    group = np.asarray(group, float)
    fit = _logit(y, np.column_stack([score, group, score * group]))
    return {"coef": fit.params[3], "se": fit.bse[3], "p": fit.pvalues[3]}


def holm(pvalues):
    """Holm step-down adjusted p-values (controls the family-wise error rate)."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    k = len(p)
    adjusted_sorted = np.maximum.accumulate((k - np.arange(k)) * p[order])
    adjusted = np.empty(k)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
