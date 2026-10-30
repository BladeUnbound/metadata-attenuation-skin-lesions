# Maths notes, iteration 2

## Why auxiliary added value can rise or fall with image-model capacity

**Setup.** Outcome $y \in \{0,1\}$, image content $Z$, auxiliary variable $A$, and an image model that sees $V_k = Z + \sigma_k \varepsilon$, with noise $\sigma_k$ shrinking as capacity grows. Suppose

$$\operatorname{logit} P(y = 1 \mid Z, A) = \alpha + w^\top Z + \gamma A, \qquad Z = Z_0 + \delta A u$$

Here $\delta$ is the mediated path (A changes what the image shows) and $\gamma$ the direct path.

**Two opposing forces act on the added value of $A$ given the image score $S_k$.**

1. **Mediated path: falls with capacity.** The part of $A$'s effect carried by $\delta A u$ reaches $y$ only through $Z$. A weak image model recovers $Z$ poorly, so $A$ partly stands in for the missing image information. As $\sigma_k \to 0$, $S_k$ recovers $w^\top Z$, including the $\delta A u$ component, and the mediated added value goes to 0. Its size tracks how *badly* the image recovers $A$: in E09, added value fell exactly as the recoverability $R^2(A \mid V_k)$ rose (0.10 → 0.58).

2. **Direct path: rises with capacity (non-collapsibility).** Logistic regression coefficients are not collapsible: leaving out a predictor of $y$ that is independent of $A$ shrinks $A$'s coefficient toward 0, even without confounding. With the probit approximation, if the image score leaves residual signal variance $\tau_k^2$ unexplained, then

$$\gamma_k^{\text{marginal}} \approx \frac{\gamma}{\sqrt{1 + c\,\tau_k^2}}, \qquad c \approx \pi^2/8 \cdot (\text{logit/probit scale})^{-1}$$

As capacity grows, $\tau_k^2 \downarrow$, so $|\gamma_k^{\text{marginal}}|$ and the log-loss improvement from adding $A$ go **up**. E09's direct scenario showed this: added value 0.029 → 0.044.

**Consequence.** The added-value curve is the sum of a falling mediated term and a rising direct term. Its **slope sign** shows which dominates, and a non-zero plateau at high capacity bounds the direct contribution from below. A flat curve is ambiguous: it can come from no effect, or from two terms cancelling. That's why the recoverability curve is needed alongside it.

**Status.** The non-collapsibility of logistic regression is **known** (Gail, Wieand & Piantadosi 1984; Zeger, Liang & Albert 1988; *from memory, not re-checked this session*). The mediated-vs-direct decomposition of the added-value curve, read against model capacity, is **derived here and supported only in simulation (E09)**. Whether it's new needs a targeted search, especially the epidemiology literature on "incremental value of biomarkers" and on mediation with non-collapsible links.

## Exact mask-shift null via FFT (pipeline/saliency_overlap.py)

For saliency $s \ge 0$ and mask $m$ on a periodic grid, the mass inside the mask after shifting it by $k$ is

$$M(k) = \frac{\sum_x s(x)\, m(x - k)}{\sum_x s(x)} = \frac{(s \star m)(k)}{\lVert s \rVert_1}$$

This is a circular cross-correlation, so $\mathcal{F}^{-1}\!\left[\mathcal{F}(s)\,\overline{\mathcal{F}(m)}\right]$ gives $M(k)$ for **all** $V$ shifts in $O(V \log V)$. The p-value is $\#\{k : M(k) \ge M(0)\} / V$, which is exact over the shift group: no Monte Carlo error. Tested against an explicit loop in `test_exact_shift_null_matches_explicit_shifts`.

**Limitations.** Circular shifts wrap around image borders. Lesions centred in dermoscopy images mean many shifted masks straddle the border, so the null mixes lesion-like and border placements. The test is valid under the stated null (exchangeability of mask placement), but the null itself is a modelling choice. **Status:** known idea (spin tests and spatial nulls for brain maps); the FFT shortcut for saliency overlap is standard signal processing; applying it to evaluating XAI maps is potentially new (unverified).
