# 04 — Mathematical foundations

*Phase 2, written 2026-10-30. Every identity below is checked numerically in notebooks/01_information_theory_verification.ipynb, and each section cites its notebook check.*

*Results fall into three kinds:*
- ***Identity:*** *an exact consequence of standard information theory, with nothing new claimed.*
- ***Proposition:*** *stated with explicit assumptions and a complete proof. Its novelty status is given in each case. All are elementary applications of known tools.*
- ***Example:*** *a controlled construction.*

*No theorem is claimed as a research contribution. Literature context: 01 §2.4–2.6, 03 §2–3.*

## 0. Setting and standing assumptions

- **Variables.**
  - X: image.
  - S = f(X) ∈ ℝ^d: frozen representation, with f measurable.
  - A: metadata (age: real-valued; site or sex: finite).
  - Y: diagnosis, taking values in a **finite** set 𝒴.
- **Transformation.** Z = T(S, ε), where ε is independent of (X, A, Y), and T is measurable. Deterministic transformations are the case where ε is constant.
- **Markov structure (M).** (Y, A) — S — Z. This holds by construction.
- **The image-to-representation link is not assumed.** (Y, A) — X — S holds because S = f(X), but nothing is assumed about how X relates to (Y, A).
- **Finiteness.** Because Y is finite, every mutual information involving Y is bounded by H(Y) < ∞. So all chain-rule manipulations below are legitimate (no ∞ − ∞) for arbitrary, possibly continuous, A, S and Z. Conditional MI is defined as I(Y; U | V) = E_{U,V}[ KL( P_{Y|U,V} ‖ P_{Y|V} ) ].

**Risks.** For an input V and a class 𝒢 of conditional distributions q(y | v):

R_Y^𝒢(V) := inf_{q ∈ 𝒢} E[ −log q(Y | V) ].

R_Y(V) means the unrestricted (Bayes) risk. The directive's quantities are, for a given 𝒢:

- C_Y = R_Y(Z) − R_Y(S): image-only diagnostic cost.
- Δ_A(V) = R_Y(V) − R_Y(V, A): incremental value of A given V.
- M_A = Δ_A(Z) − Δ_A(S): change in A's incremental value.
- **L_YA = R_Y(Z, A) − R_Y(S, A): fused (irreplaceable) cost.** This quantity is introduced here; it is not in the directive.
- R_A(Z) = inf_{g ∈ 𝒢_A} E ℓ_A(g(Z), A): attribute leakage risk. Larger means less leakage.

## 1. Log-loss risks are conditional entropies (Identity) — notebook §1

**Identity 1.** R_Y(V) = H(Y | V), attained by q = P_{Y|V}.

*Proof.* For any q, E[−log q(Y|V)] − H(Y|V) = E_V[ KL(P_{Y|V} ‖ q(·|V)) ] ≥ 0 (Gibbs), with equality iff q = P_{Y|V} a.s. ∎

**Corollary 1.**
- Δ_A(V) = H(Y|V) − H(Y|V,A) = I(Y; A | V).
- For S: R_Y(S) − R_Y(S, A) = I(Y; A | S).
- For Z: R_Y(Z) − R_Y(Z, A) = I(Y; A | Z).
- Hence **M_A = I(Y; A | Z) − I(Y; A | S)** (the directive's §2.5), under Bayes predictors and log-loss only.

For fitted models q̂, the expected held-out reduction is I(Y;A|V) + KL(P_{Y|V} ‖ q̂_0) − KL(P_{Y|V,A} ‖ q̂_1). This is already stated in the paper's Theory section. Empirical Δ̂ is therefore a CMI estimate only for well-specified, well-estimated models, and it can be negative.

## 2. The decomposition identity and its validity — notebook §2

**Identity 2 (the paper's Proposition 1).** If S = f(X) (or, more generally, (Y, A) — X — S), then

I(Y;A|S) = I(Y;A|X) + I(Y;X|S) − I(Y;X|S,A).

*Proof.* Expand I(Y; A, X | S) in two orders:

I(Y;A|S) + I(Y;X|S,A) = I(Y;X|S) + I(Y;A|X,S).

Under (Y, A) — X — S, conditioning on (X, S) is equivalent to conditioning on X, so I(Y;A|X,S) = I(Y;A|X). Rearrange. ∎

**Validity check.**
- The identity needs only the Markov condition and finiteness. It does **not** need injectivity of f, or anything about A's distribution.
- Without the Markov condition, for example if S were computed from (X, A) as in metadata-conditioned encoders, the term I(Y;A|X,S) need not equal I(Y;A|X), and the identity fails. Notebook §2 exhibits a violating case.

## 3. Metadata substitution is bounded by diagnostic cost

**Identity 3 (substitution identity).** Under (M), with Bayes predictors and log-loss:

- C_Y = I(Y; S | Z),
- L_YA = I(Y; S | Z, A),
- **M_A = C_Y − L_YA = I(Y;S|Z) − I(Y;S|Z,A).**

*Proof.*
- **C_Y.** Under (M), I(Y; Z | S) = 0. So I(Y;S) = I(Y; S, Z) = I(Y;Z) + I(Y;S|Z), and C_Y = H(Y|Z) − H(Y|S) = I(Y;S) − I(Y;Z) = I(Y;S|Z).
- **L_YA.** (M) implies I(Y; Z | S, A) = 0, because Z depends on (S, ε) with ε independent of (Y, A, S). The same argument conditional on A gives L_YA = I(Y;S|Z,A).
- **M_A.** By Corollary 1, M_A = [H(Y|Z) − H(Y|Z,A)] − [H(Y|S) − H(Y|S,A)] = [H(Y|Z) − H(Y|S)] − [H(Y|Z,A) − H(Y|S,A)] = C_Y − L_YA. ∎

This is Identity 2 with (X, S) replaced by (S, Z): the gain in A's value from coarsening S to Z is the image information lost, minus the part of it that A cannot replace.

**Corollaries** (immediate):
1. **Ceiling.** M_A ≤ C_Y, with equality iff L_YA = 0 ⇔ P(Y | S, A) = P(Y | Z, A) a.s. ("fused sufficiency").
2. **Floor.** M_A ≥ −Δ_A(S) = −I(Y;A|S), since Δ_A(Z) ≥ 0. M_A < 0 is possible: §6, Example E5 (synergy).
3. **No free substitution.** If the transformation is diagnostically sufficient on its own (C_Y = 0), then M_A = −L_YA ≤ 0. Separately observed A can never become more useful while image-only diagnosis is fully preserved.
4. **Substitution efficiency.** For C_Y > 0, define η := M_A / C_Y = 1 − L_YA / C_Y ∈ (−∞, 1]. It is the share of the image-only information loss that A replaces. This is the population counterpart of E50's restoration fraction (RF), which estimated 0.71–0.97 for LEACE and 0.03–0.10 for matched random erasure on the large sets.
5. **Co-information form.** M_A = I(Y; S; A | Z), the conditional interaction information (positive = redundancy). Positive M_A means the erased information was redundant with A; negative means it was synergistic with A.

**Answers to directive §2.5 (1)–(3).**
- I(Y;A|Z) can exceed I(Y;A|S) exactly when the erased information is redundant with A: positive conditional co-information.
- It can rise while *image-only* diagnostic utility is preserved: **never**, by Corollary 3.
- It can rise while *fused* utility (image + A) is preserved: **yes**, iff L_YA = 0, and then necessarily by exactly C_Y.

**Status.** An elementary consequence of the chain rule; the paper's Proposition 1 is the same algebra. The erasure papers reviewed (01) do not state the ceiling or η. It is presented as a framing identity, not a theorem.

## 4. Restricted predictor families — notebook §4

Empirical work uses restricted families: logistic heads, MLPs, boosted trees. Following the predictive V-information of Xu et al. (ICLR 2020), information "usable" by a family can **increase** under computation. So the ceiling needs a closure condition.

**Proposition 1 (ceiling for closed families).**
- *Assumption.* Let 𝒢 be such that for every q ∈ 𝒢_Z, the map (y, s) ↦ q(y | T(s)) lies in 𝒢_S, and likewise for (Z, A) and (S, A). This is a deterministic T with families closed under precomposition by T.
- *Claim.* Then R^𝒢(Z) ≥ R^𝒢(S) and R^𝒢(Z, A) ≥ R^𝒢(S, A). Hence C^𝒢 ≥ 0, L^𝒢 ≥ 0 and M^𝒢 = C^𝒢 − L^𝒢 ≤ C^𝒢.
- *Proof.* The infimum on the right ranges over a superset of the composed predictors on the left. ∎

**When the condition holds and when it fails.**
- It holds for **linear T with GLM heads** (LEACE, RLACE, random subspace projections, D1 scaling, D2 with independent noise), because affine maps compose to affine maps.
- It can fail for **nonlinear T with linear heads**: a nonlinear T can make diagnostic information linearly accessible, so C^lin < 0 and M^lin > C^lin are possible. Notebook §4 constructs this.
- **Consequence for MATE.** A nonlinear transformation must be evaluated with heads flexible enough to make its measured C_Y and M_A interpretable. A second check uses heads at least as rich as the transformation's own capacity.

**Dose types.**
- An *invertible* dose (E45/E50 "D1", c ↦ (1 − t)c for t < 1) changes no information: C_Y = M_A = 0 at population level. Any measured change under D1 at t < 1 is a restricted-family or finite-sample effect. This is why the registry made D2 the primary dose.
- The noise dose D2 satisfies (M), so Identity 3 and the ceiling apply.

## 5. Conditional independence statements are not interchangeable — notebook §5

Y ⊥ A | X, Y ⊥ A | S and Y ⊥ A | Z are logically independent in general:
- **Y ⊥ A | X but not Y ⊥ A | S.** A coarse representation loses image information that A proxies. By Identity 2, I(Y;A|S) = I(Y;X|S) − I(Y;X|S,A) > 0 when that information is redundant with A.
- **Y ⊥ A | S but not Y ⊥ A | Z.** Erasure removes A-redundant information (Identity 3).
- **Y ⊥ A | Z but not Y ⊥ A | S.** Removing synergistic information (Example E5).
- **No data-processing inequality for conditional MI in either direction.** Processing S into Z can raise or lower I(Y;A|·). The DPIs that do hold are I(Y;Z) ≤ I(Y;S), I(Y;Z,A) ≤ I(Y;S,A) and I(A;Z) ≤ I(A;S). These are exactly C_Y ≥ 0, L_YA ≥ 0 and "erasure cannot add attribute information".

## 6. Sufficiency, and when erasure and diagnostic preservation coexist — notebook §6

Let m(S) := P(Y = · | S) be the image-only posterior and m_A(S, a) := P(Y = · | S, A = a) the fused posterior. Write 𝔐(S) := m_A(S, ·) for the whole function a ↦ m_A(S, a).

- **Image-only sufficiency:** I(Y; S | Z) = 0 ⇔ P(Y|S) = P(Y|Z) a.s.
- **Fused sufficiency:** I(Y; S | Z, A) = 0 ⇔ P(Y|S,A) = P(Y|Z,A) a.s.
- **Directive §2.6.** P(Y|Z,A) = P(Y|Z) a.s. ⇔ I(Y;A|Z) = 0: Z makes A useless. That is the opposite of substitution and is not a sufficiency notion for diagnosis.

**Proposition 2 (minimum leakage under image-only preservation).**
- *Assumption.* T is deterministic.
- *Claim.* If C_Y = 0, then m(S) is σ(Z)-measurable (up to null sets), hence I(A; Z) ≥ I(A; m(S)). Equality is attained by Z = m(S). In particular, min{ I(A; Z) : C_Y(T) = 0 } = I(A; m(S)).
- *Proof.* C_Y = 0 gives P(Y|S) = P(Y|Z) a.s. The right-hand side is a measurable function of Z, so m(S) = φ(Z) a.s., and the data-processing inequality on A — Z — φ(Z) gives the bound. m(S) is itself sufficient: P(Y | m(S)) = m(S) by the tower property, so C_Y(m) = 0. ∎
- *Predictive form.* For every Bayes loss, R_A(Z) ≤ R_A(m(S)) for any image-only-sufficient Z. Age is at least as predictable from any diagnosis-preserving representation as it is from the diagnostic posterior alone.
- *Status.* The standard minimal-sufficiency argument (the posterior is minimal sufficient), applied to leakage.

**Proposition 3 (minimum leakage under fused preservation).**
- *Assumptions.* T is deterministic, and **positivity** holds: for P_S-almost every s, the conditional law of A given S = s has the same support as A.
- *Claim.* L_YA = 0 iff 𝔐(S) is σ(Z)-measurable. Hence min{ I(A;Z) : L_YA(T) = 0 } = I(A; 𝔐(S)), attained by Z = 𝔐(S).
- *Proof.*
  - (⇐) If 𝔐(S) = ψ(Z), then P(Y|S,A) = m_A(S, A) = ψ(Z)(A). This is a function of (Z, A), so it equals P(Y|Z,A) a.s. by the tower property.
  - (⇒) L_YA = 0 gives m_A(S, A) = g(Z, A) a.s. under P_{S,A}. By positivity, for a.e. s the equality m_A(s, a) = g(z(s), a) holds for P_A-a.e. a. So 𝔐(S) = g(Z, ·) as an element of L¹(P_A), and it is σ(Z)-measurable.
  - The minimisation follows as in Proposition 2. ∎
- *Without positivity* (A nearly determined by S, i.e. very strong age encoding), the "if" direction still holds. The minimum leakage can then be lower, because 𝔐(S) need only be recovered on the support of A given S.

**Corollary 5 (metadata enlarges the feasible erasure set).**
- m(S) = ∫ m_A(S, a) dP(a | S) depends on S through 𝔐(S) **and** through the age posterior P(A | S). So I(A; m(S)) can far exceed I(A; 𝔐(S)).
- **Example.** Let logit P(Y=1|S,A) = β·A + δ·S_Y with S = (S_A, S_Y), S_A = A + noise and S_Y ⊥ A.
  - 𝔐(S) depends on S_Y only, so I(A; 𝔐(S)) = 0: all age information can be erased at **zero fused cost**.
  - m(S) depends on S_A through E[A | S_A], so image-only preservation forces age leakage.
  - The same erasure then has C_Y = M_A > 0 and η = 1.

**Directive §2.5 (4).** When age is strongly related to diagnosis:
- Proposition 2 forces any image-only-preserving representation to leak at least the age information in the posterior.
- Zhao & Gordon (2019) bound the image-only error of age-independent representations.
- With A supplied, the binding constraint is Proposition 3 instead, which can be far weaker.

The observed HIBA failure (E50: image AUROC 0.93 → 0.65 under rank-1 age erasure) is the image-only side of this. Whether the *fused* side survives depends on L_YA, and that was not identifiable on HIBA because of the eraser overfitting (E50 §4).

**Status.** Propositions 2–3 are elementary applications of minimal sufficiency. Their side-information form (Proposition 3 and Corollary 5) was not found stated in the erasure literature; PEF and Simai et al. treat the task-agnostic H(S|Z,A) = 0. It is presented as a framing result.

## 7. Controlled examples (ground truth for Phase 3) — notebook §7

All examples use latent Gaussian factors and a logistic link for binary Y. For any linear transformation, every Bayes risk then reduces to one-dimensional Gauss–Hermite integrals: the logit given any linear-Gaussian V is Gaussian with constant variance. Ground truth is exact up to Monte Carlo error over V.

| Example | Construction | Prediction from §§3–6 |
|---|---|---|
| E1 redundant age | S_A = A + σ_u U, Y ← (A, S_Y); U carries no Y information | Erasing S_A: C_Y = M_A > 0, L_YA = 0, η = 1 |
| E2 entangled age | as E1, but Y also depends on U ("image-specific age": photo-ageing beyond chronological age) | η = 1 − I(Y;U-part \| Z, A)/C_Y < 1, falling with U's diagnostic weight |
| E3 irrelevant age | S_A = A + noise; Y ⊥ (A, S_A) given S_Y | C_Y = M_A = L_YA = 0; erasure only reduces leakage |
| E4 nonlinear encoding | S_A = h(A) with Cov(h(A), A) = 0 (e.g. h = \|A − EA\| or A²) | LEACE removes nothing (Σ_SA = 0): linear probes fail but MLP probes recover \|A\|; M_A ≈ 0 under flexible heads; restricted-family artefacts possible |
| E5 synergy | Y = 1{S_1 · A > 0} + noise | Erasing S_1: C_Y ≈ 0, L_YA > 0, M_A < 0 |
| E6 shared component | a diagnostic factor G correlated with A; S_Y = G + noise, S_A = A + noise; erase the Cov(S, A) direction (which mixes S_A and S_Y) | The erased direction carries G-information that A recovers only partly: η < 1, rising with corr(A, G) |
| E7 strong age–label dependence | Y ← A strongly; S_A = A + noise; S_Y weak | Image-only sufficiency forces leakage (Prop. 2); fused sufficiency does not (Prop. 3) |

E6 and E7 extend the directive's four regimes because §6 shows they are the cases that separate image-only from fused preservation.

**Verified values** (notebook §7; exact quadrature; ×10⁻³ nats; b = −0.5):

| Example | C_Y | M_A | L_YA | η |
|---|---|---|---|---|
| E1 redundant | 60.45 | 60.45 | 0.00 | 1.000 |
| E2 entangled (β_U = 0.8) | 107.53 | 65.32 | 42.21 | 0.607 |
| E3 irrelevant | 0.00 | 0.00 | 0.00 | undefined |
| E7 strong age–label (β_A = 2.5) | 199.04 | 199.04 | 0.00 | 1.000 |

- **E2:** η falls monotonically from 1.000 to 0.102 as β_U rises from 0 to 1.5.
- **E6:** η rises from 0.80 to 0.97 as corr(A, G) rises from 0.05 to 0.95.
- **E7:** strong age–label dependence makes the image-only cost large but leaves fused preservation intact (L_YA = 0) when the image's age trace carries nothing beyond A.
- **Oracle residual (positive control):** η = 1.000000 even in E2.
- **E4:** the linear probe gives R² = 0.0001 for A, while an MLP gives R² = 0.975 for |A|.
- **E5:** C_Y = 0, L_YA = 0.368 nats, so M_A = −0.368 nats.

## 7b. Linear erasers with the same kernel are information-equivalent — notebook §8

**Proposition 4.**
- *Claim.* Let P₁, P₂ be any two (oblique or orthogonal) projections on ℝ^d with the same kernel K. Then P₁ = P₁P₂ and P₂ = P₂P₁, so Z₁ = P₁S and Z₂ = P₂S are invertible linear functions of each other on their ranges. Hence:
  - all Bayes risks (C_Y, L_YA, M_A, leakage) coincide;
  - all risks under any predictor family closed under invertible linear maps coincide (GLMs, MLPs with a linear first layer).
- *Proof.* (I − P₂)S ∈ ker P₂ = ker P₁, so P₁(I − P₂) = 0, i.e. P₁ = P₁P₂. Symmetrically, P₂ = P₂P₁. ∎
- **Consequence.** Every exact linear eraser satisfies ker P ⊇ colsp(Σ_SA) (LEACE Thm 4.1). At minimal rank this kernel *equals* colsp(Σ_SA), so LEACE, the RLACE/OLS closed form and SAL at full rank are information-equivalent. They differ only in distortion: LEACE is minimal for every p.s.d. norm, and was minimal in 100% of 50 random 12-dimensional models.
- **Where they can still differ empirically:**
  1. penalised heads, because a ridge penalty is not invariant to the oblique map;
  2. RLACE runs that stop short of exact guardedness;
  3. categorical attributes where RLACE is run at a rank below rank(Σ_SA).

  A separate "linear adversarial erasure" arm on real data therefore adds nothing for age beyond LEACE at the same rank. Phase 4 runs RLACE only as a sanity check, and at partial ranks for categorical attributes.
- **Status.** Elementary linear algebra combined with LEACE Thm 4.1.
- **Verified** (notebook §8): the spread of R(Z) and R(Z,A) across LEACE, orthogonal and random-oblique erasers with the same kernel is 1.3 × 10⁻¹³ nats.

## 8. Deterministic versus stochastic transformations (directive §2.5 (8)–(9))

- **Deterministic T.** All identities are exact. Leakage satisfies I(A;Z) ≤ I(A;S).
- **Stochastic T with independent noise.** Again exact, since the noise preserves (M). Noise can lower leakage I(A;Z) further, but it can never raise M_A above C_Y.
- **Stochastic T whose noise depends on A or Y.** Examples are class-conditional noise, or using A at inference. This breaks (M):
  - With A at inference, Z = T(S, A) can make (Z, A) carry exactly the information of (S, A), so L_YA = 0 and M_A = C_Y, for example through the oracle residual S − E[S | A] or any A-conditional invertible map.
  - If T also uses Y (labels), the estimated quantities are no longer "properties of the image representation". Even with outer-test labels never used, training labels inside T need inner cross-fitting (see 00 §4 and Phase 5).
- **Status of the A-conditional regime.** It is the Frisch–Waugh–Lovell / oracle-LEACE regime. η = 1 holds identically in population and is a **positive control**, not a finding.

## 9. What the theory implies for the experiments

1. **Report η and L_YA, not only M_A.** At matched C_Y, comparing M_A is comparing η; that is exactly what E50's matched-cost design did.
2. **The directive's Regime 1 expectation is infeasible as worded.** "Leakage falls, diagnostic performance preserved, age more useful" violates Corollary 3 for image-only performance. The redundant regime should instead show image-only cost equal to the metadata gain (η = 1), with fused performance preserved (L_YA = 0). Phase 3 tests the corrected expectation.
3. **The meaningful method target.** Minimise L_YA (fused cost) at fixed leakage, using an attribute-free transformation. Equivalently, maximise η at matched C_Y.
   - The theoretical frontier is Proposition 3: the least leakage compatible with L_YA = 0 is I(A; 𝔐(S)).
   - LEACE is optimal only in the linear-Gaussian, additive-age case.
4. **Positive control.** The A-conditional oracle residual (η = 1 in population) measures how far finite-sample estimation pulls η below 1.
5. **Negative controls.**
   - Shams (A permuted): M_A ≈ C_Y ≈ 0.
   - Random subspaces at matched C_Y: η ≈ share of random directions' information that is A-redundant; small.
6. **Estimation caveats.**
   - Δ̂ is a difference of cross-entropies with no bound guarantee (McAllester & Stratos 2020).
   - The ceiling M_A ≤ C_Y can be violated by restricted heads (Proposition 1's condition) or by finite-sample noise. Measured η̂ > 1 is a diagnostic of those artefacts, not of super-substitution.
