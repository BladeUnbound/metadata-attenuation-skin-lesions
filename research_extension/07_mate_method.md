# 07 — MATE (Metadata-Aware Task-preserving Erasure): candidates, checks and decision

*Phase 5, written 2026-10-30. Inputs:*
- *03 (novelty);*
- *04 (theory);*
- *05 (synthetic: SYN-3, SYN-3b, SYN-3c);*
- *REAL-5 (real-data nonlinear audit, §5).*

*The directive asked for a method only if it is justified, distinct from prior work and supported. This document records why the answer is no.*

## 1. The directive's minimax formulation, checked

The directive's form is:

min_φ max_ψ [ −λ_A 𝓛_A(g_ψ(T_φ(S)), A) + λ_Y 𝓛_Y(h_θ(T_φ(S)), Y) + λ_D D(T_φ(S), S) ].

- **Signs.** The inner max over ψ of −λ_A𝓛_A makes the adversary minimise its own loss, as intended. The outer min over φ then *maximises* the adversary's loss, and minimises task loss and distortion. Both directions are correct.
- **Missing variable.** The task head θ must be optimised jointly with φ: min over (φ, θ) of max over ψ. The written form leaves θ free.
- **Unbounded reward.** Against a lagging adversary, maximising 𝓛_A can be rewarded without limit: φ can push features to where the current g_ψ extrapolates badly. The trivial loss (Var(A) for MSE, H(A-bins) for cross-entropy) is the right ceiling. scripts/nonlin.py caps 𝓛_A at it.
- **Which task term.** With an image-only head h_θ(Z), the task term fights erasure whenever Y depends on A: Proposition 2 forces leakage ≥ I(A; P(Y|S)). In a metadata-aware method the head should see (Z, A), targeting the fused cost L_YA. That fused variant is the only MATE form that 04 §9 supports.

## 2. Candidates

| Candidate | Status in the literature | Can it raise η beyond LEACE? | Decision |
|---|---|---|---|
| Linear orthogonal projection (RLACE closed form, SAL) | Established | No. Same kernel as LEACE, so information-equivalent (Proposition 4, verified to 10⁻¹³) | Kept only as a check |
| LEACE-type oblique projection | Established | It is the reference | Principal intervention |
| Linear adversarial erasure (RLACE) | Established | For a scalar attribute it converges to the same kernel. Partial ranks give partial erasure; full rank equals LEACE | Not run on real data |
| Nonlinear MLP transformation (adversarial, no task term) | Established (Edwards & Storkey; Xie et al.; Elazar & Goldberg; CF-Net) | Synthetic: no (η ≤ LEACE). Removes only what its adversary measures | Audit only (REAL-5) |
| Residual transformation | Implementation detail of the above (identity-initialised residual MLP) | — | Used in nonlin.py |
| Conditional autoencoder (decoder sees A) | Established, 2026 (Simai, Ravfogel, Shaham); earlier Moyer et al. 2018 | In the ideal regime gives perfect recoverability; η then reflects only what the encoder cannot separate from A | Not reimplemented: no distinct contribution |
| Distribution matching (MMD/HSIC) | Established (LEOPARD; Obliviator) | Same frontier constraint as any attribute-free eraser (Proposition 3) | Not reimplemented |
| Task-preserving adversarial transformation | Established (Obliviator supervised; LAFTR; FaRM; CF-Net with age) | Image-task form conflicts with erasure (Proposition 2). The fused-task form is new only as a variant, and was tested | Tested synthetically (TP-img, TP-fused) |

## 3. What was implemented and tested

scripts/nonlin.py:
- an identity-initialised residual MLP T_φ;
- an MLP adversary, either MSE for signed age or cross-entropy over age deciles, with its loss capped at the trivial value;
- an optional diagnosis head on z (TP-img) or on [z, A] (TP-fused);
- an L2 distortion penalty;
- alternating Adam updates (3 adversary steps per transform step).

All variants are attribute-free at inference and were fitted on a separate split from the heads and probes (05).

**Synthetic results** (05 SYN-3/3b/3c; 4 regimes, λ_A ∈ {0.5, 2, 8, 32}, 3 seeds):
- No nonlinear variant exceeded LEACE's η in the linear regimes (L1: LEACE 0.995 vs ≤ 0.97; L2: 0.735 vs ≤ 0.70).
- TP-fused had no lower irreplaceable cost than TP-img or adversarial erasure at comparable leakage.
- Nonlinear erasers never reached exact linear guardedness (ridge R² 0.06–0.25, against LEACE's 0).
- They removed non-monotone age information only when the adversary could see it (decile adversary: |A| R² 0.25–0.46 against 0.75–0.88 with an MSE adversary).
- Permuted-label shams were not inert (C_Y 11–17 × 10⁻³ nats).

## 4. Why a method contribution is not available here

1. **The ceiling.** M_A ≤ C_Y (Identity 3). No transformation can make age more useful without an equal or larger image-only cost. A method can only raise η, the substitutable share of that cost.
2. **The frontier is fixed by the data.** For an attribute-free eraser, zero irreplaceable cost is possible only if the age-conditional posterior 𝔐(S) carries no age information (Proposition 3). Where the image's age-aligned signal also carries diagnosis-relevant variation not explained by recorded age (the entangled regime), *no* attribute-free transformation can keep that variation while removing age. LEACE already sits on that frontier in the linear-Gaussian case, which the synthetic results confirm.
3. **The attribute-conditional route is trivial or established.** If A may be used at inference, the residual S − E[S|A] (FWL; oracle LEACE; Metadata Normalization; Weber et al.; conditional autoencoders) gives η = 1 by construction.
4. **Prior work covers the remaining design space.** Task-preserving nonlinear erasure (Obliviator), distribution matching (LEOPARD) and conditional reconstruction (Simai et al. 2026) are published. A fused-task-head variant is a minor modification, and gave no measurable benefit.

**Decision.** MATE is not proposed as a method, and no confirmatory MATE comparison is run (registry: Phase 4/6 note).

The research contribution, if any, lies in Level 3 of the directive:
- the substitution identity and efficiency η;
- the sufficiency characterisation of when erasure and diagnostic preservation can coexist (image-only vs fused);
- the empirical measurement of η on real representations with matched controls (06, 08).

## 5. Real-data audit (REAL-5)

*Run 2026-11-01 22:28–22:42 IST; full table in 08 §4.* HAM10000 (development set), ConvNeXt-T. NAE with MSE and age-decile adversaries, λ_A ∈ {2, 8}, attribute-only (no diagnostic labels), strict cross-fitting, paired with the E50 LEACE and matched-random arms on identical cases.

| | LEACE | NAE (4 arms) | matched random |
|---|---|---|---|
| C_Y | 14.1 | 7.1–10.1 | 25.5 |
| L_YA (fused cost) | 2.9 | 5.9–9.3 | 23.4 |
| η | 0.79 [0.61, 1.01] | 0.06–0.17 (each η − η_LEACE CI excludes 0) | 0.08 |
| ridge age R² (t0 0.325) | −0.004 | 0.22–0.26 | 0.25 |
| HGB within-class age R² (t0 0.337) | 0.253 | 0.25–0.27 | 0.28 |

**The registered prediction holds on both counts.** The non-linear erasers do not exceed LEACE's η (they fall far below it), and they leave far more linear leakage.

**What this adds to the synthetic result.**
- On real features, attribute-only adversarial erasure behaves like random distortion, not like substitutable removal. Its cost is small, but almost none of it is recoverable from age.
- Its training adversaries were driven close to chance in some folds, yet fresh held-out probes still decode age.
- On the metadata-aware objective (fused cost L against leakage), LEACE dominates every NAE arm on linear leakage. On non-linear leakage it is comparable, at a third to a half of their L.

**Scope.** Task-preserving variants (TP-img, TP-fused) were not run on real data. SYN-3c gave no reason to expect a benefit, and any label-using variant must be fitted in a nested way (06 §6). The decision in §4 stands.

## 6. What would change this decision
- **Real data contradicting the synthetic results.** A real representation where a nonlinear or task-aware attribute-free eraser achieves materially higher η (lower L_YA) than LEACE at matched leakage across probe families, with nested cross-fitting.
- **A plausible mechanism.** Strongly non-Gaussian structure in which the age-conditional posterior is separable from age by a nonlinear map but not by a linear one. Neither SYN-3's regimes nor the real-data audit supplied such a case.
