# CV EL: derivations, checks and where they stop

Each step has a status:

- **PROVED + CHECKED**: a proof is given here, and a test in this folder checks it numerically.
- **KNOWN**: the result is already published; the citation is given. Re-deriving it here is not a contribution.
- **STOP**: the point where going further would mean claiming something I cannot prove or verify.

Run all checks with `.venv/bin/python -m pytest -q` (26 tests, about 20 s).

---

## 0. Can novelty be proven mathematically?

No. Novelty is a claim about what has been published, so it can only be supported by a literature search, never proved. Mathematics can do two related things for this project:

1. **Prove the paper's central claim is tested correctly.** That claim is "genetic risk adds predictive value to MRI". It needs a test whose false-positive rate is provably controlled. That is Part B.
2. **Prove properties of a method**, such as what its gradient equals or how it scales. That is Part A. Whether a method is *new* is still a literature question, answered in A7.

---

## Part A: backpropagation for the spiking network

### A1. Model (definition)

For one layer of neurons, at time steps $t = 1, \dots, T$:

$$u_t = \beta\, v_{t-1} + I_t, \qquad s_t = H(u_t - \theta), \qquad v_t = u_t - \theta\, s_t$$

Here $I_t$ is the input current, $u_t$ the membrane before spiking, $s_t \in \{0,1\}$ the spike, and $v_t$ the membrane after a reset by subtraction ("soft reset"). $\beta = 1$ gives the integrate-and-fire (IF) neuron. For this project the MRI volume is static, so the first layer receives the same $I_t = Wx + b$ at every step.

$H$ has derivative zero almost everywhere, so training replaces $H'(u - \theta)$ with a surrogate $h(u - \theta)$. The code uses the rectangle $h(x) = \tfrac{1}{w}\mathbf{1}[|x| < w/2]$, with $w = \theta$ unless stated.

### A2. Backpropagation through time (BPTT): PROVED + CHECKED, KNOWN

Let $g_t = \partial L / \partial s_t$ arrive from downstream layers, and let $\varepsilon_t = \mathrm{d}L/\mathrm{d}u_t$.

- $u_t$ affects $L$ through $s_t$ and through $v_t$.
- $v_t$ affects $L$ only through $u_{t+1} = \beta v_t + I_{t+1}$, so $\partial L/\partial v_t = \beta\,\varepsilon_{t+1}$, with $\varepsilon_{T+1} = 0$.
- $\mathrm{d}v_t/\mathrm{d}u_t = 1 - \theta h_t$ when the reset stays in the graph, or $1$ when it is detached.

Therefore:

$$\varepsilon_t = h_t\, g_t + \beta\, \varepsilon_{t+1}\,(1 - \theta h_t) \quad \text{(reset kept)}, \qquad \varepsilon_t = h_t\, g_t + \beta\, \varepsilon_{t+1} \quad \text{(reset detached)}$$

Since $\partial u_t / \partial I_t = 1$, we get $\partial L/\partial I_t = \varepsilon_t$. Weight gradients follow as $\partial L/\partial W = \sum_t \varepsilon_t\, s_t^{\text{in}\top}$, and $\partial L/\partial v_0 = \beta\,\varepsilon_1$.

**Check:** `test_step2_manual_bptt_matches_autograd` compares this recursion with PyTorch autograd for $\beta \in \{1, 0.8\}$, with the reset kept and detached. They agree to $10^{-12}$.

**Known:** this is standard surrogate-gradient BPTT (Neftci, Mostafa & Zenke, *IEEE Signal Processing Magazine*, 2019). It is included because every later step builds on it.

### A3. Exact rate identity: PROVED + CHECKED, KNOWN

For $\beta = 1$ and **any** input sequence, sum $v_t - v_{t-1} = I_t - \theta s_t$ over $t$. The left side telescopes:

$$\theta N_T = v_0 + \sum_{t=1}^{T} I_t - v_T, \qquad N_T = \sum_t s_t$$

So the firing rate is $\frac{N_T}{T} = \frac{\bar I}{\theta} + \frac{v_0 - v_T}{T\theta}$: the average input divided by the threshold, plus a residual that shrinks like $1/T$ whenever $v_T$ stays bounded.

**Check:** `test_step3_rate_identity_holds_for_any_input` uses 5,000 neurons with large, negative, time-varying inputs. Error is below $10^{-9}$.

**Known:** this identity underlies ANN-to-SNN conversion error analysis (Bu et al., ICLR 2022) and "Differentiation on Spike Representation" (DSR; Meng et al., CVPR 2022).

### A4. Closed form for static input: PROVED + CHECKED, KNOWN

**Claim.** If $\beta = 1$, $I_t = I$ is constant and $v_0 \in [0, \theta)$, then

$$N_T = \operatorname{clip}\!\left(\left\lfloor \frac{v_0 + T I}{\theta} \right\rfloor,\, 0,\, T\right)$$

**Proof.**
- **Case $0 \le I < \theta$.** By induction, $v_t \in [0, \theta)$ for all $t$. If $v_{t-1} \in [0,\theta)$, then $u_t = v_{t-1} + I \in [0, 2\theta)$. The reset subtracts $\theta$ exactly when $u_t \ge \theta$, so $v_t \in [0, \theta)$. By A3, $N_T = (v_0 + TI - v_T)/\theta$ with $v_T \in [0,\theta)$. Since $N_T$ is an integer in $\left((v_0+TI)/\theta - 1,\ (v_0+TI)/\theta\right]$, it equals the floor.
- **Case $I \ge \theta$.** $u_t \ge v_{t-1} + \theta \ge \theta$ whenever $v_{t-1} \ge 0$, and then $v_t = v_{t-1} + I - \theta \ge v_{t-1} \ge 0$. So the neuron spikes every step and $N_T = T$.
- **Case $I < 0$.** $v$ only decreases from $v_0 < \theta$, so the neuron never spikes and $N_T = 0$.

The clip covers both edge cases. ∎

With $v_0 = \theta/2$, this is $\operatorname{clip}(\lfloor TI/\theta + \tfrac12 \rfloor, 0, T)$.

**Check:** `test_step4_static_if_count_closed_form` uses 80,000 neurons across $T \in \{1, 4, 16, 64\}$, with inputs covering all three cases. Counts match exactly. The test skips inputs within $10^{-9}$ of a floor boundary, where floating-point rounding decides; that is fewer than 0.1% of cases.

**Known:** this is the clip-floor relation behind the QCFS activation of Bu et al. (ICLR 2022). I have not re-checked their exact initialisation of $v_0$.

### A5. What surrogate BPTT computes for a static input: PROVED + CHECKED, novelty not claimed

**Setting:** a single IF neuron ($\beta = 1$) with constant input $I$, loss $L = N_T$ (so $g_t = 1$), and the rectangular surrogate with width $w = \theta$. Then $\theta h_t \in \{0, 1\}$ exactly, which makes the recursion in A2 solvable in closed form.

**Reset kept.** If step $t$ is in the surrogate window, $\theta h_t = 1$, so the temporal term vanishes and $\varepsilon_t = 1/\theta$. Otherwise $\varepsilon_t = \varepsilon_{t+1}$. Hence $\varepsilon_t = 1/\theta$ if some step $k \ge t$ is in the window, else $0$, and

$$\frac{\mathrm{d}N_T}{\mathrm{d}I}\bigg|_{\text{surrogate}} = \sum_t \varepsilon_t = \frac{t^\ast}{\theta}, \qquad t^\ast = \text{last in-window step (0 if none)}$$

So the gradient is never more than $T/\theta$. When the last step is in the window, it equals exactly $T/\theta$, the slope of the linear envelope $TI/\theta$ of the staircase in A4. In other words, surrogate BPTT recovers the rate-domain gradient that DSR uses.

**Reset detached.** $\varepsilon_t = h_t + \varepsilon_{t+1}$, so

$$\frac{\mathrm{d}N_T}{\mathrm{d}I}\bigg|_{\text{surrogate}} = \frac{1}{\theta}\sum_{k \text{ in window}} k$$

This grows like $T^2$.

**Check:** `test_step5_surrogate_count_gradient_closed_form` matches autograd to $10^{-9}$ for $T \in \{4, 16, 64\}$, both reset modes. Measured over 2,000 neurons with $I \in [0.05, 0.95]\theta$:

| $T$ | reset kept, max | reset detached, max | reset detached, median |
|---|---|---|---|
| 16 | 16 (= $T/\theta$) | 136 | 102 |
| 64 | 64 (= $T/\theta$) | 2,080 | 1,597 |

**Practical consequence for this project:** with static MRI input, keep the reset in the graph (or train in the rate domain). Otherwise the first layer's gradient scale grows roughly with $T^2$, and the learning rate would have to be retuned for every choice of $T$.

**Novelty not claimed.** Closely related results exist. OTTT (Xiao et al., NeurIPS 2022) shows its gradients approximate spike-representation training. Gygax & Zenke (arXiv 2404.14964, 2024) show that surrogate gradients equal the gradient of the expected output for single neurons. I have not found this exact closed form stated in those papers, but I have not read them in full, so I treat it as likely known.

### A6. Gene-conditioned thresholds are input gains: PROVED + CHECKED

The tempting "radiogenomic SNN" idea: let a patient's APOE or polygenic risk score set each channel's threshold, $\theta \to g(\text{genes})\,\theta$ with $g > 0$.

**Claim.** For any $\beta$, a neuron with threshold $g\theta$, input $I_t$ and initial membrane $g v_0$ emits exactly the same spikes as a neuron with threshold $\theta$, input $I_t/g$ and initial membrane $v_0$.

**Proof.** Divide every equation in A1 by $g$. With $u' = u/g$, $v' = v/g$ and $I' = I/g$, the equations become $u'_t = \beta v'_{t-1} + I'_t$, $s_t = H(u'_t - \theta)$ (because $H(gx) = H(x)$ for $g > 0$), and $v'_t = u'_t - \theta s_t$. The spikes $s_t$ are unchanged. ∎

So gene-conditioned thresholds are exactly multiplicative conditioning of the layer's pre-activation, $I_t \mapsto I_t / g$: the scale half of FiLM (Perez et al., AAAI 2018). They add no expressive power beyond that.

**Check:** `test_step6_threshold_scaling_equals_input_gain` uses 4,000 neurons, 20 steps, $\beta \in \{1, 0.9\}$. Spikes are identical when $g$ is a power of 2, so the arithmetic is exact. Fewer than 0.01% differ for arbitrary $g$, all at rounding ties.

### A7. STOP: no new backpropagation method

I tried four routes to a new training method for this project. Each one reduced to published work:

| Candidate | What it reduces to |
|---|---|
| Backpropagate through spike counts, using A3/A4 | DSR (Meng et al., CVPR 2022) |
| Drop the temporal gradient path to save memory on 3D volumes | SLTT (Meng et al., ICCV 2023) and OTTT (Xiao et al., NeurIPS 2022) |
| Gene-conditioned thresholds | FiLM scaling, proved equivalent in A6 |
| Per-timestep loss or early exit to cut energy | TET (Deng et al., ICLR 2022) and SEENN (Li et al., NeurIPS 2023). *Recalled from memory, not re-checked this session.* |

The next mathematical step would be extending A5 to deep layers, where inputs are spike trains rather than constants. Gygax & Zenke report that the single-neuron equivalence breaks in deep networks, and I cannot prove a replacement result. **This is where I stop.** Presenting a renamed combination of the methods above as new would be the kind of claim this document exists to prevent.

What the project can honestly use from Part A:
- A5's design rule: keep the reset in the graph, with a surrogate of width $\theta$.
- DSR or SLTT to make 3D training fit in T4 GPU memory.
- **Measured** energy on hardware as the empirical contribution. The literature search found no Alzheimer's SNN paper that measures it.

---

## Part B: testing the claim "genetics adds value beyond MRI"

### B1. Primary test: likelihood ratio on nested models: PROVED (asymptotically) + CHECKED

Let $m$ be the MRI model's score (computed out-of-sample) and $G$ the genetic predictors (APOE ε4 count, polygenic risk score):

$$\operatorname{logit} P(y = 1) = \alpha + \beta m + \gamma^\top G, \qquad H_0: \gamma = 0$$

Under $H_0$ and standard regularity conditions, $\Lambda = 2(\ell_{\text{full}} - \ell_{\text{base}}) \xrightarrow{d} \chi^2_k$ with $k = \dim G$ (Wilks, 1938). So rejecting when $\Lambda > \chi^2_{k,0.95}$ has an asymptotic false-positive rate of 5%.

**Check:** 1,000 simulated cohorts ($n = 400$) where genetics truly has no effect. The test rejected **5.6%**, inside the 95% binomial band of about 3.7–6.4%. When genetics has a real effect, power is **98%** (200 cohorts).

### B2. Why DeLong is not the primary test: KNOWN + CHECKED

Comparing the AUCs of the base and full models with DeLong's test is invalid for nested models. Under $H_0$ the AUC difference is not normally distributed, so the test is badly conservative (Demler, Pencina & D'Agostino, *Statistics in Medicine*, 2012). Their recommendation: test association first (B1), and only then report the AUC gain with a confidence interval.

**Check:** on the same 1,000 null cohorts, nested DeLong rejected only **0.2%** at nominal 5%. For non-nested models with equal true AUC, DeLong rejected **5.6%**, as it should.

### B3. DeLong variance: KNOWN + CHECKED

The AUC is the Mann–Whitney U-statistic $\hat A = \frac{1}{mn}\sum_{i,j} \psi(X_i, Y_j)$, with $\psi = \mathbf{1}[X > Y] + \tfrac12\mathbf{1}[X = Y]$, over $m$ positive and $n$ negative cases. DeLong et al. (1988) define placement values $V_{10}(X_i) = \frac1n\sum_j \psi(X_i, Y_j)$ and $V_{01}(Y_j) = \frac1m\sum_i \psi(X_i, Y_j)$, and estimate

$$\widehat{\operatorname{Cov}}(\hat A) = \frac{S_{10}}{m} + \frac{S_{01}}{n}$$

where $S_{10}$ and $S_{01}$ are the sample covariance matrices of the placement values across the two models. The code computes placements with midranks (Sun & Xu, 2014).

**Check:** AUCs match scikit-learn exactly, including with ties. The DeLong standard error agrees with a 2,000-sample bootstrap to within 10%.

### B4. Subgroup claims: CHECKED

A claim such as "genetics matters more for APOE ε4 carriers" requires a formal interaction term $m \times \text{carrier}$, not two separate subgroup AUCs. With several subgroups, adjust with Holm's method.

**Check:** with no true interaction, the Wald test rejected **4.5%** (1,000 cohorts). Holm-adjusted p-values match statsmodels in 50 random cases.

### B5. Assumptions maths cannot settle

These are design requirements, not theorems:
- The MRI score must come from subjects the network never trained on: an external cohort, or out-of-fold predictions. Otherwise B1's null distribution does not hold.
- The logistic model's linear form for $m$ and $G$.
- Transfer from ADNI to the external cohort.
- The simulations use a synthetic cohort. They check that the tests behave correctly; they say nothing about whether genetics helps in real data.

---

## References

Each reference was checked by web search in this session unless marked otherwise.

- Bu et al. *Optimal ANN-SNN conversion for high-accuracy and ultra-low-latency spiking neural networks.* ICLR 2022. arXiv 2303.04347.
- Meng et al. *Training high-performance low-latency spiking neural networks by differentiation on spike representation.* CVPR 2022. arXiv 2205.00459.
- Meng et al. *Towards memory- and time-efficient backpropagation for training spiking neural networks.* ICCV 2023. arXiv 2302.14311.
- Xiao et al. *Online training through time for spiking neural networks.* NeurIPS 2022. arXiv 2210.04195.
- Gygax & Zenke. *Elucidating the theoretical underpinnings of surrogate gradient learning in spiking neural networks.* arXiv 2404.14964, 2024.
- Demler, Pencina & D'Agostino. *Misuse of DeLong test to compare AUCs for nested models.* Statistics in Medicine 31(23):2577–2587, 2012.
- *From memory, not re-checked:* Neftci, Mostafa & Zenke 2019; Perez et al. (FiLM) 2018; Deng et al. (TET) 2022; Li et al. (SEENN) 2023; DeLong et al. 1988; Sun & Xu 2014; Wilks 1938.
