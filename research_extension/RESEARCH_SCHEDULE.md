# Research schedule — metadata-aware, task-preserving erasure (MATE, working name)

*Version 1, written 2026-10-30 10:40 IST, before any phase started. The machine-readable schedule is configs/schedule.json, and scripts/runner.py executes it. Revisions are appended as dated sections and never overwrite earlier versions. Compute estimates use the measured E50 costs in 00_project_audit.md §2.*

Resources: one Apple M5 (10 CPU cores, 24 GB unified memory, MPS GPU). Jobs run **sequentially** (standing project rule). CPU jobs use up to 8 joblib workers; GPU jobs (fine-tuning only) never overlap CPU jobs.

## Dependency graph

```
P1 literature ──► P2 mathematics ──► P3 synthetic ──► P4 real-data baselines ──► P5 MATE ──► P6 real-data validation ──► P7 final analysis
                         │                  │                    ▲
                         └── notebook ──────┘                    └── needs P3 metric validation and P2 identities
```

No phase starts before its predecessor's completion criteria are met, except literature reading for P5 candidates, which may continue in parallel with P3 writing (no compute).

## Phase 1 — Literature audit (no compute)
- **Objectives:** verify the six mandatory papers (exact title, authors, venue, status); extract formulation, objective, transformation, guarantees, assumptions, metrics, datasets, baselines, limitations and code; expand to the 16 listed areas; decide the novelty of the central question.
- **Dependencies:** none.
- **Tasks:** primary-source retrieval (arXiv/proceedings PDFs read locally with pdftotext); targeted searches per area; comparison matrix; novelty audit.
- **Runtime:** reading only, about 2–4 h of agent time. CPU/GPU: negligible.
- **Deliverables:** 01_literature_review.md, 02_literature_comparison.csv, 03_novelty_audit.md.
- **Success:** every mandatory paper verified from its primary source, or explicitly marked unverifiable; each claim tagged with a verification level.
- **Failure:** a mandatory paper cannot be found. It is then reported as unverified, never described from secondary memory.

## Phase 2 — Mathematical foundations (negligible compute)
- **Objectives:** derive R_A, C_Y, Δ_A, M_A and their population/CMI forms; the decomposition identity with assumptions; sufficiency; controlled examples and limits (redundant age, independent age, shared component, nonlinear encoding, destroyed diagnostic feature, irrelevant metadata); deterministic vs stochastic transformations.
- **Dependencies:** P1, for what is already established (cite rather than re-claim).
- **Runtime:** derivations plus a notebook (discrete distributions and Gaussian/logistic examples; Monte Carlo up to 10⁶ draws), under 10 min CPU.
- **Deliverables:** 04_mathematical_foundations.md, notebooks/01_information_theory_verification.ipynb, executed with outputs.
- **Success:** every identity verified numerically to tolerance; every proposition has explicit assumptions and a complete proof or is labelled a conjecture or example.
- **Failure:** an identity fails numerically. Its assumptions are revisited, not the check.

## Phase 3 — Synthetic experiments (CPU, about 1–3 h)
- **Objectives:** ground truth for the three metrics in at least four regimes (redundant, entangled, irrelevant, nonlinear) plus regimes the theory flags; methods none, LEACE, random (rank- and cost-matched), linear adversarial (RLACE), nonlinear adversarial, and a task-preserving transformation only if P2 justifies it.
- **Dependencies:** P2 (the Bayes-risk calculations define the ground truth).
- **Runtime:** about 4 regimes × 6 methods × 10 sweep points × 5 seeds, with small networks on n ≤ 20k and d ≤ 128: roughly 1–3 h CPU.
- **Deliverables:** scripts/synthetic.py, configs/synthetic_*.json, results/synthetic/*.csv, figures/synthetic_*.pdf, 05_synthetic_experiments.md.
- **Success:** the metrics separate redundancy removal from destruction of diagnostic information in the regimes where theory says they should, and estimator bias against ground truth is quantified.
- **Failure:** the metrics cannot separate the regimes even with known truth. The real-data plan is then revised before P4.

## Phase 4 — Real-data baselines, E1 and E2 (CPU, about 6–10 h)
- **Objectives:** the three-way frontier for existing methods on frozen ConvNeXt-T (age), with matched-erasure and matched-diagnostic-cost comparisons.
- **Dependencies:** P3 (validated metrics and probe suite); existing E50 caches (reused where definitions agree).
- **Sets:** large sets first (HAM10000 as development set; ISIC 2019, ISIC 2020, MILK10k derm). Small sets run as a documented n≈d stress test, not as evidence.
- **Runtime:** LEACE dose and rank sweeps reuse E50 or are cheap (closed form); RLACE and nonlinear adversaries run per outer fold. Estimate 1–2 h per large set, about 6–10 h in total.
- **Deliverables:** 06_baseline_experiments.md, results/E1_*, results/E2_*, Pareto figures.
- **Success:** frontiers with paired grouped CIs at matched cost.
- **Failure:** adversarial baselines are unstable (seed spread larger than their effect). Reported as such.

## Phase 5 — MATE (CPU/MPS, about 3–8 h)
- **Objectives:** decide whether a task-preserving transformation is justified and distinct from prior work; implement the most defensible formulation; quantify the optimism from using training labels inside the transformation (nested versus split cross-fitting on the development set); compare at matched diagnostic cost.
- **Dependencies:** P1 novelty audit, P3 synthetic behaviour, P4 frontier.
- **Registration:** a confirmatory comparison (MATE vs LEACE vs matched random, at matched cost) is registered in EXPERIMENT_REGISTRY.md **before** it runs on any non-development set.
- **Deliverables:** 07_mate_method.md, scripts/mate.py, configs/mate_*.json, results/E_MATE_*.
- **Failure:** MATE does not beat LEACE plus matched controls on the development frontier. It is then not taken to confirmatory testing, and the report says so.

## Phase 6 — Real-data validation, E3–E12 (CPU about 10–20 h; GPU about 3–5 h if E9 runs)
- E3 substitutability regimes (sets ordered by age–label dependence); E4 adaptive nonlinear adversaries; E5 attribute specificity (age, sex, site; 3×3 matrix); E6 representation ladder (HSV, ResNet-18, ConvNeXt-T, ViT-B/16, ViT-L/16 SWAG); E7 sample-size and dimension scaling (n/d, conditioning, shrinkage); E8 cross-dataset transfer; E9 fine-tuned encoders (split cross-fitted fine-tuning, GPU; scheduled last, optional); E10 restoration and dose; E11 missing metadata; E12 joint removal.
- **Dependencies:** P4 (all); P5 (MATE arms).
- **Deliverables:** 08_real_data_experiments.md, 09_robustness_and_failure_analysis.md, results/E3–E12.
- **Success:** each experiment's registered outcome reported with CIs, including nulls and failures.

## Phase 7 — Final analysis (CPU, under 1 h)
- **Objectives:** multiplicity, effect sizes, reproduction of any surprising result, contribution and novelty assessment, final report.
- **Deliverables:** 10_statistical_analysis.md, 11_final_research_assessment.md, RESULTS_INDEX.md completed.

## Bookkeeping
- `EXPERIMENT_REGISTRY.md`: hypothesis, protocol, compute estimate and confirmatory/exploratory status, written before each experiment, with times from `date`.
- `EXECUTION_LOG.md`: one line per job with start, end, exit code and log path, written by scripts/runner.py.
- `RESULTS_INDEX.md`: every table and figure mapped to its config, script, raw results and code hash.
- Seeds are recorded in every config. Every output directory is versioned (`results/<experiment>/<config-hash>/`). Jobs checkpoint per outer fold and resume.

---
## Revision 2 — 2026-10-30 11:46 IST (after Phases 1–3)

**What changed and why.**
1. **Phase 5 (MATE) is reduced to a decision record, with no confirmatory method comparison.**
   - Theory (04 §3, §6): metadata substitution is capped by the image-only cost, M_A ≤ C_Y. For attribute-free erasers, the fused frontier is set by the data (Proposition 3).
   - Synthetic evidence (05 SYN-3/3b/3c): task-aware nonlinear erasers never beat LEACE's substitution efficiency at matched leakage.
   - The literature (03) already covers task-preserving erasure.
   - 07_mate_method.md documents the candidates, the evidence and the decision.
2. **The outcome is now η = M_A / C_Y and L_YA**, not M_A alone.
3. **Real-data work is REAL-1 to REAL-8** (EXPERIMENT_REGISTRY.md), replacing E1–E12 as follows:

| Directive item | Registry entry |
|---|---|
| E1, E2, E3, E10, E11 | REAL-1 (re-analysis of saved E50/E51 arrays; no refits) |
| new confirmatory test | REAL-2: η on ViT-L/16, untouched by erasure; Holm over 3 sets |
| E6 | REAL-3 |
| positive control, E4 audit | REAL-4 |
| E4 nonlinear | REAL-5 |
| E7 | REAL-6 |
| E5, E12 | REAL-7 |
| E8 | REAL-8 |
| E9 (fine-tuned encoders) | deferred, cost estimate in 11 |

**Queue and estimates** (sequential; configs/schedule.json):

| Tasks | Estimate |
|---|---|
| REAL-2 (3 sets, ViT-L/16) | about 1–1.5 h |
| REAL-3 (HSV and ViT-B on 2 sets) | about 1 h |
| REAL-smoke, REAL-4 (5 sets) | about 1 h |
| REAL-5 (HAM, 4 nonlinear arms) | about 1 h |
| REAL-6 (ISIC 2019, 24 configurations) | about 1.5–2 h |
| REAL-7 (sex and joint, 2 sets) | about 1.5 h |
| REAL-8 (3 transfers) | about 1 h |

REAL-1 runs whenever no other job is running (minutes; it reads arrays only).

**Completion criteria for Phase 6.** Every REAL entry is either run, with its result reported against its registered prediction, or reported as failed or skipped with its reason.

**Status, 2026-11-02 01:17 IST (times from EXECUTION_LOG.md).**

Phases 1–7 are complete. Every REAL entry was run, and each is reported against its registered prediction in 08 §3–7. None failed or was skipped.

| Tasks | Estimate | Actual |
|---|---|---|
| REAL-2 | 1–1.5 h | 1 h 23 min (11:42–13:05, 2026-10-30) |
| REAL-3 | 1 h | 1 h 0 min (13:05–14:05) |
| REAL-smoke + REAL-4 | 1 h | 1 h 3 min (21:25–22:28, 2026-11-01) |
| REAL-5 | 1 h | 14 min |
| REAL-6 | 1.5–2 h | 37 min |
| REAL-7 | 1.5 h | 28 min |
| REAL-8 | 1 h | 41 min |

Registry deviations:
- REAL-6: within-test leakage not computed.
- REAL-8: source eraser fitted on all source rows.

Both are recorded in 08–10.

E9 is still deferred; its cost estimate is in 11 §2.6. The essential next experiments are listed in 11 §2.6 and §8. None is scheduled until the user decides.
