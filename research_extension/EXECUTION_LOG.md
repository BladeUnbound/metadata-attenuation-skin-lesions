# Execution log

Every job, analysis and milestone in time order. Times are from `date` (IST). Jobs run through scripts/runner.py, which appends one row per job.

| time | phase | event | details |
|---|---|---|---|
| 2026-10-30 10:27 IST | P0 | directive received; environment checked | torch 2.14.1, sklearn 1.9.1, numpy 2.5.4; no nbclient/LightGBM/XGBoost/concept_erasure |
| 2026-10-30 10:40 IST | P0 | 00_project_audit.md, RESEARCH_SCHEDULE.md v1 written | no experiments run |
| 2026-10-30 10:31 IST | P1 | literature audit started | |
| 2026-10-30 10:44 IST | P1 | literature audit complete | 01_literature_review.md, 02_literature_comparison.csv (49 rows), 03_novelty_audit.md, literature/paper_notes.md; 6 mandatory papers read [F]; key reframing: M_A <= C_Y (to be derived in P2) |
| 2026-10-30 10:52 IST | P2 | mathematical foundations complete | 04_mathematical_foundations.md; notebooks/01_information_theory_verification.ipynb executed (13 code cells, 0 errors, 2 figures); scripts gauss_logit.py (validated vs 2M-draw MC), nbrun.py, build_nb01.py |
| 2026-10-30 10:57 IST | P3 | task SYN-1 exit 0 | start 2026-10-30 10:56 IST; log results/logs/SYN-1.log |
| 2026-10-30 10:58 IST | P3 | SYN-3 registered (code written, not run) | runs after SYN-1/SYN-2 are inspected |
| 2026-10-30 11:00 IST | P3 | task SYN-2 exit 1 | start 2026-10-30 10:57 IST; log results/logs/SYN-2.log |
| 2026-10-30 11:00 IST | P3 | SYN-2 first run killed (worker SIGBUS, memory pressure at d=768 with 8 workers); no SYN-2 output written; rerun with 4 workers (code change: n_jobs only) | SYN-1 inspected: predictions (a)-(e) hold; unexpected: random matched eta 0.93 in R6 (strong age-label) |
| 2026-10-30 11:02 IST | P3 | task SYN-2 exit 1 | start 2026-10-30 11:00 IST; log results/logs/SYN-2.log |
| 2026-10-30 11:03 IST | P3 | SYN-2 second run: worker SIGSEGV after ~64/210 jobs (4 workers); largest config runs cleanly in-process (15 s) -> BLAS threading in parallel workers; rerun with n_jobs=1 | no SYN-2 output from failed runs |
| 2026-10-30 11:08 IST | P3 | task SYN-2 exit 1 | start 2026-10-30 11:03 IST; log results/logs/SYN-2.log |
| 2026-10-30 11:14 IST | P3 | SYN-2 third run: LinAlgError 'SVD did not converge' in gauss_logit.explained (pinv of rank-deficient view covariance, cond 1e16-1e18), intermittent (job 181 R2 d=768 n/d=1.5 failed in one scan, not in a rerun). Fix: eigh-based PSD pseudo-inverse. Re-running notebook, SYN-1 and SYN-2 on the new code | |
| 2026-10-30 11:14 IST | P2 | task P2-nb01 exit 0 | start 2026-10-30 11:14 IST; log results/logs/P2-nb01.log |
| 2026-10-30 11:14 IST | P3 | task SYN-1 exit 0 | start 2026-10-30 11:14 IST; log results/logs/SYN-1.log |
| 2026-10-30 11:24 IST | P3 | task SYN-2 exit 0 | start 2026-10-30 11:15 IST; log results/logs/SYN-2.log |
| 2026-10-30 11:24 IST | P3 | SYN-2 inspected | H-S2 (head geometry) NOT supported: pen-cost gap LEACE-orth mean 0.08e-3, corr with SD ratio 0.085; small-n/d problems are estimation (direction cos 0.79 at n/d=1.25, out-of-sample leakage 0.37, matched-control and eta-estimate variance) |
| 2026-10-30 11:27 IST | P3 | task SYN-3 exit 0 | start 2026-10-30 11:24 IST; log results/logs/SYN-3.log |
| 2026-10-30 11:36 IST | P3 | task SYN-3b exit 0 | start 2026-10-30 11:29 IST; log results/logs/SYN-3b.log |
| 2026-10-30 11:37 IST | P3 | task SYN-3c exit 1 | start 2026-10-30 11:37 IST; log results/logs/SYN-3c.log |
| 2026-10-30 11:39 IST | P3 | task SYN-3c exit 0 | start 2026-10-30 11:37 IST; log results/logs/SYN-3c.log |
| 2026-10-30 11:40 IST | P3 | Phase 3 complete | 05_synthetic_experiments.md; SYN-1, SYN-2, SYN-3, SYN-3b, SYN-3c done; MATE not justified on synthetic evidence |
| 2026-10-30 12:01 IST | P6 | task REAL-2-ham10000 exit 0 | start 2026-10-30 11:42 IST; log results/logs/REAL-2-ham10000.log |
| 2026-10-30 12:28 IST | P6 | task REAL-2-isic2019_nonham exit 0 | start 2026-10-30 12:01 IST; log results/logs/REAL-2-isic2019_nonham.log |
| 2026-10-30 13:05 IST | P6 | task REAL-2-isic2020 exit 0 | start 2026-10-30 12:28 IST; log results/logs/REAL-2-isic2020.log |
| 2026-10-30 13:05 IST | P6 | task REAL-3-ham10000-hsv exit 0 | start 2026-10-30 13:05 IST; log results/logs/REAL-3-ham10000-hsv.log |
| 2026-10-30 13:14 IST | P6 | task REAL-3-ham10000-vit_b_16 exit 0 | start 2026-10-30 13:05 IST; log results/logs/REAL-3-ham10000-vit_b_16.log |
| 2026-10-30 13:14 IST | P6 | task REAL-3-isic2019_nonham-hsv exit 0 | start 2026-10-30 13:14 IST; log results/logs/REAL-3-isic2019_nonham-hsv.log |
| 2026-10-30 14:05 IST | P6 | task REAL-3-isic2019_nonham-vit_b_16 exit 0 | start 2026-10-30 13:14 IST; log results/logs/REAL-3-isic2019_nonham-vit_b_16.log |
| 2026-11-01 21:25 IST | P6 | REAL-1 re-analysis run (reads arrays only); REAL-2 confirmatory computed: 3/3 Holm rejections | results/real/REAL-2_confirmatory.csv |
| 2026-11-01 21:27 IST | P6 | task REAL-smoke exit 0 | start 2026-11-01 21:25 IST; log results/logs/REAL-smoke.log |
| 2026-11-01 21:39 IST | P6 | task REAL-4-ham10000 exit 0 | start 2026-11-01 21:27 IST; log results/logs/REAL-4-ham10000.log |
| 2026-11-01 21:43 IST | P4/P6/P7 | docs: 06 written; 08 §1–2, 09 and 10 drafted; RESULTS_INDEX filled; e50 code-version history verified (9915166 → 4631999 → ca82ac; REAL-2 = E50b version); real_new t0/LEACE losses bit-identical to E50a on PAD; analysis-only scripts real_new_analysis.py and real_e11.py added (no jobs) | 06, 08, 09, 10, RESULTS_INDEX.md |
| 2026-11-01 22:00 IST | P6 | task REAL-4-isic2019_nonham exit 0 | start 2026-11-01 21:39 IST; log results/logs/REAL-4-isic2019_nonham.log |
| 2026-11-01 22:04 IST | P6 | task REAL-4-milk10k_derm exit 0 | start 2026-11-01 22:00 IST; log results/logs/REAL-4-milk10k_derm.log |
| 2026-11-01 22:26 IST | P6 | task REAL-4-isic2020 exit 0 | start 2026-11-01 22:04 IST; log results/logs/REAL-4-isic2020.log |
| 2026-11-01 22:28 IST | P6 | task REAL-4-hiba exit 0 | start 2026-11-01 22:26 IST; log results/logs/REAL-4-hiba.log |
| 2026-11-01 22:42 IST | P6 | task REAL-5-ham10000 exit 0 | start 2026-11-01 22:28 IST; log results/logs/REAL-5-ham10000.log |
| 2026-11-01 23:19 IST | P6 | task REAL-6-isic2019_nonham exit 0 | start 2026-11-01 22:42 IST; log results/logs/REAL-6-isic2019_nonham.log |
| 2026-11-01 23:29 IST | P6 | task REAL-7-sex-ham10000 exit 0 | start 2026-11-01 23:19 IST; log results/logs/REAL-7-sex-ham10000.log |
| 2026-11-01 23:38 IST | P6 | task REAL-7-joint-ham10000 exit 0 | start 2026-11-01 23:29 IST; log results/logs/REAL-7-joint-ham10000.log |
| 2026-11-01 23:43 IST | P6 | task REAL-7-sex-milk10k_derm exit 0 | start 2026-11-01 23:38 IST; log results/logs/REAL-7-sex-milk10k_derm.log |
| 2026-11-01 23:47 IST | P6 | task REAL-7-joint-milk10k_derm exit 0 | start 2026-11-01 23:43 IST; log results/logs/REAL-7-joint-milk10k_derm.log |
| 2026-11-02 00:03 IST | P6 | task REAL-8-ham10000-isic2019_nonham exit 0 | start 2026-11-01 23:47 IST; log results/logs/REAL-8-ham10000-isic2019_nonham.log |
| 2026-11-02 00:12 IST | P6 | task REAL-8-isic2019_nonham-ham10000 exit 0 | start 2026-11-02 00:03 IST; log results/logs/REAL-8-isic2019_nonham-ham10000.log |
| 2026-11-02 00:28 IST | P6 | task REAL-8-ham10000-isic2020 exit 0 | start 2026-11-02 00:12 IST; log results/logs/REAL-8-ham10000-isic2020.log |
| 2026-11-02 00:28 IST | P6 | queue b5jcnzjzq finished, exit 0; 15/15 tasks done (21:25 → 00:28). Logs scanned: no errors; one PyTorch UserWarning in nonlin.py:81 (scalar conversion when logging the adversary loss; no effect on results) | results/logs/REAL-*.log |
| 2026-11-02 00:28 IST | P6/P7 | analysis `real_new_analysis.py all`: oracle OK; nae failed (KeyError: arm name sliced with [:-4] instead of len("\|l0") = 3). Fixed; re-run at 00:28 completed all modes. Analysis-only bug, no job affected | results/real/REAL-new_*.json |
| 2026-11-02 00:29 IST | P6/P7 | `real_e11.py`: E11 interior CIs; endpoints reproduce L (π = 0) and C (π = 1) of REAL-1 exactly | results/real/REAL-1_E11_missing_age.csv |
| 2026-11-02 00:31 IST (stat of REAL-new_sex.json; first written as 00:35, an estimate, corrected) | P6/P7 | added `sex` mode to real_new_analysis.py (sex eraser's own C, L, M, η via real_reanalysis.analyse_tag, conc = sex; separate file, REAL-1 CSV untouched so the 37-contrast count in 10 §3 stands). Read site-probe fields from existing E50/E51 JSONs (probe_site) for leakage specificity. Per-fold breakdown of REAL-6 q = 5 rep 0 PCA (fold C: 7.5, 130.9, −11.4, 39.6, 3.1). Analysis only | results/real/REAL-new_sex.json; real_new_analysis.py 4d36db92f913 |
| 2026-11-02 01:17 IST | P7 | re-ran `real_new_analysis.py all` with the final version (4d36db92f913) so every REAL-new output comes from one script version; all 7 JSONs byte-identical to the previous outputs. Docs: 08 §3–8 written; 07 §5, 06 §5 (E11 CIs) and §7 (correction: REAL-7 did not add site leakage) updated; 09 and 10 §9 completed; 11_final_research_assessment.md written; RESULTS_INDEX rows added | 06–11, RESULTS_INDEX.md |
