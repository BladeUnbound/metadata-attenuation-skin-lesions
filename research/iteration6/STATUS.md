# Iteration 6 status
- 2026-10-29: PAD-UFES-20 shake-down run (E50, first version) superseded by amendment 1/2; log kept in logs/E50_pad_ufes.log and logs/E50_pad_ufes_first_run_superseded.json.

## E50a queue started 2026-10-29 14:58 IST
| job | start | end | exit |
|---|---|---|---|
| e50a_pad_ufes | 14:58 | 15:02 | 0 |
| e50a_hiba | 15:02 | 15:06 | 0 |
| e50a_msk2020 | 15:06 | 15:11 | 0 |
| e50a_isic2020 | 15:11 | 18:27 | 0 |
| e50a_ham10000 | 18:27 | 19:02 | 0 |
| e50a_isic2019_nonham | 19:02 | 19:56 | 0 |
| e50a_milk10k_derm | 19:56 | 20:15 | 0 |
| e50a_milk10k_clin | 20:15 | 20:36 | 0 |
E50a queue finished 2026-10-29 20:36 IST
- 2026-10-29 20:37 IST: G1 = PROCEED (results/E50_gate.json): ISIC 2020 age p = 0.0005 ≤ 0.0125 and T − sham CI above 0. HIBA and MSK 2020 age p = 1.0 (matched control overshoots; see REGISTRY amendment 3). e50 script gained --conditional (default path unchanged; E50a version kept in logs/).

## Stage-2 queue started 2026-10-29 20:37 IST
| job | start | end | exit |
|---|---|---|---|
| E51_hiba | 20:37 | 20:40 | 0 |
| E51_ham10000 | 20:40 | 21:03 | 0 |
| E51_pad_ufes | 21:03 | 21:06 | 0 |
| E51_milk10k_derm | 21:06 | 21:15 | 0 |
| E50c_hiba | 21:16 | 21:18 | 0 |
| E50c_msk2020 | 21:18 | 21:20 | 0 |
| E50c_milk10k_derm | 21:20 | 21:29 | 0 |
| E50c_pad_ufes | 21:29 | 21:31 | 0 |
| E50c_milk10k_clin | 21:31 | 21:41 | 0 |
| E50c_ham10000 | 21:41 | 22:02 | 0 |
- 2026-10-29 22:24 IST: e50_report.py rerun after E51 + 6/8 E50c. Holm family complete: only isic2020:age rejected (p_holm 0.002); hiba:site p = 1.0 (matched control overshoots, image LL T − matched −279e-3). Specificity matrix: selective_age and selective_site both TRUE on HAM10000 and MILK10k derm; FALSE on HIBA and PAD. Small sets (HIBA, MSK, PAD; n ≈ 1.1–1.2k, d = 768): coordinate test/train SD 1.97–3.13 for every targeted eraser, matching check fails in both directions, so contrasts there are not interpretable as attribute-specific (post-hoc reading, registered p-values stand). E50c HIBA +61 is non-specific (site gain also +17, image LL 0.339 → 0.624).
| E50c_isic2019_nonham | 22:02 | 22:32 | 0 |
- 2026-10-29 23:06 IST: user asked for completion overnight. Stage 3 rewritten in priority order and chained to stage 2 (starts when pid 83637 exits): E53b (5 sets with E50 CI > 0; new script experiments/e53b_erasure_fullfusion.py, rebuild checked against E50 first-stage scores) → E50b → E54 last (REGISTRY: 'if time allows after E50–E53') → reports. Deviation for compute: E50b runs with new --minimal flag (t0, targeted, matched search, 5 matched random, 5 sham; no same-rank, re-append or LW arms); summarise() guarded for absent arms; unit test on saved PAD arrays: default path reproduces stored contrasts exactly, minimal keeps primary and sham identical. E50b scope unchanged otherwise: ResNet-18 on PAD/MILK derm/HAM/ISIC 2019, fold assignments 1 and 2 on ISIC 2020/HAM/ISIC 2019. Mac slept 15:54–~17:45 during E50a ISIC 2020 (lid closed, on battery; pmset log); caffeinate cannot prevent clamshell sleep.
| E50c_isic2020 | 22:32 | 23:12 | 0 |
| E52_hiba_age | 23:12 | 23:12 | 0 |
| E52_msk2020_age | 23:12 | 23:12 | 0 |
| E52_pad_ufes_age | 23:12 | 23:12 | 0 |
| E52_milk10k_derm_age | 23:12 | 23:12 | 0 |
| E52_milk10k_clin_age | 23:12 | 23:13 | 0 |
| E52_ham10000_age | 23:13 | 23:13 | 0 |
| E52_isic2019_nonham_age | 23:13 | 23:14 | 0 |
| E52_isic2020_age | 23:14 | 23:16 | 0 |
| E52_hiba_site | 23:16 | 23:16 | 0 |
| E52_ham10000_site | 23:16 | 23:16 | 0 |
| E52_pad_ufes_site | 23:16 | 23:16 | 0 |
| E52_milk10k_derm_site | 23:16 | 23:17 | 0 |
| E53 | 23:17 | 23:17 | 1 |
- 2026-10-29 23:18 IST: E53 failed at start (exit 1, logs/E53.log): pandas 3.1 copy-on-write makes to_numpy() read-only and raw_meta() wrote NaNs into it. No E53 output was saved. Fixed with to_numpy(float, copy=True) (encoding only, no change to the method); raw_meta and miss_matrix checked on all 8 sets. E53 re-queued as the first stage-3 job (E53_rerun); stage-3 waiter restarted before it had run anything. E52 (12 runs) completed; short runtimes are genuine (ISIC 2020 age 73 s, MLP early stopping).
| E55 | 23:17 | 23:23 | 1 |
Stage-2 queue finished 2026-10-29 23:23 IST

## Stage-3 queue started 2026-10-29 23:23 IST
| job | start | end | exit |
|---|---|---|---|
- 2026-10-29 23:23 IST: E55 exit 1 (logs/E55.log): all 12 evaluations completed and were logged, then main() added summary keys to res while iterating over it (RuntimeError). Fix: iterate over the two level names. Summary code tested on the 12 logged evaluations. E55 re-queued in queue_stage4.sh after stage 3 (stage-3 script not edited while running), followed by report regeneration. Stage 2 finished 23:23; stage 3 started 23:23 with E53_rerun.
- 2026-10-29 23:24 IST: E53 field reproduces_E40_corrected = false on some rungs is a tolerance artefact. (a) score fusion matches E40 corrected per-observation delta to max |diff| 1.9e-6 to 1.7e-4 nats; means are identical to 1e-5 (hiba, msk2020, pad_ufes, milk10k_derm checked). The registered check used np.allclose atol 1e-8, below lbfgs and BLAS reproducibility. iteration6_summary.py now reports the max abs difference instead.
| E53_rerun | 23:23 | 23:29 | 0 |
| E53b_milk10k_derm | 23:29 | 23:30 | 0 |
| E53b_milk10k_clin | 23:30 | 23:30 | 0 |
| E53b_ham10000 | 23:30 | 23:32 | 0 |
| E53b_isic2019_nonham | 23:32 | 23:34 | 0 |
| E53b_isic2020 | 23:34 | 23:37 | 0 |
- 2026-10-29 23:38 IST: E53_rerun (8 sets, 6 min) and E53b (5 sets, 8 min) completed, exit 0; E53b rebuild reproduces E50 first-stage scores exactly. FINAL_REPORT §7 written. Stage 3 well ahead of estimate; E50b started 23:37.
| E50b_isic2020_f1 | 23:37 | 23:59 | 0 |
| E50b_pad_ufes_rn18 | 23:59 | 00:00 | 0 |
| E50b_milk10k_derm_rn18 | 00:00 | 00:02 | 0 |
| E50b_ham10000_rn18 | 00:02 | 00:07 | 0 |
| E50b_isic2019_nonham_rn18 | 00:07 | 00:14 | 0 |
| E50b_ham10000_f1 | 00:14 | 00:24 | 0 |
| E50b_isic2019_nonham_f1 | 00:24 | 00:38 | 0 |
| E50b_isic2020_f2 | 00:38 | 00:59 | 0 |
| E50b_ham10000_f2 | 00:59 | 01:08 | 0 |
| E50b_isic2019_nonham_f2 | 01:08 | 01:22 | 0 |
- 2026-10-30 01:23 IST: E50b complete (10 runs, exit 0). ISIC 2020 T − matched across fold assignments 0/1/2: +0.99/+1.08/+0.85, all CIs > 0; HAM10000 and ISIC 2019 stable; ResNet-18: all three E45b claims hold (PAD null). FINAL_REPORT §3.6–3.7 written. E54 started 01:22.
| E54_hiba | 01:22 | 02:01 | 0 |
| E54_pad_ufes | 02:01 | 03:13 | 0 |
| report_e50 | 03:13 | 03:13 | 0 |
| report_iter6 | 03:13 | 03:13 | 0 |
Stage-3 queue finished 2026-10-30 03:13 IST

## Stage-4 queue started 2026-10-30 03:13 IST
| job | start | end | exit |
|---|---|---|---|
| E55_rerun | 03:13 | 03:18 | 0 |
| report_e50_final | 03:18 | 03:18 | 0 |
| report_iter6_final | 03:18 | 03:18 | 0 |
Stage-4 queue finished 2026-10-30 03:18 IST
- 2026-10-30 03:19 IST: All queues finished (stage 3 at 03:13, stage 4 at 03:18; every job exit 0). E54 seed 0 reproduces E14b exactly; the fine-tuning drop holds for all seeds. E55 rerun identical to the logged first run. FINAL_REPORT.md finalised.
