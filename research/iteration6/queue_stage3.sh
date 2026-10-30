#!/bin/zsh
# Stage 3. Waits for the stage-2 queue (pid passed as $1) to exit, then runs sequentially in priority order.
# Order (REGISTRY: E50b is part of E50; E54 only "if time allows after E50-E53", so it runs last):
#   E53b (registered, gated on G1 = PROCEED; sets whose E50 contrast CI > 0)
#   E50b --minimal: ISIC 2020 fold assignment 1; ResNet-18 on the four E45b sets; HAM10000 and ISIC 2019 fold
#   assignment 1; fold assignment 2 on ISIC 2020, HAM10000, ISIC 2019
#   E54 seeds 1 and 2 on HIBA and PAD-UFES-20
#   reports
# Scope deviations from the registry (compute, recorded in STATUS.md 2026-10-29): E50b only on the sets above, and
# --minimal (t0, targeted, matched search, 5 matched random, 5 sham; no same-rank, re-append or LW arms).
cd "/Users/paradox/Desktop/CV EL/research/experiments"
PY=(../../novelty/.venv/bin/python -I)
S="../iteration6/STATUS.md"
L="../iteration6/logs"
if [[ -n "$1" ]]; then while kill -0 $1 2>/dev/null; do sleep 30; done; fi
echo "\n## Stage-3 queue started $(date '+%Y-%m-%d %H:%M %Z')\n| job | start | end | exit |\n|---|---|---|---|" >> $S
job() { local name=$1; shift; local st=$(date +%H:%M); "$PY[@]" "$@" > $L/$name.log 2>&1; local rc=$?; echo "| $name | $st | $(date +%H:%M) | $rc |" >> $S; }
job E53_rerun e53_estimators.py hiba msk2020 pad_ufes milk10k_derm milk10k_clin ham10000 isic2019_nonham isic2020
for ds in milk10k_derm milk10k_clin ham10000 isic2019_nonham isic2020; do job E53b_$ds e53b_erasure_fullfusion.py $ds; done
job E50b_isic2020_f1 e50_dose_restoration.py isic2020 --fold-seed 1 --minimal
for ds in pad_ufes milk10k_derm ham10000 isic2019_nonham; do job E50b_${ds}_rn18 e50_dose_restoration.py $ds --rung resnet18 --minimal; done
for ds in ham10000 isic2019_nonham; do job E50b_${ds}_f1 e50_dose_restoration.py $ds --fold-seed 1 --minimal; done
for ds in isic2020 ham10000 isic2019_nonham; do job E50b_${ds}_f2 e50_dose_restoration.py $ds --fold-seed 2 --minimal; done
job E54_hiba e54_finetune_seeds.py hiba 1 2
job E54_pad_ufes e54_finetune_seeds.py pad_ufes 1 2
job report_e50 e50_report.py
job report_iter6 iteration6_summary.py
echo "Stage-3 queue finished $(date '+%Y-%m-%d %H:%M %Z')" >> $S
