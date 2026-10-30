#!/bin/zsh
# Stage 2+ (after G1 = PROCEED, 2026-10-29). Sequential. Registry: REGISTRY.md (E51, E52, E53, E55, E54; E50c = amendment 3).
cd "/Users/paradox/Desktop/CV EL/research/experiments"
PY=(../../novelty/.venv/bin/python -I)
S="../iteration6/STATUS.md"
L="../iteration6/logs"
echo "\n## Stage-2 queue started $(date '+%Y-%m-%d %H:%M %Z')\n| job | start | end | exit |\n|---|---|---|---|" >> $S
job() { local name=$1; shift; local st=$(date +%H:%M); "$PY[@]" "$@" > $L/$name.log 2>&1; local rc=$?; echo "| $name | $st | $(date +%H:%M) | $rc |" >> $S; }
for ds in hiba ham10000 pad_ufes milk10k_derm; do job E51_$ds e50_dose_restoration.py $ds --concept site --light; done
for ds in hiba msk2020 milk10k_derm pad_ufes milk10k_clin ham10000 isic2019_nonham isic2020; do job E50c_$ds e50_dose_restoration.py $ds --light --conditional; done
for ds in hiba msk2020 pad_ufes milk10k_derm milk10k_clin ham10000 isic2019_nonham isic2020; do job E52_${ds}_age e52_nonlinear_probe.py $ds --concept age; done
for ds in hiba ham10000 pad_ufes milk10k_derm; do job E52_${ds}_site e52_nonlinear_probe.py $ds --concept site; done
job E53 e53_estimators.py hiba msk2020 pad_ufes milk10k_derm milk10k_clin ham10000 isic2019_nonham isic2020
job E55 e55_shift_rules.py
echo "Stage-2 queue finished $(date '+%Y-%m-%d %H:%M %Z')" >> $S
