#!/bin/zsh
# E50a: age, ConvNeXt-T, fold assignment 0. Sequential. Registered in REGISTRY.md (13:49 IST; amendments 14:55, 14:58).
cd "/Users/paradox/Desktop/CV EL/research/experiments"
PY="../../novelty/.venv/bin/python -I"
S="../iteration6/STATUS.md"
echo "\n## E50a queue started $(date '+%Y-%m-%d %H:%M %Z')\n| job | start | end | exit |\n|---|---|---|---|" >> $S
for ds in pad_ufes hiba msk2020 isic2020 ham10000 isic2019_nonham milk10k_derm milk10k_clin; do
  st=$(date +%H:%M)
  eval $PY e50_dose_restoration.py $ds > ../iteration6/logs/E50_${ds}.log 2>&1
  rc=$?
  echo "| e50a_${ds} | $st | $(date +%H:%M) | $rc |" >> $S
done
echo "E50a queue finished $(date '+%Y-%m-%d %H:%M %Z')" >> $S
