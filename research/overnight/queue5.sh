#!/bin/zsh
# Iteration 5, Stage 1 (E40, REGISTRY.md). Sequential; one job at a time.
cd "$HOME/Desktop/CV EL/research" || exit 1
P=../novelty/.venv/bin/python
STATUS=overnight/STATUS5.md
{ echo "# Queue 5 status (E40)"; echo; echo "Started $(date '+%Y-%m-%d %H:%M')."; echo; echo "| Job | Est. min | Start | End | Exit |"; echo "|---|---|---|---|---|"; } > $STATUS
echo "| e40_ham10000 | 5 | 21:58 | 22:03 | 0 (run interactively) |" >> $STATUS
run() { local name=$1 est=$2; shift 2; local start=$(date +%H:%M); "$@" > overnight/logs/$name.log 2>&1; local rc=$?
  echo "| $name | $est | $start | $(date +%H:%M) | $rc |" >> $STATUS; }
for d in pad_ufes hiba msk2020 milk10k_derm milk10k_clin isic2019_nonham isic2020; do run e40_$d 10 $P -I experiments/e40_crossfit_baseline.py $d; done
run e40_dedup_isic2019 8 $P -I experiments/e40_crossfit_baseline.py dedup_isic2019
run e40_tests 2 $P -I experiments/e40_crossfit_baseline.py tests
echo "\nQueue 5 finished at $(date +%H:%M)." >> $STATUS
