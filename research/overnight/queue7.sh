#!/bin/zsh
# Iteration 5: E44 (smoothness), E41 (labels, complete cases), E45 (erasure), E48 (mask swap). Sequential.
cd "$HOME/Desktop/CV EL/research" || exit 1
P=../novelty/.venv/bin/python
STATUS=overnight/STATUS7.md
{ echo "# Queue 7 status"; echo; echo "Started $(date '+%Y-%m-%d %H:%M')."; echo; echo "| Job | Est. min | Start | End | Exit |"; echo "|---|---|---|---|---|"; } > $STATUS
run() { local name=$1 est=$2; shift 2; local start=$(date +%H:%M); "$@" > overnight/logs/$name.log 2>&1; local rc=$?
  echo "| $name | $est | $start | $(date +%H:%M) | $rc |" >> $STATUS; }
run e44_small 5 $P -I experiments/e44_smoothness.py hiba msk2020 pad_ufes milk10k_derm milk10k_clin ham10000
run e44_isic 15 $P -I experiments/e44_smoothness.py isic2019_nonham isic2020
run e41_labels 40 $P -I experiments/e41_label_matrix.py labels
run e41_cc 25 $P -I experiments/e41_label_matrix.py cc
for d in pad_ufes milk10k_derm ham10000 isic2019_nonham; do run e45_$d 20 $P -I experiments/e45_erasure.py $d; done
run e48_mask_swap 30 $P -I experiments/e48_mask_swap.py
echo "\nQueue 7 finished at $(date +%H:%M)." >> $STATUS
