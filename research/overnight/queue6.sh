#!/bin/zsh
# Iteration 5, Stage 3-5 (E42, REGISTRY.md 21:59 IST). Sequential.
cd "$HOME/Desktop/CV EL/research" || exit 1
P=../novelty/.venv/bin/python
STATUS=overnight/STATUS6.md
{ echo "# Queue 6 status (E42)"; echo; echo "Started $(date '+%Y-%m-%d %H:%M'). e42_hiba was run interactively before the queue (smoke test, exit 0)."; echo; echo "| Job | Est. min | Start | End | Exit |"; echo "|---|---|---|---|---|"; } > $STATUS
run() { local name=$1 est=$2; shift 2; local start=$(date +%H:%M); "$@" > overnight/logs/$name.log 2>&1; local rc=$?
  echo "| $name | $est | $start | $(date +%H:%M) | $rc |" >> $STATUS; }
for d in msk2020 pad_ufes milk10k_derm milk10k_clin ham10000 isic2019_nonham isic2020; do run e42_$d 10 $P -I experiments/e42_graph.py $d; done
run e42_summary 1 $P -I experiments/e42_graph.py summarise
echo "\nQueue 6 finished at $(date +%H:%M)." >> $STATUS
