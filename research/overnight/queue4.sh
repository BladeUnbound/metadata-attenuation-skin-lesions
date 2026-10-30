#!/bin/zsh
# Runs after queue3 (pid passed as $1). Registered in iteration4/REGISTRY_E30_E34.md (E19 completion).
cd "$HOME/Desktop/CV EL/research" || exit 1
P=../novelty/.venv/bin/python
while kill -0 $1 2>/dev/null; do sleep 30; done
STATUS=overnight/STATUS3.md
run() { local name=$1 est=$2; shift 2; local start=$(date +%H:%M); "$@" > overnight/logs/$name.log 2>&1; local rc=$?
  echo "| $name | $est | $start | $(date +%H:%M) | $rc |" >> $STATUS; $P -I iteration4/make_summary.py >/dev/null 2>&1; }
run e19_embed_isic2020 40 $P -I experiments/e19_resolution_ladder.py embed isic2020
run e19_analyse_all_v2 15 $P -I experiments/e19_resolution_ladder.py analyse
echo "\nQueue 4 finished at $(date +%H:%M)." >> $STATUS
