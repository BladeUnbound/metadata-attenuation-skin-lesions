#!/bin/zsh
# Follow-up queue after paper review round 5 (registered in iteration4/REGISTRY_E30_E34.md, addendum).
# One job at a time; failures are recorded and the queue continues.
cd "$HOME/Desktop/CV EL/research" || exit 1
P=../novelty/.venv/bin/python
STATUS=overnight/STATUS3.md
{ echo "# Follow-up queue status (queue3)"; echo; echo "Started $(date '+%Y-%m-%d %H:%M')."; echo;
  echo "| Job | Est. min | Start | End | Exit |"; echo "|---|---|---|---|---|"; } > $STATUS
run() {
  local name=$1 est=$2; shift 2
  local start=$(date +%H:%M)
  "$@" > overnight/logs/$name.log 2>&1
  local rc=$?
  echo "| $name | $est | $start | $(date +%H:%M) | $rc |" >> $STATUS
  $P -I iteration4/make_summary.py >/dev/null 2>&1
}
run e31d_ft_fraction_ext 125 $P -I experiments/e31_capacity_controls.py ft_fraction_ext
run e14b_isic2019_r18 110 $P -I experiments/e14b_finetune_multi.py isic2019_nonham resnet18
run e14b_milk10k_derm_r18 35 $P -I experiments/e14b_finetune_multi.py milk10k_derm resnet18
echo "\nQueue finished at $(date +%H:%M)." >> $STATUS
