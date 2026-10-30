#!/bin/zsh
# Integrated overnight queue (replaces queue.sh, which was stopped before running anything).
# One job at a time; no new job starts after 05:45; each job has a runtime estimate (minutes).
# A job is skipped if its estimate would run past 06:00. Failures are recorded and the queue continues.
# Order and rationale: research/iteration4/REGISTRY_E30_E34.md ("Queue priority").
cd "$HOME/Desktop/CV EL/research" || exit 1
P=../novelty/.venv/bin/python
STATUS=overnight/STATUS.md
DEADLINE=$(date -j -v+1d -f "%H:%M:%S" "05:45:00" +%s); HARD=$(date -j -v+1d -f "%H:%M:%S" "06:00:00" +%s)
[ "$(date +%H)" -lt 6 ] && DEADLINE=$(date -j -f "%H:%M:%S" "05:45:00" +%s) && HARD=$(date -j -f "%H:%M:%S" "06:00:00" +%s)

{ echo "# Overnight queue status (queue2)"; echo; echo "Started $(date '+%Y-%m-%d %H:%M'); last start $(date -r $DEADLINE '+%H:%M'); hard stop for estimates $(date -r $HARD '+%H:%M')."; echo;
  echo "| Job | Est. min | Start | End | Exit |"; echo "|---|---|---|---|---|"; } > $STATUS

run() {
  local name=$1 est=$2; shift 2
  local now=$(date +%s)
  if [ $now -ge $DEADLINE ]; then echo "| $name | $est | skipped: past 05:45 | | |" >> $STATUS; return; fi
  if [ $((now + est * 60)) -gt $HARD ]; then echo "| $name | $est | skipped: estimate exceeds 06:00 | | |" >> $STATUS; return; fi
  local start=$(date +%H:%M)
  "$@" > overnight/logs/$name.log 2>&1
  local rc=$?
  echo "| $name | $est | $start | $(date +%H:%M) | $rc |" >> $STATUS
  $P -I iteration4/make_summary.py >/dev/null 2>&1
}

# Wait for jobs already running: E14 chain (39594) and the ISIC Archive downloads.
while kill -0 39594 2>/dev/null || pgrep -f fetch_isic_collection >/dev/null; do sleep 30; done
echo "\nPrior jobs finished at $(date +%H:%M).\n" >> $STATUS

# 1. Pre-registered second wave (E22)
for d in hiba msk2020 milk10k_derm milk10k_clin; do run embed_$d 8 $P -I experiments/e13_replication.py $d embed; done
for d in hiba msk2020 milk10k_derm milk10k_clin; do run analyse_$d 12 $P -I experiments/e13_replication.py $d analyse; done
run e22_paired_modality 8 $P -I experiments/e22_paired_modality.py
run e17_all 25 $P -I experiments/e17_retained_fraction.py ham10000 isic2020 isic2019_nonham pad_ufes hiba msk2020 milk10k_derm milk10k_clin
run e21_wave2 10 $P -I experiments/e21_complete_case.py hiba msk2020 milk10k_derm milk10k_clin

# 2. Requested CPU experiments on cached embeddings
run e30_recoverability 90 $P -I experiments/e30_recoverability_utility.py
run e32_permutation 40 $P -I experiments/e32_permutation_controls.py
run e33_shift 35 $P -I experiments/e33_shift.py
run e34_rule 5 $P -I experiments/e34_collection_rule.py

# 3. E31 size ladders (GPU embeddings, then CPU analysis) and frozen data scaling
for d in ham10000 pad_ufes isic2019_nonham; do run e31_embed_$d 15 $P -I experiments/e31_capacity_controls.py embed_family $d; done
run e31a_family 30 $P -I experiments/e31_capacity_controls.py analyse_family
run e31b_scaling 45 $P -I experiments/e31_capacity_controls.py scaling_frozen

# 4. Foundation-model rung, smallest datasets first
for d in pad_ufes msk2020 hiba milk10k_clin milk10k_derm ham10000 isic2019_nonham isic2020; do
  run e18_embed_$d 20 $P -I experiments/e18_foundation_rung.py embed $d
done
run e18_analyse 20 $P -I experiments/e18_foundation_rung.py analyse

# 5. Fine-tuning (checkpointed per fold; resumes if interrupted)
run e14b_pad 45 $P -I experiments/e14b_finetune_multi.py pad_ufes resnet18 convnext_tiny
run e31c_ft_fraction 45 $P -I experiments/e31_capacity_controls.py ft_fraction
run e14b_hiba 25 $P -I experiments/e14b_finetune_multi.py hiba resnet18 convnext_tiny

# 6. Lower priority, only if time remains
for d in hiba milk10k_derm milk10k_clin; do run e19_embed_$d 10 $P -I experiments/e19_resolution_ladder.py embed $d; done
run e19_analyse_all 15 $P -I experiments/e19_resolution_ladder.py analyse
run e14b_isic2019_r18 120 $P -I experiments/e14b_finetune_multi.py isic2019_nonham resnet18

echo "\nQueue finished at $(date +%H:%M)." >> $STATUS
$P -I iteration4/make_summary.py >/dev/null 2>&1
