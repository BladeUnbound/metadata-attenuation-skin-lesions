#!/bin/zsh
# Overnight sequential queue (27 -> 28 Oct 2026). One GPU job at a time; no new job starts after 05:45.
# Each job logs to overnight/logs/<name>.log and appends a row to overnight/STATUS.md.
# A failed job is recorded and the queue continues with the next one.
cd "$HOME/Desktop/CV EL/research" || exit 1
P=../novelty/.venv/bin/python
STATUS=overnight/STATUS.md
DEADLINE=$(date -j -v+1d -f "%H:%M:%S" "05:45:00" +%s)
[ "$(date +%H)" -lt 6 ] && DEADLINE=$(date -j -f "%H:%M:%S" "05:45:00" +%s)

echo "# Overnight queue status\n\nStarted $(date '+%Y-%m-%d %H:%M'); deadline $(date -r $DEADLINE '+%Y-%m-%d %H:%M').\n\n| Job | Start | End | Exit |\n|---|---|---|---|" > $STATUS

run() {
  local name=$1; shift
  if [ "$(date +%s)" -ge "$DEADLINE" ]; then echo "| $name | skipped: deadline | | |" >> $STATUS; return; fi
  local start=$(date +%H:%M)
  "$@" > overnight/logs/$name.log 2>&1
  local rc=$?
  echo "| $name | $start | $(date +%H:%M) | $rc |" >> $STATUS
}

# Wait for jobs already running: E14 chain (39594), E19 chain (40957), ISIC downloads.
while kill -0 39594 2>/dev/null || kill -0 40957 2>/dev/null || pgrep -f fetch_isic_collection >/dev/null; do sleep 30; done
echo "\nPrior jobs finished at $(date +%H:%M).\n" >> $STATUS

# 1. Second-wave datasets: embeddings (GPU), then pre-registered analysis (CPU)
for d in hiba msk2020 milk10k_derm milk10k_clin; do run embed_$d $P -I experiments/e13_replication.py $d embed; done
for d in hiba msk2020 milk10k_derm milk10k_clin; do run analyse_$d $P -I experiments/e13_replication.py $d analyse; done
run e22_paired_modality $P -I experiments/e22_paired_modality.py
run e17_all $P -I experiments/e17_retained_fraction.py ham10000 isic2020 isic2019_nonham pad_ufes hiba msk2020 milk10k_derm milk10k_clin
run e21_wave2 $P -I experiments/e21_complete_case.py hiba msk2020 milk10k_derm milk10k_clin

# 2. Foundation-model rung (ViT-L/16 SWAG), smallest datasets first; ISIC 2020 (33k) last
for d in pad_ufes msk2020 hiba milk10k_clin milk10k_derm ham10000 isic2019_nonham isic2020; do
  run e18_embed_$d $P -I experiments/e18_foundation_rung.py embed $d
done
run e18_analyse $P -I experiments/e18_foundation_rung.py analyse

# 3. Fine-tuning on further datasets (checkpointed per fold; resumes if interrupted)
run e14b_pad $P -I experiments/e14b_finetune_multi.py pad_ufes resnet18 convnext_tiny
run e14b_hiba $P -I experiments/e14b_finetune_multi.py hiba resnet18 convnext_tiny

# 4. Resolution ladder on the remaining datasets, then a full E19 analysis
for d in hiba milk10k_derm milk10k_clin isic2020; do run e19_embed_$d $P -I experiments/e19_resolution_ladder.py embed $d; done
run e19_analyse_all $P -I experiments/e19_resolution_ladder.py analyse

# 5. Long fine-tuning runs, only if time remains
run e14b_isic2019_r18 $P -I experiments/e14b_finetune_multi.py isic2019_nonham resnet18
run e14b_milk10k_derm $P -I experiments/e14b_finetune_multi.py milk10k_derm resnet18 convnext_tiny

echo "\nQueue finished at $(date +%H:%M)." >> $STATUS
