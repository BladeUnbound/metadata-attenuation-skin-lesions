#!/bin/zsh
# Stage 4: waits for stage 3 (pid $1), then reruns E55 (summary-loop fix, STATUS 2026-10-29) and regenerates reports.
cd "/Users/paradox/Desktop/CV EL/research/experiments"
PY=(../../novelty/.venv/bin/python -I)
S="../iteration6/STATUS.md"
L="../iteration6/logs"
if [[ -n "$1" ]]; then while kill -0 $1 2>/dev/null; do sleep 30; done; fi
echo "\n## Stage-4 queue started $(date '+%Y-%m-%d %H:%M %Z')\n| job | start | end | exit |\n|---|---|---|---|" >> $S
job() { local name=$1; shift; local st=$(date +%H:%M); "$PY[@]" "$@" > $L/$name.log 2>&1; local rc=$?; echo "| $name | $st | $(date +%H:%M) | $rc |" >> $S; }
job E55_rerun e55_shift_rules.py
job report_e50_final e50_report.py
job report_iter6_final iteration6_summary.py
echo "Stage-4 queue finished $(date '+%Y-%m-%d %H:%M %Z')" >> $S
