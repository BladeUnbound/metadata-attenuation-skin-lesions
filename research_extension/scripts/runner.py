"""Sequential, resumable task runner for research_extension.

Tasks are defined in configs/schedule.json as {"id", "phase", "cmd" (list, run from scripts/), "deps" (task ids)}.
A task runs only when all its dependencies have a success marker. Its stdout/stderr go to results/logs/<id>.log, and
one row (start, end, exit, log) is appended to EXECUTION_LOG.md. A successful task leaves results/markers/<id>.ok and
is skipped on re-runs unless --force is given.

Usage: python runner.py <task-id> [<task-id> ...] [--force]
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = str(ROOT.parent / "novelty/.venv/bin/python")


def now():
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")


def main(ids, force):
    sched = {t["id"]: t for t in json.loads((ROOT / "configs/schedule.json").read_text())["tasks"]}
    (ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
    (ROOT / "results/markers").mkdir(parents=True, exist_ok=True)
    for tid in ids:
        t = sched[tid]
        mark = ROOT / f"results/markers/{tid}.ok"
        if mark.exists() and not force:
            print(f"skip {tid} (done)")
            continue
        missing = [d for d in t.get("deps", []) if not (ROOT / f"results/markers/{d}.ok").exists()]
        if missing:
            print(f"BLOCKED {tid}: missing {missing}")
            return 2
        log = ROOT / f"results/logs/{tid}.log"
        start = now()
        cmd = [PY if c == "PY" else c for c in t["cmd"]]
        with open(log, "w") as fh:
            rc = subprocess.call(cmd, cwd=HERE, stdout=fh, stderr=subprocess.STDOUT)
        end = now()
        with open(ROOT / "EXECUTION_LOG.md", "a") as fh:
            fh.write(f"| {end} | {t['phase']} | task {tid} exit {rc} | start {start}; log results/logs/{tid}.log |\n")
        if rc == 0:
            mark.write_text(end + "\n")
        else:
            print(f"FAILED {tid} (exit {rc}); see {log}")
            return rc
        print(f"done {tid} ({start} -> {end})")
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sys.exit(main(args, "--force" in sys.argv))
