"""Shared experiment logging: every run writes one JSON record to research/results/.

A record holds the experiment ID, hypothesis, data provenance, configuration,
seed, results, runtime, peak memory and an interpretation, so the registry in
research/EXPERIMENTS.md can be rebuilt from the JSON files alone.
"""
import json
import platform
import resource
import time
from datetime import datetime, timezone
from pathlib import Path

RESEARCH = Path(__file__).resolve().parents[1]
RESULTS = RESEARCH / "results"
PROJECT = RESEARCH.parent


class Experiment:
    def __init__(self, exp_id, hypothesis, data, config, seed):
        self.record = {
            "id": exp_id,
            "hypothesis": hypothesis,
            "data": data,
            "config": config,
            "seed": seed,
            "machine": f"{platform.machine()} {platform.platform()}",
            "started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self._t0 = time.perf_counter()

    def finish(self, results, interpretation, failures=None, next_step=None):
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
        self.record.update(
            results=results,
            interpretation=interpretation,
            failures=failures or [],
            next_step=next_step,
            runtime_s=round(time.perf_counter() - self._t0, 2),
            peak_rss_mb=round(peak_rss / 2**20, 1),
        )
        RESULTS.mkdir(exist_ok=True)
        path = RESULTS / f"{self.record['id']}.json"
        path.write_text(json.dumps(self.record, indent=2, default=float))
        print(json.dumps(self.record, indent=2, default=float))
        return path
