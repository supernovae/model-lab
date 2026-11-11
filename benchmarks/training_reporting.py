"""Bounded offline comparison of durable per-update metric writes.

Run: uv run --locked --no-sync python benchmarks/training_reporting.py
Uses disposable SQLite stores, no model, dataset, accelerator or subprocess.
This isolates store overhead; it does not estimate end-to-end training speedup.
"""

from __future__ import annotations

import json
import statistics
import tempfile
import time
from contextlib import nullcontext
from pathlib import Path

from klab.training.metrics import ExperimentStore


def main() -> None:
    samples: dict[str, list[float]] = {"per_call": [], "session": []}
    for repeat in range(5):
        # Alternate order to reduce warm-up/order bias.
        for mode in samples if repeat % 2 == 0 else reversed(samples):
            with tempfile.TemporaryDirectory() as directory:
                store = ExperimentStore(Path(directory))
                store.create_run("run", {}, {})
                started = time.perf_counter()
                with store.session() if mode == "session" else nullcontext():
                    for step in range(1000):
                        store.log_metrics(
                            "run",
                            step,
                            step * 4096,
                            step * 0.1,
                            {
                                "train/loss": 1.0,
                                "performance/step_seconds": 0.1,
                                "performance/tokens_per_second": 40960.0,
                            },
                        )
                samples[mode].append(time.perf_counter() - started)
                records = store.export_records()["records"]
                records += store.export_records(after_sequence=1000)["records"]
                assert len(records) == 1001
                assert [record["sequence"] for record in records] == list(
                    range(1, 1002)
                )
    print(
        json.dumps(
            {
                "updates_per_sample": 1000,
                "seconds": samples,
                "median_seconds": {
                    key: statistics.median(value) for key, value in samples.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
