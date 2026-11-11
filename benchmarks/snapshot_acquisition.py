"""Offline CPU snapshot benchmark; run against each checkout with identical arguments.

Uses the maintained source fixture builder. No Hub calls or downloads occur.
"""

from __future__ import annotations

import argparse
import json
import resource
import sqlite3
import sys
import tempfile
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
from test_dataset_sources import acquire, declaration, inventory

from klab.data import snapshot_writer, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=5000)
    parser.add_argument("--baseline-source", type=Path)
    args = parser.parse_args()
    if args.baseline_source:
        # Evaluate the exact saved base module, retaining its package context.
        exec(  # noqa: S102 -- explicitly selected trusted base code
            compile(
                args.baseline_source.read_bytes(), str(args.baseline_source), "exec"
            ),
            sources.__dict__,
        )
    payload = declaration()
    payload["resources"].update(
        max_source_records=args.records,
        max_text_bytes=32 * 1024**2,
        max_work_bytes=128 * 1024**2,
    )
    metrics = {"commits": 0, "capacity_checks": 0}
    connect, admit = sqlite3.connect, snapshot_writer._admit

    def trace(sql):
        if sql == "COMMIT":
            metrics["commits"] += 1

    def traced_connect(*args, **kwargs):
        db = connect(*args, **kwargs)
        db.set_trace_callback(trace)
        return db

    def observed_admit(*args, **kwargs):
        metrics["capacity_checks"] += 1
        return admit(*args, **kwargs)

    def stream(source, split, cache, **kwargs):
        count = args.records if split == "train" else 32
        for i in range(count):
            yield {"text": f"{split} {i} " + "example text " * 40}

    with (
        tempfile.TemporaryDirectory() as directory,
        patch.object(sources, "_hub_identity", inventory),
        patch.object(sources, "stream_source", stream),
        patch.object(snapshot_writer, "_admit", observed_admit),
        patch.object(sqlite3, "connect", traced_connect),
    ):
        started = perf_counter()
        manifest = acquire(Path(directory), payload)
        metrics.update(
            seconds=perf_counter() - started,
            peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            owned_bytes=sum(
                p.stat().st_size for p in Path(directory).rglob("*") if p.is_file()
            ),
            manifest_sha256=sources.verify_snapshot(manifest)["manifest_sha256"],
            records=args.records + 32,
        )
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
