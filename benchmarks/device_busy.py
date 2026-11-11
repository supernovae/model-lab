"""Sample per-GPU busy time of one process and draw a busy/idle timeline.

Linux DRM drivers publish per-client engine counters in ``/proc/PID/fdinfo``.
This reads them for a training process every 50 ms, without touching the
process, and reports for each device how much of the time it was busy and where
the idle gaps are. It was written for Intel Xe (Arc) devices, whose compute
engine is ``ccs``; pass ``--engine`` for other drivers.

    python benchmarks/device_busy.py record PID trace.json   # until PID exits
    python benchmarks/device_busy.py show trace.json

A device that is always busy is the bottleneck. Regular idle gaps mean the
device is waiting: for the host, for another device, or for I/O.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from itertools import pairwise
from pathlib import Path


def _read(pid: int) -> dict[str, int] | None:
    totals: dict[str, int] = {}
    try:
        entries = os.listdir(f"/proc/{pid}/fdinfo")
    except FileNotFoundError:
        return None
    seen: set[tuple[str, str]] = set()
    for entry in entries:
        try:
            text = Path(f"/proc/{pid}/fdinfo/{entry}").read_text()
        except OSError:
            continue
        fields = {
            key.strip(): value.strip()
            for key, _, value in (line.partition(":") for line in text.splitlines())
        }
        device, client = fields.get("drm-pdev"), fields.get("drm-client-id")
        if device is None or (device, client or entry) in seen:
            continue
        seen.add((device, client or entry))
        for key, value in fields.items():
            if key.startswith(("drm-cycles-", "drm-total-cycles-")):
                name = f"{device}|{key}"
                totals[name] = totals.get(name, 0) + int(value.split()[0])
    return totals


def record(pid: int, output: Path, period: float) -> None:
    rows = []
    while (sample := _read(pid)) is not None:
        rows.append([time.time(), sample])
        time.sleep(period)
    output.write_text(json.dumps(rows))


def show(trace: Path, engine: str, width: int) -> None:
    rows = [row for row in json.loads(trace.read_text()) if row[1]]
    devices = sorted({key.split("|")[0] for _, sample in rows for key in sample})
    for device in devices:
        used, total = (
            f"{device}|drm-cycles-{engine}",
            f"{device}|drm-total-cycles-{engine}",
        )
        ratios = []
        for (_, before), (_, after) in pairwise(rows):
            elapsed = after.get(total, 0) - before.get(total, 0)
            ratios.append(
                (after.get(used, 0) - before.get(used, 0)) / elapsed
                if elapsed > 0
                else 0.0
            )
        if not ratios or max(ratios) == 0:
            continue
        # The counters' ratio saturates below one; scale by the observed ceiling.
        ceiling = sorted(ratios)[int(len(ratios) * 0.95)]
        busy = [ratio > 0.5 * ceiling for ratio in ratios]
        first = busy.index(True)
        last = len(busy) - busy[::-1].index(True)
        active = busy[first:last]
        print(
            f"{device}: busy {sum(active) / len(active):.1%} between its first and "
            f"last busy sample ({len(active) * (rows[1][0] - rows[0][0]):.0f} s)"
        )
        line = "".join("#" if value else "." for value in busy)
        for start in range(0, len(line), width):
            print("   ", line[start : start + width])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    recording = commands.add_parser("record")
    recording.add_argument("pid", type=int)
    recording.add_argument("output", type=Path)
    recording.add_argument("--period", type=float, default=0.05)
    showing = commands.add_parser("show")
    showing.add_argument("trace", type=Path)
    showing.add_argument("--engine", default="ccs")
    showing.add_argument("--width", type=int, default=100)
    args = parser.parse_args()
    if args.command == "record":
        record(args.pid, args.output, args.period)
    else:
        show(args.trace, args.engine, args.width)


if __name__ == "__main__":
    main()
