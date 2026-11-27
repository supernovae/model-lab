"""Reorder verified mixture records without selecting or encoding documents."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import OrderedDict
from itertools import pairwise
from pathlib import Path

from klab.corpus.mixture_index import MAX_ROW_BYTES
from klab.corpus.mixture_schedule import (
    SCHEDULE_RANGE,
    TRAIN_SCHEDULE,
    VerifiedSchedule,
    open_verified_schedule_member,
)
from klab.corpus.release_verify import scratch_root
from klab.data.verification import HashingWriter
from klab.workspace_preflight import check_storage, require_storage

_OPEN_GROUPS = 16


def order_by_source(
    records: Path,
    base: dict,
    schedule: VerifiedSchedule,
    destination: Path,
    source_strata: dict[str, str],
    exposure_limits: dict[str, int],
) -> tuple[dict, dict]:
    """Stably group records inside each stratum and exposure's selection window."""
    scratch = scratch_root()
    size = base["train_records_bytes"] + schedule.schedule.size_bytes
    inode_bound = min(
        base["records"],
        sum(exposure_limits[stratum] for stratum in source_strata.values()),
    )
    if scratch.stat().st_dev == destination.stat().st_dev:
        require_storage(
            [
                check_storage(
                    scratch, projected_bytes=2 * size, projected_inodes=inode_bound + 4
                )
            ]
        )
    else:
        require_storage(
            [
                check_storage(
                    scratch, projected_bytes=size, projected_inodes=inode_bound + 2
                ),
                check_storage(destination, projected_bytes=size, projected_inodes=2),
            ]
        )
    with tempfile.TemporaryDirectory(prefix="source-sequence-", dir=scratch) as temp:
        groups, transitions = _partition(
            records, base, schedule, Path(temp), source_strata, exposure_limits
        )
        windows = {}
        for key, ordinal in groups.items():
            windows.setdefault(key[:2], ordinal)
        ordered = sorted(groups, key=lambda key: (windows[key[:2]], groups[key]))
        after = sum(left[2] != right[2] for left, right in pairwise(ordered))
        with (
            (destination / records.name).open("xb") as record_output,
            (destination / TRAIN_SCHEDULE).open("xb") as schedule_output,
        ):
            record_writer = HashingWriter(record_output)
            schedule_writer = HashingWriter(schedule_output)
            for key in ordered:
                with (Path(temp) / str(groups[key])).open("rb") as group:
                    while item := group.read(SCHEDULE_RANGE.size):
                        schedule_writer.write(item)
                        record_writer.write(group.readline(MAX_ROW_BYTES + 1))
            record_proof = record_writer.seal(destination / records.name)
            schedule_proof = schedule_writer.seal(destination / TRAIN_SCHEDULE)
    if (
        record_proof.size_bytes != base["train_records_bytes"]
        or schedule_proof.size_bytes != schedule.schedule.size_bytes
    ):
        raise ValueError("source ordering changed the mixture inventory size")
    schedule.require_unchanged()
    return (
        {
            "train_records_sha256": record_proof.sha256,
            "token_schedule": {
                **base["token_schedule"],
                "sha256": schedule_proof.sha256,
            },
        },
        {
            "groups": len(groups),
            "source_transitions_before": transitions,
            "source_transitions_after": after,
        },
    )


def _partition(records, base, schedule, directory, source_strata, exposure_limits):
    groups, streams = {}, OrderedDict()
    previous, transitions, count = None, 0, 0
    digest = hashlib.sha256()
    try:
        with (
            records.open("rb") as source,
            open_verified_schedule_member(schedule.schedule) as ranges,
        ):
            while line := source.readline(MAX_ROW_BYTES + 1):
                item = ranges.read(SCHEDULE_RANGE.size)
                if (
                    len(line) > MAX_ROW_BYTES
                    or not line.endswith(b"\n")
                    or len(item) != SCHEDULE_RANGE.size
                ):
                    raise ValueError(
                        "source ordering input records differ from the schedule"
                    )
                digest.update(line)
                row = json.loads(line)
                name, exposure = row["source_id"], row["exposure"]
                if (
                    name not in source_strata
                    or source_strata[name] != row["stratum"]
                    or type(exposure) is not int
                    or not 1 <= exposure <= exposure_limits[row["stratum"]]
                ):
                    raise ValueError(
                        "source ordering input has an invalid stratum or exposure"
                    )
                if SCHEDULE_RANGE.unpack(item)[2] != row["tokens"]:
                    raise ValueError(
                        "source ordering input token count differs from the schedule"
                    )
                if previous is not None and name != previous:
                    transitions += 1
                previous = name
                count += 1
                key = (row["stratum"], exposure, name)
                ordinal = groups.setdefault(key, len(groups))
                if ordinal not in streams:
                    if len(streams) == _OPEN_GROUPS:
                        streams.popitem(last=False)[1].close()
                    streams[ordinal] = (directory / str(ordinal)).open("ab")
                streams.move_to_end(ordinal)
                streams[ordinal].write(item)
                streams[ordinal].write(line)
            if (
                ranges.read(1)
                or count != base["records"]
                or digest.hexdigest() != base["train_records_sha256"]
            ):
                raise ValueError("source ordering input changed during partitioning")
    finally:
        for stream in streams.values():
            stream.close()
    return groups, transitions
