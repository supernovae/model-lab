"""Physical byte accounting for a verified mixture's offline dispatch."""

from __future__ import annotations

import tempfile
from itertools import groupby
from operator import itemgetter
from pathlib import Path

from klab.corpus.mixture_schedule import SCHEDULE_RANGE, VerifiedSchedule
from klab.corpus.release_verify import scratch_root
from klab.corpus.split_runs import FAN_IN, external_sort, rows
from klab.workspace_preflight import check_storage, require_storage

_SORT_BUFFER_BYTES = 2 * 1024**2
# Canonical JSON array of one u32 ordinal and two u64 positions, plus newline.
_RANGE_LINE_BOUND = 55


def plan_transfer(schedule: VerifiedSchedule) -> dict[str, object]:
    """Union physical token ranges without rereading their payload bytes."""
    records = schedule.schedule.size_bytes // SCHEDULE_RANGE.size
    interval_bytes = records * _RANGE_LINE_BOUND
    runs = max(1, (interval_bytes + _SORT_BUFFER_BYTES - 1) // _SORT_BUFFER_BYTES)
    files, generations = runs, 1
    while runs > 1:
        runs = (runs + FAN_IN - 1) // FAN_IN
        files += runs
        generations += 1
    scratch_bytes = interval_bytes * generations
    scratch = scratch_root()
    require_storage(
        [
            check_storage(
                scratch, projected_bytes=scratch_bytes, projected_inodes=files + 2
            )
        ]
    )
    width = schedule.token_dtype.itemsize
    without_content = len(schedule.tokens)
    covered_tokens = coalesced_ranges = 0
    fully_used = sum(member.size_bytes == 0 for member in schedule.tokens)
    with tempfile.TemporaryDirectory(prefix="mixture-transfer-", dir=scratch) as temp:
        ordered = external_sort(
            (
                (segment, start, start + count - 1)
                for segment, start, count in schedule.ranges()
                if count > 1
            ),
            Path(temp) / "ranges",
            buffer_bytes=_SORT_BUFFER_BYTES,
        )
        for segment, intervals in groupby(rows(ordered["path"]), key=itemgetter(0)):
            without_content -= 1
            end = -1
            covered = 0
            for _, start, stop in intervals:
                if start > end:
                    coalesced_ranges += 1
                covered += max(0, stop - max(start, end))
                end = max(stop, end)
            covered_tokens += covered
            fully_used += covered * width == schedule.tokens[segment].size_bytes

    schedule.require_unchanged()
    token_bytes = sum(member.size_bytes for member in schedule.tokens)
    index_bytes = sum(member.size_bytes for member in schedule.indexes)
    whole_bytes, whole_files = schedule.dependency_footprint()
    selected_bytes = covered_tokens * width
    return {
        "schedule_sha256": schedule.schedule.sha256,
        "segment_dtype": schedule.token_dtype.name,
        "segments": len(schedule.tokens),
        "records": records,
        "logical_target_bytes": schedule.target_count * 4,
        "whole_segments": {
            "files": whole_files,
            "bytes": whole_bytes,
            "token_bytes": token_bytes,
            "index_bytes": index_bytes,
            "manifest_bytes": whole_bytes - token_bytes - index_bytes,
            "unselected_token_bytes": token_bytes - selected_bytes,
            "segments_with_unselected_tokens": len(schedule.tokens) - fully_used,
        },
        "selected_ranges": {
            "unique_token_bytes": selected_bytes,
            "repeated_token_bytes": (schedule.target_count - records - covered_tokens)
            * width,
            "coalesced_ranges": coalesced_ranges,
            "segments_without_selected_content": without_content,
        },
        "scratch": {"planned_byte_bound": scratch_bytes, **ordered["sort_work"]},
    }
