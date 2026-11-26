"""Metadata-only inventory of the shared native normalized partition store."""

from __future__ import annotations

import os
import re
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from heapq import nsmallest
from operator import itemgetter
from pathlib import Path
from typing import Any

from klab.corpus.prepared_shards import normalized_store_root
from klab.safe_io import symlinked

_SOURCE = re.compile(r"[a-z][a-z0-9_-]*")
_KEY = re.compile(r"[0-9a-f]{64}")
_INCOMPLETE = ".incomplete-"


@contextmanager
def _directory(path: str | Path, *, parent: int | None = None) -> Iterator[int]:
    descriptor = os.open(
        path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent
    )
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def _published(root: int, source_id: str | None, totals: dict[str, int]):
    with os.scandir(root) as sources:
        for source in sources:
            if not _SOURCE.fullmatch(source.name) or (
                source_id is not None and source.name != source_id
            ):
                continue
            with (
                _directory(source.name, parent=root) as namespace,
                os.scandir(namespace) as entries,
            ):
                for entry in entries:
                    key = entry.name.removeprefix(_INCOMPLETE)
                    if not _KEY.fullmatch(key):
                        continue
                    if entry.name.startswith(_INCOMPLETE):
                        if entry.is_dir(follow_symlinks=False):
                            totals["staging_directories"] += 1
                        continue
                    try:
                        with (
                            _directory(key, parent=namespace) as partition,
                            os.scandir(partition) as members,
                        ):
                            size = 0
                            for member in members:
                                info = member.stat(follow_symlinks=False)
                                if not stat.S_ISREG(info.st_mode):
                                    raise ValueError(
                                        f"invalid normalized member: {source.name}/{key}/{member.name}"
                                    )
                                size += info.st_size
                    except FileNotFoundError:
                        totals["disappeared_partitions"] += 1
                        continue
                    totals["published_partitions"] += 1
                    totals["retained_bytes"] += size
                    yield {
                        "artifact_id": f"{source.name}/{key}",
                        "source_id": source.name,
                        "computation_key": key,
                        "retained_bytes": size,
                    }


def normalized_inventory(
    work_root: Path, *, source_id: str | None, limit: int, after: str | None
) -> dict[str, Any]:
    """Count visible published bytes and return one bounded page of partition keys."""
    if source_id is not None and not _SOURCE.fullmatch(source_id):
        raise ValueError("invalid source ID")
    if not 1 <= limit <= 1000:
        raise ValueError("artifact limit must be between 1 and 1000")
    if after is not None:
        source, _, key = after.partition("/")
        if (
            not _SOURCE.fullmatch(source)
            or not _KEY.fullmatch(key)
            or (source_id is not None and source_id != source)
        ):
            raise ValueError("artifact cursor must be a source ID and computation key")
    root = normalized_store_root(work_root)
    if symlinked(root):
        raise ValueError("symlinked normalized store")
    totals = {
        "published_partitions": 0,
        "retained_bytes": 0,
        "staging_directories": 0,
        "disappeared_partitions": 0,
    }
    page = []
    if root.exists():
        with _directory(root) as directory:
            page = nsmallest(
                limit + 1,
                (
                    row
                    for row in _published(directory, source_id, totals)
                    if after is None or row["artifact_id"] > after
                ),
                key=itemgetter("artifact_id"),
            )
    return {
        "artifact_type": "normalized_source_partition",
        "verification": "advisory",
        "source_id": source_id,
        **totals,
        "partitions": page[:limit],
        "next_after": page[limit - 1]["artifact_id"] if len(page) > limit else None,
    }
