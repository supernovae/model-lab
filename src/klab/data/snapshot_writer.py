"""Snapshot acquisition/publication only; never used to read prepared inputs.

Published snapshots are independently authenticated by sources.verify_snapshot.
This module is intentionally outside the preparation implementation fingerprint.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING

from klab.canonical import canonical_digest, canonical_json
from klab.corpus import acquisition as forge
from klab.engram.packs import _rename_noreplace
from klab.result_publication import sync_directory
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.data.sources import DatasetSource

# Operational bounds, independent of dataset identity and selection.
_SNAPSHOT_BATCH_RECORDS = 256
_SNAPSHOT_BATCH_BYTES = 8 * 1024 * 1024
_SNAPSHOT_BATCH_SECONDS = 0.25


def _admit(work: Path, cache: Path, source: DatasetSource, *, growth: int = 0) -> int:
    headroom = source.resources.max_work_bytes
    for path in (work, cache):
        usage = shutil.disk_usage(path)
        headroom = min(headroom, usage.free - source.resources.min_free_bytes)
        if usage.free < source.resources.min_free_bytes + growth:
            raise ValueError("resource bound: insufficient free bytes")
        if hasattr(os, "statvfs"):
            stat = os.statvfs(path)
            if stat.f_files and stat.f_favail < 16:
                raise ValueError("resource bound: insufficient free inodes")
    size = sum(p.stat().st_size for p in work.rglob("*") if p.is_file())
    if size + growth > source.resources.max_work_bytes:
        raise ValueError("resource bound: max_work_bytes")
    return min(headroom, source.resources.max_work_bytes - size)


def write_snapshot(
    lock: dict,
    source: DatasetSource,
    output: Path,
    cache: Path,
    work: Path,
    resume: bool,
) -> Path:
    from klab.data.sources import (
        count_snapshot_event,
        forge_bindings,
        seal_snapshot_payload,
        snapshot_counts,
        snapshot_decision,
        stream_source,
        verify_snapshot,
    )

    if source.kind == "forge_files" and forge_bindings(source) != lock["provenance"]:
        raise ValueError("Forge snapshot changed since lock")
    publication = work / "publication"
    if publication.exists():
        # An interrupted export must be discarded before resource admission;
        # otherwise its partial duplicate bytes could prevent any retry.
        if publication.is_symlink() or not publication.is_dir():
            raise ValueError("unsafe publication directory")
        for child in publication.iterdir():
            forge._regular(child)
            if child.name not in {
                "manifest.json",
                "events.jsonl",
                "excluded.jsonl",
                *(s + ".jsonl" for s in source.splits),
            }:
                raise ValueError("unexpected publication file")
        for child in publication.iterdir():
            child.unlink()
    _admit(work, cache, source)
    journal = work / "journal.sqlite"
    if resume:
        forge._regular(journal)
    elif journal.exists():
        raise FileExistsError(journal)
    db = sqlite3.connect(journal)
    try:
        db.execute("PRAGMA synchronous=FULL")
        db.execute("PRAGMA temp_store=FILE")
        if resume:
            if db.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise ValueError("journal corruption")
            if db.execute("SELECT identity FROM binding").fetchall() != [
                (lock["lock_sha256"],)
            ]:
                raise ValueError("resume lock mismatch")
        else:
            db.execute("CREATE TABLE binding (identity TEXT NOT NULL)")
            db.execute("INSERT INTO binding VALUES (?)", (lock["lock_sha256"],))
            db.execute(
                "CREATE TABLE events (id INTEGER PRIMARY KEY, payload BLOB NOT NULL)"
            )
            db.commit()
        # Keep the index in the accounted work filesystem, not SQLite's system
        # temporary directory. It is reconstructed from the authenticated prefix.
        db.execute(
            "CREATE TABLE IF NOT EXISTS seen (digest TEXT, text TEXT, split TEXT, PRIMARY KEY(digest, split))"
        )
        db.execute("DELETE FROM seen")
        db.commit()
        saved = db.execute("SELECT count(*) FROM events").fetchone()[0]
        index = total_bytes = 0
        splits = {}
        pending = pending_bytes = allowance = 0
        started = monotonic()

        def flush():
            nonlocal pending, pending_bytes, allowance, started
            # Observe the live rollback journal before committing, as well as
            # the durable database afterwards. Never publish pending counters.
            _admit(work, cache, source)
            db.commit()
            _admit(work, cache, source)
            pending = pending_bytes = allowance = 0
            started = monotonic()

        for split in source.dedup.priority:
            counts = snapshot_counts()
            iterator = iter(
                stream_source(source, split, cache, inventory=lock["provenance"])
            )
            stop = "source_exhausted"
            try:
                while True:
                    if (
                        source.selection.mode == "bounded"
                        and counts["count"] >= source.selection.documents[split]
                    ):
                        stop = "document_target"
                        break
                    try:
                        row = next(iterator)
                    except StopIteration:
                        break
                    if (
                        counts["source_records_consumed"]
                        >= source.resources.max_source_records
                    ):
                        raise ValueError(
                            "resource bound: max_source_records; journal retained"
                        )
                    raw_size = len(canonical_json(row))
                    if raw_size > source.resources.max_record_bytes:
                        raise ValueError("resource bound: max_record_bytes")
                    # Conservative existing per-row growth estimate includes
                    # index and rollback pages. No Python list retains rows.
                    growth = 4 * raw_size + 32768
                    if pending and (
                        pending >= _SNAPSHOT_BATCH_RECORDS
                        or pending_bytes + growth > allowance
                        or monotonic() - started >= _SNAPSHOT_BATCH_SECONDS
                    ):
                        flush()
                    if not pending:
                        headroom = _admit(work, cache, source, growth=growth)
                        allowance = min(_SNAPSHOT_BATCH_BYTES, headroom)
                        started = monotonic()
                    event = snapshot_decision(
                        db, source, split, counts["source_records_consumed"], row
                    )
                    event_bytes = (
                        len(event["text"].encode())
                        if event["reason"] == "selected"
                        else 0
                    )
                    if total_bytes + event_bytes > source.resources.max_text_bytes:
                        db.rollback()
                        raise ValueError(
                            "resource bound: max_text_bytes; journal retained"
                        )
                    payload = canonical_json(event)
                    if index < saved:
                        previous = db.execute(
                            "SELECT payload FROM events WHERE id=?", (index,)
                        ).fetchone()
                        if previous != (payload,):
                            raise ValueError("resume journal/source prefix mismatch")
                    else:
                        db.execute("INSERT INTO events VALUES (?, ?)", (index, payload))
                    pending += 1
                    pending_bytes += growth
                    index += 1
                    counts.setdefault("text_bytes_limit", 0)
                    count_snapshot_event(counts, event)
                    total_bytes += event_bytes if event["reason"] == "selected" else 0
                    if (
                        pending >= _SNAPSHOT_BATCH_RECORDS
                        or pending_bytes >= allowance
                        or monotonic() - started >= _SNAPSHOT_BATCH_SECONDS
                    ):
                        flush()
                if pending:
                    flush()
            finally:
                if hasattr(iterator, "close"):
                    iterator.close()
            splits[split] = {
                **counts,
                "stop_reason": stop,
                "source_exhausted": stop == "source_exhausted",
            }
        if index < saved:
            raise ValueError("resume source ended before committed journal")
        publication.mkdir(exist_ok=True)

        export_allowance = 0
        export_checked = monotonic()

        def write_record(handle, payload):
            nonlocal export_allowance, export_checked
            if (
                len(payload) > export_allowance
                or monotonic() - export_checked >= _SNAPSHOT_BATCH_SECONDS
            ):
                headroom = _admit(work, cache, source, growth=len(payload))
                export_allowance = min(_SNAPSHOT_BATCH_BYTES, headroom)
                export_checked = monotonic()
            handle.write(payload)
            export_allowance -= len(payload)

        for split in source.splits:
            with (publication / f"{split}.jsonl").open("xb", buffering=0) as handle:
                for (payload,) in db.execute("SELECT payload FROM events ORDER BY id"):
                    event = json.loads(payload)
                    if event["split"] == split and event["reason"] == "selected":
                        write_record(
                            handle,
                            canonical_json(
                                {"ordinal": event["ordinal"], "text": event["text"]}
                            )
                            + b"\n",
                        )
                handle.flush()
                os.fsync(handle.fileno())
            splits[split].update(
                path=f"{split}.jsonl",
                sha256=sha256_file(publication / f"{split}.jsonl"),
            )
        with (publication / "events.jsonl").open("xb", buffering=0) as handle:
            for (payload,) in db.execute("SELECT payload FROM events ORDER BY id"):
                write_record(handle, payload + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        with (publication / "excluded.jsonl").open("xb", buffering=0) as handle:
            excluded_count = 0
            for (payload,) in db.execute("SELECT payload FROM events ORDER BY id"):
                event = json.loads(payload)
                if event["reason"] != "selected":
                    write_record(
                        handle,
                        canonical_json(
                            {k: event[k] for k in ("split", "ordinal", "reason")}
                        )
                        + b"\n",
                    )
                    excluded_count += 1
            handle.flush()
            os.fsync(handle.fileno())
        manifest = seal_snapshot_payload(
            {
                "snapshot_source_sha256": canonical_digest(lock["source"]),
                "excluded": {
                    "path": "excluded.jsonl",
                    "count": excluded_count,
                    "sha256": sha256_file(publication / "excluded.jsonl"),
                },
                "format": "klab-dataset-snapshot-v1",
                "lock": lock,
                "splits": splits,
                "events_sha256": sha256_file(publication / "events.jsonl"),
                "source_exhausted": all(s["source_exhausted"] for s in splits.values()),
            },
            "manifest_sha256",
        )
        _admit(work, cache, source, growth=len(canonical_json(manifest)) + 1024)
        forge._write_json(publication / "manifest.json", manifest)
        _admit(work, cache, source)
        verify_snapshot(publication / "manifest.json", verification_mode="cold")
        sync_directory(publication)
        _rename_noreplace(publication, output)
        sync_directory(output.parent)
        return output / "manifest.json"
    finally:
        db.close()
