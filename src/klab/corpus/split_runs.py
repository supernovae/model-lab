"""Immutable bounded sort runs and durable completion for split preparation."""

from __future__ import annotations

import hashlib
import heapq
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.admission_inspection_cache import InspectionCache, capture
from klab.result_publication import sync_directory
from klab.training.manifest import sha256_file

BUFFER_BYTES = 32 * 1024**2  # Sort buffer, never a limit on the workload.
FAN_IN = 32


def rows(path):
    with Path(path).open("rb") as stream:
        for line in stream:
            yield json.loads(line)


def write_rows(path, records):
    return _write_raw(path, (canonical_json(record) + b"\n" for record in records))


def _write_raw(path, encoded):
    """An incomplete writer never publishes a completed file."""
    path = Path(path)
    partial = path.with_name(path.name + ".partial-" + uuid4().hex)
    digest, count = hashlib.sha256(), 0
    with partial.open("xb") as stream:
        for raw in encoded:
            stream.write(raw)
            digest.update(raw)
            count += 1
        stream.flush()
        os.fsync(stream.fileno())
    os.link(partial, path)
    partial.unlink()
    sync_directory(path.parent)
    return {
        "path": str(path),
        "sha256": digest.hexdigest(),
        "rows": count,
        "bytes": path.stat().st_size,
    }


def external_sort(
    records, directory, *, key=lambda row: row, buffer_bytes=BUFFER_BYTES
):
    """Bounded fan-in merge; sorting never needs a corpus-sized Python object."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    paths, batch, size, run, published = [], [], 0, 0, {}
    for record in records:
        raw = canonical_json(record) + b"\n"
        batch.append((key(record), raw))
        size += len(raw)
        if size >= buffer_bytes:
            batch.sort(key=lambda item: item[0])
            path = directory / f"run-{run:06d}.jsonl"
            published[path] = _write_raw(path, (item[1] for item in batch))
            paths.append(path)
            batch, size, run = [], 0, run + 1
    if batch or not paths:
        batch.sort(key=lambda item: item[0])
        path = directory / f"run-{run:06d}.jsonl"
        published[path] = _write_raw(path, (item[1] for item in batch))
        paths.append(path)
    generation = 0
    initial_runs = len(paths)
    while len(paths) > 1:
        merged = []
        for start in range(0, len(paths), FAN_IN):
            selected = paths[start : start + FAN_IN]
            output = directory / f"merge-{generation:03d}-{start:06d}.jsonl"
            with ExitStack() as stack:
                streams = [stack.enter_context(path.open("rb")) for path in selected]
                iterators = [
                    ((key(json.loads(line)), line) for line in stream)
                    for stream in streams
                ]
                merged_records = heapq.merge(*iterators, key=lambda item: item[0])
                published[output] = _write_raw(
                    output, (item[1] for item in merged_records)
                )
            merged.append(output)
        paths, generation = merged, generation + 1
    return {
        **published[paths[0]],
        "sort_work": {
            "initial_runs": initial_runs,
            "merge_generations": generation,
            "logical_bytes_written": sum(item["bytes"] for item in published.values()),
        },
    }


def _stamp(path):
    path = Path(path)
    if any(part.is_symlink() for part in (path, *path.parents)) or not path.is_file():
        raise ValueError("unsafe split checkpoint file")
    info = path.stat()
    return [info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns]


class SplitStages:
    """Fixed native split phases, not a scheduler or execution-authority grant.

    The existing native proof store authenticates resumed phase completion and
    unchanged inputs/outputs. The append-only manifest chain is an audit trail,
    not an execution-authority grant. Invalidated work is recomputed.
    """

    def __init__(self, directory, request, *, work_root=None):
        self.request = request
        self.identity = canonical_digest(request)
        self.root = Path(directory) / self.identity
        self.root.mkdir(parents=True, exist_ok=True)
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError("symlink in split checkpoint root")
        self.journal = self.root / "completion.jsonl"
        self.completed, self.previous, self.reused = {}, None, []
        self.authenticated = set()
        self.cache = InspectionCache(work_root or self.root)
        self.cache.directory = self.root / "sealed"
        self.cache.authority = {
            **self.cache.authority,
            "streaming_producer": request.get("producer"),
        }
        self.sequence = 0
        # One atomic immutable record per commit survives an interrupted append.
        # The JSONL journal is a readable mirror, never the recovery authority.
        for path in sorted(self.root.glob("completion-*.json")):
            for event in rows(path):
                body = event["record"]
                digest = canonical_digest(body)
                if event["sha256"] != digest or body["previous"] != self.previous:
                    raise ValueError("corrupt split completion chain")
                if body["request"] != self.identity:
                    raise ValueError("split checkpoint request mismatch")
                self._remember(body["phase"], body["result"])
                proof = self.cache.load(
                    "split-phase",
                    {
                        "request": self.identity,
                        "completion": digest,
                    },
                    limit=4 * 1024**2,
                )
                if proof is not None and proof[0] == body["result"]:
                    self.authenticated.add(body["phase"])
                self.previous = digest
                self.sequence += 1

    def _remember(self, name, result):
        if name in self.completed:
            keys = list(self.completed)
            for key in keys[keys.index(name) :]:
                del self.completed[key]
                self.authenticated.discard(key)
        self.completed[name] = result

    def completed_phase(self, name):
        """Read an authenticated unchanged phase without starting any work."""
        prior = self.completed.get(name)
        if prior is not None:
            try:
                valid = all(
                    _stamp(item["path"]) == item["stamp"] for item in prior["files"]
                )
            except OSError, ValueError:
                valid = False
            if valid and name in self.authenticated:
                return prior
        return None

    def phase(self, name, produce):
        prior = self.completed_phase(name)
        if prior is not None:
            self.reused.append(name)
            return prior
        directory = self.root / (name + "-" + uuid4().hex)
        directory.mkdir()
        started = time.monotonic()
        with (self.root / "phase-events.jsonl").open("ab") as stream:
            stream.write(
                canonical_json(
                    {"kind": "phase_started", "phase": name, "monotonic": started}
                )
                + b"\n"
            )
            stream.flush()
            os.fsync(stream.fileno())
        result = produce(directory)
        result["files"] = [
            {**item, "stamp": _stamp(item["path"])} for item in result["files"]
        ]
        body = {
            "previous": self.previous,
            "request": self.identity,
            "phase": name,
            "result": result,
            "elapsed_seconds": time.monotonic() - started,
        }
        digest = canonical_digest(body)
        event = {"record": body, "sha256": digest}
        completion = self.root / f"completion-{self.sequence:06d}-{digest}.json"
        write_rows(completion, [event])
        dependencies = capture(
            [
                completion,
                *[item["path"] for item in result["files"]],
                *self.request.get("inputs", {}),
            ]
        )
        self.cache.publish(
            "split-phase",
            {"request": self.identity, "completion": digest},
            result,
            dependencies,
            limit=4 * 1024**2,
        )
        raw = canonical_json(event) + b"\n"
        with self.journal.open("ab") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        sync_directory(self.root)
        self._remember(name, result)
        self.authenticated.add(name)
        self.previous = digest
        self.sequence += 1
        return result


def file_record(path):
    path = Path(path)
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
