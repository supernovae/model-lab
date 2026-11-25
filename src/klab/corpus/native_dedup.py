"""Duplicate groups, checkpointed resolution and compact native decisions."""

from __future__ import annotations

import hashlib
import mmap
import os
import sqlite3
import struct
import tempfile
from array import array
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, closing, contextmanager
from functools import partial
from itertools import chain as iter_chain
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

from klab.canonical import canonical_json
from klab.corpus.dedup_policy import Member, describe_failure, find_root, resolve_group
from klab.corpus.native_index import DEDUP_SCHEMA
from klab.corpus.progress import BuildProgress
from klab.corpus.shards import range_file_identity
from klab.corpus.view_checkpoints import ViewCheckpoints
from klab.safe_io import write_canonical_json
from klab.training.manifest import sha256_file

DEDUP_IMPLEMENTATION_FILES = (
    "dedup_policy.py",
    "native_dedup.py",
    "view_checkpoints.py",
)


def open_dedup(index: Path, database: Path) -> sqlite3.Connection:
    """A new decisions database over a read-only index, ready for `deduplicate`.

    `dedup`, the narrow table the passes read and write, lives in `database`;
    `documents` and `origins` are read from the attached index, which is never
    changed.
    """
    connection = sqlite3.connect(f"file:{quote(str(database))}", uri=True)
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(
        "ATTACH DATABASE ? AS indexed",
        (f"file:{quote(str(index / 'index.sqlite'))}?mode=ro",),
    )
    connection.executescript(
        DEDUP_SCHEMA.format(unique=" UNIQUE")
        + "INSERT INTO main.dedup (document_number,document_id,split,source_rank,"
        "document_kind,metadata_only,rights_signature) SELECT document_number,"
        "document_id,split,source_rank,document_kind,metadata_only,rights_signature "
        "FROM indexed.dedup "
        "ORDER BY document_number;"
    )
    connection.commit()
    return connection


def _duplicate_diagnostics(
    groups_path: Path,
    source_events: Iterable[Path | dict[str, Any]],
    diagnostics: Path,
    error: str,
) -> None:
    """Preserve the complete observed groups without loading them."""
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    with diagnostics.open("wb") as report:
        report.write(b'{"duplicates":[')
        with groups_path.open("rb") as rows:
            # Only record separators are literal newlines in canonical JSONL.
            tail = b""
            while block := rows.read(64 * 1024):
                report.write((tail + block[:-1]).replace(b"\n", b","))
                tail = block[-1:]
            if tail != b"\n":
                report.write(tail)
        report.write(b'],"error":' + canonical_json(error) + b',"rejected":[')
        for index, row in enumerate(rejected_rows(source_events)):
            if index:
                report.write(b",")
            report.write(row.rstrip(b"\n"))
        report.write(b"]}\n")
        report.flush()
        os.fsync(report.fileno())


_DEDUP_WINDOW_KEYS = 100_000
_DEDUP_WINDOW_BYTES = 64 * 1024**2
_DEDUP_GROUP_ROWS = 4096
_DEDUP_LINK = struct.Struct("<I")
_DEDUP_LINK_HEADER = struct.Struct("<III")


def _link_numbers(stream: BinaryIO, count: int, journal=None) -> Iterator[int]:
    """Read or copy one bounded binary member list as it is consumed."""
    while count:
        batch = min(count, _DEDUP_GROUP_ROWS)
        raw = stream.read(batch * _DEDUP_LINK.size)
        if len(raw) != batch * _DEDUP_LINK.size:
            raise ValueError("truncated duplicate group link journal")
        if journal is not None:
            journal.write(raw)
        yield from struct.unpack(f"<{batch}I", raw)
        count -= batch


def _write_large_duplicate_group(
    rows, method, digest, policy, target, links, connection, apply, reject, scratch
):
    """Write an oversized audit record and its links without holding the group."""
    chain = policy.resolution == "chain_smallest_id"
    target.write(b'{"method":' + canonical_json(method) + b',"origins":[')
    with (
        tempfile.TemporaryFile(dir=scratch) as other,
        tempfile.TemporaryFile(dir=scratch) as train,
    ):
        batch = []
        sample = []
        count = train_count = 0

        def write_ids():
            if batch:
                if count > len(batch):
                    target.write(b",")
                target.write(canonical_json(batch)[1:-1])
                batch.clear()

        def members():
            nonlocal count, train_count
            for number, identifier, split, rank, kind, signature in rows:
                count += 1
                batch.append(identifier)
                if len(batch) == _DEDUP_GROUP_ROWS:
                    write_ids()
                if len(sample) < 5:
                    sample.append(identifier)
                if chain and split == "train":
                    train.write(_DEDUP_LINK.pack(number))
                    train_count += 1
                else:
                    other.write(_DEDUP_LINK.pack(number))
                yield Member(identifier, split, rank, kind, signature)

        group = resolve_group(method, members(), policy)
        write_ids()
        target.write(
            b'],"representative":'
            + canonical_json(group.representative)
            + b',"sha256":'
            + canonical_json(digest)
            + b',"splits":'
            + canonical_json(group.splits)
            + b"}\n"
        )
        if group.error:
            reject(group, sample, count)
        train_links = train_count if group.contaminated else 0
        if links is not None:
            number = 0
            if chain:
                number = connection.execute(
                    "SELECT document_number FROM dedup WHERE document_id=?",
                    (group.representative,),
                ).fetchone()[0]
            links.write(
                _DEDUP_LINK_HEADER.pack(number, count - train_links, train_links)
            )
        other.seek(0)
        train.seek(0)
        ordinary = _link_numbers(other, count - train_count, links)
        trainees = _link_numbers(train, train_count, links)
        if not train_links:
            ordinary = iter_chain(ordinary, trainees)
            trainees = ()
        apply(group.representative, ordinary, trainees, count)


def _duplicate_key_bounds(connection, method, after):
    """Seek a batch boundary in the covering index, including unique origins."""
    predicate, values = _duplicate_key_start(method, after)
    row = connection.execute(
        f"SELECT origin FROM origins WHERE {predicate} ORDER BY origin LIMIT 1 OFFSET ?",
        (*values, _DEDUP_WINDOW_KEYS - 1),
    ).fetchone()
    return row[0] if row else None


def _duplicate_key_start(method, after):
    return (
        ("method=?", (method,))
        if after is None
        else ("method=? AND origin>?", (method, after))
    )


def _duplicate_keys(connection, method, after, end):
    predicate, values = _duplicate_key_start(method, after)
    if end is not None:
        predicate += " AND origin<=?"
        values = (*values, end)
    return connection.execute(
        f"SELECT origin FROM origins WHERE {predicate} "
        "GROUP BY origin HAVING COUNT(*)>1 ORDER BY origin",
        values,
    )


_DEDUP_DECISION_ROWS = 100_000
_DEDUP_COMPRESSION_ROWS = 100_000
_DEDUP_REPRESENTATIVE = struct.Struct("<II")
_DEDUP_DECISION = struct.Struct("<IIB")
_RANKED_COMPONENT = struct.Struct("<IIB")
_DEDUP_DROP_REASONS = (
    "metadata_only_front_matter",
    "duplicate",
    "contaminated_heldout",
)
_DEDUP_REASON_CODES = {reason: code for code, reason in enumerate(_DEDUP_DROP_REASONS)}


class _DedupJournal:
    """Receipted sparse changes to the disposable SQLite decisions table."""

    def __init__(
        self,
        checkpoints: ViewCheckpoints,
        stack: ExitStack,
        *,
        stage: str,
        filename: str,
        record: struct.Struct,
    ) -> None:
        self.root = checkpoints.root / stage
        self.identity = checkpoints.identity
        self.filename = filename
        self.record = record
        directory = self.root / "build"
        path = directory / filename
        if any(part.is_symlink() for part in (self.root, directory, path)):
            raise ValueError(f"symlinked dedup decision checkpoint: {path}")
        directory.mkdir(parents=True, exist_ok=True)
        self.stream: BinaryIO = stack.enter_context(path.open("a+b"))
        self.journal = ViewCheckpoints(self.root, self.identity)
        self.output = self.journal.add(filename, self.stream)
        self.state = self.journal.restore()

    def reset(self) -> None:
        for receipt in self.root.glob("commit-*.json"):
            receipt.unlink()
        self.stream.seek(0)
        self.stream.truncate()
        self.journal = ViewCheckpoints(self.root, self.identity)
        self.output = self.journal.add(self.filename, self.stream)
        self.state = {}

    def records(self) -> Iterator[tuple[int, ...]]:
        self.stream.seek(0)
        for block in iter(lambda: self.stream.read(16_384 * self.record.size), b""):
            yield from self.record.iter_unpack(block)


def _apply_decisions(connection, records, *, representatives):
    if representatives:
        connection.executemany(
            "UPDATE dedup SET representative_id=(SELECT p.document_id FROM dedup p "
            "WHERE p.document_number=?), drop_reason=? WHERE document_number=?",
            (
                (representative, _DEDUP_DROP_REASONS[code], number)
                for number, representative, code in records
            ),
        )
    else:
        connection.executemany(
            "UPDATE dedup SET drop_reason=? WHERE document_number=?",
            ((_DEDUP_DROP_REASONS[code], number) for number, _, code in records),
        )


@contextmanager
def _mapped_array(scratch: Path, length: int, format: str) -> Iterator[memoryview]:
    with tempfile.TemporaryFile(dir=scratch) as stream:
        stream.truncate(length * struct.calcsize(format))
        with (
            mmap.mmap(stream.fileno(), 0) as mapping,
            memoryview(mapping).cast(format) as values,
        ):
            yield values


def deduplicate(
    connection: sqlite3.Connection,
    groups_path: Path,
    *,
    policy: Any,
    diagnostics: Path,
    progress: BuildProgress,
    source_events: Iterable[Path | dict[str, Any]] = (),
    checkpoints: ViewCheckpoints | None = None,
) -> tuple[int, int]:
    """Apply the shared duplicate policy with disk-backed joins."""
    chain = policy.resolution == "chain_smallest_id"
    with ExitStack() as stack:
        decisions = (
            _DedupJournal(
                checkpoints,
                stack,
                stage="chain-decisions" if chain else "ranked-decisions",
                filename="decisions.bin",
                record=_DEDUP_DECISION,
            )
            if checkpoints
            else None
        )
        if decisions is not None:
            if decisions.journal.damaged:
                progress.record("discarded_damaged_dedup_decisions")
            report = checkpoints.root / "build" / "groups.jsonl"
            state = decisions.state
            if (
                state.get("complete")
                and not (report.is_symlink() or report.parent.is_symlink())
                and report.is_file()
            ):
                generation = range_file_identity(report)
                if sha256_file(report) == state["groups_sha256"]:
                    progress.record("reused_completed_dedup_decisions")
                    connection.execute("UPDATE dedup SET representative_id=document_id")
                    _apply_decisions(
                        connection, decisions.records(), representatives=True
                    )
                    connection.commit()
                    if range_file_identity(report) != generation:
                        raise ValueError(
                            "dedup group report changed while replaying final decisions"
                        )
                    os.link(report, groups_path)
                    return state["groups"], state["dropped"]

        parents = None
        if not chain:
            width = _document_count(connection) + 1
            parents = stack.enter_context(_mapped_array(groups_path.parent, width, "I"))
            for start in range(0, width, _DEDUP_DECISION_ROWS):
                end = min(width, start + _DEDUP_DECISION_ROWS)
                parents[start:end] = array("I", range(start, end))
        groups, report_digest = _resolve_duplicate_groups(
            connection,
            groups_path,
            policy=policy,
            diagnostics=diagnostics,
            progress=progress,
            source_events=source_events,
            checkpoints=checkpoints,
            parents=parents,
        )
        if chain:
            compression = (
                _DedupJournal(
                    checkpoints,
                    stack,
                    stage="chain-compression",
                    filename="representatives.bin",
                    record=_DEDUP_REPRESENTATIVE,
                )
                if groups and checkpoints
                else None
            )
            return groups, _finalize_chain(
                connection,
                groups,
                report_digest,
                compression,
                decisions,
                groups_path.parent,
                progress,
            )
        assert parents is not None
        return groups, _finalize_ranked(
            connection,
            parents,
            groups,
            report_digest,
            diagnostics,
            progress,
            checkpoints=checkpoints,
            decisions=decisions,
            scratch=groups_path.parent,
        )


def _resolve_duplicate_groups(
    connection: sqlite3.Connection,
    groups_path: Path,
    *,
    policy: Any,
    diagnostics: Path,
    progress: BuildProgress,
    source_events: Iterable[Path | dict[str, Any]] = (),
    checkpoints: ViewCheckpoints | None = None,
    parents: memoryview | None = None,
) -> tuple[int, str | None]:
    """Find or replay ordered group links and retain the audit report."""
    chain = policy.resolution == "chain_smallest_id"
    methods = [
        method
        for method in ("raw", "normalized", "canonical_origin")
        if method in policy.keys
    ]
    progress.update(phase="dedup_groups")

    if chain:
        connection.execute("UPDATE dedup SET representative_id=document_id")

    members_query = (
        "SELECT x.document_number,x.document_id,x.split,x.source_rank,"
        "x.document_kind,x.rights_signature FROM origins o "
        "JOIN dedup x ON x.document_number=o.document_number "
        "WHERE o.method=? AND o.origin=? ORDER BY x.document_id"
    )
    groups = 0

    def apply(representative, linked, train, count):
        nonlocal groups
        if chain:
            connection.executemany(
                "UPDATE dedup SET "
                "representative_id=MIN(representative_id,?) WHERE document_number=?",
                ((representative, number) for number in linked),
            )
            if train:
                connection.executemany(
                    "UPDATE dedup SET representative_id=? WHERE document_number=?",
                    ((representative, number) for number in train),
                )
        else:
            assert parents is not None
            members = iter(linked)
            first = find_root(parents, next(members))
            for number in members:
                parents[find_root(parents, number)] = first
        groups += 1
        progress.update(documents=count)

    report_path = groups_path
    if checkpoints is not None:
        report_path = checkpoints.root / "build" / "groups.jsonl"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        if any(
            path.is_symlink()
            for path in (
                report_path.parent,
                report_path,
                report_path.with_name("links.bin"),
            )
        ):
            raise ValueError(f"symlinked dedup output: {report_path.parent}")
    report_digest = None
    with ExitStack() as stack:
        output = stack.enter_context(report_path.open("a+b" if checkpoints else "wb"))
        links = None
        first_method, after = 0, None
        if checkpoints is not None:
            links_stream = stack.enter_context(
                report_path.with_name("links.bin").open("a+b")
            )
            output_view = checkpoints.add("groups.jsonl", output)
            links = checkpoints.add("links.bin", links_stream)
            state = checkpoints.restore()
            if checkpoints.damaged:
                progress.record("discarded_damaged_dedup_checkpoint")
            if state:
                first_method, after = state["method"], state["after"]
                progress.record("reused_dedup_group_windows")
                links_stream.seek(0)
                for _ in range(state["groups"]):
                    number, linked, train = _DEDUP_LINK_HEADER.unpack(
                        links_stream.read(_DEDUP_LINK_HEADER.size)
                    )
                    representative = None
                    if chain:
                        representative = connection.execute(
                            "SELECT document_id FROM dedup WHERE document_number=?",
                            (number,),
                        ).fetchone()[0]
                    count = linked + train
                    if count <= _DEDUP_GROUP_ROWS:
                        numbers = struct.unpack(
                            f"<{count}I", links_stream.read(count * _DEDUP_LINK.size)
                        )
                        apply(representative, numbers[:linked], numbers[linked:], count)
                    else:
                        apply(
                            representative,
                            _link_numbers(links_stream, linked),
                            _link_numbers(links_stream, train) if train else (),
                            count,
                        )
                if links_stream.read(1):
                    raise ValueError("dedup links and group report disagree")
                output.seek(0, os.SEEK_END)
                links_stream.seek(0, os.SEEK_END)
            target = output_view
        else:
            target = output

        def reject(method, key, group, ids, count):
            output.flush()
            os.fsync(output.fileno())
            _duplicate_diagnostics(report_path, source_events, diagnostics, group.error)
            raise ValueError(
                describe_failure(
                    group.error, method, key, ids, group.splits, member_count=count
                )
            )

        def scan(method, after, end):
            needs_rights = method == "normalized" and policy.rights_conflict == "error"
            written = 0
            with (
                closing(connection.cursor()) as cursor,
                closing(_duplicate_keys(connection, method, after, end)) as keys,
            ):
                for (key,) in keys:
                    digest = (
                        hashlib.sha256(key.encode()).hexdigest()
                        if method == "canonical_origin"
                        else key
                    )
                    rows = cursor.execute(members_query, (method, key)).fetchmany(
                        _DEDUP_GROUP_ROWS + 1
                    )
                    if len(rows) > _DEDUP_GROUP_ROWS:
                        report_start = output.tell()
                        links_start = links_stream.tell() if links is not None else 0
                        _write_large_duplicate_group(
                            iter_chain(rows, cursor),
                            method,
                            digest,
                            policy,
                            target,
                            links,
                            connection,
                            apply,
                            partial(reject, method, key),
                            report_path.parent,
                        )
                        if links is not None:
                            written += output.tell() - report_start
                            written += links_stream.tell() - links_start
                        if written >= _DEDUP_WINDOW_BYTES:
                            return key
                        continue
                    group = resolve_group(
                        method,
                        (
                            Member(
                                identifier,
                                split,
                                rank,
                                kind,
                                signature if needs_rights else None,
                            )
                            for _, identifier, split, rank, kind, signature in rows
                        ),
                        policy,
                    )
                    ids = [row[1] for row in rows]
                    rendered = (
                        canonical_json(
                            {
                                "method": method,
                                "sha256": digest,
                                "origins": ids,
                                "splits": group.splits,
                                "representative": group.representative,
                            }
                        )
                        + b"\n"
                    )
                    target.write(rendered)
                    if group.error:
                        reject(method, key, group, ids, len(rows))
                    linked, train = [], []
                    for number, _, split, *_ in rows:
                        if chain and group.contaminated and split == "train":
                            train.append(number)
                        else:
                            linked.append(number)
                    if links is not None:
                        number = (
                            next(
                                row[0] for row in rows if row[1] == group.representative
                            )
                            if chain
                            else 0
                        )
                        encoded_links = _DEDUP_LINK_HEADER.pack(
                            number, len(linked), len(train)
                        ) + struct.pack(
                            f"<{len(linked) + len(train)}I", *linked, *train
                        )
                        links.write(encoded_links)
                        written += len(rendered) + len(encoded_links)
                    apply(group.representative, linked, train, len(rows))
                    if written >= _DEDUP_WINDOW_BYTES:
                        return key
            return end

        for index in range(first_method, len(methods)):
            method = methods[index]
            if checkpoints is None:
                scan(method, None, None)
                continue
            while True:
                end = _duplicate_key_bounds(connection, method, after)
                after = scan(method, after, end)
                checkpoints.commit(
                    {
                        "method": index if after is not None else index + 1,
                        "after": after,
                        "groups": groups,
                    }
                )
                if after is None:
                    break
        output.flush()
        os.fsync(output.fileno())
        if checkpoints is not None:
            report_digest = output_view.file_digest().sha256

    if checkpoints is not None:
        # The stage can seal this immutable link without copying the report.
        os.link(report_path, groups_path)
    return groups, report_digest


def _chain_root(parents, roots, number: int) -> int:
    """Resolve one ordinal and memoize the traversed path without a Python list."""
    cursor = number
    while parents[cursor] and not roots[cursor]:
        cursor = parents[cursor]
    root = roots[cursor] or cursor
    while parents[number] and not roots[number]:
        roots[number] = root
        number = parents[number]
    return root


_SET_CHAIN_ROOT = (
    "UPDATE dedup SET representative_id=(SELECT p.document_id FROM dedup p "
    "WHERE p.document_number=?) WHERE document_number=?"
)


def _compress_chain(
    connection: sqlite3.Connection,
    total: int,
    report_digest: str | None,
    compression: _DedupJournal | None,
    scratch: Path,
    progress: BuildProgress,
) -> None:
    progress.update(phase="dedup_compress")
    last = 0
    if compression is not None:
        if compression.journal.damaged:
            progress.record("discarded_damaged_dedup_compression")
        state = compression.state
        if state and state["groups_sha256"] != report_digest:
            compression.reset()
            progress.record("discarded_dedup_compression_for_changed_groups")
        elif state:
            connection.executemany(
                _SET_CHAIN_ROOT,
                (
                    (representative, number)
                    for number, representative in compression.records()
                ),
            )
            progress.record("reused_dedup_compression_windows")
            if state["complete"]:
                return
            last = state["last"]

    # Train copies can link to held-out documents, which never link back.
    # Other links decrease lexicographically, so paths terminate.
    edges = (
        "FROM dedup x JOIN dedup p ON p.document_id=x.representative_id "
        "WHERE x.document_id<>x.representative_id"
    )
    transitive = connection.execute(
        "SELECT 1 FROM dedup WHERE document_id<>representative_id "
        "AND representative_id<>(SELECT p.representative_id FROM dedup p "
        "WHERE p.document_id=dedup.representative_id) LIMIT 1"
    ).fetchone()
    if transitive is None:
        if compression is not None:
            compression.journal.commit(
                {"last": total, "complete": True, "groups_sha256": report_digest}
            )
        return

    # Zero denotes a self link in the first array and an unknown root in the
    # second. The scratch file can be paged out under memory pressure.
    with tempfile.TemporaryFile(dir=scratch) as stream:
        width = total + 1
        stream.truncate(2 * width * struct.calcsize("I"))
        with (
            mmap.mmap(stream.fileno(), 0) as mapping,
            memoryview(mapping).cast("I") as ordinals,
            ordinals[:width] as parents,
            ordinals[width:] as roots,
        ):
            for number, parent in connection.execute(
                "SELECT x.document_number,p.document_number " + edges
            ):
                parents[number] = parent
            while last < total:
                end = min(total, last + _DEDUP_COMPRESSION_ROWS)
                updates = bytearray()
                for number in range(last + 1, end + 1):
                    parent = parents[number]
                    if parent and parents[parent]:
                        updates.extend(
                            _DEDUP_REPRESENTATIVE.pack(
                                number, _chain_root(parents, roots, number)
                            )
                        )
                connection.executemany(
                    _SET_CHAIN_ROOT,
                    (
                        (root, number)
                        for number, root in _DEDUP_REPRESENTATIVE.iter_unpack(updates)
                    ),
                )
                if compression is not None:
                    compression.output.write(bytes(updates))
                    compression.journal.commit(
                        {
                            "last": end,
                            "complete": end == total,
                            "groups_sha256": report_digest,
                        }
                    )
                progress.update(documents=len(updates) // _DEDUP_REPRESENTATIVE.size)
                last = end


def _finalize_chain(
    connection: sqlite3.Connection,
    groups: int,
    report_digest: str | None,
    compression: _DedupJournal | None,
    decisions: _DedupJournal | None,
    scratch: Path,
    progress: BuildProgress,
) -> int:
    from klab.corpus.dedup_policy import drop_reason

    connection.commit()
    # Range over document numbers, even where no representatives change.
    total = connection.execute(
        "SELECT COALESCE(MAX(document_number), 0) FROM dedup"
    ).fetchone()[0]
    if groups:
        _compress_chain(
            connection, total, report_digest, compression, scratch, progress
        )
    connection.commit()
    progress.update(phase="dedup_reasons")
    last, dropped = 0, 0
    if decisions is not None and decisions.state:
        if decisions.state["groups_sha256"] == report_digest:
            last = decisions.state["last"]
            dropped = decisions.state["dropped"]
            _apply_decisions(connection, decisions.records(), representatives=False)
            progress.record("reused_dedup_decision_windows")
        else:
            decisions.reset()
            progress.record("discarded_dedup_decisions_for_changed_groups")

    while last < total:
        end = min(total, last + _DEDUP_DECISION_ROWS)
        updates = []
        records = bytearray()
        rows = connection.execute(
            "SELECT x.document_number,x.metadata_only,x.split,x.document_id,"
            "x.representative_id,p.split,p.document_number FROM dedup x JOIN dedup p "
            "ON p.document_id=x.representative_id "
            "WHERE x.document_number>? AND x.document_number<=? "
            "AND (x.metadata_only=1 OR x.document_id<>x.representative_id) "
            "ORDER BY x.document_number",
            (last, end),
        )
        for (
            number,
            metadata_only,
            split,
            identifier,
            representative,
            kept,
            rep_number,
        ) in rows:
            reason = drop_reason(
                metadata_only=bool(metadata_only),
                split=split,
                is_representative=identifier == representative,
                representative_split=kept,
            )
            if reason is not None:
                updates.append((reason, number))
                if decisions is not None:
                    records.extend(
                        _DEDUP_DECISION.pack(
                            number, rep_number, _DEDUP_REASON_CODES[reason]
                        )
                    )
        connection.executemany(
            "UPDATE dedup SET drop_reason=? WHERE document_number=?", updates
        )
        dropped += len(updates)
        if decisions is not None:
            decisions.output.write(bytes(records))
            decisions.journal.commit(
                {
                    "last": end,
                    "dropped": dropped,
                    "complete": end == total,
                    "groups": groups,
                    "groups_sha256": report_digest,
                }
            )
        progress.update(documents=end - last)
        last = end
    connection.commit()
    return dropped


_RANKED_SPLIT_MASK = {"train": 0, "validation": 1, "test": 2}
_RANKED_HOLDOUT = 3
_RANKED_CLEANING = 12
_RANKED_GROUPED = 16


def _document_count(connection: sqlite3.Connection) -> int:
    return connection.execute(
        "SELECT COALESCE(MAX(document_number), 0) FROM dedup"
    ).fetchone()[0]


def _choose_ranked_components(
    connection, parents, masks, winners, report_digest, journal, progress
):
    from klab.corpus.dedup_policy import find_root, ranked_choice

    def key(identifier, split, rank, kind):
        # A valid component has at most one held-out split. Put held-out
        # members first while accumulating the evidence that enforces that.
        return (
            split == "train",
            *ranked_choice(split, None, rank, kind, identifier)[1:],
        )

    last, conflicts = 0, 0
    if journal is not None:
        if journal.journal.damaged:
            progress.record("discarded_damaged_ranked_components")
        state = journal.state
        if state and state["groups_sha256"] != report_digest:
            journal.reset()
            progress.record("discarded_ranked_components_for_changed_groups")
        elif state:
            for root, winner, mask in journal.records():
                masks[root], winners[root] = mask, winner
            progress.record("reused_ranked_component_windows")
            if state["complete"]:
                return state["conflicts"]
            last, conflicts = state["last"], state["conflicts"]

    # Singletons select themselves. Give only real components reducer state.
    total = len(parents) - 1
    for number in range(1, total + 1):
        if parents[number] != number:
            masks[find_root(parents, number)] |= _RANKED_GROUPED

    progress.update(phase="dedup_components")
    while last < total:
        end = min(total, last + _DEDUP_DECISION_ROWS)
        # Winner keys live only for the current window. A prior window
        # carries its winner ordinal, resolved with one indexed row lookup.
        keys, changed = {}, set()
        for number, identifier, split, rank, kind, metadata_only in connection.execute(
            "SELECT document_number,document_id,split,source_rank,document_kind,metadata_only "
            "FROM dedup WHERE document_number>? AND document_number<=? ORDER BY document_number",
            (last, end),
        ):
            root = find_root(parents, number)
            old_mask = masks[root]
            if not old_mask & _RANKED_GROUPED:
                continue
            mask = old_mask | _RANKED_SPLIT_MASK[split] | (8 if metadata_only else 4)
            if mask != old_mask:
                masks[root] = mask
                changed.add(root)
                conflicts |= int(mask & _RANKED_HOLDOUT == _RANKED_HOLDOUT)
                conflicts |= int(mask & _RANKED_CLEANING == _RANKED_CLEANING) << 1
            candidate = key(identifier, split, rank, kind)
            winner = winners[root]
            if winner and root not in keys:
                keys[root] = key(
                    *connection.execute(
                        "SELECT document_id,split,source_rank,document_kind "
                        "FROM dedup WHERE document_number=?",
                        (winner,),
                    ).fetchone()
                )
            if not winner or candidate < keys[root]:
                winners[root] = number
                keys[root] = candidate
                changed.add(root)
        if journal is not None:
            journal.output.write(
                b"".join(
                    _RANKED_COMPONENT.pack(root, winners[root], masks[root])
                    for root in sorted(changed)
                )
            )
            journal.journal.commit(
                {
                    "last": end,
                    "conflicts": conflicts,
                    "complete": end == total,
                    "groups_sha256": report_digest,
                }
            )
        progress.update(documents=end - last)
        last = end
    return conflicts


def _ranked_conflict(connection, parents, masks, conflicts, diagnostics):
    from klab.corpus.dedup_policy import (
        CLEANING_CONFLICT,
        PAGE_OVERLAP,
        describe_failure,
        find_root,
    )

    if conflicts & 1:
        flag, message = _RANKED_HOLDOUT, PAGE_OVERLAP
    else:
        flag, message = _RANKED_CLEANING, CLEANING_CONFLICT
    # Preserve the first offending component in document-number order.
    root = next(
        root
        for (number,) in connection.execute(
            "SELECT document_number FROM dedup ORDER BY document_number"
        )
        if masks[root := find_root(parents, number)] & flag == flag
    )
    sample, splits, count = [], set(), 0
    for number, identifier, split in connection.execute(
        "SELECT document_number,document_id,split FROM dedup ORDER BY document_number"
    ):
        if find_root(parents, number) == root:
            if count < 5:
                sample.append(identifier)
            splits.add(split)
            count += 1
    text = describe_failure(
        message,
        "component",
        sample[0],
        sample,
        splits,
        member_count=count,
    )
    diagnostics.parent.mkdir(parents=True, exist_ok=True)
    write_canonical_json(diagnostics, {"error": text})
    raise ValueError(text)


def _finalize_ranked(
    connection,
    parents,
    groups,
    report_digest,
    diagnostics,
    progress,
    *,
    checkpoints,
    decisions,
    scratch,
):
    from klab.corpus.dedup_policy import drop_reason, find_root

    total = len(parents) - 1
    with ExitStack() as stack:
        masks = winners = None
        if groups:
            masks = stack.enter_context(_mapped_array(scratch, total + 1, "B"))
            winners = stack.enter_context(_mapped_array(scratch, total + 1, "I"))
            components = (
                _DedupJournal(
                    checkpoints,
                    stack,
                    stage="ranked-components",
                    filename="winners.bin",
                    record=_RANKED_COMPONENT,
                )
                if checkpoints is not None
                else None
            )
            conflicts = _choose_ranked_components(
                connection, parents, masks, winners, report_digest, components, progress
            )
            if conflicts:
                _ranked_conflict(connection, parents, masks, conflicts, diagnostics)

        connection.execute("UPDATE dedup SET representative_id=document_id")
        connection.commit()
        progress.update(phase="dedup_reasons")
        last, dropped = 0, 0
        if decisions is not None and decisions.state:
            if decisions.state["groups_sha256"] == report_digest:
                last, dropped = decisions.state["last"], decisions.state["dropped"]
                _apply_decisions(connection, decisions.records(), representatives=True)
                progress.record("reused_dedup_decision_windows")
            else:
                decisions.reset()
                progress.record("discarded_dedup_decisions_for_changed_groups")
        while last < total:
            end = min(total, last + _DEDUP_DECISION_ROWS)
            records = bytearray()
            for number, split, metadata_only in connection.execute(
                "SELECT document_number,split,metadata_only FROM dedup "
                "WHERE document_number>? AND document_number<=? ORDER BY document_number",
                (last, end),
            ):
                representative, kept = number, split
                if winners is not None:
                    root = find_root(parents, number)
                    winner = winners[root]
                    if winner:
                        representative = winner
                        kept = ("train", "validation", "test")[
                            masks[root] & _RANKED_HOLDOUT
                        ]
                reason = drop_reason(
                    metadata_only=bool(metadata_only),
                    split=split,
                    is_representative=representative == number,
                    representative_split=kept,
                )
                if reason is not None:
                    records.extend(
                        _DEDUP_DECISION.pack(
                            number, representative, _DEDUP_REASON_CODES[reason]
                        )
                    )
            _apply_decisions(
                connection, _DEDUP_DECISION.iter_unpack(records), representatives=True
            )
            dropped += len(records) // _DEDUP_DECISION.size
            if decisions is not None:
                decisions.output.write(bytes(records))
                decisions.journal.commit(
                    {
                        "last": end,
                        "dropped": dropped,
                        "complete": end == total,
                        "groups": groups,
                        "groups_sha256": report_digest,
                    }
                )
            connection.commit()
            progress.update(documents=end - last)
            last = end
        connection.commit()
        return dropped


def rejected_rows(source_events: Iterable[Path | dict[str, Any]]) -> Iterator[bytes]:
    for event in source_events:
        if isinstance(event, Path):
            with (event / "rejected.jsonl").open("rb") as source:
                yield from source
        else:
            yield canonical_json(event) + b"\n"
