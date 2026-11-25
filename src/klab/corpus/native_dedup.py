"""Duplicate groups, checkpointed resolution and compact native decisions."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import sqlite3
import struct
import tempfile
from array import array
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, closing
from pathlib import Path
from typing import Any, BinaryIO
from urllib.parse import quote

from klab.canonical import canonical_json
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
            for index, row in enumerate(rows):
                if index:
                    report.write(b",")
                report.write(row.rstrip(b"\n"))
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
_DEDUP_LINK_COUNTS = struct.Struct("<II")


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
_DEDUP_DROP_REASONS = (
    "metadata_only_front_matter",
    "duplicate",
    "contaminated_heldout",
)
_DEDUP_REASON_CODES = {reason: code for code, reason in enumerate(_DEDUP_DROP_REASONS)}


class _ChainJournal:
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


def _replay_decisions(connection, decisions, *, representatives):
    if representatives:
        connection.executemany(
            "UPDATE dedup SET representative_id=(SELECT p.document_id FROM dedup p "
            "WHERE p.document_number=?), drop_reason=? WHERE document_number=?",
            (
                (representative, _DEDUP_DROP_REASONS[code], number)
                for number, representative, code in decisions.records()
            ),
        )
    else:
        connection.executemany(
            "UPDATE dedup SET drop_reason=? WHERE document_number=?",
            (
                (_DEDUP_DROP_REASONS[code], number)
                for number, _, code in decisions.records()
            ),
        )


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
            _ChainJournal(
                checkpoints,
                stack,
                stage="chain-decisions",
                filename="decisions.bin",
                record=_DEDUP_DECISION,
            )
            if chain and checkpoints
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
                    _replay_decisions(connection, decisions, representatives=True)
                    connection.commit()
                    if range_file_identity(report) != generation:
                        raise ValueError(
                            "dedup group report changed while replaying final decisions"
                        )
                    os.link(report, groups_path)
                    return state["groups"], state["dropped"]

        groups, parents, report_digest = _resolve_duplicate_groups(
            connection,
            groups_path,
            policy=policy,
            diagnostics=diagnostics,
            progress=progress,
            source_events=source_events,
            checkpoints=checkpoints,
        )
        if chain:
            compression = (
                _ChainJournal(
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
        return groups, _finalize_ranked(connection, parents, diagnostics, progress)


def _resolve_duplicate_groups(
    connection: sqlite3.Connection,
    groups_path: Path,
    *,
    policy: Any,
    diagnostics: Path,
    progress: BuildProgress,
    source_events: Iterable[Path | dict[str, Any]] = (),
    checkpoints: ViewCheckpoints | None = None,
) -> tuple[int, array | None, str | None]:
    """Find or replay ordered group links and retain the audit report."""
    from klab.corpus.dedup_policy import (
        Member,
        describe_failure,
        find_root,
        resolve_group,
    )

    chain = policy.resolution == "chain_smallest_id"
    methods = [
        method
        for method in ("raw", "normalized", "canonical_origin")
        if method in policy.keys
    ]
    parents = None
    if not chain:
        total = connection.execute("SELECT COUNT(*) FROM dedup").fetchone()[0]
        parents = array("I", range(total + 1))
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

    def apply(representative, linked, train):
        nonlocal groups
        if chain:
            if train:
                connection.executemany(
                    "UPDATE dedup SET representative_id=? WHERE document_number=?",
                    ((representative, number) for number in train),
                )
            connection.executemany(
                "UPDATE dedup SET "
                "representative_id=MIN(representative_id,?) WHERE document_number=?",
                ((representative, number) for number in linked),
            )
        else:
            assert parents is not None
            members = iter(linked)
            first = find_root(parents, next(members))
            for number in members:
                parents[find_root(parents, number)] = first
        groups += 1
        progress.update(documents=len(linked) + len(train))

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
                output.seek(0)
                links_stream.seek(0)
                for rendered in output:
                    linked, train = _DEDUP_LINK_COUNTS.unpack(
                        links_stream.read(_DEDUP_LINK_COUNTS.size)
                    )
                    numbers = struct.unpack(
                        f"<{linked + train}I", links_stream.read(4 * (linked + train))
                    )
                    representative = (
                        json.loads(rendered)["representative"] if chain else None
                    )
                    apply(representative, numbers[:linked], numbers[linked:])
                if links_stream.read(1):
                    raise ValueError("dedup links and group report disagree")
                output.seek(0, os.SEEK_END)
                links_stream.seek(0, os.SEEK_END)
            target = output_view
        else:
            target = output

        def scan(method, after, end):
            needs_rights = method == "normalized" and policy.rights_conflict == "error"
            written = 0
            with closing(_duplicate_keys(connection, method, after, end)) as keys:
                for (key,) in keys:
                    rows = connection.execute(members_query, (method, key)).fetchall()
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
                                "sha256": hashlib.sha256(key.encode()).hexdigest()
                                if method == "canonical_origin"
                                else key,
                                "origins": ids,
                                "splits": group.splits,
                                "representative": group.representative,
                            }
                        )
                        + b"\n"
                    )
                    target.write(rendered)
                    if group.error:
                        output.flush()
                        os.fsync(output.fileno())
                        _duplicate_diagnostics(
                            report_path, source_events, diagnostics, group.error
                        )
                        raise ValueError(
                            describe_failure(
                                group.error, method, key, ids, group.splits
                            )
                        )
                    linked, train = [], []
                    for number, _, split, *_ in rows:
                        if chain and group.contaminated and split == "train":
                            train.append(number)
                        else:
                            linked.append(number)
                    if links is not None:
                        encoded_links = _DEDUP_LINK_COUNTS.pack(
                            len(linked), len(train)
                        ) + struct.pack(
                            f"<{len(linked) + len(train)}I", *linked, *train
                        )
                        links.write(encoded_links)
                        written += len(rendered) + len(encoded_links)
                    apply(group.representative, linked, train)
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
    return groups, parents, report_digest


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
    compression: _ChainJournal | None,
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
    compression: _ChainJournal | None,
    decisions: _ChainJournal | None,
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
            _replay_decisions(connection, decisions, representatives=False)
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


def _finalize_ranked(
    connection: sqlite3.Connection,
    parents: array,
    diagnostics: Path,
    progress: BuildProgress,
) -> int:
    from klab.corpus.dedup_policy import (
        CLEANING_CONFLICT,
        PAGE_OVERLAP,
        describe_failure,
        drop_reason,
        find_root,
        ranked_choice,
    )

    # A connected component can link distinct raw, normalized and page origins.
    # Whole components select one deterministic representative, preserving the
    # validation/test isolation gate even for transitive overlaps.
    assert parents is not None
    winner: dict[int, tuple[Any, ...]] = {}
    heldout_mask: dict[int, int] = defaultdict(int)
    cleaning_mask: dict[int, int] = defaultdict(int)
    progress.update(phase="dedup_components")
    for number, split, metadata_only in connection.execute(
        "SELECT document_number,split,metadata_only FROM dedup ORDER BY document_number"
    ):
        component = find_root(parents, number)
        heldout_mask[component] |= (
            1 if split == "validation" else 2 if split == "test" else 0
        )
        cleaning_mask[component] |= 2 if metadata_only else 1
    for masks, ok, message in (
        (heldout_mask, lambda mask: mask != 3, PAGE_OVERLAP),
        (cleaning_mask, lambda mask: mask != 3, CLEANING_CONFLICT),
    ):
        bad = next((root for root, mask in masks.items() if not ok(mask)), None)
        if bad is not None:
            members = [
                (identifier, split)
                for number, identifier, split in connection.execute(
                    "SELECT document_number,document_id,split FROM dedup "
                    "ORDER BY document_number"
                )
                if find_root(parents, number) == bad
            ]
            text = describe_failure(
                message,
                "component",
                members[0][0],
                [item[0] for item in members],
                {item[1] for item in members},
            )
            diagnostics.parent.mkdir(parents=True, exist_ok=True)
            write_canonical_json(diagnostics, {"error": text})
            raise ValueError(text)
    for number, identifier, split, rank, kind in connection.execute(
        "SELECT document_number,document_id,split,source_rank,document_kind "
        "FROM dedup ORDER BY document_number"
    ):
        component = find_root(parents, number)
        preferred_split = (
            "validation"
            if heldout_mask[component] == 1
            else "test"
            if heldout_mask[component] == 2
            else None
        )
        choice = ranked_choice(split, preferred_split, rank, kind, identifier)
        if component not in winner or choice < winner[component][:4]:
            winner[component] = (*choice, identifier, split)
    dropped = 0
    updates = []
    for number, identifier, split, metadata_only in connection.execute(
        "SELECT document_number,document_id,split,metadata_only FROM dedup "
        "ORDER BY document_number"
    ):
        selected, retained_split = winner[find_root(parents, number)][4:6]
        reason = drop_reason(
            metadata_only=bool(metadata_only),
            split=split,
            is_representative=selected == identifier,
            representative_split=retained_split,
        )
        dropped += reason is not None
        updates.append((selected, reason, number))
        if len(updates) >= 10_000:
            connection.executemany(
                "UPDATE dedup SET representative_id=?,drop_reason=? "
                "WHERE document_number=?",
                updates,
            )
            connection.commit()
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE dedup SET representative_id=?,drop_reason=? "
            "WHERE document_number=?",
            updates,
        )
        connection.commit()
    return dropped


def rejected_rows(source_events: Iterable[Path | dict[str, Any]]) -> Iterator[bytes]:
    for event in source_events:
        if isinstance(event, Path):
            with (event / "rejected.jsonl").open("rb") as source:
                yield from source
        else:
            yield canonical_json(event) + b"\n"
