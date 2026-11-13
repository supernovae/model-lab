"""Bounded-memory, resumable Corpus Forge build for prospective LM source pools.

Selected source bytes remain in the verified acquisition snapshots. A prepared
file is immutable only after its receipt and all three output hashes validate;
incomplete staging directories are never inputs to a resumed build.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from array import array
from collections import Counter, defaultdict, deque
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.progress import BuildProgress
from klab.corpus.project import CURATED_SOURCE_KINDS, ROW_SOURCE_KINDS
from klab.corpus.provenance import (
    HUMAN_ORIGIN,
    SOURCE_ORIGIN,
    rendered_digest,
    shape_for_record,
    validate_lineage,
    verification,
)
from klab.corpus.rights import (
    FILE_ADMISSION_KINDS,
    reported_spdx_expression,
    resolve_file_rights,
)
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.verification_proofs import ProofStore, VerificationMode


_SPLITS = ("train", "validation", "test")
_PREPARED_FILES = ("docs.jsonl", "spans.jsonl", "rejected.jsonl")
_PROCESS_SHARD_MIN_BYTES = 32 * 1024**2
_PROCESS_SHARD_MAX_LINE_BYTES = 1024**2
_ADMITTED_RECORD_MAX_BYTES = 8 * 1024**2


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(canonical_json(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())


def _output(handle: Any, value: Any) -> None:
    handle.write(canonical_json(value) + b"\n")


def _receipt_identity(
    build_id: str, source: Any, snapshot_sha: str, file: dict[str, Any]
) -> dict[str, Any]:
    return {
        "stage_version": "normalized-source-v2",
        "implementation": {
            name: sha256_file(Path(__file__).with_name(name))
            for name in (
                "large_build.py",
                "pipeline.py",
                "jsonl_records.py",
                "structure_cleaning.py",
                "rights.py",
                "provenance.py",
            )
        },
        "source": source.model_dump(mode="json"),
        "source_id": source.id,
        "revision": source.revision,
        "snapshot_sha256": snapshot_sha,
        "path": file["path"],
        "file_sha256": file["sha256"],
        "file_size": file["size"],
    }


def _verify_prepared(path: Path, identity: dict[str, Any]) -> dict[str, Any]:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("prepared source shard is not a directory")
    receipt = json.loads((path / "receipt.json").read_text(encoding="utf-8"))
    if receipt["identity"] != identity or set(receipt["files"]) != set(_PREPARED_FILES):
        raise ValueError("prepared source shard provenance mismatch")
    for name, expected in receipt["files"].items():
        item = path / name
        if (
            item.is_symlink()
            or not item.is_file()
            or item.stat().st_size != expected["size"]
            or sha256_file(item) != expected["sha256"]
        ):
            raise ValueError(f"prepared source shard changed: {name}")
    return receipt


class _SourceParseError(ValueError):
    """Input file rejected by the normalizer, not a broken receipt or build."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.documents = 0
        self.input_bytes = 0
        self.output_records = 0


class _ShardProgress:
    """Accumulate worker counters; the parent journals in input order."""

    def __init__(self) -> None:
        self.documents = 0
        self.input_bytes = 0
        self.output_records = 0
        self.reused = False

    def update(
        self, *, documents: int = 0, input_bytes: int = 0, output_records: int = 0
    ) -> None:
        self.documents += documents
        self.input_bytes += input_bytes
        self.output_records += output_records

    def record(self, event: str) -> None:
        if event == "reused_verified_shard":
            self.reused = True


def _prepare_shard(**kwargs: Any) -> tuple[Path, int, int, int, bool]:
    progress = _ShardProgress()
    try:
        shard = _prepare_file(**kwargs, progress=progress)
    except _SourceParseError as error:
        error.documents = progress.documents
        error.input_bytes = progress.input_bytes
        error.output_records = progress.output_records
        raise
    return (
        shard,
        progress.documents,
        progress.input_bytes,
        progress.output_records,
        progress.reused,
    )


def _process_shard_workers(count: int) -> int:
    """Use the shared planner with the measured 512 MiB per-process bound."""
    if count < 2:
        return 1
    import psutil

    from klab import host_capacity
    from klab.host_capacity import plan_host_workers

    try:
        total = host_capacity.measure_memory().total
        return plan_host_workers(
            "corpus_jsonl_shards",
            worker_memory_bytes=512 * 1024**2,
            reserve_bytes=max(1024**3, total // 10),
            operator_cap=min(2, count),
        ).workers
    except AttributeError, OSError, psutil.Error, TypeError, ValueError:
        # Unknown or insufficient capacity retains the existing serial parser.
        return 1


def _bounded_jsonl_shard(path: Path) -> bool:
    """Keep oversized JSON rows on the serial path rather than guessing RSS."""
    with path.open("rb") as stream:
        while line := stream.readline(_PROCESS_SHARD_MAX_LINE_BYTES + 1):
            if len(line) > _PROCESS_SHARD_MAX_LINE_BYTES:
                return False
    return True


# Rights resolution inspects only 30 LF-delimited lines. This is a header
# evidence limit, not a bound on fallback normalization or admission screening.
_RIGHTS_HEADER_MAX_BYTES = 64 * 1024


def _read_rights_header(path: Path, name: str) -> bytes:
    """Read complete rights evidence within 64 KiB, plus one overflow probe byte.

    Match resolve_file_rights' case-sensitive dataset suffixes and binary LF
    boundaries (including CRLF and a final unterminated line). Never classify a
    truncated header: fewer than 30 lines require EOF within the byte ceiling.
    """
    if name.endswith((".jsonl", ".json", ".parquet")):
        return b""
    header = bytearray()
    with path.open("rb") as stream:
        for _ in range(30):
            line = stream.readline(_RIGHTS_HEADER_MAX_BYTES - len(header) + 1)
            if len(header) + len(line) > _RIGHTS_HEADER_MAX_BYTES:
                raise ValueError(
                    f"rights header exceeds {_RIGHTS_HEADER_MAX_BYTES} bytes: {name}"
                )
            header.extend(line)
            if not line.endswith(b"\n"):
                break
    return bytes(header)


def _prepare_file(
    *,
    build_id: str,
    prepared_root: Path,
    source: Any,
    file: dict[str, Any],
    snapshot_sha: str,
    source_path: Path,
    decision: Any,
    progress: BuildProgress,
    normalizer_version: str = "normalizer-nfc-markdown-v1",
    record_admission: dict[str, Any] | None = None,
    admission_sha256: str | None = None,
) -> Path:
    from klab.corpus.pipeline import _records_for_file
    from klab.engram.packs import _rename_noreplace

    # Internal primitive: callers must first use verify_record_admission against
    # the pinned original rows. The digest binds that complete reviewed artifact,
    # not a newly inferred policy or an approval supplied by this helper.
    if (record_admission is None) != (admission_sha256 is None):
        raise ValueError(
            "admitted shard requires verified decisions and manifest digest"
        )
    if record_admission is not None:
        if not isinstance(admission_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", admission_sha256
        ):
            raise ValueError("invalid admission manifest digest")
        if "decisions" in record_admission:
            if source.kind != "huggingface_dataset" or not file["path"].endswith(
                ".jsonl"
            ):
                raise ValueError("streamed record admission requires an HF JSONL shard")
        elif source.kind not in FILE_ADMISSION_KINDS:
            raise ValueError("file admission requires a file-admitted source")
        if decision != record_admission["rights"]:
            raise ValueError("admitted shard file rights differ from verified policy")
    identity = _receipt_identity(build_id, source, snapshot_sha, file)
    identity["normalizer_version"] = normalizer_version
    identity["rights"] = decision.model_dump(mode="json")
    if record_admission is not None:
        # Bind only this verified file's decisions, not the admission plan for
        # unrelated sources. Stream disk-backed decisions without materializing.
        digest = hashlib.sha256()
        for key in sorted(record_admission):
            value = record_admission[key]
            digest.update(canonical_json(key))
            if key == "decisions":
                for row_index, row in value.items():
                    digest.update(canonical_json([row_index, row]))
            else:
                if hasattr(value, "model_dump"):
                    value = value.model_dump(mode="json")
                digest.update(canonical_json(value))
        identity["file_admission_sha256"] = digest.hexdigest()
    parent = prepared_root / source.id
    destination = parent / canonical_digest(identity)
    if destination.exists():
        receipt = _verify_prepared(destination, identity)
        progress.update(
            documents=receipt["counts"]["documents"],
            input_bytes=receipt["counts"]["input_bytes"],
            output_records=receipt["counts"]["documents"],
        )
        progress.record("reused_verified_shard")
        return destination
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".incomplete-", dir=parent) as temporary:
        staging = Path(temporary)
        counts = {"documents": 0, "rejected": 0, "input_bytes": 0}
        with ExitStack() as stack:
            docs = stack.enter_context((staging / "docs.jsonl").open("wb"))
            spans = stack.enter_context((staging / "spans.jsonl").open("wb"))
            rejected = stack.enter_context((staging / "rejected.jsonl").open("wb"))
            if source.kind in ROW_SOURCE_KINDS and file["path"].endswith(".jsonl"):
                with source_path.open("rb") as stream:
                    index = 0
                    from klab.corpus.jsonl_records import iter_lf_lines

                    input_digest = hashlib.sha256()
                    for _, raw in iter_lf_lines(
                        stream,
                        max_record_bytes=(
                            _ADMITTED_RECORD_MAX_BYTES
                            if record_admission is not None
                            else None
                        ),
                    ):
                        input_digest.update(raw)
                        counts["input_bytes"] += len(raw)
                        if not raw.strip():
                            progress.update(input_bytes=len(raw))
                            continue
                        index += 1
                        if index > source.acquisition.max_rows:
                            progress.update(input_bytes=len(raw))
                            continue
                        dropped: list[dict[str, Any]] = []
                        try:
                            rows = _records_for_file(
                                raw,
                                file["path"],
                                source,
                                snapshot_sha,
                                file_rights=decision,
                                record_admission=record_admission,
                                rejected_records=dropped,
                                full_file_sha256=file["sha256"],
                                first_row_index=index,
                                normalizer_version=normalizer_version,
                            )
                        except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                            raise _SourceParseError(str(exc)) from exc
                        for doc, span in rows:
                            _output(docs, doc)
                            _output(spans, span)
                            counts["documents"] += 1
                        for row in dropped:
                            _output(rejected, row)
                            counts["rejected"] += 1
                        progress.update(
                            documents=1, input_bytes=len(raw), output_records=len(rows)
                        )
                    if (
                        record_admission is not None
                        and input_digest.hexdigest() != file["sha256"]
                    ):
                        raise ValueError(
                            "admitted source shard bytes differ from acquisition"
                        )
            else:
                if record_admission is not None:
                    with source_path.open("rb") as source_stream:
                        raw = source_stream.read(_ADMITTED_RECORD_MAX_BYTES + 1)
                    if len(raw) > _ADMITTED_RECORD_MAX_BYTES:
                        raise ValueError("admitted file-byte cap exceeded")
                    if hashlib.sha256(raw).hexdigest() != file["sha256"]:
                        raise ValueError(
                            "admitted source bytes differ from acquisition"
                        )
                else:
                    raw = source_path.read_bytes()
                dropped = []
                try:
                    rows = _records_for_file(
                        raw,
                        file["path"],
                        source,
                        snapshot_sha,
                        file_rights=decision,
                        record_admission=record_admission,
                        rejected_records=dropped,
                        normalizer_version=normalizer_version,
                    )
                except (ValueError, UnicodeError, KeyError, TypeError) as exc:
                    raise _SourceParseError(str(exc)) from exc
                for doc, span in rows:
                    _output(docs, doc)
                    _output(spans, span)
                for row in dropped:
                    _output(rejected, row)
                counts = {
                    "documents": len(rows),
                    "rejected": len(dropped),
                    "input_bytes": len(raw),
                }
                progress.update(
                    documents=len(rows), input_bytes=len(raw), output_records=len(rows)
                )
            for output in (docs, spans, rejected):
                output.flush()
                os.fsync(output.fileno())
        if counts["input_bytes"] != file["size"]:
            raise ValueError(
                "prepared source shard input size differs from acquisition"
            )
        receipt = {
            "schema_version": 1,
            "identity": identity,
            "provenance": {"build_id": build_id, "admission_sha256": admission_sha256},
            "counts": counts,
            "files": {
                name: {
                    "size": (staging / name).stat().st_size,
                    "sha256": sha256_file(staging / name),
                }
                for name in _PREPARED_FILES
            },
        }
        _write_json(staging / "receipt.json", receipt)
        if destination.exists():
            _verify_prepared(destination, identity)
        else:
            _rename_noreplace(staging, destination)
    _verify_prepared(destination, identity)
    return destination


def _sources_and_rights(
    project: Any, lock: dict[str, Any], rights_files: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sources = [
        {
            "id": source.id,
            "kind": source.kind,
            "canonical_uri": source.canonical_uri,
            "revision": source.revision,
            "license": source.license,
            "redistribution": source.rights.redistribution_mode
            if source.rights
            else source.redistribution,
            **(
                {
                    "license_url": source.license_url,
                    "rights_policy": source.rights.model_dump(mode="json"),
                }
                if source.rights
                else {}
            ),
            **(
                {"explicit_training_restriction": source.explicit_training_restriction}
                if source.schema_version == 3
                else {}
            ),
            "origin": source.origin,
            "source_family": source.source_family,
            "snapshot_sha256": lock["sources"]
            .get(source.id, {})
            .get("snapshot_sha256"),
            "reproducibility_class": "script_reproducible_not_redistributed"
            if source.rights.redistribution_mode
            in {
                "reference_only",
                "derived_only",
                "unknown",
                "rejected",
                "metadata_reconstruction_only",
                "not_redistributable",
                "review_required",
            }
            else "fully_reproducible",
        }
        for source in project.sources
    ]
    files = [row for row in rights_files if row["role"] == "document"]
    eligibility = Counter(row["rights"]["training_eligibility"] for row in files)
    bytes_by_state: Counter[str] = Counter()
    expressions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    source_spdx = {
        source.id: source.rights.spdx_expression for source in project.sources
    }
    for item in files:
        rights = item["rights"]
        bytes_by_state[rights["training_eligibility"]] += item["size"]
        expressions[reported_spdx_expression(item, source_spdx[item["source_id"]])] += 1
        modes[rights["redistribution_mode"]] += 1
    rights_report = {
        "schema_version": project.release.schema_version,
        "publication_mode": project.release.publication_mode,
        **(
            {"training_use_policy": project.release.training_use_policy}
            if project.release.schema_version == 3
            else {}
        ),
        "weight_license_status": "separate_analysis_required",
        "sources": sources,
        "files": rights_files,
        "training_eligibility": {
            state: {
                "files": eligibility[state],
                "source_bytes": bytes_by_state[state],
                "source_tokens": None,
                "token_count_reason": "tokenizer_not_declared",
            }
            for state in (
                "eligible",
                "eligible_with_obligations",
                "review_required",
                "ineligible",
            )
        },
        "spdx_expressions": dict(expressions),
        "redistribution_modes": dict(modes),
        "unresolved_rights_files": eligibility["review_required"],
        "advisory": "Source/derived-data rights and model-weight licensing are separate decisions; not a legal conclusion.",
    }
    return sources, rights_report


def _index_prepared(
    database: Path, prepared: list[Path], project: Any, progress: BuildProgress
) -> sqlite3.Connection:
    from klab.corpus.pipeline import _origin_keys, _split

    connection = sqlite3.connect(database)
    connection.execute("PRAGMA cache_size=-16384")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA synchronous=FULL")
    connection.executescript(
        """
        CREATE TABLE documents (
            document_id TEXT NOT NULL UNIQUE,
            payload BLOB NOT NULL,
            span BLOB NOT NULL,
            split TEXT NOT NULL,
            source_id TEXT NOT NULL,
            source_family TEXT NOT NULL,
            document_kind TEXT NOT NULL,
            source_rank INTEGER NOT NULL,
            representative_id TEXT,
            drop_reason TEXT
        );
        CREATE TABLE origins (
            method TEXT NOT NULL,
            origin TEXT NOT NULL,
            document_number INTEGER NOT NULL
        );
        """
    )
    from klab.corpus.split_assignments import split_policy

    policy = split_policy(project)
    rank = {
        source.id: (
            0
            if source.kind in CURATED_SOURCE_KINDS
            else 1
            if source.id.startswith("pes2o")
            else 2
        )
        for source in project.sources
    }
    total = 0
    for shard in prepared:
        with (
            (shard / "docs.jsonl").open("rb") as docs,
            (shard / "spans.jsonl").open("rb") as spans,
        ):
            for raw_doc, raw_span in zip(docs, spans, strict=True):
                doc = json.loads(raw_doc)
                span = json.loads(raw_span)
                if doc["document_id"] != span["record_id"]:
                    raise ValueError("prepared document/span identity mismatch")
                split = _split(doc, policy)
                cursor = connection.execute(
                    "INSERT INTO documents (document_id,payload,span,split,source_id,source_family,document_kind,source_rank) VALUES (?,?,?,?,?,?,?,?)",
                    (
                        doc["document_id"],
                        raw_doc.rstrip(b"\n"),
                        raw_span.rstrip(b"\n"),
                        split,
                        doc["source_id"],
                        doc["source_family"],
                        doc["document_kind"],
                        rank[doc["source_id"]],
                    ),
                )
                number = cursor.lastrowid
                origins = (
                    ("raw", doc["raw_content_sha256"]),
                    ("normalized", doc["content_sha256"]),
                    *(("canonical_origin", value) for value in _origin_keys(doc)),
                )
                connection.executemany(
                    "INSERT INTO origins(method,origin,document_number) VALUES (?,?,?)",
                    ((method, key, number) for method, key in origins),
                )
                total += 1
                progress.update(documents=1)
                if total % 10_000 == 0:
                    connection.commit()
    connection.commit()
    connection.execute(
        "CREATE INDEX origins_by_key ON origins(method,origin,document_number)"
    )
    connection.execute("CREATE INDEX documents_source ON documents(source_id)")
    connection.commit()
    return connection


def _deduplicate_v2(
    connection: sqlite3.Connection,
    groups_path: Path,
    *,
    diagnostics: Path,
    progress: BuildProgress,
    rejected_path: Path | None = None,
) -> tuple[int, int]:
    """Apply the native v2 raw/normalized policy without corpus-sized maps."""
    connection.execute("UPDATE documents SET representative_id=document_id")
    groups = 0
    with groups_path.open("wb") as output:
        for method in ("raw", "normalized"):
            keys = connection.execute(
                "SELECT origin FROM origins WHERE method=? GROUP BY origin "
                "HAVING COUNT(*)>1 ORDER BY origin",
                (method,),
            )
            for (key,) in keys:
                query = (
                    "SELECT d.document_id,d.split,d.payload FROM origins o "
                    "JOIN documents d ON d.rowid=o.document_number "
                    "WHERE o.method=? AND o.origin=? ORDER BY d.document_id"
                )
                first = None
                splits = set()
                signature = None
                incompatible = False
                count = 0
                for identifier, split, payload in connection.execute(
                    query, (method, key)
                ):
                    first = first or identifier
                    splits.add(split)
                    count += 1
                    if method == "normalized":
                        doc = json.loads(payload)
                        rights = doc["rights"]
                        current = canonical_json(
                            {
                                "license": doc["license"],
                                "redistribution": doc["redistribution"],
                                **{
                                    name: rights[name]
                                    for name in (
                                        "training_eligibility",
                                        "license_references",
                                        "notices",
                                        "training_restriction",
                                    )
                                },
                            }
                        )
                        if signature is not None and current != signature:
                            incompatible = True
                        signature = current
                # Stream even a single very large duplicate group, in canonical order.
                output.write(b'{"method":' + canonical_json(method) + b',"origins":[')
                for index, (identifier, _, _) in enumerate(
                    connection.execute(query, (method, key))
                ):
                    if index:
                        output.write(b",")
                    output.write(canonical_json(identifier))
                output.write(
                    b'],"representative":'
                    + canonical_json(first)
                    + b',"sha256":'
                    + canonical_json(key)
                    + b',"splits":'
                    + canonical_json(sorted(splits))
                    + b"}\n"
                )
                error = (
                    "cross-split exact text overlap"
                    if len(splits) > 1
                    else "duplicate source text has incompatible rights evidence"
                    if incompatible
                    else None
                )
                if error:
                    output.flush()
                    os.fsync(output.fileno())
                    # Preserve the complete observed groups without loading them.
                    diagnostics.parent.mkdir(parents=True, exist_ok=True)
                    with diagnostics.open("wb") as report:
                        report.write(b'{"duplicates":[')
                        with groups_path.open("rb") as rows:
                            for index, row in enumerate(rows):
                                if index:
                                    report.write(b",")
                                report.write(row.rstrip(b"\n"))
                        report.write(
                            b'],"error":' + canonical_json(error) + b',"rejected":['
                        )
                        if rejected_path is not None:
                            with rejected_path.open("rb") as rejected:
                                for index, row in enumerate(rejected):
                                    if index:
                                        report.write(b",")
                                    report.write(row.rstrip(b"\n"))
                        report.write(b"]}\n")
                        report.flush()
                        os.fsync(report.fileno())
                    raise ValueError(error)
                connection.execute(
                    "UPDATE documents SET representative_id=MIN(representative_id,?) "
                    "WHERE rowid IN (SELECT document_number FROM origins WHERE method=? AND origin=?)",
                    (first, method, key),
                )
                groups += 1
                progress.update(documents=count)
        output.flush()
        os.fsync(output.fileno())
    # Every link decreases lexicographically, so compression terminates without
    # an in-memory union table and matches the native v2 representative chain.
    while True:
        changed = connection.execute(
            "UPDATE documents SET representative_id=(SELECT p.representative_id "
            "FROM documents p WHERE p.document_id=documents.representative_id) "
            "WHERE representative_id<>(SELECT p.representative_id FROM documents p "
            "WHERE p.document_id=documents.representative_id)"
        ).rowcount
        if not changed:
            break
    dropped = 0
    for number, identifier, representative, payload in connection.execute(
        "SELECT rowid,document_id,representative_id,payload FROM documents ORDER BY rowid"
    ):
        doc = json.loads(payload)
        reason = (
            "metadata_only_front_matter"
            if (doc.get("structure") or {}).get("decision")
            == "exclude_lm_metadata_only"
            else "duplicate"
            if identifier != representative
            else None
        )
        dropped += reason is not None
        connection.execute(
            "UPDATE documents SET drop_reason=? WHERE rowid=?", (reason, number)
        )
    connection.commit()
    return groups, dropped


def _deduplicate(
    connection: sqlite3.Connection,
    groups_path: Path,
    *,
    diagnostics: Path,
    progress: BuildProgress,
    schema_version: int = 3,
    rejected_path: Path | None = None,
) -> tuple[int, int]:
    """Apply the declared duplicate policy with disk-backed input joins."""
    if schema_version == 2:
        return _deduplicate_v2(
            connection,
            groups_path,
            diagnostics=diagnostics,
            progress=progress,
            rejected_path=rejected_path,
        )
    if schema_version != 3:
        raise ValueError("unsupported disk deduplication policy")
    total = connection.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    parents = array("I", range(total + 1))

    def root(number: int) -> int:
        while parents[number] != number:
            parents[number] = parents[parents[number]]
            number = parents[number]
        return number

    methods = ("raw", "normalized", "canonical_origin")
    groups = 0
    with groups_path.open("wb") as output:
        for method in methods:
            keys = connection.execute(
                "SELECT origin FROM origins WHERE method=? GROUP BY origin HAVING COUNT(*)>1 ORDER BY origin",
                (method,),
            )
            for (key,) in keys:
                rows = list(
                    connection.execute(
                        """
                        SELECT d.rowid,d.document_id,d.split,d.source_rank,d.document_kind
                        FROM origins o JOIN documents d ON d.rowid=o.document_number
                        WHERE o.method=? AND o.origin=? ORDER BY d.document_id
                        """,
                        (method, key),
                    )
                )
                splits = {row[2] for row in rows}
                heldout = splits - {"train"}
                if len(heldout) > 1:
                    _write_json(
                        diagnostics,
                        {
                            "error": "cross-split exact text overlap",
                            "method": method,
                            "sha256": canonical_digest(key),
                        },
                    )
                    raise ValueError("cross-split exact text overlap")
                priority = next(iter(heldout)) if heldout else None
                selected = min(
                    rows,
                    key=lambda row: (
                        row[2] != priority if priority else False,
                        row[3],
                        row[4] != "paper",
                        row[1],
                    ),
                )
                _output(
                    output,
                    {
                        "method": method,
                        "sha256": (
                            hashlib.sha256(key.encode()).hexdigest()
                            if method == "canonical_origin"
                            else key
                        ),
                        "origins": [row[1] for row in rows],
                        "splits": sorted(splits),
                        "representative": selected[1],
                    },
                )
                first = root(rows[0][0])
                for row in rows[1:]:
                    parents[root(row[0])] = first
                groups += 1
                progress.update(documents=len(rows))
        output.flush()
        os.fsync(output.fileno())

    # A connected component can link distinct raw, normalized and page origins.
    # Whole components select one deterministic representative, preserving the
    # validation/test isolation gate even for transitive overlaps.
    winner: dict[int, tuple[Any, ...]] = {}
    heldout_mask: dict[int, int] = defaultdict(int)
    cleaning_mask: dict[int, int] = defaultdict(int)
    for number, split, payload in connection.execute(
        "SELECT rowid,split,payload FROM documents ORDER BY rowid"
    ):
        component = root(number)
        heldout_mask[component] |= (
            1 if split == "validation" else 2 if split == "test" else 0
        )
        cleaning_mask[component] |= (
            2
            if (json.loads(payload).get("structure") or {}).get("decision")
            == "exclude_lm_metadata_only"
            else 1
        )
    if any(mask == 3 for mask in heldout_mask.values()):
        _write_json(
            diagnostics, {"error": "validation/test page or paper origin overlap"}
        )
        raise ValueError("validation/test page or paper origin overlap")
    if any(mask == 3 for mask in cleaning_mask.values()):
        message = "discordant metadata-only cleaning decisions in duplicate component"
        _write_json(diagnostics, {"error": message})
        raise ValueError(message)
    for number, identifier, split, source_rank, kind in connection.execute(
        "SELECT rowid,document_id,split,source_rank,document_kind FROM documents ORDER BY rowid"
    ):
        component = root(number)
        preferred_split = (
            "validation"
            if heldout_mask[component] == 1
            else "test"
            if heldout_mask[component] == 2
            else None
        )
        choice = (
            split != preferred_split if preferred_split else False,
            source_rank,
            kind != "paper",
            identifier,
        )
        if component not in winner or choice < winner[component][:4]:
            winner[component] = (*choice, identifier, split)
    dropped = 0
    updates = []
    for number, identifier, split, payload in connection.execute(
        "SELECT rowid,document_id,split,payload FROM documents ORDER BY rowid"
    ):
        selected = winner[root(number)][4]
        retained_split = winner[root(number)][5]
        reason = (
            "metadata_only_front_matter"
            if (json.loads(payload).get("structure") or {}).get("decision")
            == "exclude_lm_metadata_only"
            else "contaminated_heldout"
            if selected != identifier and split == "train" and retained_split != "train"
            else "duplicate"
            if selected != identifier
            else None
        )
        dropped += reason is not None
        updates.append((selected, reason, number))
        if len(updates) >= 10_000:
            connection.executemany(
                "UPDATE documents SET representative_id=?,drop_reason=? WHERE rowid=?",
                updates,
            )
            connection.commit()
            updates.clear()
    if updates:
        connection.executemany(
            "UPDATE documents SET representative_id=?,drop_reason=? WHERE rowid=?",
            updates,
        )
        connection.commit()
    return groups, dropped


def _write_audit(
    target: Path, groups: Path, rejected: Path, *, dropped: int, leakage=(), warnings=()
) -> None:
    with target.open("wb") as output:
        output.write(b'{"drop_count":' + str(dropped).encode() + b',"duplicates":[')
        with groups.open("rb") as source:
            for index, row in enumerate(source):
                if index:
                    output.write(b",")
                output.write(row.rstrip(b"\n"))
        output.write(b'],"leakage":' + canonical_json(list(leakage)) + b',"rejected":[')
        with rejected.open("rb") as source:
            for index, row in enumerate(source):
                if index:
                    output.write(b",")
                output.write(row.rstrip(b"\n"))
        output.write(b'],"warnings":' + canonical_json(list(warnings)) + b"}\n")
        output.flush()
        os.fsync(output.fileno())


def _document_lineage(
    doc: dict[str, Any], source_origins: dict[str, str]
) -> dict[str, Any]:
    record_id = doc["document_id"]
    representative = doc["representative_id"]
    row = {
        "record_id": record_id,
        "record_kind": "document",
        "transform_id": doc.get("metadata", {}).get(
            "normalizer", "normalizer-nfc-markdown-v1"
        ),
        "parent_document_ids": [record_id],
        "original_parent_document_ids": [record_id],
        "representative_id": representative,
        "drop_reason": doc["drop_reason"],
        "source_family_ids": [doc["source_family"]],
        "split": doc["split"],
        "representative_parent_document_ids": [representative],
        "origin": (
            HUMAN_ORIGIN
            if source_origins[doc["source_id"]] == HUMAN_ORIGIN
            else SOURCE_ORIGIN
        ),
        "modalities": [doc["modality"]],
        "origin_schema_version": 1,
        "domains": doc["domains"],
        "shape": shape_for_record(
            "document",
            "raw_document",
            domains=doc["domains"],
            parent_document_ids=[record_id],
        ),
        "verification": verification(
            "schema_validated",
            doc.get("metadata", {}).get("normalizer", "normalizer_v1"),
            {"schema_id": "normalized_document_v1"},
        ),
        "validation_status": "schema_validated",
        "rendered_sha256": rendered_digest(doc["text"], text=True),
    }
    validate_lineage(row, documents={record_id: doc})
    return row


def _write_rejected(target: Path, source_events: list[Path | dict[str, Any]]) -> None:
    with target.open("wb") as output:
        for event in source_events:
            if isinstance(event, Path):
                with (event / "rejected.jsonl").open("rb") as source:
                    for line in source:
                        output.write(line)
            else:
                _output(output, event)
        output.flush()
        os.fsync(output.fileno())


class _SQLBuildRows:
    """The ordinary builder's ordered inputs to the shared release writer."""

    def __init__(self, connection):
        self.connection = connection

    def spans(self):
        for (span,) in self.connection.execute(
            "SELECT span FROM documents ORDER BY rowid"
        ):
            yield span

    def documents(self):
        yield from self.connection.execute(
            "SELECT payload,split,representative_id,drop_reason FROM documents ORDER BY document_id"
        )

    def family_splits(self):
        for (family,) in self.connection.execute(
            "SELECT source_family FROM documents "
            "GROUP BY source_family HAVING COUNT(DISTINCT split)>1 ORDER BY source_family"
        ):
            yield (
                family,
                [
                    row[0]
                    for row in self.connection.execute(
                        "SELECT DISTINCT split FROM documents WHERE source_family=? ORDER BY split",
                        (family,),
                    )
                ],
            )


def _emit_build(
    *,
    connection: sqlite3.Connection,
    staging: Path,
    project: Any,
    lock: dict[str, Any],
    identity: dict[str, Any],
    build_id: str,
    transform: Any,
    prepared: list[Path],
    rights_files: list[dict[str, Any]],
    source_events: list[Path | dict[str, Any]],
    groups_path: Path,
    group_count: int,
    dropped: int,
    progress: BuildProgress,
    ordered_rows=None,
    descriptive_census=True,
) -> None:
    from klab.corpus.measurement import summarize_release
    from klab.corpus.pipeline import _model

    ordered_rows = ordered_rows or _SQLBuildRows(connection)
    sources, rights_report = _sources_and_rights(project, lock, rights_files)
    source_origins = {source.id: source.origin for source in project.sources}
    release_spec = _model(project.release)
    stage_spec = _model(transform)
    stage_id = stage_spec["id"]
    selected_sources = set(stage_spec["inputs"])
    mixture = release_spec.get("mixture", {})
    source_counts: Counter[str] = Counter()
    kinds: Counter[str] = Counter()
    source_scale: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    domain_scale: dict[str, dict[str, int]] = defaultdict(
        lambda: {"documents": 0, "utf8_bytes": 0}
    )
    split_counts: Counter[str] = Counter()
    lm_counts: Counter[str] = Counter()
    count = 0
    stage_count = 0
    stage_digest = hashlib.sha256()
    stage_digest.update(b"[" + canonical_json(stage_id) + b",[")

    with ExitStack() as stack:

        def writer(name: str) -> Any:
            path = staging / name
            path.parent.mkdir(parents=True, exist_ok=True)
            return stack.enter_context(path.open("wb"))

        documents = writer("documents.jsonl")
        spans = writer("spans.jsonl")
        lineage = writer("lineage.jsonl")
        stage = writer(f"stages/{stage_id}.jsonl")
        views = {split: writer(f"lm/{split}.jsonl") for split in _SPLITS}
        links = {split: writer(f"lm/{split}.lineage.jsonl") for split in _SPLITS}
        for path in (
            "lexical/candidates.jsonl",
            "semantic/candidates.jsonl",
            "scenarios.jsonl",
            "generations.jsonl",
            "tool_episodes.jsonl",
            "chat/records.jsonl",
        ):
            writer(path)
        for split in _SPLITS:
            writer(f"chat/{split}.jsonl")
            writer(f"chat/{split}.lineage.jsonl")

        # Source spans retain acquisition order; documents and views use ID order.
        for span in ordered_rows.spans():
            spans.write(span + b"\n")
        for raw_doc, split, representative, reason in ordered_rows.documents():
            doc = json.loads(raw_doc)
            doc["split"] = split
            doc["representative_id"] = representative
            doc["drop_reason"] = reason
            _output(documents, doc)
            _output(lineage, _document_lineage(doc, source_origins))
            count += 1
            record_id = doc["document_id"]
            origin = (
                HUMAN_ORIGIN
                if source_origins[doc["source_id"]] == HUMAN_ORIGIN
                else SOURCE_ORIGIN
            )
            if reason is not None:
                continue
            split_counts[split] += 1
            source_counts[doc["source_id"]] += 1
            kinds[doc["document_kind"]] += 1
            text = doc["text"]
            byte_size = len(text.encode("utf-8"))
            for domain in doc["domains"]:
                domain_scale[domain]["documents"] += 1
                domain_scale[domain]["utf8_bytes"] += byte_size
            counts = source_scale[doc["source_id"]].setdefault(
                split,
                {
                    "documents": 0,
                    "utf8_bytes": 0,
                    "characters": 0,
                    "whitespace_words": 0,
                },
            )
            counts["documents"] += 1
            counts["utf8_bytes"] += byte_size
            counts["characters"] += len(text)
            counts["whitespace_words"] += len(text.split())

            selected_for_stage = (
                not selected_sources or doc["source_id"] in selected_sources
            )
            if selected_for_stage:
                output = {"record_id": record_id, "text": text, "split": split}
                encoded = canonical_json(output)
                stage.write(encoded + b"\n")
                if stage_count:
                    stage_digest.update(b",")
                stage_digest.update(encoded)
                stage_count += 1
            if (
                selected_for_stage
                and (
                    release_spec.get("include_shapes") is None
                    or "raw_document" in release_spec["include_shapes"]
                )
                and (
                    release_spec.get("include_origins") is None
                    or origin in release_spec["include_origins"]
                )
            ):
                _output(views[split], {"text": text})
                _output(links[split], {"record_id": record_id, "split": split})
                lm_counts[split] += 1
                progress.update(output_records=1)
        for handle in (
            documents,
            spans,
            lineage,
            stage,
            *views.values(),
            *links.values(),
        ):
            handle.flush()
            os.fsync(handle.fileno())

    stage_digest.update(b"]]")
    stage_path = staging / "stages" / f"{stage_id}.jsonl"
    stages = [
        {
            "id": stage_id,
            "kind": stage_spec["kind"],
            "version": stage_spec["version"],
            "parameters_sha256": canonical_digest(stage_spec["parameters"]),
            "inputs": stage_spec["inputs"],
            "implementation_sha256": identity["implementation_sha256"],
            "output_sha256": sha256_file(stage_path),
            "output_count": stage_count,
            "output_id": stage_digest.hexdigest(),
        }
    ]
    _write_rejected(staging / "rejected.jsonl", source_events)
    warnings = []
    leakage = []
    if project.release.schema_version == 2:
        # LM-only lineages have no generated families. Source-family overlap is
        # descriptive unless that family is the declared split policy unit.
        for family, splits in ordered_rows.family_splits():
            leakage.append(
                {
                    "family_type": "source_family_ids",
                    "family_id": family,
                    "splits": splits,
                }
            )
            if project.splits.unit in {"source_document_family", "source_repository"}:
                _write_json(
                    staging.parent / "diagnostics" / f"{build_id}.json",
                    {"error": "shared split policy unit", "leakage": leakage},
                )
                raise ValueError(f"split policy unit {family} spans splits")
            warnings.append(f"source_family_ids {family} overlaps {', '.join(splits)}")
    _write_audit(
        staging / "audit.json",
        groups_path,
        staging / "rejected.jsonl",
        dropped=dropped,
        leakage=leakage,
        warnings=warnings,
    )
    _write_json(staging / "splits.json", project.splits.model_dump(mode="json"))
    _write_json(staging / "sources.json", sources)
    _write_json(staging / "license-report.json", rights_report)

    train_scale = {
        key: sum(split.get("train", {}).get(key, 0) for split in source_scale.values())
        for key in ("documents", "utf8_bytes", "characters", "whitespace_words")
    }
    generation_statuses = {"oracle_verified": 0}
    report = {
        "schema_version": project.release.schema_version,
        "corpus_id": project.config.id,
        "requested_mixture": mixture,
        "actual_mixture": {
            domain: {
                "documents": domain_scale[domain]["documents"],
                "utf8_bytes": domain_scale[domain]["utf8_bytes"],
                "actual_tokens": None,
                "token_count_reason": "tokenizer_not_declared",
            }
            for domain in sorted(set(mixture) | set(domain_scale))
        },
        "document_count": count,
        "kept_document_count": count - dropped,
        "split_counts": {split: split_counts[split] for split in _SPLITS},
        "view_counts": {
            "lm": {split: lm_counts[split] for split in _SPLITS},
            "chat": {split: 0 for split in _SPLITS},
        },
        "source_counts": dict(source_counts),
        "source_concentration": {
            source: amount / (count - dropped)
            for source, amount in sorted(source_counts.items())
        }
        if count != dropped
        else {},
        "kind_counts": dict(kinds),
        "lexical_count": 0,
        "semantic_count": 0,
        "chat_count": 0,
        "generation_statuses": generation_statuses,
        "generation_validation_ratios": {},
        "generator_count": 0,
        "scenario_count": 0,
        "dedupe": {"groups": group_count, "dropped": dropped},
        "warnings": warnings,
        "token_count_reason": "tokenizer_not_declared",
        "source_scale": {
            source: dict(sorted(splits.items()))
            for source, splits in sorted(source_scale.items())
        },
        "train_source_scale": {
            **train_scale,
            "utf8_bytes_div4_proxy": train_scale["utf8_bytes"] // 4,
            "actual_tokens": None,
            "token_count_reason": "final_tokenizer_not_fitted",
            "proxy_warning": (
                "Whitespace words and UTF-8 bytes/4 are tokenizer-independent "
                "descriptors, not measured token counts."
            ),
        },
        "rights": {
            key: value
            for key, value in rights_report.items()
            if key
            in {
                "training_eligibility",
                "spdx_expressions",
                "redistribution_modes",
                "unresolved_rights_files",
                "weight_license_status",
                "publication_mode",
                "training_use_policy",
            }
        },
    }
    if project.release.schema_version == 2:
        report.pop("source_scale")
        report.pop("train_source_scale")
    progress.update(phase="measuring")
    report["measurement"] = (
        summarize_release(staging, release_spec=release_spec, tokenizer=None)
        if descriptive_census
        else {
            "status": "deferred",
            "reason": "Descriptive lineage census is outside the selected completed local preparation scope",
            "token_count_reason": "tokenizer_not_declared",
        }
    )
    _write_json(staging / "report.json", report)
    (staging / "report.md").write_text(
        f"# Corpus {project.config.id}\n\nDocuments: {count - dropped} retained / {count} total.\n\n"
        "No tokenizer declared; token counts unavailable.\n",
        encoding="utf-8",
    )
    _write_json(
        staging / "build.json",
        {
            "schema_version": 1,
            "build_id": build_id,
            "identity": identity,
            "stages": stages,
            "files": {
                str(path.relative_to(staging)): {
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            },
            "snapshots": [
                {"source_id": source_id, "sha256": entry["snapshot_sha256"]}
                for source_id, entry in sorted(lock["sources"].items())
                if entry.get("snapshot_path")
            ],
        },
    )


def _verify_admitted_inventory(
    connection: sqlite3.Connection, admissions: dict
) -> None:
    """Compare qualified and observed source identities without materialized sets."""
    from klab.corpus.admission_coverage import AdmissionCoverage

    if not admissions:
        return
    with AdmissionCoverage(connection) as coverage:
        for (source, path), admitted in admissions.items():
            for index, choice in admitted.get("decisions", {}).items():
                if choice["decision"] == "qualify":
                    coverage.expect(source, path, index)
        for source, raw in connection.execute(
            "SELECT source_id,payload FROM documents"
        ):
            doc = json.loads(raw)
            path = doc["source_location"].split("#", 1)[0]
            if "decisions" in admissions.get((source, path), {}):
                coverage.observe(source, path, doc["metadata"]["source_row_index"])
        coverage.verify()


def build_large(
    project: Any,
    workspace: Path,
    lock: dict[str, Any],
    identity: dict[str, Any],
    build_id: str,
    target: Path,
    progress: BuildProgress,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    record_admission: dict | None = None,
    admission_path: Path | None = None,
) -> Path:
    """Build LM-only v2/v3 inputs using their declared admission/dedup policies."""
    record_admission = record_admission or {}
    from klab.corpus.acquisition import verify_snapshot

    root = workspace / "builds"
    root.mkdir(parents=True, exist_ok=True)
    prepared_root = root / "prepared" / "normalized-source-v2"
    prepared_root.mkdir(parents=True, exist_ok=True)
    sources = sorted(project.sources, key=lambda source: source.id)
    transform = project.transforms[0]
    rights_files: list[dict[str, Any]] = []
    source_events: list[Path | dict[str, Any]] = []
    prepared: list[Path] = []
    inputs: list[tuple[Any, dict[str, Any], Path, dict[str, Any], dict[str, Any]]] = []
    total_bytes = 0
    progress.record("verifying_snapshots")
    for source in sources:
        lock_row = lock["sources"].get(source.id)
        if (
            source.redistribution == "rejected"
            or not lock_row
            or not lock_row.get("snapshot_path")
        ):
            continue
        snapshot_path = Path(lock_row["snapshot_path"])
        snapshot = verify_snapshot(
            snapshot_path, proof_store=proof_store, verification_mode=verification_mode
        )
        nested_path = source.rights.nested_metadata_path if source.rights else None
        nested_metadata = (
            {
                nested_path: json.loads(
                    (snapshot_path / "files" / nested_path).read_text(encoding="utf-8")
                )
            }
            if nested_path
            else {}
        )
        for file in sorted(snapshot["files"], key=lambda item: item["path"]):
            inputs.append(
                (
                    source,
                    file,
                    snapshot_path / "files" / file["path"],
                    lock_row,
                    nested_metadata,
                )
            )
            if file["path"] != nested_path:
                total_bytes += file["size"]
    progress.expected_input_bytes = total_bytes
    progress.update(phase="normalizing")
    eligible_jsonl = {
        path
        for source, file, path, lock_row, nested_metadata in inputs
        if source.kind in ROW_SOURCE_KINDS
        and file["path"].endswith(".jsonl")
        and file["size"] >= _PROCESS_SHARD_MIN_BYTES
        and file["path"] not in nested_metadata
        # Cached sources reach exact identity/rights checking in _prepare_file.
        # Do not pre-scan their bodies merely to choose parallel workers.
        and not (prepared_root / source.id).exists()
    }
    if record_admission:
        # Operation-local SQLite decisions must stay in their owning process.
        eligible_jsonl.clear()
    workers = _process_shard_workers(len(eligible_jsonl))
    if workers > 1:
        eligible_jsonl = {path for path in eligible_jsonl if _bounded_jsonl_shard(path)}
        workers = _process_shard_workers(len(eligible_jsonl))
    pool = (
        ProcessPoolExecutor(max_workers=workers) if workers > 1 else nullcontext(None)
    )
    pending: deque[
        tuple[Future[tuple[Path, int, int, int, bool]], Any, dict[str, Any]]
    ] = deque()

    def finish_one() -> None:
        future, source, file = pending.popleft()
        try:
            shard, documents, input_bytes, output_records, reused = future.result()
        except _SourceParseError as exc:
            source_events.append(
                {"source_id": source.id, "path": file["path"], "reason": str(exc)}
            )
            progress.update(
                documents=exc.documents,
                input_bytes=file["size"] + exc.input_bytes,
                output_records=exc.output_records,
            )
            progress.record("source_rejected")
            return
        progress.update(
            documents=documents, input_bytes=input_bytes, output_records=output_records
        )
        if reused:
            progress.record("reused_verified_shard")
        prepared.append(shard)
        source_events.append(shard)

    with pool as executor:
        for source, file, path, lock_row, nested_metadata in inputs:
            name = file["path"]
            if name in nested_metadata:
                while pending:
                    finish_one()
                rights_files.append(
                    {
                        "source_id": source.id,
                        "path": name,
                        "sha256": file["sha256"],
                        "size": file["size"],
                        "role": "license_metadata",
                        "canonical_uri": source.canonical_uri,
                        "revision": source.revision,
                    }
                )
                continue
            decision = (
                resolve_file_rights(
                    source.rights,
                    name,
                    _read_rights_header(path, name),
                    nested_metadata=nested_metadata,
                    prospective_private_research=source.schema_version == 3,
                )
                if source.rights
                else None
            )
            admitted = record_admission.get((source.id, name))
            if admitted is not None:
                decision = admitted["rights"]
            if decision:
                rights_files.append(
                    {
                        "source_id": source.id,
                        "path": name,
                        "sha256": file["sha256"],
                        "size": file["size"],
                        "role": "document",
                        "canonical_uri": source.canonical_uri,
                        "revision": source.revision,
                        "license_url": source.license_url,
                        "rights": decision.model_dump(mode="json"),
                        **(
                            {
                                "admission_policy_id": admitted["policy_id"],
                                "admission_policy_sha256": admitted["policy_sha256"],
                                "admission_license_label": admitted["license_label"],
                                "admission_policy_spdx_expression": admitted[
                                    "policy_spdx_expression"
                                ],
                            }
                            if admitted is not None
                            else {}
                        ),
                    }
                )
                if decision.training_eligibility not in {
                    "eligible",
                    "eligible_with_obligations",
                }:
                    while pending:
                        finish_one()
                    source_events.append(
                        {
                            "source_id": source.id,
                            "path": name,
                            "reason": f"rights {decision.training_eligibility}: {decision.reason}",
                        }
                    )
                    progress.update(input_bytes=file["size"])
                    continue
            kwargs = {
                "source": source,
                "file": file,
                "source_path": path,
                "decision": decision,
                "snapshot_sha": lock_row["snapshot_sha256"],
                "prepared_root": prepared_root,
                "build_id": build_id,
                "record_admission": admitted,
                "admission_sha256": project.release.record_admission.sha256.lower()
                if admitted is not None
                else None,
                "normalizer_version": project.release.normalizer
                or "normalizer-nfc-markdown-v1",
            }
            if executor is not None and path in eligible_jsonl:
                pending.append(
                    (executor.submit(_prepare_shard, **kwargs), source, file)
                )
                if len(pending) >= workers:
                    finish_one()
                continue
            while pending:
                finish_one()
            try:
                shard = _prepare_file(**kwargs, progress=progress)
            except _SourceParseError as exc:
                source_events.append(
                    {"source_id": source.id, "path": name, "reason": str(exc)}
                )
                progress.update(input_bytes=file["size"])
                progress.record("source_rejected")
                continue
            prepared.append(shard)
            source_events.append(shard)
        while pending:
            finish_one()

    progress.update(phase="indexing")
    index_root = root / "indices" / build_id
    index_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="index-", dir=index_root) as index_dir:
        connection = _index_prepared(
            Path(index_dir) / "documents.sqlite", prepared, project, progress
        )
        try:
            _verify_admitted_inventory(connection, record_admission)
            progress.update(phase="deduplicating")
            groups_path = Path(index_dir) / "groups.jsonl"
            rejected_path = Path(index_dir) / "rejected.jsonl"
            _write_rejected(rejected_path, source_events)
            group_count, dropped = _deduplicate(
                connection,
                groups_path,
                diagnostics=root / "diagnostics" / f"{build_id}.json",
                progress=progress,
                schema_version=project.release.schema_version,
                rejected_path=rejected_path,
            )
            progress.update(phase="writing_views")
            with tempfile.TemporaryDirectory(prefix=".build-", dir=root) as temporary:
                staging = Path(temporary)
                from klab.corpus.split_assignments import copy_assignments

                copy_assignments(project, staging)
                if admission_path is not None:
                    copy = staging / "record-admission.json"
                    shutil.copyfile(admission_path, copy)
                    if (
                        sha256_file(copy)
                        != project.release.record_admission.sha256.lower()
                    ):
                        raise ValueError("record admission changed during build")
                _emit_build(
                    connection=connection,
                    staging=staging,
                    project=project,
                    lock=lock,
                    identity=identity,
                    build_id=build_id,
                    transform=transform,
                    prepared=prepared,
                    rights_files=rights_files,
                    source_events=source_events,
                    groups_path=groups_path,
                    group_count=group_count,
                    dropped=dropped,
                    progress=progress,
                )
                progress.update(phase="publishing")
                staging.rename(target)
        finally:
            connection.close()
    progress.record("complete")
    return target
