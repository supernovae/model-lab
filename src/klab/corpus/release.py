"""Immutable publication, verification and read-only corpus inspection."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import tempfile
from collections import Counter
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.bounded_metadata import metadata_operation, read_metadata
from klab.corpus.jsonl_records import iter_rows, read_rows
from klab.corpus.project import NormalizerOptions
from klab.corpus.release_copy import copy_inventory
from klab.corpus.release_verify import (
    chunk_ranges,
    current_settings,
    hash_inventory,
    line_boundaries,
    open_progress,
    range_records,
    run_tasks,
    scratch_root,
)
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.corpus.progress import BuildProgress
    from klab.verification_proofs import ProofStore, VerificationMode


@dataclass(frozen=True)
class _VerifiedRelease:
    """Declared-policy proof valid only while its typed closure is unchanged."""

    manifest: dict[str, Any]
    closure_key: tuple[object, ...]


_verified_releases: ContextVar[dict[Path, _VerifiedRelease] | None] = ContextVar(
    "verified_releases", default=None
)


@contextmanager
def _verification_operation():
    """Share full authentication, snapshots included, with nested consumers, never across operations."""
    if _verified_releases.get() is not None:
        yield
        return
    token, snapshots = _verified_releases.set({}), _snapshot_receipts.set({})
    try:
        yield
    finally:
        _snapshot_receipts.reset(snapshots)
        _verified_releases.reset(token)


_snapshot_receipts: ContextVar[dict[Path, dict[str, Any]] | None] = ContextVar(
    "release_snapshot_receipts", default=None
)


def _verified_snapshot(
    path: Path, *, proof_store: ProofStore | None, verification_mode: VerificationMode
) -> dict[str, Any]:
    from klab.corpus.acquisition import verify_snapshot

    memo = _snapshot_receipts.get()
    if memo is not None and path in memo:
        return memo[path]
    receipt = verify_snapshot(
        path, proof_store=proof_store, verification_mode=verification_mode
    )
    if memo is not None:
        memo[path] = receipt
    return receipt


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise ValueError("unsafe release artifact path")
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError("release artifact escapes directory")
    return target


def _files(
    root: Path,
    inventory: dict[str, Any],
    progress: BuildProgress | None = None,
    hashed: frozenset[str] = frozenset(),
) -> None:
    if not isinstance(inventory, dict):
        raise TypeError("missing artifact inventory")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if actual != set(inventory):
        raise ValueError("release artifact inventory mismatch")
    if progress is not None:
        progress.start_phase(
            "hash_files",
            expected_input_bytes=sum(entry["size"] for entry in inventory.values()),
        )
    hash_inventory(root, inventory, _safe, progress, skip=hashed)


def _verify_rights_files(
    root: Path,
    sources: dict[str, dict[str, Any]],
    snapshots: dict[str, dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Bind prospective file decisions to the pinned bytes and metadata."""
    from klab.corpus.rights import (
        RightsPolicy,
        reported_spdx_expression,
        resolve_file_rights,
        verify_record_admission,
    )

    report = _load(root / "license-report.json")
    if report.get("schema_version") != 1 or report.get("sources") != list(
        sources.values()
    ):
        raise ValueError("prospective rights report/source mismatch")
    rows = report["files"]
    indexed = {(row["source_id"], row["path"]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError("duplicate rights file entry")
    expected = {
        (source_id, item["path"])
        for source_id, snapshot in snapshots.items()
        for item in snapshot["files"]
    }
    if set(indexed) != expected:
        raise ValueError("rights file inventory differs from pinned snapshots")
    identity = _load(
        root / ("build.json" if (root / "build.json").exists() else "manifest.json")
    ).get("identity")
    if identity is None:
        identity = _load(root / "manifest.json")["build_identity"]
    admission_reference = identity["release"].get("record_admission")
    admission: dict[tuple[str, str], dict[str, Any]] = {}
    if admission_reference is not None:
        admission_path = root / "record-admission.json"
        if (
            admission_path.is_symlink()
            or sha256_file(admission_path) != admission_reference["sha256"].lower()
        ):
            raise ValueError("record admission reference differs from frozen bytes")
        admission = verify_record_admission(
            read_metadata(admission_path),
            {source_id: snapshots[source_id]["declaration"] for source_id in sources},
            snapshots,
            root.parent.parent / "snapshots",
        )
    elif (root / "record-admission.json").exists():
        raise ValueError("undeclared record admission artifact")
    for source_id, snapshot in snapshots.items():
        source = sources[source_id]
        policy = RightsPolicy.model_validate(source["rights_policy"])
        nested_path = policy.nested_metadata_path
        folder = (
            root.parent.parent
            / "snapshots"
            / source_id
            / source["snapshot_sha256"]
            / "files"
        )
        metadata = (
            {nested_path: _load(_safe(folder, nested_path))} if nested_path else {}
        )
        for file in snapshot["files"]:
            path = file["path"]
            recorded = indexed[source_id, path]
            if any(
                recorded.get(key) != value
                for key, value in (
                    ("sha256", file["sha256"]),
                    ("size", file["size"]),
                    ("canonical_uri", source["canonical_uri"]),
                    ("revision", source["revision"]),
                )
            ):
                raise ValueError("rights source file attribution mismatch")
            if path == nested_path:
                if recorded["role"] != "license_metadata" or "rights" in recorded:
                    raise ValueError("invalid rights metadata role")
                continue
            if recorded["role"] not in {"document", "transform_input"}:
                raise ValueError("unexpected rights file role")
            # The resolver examines only the first 30 lines (and no lines for
            # dataset shards). Never materialize a whole source shard here.
            with _safe(folder, path).open("rb") as stream:
                raw = (
                    b""
                    if path.endswith((".jsonl", ".json", ".parquet"))
                    else b"".join(stream.readline() for _ in range(30))
                )
            admitted = admission.get((source_id, path))
            resolved = (
                admitted["rights"]
                if admitted is not None
                else resolve_file_rights(
                    policy,
                    path,
                    raw,
                    nested_metadata=metadata,
                    prospective_private_research=source.get(
                        "explicit_training_restriction"
                    )
                    == "none_found",
                )
            )
            if recorded["rights"] != resolved.model_dump(mode="json") or (
                recorded.get("license_url") != source["license_url"]
            ):
                raise ValueError("rights decision differs from pinned file evidence")
            if admitted is not None:
                if "admission_policy_spdx_expression" not in recorded or any(
                    recorded.get(key) != value
                    for key, value in (
                        ("admission_policy_id", admitted["policy_id"]),
                        ("admission_policy_sha256", admitted["policy_sha256"]),
                        ("admission_license_label", admitted["license_label"]),
                        (
                            "admission_policy_spdx_expression",
                            admitted["policy_spdx_expression"],
                        ),
                    )
                ):
                    raise ValueError("file admission policy attribution mismatch")
                if "decisions" in admitted:
                    recorded["_admission_decisions"] = admitted["decisions"]
                else:
                    recorded["_admission_file_decision"] = admitted["decision"]
            elif any(key.startswith("admission_") for key in recorded):
                raise ValueError("undeclared file admission policy")
    if admission:
        expected_spdx = Counter(
            reported_spdx_expression(
                row, sources[row["source_id"]]["rights_policy"]["spdx_expression"]
            )
            for row in rows
            if row["role"] == "document"
        )
        if report.get("spdx_expressions") != dict(expected_spdx):
            raise ValueError("admission SPDX summary differs from reviewed file policy")
    return indexed


def _release_normalizer(identity: dict[str, Any]) -> NormalizerOptions:
    return NormalizerOptions.model_validate(identity["release"]["normalizer"])


def _verify_structure_replay(
    doc: dict[str, Any], raw_text: str, *, options: NormalizerOptions
) -> None:
    """Recreate the declared cleaning from the authenticated source text."""
    from klab.corpus.pipeline import _normalized
    from klab.corpus.structure_cleaning import clean_structure

    original = _normalized(raw_text)
    if (doc.get("structure") or {}).get("normalizer") != options.identity:
        raise ValueError("structured document normalizer mismatch")
    replay = clean_structure(original, source_id=doc["source_id"], options=options)
    if (
        replay["text"] != doc["text"]
        or replay["structure"] != doc.get("structure")
        or (
            "#row=" in doc["source_location"]
            and hashlib.sha256(original.encode()).hexdigest()
            != doc["raw_content_sha256"]
        )
    ):
        raise ValueError("structured document replay mismatch")


def _verify_cleaning_decision(doc: dict[str, Any], options: NormalizerOptions) -> None:
    excluded = (
        options.structure_cleaning
        and (doc.get("structure") or {}).get("decision") == "exclude_lm_metadata_only"
    )
    if (doc.get("drop_reason") == "metadata_only_front_matter") != excluded:
        raise ValueError("metadata-only cleaning decision mismatch")


def _verify_source_snapshots(
    root: Path,
    sources: dict[str, dict[str, Any]],
    *,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> dict[str, dict[str, Any]]:
    """Authenticate each source's snapshot and bind the source row to its declaration."""
    snapshots: dict[str, dict[str, Any]] = {}
    for source in sources.values():
        snapshot_id = source["snapshot_sha256"]
        if not snapshot_id:
            continue
        snapshot = _verified_snapshot(
            root.parent.parent / "snapshots" / source["id"] / snapshot_id,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        snapshots[source["id"]] = snapshot
        declaration = snapshot["declaration"]
        if any(
            source[key] != declaration[key]
            for key in (
                "id",
                "kind",
                "canonical_uri",
                "revision",
                "license",
                "source_family",
            )
        ) or source.get("origin", "primary_source") != declaration.get(
            "origin", "primary_source"
        ):
            raise ValueError("source declaration attribution mismatch")
        if (
            source["rights_policy"] != declaration["rights"]
            or source["license_url"] != declaration["license_url"]
            or source["redistribution"] != declaration["rights"]["redistribution_mode"]
        ):
            raise ValueError("prospective source rights declaration mismatch")
        if source.get("explicit_training_restriction") != declaration.get(
            "explicit_training_restriction"
        ):
            raise ValueError(
                "source training restriction state differs from declaration"
            )
    return snapshots


def _streaming_rows(identity: dict[str, Any]) -> bool:
    """Whether the bounded-memory row validator applies (no reviewed admission)."""
    return (
        identity["release"].get("record_admission") is None
        and not identity["release"]["chat"]["selected"]
        and all(spec["kind"] == "lm_text" for spec in identity["transforms"])
        and identity["release"].get("fraction") is None
    )


# Lines per parallel LM-view task; both files of a view are cut at the same line.
VIEW_LINES = 50_000


def _read_json(stream, offset: int, length: int) -> Any:
    stream.seek(offset)
    return json.loads(stream.read(length))


def _readonly(database: str) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{database}?mode=ro", uri=True)


def _span_rows(spans: str, start: int, end: int):
    rows = [
        (span["record_id"], span["raw_path"], span["line_start"], offset, length)
        for span, offset, length in range_records(Path(spans), start, end)
    ]
    return rows, end - start


def _document_rows(
    span_database: str,
    documents: str,
    start: int,
    end: int,
    options: NormalizerOptions,
):
    from klab.corpus.pipeline import _origin_keys

    db = _readonly(span_database)
    try:
        rows, keys = [], []
        for doc, offset, length in range_records(Path(documents), start, end):
            if (
                doc["content_sha256"]
                != hashlib.sha256(doc["text"].encode("utf-8")).hexdigest()
            ):
                raise ValueError("document content digest mismatch")
            _verify_cleaning_decision(doc, options)
            span = db.execute(
                "SELECT raw_path, line_start, byte_offset, byte_length FROM spans WHERE id=?",
                (doc["document_id"],),
            ).fetchone()
            if span is None:
                raise ValueError("document evidence span inventory mismatch")
            rows.append(
                (
                    doc["document_id"],
                    doc["source_id"],
                    doc["split"],
                    doc["representative_id"],
                    doc["drop_reason"],
                    span[0],
                    span[1],
                    doc["content_sha256"],
                    doc["source_family"],
                    doc["modality"],
                    json.dumps(doc["domains"]),
                    offset,
                    length,
                    span[2],
                    span[3],
                )
            )
            for key in (
                "raw:" + doc["raw_content_sha256"],
                "normalized:" + doc["content_sha256"],
                *("origin:" + value for value in _origin_keys(doc)),
            ):
                keys.append((key, doc["document_id"]))
        return rows, keys, end - start
    finally:
        db.close()


def _raw_group(
    database: str,
    documents: str,
    spans: str,
    source: dict[str, Any],
    raw_name: str,
    file: dict[str, Any] | None,
    files_root: str,
    text_field: str | None,
    options: NormalizerOptions,
):
    """Replay one raw snapshot file against every document that cites it."""
    from klab.corpus.jsonl_records import iter_jsonl_records
    from klab.corpus.pipeline import _normalized

    if file is None or file["role"] != "document":
        raise ValueError("document lacks pinned file rights")
    decision = file["rights"]
    raw_path = _safe(Path(files_root), raw_name)
    raw_sha = sha256_file(raw_path)
    normalizer = (
        "cnxml-text-v1" if raw_path.suffix.lower() == ".cnxml" else options.identity
    )
    db = _readonly(database)
    count = 0
    try:
        with (
            raw_path.open("rb") as raw_stream,
            Path(documents).open("rb") as document_stream,
            Path(spans).open("rb") as span_stream,
        ):
            current_line = current_offset = 0
            current_item = None
            for doc_offset, doc_length, span_offset, span_length in db.execute(
                "SELECT doc_offset, doc_length, span_offset, span_length FROM docs "
                "WHERE source=? AND raw_path=? ORDER BY line_start, id",
                (source["id"], raw_name),
            ):
                count += 1
                doc = _read_json(document_stream, doc_offset, doc_length)
                span = _read_json(span_stream, span_offset, span_length)
                if (
                    source.get("explicit_training_restriction", "none_found")
                    != "none_found"
                    or doc["source_revision"] != source["revision"]
                    or doc["source_family"] != source["source_family"]
                    or decision["training_eligibility"]
                    not in ("eligible", "eligible_with_obligations")
                    or doc.get("rights") != decision
                    or doc.get("file_sha256") != file["sha256"]
                    or doc["license"]
                    != (
                        decision["detected_spdx_expression"]
                        or source["rights_policy"]["spdx_expression"]
                        or source["license"]
                    )
                    or doc["redistribution"] != decision["redistribution_mode"]
                ):
                    raise ValueError("document rights attribution mismatch")
                if (
                    span["source_id"] != doc["source_id"]
                    or span["snapshot_sha256"] != source["snapshot_sha256"]
                    or span["raw_sha256"] != raw_sha
                    or span["raw_content_sha256"] != doc["raw_content_sha256"]
                ):
                    raise ValueError("document snapshot span mismatch")
                if (
                    span.get("normalizer") != normalizer
                    or doc["document_id"]
                    != canonical_digest(
                        [
                            source["snapshot_sha256"],
                            doc["source_location"],
                            raw_sha,
                            normalizer,
                        ]
                    )
                    or span.get("section_path") != doc["section_path"]
                ):
                    raise ValueError("document snapshot identity mismatch")
                if "#lines=" in doc["source_location"]:
                    start, end = span["line_start"], span["line_end"]
                    if start < 1 or end < start:
                        raise ValueError("document raw line range mismatch")
                    if start <= current_line:
                        raw_stream.seek(0)
                        current_line, current_offset = 0, 0
                    while current_line < start - 1:
                        line = raw_stream.readline()
                        if not line:
                            raise ValueError("document raw line range mismatch")
                        current_line += 1
                        current_offset += len(line)
                    byte_start = current_offset
                    digest = hashlib.sha256()
                    lines = []
                    while current_line < end:
                        line = raw_stream.readline()
                        if not line:
                            raise ValueError("document raw line range mismatch")
                        current_line += 1
                        current_offset += len(line)
                        digest.update(line)
                        lines.append(line)
                    if (
                        span["byte_start"] != byte_start
                        or span["byte_end"] != current_offset
                        or digest.hexdigest() != doc["raw_content_sha256"]
                    ):
                        raise ValueError("document raw byte range mismatch")
                    if options.structure_cleaning and normalizer == options.identity:
                        _verify_structure_replay(
                            doc, b"".join(lines).decode("utf-8"), options=options
                        )
                    if doc["source_location"] != (
                        f"{span['raw_path']}#lines={start}-{end}"
                    ):
                        raise ValueError("document source location mismatch")
                elif "#row=" in doc["source_location"] and raw_path.suffix == ".jsonl":
                    index = span["line_start"]
                    if (
                        index < 1
                        or span["line_end"] != index
                        or doc["source_location"] != f"{span['raw_path']}#row={index}"
                        or span["byte_start"] is not None
                        or span["byte_end"] is not None
                    ):
                        raise ValueError("document raw row range mismatch")
                    if index <= current_line:
                        raw_stream.seek(0)
                        current_line = 0
                    while current_line < index:
                        parsed = next(
                            iter_jsonl_records(raw_stream, source=str(raw_path)), None
                        )
                        if parsed is None:
                            raise ValueError("document raw row range mismatch")
                        current_line += 1
                        current_item = parsed.value
                    if current_item is None:
                        raise ValueError("document raw row range mismatch")
                    if not isinstance(current_item, dict) or not isinstance(
                        current_item.get(text_field), str
                    ):
                        raise ValueError("document raw row content mismatch")
                    source_text = current_item[text_field]
                    if options.structure_cleaning and normalizer == options.identity:
                        _verify_structure_replay(doc, source_text, options=options)
                    elif (
                        _normalized(source_text) != doc["text"]
                        or doc["raw_content_sha256"] != doc["content_sha256"]
                    ):
                        raise ValueError("document raw row content mismatch")
                else:
                    raise ValueError("unsupported document snapshot span")
    finally:
        db.close()
    return count, raw_path.stat().st_size


def _lineage_rows(database: str, lineage: str, start: int, end: int, origins):
    from klab.corpus.provenance import shape_for_record, validate_lineage

    db = _readonly(database)
    try:
        ids = []
        for row, _, _ in range_records(Path(lineage), start, end):
            record_id = row["record_id"]
            found = db.execute(
                "SELECT representative, dropped, source, split, content_sha256, "
                "source_family, modality, domains FROM docs WHERE id=?",
                (record_id,),
            ).fetchone()
            if found is None or row["record_kind"] != "document":
                raise ValueError("unexpected LM-only lineage record")
            (
                representative,
                dropped,
                source,
                split,
                digest,
                family,
                modality,
                domains,
            ) = found
            domains = json.loads(domains)
            validate_lineage(row)
            if (
                row["parent_document_ids"] != [record_id]
                or row["original_parent_document_ids"] != [record_id]
                or row["representative_parent_document_ids"] != [representative]
                or row["representative_id"] != representative
                or row["drop_reason"] != dropped
                or row["source_family_ids"] != [family]
                or row["split"] != split
                or row["origin"] != origins[source]
                or row["modalities"] != [modality]
                or row["domains"] != domains
                or row["shape"]
                != shape_for_record(
                    "document",
                    "raw_document",
                    domains=domains,
                    parent_document_ids=[record_id],
                )
                or row["verification"]["status"] != "schema_validated"
                or row["verification"]["evidence"].get("schema_id")
                != "normalized_document_v1"
                or row["rendered_sha256"] != digest
            ):
                raise ValueError("document lineage mismatch")
            ids.append((record_id,))
        return ids, end - start
    finally:
        db.close()


def _stage_rows(
    database: str, stage: str, path: str, start: int, end: int, selected: list[str]
):
    db = _readonly(database)
    try:
        ids = []
        for row, _, _ in range_records(Path(path), start, end):
            doc = db.execute(
                "SELECT content_sha256, split, source, dropped FROM docs WHERE id=?",
                (row["record_id"],),
            ).fetchone()
            if (
                doc is None
                or set(row) != {"record_id", "text", "split"}
                or hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != doc[0]
                or row["split"] != doc[1]
                or (selected and doc[2] not in selected)
                or doc[3]
            ):
                raise ValueError("LM stage record differs from selected document")
            ids.append((stage, row["record_id"]))
        return ids, end - start
    finally:
        db.close()


def _view_rows(
    database: str,
    split: str,
    view: str,
    links: str,
    view_start: int,
    link_start: int,
    count: int,
    eligible: bool,
    allowed: list[str],
    origins: dict[str, str],
    wanted_origins: list[str] | None,
):
    db = _readonly(database)
    try:
        ids, size = [], 0
        with Path(view).open("rb") as payloads, Path(links).open("rb") as refs:
            payloads.seek(view_start)
            refs.seek(link_start)
            for _ in range(count):
                payload, link = payloads.readline(), refs.readline()
                size += len(payload)
                row, ref = json.loads(payload), json.loads(link)
                if (
                    set(row) != {"text"}
                    or not isinstance(row["text"], str)
                    or not row["text"].strip()
                    or ref.get("split") != split
                ):
                    raise ValueError("invalid LM record or lineage")
                found = db.execute(
                    "SELECT content_sha256, split, dropped, source FROM docs WHERE id=?",
                    (ref["record_id"],),
                ).fetchone()
                if (
                    found is None
                    or (
                        hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                        split,
                        None,
                    )
                    != found[:3]
                ):
                    raise ValueError("invalid LM record or lineage")
                if (
                    not eligible
                    or found[3] not in allowed
                    or wanted_origins is not None
                    and origins[found[3]] not in wanted_origins
                ):
                    raise ValueError("LM view includes an unselected document")
                ids.append((ref["record_id"],))
        return ids, size
    finally:
        db.close()


def _unique_index(
    db: sqlite3.Connection, name: str, table: str, columns: str, error: str
):
    try:
        db.execute(f"CREATE UNIQUE INDEX {name} ON {table}({columns})")
    except sqlite3.IntegrityError as exception:
        raise ValueError(error) from exception


def _validate_rows_streaming(
    root: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    progress: BuildProgress | None = None,
) -> None:
    """Validate LM-only source evidence in parallel, indexing only identities.

    Document text, spans and lineage stay in the release files; the disposable
    scratch index holds identity columns and byte offsets, so it is a small
    fraction of the release rather than a copy of it.
    """

    source_rows = _load(root / "sources.json")
    sources = {row["id"]: row for row in source_rows}
    if len(sources) != len(source_rows):
        raise ValueError("duplicate source identity")
    receipt = _load(
        root / ("build.json" if (root / "build.json").exists() else "manifest.json")
    )
    identity = receipt.get("identity", receipt.get("build_identity"))
    options = _release_normalizer(identity)
    pinned_snapshots = {row["source_id"]: row["sha256"] for row in receipt["snapshots"]}
    if len(pinned_snapshots) != len(receipt["snapshots"]) or pinned_snapshots != {
        source["id"]: source["snapshot_sha256"]
        for source in source_rows
        if source["snapshot_sha256"]
    }:
        raise ValueError("source snapshot inventory mismatch")
    snapshots = _verify_source_snapshots(
        root, sources, proof_store=proof_store, verification_mode=verification_mode
    )
    rights = _verify_rights_files(root, sources, snapshots)
    options = _release_normalizer(identity)
    origins = {
        source_id: source.get("origin", "primary_source")
        for source_id, source in sources.items()
    }

    def phase(name: str, *paths: Path) -> int:
        total = sum(path.stat().st_size for path in paths)
        if progress is not None:
            progress.start_phase(name, expected_input_bytes=total)
        return total

    def advance(documents: int = 0, size: int = 0) -> None:
        if progress is not None:
            progress.update(documents=documents, input_bytes=size)

    spans_path, documents_path = root / "spans.jsonl", root / "documents.jsonl"
    with tempfile.TemporaryDirectory(
        prefix=".verify-corpus-", dir=scratch_root()
    ) as temporary:
        # Workers read one file while this process fills another, so writers and
        # readers never share a database file.
        span_database = str(Path(temporary) / "spans.sqlite")
        database = str(Path(temporary) / "evidence.sqlite")
        tally = str(Path(temporary) / "tally.sqlite")
        db = sqlite3.connect(database)
        db.execute("ATTACH ? AS span_store", (span_database,))
        db.execute("ATTACH ? AS tally", (tally,))
        for schema in ("main", "span_store", "tally"):
            db.execute(f"PRAGMA {schema}.journal_mode=OFF")
            db.execute(f"PRAGMA {schema}.synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-65536")
        try:
            db.execute(
                "CREATE TABLE span_store.spans (id TEXT, raw_path TEXT, line_start INTEGER, "
                "byte_offset INTEGER, byte_length INTEGER)"
            )
            db.execute(
                "CREATE TABLE docs (id TEXT, source TEXT, split TEXT, "
                "representative TEXT, dropped TEXT, raw_path TEXT, "
                "line_start INTEGER, content_sha256 TEXT, source_family TEXT, "
                "modality TEXT, domains TEXT, doc_offset INTEGER, "
                "doc_length INTEGER, span_offset INTEGER, span_length INTEGER)"
            )
            db.execute("CREATE TABLE duplicate_keys (key TEXT, id TEXT)")
            db.execute("CREATE TABLE tally.lineage (id TEXT)")
            db.execute("CREATE TABLE tally.stage_docs (stage TEXT, id TEXT)")
            db.execute("CREATE TABLE tally.view_docs (id TEXT)")

            def load(table: str, rows: list, width: int) -> None:
                db.executemany(
                    f"INSERT INTO {table} VALUES ({','.join('?' * width)})", rows
                )

            total = phase("spans", spans_path)
            run_tasks(
                _span_rows,
                (
                    (str(spans_path), start, end)
                    for start, end in chunk_ranges(spans_path)
                ),
                lambda result: (
                    load("spans", result[0], 5),
                    advance(len(result[0]), result[1]),
                ),
                total_bytes=total,
            )
            _unique_index(
                db,
                "span_store.spans_id",
                "spans",
                "id",
                "duplicate document evidence span",
            )
            db.commit()

            total = phase("documents", documents_path)
            run_tasks(
                _document_rows,
                (
                    (span_database, str(documents_path), start, end, options)
                    for start, end in chunk_ranges(documents_path)
                ),
                lambda result: (
                    load("docs", result[0], 15),
                    load("duplicate_keys", result[1], 2),
                    advance(len(result[0]), result[2]),
                ),
                total_bytes=total,
            )
            db.commit()
            _unique_index(db, "docs_id", "docs", "id", "duplicate document identity")
            if (
                db.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
                != db.execute("SELECT COUNT(*) FROM spans").fetchone()[0]
            ):
                raise ValueError("document evidence span inventory mismatch")
            db.execute(
                "CREATE INDEX docs_raw_order_idx "
                "ON docs(source, raw_path, line_start, id)"
            )
            if db.execute(
                "SELECT COUNT(*) FROM docs AS doc LEFT JOIN docs AS representative "
                "ON doc.representative=representative.id WHERE representative.id IS NULL "
                "OR representative.representative!=representative.id OR "
                "(doc.id=doc.representative AND doc.dropped IS NOT NULL "
                "AND doc.dropped!='metadata_only_front_matter') OR "
                "(doc.id!=doc.representative AND doc.dropped IS NULL) OR "
                "(doc.dropped NOT IN ('duplicate','contaminated_heldout',"
                "'metadata_only_front_matter')) OR "
                "(doc.split!='train' AND representative.split!=doc.split) OR "
                "(doc.dropped='contaminated_heldout' AND "
                "(doc.split!='train' OR representative.split='train')) OR "
                "(doc.dropped='duplicate' AND doc.split='train' "
                "AND representative.split!='train')"
            ).fetchone()[0]:
                raise ValueError("invalid duplicate representative identity")
            db.execute("CREATE INDEX duplicate_key_idx ON duplicate_keys (key)")
            if db.execute(
                "SELECT COUNT(*) FROM (SELECT duplicate_keys.key FROM duplicate_keys "
                "JOIN docs ON docs.id=duplicate_keys.id GROUP BY duplicate_keys.key "
                "HAVING COUNT(DISTINCT docs.representative)!=1)"
            ).fetchone()[0]:
                raise ValueError("duplicate group representative mismatch")
            db.commit()

            groups = db.execute(
                "SELECT source, raw_path, COUNT(*) FROM docs GROUP BY source, raw_path "
                "ORDER BY COUNT(*) DESC"
            ).fetchall()
            files_roots = {
                source_id: root.parent.parent
                / "snapshots"
                / source_id
                / sources[source_id]["snapshot_sha256"]
                / "files"
                for source_id in {group[0] for group in groups}
            }
            for source_id, raw_name, _ in groups:
                # Fail on an unsafe path before spending any time on the replay.
                _safe(files_roots[source_id], raw_name)
            total = sum(
                _safe(files_roots[source_id], raw_name).stat().st_size
                for source_id, raw_name, _ in groups
            )
            if progress is not None:
                progress.start_phase("raw_replay", expected_input_bytes=total)
            run_tasks(
                _raw_group,
                (
                    (
                        database,
                        str(documents_path),
                        str(spans_path),
                        sources[source_id],
                        raw_name,
                        rights.get((source_id, raw_name)),
                        str(files_roots[source_id]),
                        snapshots[source_id]["declaration"]["acquisition"].get(
                            "text_field"
                        ),
                        options,
                    )
                    for source_id, raw_name, _ in groups
                ),
                lambda result: advance(*result),
                total_bytes=total,
            )

            lineage_path = root / "lineage.jsonl"
            total = phase("lineage", lineage_path)
            run_tasks(
                _lineage_rows,
                (
                    (database, str(lineage_path), start, end, origins)
                    for start, end in chunk_ranges(lineage_path)
                ),
                lambda result: (
                    load("lineage", result[0], 1),
                    advance(len(result[0]), result[1]),
                ),
                total_bytes=total,
            )
            _unique_index(
                db, "tally.lineage_id", "lineage", "id", "duplicate lineage ID"
            )
            if db.execute(
                "SELECT COUNT(*) FROM docs LEFT JOIN lineage ON docs.id=lineage.id "
                "WHERE lineage.id IS NULL"
            ).fetchone()[0]:
                raise ValueError("document lineage mismatch")
            db.commit()

            transforms = identity["transforms"]
            for stage in receipt["stages"]:
                transform = next(
                    spec for spec in transforms if spec["id"] == stage["id"]
                )
                selected = sorted(set(transform["inputs"]) & set(sources))
                stage_path = root / "stages" / f"{stage['id']}.jsonl"
                total = phase(f"stage:{stage['id']}", stage_path)
                run_tasks(
                    _stage_rows,
                    (
                        (database, stage["id"], str(stage_path), start, end, selected)
                        for start, end in chunk_ranges(stage_path)
                    ),
                    lambda result: (
                        load("stage_docs", result[0], 2),
                        advance(len(result[0]), result[1]),
                    ),
                    total_bytes=total,
                )
            db.commit()
            _unique_index(
                db,
                "tally.stage_docs_id",
                "stage_docs",
                "stage, id",
                "duplicate LM stage identity",
            )
            for stage in receipt["stages"]:
                transform = next(
                    spec for spec in transforms if spec["id"] == stage["id"]
                )
                selected = set(transform["inputs"]) & set(sources)
                if db.execute(
                    "SELECT COUNT(*) FROM docs WHERE dropped IS NULL AND "
                    "(? = 1 OR source IN (SELECT value FROM json_each(?))) "
                    "AND id NOT IN (SELECT id FROM stage_docs WHERE stage=?)",
                    (not bool(selected), json.dumps(sorted(selected)), stage["id"]),
                ).fetchone()[0]:
                    raise ValueError("LM stage omitted selected documents")

            release = identity["release"]
            allowed_sources = set().union(
                *(
                    set(spec["inputs"]) & set(sources)
                    if set(spec["inputs"]) & set(sources)
                    else set(sources)
                    for spec in transforms
                )
            )
            shapes = release.get("include_shapes")
            wanted_origins = release.get("include_origins")
            eligible = (shapes is None or "raw_document" in shapes) and (
                wanted_origins is None
                or any(
                    source.get("origin", "primary_source") in wanted_origins
                    for source in sources.values()
                )
            )
            for split in ("train", "validation", "test"):
                view, links = (
                    root / "lm" / f"{split}.jsonl",
                    root / "lm" / f"{split}.lineage.jsonl",
                )
                total = phase(f"view:{split}", view, links)
                view_starts, view_lines = line_boundaries(view, VIEW_LINES)
                link_starts, link_lines = line_boundaries(links, VIEW_LINES)
                if view_lines != link_lines:
                    raise ValueError("LM lineage count mismatch")
                run_tasks(
                    _view_rows,
                    (
                        (
                            database,
                            split,
                            str(view),
                            str(links),
                            view_start,
                            link_start,
                            min(VIEW_LINES, view_lines - index * VIEW_LINES),
                            eligible,
                            sorted(allowed_sources),
                            origins,
                            wanted_origins,
                        )
                        for index, (view_start, link_start) in enumerate(
                            zip(view_starts, link_starts, strict=True)
                        )
                    ),
                    lambda result: (
                        load("view_docs", result[0], 1),
                        advance(len(result[0]), result[1]),
                    ),
                    total_bytes=total,
                )
                for name in (f"chat/{split}.jsonl", f"chat/{split}.lineage.jsonl"):
                    if (root / name).stat().st_size:
                        raise ValueError("unexpected chat records in LM-only release")
            db.commit()
            _unique_index(
                db,
                "tally.view_docs_id",
                "view_docs",
                "id",
                "duplicate LM view identity",
            )
            if eligible:
                for source_id in allowed_sources:
                    if (
                        wanted_origins is not None
                        and origins[source_id] not in wanted_origins
                    ):
                        continue
                    if db.execute(
                        "SELECT COUNT(*) FROM docs WHERE source=? AND dropped IS NULL "
                        "AND id NOT IN (SELECT id FROM view_docs)",
                        (source_id,),
                    ).fetchone()[0]:
                        raise ValueError("LM view omitted selected documents")
            for name in (
                "chat/records.jsonl",
                "scenarios.jsonl",
                "generations.jsonl",
                "lexical/candidates.jsonl",
                "semantic/candidates.jsonl",
                "tool_episodes.jsonl",
            ):
                if (root / name).stat().st_size:
                    raise ValueError("unexpected generated records in LM-only release")
        finally:
            db.close()


def _validate_rows(
    root: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    progress: BuildProgress | None = None,
) -> None:
    if progress is not None:
        progress.start_phase("reviewed_rows")
    from klab.corpus.admission_coverage import AdmissionCoverage

    with (
        metadata_operation(scratch_root(), max_database_bytes=None) as metadata,
        AdmissionCoverage(metadata.connection) as coverage,
        ExitStack() as replay_stack,
    ):
        _validate_rows_stored(
            root,
            replay_stack=replay_stack,
            coverage=coverage,
            row_connection=metadata.connection,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )


def _validate_rows_stored(
    root: Path,
    *,
    replay_stack: ExitStack,
    coverage,
    row_connection,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> None:
    receipt = _load(
        root / ("build.json" if (root / "build.json").exists() else "manifest.json")
    )
    identity = receipt.get("identity", receipt.get("build_identity"))
    options = _release_normalizer(identity)
    selected_normalizer = options.identity
    from klab.corpus.jsonl_records import records_from_path
    from klab.corpus.row_index import RowIndex

    def rows(path):
        if path.is_symlink():
            raise ValueError("symlinked evidence input")
        return (record.value for record in records_from_path(path))

    def row_index(path, key="record_id", *, duplicate="duplicate evidence identity"):
        return RowIndex(
            row_connection, rows(path), lambda row: row[key], duplicate=duplicate
        )

    document_map = row_index(
        root / "documents.jsonl", "document_id", duplicate="duplicate document identity"
    )
    for doc in document_map.values():
        if doc["content_sha256"] != hashlib.sha256(doc["text"].encode()).hexdigest():
            raise ValueError("document content digest mismatch")
        _verify_cleaning_decision(doc, options)
    spans = row_index(root / "spans.jsonl")
    sources = {item["id"]: item for item in _load(root / "sources.json")}

    source_origins = {
        key: value.get("origin", "primary_source") for key, value in sources.items()
    }

    snapshots = _verify_source_snapshots(
        root, sources, proof_store=proof_store, verification_mode=verification_mode
    )
    rights_files = _verify_rights_files(root, sources, snapshots)
    if not spans.same_keys(document_map):
        raise ValueError("document evidence span inventory mismatch")
    previous_file: Path | None = None
    raw = b""
    raw_sha = ""
    offsets: list[int] = []
    sample_rows = None
    for doc in document_map.source_order(spans):
        span = spans[doc["document_id"]]
        source = sources[doc["source_id"]]
        if (
            doc["source_revision"] != source["revision"]
            or doc["source_family"] != source["source_family"]
        ):
            raise ValueError("document attribution mismatch")
        file = rights_files.get((doc["source_id"], span["raw_path"]))
        if file is None or file["role"] != "document":
            raise ValueError("document lacks pinned file rights")
        decision = file["rights"]
        if source.get("explicit_training_restriction", "none_found") != "none_found":
            raise ValueError("document source has unresolved training restriction")
        if (
            decision["training_eligibility"]
            not in ("eligible", "eligible_with_obligations")
            or doc.get("rights") != decision
            or doc.get("file_sha256") != file["sha256"]
            or doc["license"]
            != (
                file.get("admission_license_label")
                or decision["detected_spdx_expression"]
                or source["rights_policy"]["spdx_expression"]
                or source["license"]
            )
            or doc["redistribution"] != decision["redistribution_mode"]
        ):
            raise ValueError("document rights attribution mismatch")
        if file.get("_admission_file_decision", "qualify") != "qualify":
            raise ValueError("document lacks qualifying file admission")
        if "_admission_decisions" in file:
            metadata = doc.get("metadata") or {}
            index = metadata.get("source_row_index")
            choice = file["_admission_decisions"].get(index)
            if (
                type(index) is not int
                or choice is None
                or choice["decision"] != "qualify"
                or metadata.get("source_row_sha256") != choice["source_row_sha256"]
            ):
                raise ValueError("document lacks qualifying record admission")
            coverage.observe(doc["source_id"], span["raw_path"], index, unique=True)
        snapshot = (
            root.parent.parent
            / "snapshots"
            / doc["source_id"]
            / source["snapshot_sha256"]
        )
        raw_path = _safe(snapshot / "files", span["raw_path"])
        if raw_path != previous_file:
            # Finishing the previous cursor validates trailing unselected rows.
            replay_stack.close()
            raw_sha = sha256_file(raw_path)
            offsets = [0]
            sample_rows = None
            needs_rows = ("_admission_decisions" in file) or (
                options.structure_cleaning and "#row=" in doc["source_location"]
            )
            if needs_rows and raw_path.suffix == ".jsonl":
                from klab.corpus.jsonl_records import forward_rows

                raw = b""
                sample_rows = replay_stack.enter_context(
                    forward_rows(
                        raw_path,
                        max_record_bytes=(
                            8 * 1024**2 if "_admission_decisions" in file else None
                        ),
                        expected_sha256=raw_sha,
                    )
                )
            else:
                raw = raw_path.read_bytes()
                if "#lines=" in doc["source_location"]:
                    for line in raw.split(b"\n")[:-1]:
                        offsets.append(offsets[-1] + len(line) + 1)
                    if raw and not raw.endswith(b"\n"):
                        offsets.append(len(raw))
                if needs_rows:
                    from klab.corpus.jsonl_records import records_from_bytes

                    sample_rows = list(records_from_bytes(raw, source=str(raw_path)))
            previous_file = raw_path
        selected_row = None
        if sample_rows is not None:
            from klab.corpus.jsonl_records import ForwardRows

            ordinal = span["line_start"]
            if isinstance(sample_rows, ForwardRows):
                selected_row = sample_rows.get(ordinal).value
            elif type(ordinal) is int and 1 <= ordinal <= len(sample_rows):
                selected_row = sample_rows[ordinal - 1].value
            else:
                raise ValueError("source replay row exceeds source inventory")
        text_field = (
            snapshots[doc["source_id"]]["declaration"]
            .get("acquisition", {})
            .get("text_field", "text")
        )
        if (
            span["source_id"] != doc["source_id"]
            or span["snapshot_sha256"] != source["snapshot_sha256"]
            or raw_sha != span["raw_sha256"]
            or span["raw_content_sha256"] != doc["raw_content_sha256"]
        ):
            raise ValueError("document snapshot span mismatch")
        expected_normalizer = (
            "cnxml-text-v1"
            if raw_path.suffix.lower() == ".cnxml"
            else selected_normalizer
        )
        if options.structure_cleaning and (
            span.get("normalizer") != expected_normalizer
            or doc["document_id"]
            != canonical_digest(
                [
                    source["snapshot_sha256"],
                    doc["source_location"],
                    raw_sha,
                    expected_normalizer,
                ]
            )
        ):
            raise ValueError("structured document identity mismatch")
        if "_admission_decisions" in file:
            from klab.corpus.pipeline import _normalized

            line_number = span["line_start"]
            if (
                selected_row is None
                or type(line_number) is not int
                or line_number < 1
                or span["line_end"] != line_number
                or doc["source_location"] != f"{span['raw_path']}#row={line_number}"
                or (
                    not options.structure_cleaning
                    and _normalized(selected_row[text_field]) != doc["text"]
                )
            ):
                raise ValueError("admitted document differs from selected source row")
        if options.structure_cleaning and "#row=" in doc["source_location"]:
            if selected_row is None:
                raise ValueError("structured source row unavailable")
            _verify_structure_replay(doc, selected_row[text_field], options=options)
        if "#lines=" in doc["source_location"]:
            start, end = span["line_start"], span["line_end"]
            if start < 1 or end < start or end >= len(offsets):
                raise ValueError("document raw line range mismatch")
            byte_start, byte_end = span["byte_start"], span["byte_end"]
            if (
                byte_start != offsets[start - 1]
                or byte_end != offsets[end]
                or hashlib.sha256(raw[byte_start:byte_end]).hexdigest()
                != doc["raw_content_sha256"]
            ):
                raise ValueError("document raw byte range mismatch")
            if options.structure_cleaning and expected_normalizer != "cnxml-text-v1":
                _verify_structure_replay(
                    doc, raw[byte_start:byte_end].decode("utf-8"), options=options
                )
    replay_stack.close()
    for (source_id, path), file in rights_files.items():
        for index, choice in file.get("_admission_decisions", {}).items():
            if choice["decision"] == "qualify":
                coverage.expect(source_id, path, index)
    coverage.verify()
    lineage_map = row_index(root / "lineage.jsonl", duplicate="duplicate lineage ID")
    classified = any("origin" in row for row in lineage_map.values())
    if classified:
        from klab.corpus.pipeline import _path_scenario, _scenario, _scenario_shape
        from klab.corpus.provenance import (
            rendered_digest,
            shape_for_record,
            validate_lineage,
        )
        from klab.data.conversations import _v2_document

        receipt = _load(
            root / ("build.json" if (root / "build.json").exists() else "manifest.json")
        )
        stage_rows = RowIndex(
            row_connection,
            (
                {"stage": stage["id"], "record": record}
                for stage in receipt["stages"]
                for record in rows(root / "stages" / f"{stage['id']}.jsonl")
                if record.get("record_id") or record.get("scenario_id")
            ),
            lambda item: (
                item["stage"],
                item["record"].get("record_id", item["record"].get("scenario_id")),
            ),
        )
        chats = row_index(root / "chat/records.jsonl")
        scenarios = row_index(root / "scenarios.jsonl", "scenario_id")
        generations = row_index(root / "generations.jsonl")
        semantic = row_index(root / "semantic/candidates.jsonl")
        for row in lineage_map.values():
            validate_lineage(row, documents=document_map)
            expected_origin = (
                source_origins[document_map[row["record_id"]]["source_id"]]
                if row["record_kind"] == "document"
                else "deterministic_synthetic"
                if row["record_kind"] in {"scenario", "tool_episode"}
                or row["record_kind"] == "chat_sft"
                and row.get("scenario_id")
                else "multi_source_synthetic"
                if len(
                    {
                        document_map[parent]["source_id"]
                        for parent in row["parent_document_ids"]
                    }
                )
                > 1
                else "free_generated_synthetic"
                if row["record_kind"] == "generation"
                and not row["parent_document_ids"]
                or row["record_kind"] == "chat_sft"
                and row.get("generation_id")
                and not row["parent_document_ids"]
                else "source_transformed_synthetic"
            )
            if row["origin"] != expected_origin:
                raise ValueError("lineage origin disagrees with source and transform")
            expected_domains = (
                sorted(document_map[row["record_id"]]["domains"])
                if row["record_kind"] == "document"
                else ["systems_scenarios"]
                if row.get("generator_world_id")
                else sorted(
                    {
                        domain
                        for parent in row["parent_document_ids"]
                        for domain in document_map[parent]["domains"]
                    }
                )
            )
            if (
                sorted(row.get("domains", [])) != expected_domains
                or row["shape"]["attributes"]["source_domains"] != expected_domains
            ):
                raise ValueError("shape source domain attribution mismatch")
            evidence = row["verification"]["evidence"]
            if row["verification"]["status"] == "oracle_verified":
                scenario = scenarios.get(row.get("scenario_id", row["record_id"]))
                if scenario is None:
                    raise ValueError("oracle scenario reference mismatch")
                world = scenario["world_state"]
                generator_id = scenario["generator_id"]
                if generator_id == "pathlib_path_suffix_v1":
                    expected_scenario = _path_scenario(
                        scenario["world_seed"],
                        scenario["scenario_family_id"],
                        scenario["transform_id"],
                    )
                    expected_result = PurePosixPath(world["path"]).suffix
                else:
                    expected_scenario = _scenario(
                        generator_id,
                        scenario["world_seed"],
                        scenario["scenario_family_id"],
                        scenario["template_family_id"],
                        scenario["transform_id"],
                    )
                    expected_result = expected_scenario["oracle_answer"]
                implementation = receipt.get("identity", receipt.get("build_identity"))[
                    "implementation_sha256"
                ]
                if (
                    evidence["oracle_answer"] != scenario["oracle_answer"]
                    or evidence["generator_world_id"] != scenario["generator_world_id"]
                    or evidence["oracle_identity"]
                    != canonical_digest([world, scenario["oracle_answer"]])
                    or evidence["world_state"] != world
                    or scenario["generator_id"].rsplit("_v", 1)[-1]
                    != evidence["oracle_version"]
                    or evidence["oracle_implementation"] != implementation
                    or evidence["interpreter"] != scenario["interpreter"]
                    or evidence["actual_result"] != expected_result
                    or evidence["comparison_status"] != "match"
                    or expected_result != scenario["oracle_answer"]
                    or expected_scenario
                    != {key: value for key, value in scenario.items() if key != "split"}
                    or (
                        evidence.get("receipt") != scenario.get("oracle_receipt")
                        if generator_id != "pathlib_path_suffix_v1"
                        else "receipt" in evidence
                    )
                ):
                    raise ValueError("oracle verification evidence mismatch")
            kind = row["record_kind"]
            shape_id = row["shape"]["id"]
            attributes = row["shape"]["attributes"]
            if kind == "document" and (
                shape_id != "raw_document"
                or row["shape"]
                != shape_for_record(
                    kind,
                    shape_id,
                    domains=row["domains"],
                    parent_document_ids=row["parent_document_ids"],
                )
            ):
                raise ValueError("document shape mismatch")
            if kind == "document":
                if (
                    row["verification"]["status"] != "schema_validated"
                    or evidence.get("schema_id") != "normalized_document_v1"
                ):
                    raise ValueError("document verification classification mismatch")
                continue
            if kind in {"chat_sft", "tool_episode"}:
                record = chats.get(row["record_id"])
                payload = (
                    {
                        key: record[key]
                        for key in ("format_version", "loss_mode", "messages")
                    }
                    if record is not None
                    else None
                )
            else:
                stored = stage_rows.get((row["transform_id"], row["record_id"]))
                payload = stored["record"] if stored is not None else None
            source_payload = record if kind in {"chat_sft", "tool_episode"} else payload
            expected_shape = {
                "lexical_candidate": "lexical_inventory",
                "semantic_candidate": "definition",
                "scenario": _scenario_shape(source_payload, "scenario")
                if source_payload is not None
                else None,
                "generation": (source_payload.get("parsed_output") or {}).get(
                    "shape", "direct_qa"
                )
                if source_payload is not None
                else None,
                "tool_episode": "tool_trace",
                "chat_sft": (
                    source_payload.get("semantic_shape")
                    or (
                        "troubleshooting_scenario"
                        if source_payload.get("scenario_id")
                        else "direct_qa"
                    )
                )
                if source_payload is not None
                else None,
            }.get(kind)
            if (
                shape_id != expected_shape
                or attributes["task_family"] != expected_shape
            ):
                raise ValueError("lineage shape classification mismatch")
            if row["shape"] != shape_for_record(
                kind,
                shape_id,
                domains=row["domains"],
                parent_document_ids=row["parent_document_ids"],
                scenario_id=row.get("scenario_id"),
            ):
                raise ValueError("lineage shape attributes mismatch")
            if source_payload is not None:
                evidence_payload = (
                    generations.get(source_payload["generation_id"])
                    if source_payload.get("generation_id")
                    else semantic.get(source_payload["semantic_id"])
                    if source_payload.get("semantic_id")
                    else source_payload
                )
                if (
                    kind == "generation"
                    or kind == "chat_sft"
                    and source_payload.get("generation_id")
                ) and row.get("generator") != evidence_payload.get("generator"):
                    raise ValueError("lineage generator metadata mismatch")
                if (
                    kind != "generation"
                    and not (kind == "chat_sft" and source_payload.get("generation_id"))
                    and "generator" in row
                ):
                    raise ValueError("unexpected lineage generator metadata")
                expected_status = (
                    "oracle_verified"
                    if kind == "scenario" or source_payload.get("scenario_id")
                    else evidence_payload.get("validation_status", "schema_validated")
                )
                if (
                    row["verification"]["status"] != expected_status
                    or row.get("validation_status") != expected_status
                ):
                    raise ValueError(
                        "lineage verification differs from recorded status"
                    )
                if (
                    expected_status == "schema_validated"
                    and evidence.get("schema_id") != f"{kind}_v1"
                ):
                    raise ValueError("lineage schema evidence mismatch")
            if row["verification"]["status"] == "source_entailed":
                expected_answer = (
                    source_payload["messages"][-1]["content"]
                    if kind == "chat_sft"
                    else source_payload["parsed_output"]["answer"]
                    if kind == "generation"
                    else source_payload["value"]
                )
                if evidence.get("answer") != expected_answer:
                    raise ValueError(
                        "source-entailed answer differs from rendered record"
                    )
            if payload is None:
                raise ValueError("lineage payload missing")
            if kind in {"chat_sft", "tool_episode"}:
                actual_rendered = rendered_digest(
                    _v2_document(payload, Path("<corpus-record>"), 1).text, text=True
                )
            else:
                actual_rendered = rendered_digest(payload)
            if actual_rendered != row["rendered_sha256"]:
                raise ValueError("lineage rendered digest mismatch")
            if kind == "generation":
                generation = generations.get(row["record_id"])
                if row["verification"]["status"] == "source_entailed" and (
                    generation["parsed_output"] is None
                    or generation["parsed_output"]["answer"] not in evidence["passage"]
                    or generation["parsed_output"]["citation_id"]
                    != evidence["document_id"]
                ):
                    raise ValueError("generation answer not entailed by citation")
                if generation != payload or canonical_digest(
                    generation["generator"]
                ) != row.get("generator_identity"):
                    raise ValueError("generation identity mismatch")
            elif kind == "scenario":
                scenario = scenarios.get(row["record_id"])
                if scenario != payload or canonical_digest(
                    [scenario["generator_id"]]
                ) != row.get("generator_identity"):
                    raise ValueError("scenario generator identity mismatch")
            elif kind in {"chat_sft", "tool_episode"}:
                if generation_id := row.get("generation_id"):
                    source = generations.get(generation_id)
                    if source is None or source["generator_identity"] != row.get(
                        "generator_identity"
                    ):
                        raise ValueError("chat generation identity mismatch")
                    if source.get("parsed_output") is None or row["verification"][
                        "status"
                    ] != source["validation_status"].replace(
                        "source_grounded", "source_entailed"
                    ):
                        raise ValueError("chat verification differs from generation")
                elif scenario_id := row.get("scenario_id"):
                    source = scenarios.get(scenario_id)
                    if source is None or canonical_digest(
                        [source["generator_id"]]
                    ) != row.get("generator_identity"):
                        raise ValueError("chat scenario identity mismatch")
                elif semantic_id := row.get("semantic_id"):
                    fact = semantic.get(semantic_id)
                    if (
                        fact is None
                        or record.get("semantic_id") != semantic_id
                        or row.get("generator_identity")
                        != canonical_digest(
                            [
                                "semantic_chat_v1",
                                row["transform_id"],
                                record["semantic_shape"],
                            ]
                        )
                        or evidence["document_id"] != fact["evidence_document_id"]
                        or evidence["passage"] != fact["evidence_passage"]
                        or evidence["span"] != fact["evidence_span"]
                        or record["messages"][-1]["content"]
                        not in fact["evidence_passage"]
                    ):
                        raise ValueError("semantic chat generator/evidence mismatch")
                else:
                    raise ValueError("chat missing generator reference")
    for row in lineage_map.values():
        parents = row["parent_document_ids"]
        if any(parent not in document_map for parent in parents) or (
            row["split"] not in {"train", "validation", "test"}
            and not (
                row["record_kind"] == "lexical_candidate"
                and row["split"] == "inventory"
            )
        ):
            raise ValueError("dangling lineage or invalid split")
        if any(
            parent not in document_map
            for parent in row["representative_parent_document_ids"]
        ):
            raise ValueError("invalid representative lineage")
        if row["split"] == "inventory":
            if len({document_map[parent]["split"] for parent in parents}) < 2:
                raise ValueError("unnecessary cross-split lexical inventory")
        elif any(document_map[parent]["split"] != row["split"] for parent in parents):
            raise ValueError("cross-split lineage")
        if row["representative_parent_document_ids"] != sorted(
            {document_map[parent]["representative_id"] for parent in parents}
        ):
            raise ValueError("incorrect duplicate representative lineage")
    for doc in document_map.values():
        row = lineage_map.get(doc["document_id"])
        if (
            row is None
            or row["record_kind"] != "document"
            or row["split"] != doc["split"]
        ):
            raise ValueError("document lineage mismatch")
    if not classified:
        chats = row_index(root / "chat/records.jsonl")
    for split in ("train", "validation", "test"):
        from itertools import zip_longest

        lm = rows(root / "lm" / f"{split}.jsonl")
        lm_lineage = rows(root / "lm" / f"{split}.lineage.jsonl")
        for row, link in zip_longest(lm, lm_lineage):
            if row is None or link is None:
                raise ValueError("LM lineage count mismatch")
            doc = document_map[link["record_id"]]
            if classified and (
                link["split"] != split
                or lineage_map[link["record_id"]]["record_kind"] != "document"
                or lineage_map[link["record_id"]]["rendered_sha256"]
                != hashlib.sha256(row["text"].encode("utf-8")).hexdigest()
            ):
                raise ValueError("LM selected-view lineage mismatch")
            if (
                set(row) != {"text"}
                or not isinstance(row["text"], str)
                or not row["text"].strip()
                or doc["text"] != row["text"]
                or doc["split"] != split
                or doc["drop_reason"]
            ):
                raise ValueError("invalid LM record or lineage")
        from itertools import zip_longest

        chat = rows(root / "chat" / f"{split}.jsonl")
        chat_lineage = rows(root / "chat" / f"{split}.lineage.jsonl")
        for row, link in zip_longest(chat, chat_lineage):
            if row is None or link is None:
                raise ValueError("chat lineage count mismatch")
            record = chats[link["record_id"]]
            if classified and (
                link["split"] != split
                or lineage_map[link["record_id"]]["record_kind"]
                not in {"chat_sft", "tool_episode"}
                or lineage_map[link["record_id"]]["rendered_sha256"]
                != hashlib.sha256(
                    _v2_document(row, Path("<corpus-record>"), 1).text.encode("utf-8")
                ).hexdigest()
            ):
                raise ValueError("chat selected-view lineage mismatch")
            if (
                record["split"] != split
                or lineage_map[link["record_id"]]["split"] != split
                or row
                != {
                    key: record[key]
                    for key in ("format_version", "loss_mode", "messages")
                }
            ):
                raise ValueError("invalid chat record lineage")
        from klab.data.conversations import iter_rendered_conversations

        for _ in iter_rendered_conversations(root / "chat" / f"{split}.jsonl"):
            pass


def _canonical_lines(path: str, start: int, end: int) -> tuple[int, int]:
    """Count the rows in a range, refusing any line that is not canonical JSON."""
    count = 0
    with Path(path).open("rb") as stream:
        stream.seek(start)
        offset = start
        for line in stream:
            if offset >= end:
                break
            offset += len(line)
            if not line.strip():
                continue
            if canonical_json(json.loads(line)) != line.rstrip(b"\n"):
                raise ValueError("stage receipt does not match output")
            count += 1
    return count, end - start


def _verify_stages(
    root: Path,
    stages: list[dict[str, Any]],
    identity: dict[str, Any],
    progress: BuildProgress | None = None,
) -> None:
    declarations = {spec["id"]: spec for spec in identity["transforms"]}
    if len(stages) != len(declarations) or {stage["id"] for stage in stages} != set(
        declarations
    ):
        raise ValueError("stage receipt inventory mismatch")
    for stage in stages:
        declaration = declarations[stage["id"]]
        if (
            stage["kind"] != declaration["kind"]
            or stage["version"] != declaration["version"]
            or stage["parameters_sha256"] != canonical_digest(declaration["parameters"])
            or stage["inputs"] != declaration["inputs"]
            or stage["implementation_sha256"] != identity["implementation_sha256"]
        ):
            raise ValueError("stage declaration identity mismatch")
        output = _safe(root, f"stages/{stage['id']}.jsonl")
        # Workers confirm every line is the canonical encoding of its row; this
        # process then streams the file once, deriving the file digest and the
        # stage identity (a hash over the canonical rows) from the same bytes.
        rows = []
        total = output.stat().st_size

        def progress_bytes(size: int) -> None:
            if progress is not None:
                progress.update(input_bytes=size)

        def consume(result: tuple[int, int], rows: list[int] = rows) -> None:
            rows.append(result[0])
            progress_bytes(result[1])

        run_tasks(
            _canonical_lines,
            ((str(output), start, end) for start, end in chunk_ranges(output)),
            consume,
            total_bytes=total,
        )
        file_digest, identity_digest = hashlib.sha256(), hashlib.sha256()
        identity_digest.update(canonical_json([stage["id"]])[:-1] + b",[")
        count = 0
        with output.open("rb") as stream:
            for line in stream:
                file_digest.update(line)
                if not line.strip():
                    continue
                if count:
                    identity_digest.update(b",")
                identity_digest.update(line.rstrip(b"\n"))
                count += 1
        identity_digest.update(b"]]")
        if (
            file_digest.hexdigest() != stage["output_sha256"]
            or count != stage["output_count"]
            or count != sum(rows)
            or identity_digest.hexdigest() != stage["output_id"]
        ):
            raise ValueError("stage receipt does not match output")


def verify_build(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    _domain_cold: bool = False,
) -> dict[str, Any]:
    """Verify staged build and its pinned snapshots."""
    path = Path(path).resolve()
    with _verification_operation():
        if (
            not _domain_cold
            and proof_store is not None
            and verification_mode == "verified_reuse"
        ):
            from klab.experiments.artifacts import Artifact, verify_artifact

            manifest = _load(path / "build.json")
            verify_artifact(
                Artifact(
                    kind="corpus_build",
                    version=manifest["schema_version"],
                    producer="klab",
                    identifier=manifest["build_id"],
                    sha256=manifest["build_id"],
                    path=str(path),
                ),
                path / "build.json",
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            return manifest
        return _verify_build_cold(
            path,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )


def _verify_build_cold(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Verify all staged outputs and their claimed immutable input snapshots."""
    path = Path(path).resolve()
    manifest = _load(path / "build.json")
    from klab.corpus.completed_preparation import (
        local_scope,
        verify_completed_build,
    )

    if local_scope(manifest) is not None:
        return verify_completed_build(path, manifest)
    if (
        manifest["schema_version"] != 1
        or manifest["build_id"] != path.name
        or canonical_digest(manifest["identity"]) != path.name
    ):
        raise ValueError("invalid build identity")
    progress = open_progress(manifest["build_id"])
    _files(
        path,
        {
            **manifest["files"],
            "build.json": {
                "sha256": sha256_file(path / "build.json"),
                "size": (path / "build.json").stat().st_size,
            },
        },
        progress,
    )
    progress.start_phase("stage_receipts")
    _verify_stages(path, manifest["stages"], manifest["identity"], progress)
    from klab.corpus.split_assignments import verify_document_assignments

    progress.start_phase("snapshots")
    for snapshot in manifest["snapshots"]:
        receipt = _verified_snapshot(
            path.parent.parent
            / "snapshots"
            / snapshot["source_id"]
            / snapshot["sha256"],
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if receipt["snapshot_sha256"] != snapshot["sha256"]:
            raise ValueError("build snapshot identity mismatch")
    (
        _validate_rows_streaming
        if _streaming_rows(manifest["identity"])
        else _validate_rows
    )(
        path,
        proof_store=proof_store,
        verification_mode=verification_mode,
        progress=progress,
    )
    progress.start_phase("document_assignments")
    with metadata_operation(scratch_root(), max_database_bytes=None):
        verify_document_assignments(
            path, manifest["identity"], iter_rows(path / "documents.jsonl")
        )
    progress.close()
    return manifest


def _manifest_payload(build: dict[str, Any]) -> dict[str, Any]:
    files = {key: value for key, value in build["files"].items() if key != "build.json"}
    return {
        "schema_version": 1,
        "corpus_id": build["identity"]["project_id"],
        "build_id": build["build_id"],
        "build_identity": build["identity"],
        "stages": build["stages"],
        "snapshots": build["snapshots"],
        "files": files,
    }


def freeze(
    build_dir: Path,
    work_root: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    reuse_completed_preparation: bool = False,
) -> Path:
    """Publish a release identity, optionally retaining its completed local payload."""
    build_dir = Path(build_dir).resolve()
    if reuse_completed_preparation:
        from klab.corpus.completed_release import freeze_retained

        return freeze_retained(build_dir, work_root)
    # One operation, so a snapshot authenticated for the build is not hashed again
    # when the staged release is verified.
    with _verification_operation():
        return _freeze(build_dir, work_root, proof_store, verification_mode)


def _freeze(
    build_dir: Path,
    work_root: Path,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> Path:
    build = verify_build(
        build_dir, proof_store=proof_store, verification_mode=verification_mode
    )
    release_config = build["identity"]["release"]
    for view in ("lm", "chat"):
        selected = release_config[view]
        if not selected.get("selected", False):
            continue
        for split in selected.get("training_splits", ("train", "validation")):
            if split not in {"train", "validation"}:
                raise ValueError("test split cannot be a training view")
            with (build_dir / view / f"{split}.jsonl").open("rb") as stream:
                if not any(
                    chunk.strip() for chunk in iter(lambda: stream.read(64 * 1024), b"")
                ):
                    raise ValueError(f"selected {view}/{split} training view is empty")
    payload = _manifest_payload(build)
    release_id = canonical_digest(payload)
    destination = (
        Path(work_root) / "corpora" / payload["corpus_id"] / "releases" / release_id
    )
    if destination.exists():
        verify_release(
            destination, proof_store=proof_store, verification_mode=verification_mode
        )
        if _load(destination / "manifest.json") != {
            **payload,
            "release_id": release_id,
        }:
            raise ValueError("existing release has different identity")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".release-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary)
        progress = open_progress(release_id)
        progress.start_phase(
            "copy",
            expected_input_bytes=sum(
                item["size"] for item in payload["files"].values()
            ),
        )
        # One read of each build file: the bytes are hashed as they are written and
        # must match the digests verify_build just checked, so publication cannot
        # carry a file that changed in between.
        copy_inventory(build_dir, staging, payload["files"], _safe, progress)
        progress.close()
        manifest_path = staging / "manifest.json"
        with manifest_path.open("xb") as stream:
            stream.write(canonical_json({**payload, "release_id": release_id}) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        verify_release(
            staging,
            expected_id=release_id,
            proof_store=proof_store,
            verification_mode=verification_mode,
            _domain_cold=True,
            _hashed=frozenset(payload["files"]),
        )
        from klab.result_publication import sync_directory
        from klab.safe_io import rename_noreplace

        for entry in sorted(staging.rglob("*"), reverse=True):
            if entry.is_dir():
                sync_directory(entry)
        sync_directory(staging)
        try:
            rename_noreplace(staging, destination)
            sync_directory(destination.parent)
        except FileExistsError:
            verify_release(
                destination,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
    return destination


def verify_release(
    path: Path,
    *,
    expected_id: str | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode | None = None,
    _domain_cold: bool = False,
    _hashed: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Verify a release, reusing a signed receipt when the caller or operator allows.

    A caller passing `verification_mode` decides for itself. Otherwise the
    receipt store chosen by `verification_settings` applies, which is how every
    command that reads a release shares one cold verification.
    """
    if verification_mode is None:
        ambient = current_settings()
        proof_store, verification_mode = (
            ambient.proof_store,
            ambient.verification_mode,
        )
    path = Path(path).resolve()
    with _verification_operation():
        if (
            not _domain_cold
            and proof_store is not None
            and verification_mode == "verified_reuse"
        ):
            from klab.experiments.artifacts import Artifact, verify_artifact

            manifest = _load(path / "manifest.json")
            if manifest["release_id"] != _release_location_id(
                path, manifest, expected_id
            ):
                raise ValueError("release directory identity mismatch")
            verify_artifact(
                Artifact(
                    kind="corpus_release",
                    version=manifest["schema_version"],
                    producer="klab",
                    identifier=manifest["release_id"],
                    sha256=manifest["release_id"],
                    path=str(path),
                ),
                path / "manifest.json",
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            return manifest
        return _verify_release_cold(
            path,
            expected_id=expected_id,
            proof_store=proof_store,
            verification_mode=verification_mode,
            hashed=_hashed,
        )


def _release_closure_key(path: Path, manifest: dict[str, Any]) -> tuple[object, ...]:
    """Use the artifact verifier's typed inventory and external snapshot binding."""
    from klab.experiments.artifacts import Artifact, _artifact_key

    release_id = manifest["release_id"]
    return _artifact_key(
        Artifact(
            kind="corpus_release",
            version=manifest["schema_version"],
            producer="klab",
            identifier=release_id,
            sha256=release_id,
            path=str(path),
        ),
        path,
    )


def _release_location_id(path, manifest, expected_id):
    from klab.corpus.completed_release import in_place

    if expected_id is not None:
        return expected_id
    return manifest["release_id"] if in_place(path, manifest) else path.name


def _verify_release_cold(
    path: Path,
    *,
    expected_id: str | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
    hashed: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Verify the complete release including snapshotted source and lineage evidence.

    `hashed` names files whose bytes the caller already hashed and matched
    against the manifest while writing them; only their presence and size are
    rechecked here.
    """
    path = Path(path).resolve()
    proofs = _verified_releases.get()
    if proofs is not None and path in proofs:
        proof = proofs[path]
        if proof.closure_key != _release_closure_key(
            path, proof.manifest
        ) or proof.manifest["release_id"] != _release_location_id(
            path, proof.manifest, expected_id
        ):
            raise ValueError("release changed within verification operation")
        return proof.manifest
    manifest = _load(path / "manifest.json")
    from klab.corpus.completed_preparation import (
        local_scope,
        verify_completed_release,
    )

    if local_scope(manifest) is not None:
        from klab.corpus.completed_release import in_place, verify_retained_release

        manifest = (
            verify_retained_release(path, manifest, expected_id=expected_id)
            if in_place(path, manifest)
            else verify_completed_release(path, manifest, expected_id=expected_id)
        )
        if proofs is not None:
            proofs[path] = _VerifiedRelease(
                manifest, _release_closure_key(path, manifest)
            )
        return manifest
    release_id = manifest.get("release_id")
    if release_id != (expected_id or path.name) or len(release_id) != 64:
        raise ValueError("release directory identity mismatch")
    if (
        canonical_digest(
            {key: value for key, value in manifest.items() if key != "release_id"}
        )
        != release_id
    ):
        raise ValueError("release manifest identity mismatch")
    closure_key = _release_closure_key(path, manifest) if proofs is not None else None
    progress = open_progress(release_id)
    _files(
        path,
        {
            **manifest["files"],
            "manifest.json": {
                "sha256": sha256_file(path / "manifest.json"),
                "size": (path / "manifest.json").stat().st_size,
            },
        },
        progress,
        hashed,
    )
    progress.start_phase("stage_receipts")
    _verify_stages(path, manifest["stages"], manifest["build_identity"], progress)
    from klab.corpus.split_assignments import verify_document_assignments

    progress.start_phase("snapshots")
    for snapshot in manifest["snapshots"]:
        receipt = _verified_snapshot(
            path.parent.parent
            / "snapshots"
            / snapshot["source_id"]
            / snapshot["sha256"],
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        if receipt["snapshot_sha256"] != snapshot["sha256"]:
            raise ValueError("release snapshot identity mismatch")
    (
        _validate_rows_streaming
        if _streaming_rows(manifest["build_identity"])
        else _validate_rows
    )(
        path,
        proof_store=proof_store,
        verification_mode=verification_mode,
        progress=progress,
    )
    progress.start_phase("document_assignments")
    with metadata_operation(scratch_root(), max_database_bytes=None):
        verify_document_assignments(
            path, manifest["build_identity"], iter_rows(path / "documents.jsonl")
        )
    progress.close()
    if proofs is not None:
        if closure_key != _release_closure_key(path, manifest):
            raise ValueError("release changed during verification operation")
        proofs[path] = _VerifiedRelease(manifest, closure_key)
    return manifest


def describe(path: Path, *, tokenizer: Path | None = None) -> dict[str, Any]:
    manifest = verify_release(path)
    report = _load(Path(path) / "report.json")
    from klab.corpus.measurement import measure_views

    measured = measure_views(
        Path(path), tokenizer, release_spec=manifest["build_identity"]["release"]
    )
    if tokenizer is not None:
        from klab.corpus.measurement import measure_source_rights

        counted = measure_source_rights(Path(path), tokenizer)
        rights = report["rights"]
        report = {
            **report,
            "rights": {
                **rights,
                "training_eligibility": {
                    state: {
                        **info,
                        "training_documents": counted[state]["documents"],
                        "source_tokens": counted[state]["source_tokens"],
                        "token_count_reason": (
                            None
                            if counted[state]["source_tokens"] is not None
                            else "not_eligible_for_training"
                        ),
                    }
                    for state, info in rights["training_eligibility"].items()
                },
            },
        }
    result = {
        "release_id": manifest["release_id"],
        "report": report,
        "measurement": measured,
    }
    if tokenizer is not None:
        result["tokenizer_sha256"] = measured["tokenizer_sha256"]
        result["token_totals"] = {
            f"{view}/{split}": item["actual_tokens"]
            for view, splits in measured["views"].items()
            for split, item in splits.items()
        }
    return result


def sources(path: Path) -> dict[str, Any]:
    verify_release(path)
    return {"sources": _load(Path(path) / "sources.json")}


def audit(path: Path) -> dict[str, Any]:
    verify_release(path)
    return _load(Path(path) / "audit.json")


def sample(
    path: Path,
    *,
    domain: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    verify_release(path)
    if limit < 0 or limit > 1000:
        raise ValueError("sample limit must be between 0 and 1000")
    if limit == 0:
        return {"records": []}
    root = Path(path)
    documents = {row["document_id"]: row for row in read_rows(root / "documents.jsonl")}
    result = []
    for row in read_rows(root / "lineage.jsonl"):
        if kind and row["record_kind"] not in (
            {"chat_sft", "tool_episode"} if kind == "chat" else {kind}
        ):
            continue
        if status and row.get("validation_status") != status:
            continue
        domains = sorted(
            set(row.get("domains", []))
            | {
                value
                for parent in row["parent_document_ids"]
                for value in documents[parent]["domains"]
            }
        )
        if domain and domain not in domains:
            continue
        result.append({**row, "domains": domains})
        if len(result) >= limit:
            break
    return {"records": result}


def review(path: Path, **filters: Any) -> dict[str, Any]:
    return sample(path, **filters)


def lineage(path: Path, record_id: str) -> dict[str, Any]:
    verify_release(path)
    root = Path(path)
    document_map = {
        row["document_id"]: row for row in read_rows(root / "documents.jsonl")
    }
    spans = {row["record_id"]: row for row in read_rows(root / "spans.jsonl")}
    sources_map = {row["id"]: row for row in _load(root / "sources.json")}
    for row in read_rows(root / "lineage.jsonl"):
        if row["record_id"] == record_id:
            parents = [
                {
                    "document": document_map[key],
                    "span": spans[key],
                    "source": sources_map[document_map[key]["source_id"]],
                }
                for key in row["parent_document_ids"]
            ]
            result = {"lineage": row, "parents": parents}
            if scenario_id := row.get("scenario_id"):
                result["scenario"] = next(
                    item
                    for item in read_rows(root / "scenarios.jsonl")
                    if item["scenario_id"] == scenario_id
                )
            if generation_id := row.get("generation_id"):
                result["generation"] = next(
                    item
                    for item in read_rows(root / "generations.jsonl")
                    if item["record_id"] == generation_id
                )
            return result
    raise ValueError(f"unknown record ID: {record_id}")


def consumers(path: Path, runs_dir: Path) -> dict[str, Any]:
    manifest = verify_release(path)
    release = Path(path)
    result = []
    from klab.training.manifest import read_manifest

    evidence = ("manifest.json", "report.json", "license-report.json", "audit.json")
    for run_manifest in sorted(Path(runs_dir).rglob("manifest.json")):
        if run_manifest.parent.name == "corpus":
            continue
        run = run_manifest.parent
        try:
            record = read_manifest(run_manifest)
            dataset = record["effective_config"]["dataset"]
            if dataset.get("revision") != manifest["release_id"]:
                continue
            inventory = {item["relative_path"]: item for item in record["artifacts"]}
            for name in (*evidence, "export.json"):
                relative = f"corpus/{name}"
                artifact = inventory[relative]
                owned = _safe(run, relative)
                if (
                    owned.stat().st_size != artifact["size_bytes"]
                    or sha256_file(owned) != artifact["sha256"]
                ):
                    raise ValueError("tampered run evidence artifact")
                if name != "export.json" and sha256_file(owned) != sha256_file(
                    release / name
                ):
                    raise ValueError("run corpus evidence does not match release")
            export = _load(run / "corpus/export.json")
            if export.get("release_id") != manifest["release_id"]:
                raise ValueError("run export release mismatch")
            checkpoints = [
                {"path": str(file.relative_to(run)), "sha256": sha256_file(file)}
                for file in sorted((run / "checkpoints").rglob("manifest.json"))
            ]
            result.append(
                {
                    "run_id": record["run_id"],
                    "run_path": str(run),
                    "checkpoints": checkpoints,
                }
            )
        except OSError, ValueError, KeyError, TypeError:
            continue
    return {"release_id": manifest["release_id"], "consumers": result}
