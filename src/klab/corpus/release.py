"""Immutable publication, verification and read-only corpus inspection."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
from array import array
from collections import Counter
from collections.abc import Iterator
from contextlib import ExitStack, closing, contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.bounded_metadata import metadata_operation, read_metadata
from klab.corpus.build_stages import StageStore, run_stage
from klab.corpus.document_coverage import document_coverage
from klab.corpus.jsonl_records import iter_rows
from klab.corpus.measurement_inputs import MeasurementTokenizer
from klab.corpus.normalized_spans import (
    NORMALIZED_SPANS,
    SPAN_MEMBER,
    has_span_projection,
    span_dependencies,
    span_partitions,
    verify_span_projection,
)
from klab.corpus.project import NormalizerOptions
from klab.corpus.quota import require_estimated_supply
from klab.corpus.release_members import (
    DOCUMENT_MEMBERS,
    SHARED_DOCUMENT,
    SHARED_STAGE,
    ReleasePlacement,
    SharedPublication,
    normalized_retention_lease,
    publish_shared_members,
    release_retention_lease,
    shared_member_path,
    shared_members,
)
from klab.corpus.release_receipt import receipt_first, verifier_digest
from klab.corpus.release_verify import (
    INVENTORY_RECEIPT_MIN_BYTES,
    current_settings,
    hash_inventory,
    line_boundaries,
    open_progress,
    range_records,
    scratch_root,
)
from klab.corpus.shards import chunk_ranges, range_file_identity, run_tasks
from klab.corpus.split_assignments import document_assignments
from klab.experiments.artifacts import snapshot_dependencies
from klab.safe_io import read_file_range
from klab.training.manifest import sha256_file

if TYPE_CHECKING:
    from klab.corpus.progress import BuildProgress
    from klab.verification_proofs import ProofStore, VerificationMode


@dataclass(frozen=True)
class _VerifiedRelease:
    """A release verified in this operation and the stat fingerprint it was verified at."""

    manifest: dict[str, Any]
    fingerprint: str


_verified_releases: ContextVar[dict[Path, _VerifiedRelease] | None] = ContextVar(
    "verified_releases", default=None
)


@contextmanager
def _verification_operation():
    """Share full authentication, snapshots included, with nested consumers, never across operations."""
    if _verified_releases.get() is not None:
        yield
        return
    from klab.corpus.release_receipt import fingerprint

    verified: dict[Path, _VerifiedRelease] = {}
    token, snapshots = _verified_releases.set(verified), _snapshot_receipts.set({})
    try:
        yield
        for path, proof in verified.items():
            if fingerprint(path) != proof.fingerprint:
                raise ValueError("release changed during verification operation")
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
    *,
    stage_outputs: frozenset[str] = frozenset(),
    checked_shared: dict[Path, list[int]] | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> None:
    if not isinstance(inventory, dict):
        raise TypeError("missing artifact inventory")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    shared = shared_members(root, inventory)
    parts = span_partitions(root, inventory)
    if any(
        inventory[name]["storage"] == SHARED_STAGE and name not in stage_outputs
        for name in shared
    ):
        raise ValueError("shared member is not a declared stage output")
    external = set(shared)
    if parts:
        external.add(SPAN_MEMBER)
    if actual != inventory.keys() - external:
        raise ValueError("release artifact inventory mismatch")
    eligible = {
        name: entry for name, entry in inventory.items() if name not in stage_outputs
    }
    reused = {
        name
        for name, path in shared.items()
        if checked_shared
        and path in checked_shared
        and name in eligible
        and range_file_identity(path) == checked_shared[path]
    }
    if (
        parts
        and checked_shared
        and all(
            path in checked_shared and range_file_identity(path) == checked_shared[path]
            for path in span_dependencies(parts)
        )
    ):
        reused.add(SPAN_MEMBER)
    verified_separately = reused | ({SPAN_MEMBER} if parts else set())
    to_hash = (
        {
            name: entry
            for name, entry in eligible.items()
            if name not in verified_separately
        }
        if verified_separately
        else eligible
    )
    if progress is not None:
        progress.start_phase(
            "hash_files",
            expected_input_bytes=sum(entry["size"] for entry in eligible.values()),
        )
        if reused:
            progress.update(
                input_bytes_reused=sum(eligible[name]["size"] for name in reused)
            )
    hash_inventory(
        root,
        to_hash,
        lambda base, name: shared[name] if name in shared else _safe(base, name),
        progress,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    if parts and SPAN_MEMBER not in reused:
        verify_span_projection(
            parts,
            inventory[SPAN_MEMBER],
            on_read=(lambda size: progress.update(input_bytes=size))
            if progress is not None
            else None,
        )


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
            {source_id: identity["sources"][source_id] for source_id in sources},
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
    from klab.corpus.document_normalization import normalize_text
    from klab.corpus.structure_cleaning import clean_structure

    original = normalize_text(raw_text)
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
    identity: dict[str, Any],
    *,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> dict[str, dict[str, Any]]:
    """Bind each source row to the build identity, and the identity to its snapshot.

    Rights, licenses and restriction state are build inputs: the build identity
    records every source's full declaration, and a source row must be exactly
    the row derived from it. A snapshot records only the acquisition identity,
    which must match that declaration and the identity's lock.
    """
    from klab.corpus.acquisition import declaration_sha256
    from klab.corpus.build_outputs import source_row
    from klab.corpus.project import SourceDeclaration, acquisition_identity_payload

    declared = identity["sources"]
    if set(declared) != set(sources):
        raise ValueError("source inventory differs from build identity")
    snapshots: dict[str, dict[str, Any]] = {}
    for source in sources.values():
        declaration = declared[source["id"]]
        if source != source_row(declaration, source["snapshot_sha256"]):
            raise ValueError("source row differs from its build identity declaration")
        snapshot_id = source["snapshot_sha256"]
        if not snapshot_id:
            continue
        snapshot = _verified_snapshot(
            root.parent.parent / "snapshots" / source["id"] / snapshot_id,
            proof_store=proof_store,
            verification_mode=verification_mode,
        )
        snapshots[source["id"]] = snapshot
        declared_source = SourceDeclaration.model_validate(declaration)
        if (
            snapshot["declaration"] != acquisition_identity_payload(declared_source)
            or snapshot["declaration_sha256"] != declaration_sha256(declared_source)
            or identity["lock"][source["id"]]["declaration_sha256"]
            != snapshot["declaration_sha256"]
        ):
            raise ValueError("source acquisition differs from build identity")
    return snapshots


def streaming_verification(identity: dict[str, Any]) -> bool:
    """Whether the bounded-memory row validator applies (no reviewed admission)."""
    return (
        identity["release"].get("record_admission") is None
        and not identity["release"]["chat"]["selected"]
        and all(spec["kind"] == "lm_text" for spec in identity["transforms"])
        and identity["release"].get("fraction") is None
    )


# Lines per parallel LM-view task; both files of a view are cut at the same line.
VIEW_LINES = 50_000


def _readonly(database: str) -> sqlite3.Connection:
    return sqlite3.connect(Path(database).resolve().as_uri() + "?mode=ro", uri=True)


def _span_rows(
    spans: str, start: int, end: int, source_file: tuple[str, str] | None = None
):
    rows = []
    for span, offset, length in range_records(Path(spans), start, end):
        if (
            source_file is not None
            and (span["source_id"], span["raw_path"]) != source_file
        ):
            raise ValueError("span is outside its normalized source partition")
        rows.append(
            (span["record_id"], span["raw_path"], span["line_start"], offset, length)
        )
    return rows, end - start


def _declared_keys(doc: dict[str, Any], methods: tuple[str, ...]) -> list[str]:
    """The duplicate keys a release's policy declares for one document."""
    from klab.corpus.duplicate_evidence import origin_keys

    by_method = {
        "raw": lambda: ["raw:" + doc["raw_content_sha256"]],
        "normalized": lambda: ["normalized:" + doc["content_sha256"]],
        "canonical_origin": lambda: ["origin:" + key for key in origin_keys(doc)],
    }
    return [key for method in methods for key in by_method[method]()]


def _document_rows(
    span_database: str,
    documents: str,
    start: int,
    end: int,
    options: NormalizerOptions,
    keys: tuple[str, ...],
):
    db = _readonly(span_database)
    try:
        rows, duplicate_keys = [], []
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
            duplicate_keys.extend(
                (key, doc["document_id"]) for key in _declared_keys(doc, keys)
            )
        return rows, duplicate_keys, end - start
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
    from klab.corpus.document_normalization import normalize_text
    from klab.corpus.jsonl_records import iter_jsonl_records, verified_line_replay

    if file is None or file["role"] != "document":
        raise ValueError("document lacks pinned file rights")
    decision = file["rights"]
    raw_path = _safe(Path(files_root), raw_name)
    raw_sha = file["sha256"]
    normalizer = (
        "cnxml-text-v1" if raw_path.suffix.lower() == ".cnxml" else options.identity
    )
    db = _readonly(database)
    count = 0
    try:
        with (
            verified_line_replay(raw_path, expected_sha256=raw_sha) as raw_stream,
            Path(documents).open("rb") as document_stream,
            Path(spans).open("rb") as span_stream,
        ):
            current_line = current_offset = 0
            current_item = None
            raw_rows = iter_jsonl_records(raw_stream, source=str(raw_path))
            for doc_offset, doc_length, span_offset, span_length in db.execute(
                "SELECT doc_offset, doc_length, span_offset, span_length FROM docs "
                "WHERE source=? AND raw_path=? ORDER BY line_start, id",
                (source["id"], raw_name),
            ):
                count += 1
                doc = json.loads(
                    read_file_range(document_stream, doc_offset, doc_length)
                )
                span = json.loads(
                    read_file_range(span_stream, span_offset, span_length)
                )
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
                        raw_stream.rewind()
                        raw_rows = iter_jsonl_records(raw_stream, source=str(raw_path))
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
                        raw_stream.rewind()
                        raw_rows = iter_jsonl_records(raw_stream, source=str(raw_path))
                        current_line = 0
                    while current_line < index:
                        parsed = next(raw_rows, None)
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
                        normalize_text(source_text) != doc["text"]
                        or doc["raw_content_sha256"] != doc["content_sha256"]
                    ):
                        raise ValueError("document raw row content mismatch")
                else:
                    raise ValueError("unsupported document snapshot span")
    finally:
        db.close()
    return source["id"], raw_name, count, raw_path.stat().st_size


def _replay_sources(
    root: Path,
    database: str,
    groups: list[tuple[str, str, int]],
    receipt: dict[str, Any],
    sources: dict[str, dict],
    snapshots: dict[str, dict],
    rights: dict[tuple[str, str], dict],
    options: NormalizerOptions,
    progress: BuildProgress | None,
    *,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
) -> None:
    """Replay source files or reuse checks of the exact sealed replay inputs."""
    from klab.corpus import release_receipt

    files_roots = {
        source_id: root.parent.parent
        / "snapshots"
        / source_id
        / sources[source_id]["snapshot_sha256"]
        / "files"
        for source_id in {group[0] for group in groups}
    }
    members = ReleaseMembers(root, receipt)
    documents = members.member("documents.jsonl")
    support = [members.member(name) for name in ("license-report.json", "sources.json")]
    parts = span_partitions(root, receipt["files"])
    span_files = {part.source_file: part.spans for part in parts}
    inline_spans = None if parts else members.member(SPAN_MEMBER)
    use_receipts = proof_store is not None and verification_mode == "verified_reuse"
    if use_receipts:
        # The path separates build, staged, and release checks at distinct locations.
        replay_inputs = canonical_digest(
            [
                str(root),
                receipt["build_id"],
                [
                    receipt["files"][name]
                    for name in (
                        *DOCUMENT_MEMBERS,
                        "license-report.json",
                        "sources.json",
                    )
                ],
            ]
        )
    tasks = []
    unfinished = {}
    total = reused = 0
    for source_id, raw_name, count in groups:
        raw_path = _safe(files_roots[source_id], raw_name)
        spans = (
            inline_spans
            if inline_spans is not None
            else span_files.get((source_id, raw_name))
        )
        if spans is None:
            raise ValueError(
                f"missing normalized span partition: {source_id}/{raw_name}"
            )
        size = raw_path.stat().st_size
        total += size
        file = rights.get((source_id, raw_name))
        if use_receipts and file is not None and size >= INVENTORY_RECEIPT_MIN_BYTES:
            dependencies = [documents, spans, *support]
            identity = canonical_digest(
                [
                    replay_inputs,
                    source_id,
                    raw_name,
                    count,
                    sources[source_id]["snapshot_sha256"],
                    file["sha256"],
                ]
            )
            check = release_receipt.lookup(
                proof_store, "raw_replay", raw_path, identity, dependencies
            )
            if check.hit:
                reused += size
                continue
            if check.index is not None:
                unfinished[source_id, raw_name] = (
                    raw_path,
                    identity,
                    check.index,
                    dependencies,
                )
        tasks.append(
            (
                database,
                str(documents),
                str(spans),
                sources[source_id],
                raw_name,
                file,
                str(files_roots[source_id]),
                snapshots[source_id]["declaration"]["acquisition"].get("text_field"),
                options,
            )
        )

    if progress is not None:
        progress.start_phase("raw_replay", expected_input_bytes=total)
        progress.update(input_bytes_reused=reused)

    def consume(result):
        source_id, raw_name, count, size = result
        if proof := unfinished.pop((source_id, raw_name), None):
            path, identity, index, dependencies = proof
            release_receipt.record(
                proof_store, "raw_replay", path, identity, dependencies, index=index
            )
        if progress is not None:
            progress.update(documents=count, input_bytes=size)

    if tasks:
        run_tasks(_raw_group, tasks, consume, total_bytes=total - reused)


def _lineage_rows(database: str, lineage: str, start: int, end: int, origins):
    from klab.corpus.provenance import shape_for_record, validate_lineage

    db = _readonly(database)
    try:
        ids = array("q")
        for row, _, _ in range_records(Path(lineage), start, end):
            record_id = row["record_id"]
            found = db.execute(
                "SELECT representative, dropped, source, split, content_sha256, "
                "source_family, modality, domains, rowid FROM docs WHERE id=?",
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
                ordinal,
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
            ids.append(ordinal)
        return ids, end - start
    finally:
        db.close()


def _stage_rows(
    database: str, path: str, start: int, end: int, selected: frozenset[str]
):
    db = _readonly(database)
    try:
        ids = array("q")
        for row, _, _ in range_records(Path(path), start, end):
            doc = db.execute(
                "SELECT content_sha256, split, source, dropped, rowid FROM docs WHERE id=?",
                (row["record_id"],),
            ).fetchone()
            if (
                doc is None
                or set(row) != {"record_id", "text", "split"}
                or hashlib.sha256(row["text"].encode("utf-8")).hexdigest() != doc[0]
                or row["split"] != doc[1]
                or (selected and doc[2] not in selected)
                or doc[3] is not None
            ):
                raise ValueError("LM stage record differs from selected document")
            ids.append(doc[4])
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
    eligible_sources: frozenset[str],
):
    db = _readonly(database)
    try:
        ids, size = array("q"), 0
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
                    "SELECT content_sha256, split, dropped, source, rowid FROM docs WHERE id=?",
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
                if found[3] not in eligible_sources:
                    raise ValueError("LM view includes an unselected document")
                ids.append(found[4])
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


def release_index_workspace(root: Path, scratch: Path) -> Path:
    """Locate row index checkpoints for this verification path."""
    return scratch / "release-indexes" / canonical_digest(str(root.resolve()))


@contextmanager
def _release_index_store(root, verification_mode):
    """Own temporary index checkpoints until the row verdict completes."""
    from klab.verification_proofs import verification_options

    settings = current_settings()
    if settings.scratch_dir is None or verification_mode != "verified_reuse":
        yield None
        return
    scratch = scratch_root()
    trust = verification_options(scratch, read_only=True)["proof_store"]
    if trust is None:
        if settings.echo is not None:
            print(
                "release row index checkpoints require owned scratch",
                file=settings.echo,
            )
        yield None
        return
    workspace = release_index_workspace(root, scratch)
    parent = workspace.parent
    parent.mkdir(exist_ok=True, mode=0o700)
    if not trust.trusted(parent):
        raise ValueError(f"unsafe release index root: {parent}")
    workspace.mkdir(exist_ok=True, mode=0o700)
    if not trust.trusted(workspace):
        raise ValueError(f"unsafe release index workspace: {workspace}")
    descriptor = os.open(
        workspace / "lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    with os.fdopen(descriptor, "wb") as lock:
        # Process record locks are released on crash even if a worker inherits the fd.
        fcntl.lockf(lock, fcntl.LOCK_EX)
        for abandoned in workspace.glob(".verify-corpus-*"):
            if abandoned.is_dir() and not abandoned.is_symlink():
                shutil.rmtree(abandoned)
        store = StageStore(workspace)
        yield store
        for stage in ("spans", "documents"):
            store.prune(stage, keep=0)


@contextmanager
def _row_index_writer(path):
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("PRAGMA synchronous=OFF")
        db.execute("PRAGMA temp_store=FILE")
        db.execute("PRAGMA cache_size=-65536")
        yield db
        db.commit()


def release_row_index_inputs(
    paths: dict[str, tuple[Path, ...]],
    files: dict[str, Any],
    identity: dict[str, Any],
    options: NormalizerOptions,
    *,
    checkpointed: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind the row index keys to the exact files and verification rules."""
    producer = verifier_digest("release") if checkpointed else None
    file_bindings = {
        name: {
            "entry": files[name],
            "generation": [range_file_identity(path) for path in members],
        }
        for name, members in paths.items()
    }
    return (
        {"files": {"spans.jsonl": file_bindings["spans.jsonl"]}, "producer": producer},
        {
            "files": file_bindings,
            "producer": producer,
            "normalizer": options.model_dump(mode="json"),
            "keys": identity["release"]["policy"]["duplicates"]["keys"],
        },
    )


def _release_row_indexes(
    root, receipt, identity, options, temporary, store, phase, advance, progress
):
    """Build or restore the checked indexes used by downstream replay queries."""
    members = ReleaseMembers(root, receipt)
    parts = span_partitions(root, receipt["files"])
    span_sources = (
        tuple((part.spans, part.source_file) for part in parts)
        if parts
        else ((members.member(SPAN_MEMBER), None),)
    )
    spans_paths = tuple(path for path, _ in span_sources)
    documents_path = members.member("documents.jsonl")
    inputs_by_name = {"documents.jsonl": (documents_path,), SPAN_MEMBER: spans_paths}
    span_inputs, document_inputs = release_row_index_inputs(
        inputs_by_name,
        receipt["files"],
        identity,
        options,
        checkpointed=store is not None,
    )
    file_bindings = document_inputs["files"]

    def check_inputs(files):
        if any(
            [range_file_identity(path) for path in inputs_by_name[name]]
            != item["generation"]
            for name, item in files.items()
        ):
            raise ValueError("release row index inputs changed")

    def run(name, inputs, produce):
        if store is None:
            result = {}
            produce(temporary, result)
            return str(temporary / f"{name}.sqlite"), result
        store.discard_incomplete(name)

        def reused():
            if progress is not None:
                progress.update(
                    input_bytes_reused=sum(
                        receipt["files"][member]["size"] for member in inputs["files"]
                    )
                )
                progress.record(f"reused_release_{name}_index")

        def checked(directory, result):
            produce(directory, result)
            check_inputs(inputs["files"])

        directory, result = run_stage(store, name, inputs, checked, on_reuse=reused)
        store.prune(name, keep=1, protect=store.key(inputs))
        return str(directory / f"{name}.sqlite"), result

    def build_spans(directory, result):
        with _row_index_writer(directory / "spans.sqlite") as db:
            db.execute(
                "CREATE TABLE spans (id TEXT, raw_path TEXT, line_start INTEGER, "
                "byte_offset INTEGER, byte_length INTEGER)"
            )

            def load(result):
                rows, size = result
                db.executemany("INSERT INTO spans VALUES (?,?,?,?,?)", rows)
                advance(len(rows), size)

            run_tasks(
                _span_rows,
                (
                    (str(path), start, end, source_file)
                    for path, source_file in span_sources
                    for start, end in chunk_ranges(path)
                ),
                load,
                total_bytes=file_bindings["spans.jsonl"]["entry"]["size"],
            )
            _unique_index(db, "spans_id", "spans", "id", "duplicate evidence span")
            result["rows"] = db.execute("SELECT COUNT(*) FROM spans").fetchone()[0]

    def build_documents(directory, result):
        phase("spans", *spans_paths)
        span_database, spans = run("spans", span_inputs, build_spans)
        with _row_index_writer(directory / "documents.sqlite") as db:
            db.execute(
                "CREATE TABLE docs (id TEXT, source TEXT, split TEXT, "
                "representative TEXT, dropped TEXT, raw_path TEXT, "
                "line_start INTEGER, content_sha256 TEXT, source_family TEXT, "
                "modality TEXT, domains TEXT, doc_offset INTEGER, doc_length INTEGER, "
                "span_offset INTEGER, span_length INTEGER)"
            )
            db.execute("CREATE TABLE duplicate_keys (key TEXT, id TEXT)")

            def load(rows):
                docs, keys, size = rows
                db.executemany(
                    "INSERT INTO docs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", docs
                )
                db.executemany("INSERT INTO duplicate_keys VALUES (?,?)", keys)
                advance(len(docs), size)

            total = phase("documents", documents_path)
            run_tasks(
                _document_rows,
                (
                    (
                        span_database,
                        str(documents_path),
                        start,
                        end,
                        options,
                        tuple(document_inputs["keys"]),
                    )
                    for start, end in chunk_ranges(documents_path)
                ),
                load,
                total_bytes=total,
            )
            db.commit()
            _unique_index(db, "docs_id", "docs", "id", "duplicate document identity")
            if db.execute("SELECT COUNT(*) FROM docs").fetchone()[0] != spans["rows"]:
                raise ValueError("document evidence span inventory mismatch")
            db.execute(
                "CREATE INDEX docs_raw_order_idx ON docs(source, raw_path, line_start, id)"
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

    phase("documents", documents_path, *spans_paths)
    database, _ = run("documents", document_inputs, build_documents)
    check_inputs(file_bindings)
    if identity["split"].get("assignment_file") is not None:
        if progress is not None:
            progress.start_phase("document_assignments")
        with metadata_operation(temporary, max_database_bytes=None):
            assignments = document_assignments(root, identity)
            with closing(_readonly(database)) as db:
                for document_id, split in db.execute("SELECT id, split FROM docs"):
                    assignments.verify_document(document_id, split)
    return database


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
        root,
        sources,
        identity,
        proof_store=proof_store,
        verification_mode=verification_mode,
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

    with (
        _release_index_store(root, verification_mode) as store,
        tempfile.TemporaryDirectory(
            prefix=".verify-corpus-",
            dir=store.root if store is not None else scratch_root(),
        ) as temporary,
        progress.track_scratch(store.root if store is not None else Path(temporary))
        if progress is not None
        else nullcontext(),
    ):
        database = _release_row_indexes(
            root,
            receipt,
            identity,
            options,
            Path(temporary),
            store,
            phase,
            advance,
            progress,
        )
        coverage_path = Path(temporary) / "coverage.bits"
        with closing(_readonly(database)) as db:
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA cache_size=-65536")

            groups = db.execute(
                "SELECT source, raw_path, COUNT(*) FROM docs GROUP BY source, raw_path "
                "ORDER BY COUNT(*) DESC"
            ).fetchall()
            _replay_sources(
                root,
                database,
                groups,
                receipt,
                sources,
                snapshots,
                rights,
                options,
                progress,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )

            # The index inserts dense positive row IDs and never deletes rows.
            documents = sum(count for _, _, count in groups)
            kept_by_source = dict(
                db.execute(
                    "SELECT source, COUNT(*) FROM docs WHERE dropped IS NULL GROUP BY source"
                )
            )

            def consume(coverage, result):
                rows, size = result
                coverage.add(rows)
                advance(len(rows), size)

            lineage_path = root / "lineage.jsonl"
            total = phase("lineage", lineage_path)
            with document_coverage(
                coverage_path, documents, duplicate="duplicate lineage ID"
            ) as coverage:
                run_tasks(
                    _lineage_rows,
                    (
                        (database, str(lineage_path), start, end, origins)
                        for start, end in chunk_ranges(lineage_path)
                    ),
                    partial(consume, coverage),
                    total_bytes=total,
                )
                if coverage.count != documents:
                    raise ValueError("document lineage mismatch")

            transforms = identity["transforms"]
            for stage in receipt["stages"]:
                transform = next(
                    spec for spec in transforms if spec["id"] == stage["id"]
                )
                selected = frozenset(transform["inputs"]).intersection(sources)
                stage_path = ReleaseMembers(root, receipt).member(
                    f"stages/{stage['id']}.jsonl"
                )
                total = phase(f"stage:{stage['id']}", stage_path)
                expected = sum(
                    kept_by_source.get(source, 0) for source in (selected or sources)
                )
                with document_coverage(
                    coverage_path, documents, duplicate="duplicate LM stage identity"
                ) as coverage:
                    run_tasks(
                        _stage_rows,
                        (
                            (database, str(stage_path), start, end, selected)
                            for start, end in chunk_ranges(stage_path)
                        ),
                        partial(consume, coverage),
                        total_bytes=total,
                    )
                    if coverage.count != expected:
                        raise ValueError("LM stage omitted selected documents")

            release = identity["release"]
            allowed_sources = set().union(
                *(
                    set(spec["inputs"]).intersection(sources) or sources.keys()
                    for spec in transforms
                )
            )
            shapes = release.get("include_shapes")
            wanted_origins = release.get("include_origins")
            eligible_sources = frozenset(
                source
                for source in allowed_sources
                if (shapes is None or "raw_document" in shapes)
                and (wanted_origins is None or origins[source] in wanted_origins)
            )
            expected = sum(kept_by_source.get(source, 0) for source in eligible_sources)
            with document_coverage(
                coverage_path, documents, duplicate="duplicate LM view identity"
            ) as coverage:
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
                                eligible_sources,
                            )
                            for index, (view_start, link_start) in enumerate(
                                zip(view_starts, link_starts, strict=True)
                            )
                        ),
                        partial(consume, coverage),
                        total_bytes=total,
                    )
                    for name in (f"chat/{split}.jsonl", f"chat/{split}.lineage.jsonl"):
                        if (root / name).stat().st_size:
                            raise ValueError(
                                "unexpected chat records in LM-only release"
                            )
                if coverage.count != expected:
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
        progress.track_scratch(Path(metadata.temporary.name))
        if progress is not None
        else nullcontext(),
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

    members = ReleaseMembers(root, receipt)
    document_map = row_index(
        members.member("documents.jsonl"),
        "document_id",
        duplicate="duplicate document identity",
    )
    assignments = document_assignments(root, identity)
    for doc in document_map.values():
        if doc["content_sha256"] != hashlib.sha256(doc["text"].encode()).hexdigest():
            raise ValueError("document content digest mismatch")
        _verify_cleaning_decision(doc, options)
        if assignments is not None:
            assignments.verify_document(doc["document_id"], doc["split"])
    spans = row_index(members.member("spans.jsonl"))
    sources = {item["id"]: item for item in _load(root / "sources.json")}

    source_origins = {
        key: value.get("origin", "primary_source") for key, value in sources.items()
    }

    snapshots = _verify_source_snapshots(
        root,
        sources,
        identity,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    rights_files = _verify_rights_files(root, sources, snapshots)
    if not spans.same_keys(document_map):
        raise ValueError("document evidence span inventory mismatch")
    previous_file: Path | None = None
    raw = b""
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
        raw_sha = file["sha256"]
        if raw_path != previous_file:
            # Finishing the previous cursor validates trailing unselected rows.
            replay_stack.close()
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
                if hashlib.sha256(raw).hexdigest() != raw_sha:
                    raise ValueError(f"source replay SHA-256 mismatch: {raw_path}")
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
            from klab.corpus.document_normalization import normalize_text

            line_number = span["line_start"]
            if (
                selected_row is None
                or type(line_number) is not int
                or line_number < 1
                or span["line_end"] != line_number
                or doc["source_location"] != f"{span['raw_path']}#row={line_number}"
                or (
                    not options.structure_cleaning
                    and normalize_text(selected_row[text_field]) != doc["text"]
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
                for record in rows(members.member(f"stages/{stage['id']}.jsonl"))
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
    inventory: dict[str, Any],
    progress: BuildProgress | None = None,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
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
        name = f"stages/{stage['id']}.jsonl"
        entry = inventory[name]
        output = shared_member_path(root, name, entry) or _safe(root, name)
        if not output.is_file():
            raise ValueError(f"tampered artifact: {name}")
        generation = range_file_identity(output)
        if generation[2] != entry["size"]:
            raise ValueError(f"tampered artifact: {name}")
        if stage["output_sha256"] != entry["sha256"]:
            raise ValueError("stage receipt does not match output")
        verify = partial(_verify_stage_output, output, stage, generation, progress)
        if (
            proof_store is not None
            and verification_mode == "verified_reuse"
            and entry["size"] >= INVENTORY_RECEIPT_MIN_BYTES
        ):
            receipt_first(
                "stage_output",
                output,
                canonical_digest({"stage": stage, "file": entry}),
                [],
                proof_store,
                verification_mode,
                verify,
                partial(progress.update, input_bytes_reused=2 * generation[2])
                if progress is not None
                else lambda: None,
            )
        else:
            verify()


def _verify_stage_output(
    output: Path,
    stage: dict[str, Any],
    generation: list[int],
    progress: BuildProgress | None,
) -> None:
    """Validate canonical rows, then hash the raw file and ordered stage identity."""
    canonical_count = 0
    total = generation[2]

    def progress_bytes(size: int) -> None:
        if progress is not None:
            progress.update(input_bytes=size)

    def consume(result: tuple[int, int]) -> None:
        nonlocal canonical_count
        canonical_count += result[0]
        progress_bytes(result[1])

    run_tasks(
        _canonical_lines,
        ((str(output), start, end) for start, end in chunk_ranges(output)),
        consume,
        total_bytes=total,
    )
    file_digest, identity_digest = hashlib.sha256(), hashlib.sha256()
    identity_digest.update(canonical_json([stage["id"]])[:-1] + b",[")
    count = size = reported = 0
    with output.open("rb") as stream:
        for line in stream:
            file_digest.update(line)
            size += len(line)
            if size - reported >= 64 * 1024**2:
                progress_bytes(size - reported)
                reported = size
            if not line.strip():
                continue
            if count:
                identity_digest.update(b",")
            identity_digest.update(line.rstrip(b"\n"))
            count += 1
    progress_bytes(size - reported)
    identity_digest.update(b"]]")
    if range_file_identity(output) != generation:
        raise ValueError(f"stage output changed during verification: {output}")
    if (
        size != total
        or file_digest.hexdigest() != stage["output_sha256"]
        or count != stage["output_count"]
        or count != canonical_count
        or identity_digest.hexdigest() != stage["output_id"]
    ):
        raise ValueError("stage receipt does not match output")


def _verify_native_outputs(
    path: Path,
    manifest: dict[str, Any],
    identity: dict[str, Any],
    metadata_name: str,
    progress: BuildProgress,
    *,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
    checked_shared: dict[Path, list[int]] | None = None,
) -> None:
    """Check file membership, deferring stage checksums to ordered stage replay."""
    metadata = path / metadata_name
    inventory = {
        **manifest["files"],
        metadata_name: {
            "sha256": sha256_file(metadata),
            "size": metadata.stat().st_size,
        },
    }
    stages = manifest["stages"]
    stage_outputs = frozenset(f"stages/{stage['id']}.jsonl" for stage in stages)
    if not stage_outputs <= inventory.keys():
        raise ValueError("stage output missing from artifact inventory")
    _files(
        path,
        inventory,
        progress,
        stage_outputs=stage_outputs,
        checked_shared=checked_shared,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
    progress.start_phase(
        "stage_receipts",
        expected_input_bytes=2 * sum(inventory[name]["size"] for name in stage_outputs),
    )
    _verify_stages(
        path,
        stages,
        identity,
        inventory,
        progress,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )


def verify_build(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> dict[str, Any]:
    """Verify staged build and its pinned snapshots, by receipt when one still holds."""
    path = Path(path).resolve()
    if (released := consumed_release(path)) is not None:
        raise ValueError(
            f"build was frozen in place into release {released}; verify the release"
        )
    with _verification_operation():
        manifest = _load(path / "build.json")
        verified, _ = receipt_first(
            "build",
            path,
            manifest["build_id"],
            list(snapshot_dependencies(path, "build.json")),
            proof_store,
            verification_mode,
            lambda: _verify_build_cold(
                path, proof_store=proof_store, verification_mode=verification_mode
            ),
            lambda: manifest,
        )
        return verified


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
    if has_span_projection(manifest["files"]) != bool(
        manifest["identity"].get("reference_normalized_spans")
    ):
        raise ValueError("build span placement differs from its identity")
    progress = open_progress(manifest["build_id"], verification_mode=verification_mode)
    _verify_native_outputs(
        path,
        manifest,
        manifest["identity"],
        "build.json",
        progress,
        proof_store=proof_store,
        verification_mode=verification_mode,
    )
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
        if streaming_verification(manifest["identity"])
        else _validate_rows
    )(
        path,
        proof_store=proof_store,
        verification_mode=verification_mode,
        progress=progress,
    )
    progress.close()
    return manifest


_INLINE_PLACEMENT = ReleasePlacement()


def release_manifest_payload(
    build: dict[str, Any], *, placement: ReleasePlacement = _INLINE_PLACEMENT
) -> dict[str, Any]:
    """Resolve the ordinary freeze payload before computing the release identity."""
    if placement.shared:
        from klab.corpus.preparation_inputs import local_scope

        if local_scope(build) is not None:
            raise ValueError(
                "shared release artifacts require an ordinary Corpus build"
            )
    files = {key: value for key, value in build["files"].items() if key != "build.json"}
    projected_spans = has_span_projection(files)
    if placement.stages:
        for stage in build["stages"]:
            name = f"stages/{stage['id']}.jsonl"
            files[name] = {**files[name], "storage": SHARED_STAGE}
    if placement.documents:
        for name in DOCUMENT_MEMBERS:
            if name != SPAN_MEMBER or not projected_spans:
                files[name] = {**files[name], "storage": SHARED_DOCUMENT}
    if placement.normalized_spans and not projected_spans:
        if not build.get("span_partitions") or not streaming_verification(
            build["identity"]
        ):
            raise ValueError(
                "normalized spans require a native build with a span partition inventory"
            )
        files[SPAN_MEMBER] = {
            **files[SPAN_MEMBER],
            "storage": NORMALIZED_SPANS,
            "partitions": build["span_partitions"],
        }
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
    share_stage_artifacts: bool = False,
    share_document_artifacts: bool = False,
    reference_normalized_spans: bool = False,
) -> Path:
    """Publish a release identity, optionally retaining its completed local payload."""
    build_dir = Path(build_dir).resolve()
    work_root = Path(work_root).resolve()
    placement = ReleasePlacement(
        stages=share_stage_artifacts,
        documents=share_document_artifacts,
        normalized_spans=reference_normalized_spans,
    )
    if placement.shared and build_dir.parent.parent.parent != work_root / "corpora":
        raise ValueError(
            "shared release artifacts require a build in the selected work root"
        )
    if reuse_completed_preparation:
        from klab.corpus.completed_release import freeze_retained

        if placement.shared:
            raise ValueError("shared release artifacts require ordinary Corpus freeze")
        return freeze_retained(build_dir, work_root)
    # One operation, so a snapshot authenticated for the build is hashed once.
    with (
        _verification_operation(),
        normalized_retention_lease(work_root)
        if reference_normalized_spans
        else nullcontext(),
    ):
        return _freeze(build_dir, work_root, proof_store, verification_mode, placement)


CONSUMED = "consumed.json"


def consumed_release(build_dir: Path) -> str | None:
    """The release a build was frozen into, or None if the build still holds its files."""
    marker = Path(build_dir) / CONSUMED
    return _load(marker)["release_id"] if marker.is_file() else None


def _freeze(
    build_dir: Path,
    work_root: Path,
    proof_store: ProofStore | None,
    verification_mode: VerificationMode,
    placement: ReleasePlacement,
) -> Path:
    """Publish the verified build directory as the release by renaming it.

    The payload is never copied. The build directory is renamed to a hidden
    staging name next to the release, `manifest.json` replaces `build.json`
    there, and one more rename publishes it under the release ID. The build's
    path is left as a small tombstone (its `build.json` and a `consumed.json`
    naming the release): the build is consumed, so its files exist once. A
    freeze that stopped part way finds the staging directory on the next run and
    authenticates it before publication.
    """
    from klab.corpus.release_receipt import fingerprint

    releases = Path(work_root) / "corpora" / build_dir.parent.parent.name / "releases"
    staging = releases / f".freezing-{build_dir.name}"
    # Between the first rename and the tombstone the build is only in staging.
    manifest = _load(
        build_dir / "build.json"
        if not staging.exists()
        else _resume_source(staging, build_dir)
    )
    projected_spans = has_span_projection(manifest["files"])
    if projected_spans and build_dir.parents[2] != work_root / "corpora":
        raise ValueError("referenced spans require freeze in the build's work area")
    prospective = (
        release_manifest_payload(manifest, placement=placement)
        if placement.shared
        else None
    )
    if staging.exists():
        return _publish(
            staging,
            build_dir,
            manifest,
            releases,
            proof_store,
            verification_mode=verification_mode,
            verified=False,
            placement=placement,
        )
    if (released := consumed_release(build_dir)) is not None:
        if placement.shared and canonical_digest(prospective) != released:
            raise ValueError("build was already frozen with different shared placement")
        destination = releases / released
        verify_release(
            destination, proof_store=proof_store, verification_mode=verification_mode
        )
        return destination
    before = fingerprint(build_dir, "build.json")
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
    require_estimated_supply(_load(build_dir / "report.json"))
    payload = release_manifest_payload(build, placement=placement)
    release_id = canonical_digest(payload)
    destination = releases / release_id
    shared_members(destination, payload["files"])
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
    parts = span_partitions(destination, payload["files"])
    checked_spans = None
    if parts:
        checked_spans = (
            {path: range_file_identity(path) for path in span_dependencies(parts)}
            if projected_spans
            else verify_span_projection(parts, payload["files"][SPAN_MEMBER])
        )
    # A file written between verifying the build and renaming it would be
    # published unverified; the stat index is what notices.
    if fingerprint(build_dir, "build.json") != before:
        raise ValueError("build changed while it was being verified")
    releases.mkdir(parents=True, exist_ok=True)
    from klab.safe_io import rename_noreplace

    rename_noreplace(build_dir, staging)
    return _publish(
        staging,
        build_dir,
        build,
        releases,
        proof_store,
        verification_mode=verification_mode,
        verified=True,
        placement=placement,
        checked_spans=checked_spans,
    )


def _publish(
    staging: Path,
    build_dir: Path,
    build: dict[str, Any],
    releases: Path,
    proof_store: ProofStore | None,
    *,
    verification_mode: VerificationMode,
    verified: bool,
    placement: ReleasePlacement,
    checked_spans: dict[Path, list[int]] | None = None,
) -> Path:
    """Finish a freeze from the renamed build: tombstone, manifest, one rename.

    `verified` says the staged files are the ones just verified. A resumed freeze
    needs a check of the staging path itself before publishing.
    """
    from klab.corpus.release_receipt import stat_index
    from klab.result_publication import sync_directory
    from klab.safe_io import rename_noreplace

    payload = release_manifest_payload(build, placement=placement)
    release_id = canonical_digest(payload)
    destination = releases / release_id
    if destination.exists():
        raise ValueError(
            f"release {release_id} exists and so does {staging.name}; "
            "the earlier freeze finished, remove the staging directory"
        )
    sealed = canonical_json({**payload, "release_id": release_id}) + b"\n"
    parts = span_partitions(staging, payload["files"])
    shared_placement = placement.shared or bool(parts)
    # Keep a staged manifest visible before any shared payload can be moved.
    # The durable manifest protects the files during subsequent cold replay.
    with (
        release_retention_lease(staging.parents[3])
        if shared_placement
        else nullcontext()
    ):
        tombstone_json = staging / "build.json"
        if tombstone_json.is_file():
            build_dir.mkdir(exist_ok=True)
            if not (build_dir / "build.json").exists():
                _write_durably(build_dir / "build.json", tombstone_json.read_bytes())
        marker = canonical_json({"release_id": release_id}) + b"\n"
        previous_release = consumed_release(build_dir)
        if previous_release is None:
            _write_durably(build_dir / CONSUMED, marker)
        elif previous_release != release_id:
            raise ValueError("retry the staged release with the same freeze options")
        sync_directory(build_dir)
        manifest_path = staging / "manifest.json"
        if manifest_path.exists():
            if manifest_path.read_bytes() != sealed:
                raise ValueError("staged release manifest differs from its build")
        else:
            _write_durably(manifest_path, sealed)
        if shared_placement:
            sync_directory(staging)
            sync_directory(releases)
        shared = (
            publish_shared_members(
                staging,
                payload["files"],
                verified=verified,
                proof_store=proof_store,
                verification_mode=verification_mode,
            )
            if placement.shared
            else SharedPublication({}, {})
        )
        if parts:
            inline = staging / SPAN_MEMBER
            inline_present = inline.exists()
            projected_spans = has_span_projection(build["files"])
            if inline.is_symlink() or (inline_present and projected_spans):
                raise ValueError("unexpected inline span payload in staging")
            if inline_present:
                checked_spans = checked_spans or verify_span_projection(
                    parts, payload["files"][SPAN_MEMBER]
                )
            elif verified and not projected_spans:
                raise ValueError("verified build is missing its inline span payload")
            if checked_spans:
                if any(
                    range_file_identity(path) != stamp
                    for path, stamp in checked_spans.items()
                ):
                    raise ValueError(
                        "normalized span inputs changed before publication"
                    )
                shared.generations.update(checked_spans)
                shared.checked.update(checked_spans)
            if inline_present:
                inline.unlink()
                sync_directory(staging)
        tombstone_json.unlink(missing_ok=True)
    if not verified:
        # Staging will be renamed, so keep this check out of the operation's
        # memo of releases whose paths must still exist when the call returns.
        staged_path = staging.absolute()
        dependencies = list(snapshot_dependencies(staged_path, "manifest.json"))
        _, generation = receipt_first(
            "staged_release",
            staged_path,
            release_id,
            dependencies,
            proof_store,
            verification_mode,
            partial(
                _verify_release_cold,
                staged_path,
                expected_id=release_id,
                proof_store=proof_store,
                verification_mode=verification_mode,
                checked_shared=shared.checked,
            ),
            lambda: None,
        )
    for entry in sorted(staging.rglob("*"), reverse=True):
        if entry.is_dir():
            sync_directory(entry)
    sync_directory(staging)
    if any(
        range_file_identity(path) != before
        for path, before in shared.generations.items()
    ):
        raise ValueError("shared payload changed before release publication")
    if (
        not verified
        and stat_index([staged_path, *dependencies], strict=False)[0] != generation
    ):
        raise ValueError("staged release changed while it was being verified")
    rename_noreplace(staging, destination)
    sync_directory(releases)
    # The files are the verified build's, unchanged; record that for the release.
    _record_frozen(destination, release_id, proof_store, shared.generations)
    return destination


def _resume_source(staging: Path, build_dir: Path) -> Path:
    """Where an interrupted freeze can still read the build's `build.json`."""
    return (
        staging / "build.json"
        if (staging / "build.json").is_file()
        else build_dir / "build.json"
    )


def _write_durably(path: Path, content: bytes) -> None:
    """Write a file that is either absent or complete, even if the process is killed."""
    partial = path.with_name(f".{path.name}.partial")
    with partial.open("wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    partial.replace(path)


def _record_frozen(
    release: Path,
    release_id: str,
    proof_store: ProofStore | None,
    shared_generations: dict[Path, list[int]],
) -> None:
    """Receipt a release whose files were verified as the build it was renamed from."""
    if proof_store is None:
        return
    from klab.corpus import release_receipt

    release_receipt.record(
        proof_store,
        "release",
        release,
        release_id,
        list(snapshot_dependencies(release, "manifest.json")),
        shared_generations=shared_generations,
    )


def verify_release(
    path: Path,
    *,
    expected_id: str | None = None,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode | None = None,
) -> dict[str, Any]:
    """Verify a release, by signed receipt when one still holds and the mode allows.

    A caller passing `verification_mode` decides for itself. Otherwise the
    receipt store chosen by `verification_settings` applies, which is how every
    command that reads a release shares one verification.
    """
    if verification_mode is None:
        ambient = current_settings()
        proof_store, verification_mode = (
            ambient.proof_store,
            ambient.verification_mode,
        )
    path = Path(path).resolve()
    with _verification_operation():
        proofs = _verified_releases.get()
        assert proofs is not None
        if path in proofs:
            return proofs[path].manifest
        manifest = _load(path / "manifest.json")
        identity = manifest["release_id"]
        if identity != _release_location_id(path, manifest, expected_id):
            raise ValueError("release directory identity mismatch")
        manifest, fingerprint = receipt_first(
            "release",
            path,
            identity,
            list(snapshot_dependencies(path, "manifest.json")),
            proof_store,
            verification_mode,
            lambda: _verify_release_cold(
                path,
                expected_id=expected_id,
                proof_store=proof_store,
                verification_mode=verification_mode,
            ),
            lambda: manifest,
        )
        proofs[path] = _VerifiedRelease(manifest, fingerprint)
        return manifest


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
    checked_shared: dict[Path, list[int]] | None = None,
) -> dict[str, Any]:
    """Verify the complete release including snapshotted source and lineage evidence."""
    path = Path(path).resolve()
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
    if canonical_digest(manifest["build_identity"]) != manifest["build_id"]:
        raise ValueError("release build identity differs from its build")
    if manifest["build_identity"].get(
        "reference_normalized_spans"
    ) and not has_span_projection(manifest["files"]):
        raise ValueError("release span placement differs from its build identity")
    progress = open_progress(release_id, verification_mode=verification_mode)
    _verify_native_outputs(
        path,
        manifest,
        manifest["build_identity"],
        "manifest.json",
        progress,
        proof_store=proof_store,
        verification_mode=verification_mode,
        checked_shared=checked_shared,
    )
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
        if streaming_verification(manifest["build_identity"])
        else _validate_rows
    )(
        path,
        proof_store=proof_store,
        verification_mode=verification_mode,
        progress=progress,
    )
    progress.close()
    return manifest


def describe(
    path: Path, *, tokenizer: Path | None = None, mixture: Path | None = None
) -> dict[str, Any]:
    if mixture is not None and tokenizer is None:
        raise ValueError("--mixture needs --tokenizer to count tokens")
    with open_release(path) as release:
        bound = (
            MeasurementTokenizer.from_path(tokenizer) if tokenizer is not None else None
        )
        result = _describe_verified(release, bound, mixture)
        if bound is not None:
            bound.require_unchanged()
        return result


def _description_measurements(
    release: ReleaseReader, tokenizer: MeasurementTokenizer | None
) -> dict[str, Any]:
    settings = current_settings()
    store = settings.proof_store
    base = store.root / "cache" if store is not None else settings.scratch_dir
    cache = (
        base / "describe" / release.manifest["release_id"] if base is not None else None
    )

    def measure() -> dict[str, Any]:
        from klab.corpus.measurement import measure_source_rights, measure_views

        return {
            "views": measure_views(
                release,
                tokenizer,
                release_spec=release.manifest["build_identity"]["release"],
                cache=cache,
            ),
            "rights": measure_source_rights(release, tokenizer, cache=cache)
            if tokenizer is not None
            else None,
        }

    if cache is None:
        return measure()
    inputs = {
        "release_id": release.manifest["release_id"],
        "tokenizer": tokenizer.identity() if tokenizer is not None else None,
        "implementation": verifier_digest("release"),
    }
    stages = StageStore(cache)
    locks = cache / "locks"
    locks.mkdir(parents=True, exist_ok=True)
    with (locks / f"{stages.key(inputs)}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = bytearray()
        if (
            stages.verified(
                "description", inputs, hash_chunks={"measurement.json": data.extend}
            )
            is not None
        ):
            if settings.echo is not None:
                print("describe: reused completed measurement", file=settings.echo)
            return json.loads(data)
        del data
        with stages.build("description", inputs) as (staging, _):
            from klab.corpus.build_outputs import write_json

            value = measure()
            write_json(staging / "measurement.json", value)
            if tokenizer is not None:
                tokenizer.require_unchanged()
            release.require_unchanged()
        return value


def _describe_verified(
    release: ReleaseReader,
    tokenizer: MeasurementTokenizer | None,
    mixture: Path | None,
) -> dict[str, Any]:
    report = _load(release.member("report.json"))
    completed = _description_measurements(release, tokenizer)
    measured, counted = completed["views"], completed["rights"]
    if counted is not None:
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
        "release_id": release.manifest["release_id"],
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
    if mixture is not None:
        from klab.corpus.quota import require_supply, template_requirements

        quotas, exposures = template_requirements(mixture)
        domains = measured["views"]["lm"]["train"]["dimensions"]["source_domain"]
        require_supply(
            {stratum: row["actual_tokens"] for stratum, row in domains.items()},
            quotas,
            exposures,
        )
    return result


@dataclass(frozen=True)
class ReleaseMembers:
    """Map logical Corpus member names through a release manifest."""

    root: Path
    manifest: dict[str, Any]

    def member(self, name: str) -> Path:
        if name not in self.manifest["files"]:
            raise ValueError(f"release does not declare member: {name}")
        entry = self.manifest["files"][name]
        if name == SPAN_MEMBER and entry.get("storage") == NORMALIZED_SPANS:
            raise ValueError("referenced spans have multiple backing files")
        return shared_member_path(self.root, name, entry) or _safe(self.root, name)

    def member_paths(self, name: str) -> tuple[Path, ...]:
        if name == SPAN_MEMBER:
            parts = span_partitions(self.root, self.manifest["files"])
            if parts:
                return tuple(part.spans for part in parts)
        return (self.member(name),)

    def rows(self, name: str) -> Iterator[dict[str, Any]]:
        for path in self.member_paths(name):
            yield from iter_rows(path)


@dataclass(frozen=True)
class ReleaseReader(ReleaseMembers):
    """Logical release members bound to the generation verified in this operation."""

    generation: str

    def require_unchanged(self) -> None:
        """Check the verified generation before publishing a dependent artifact."""
        from klab.corpus.release_receipt import fingerprint

        if fingerprint(self.root) != self.generation:
            raise ValueError("release changed after verification")


@contextmanager
def open_release(
    path: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode | None = None,
) -> Iterator[ReleaseReader]:
    """Verify once and check that the inspected release generation stays stable."""
    root = Path(path).resolve()
    with _verification_operation():
        manifest = verify_release(
            root, proof_store=proof_store, verification_mode=verification_mode
        )
        proofs = _verified_releases.get()
        assert proofs is not None
        yield ReleaseReader(root, manifest, proofs[root].fingerprint)


def resolve_release_reference(value: str, work_root: Path) -> Path:
    """Resolve a directory or a full Corpus release reference to its location."""
    if "@" in value and not Path(value).exists():
        corpus_id, digest = value.split("@", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", corpus_id) or not re.fullmatch(
            r"[0-9a-f]{64}", digest
        ):
            raise ValueError("release reference must be <corpus-id>@<full-sha256>")
        return work_root / "corpora" / corpus_id / "releases" / digest
    return Path(value).resolve()


def sources(path: Path) -> dict[str, Any]:
    with open_release(path) as release:
        return {"sources": _load(release.member("sources.json"))}


def audit(path: Path) -> dict[str, Any]:
    with open_release(path) as release:
        return _load(release.member("audit.json"))


def _parent_domain_index(release: ReleaseReader, stack: ExitStack):
    """Spill only document IDs and domains when lineage needs supplementary domains."""
    from klab.corpus.row_index import RowIndex

    temporary = Path(
        stack.enter_context(
            tempfile.TemporaryDirectory(prefix="sample-domains-", dir=scratch_root())
        )
    )
    db = stack.enter_context(closing(sqlite3.connect(temporary / "domains.sqlite")))
    db.execute("PRAGMA cache_size=-4096")
    return RowIndex(
        db,
        (
            {"id": row["document_id"], "domains": row["domains"]}
            for row in release.rows("documents.jsonl")
        ),
        lambda row: row["id"],
    )


def sample(
    path: Path,
    *,
    domain: str | None = None,
    kind: str | None = None,
    status: str | None = None,
    limit: int = 10,
) -> dict[str, Any]:
    if limit < 0 or limit > 1000:
        raise ValueError("sample limit must be between 0 and 1000")
    kinds = {"chat_sft", "tool_episode"} if kind == "chat" else {kind}
    result: list[dict[str, Any]] = []
    with open_release(path) as release, ExitStack() as stack:
        if limit == 0:
            return {"records": result}
        parent_domains = None
        for row in release.rows("lineage.jsonl"):
            if kind and row["record_kind"] not in kinds:
                continue
            if status and row.get("validation_status") != status:
                continue
            domains = set(row.get("domains", []))
            parents = row["parent_document_ids"]
            # Classified lineage pins its parent domain union. Generated worlds
            # pin their own domain and can also need source domains for inspection.
            if parents and ("origin" not in row or row.get("generator_world_id")):
                if parent_domains is None:
                    parent_domains = _parent_domain_index(release, stack)
                for parent in parents:
                    domains.update(parent_domains[parent]["domains"])
            if domain and domain not in domains:
                continue
            result.append({**row, "domains": sorted(domains)})
            if len(result) >= limit:
                break
        return {"records": result}


def review(path: Path, **filters: Any) -> dict[str, Any]:
    return sample(path, **filters)


def _find_rows(
    release: ReleaseReader, name: str, field: str, wanted: set[str]
) -> dict[str, dict[str, Any]]:
    found = {}
    for row in release.rows(name):
        key = row[field]
        if key in wanted:
            found[key] = row
            if len(found) == len(wanted):
                return found
    return found


def lineage(path: Path, record_id: str) -> dict[str, Any]:
    with open_release(path) as release:
        row = _find_rows(release, "lineage.jsonl", "record_id", {record_id}).get(
            record_id
        )
        if row is None:
            raise ValueError(f"unknown record ID: {record_id}")
        parents = []
        if wanted := set(row["parent_document_ids"]):
            documents = _find_rows(release, "documents.jsonl", "document_id", wanted)
            spans = _find_rows(release, "spans.jsonl", "record_id", wanted)
            sources_map = {
                item["id"]: item for item in _load(release.member("sources.json"))
            }
            parents = [
                {
                    "document": documents[key],
                    "span": spans[key],
                    "source": sources_map[documents[key]["source_id"]],
                }
                for key in row["parent_document_ids"]
            ]
        result = {"lineage": row, "parents": parents}
        if scenario_id := row.get("scenario_id"):
            result["scenario"] = _find_rows(
                release, "scenarios.jsonl", "scenario_id", {scenario_id}
            )[scenario_id]
        if generation_id := row.get("generation_id"):
            result["generation"] = _find_rows(
                release, "generations.jsonl", "record_id", {generation_id}
            )[generation_id]
        return result


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
