"""Bounded, pre-release inspection of verified admission draft rows and files.

This is an inspection receipt, not a rights decision. It reuses the acquired
snapshot, physical JSONL reader, record-admission verifier and versioned cleaner.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest, canonical_json
from klab.corpus.acquisition import verify_acquisition
from klab.corpus.admission_inspection_cache import (
    InspectionCache,
    SourceIndex,
    capture,
    metadata_sha256,
    unchanged,
)
from klab.corpus.bounded_metadata import (
    current_metadata_store,
    metadata_operation,
    read_metadata,
)
from klab.corpus.jsonl_records import records_from_bytes
from klab.corpus.project import Project, load_project, source_declaration_payload
from klab.corpus.rights import FILE_ADMISSION_KINDS, verify_record_admission
from klab.corpus.structure_cleaning import (
    NORMALIZER_V2,
    NORMALIZER_V3,
    clean_structure,
)
from klab.result_publication import sync_directory
from klab.training.manifest import sha256_file

_MAX_RECORD_BYTES = 8 * 1024**2
_MAX_FILE_BYTES = 8 * 1024**2
_MAX_POOL_BYTES = 64 * 1024**2


def _inside(path: Path, root: Path | None = None) -> Path:
    if (
        any(part.is_symlink() for part in (path, *path.parents))
        or not path.is_file()
        or (root is not None and not path.resolve().is_relative_to(root))
    ):
        raise ValueError("unsafe or missing admission inspection input")
    return path.resolve()


def _selection(path: Path, root: Path, source_ids: set[str]) -> dict[str, Any]:
    selected = _inside(path)
    if selected.stat().st_size > 65536:
        raise ValueError("admission inspection selection is too large")
    value = json.loads(selected.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or set(value) != {"format", "seed", "normalizer", "sources"}
        or value["format"] != "klab-admission-inspection-selection-v1"
        or not isinstance(value["seed"], str)
        or not 1 <= len(value["seed"]) <= 128
        or value["normalizer"] not in {NORMALIZER_V2, NORMALIZER_V3}
        or not isinstance(value["sources"], list)
        or len(value["sources"]) != len(source_ids)
    ):
        raise ValueError("invalid admission inspection selection")
    seen: set[str] = set()
    for source in value["sources"]:
        if (
            not isinstance(source, dict)
            or set(source) != {"source_id", "strata", "exception_count"}
            or source["source_id"] not in source_ids
            or source["source_id"] in seen
            or type(source["exception_count"]) is not int
            or not 0 <= source["exception_count"] <= 32
            or not isinstance(source["strata"], list)
            or not 1 <= len(source["strata"]) <= 16
        ):
            raise ValueError("invalid admission inspection source selection")
        seen.add(source["source_id"])
        names: set[str] = set()
        for stratum in source["strata"]:
            if (
                not isinstance(stratum, dict)
                or set(stratum)
                != {"id", "count", "path_prefix", "length_band", "issues_nonempty"}
                or not isinstance(stratum["id"], str)
                or not 1 <= len(stratum["id"]) <= 64
                or stratum["id"] in names
                or type(stratum["count"]) is not int
                or not 0 <= stratum["count"] <= 128
                or stratum["length_band"] not in {None, "short", "long"}
                or stratum["issues_nonempty"] not in {None, True, False}
                or (
                    stratum["path_prefix"] is not None
                    and (
                        not isinstance(stratum["path_prefix"], str)
                        or not stratum["path_prefix"]
                    )
                )
            ):
                raise ValueError("invalid admission inspection stratum")
            names.add(stratum["id"])
        if not 1 <= sum(group["count"] for group in source["strata"]) <= 128:
            raise ValueError("admission inspection sample count is out of bounds")
    return value


def _candidate(
    source: Any,
    entry: dict[str, Any],
    decision: dict[str, Any],
    *,
    path: str,
    raw: str,
    digest: str,
    row_index: int | None = None,
    physical_line: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = {
        "source_id": source.id,
        "path": path,
        "digest": digest,
        "row_index": row_index,
    }
    return {
        "item_id": canonical_digest(identity),
        "source_id": source.id,
        "path": path,
        "digest": digest,
        "row_index": row_index,
        "physical_line": physical_line,
        "decision": decision["decision"],
        "reason": decision["reason"],
        "issues": decision["issues"],
        "rights": entry["rights"],
        "license_label": entry["license_label"],
        "source_revision": entry["source_revision"],
        "canonical_uri": source.canonical_uri,
        "license_url": source.license_url,
        "snapshot_sha256": entry["snapshot_sha256"],
        "metadata": metadata or {},
        **(
            {"row_evidence": decision["row_evidence"]}
            if "row_evidence" in decision
            else {}
        ),
        "raw": raw,
        "raw_text_sha256": hashlib.sha256(raw.encode()).hexdigest(),
        "length_bytes": len(raw.encode("utf-8")),
    }


def _scan_candidates(
    project,
    root,
    draft,
    lock,
    max_input_bytes,
    visit,
    *,
    only_source=None,
    snapshots=None,
):
    """One native admission-validation pass; candidates come from verified rows."""
    consumed = 0
    entries = {entry["source_id"]: entry for entry in draft["sources"]}
    for source in project.sources:
        if only_source is not None and source.id != only_source:
            continue
        entry = entries[source.id]
        snapshot = Path(lock["sources"][source.id]["snapshot_path"])
        manifest = (
            snapshots[source.id]
            if snapshots is not None
            else read_metadata(snapshot / "manifest.json")
        )
        for file in manifest["files"]:
            size = (snapshot / "files" / file["path"]).stat().st_size
            if source.kind in FILE_ADMISSION_KINDS and size > _MAX_FILE_BYTES:
                raise ValueError("inspection file-byte cap exceeded")
            consumed += size
            if consumed > max_input_bytes:
                raise ValueError("admission inspection input-byte cap exceeded")

        def observed(
            source_id, name, decision, record, ordinal, source=source, entry=entry
        ):
            if source.kind in FILE_ADMISSION_KINDS:
                candidate = _candidate(
                    source,
                    entry,
                    decision,
                    path=name,
                    raw=record.decode("utf-8"),
                    digest=decision["sha256"],
                )
                offset = 0
            else:
                value = record.value
                field = source.acquisition.text_field
                if not isinstance(value.get(field), str):
                    raise TypeError(
                        f"inspection row lacks declared text field: {field}"
                    )
                ordered = entry["records"][ordinal]
                if ordered["source_row_index"] != decision["source_row_index"]:
                    raise ValueError(
                        "inspection row identity differs from draft or snapshot"
                    )
                candidate = _candidate(
                    source,
                    entry,
                    decision,
                    path=name,
                    raw=value[field],
                    digest=decision["source_row_sha256"],
                    row_index=decision["source_row_index"],
                    physical_line=record.physical_line,
                    metadata=value.get("metadata")
                    if isinstance(value.get("metadata"), dict)
                    else {},
                )
                offset = record.byte_offset
            candidate["_ordinal"], candidate["_offset"] = ordinal, offset
            visit(source_id, candidate)

        verify_record_admission(
            {**draft, "sources": [entry]},
            {source.id: source_declaration_payload(source)},
            {source.id: manifest},
            root / "corpora" / project.config.id / "snapshots",
            _visit=observed,
            _max_record_bytes=_MAX_RECORD_BYTES,
        )
    return consumed


def _rank(
    seed: str, source: str, stratum: str, row: dict[str, Any]
) -> tuple[str, str, int]:
    value = f"{seed} | {source} | {stratum} | {row['digest']}".encode()
    return hashlib.sha256(value).hexdigest(), row["path"], row["row_index"] or 0


def _clip(value: str, limit: int) -> tuple[str, bool]:
    raw = value.encode("utf-8")
    if len(raw) <= limit:
        return value, False
    return raw[:limit].decode("utf-8", errors="ignore"), True


def _display(
    row: dict[str, Any], *, normalizer: str, max_excerpt_bytes: int
) -> dict[str, Any]:
    cleaned = clean_structure(
        row["raw"], source_id=row["source_id"], normalizer=normalizer
    )
    raw_excerpt, raw_truncated = _clip(row["raw"], max_excerpt_bytes)
    clean_excerpt, clean_truncated = _clip(cleaned["text"], max_excerpt_bytes)
    structure = cleaned["structure"]
    return {
        **{
            key: value
            for key, value in row.items()
            if key != "raw" and not key.startswith("_")
        },
        "source_span": {
            "path": row["path"],
            "row_index": row["row_index"],
            "physical_line": row["physical_line"],
        },
        "raw_excerpt": raw_excerpt,
        "raw_truncated": raw_truncated,
        "cleaned_excerpt": clean_excerpt,
        "cleaned_truncated": clean_truncated,
        "cleaned_bytes": len(cleaned["text"].encode("utf-8")),
        "cleaned_text_sha256": hashlib.sha256(cleaned["text"].encode()).hexdigest(),
        "cleaning_decision": structure["decision"],
        "cleaning_flags": structure["flags"],
        "removed_spans": structure["removed_spans"][:32],
        "removed_spans_complete": len(structure["removed_spans"]) <= 32,
        "retained_spans": structure["retained_spans"][:32],
        "retained_spans_complete": len(structure["retained_spans"]) <= 32,
    }


def _fetch_candidate(source, entry, snapshot, row, stats):
    """Replay only selected offsets, checking their content identity again."""
    path = _inside(snapshot / "files" / row["path"], snapshot.resolve())
    if source.kind in FILE_ADMISSION_KINDS:
        raw = path.read_bytes()
        if len(raw) > _MAX_FILE_BYTES:
            raise ValueError("inspection file-byte cap exceeded")
        digest = hashlib.sha256(raw).hexdigest()
        value, metadata = raw.decode("utf-8"), {}
        decision = entry["files"][row["_ordinal"]]
    else:
        with path.open("rb") as stream:
            stream.seek(row["_offset"])
            raw = stream.readline(_MAX_RECORD_BYTES + 1)
        if len(raw) > _MAX_RECORD_BYTES:
            raise ValueError("JSONL record-byte cap exceeded")
        record = next(records_from_bytes(raw, source=str(path)))
        original = {
            key: value for key, value in record.value.items() if key != "_klab_source"
        }
        digest = canonical_digest(original)
        value = original[source.acquisition.text_field]
        metadata = original.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        decision = entry["records"][row["_ordinal"]]
    if digest != row["digest"]:
        raise ValueError("selected inspection record changed")
    stats["selected_records_read"] += 1
    stats["selected_bytes_read"] += len(raw)
    result = _candidate(
        source,
        entry,
        decision,
        path=row["path"],
        raw=value,
        digest=digest,
        row_index=row["row_index"],
        physical_line=row["physical_line"],
        metadata=metadata,
    )
    if row["decision"] == "qualify":
        result["retained_length_bytes"] = row["retained_length_bytes"]
    return result


def _sample_source(index, source, entry, snapshot, selection, spec, excerpt, stats):
    source_id = source.id
    median, population = index.median()
    pools = {group["id"]: [] for group in spec["strata"]}
    held = []
    counts = dict.fromkeys(pools, 0)
    outside = held_count = pool_bytes = 0
    requested = sum(group["count"] for group in spec["strata"])

    def retain(pool, name, row, count):
        nonlocal pool_bytes
        if not count:
            return
        rank = _rank(selection["seed"], source_id, name, row)
        if len(pool) == count and rank >= pool[-1][0]:
            return
        size = len(canonical_json(row))
        removed = pool[-1][2] if len(pool) == count else 0
        if pool_bytes + size - removed > _MAX_POOL_BYTES:
            raise ValueError("inspection selection-pool byte cap exceeded")
        if len(pool) == count:
            pool.pop()
        pool.append((rank, row, size))
        pool.sort(key=lambda item: item[0])
        pool_bytes += size - removed

    for row in index:
        if row["decision"] == "quarantine":
            held_count += 1
            retain(held, "quarantine", row, spec["exception_count"])
        if row["decision"] != "qualify":
            continue
        for group in spec["strata"]:
            prefix, band, issue = (
                group["path_prefix"],
                group["length_band"],
                group["issues_nonempty"],
            )
            if (
                (prefix is None or row["path"].startswith(prefix))
                and (
                    band is None
                    or (
                        row["retained_length_bytes"] * 2 <= median
                        if band == "short"
                        else row["retained_length_bytes"] * 2 > median
                    )
                )
                and (issue is None or bool(row["issues"]) == issue)
            ):
                counts[group["id"]] += 1
                if group["count"]:
                    retain(pools[group["id"]], group["id"], row, requested)
                else:
                    outside += 1
                break
        else:
            outside += 1
    selected, remaining, coverage = [], [], []
    sample_population = 0
    for group in spec["strata"]:
        name, count = group["id"], group["count"]
        pool = pools[name]
        selected.extend((name, row) for _, row, _ in pool[:count])
        if count:
            sample_population += counts[name]
            remaining.extend((rank, name, row) for rank, row, _ in pool[count:])
        coverage.append(
            {
                "source_id": source_id,
                "stratum": name,
                "population": counts[name],
                "requested": count,
                "initial_selected": min(count, counts[name]),
            }
        )
    remaining.sort(key=lambda item: item[0])
    selected.extend(
        (name, row) for _, name, row in remaining[: requested - len(selected)]
    )
    if len(selected) != min(requested, sample_population):
        raise ValueError("inspection selection is incomplete")

    display_bytes = 0

    def display(row):
        nonlocal display_bytes
        result = _display(
            _fetch_candidate(source, entry, snapshot, row, stats),
            normalizer=selection["normalizer"],
            max_excerpt_bytes=excerpt,
        )
        display_bytes += len(canonical_json(result))
        if display_bytes > _MAX_POOL_BYTES:
            raise ValueError("inspection selection-pool byte cap exceeded")
        return result

    items = [{**display(row), "stratum": name} for name, row in selected]
    exceptions = [display(row) for _, row, _ in held]
    coverage.append(
        {
            "source_id": source_id,
            "stratum": "__summary__",
            "qualify_population": population,
            "outside_protocol_population": outside,
            "requested": requested,
            "selected": len(selected),
            "sample_shortfall": requested - len(selected),
            "quarantine_population": held_count,
            "exceptions_shown": len(held),
            "exceptions_incomplete": held_count > len(held),
        }
    )
    return {"items": items, "quarantine_exceptions": exceptions, "coverage": coverage}


def _inspect_stored(
    project,
    project_path,
    root,
    draft_path,
    policy,
    selection_path,
    selection,
    cache,
    stats,
    dependencies,
    **limits,
):
    draft = read_metadata(draft_path)
    if draft.get("schema_version") != 2 or draft.get("policy_sha256") != sha256_file(
        policy
    ):
        raise ValueError("inspection draft or policy identity mismatch")
    options = {
        "proof_store": cache.proofs,
        "verification_mode": "verified_reuse" if cache.proofs is not None else "cold",
    }
    before_proofs = cache.proofs.diagnostics() if cache.proofs is not None else None
    lock = verify_acquisition(project, root, **options)
    if before_proofs is not None:
        after_proofs = cache.proofs.diagnostics()
        stats["snapshot_proof_hits"] += after_proofs["hits"] - before_proofs["hits"]
        stats["snapshot_cold_checks"] += (
            after_proofs["misses"] - before_proofs["misses"]
        )
    else:
        stats["snapshot_cold_checks"] += len(project.sources)
    snapshots = {
        source.id: read_metadata(
            Path(lock["sources"][source.id]["snapshot_path"]) / "manifest.json"
        )
        for source in project.sources
    }
    entries = {entry["source_id"]: entry for entry in draft["sources"]}
    if len(entries) != len(draft["sources"]) or set(entries) != {
        s.id for s in project.sources
    }:
        raise ValueError("record admission source inventory mismatch")
    specs = {item["source_id"]: item for item in selection["sources"]}
    results = {}
    output_bytes = 0
    input_bytes = 0
    for source in project.sources:
        entry = entries[source.id]
        snapshot = Path(lock["sources"][source.id]["snapshot_path"])
        source_dependencies = capture([snapshot])
        dependencies.update(source_dependencies)
        request = {
            "source": source_declaration_payload(source),
            "admission": metadata_sha256(
                {
                    "policy_id": draft["policy_id"],
                    "policy_sha256": draft["policy_sha256"],
                    "entry": entry,
                }
            ),
            "snapshot": lock["sources"][source.id]["snapshot_sha256"],
            "selection": specs[source.id],
            "seed": selection["seed"],
            "normalizer": selection["normalizer"],
            "excerpt": limits["max_excerpt_bytes"],
            "record_cap": _MAX_RECORD_BYTES,
            "file_cap": _MAX_FILE_BYTES,
            "pool_cap": _MAX_POOL_BYTES,
        }
        hit = cache.load("source", request, limit=_MAX_POOL_BYTES + 65536)
        if hit is not None:
            result, signed_dependencies = hit
            dependencies.update(signed_dependencies)
            stats["source_cache_hits"] += 1
        else:
            index = SourceIndex()

            def collect(source_id, row, index=index):
                length = (
                    len(
                        clean_structure(
                            row["raw"],
                            source_id=source_id,
                            normalizer=selection["normalizer"],
                        )["text"].encode("utf-8")
                    )
                    if row["decision"] == "qualify"
                    else None
                )
                index.add(row, length)

            consumed = _scan_candidates(
                project,
                root,
                draft,
                lock,
                limits["max_input_bytes"],
                collect,
                only_source=source.id,
                snapshots=snapshots,
            )
            stats["full_source_passes"] += 1
            stats["records_validated"] += index.rows
            before = stats["selected_bytes_read"]
            result = _sample_source(
                index,
                source,
                entry,
                snapshot,
                selection,
                specs[source.id],
                limits["max_excerpt_bytes"],
                stats,
            )
            result.update(
                input_bytes=consumed,
                read_bytes=consumed + stats["selected_bytes_read"] - before,
            )
            cache.publish(
                "source",
                request,
                result,
                source_dependencies,
                limit=_MAX_POOL_BYTES + 65536,
            )
        results[source.id] = result
        input_bytes += result["input_bytes"]
        if input_bytes > limits["max_input_bytes"]:
            raise ValueError("admission inspection input-byte cap exceeded")
        output_bytes += sum(
            len(canonical_json(result[key]))
            for key in ("items", "quarantine_exceptions", "coverage")
        )
        if output_bytes > limits["max_output_bytes"]:
            raise ValueError("admission inspection output-byte cap exceeded")
    items, exceptions, coverage = [], [], []
    for spec in selection["sources"]:
        result = results[spec["source_id"]]
        items.extend(result["items"])
        exceptions.extend(result["quarantine_exceptions"])
        coverage.extend(result["coverage"])
    payload = {
        "format": "klab-admission-inspection-v1",
        "project_id": project.config.id,
        "project_path": str(_inside(project_path)),
        "project_sha256": lock["project_sha256"],
        "acquisition_lock_sha256": sha256_file(
            root / "corpora" / project.config.id / "acquisition.json"
        ),
        "draft_path": str(draft_path.resolve()),
        "draft_sha256": sha256_file(draft_path),
        "policy_path": str(policy),
        "policy_sha256": sha256_file(policy),
        "selection_path": str(selection_path.resolve()),
        "selection_sha256": sha256_file(selection_path),
        "normalizer": selection["normalizer"],
        "normalizer_source_sha256": sha256_file(
            Path(__file__).with_name("structure_cleaning.py")
        ),
        "snapshot_sha256": {
            s.id: lock["sources"][s.id]["snapshot_sha256"] for s in project.sources
        },
        **limits,
        "input_bytes": sum(v["input_bytes"] for v in results.values()),
        "read_bytes": sum(v["read_bytes"] for v in results.values()),
        "items": items,
        "quarantine_exceptions": exceptions,
        "coverage": coverage,
    }
    if len(canonical_json(payload)) + 1 > limits["max_output_bytes"]:
        raise ValueError("admission inspection output-byte cap exceeded")
    return payload


def inspect_admission(
    project: Project,
    project_path: Path,
    work_root: Path,
    draft: Path,
    policy_document: Path,
    selection: Path,
    output: Path,
    *,
    max_input_bytes: int,
    max_excerpt_bytes: int,
    max_output_bytes: int,
) -> dict[str, Any]:
    root = work_root.resolve(strict=True)
    if any(
        part.is_symlink() for part in (output, *output.parents)
    ) or not output.resolve().is_relative_to(root):
        raise ValueError("inspection requires safe output within work root")
    if output.exists() and (
        not output.is_file() or output.stat().st_size > 4 * 1024**2
    ):
        raise ValueError("invalid existing inspection output")
    work_stats = {}
    payload = _inspect(
        project,
        project_path,
        root,
        draft,
        policy_document,
        selection,
        max_input_bytes=max_input_bytes,
        max_excerpt_bytes=max_excerpt_bytes,
        max_output_bytes=max_output_bytes,
        _stats=work_stats,
    )
    if not output.exists():
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=".inspection-", dir=output.parent, delete=False
        ) as stream:
            temporary = Path(stream.name)
            try:
                stream.write(canonical_json(payload) + b"\n")
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        try:
            os.link(temporary, output)
            sync_directory(output.parent)
        finally:
            temporary.unlink(missing_ok=True)
    _inside(output, root)
    output_guard = capture([output])
    with output.open("rb") as stream:
        raw = stream.read(4 * 1024**2 + 1)
    unchanged(output_guard)
    if len(raw) > 4 * 1024**2:
        raise ValueError("invalid existing inspection output")
    try:
        observed = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("invalid/incomplete existing inspection output") from error
    if observed != payload:
        raise ValueError("existing inspection output differs from verified inputs")
    return {
        "path": str(output),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "items": len(payload["items"]),
        "exceptions": len(payload["quarantine_exceptions"]),
        "input_bytes": payload["input_bytes"],
        "read_bytes": payload["read_bytes"],
        "work": work_stats,
    }


def verify_admission_inspection(path: Path, work_root: Path) -> dict[str, Any]:
    root = work_root.resolve(strict=True)
    verified_path = _inside(path, root)
    if verified_path.stat().st_size > 4 * 1024**2:
        raise ValueError("admission inspection receipt is too large")
    observed = json.loads(verified_path.read_text(encoding="utf-8"))
    if (
        not isinstance(observed, dict)
        or observed.get("format") != "klab-admission-inspection-v1"
    ):
        raise ValueError("invalid admission inspection receipt")
    project_path = Path(observed["project_path"])
    project = load_project(project_path)
    expected = _inspect(
        project,
        project_path,
        root,
        Path(observed["draft_path"]),
        Path(observed["policy_path"]),
        Path(observed["selection_path"]),
        max_input_bytes=observed["max_input_bytes"],
        max_excerpt_bytes=observed["max_excerpt_bytes"],
        max_output_bytes=observed["max_output_bytes"],
    )
    if observed != expected:
        raise ValueError("admission inspection receipt differs from verified inputs")
    return observed


def _inspect(
    project,
    project_path,
    work_root,
    draft_path,
    policy_document,
    selection_path,
    *,
    _stats=None,
    **limits,
):
    root = Path(work_root).resolve(strict=True)
    for value in limits.values():
        if type(value) is not int or value <= 0:
            raise ValueError("inspection caps must be positive integers")
    if limits["max_excerpt_bytes"] > 4096 or limits["max_output_bytes"] > 4 * 1024**2:
        raise ValueError("inspection cap exceeds native maximum")
    if (
        sum(s.acquisition.max_bytes for s in project.sources)
        > limits["max_input_bytes"]
    ):
        raise ValueError(
            "admission inspection input-byte cap is below declared source maxima"
        )
    draft_path = _inside(Path(draft_path), root)
    project_path = _inside(Path(project_path))
    policy = _inside(Path(policy_document))
    if policy.stat().st_size > 1024**2:
        raise ValueError("admission inspection policy is too large")
    selection_path = _inside(Path(selection_path))
    selection = _selection(selection_path, root, {s.id for s in project.sources})
    request = {
        "paths": [str(p) for p in (project_path, draft_path, policy, selection_path)],
        "project": project.config.id,
        "sources": [source_declaration_payload(s) for s in project.sources],
        "project_file": sha256_file(project_path),
        "policy": sha256_file(policy),
        "selection": sha256_file(selection_path),
        "limits": limits,
        "record_cap": _MAX_RECORD_BYTES,
        "file_cap": _MAX_FILE_BYTES,
        "pool_cap": _MAX_POOL_BYTES,
    }
    stats = _stats if _stats is not None else {}
    stats.update(
        operation_cache_hits=0,
        source_cache_hits=0,
        full_source_passes=0,
        records_validated=0,
        selected_records_read=0,
        selected_bytes_read=0,
        snapshot_proof_hits=0,
        snapshot_cold_checks=0,
    )
    cache = InspectionCache(root)
    stats.update(
        cache_read_available=cache.proofs is not None,
        cache_write_available=cache.writable,
        cache_unavailable_reason=cache.unavailable_reason,
    )
    try:
        active_store = current_metadata_store()
    except RuntimeError:
        active_store = None
    # An explicit caller-selected metadata cap still applies to its input read.
    hit = (
        cache.load("operation", request, limit=limits["max_output_bytes"] + 65536)
        if active_store is None or active_store.max_database_bytes is None
        else None
    )
    if hit is not None:
        payload, _ = hit
        stats["operation_cache_hits"] = 1
        stats["source_cache_hits"] = len(project.sources)
        return payload
    dependencies = capture(
        [
            project_path,
            draft_path,
            policy,
            selection_path,
            root / "corpora" / project.config.id / "acquisition.json",
        ]
    )
    with metadata_operation(
        root / "corpora" / project.config.id / "metadata-scratch",
        max_database_bytes=None,
    ):
        payload = _inspect_stored(
            project,
            project_path,
            root,
            draft_path,
            policy,
            selection_path,
            selection,
            cache,
            stats,
            dependencies,
            **limits,
        )
    unchanged(dependencies)
    cache.publish(
        "operation",
        request,
        payload,
        dependencies,
        limit=limits["max_output_bytes"] + 65536,
    )
    stats.update(
        cache_read_available=cache.proofs is not None,
        cache_write_available=cache.writable,
        cache_unavailable_reason=cache.unavailable_reason,
    )
    return payload
