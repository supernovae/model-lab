"""Sparse, typed evidence from a normalized partition's recorded rejections."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.admission_inspection_cache import capture, unchanged
from klab.corpus.artifact_inventory import parse_normalized_partition_id
from klab.corpus.build_stages import implementation_digests
from klab.corpus.jsonl_records import records_from_path
from klab.corpus.prepared_shards import normalized_store_root
from klab.corpus.release_members import normalized_retention_lease, shared_payload_lease
from klab.corpus.split_runs import SplitStages, checkpoint_stamp
from klab.data.verification import HashingWriter
from klab.progress import progress_phase
from klab.result_publication import sync_directory
from klab.safe_io import rename_noreplace, symlinked
from klab.workspace_preflight import check_storage, require_storage

_KIND = "privacy-exclusions"
_ROW = struct.Struct("<QQB")
_REASONS = ("secret_or_private_identifier", "excluded_platform_terms_or_private_paste")
_CODES = {reason: code for code, reason in enumerate(_REASONS, 1)}
_DIGEST = re.compile(r"[0-9a-f]{64}")


def _rejections(work_root, partition):
    source, key = parse_normalized_partition_id(partition)
    directory = normalized_store_root(work_root) / source / key
    receipt_path = directory / "receipt.json"
    path = directory / "rejected.jsonl"
    if not receipt_path.is_file() or not path.is_file():
        raise ValueError(f"missing normalized rejection inventory: {partition}")
    dependencies = capture([receipt_path, path])
    with receipt_path.open("rb") as stream:
        raw = stream.read(4 * 1024**2 + 1)
    if len(raw) > 4 * 1024**2:
        raise ValueError("normalized partition receipt exceeds its byte limit")
    try:
        receipt = json.loads(raw)
        identity = receipt["identity"]
        member = receipt["files"][path.name]
        count = receipt["counts"]["rejected"]
        policy = identity["document_policy"]["exclusions"]
        raw_path = identity["path"]
        size, digest = member["size"], member["sha256"]
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("invalid normalized rejection receipt") from exc
    if (
        receipt.get("schema_version") != 1
        or receipt.get("key") != key
        or canonical_digest(identity) != key
        or identity.get("artifact_type") != "normalized_source_partition"
        or identity.get("source_id") != source
        or not isinstance(raw_path, str)
        or type(count) is not int
        or count < 0
        or type(size) is not int
        or path.stat().st_size != size
        or not isinstance(digest, str)
        or not _DIGEST.fullmatch(digest)
        or policy not in ("none", "research_privacy")
    ):
        raise ValueError(
            "normalized partition receipt disagrees with its rejection inventory"
        )
    return (
        path,
        dependencies,
        source,
        raw_path,
        {
            "partition": partition,
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "rejected_sha256": digest,
            "rejected_bytes": size,
            "rejected_records": count,
            "exclusions": policy,
        },
    )


def _write(directory, pending, path, inputs, source, raw_path, dependencies):
    if symlinked(pending):
        raise ValueError("unsafe privacy evidence staging file")
    pending.unlink(missing_ok=True)
    expected = inputs["rejected_records"]
    enabled = inputs["exclusions"] == "research_privacy"
    require_storage(
        [
            check_storage(
                directory, projected_bytes=expected * _ROW.size if enabled else 0
            )
        ]
    )
    counts = dict.fromkeys(_REASONS, 0)
    ordinal = 0
    try:
        with (
            pending.open("xb") as stream,
            progress_phase(
                "corpus_privacy_exclusions", total_work=expected, unit="rejections"
            ) as progress,
        ):
            writer = HashingWriter(stream)
            for ordinal, record in enumerate(
                records_from_path(
                    path,
                    max_record_bytes=64 * 1024**2,
                    expected_sha256=inputs["rejected_sha256"],
                ),
                1,
            ):
                row = record.value
                if (
                    ordinal > expected
                    or not isinstance(row, dict)
                    or row.get("source_id") != source
                    or row.get("path") != raw_path
                    or not isinstance(row.get("reason"), str)
                ):
                    raise ValueError("rejection differs from its normalized partition")
                reason = row["reason"]
                if code := _CODES.get(reason):
                    position = row["row"]
                    if not enabled or (
                        position is not None
                        and (type(position) is not int or not 1 <= position < 2**64)
                    ):
                        raise ValueError("invalid recorded privacy exclusion")
                    writer.write(_ROW.pack(ordinal, position or 0, code))
                    counts[reason] += 1
                if ordinal % 8192 == 0:
                    progress.update(completed_work=ordinal, emit=False)
            if ordinal != expected:
                raise ValueError("rejection count differs from the normalized receipt")
            progress.update(completed_work=ordinal)
            proof = writer.seal(pending)
        unchanged(dependencies)
        output = directory / f"{_KIND}.bin"
        rename_noreplace(pending, output)
        sync_directory(directory)
        return {
            "totals": counts,
            "files": [
                {"path": str(output), "sha256": proof.sha256, "bytes": proof.size_bytes}
            ],
        }
    finally:
        pending.unlink(missing_ok=True)


def _page(path, count, offset, limit):
    if not limit or offset >= count:
        return {"rows": [], "next_offset": None}
    end = min(count, offset + limit)
    with path.open("rb") as stream:
        stream.seek(offset * _ROW.size)
        data = stream.read((end - offset) * _ROW.size)
    if len(data) != (end - offset) * _ROW.size:
        raise ValueError("privacy exclusion page was truncated")
    rows = []
    for ordinal, row, code in _ROW.iter_unpack(data):
        if not 1 <= code <= len(_REASONS):
            raise ValueError("invalid privacy exclusion reason code")
        rows.append(
            {
                "rejection_ordinal": ordinal,
                "snapshot_row": row or None,
                "reason": _REASONS[code - 1],
            }
        )
    return {
        "rows": rows,
        "next_offset": end if end < count else None,
    }


def annotate_privacy_exclusions(
    work_root: Path, partition: str, *, offset=0, limit=0
) -> dict:
    """Publish or page existing privacy-related exclusions, without source bodies."""
    if offset < 0 or not 0 <= limit <= 1000:
        raise ValueError(
            "annotation needs a nonnegative offset and a limit from 0 to 1000"
        )
    with normalized_retention_lease(work_root):
        path, deps, source, raw_path, parent = _rejections(work_root, partition)
        request = {
            "artifact_type": _KIND,
            "schema_version": 1,
            "producer": implementation_digests(
                "privacy_exclusions.py", "jsonl_records.py"
            ),
            "input": parent,
        }
        key = canonical_digest(request)
        root = work_root / "cache" / "corpus" / "annotations" / _KIND / key[:2] / key
        with shared_payload_lease(root):
            stages = SplitStages(
                root,
                {**request, "inputs": deps},
                work_root=work_root,
                producer_scoped=True,
            )
            result = stages.phase(
                _KIND,
                lambda directory: _write(
                    directory,
                    stages.root / ".pending.bin",
                    path,
                    parent,
                    source,
                    raw_path,
                    deps,
                ),
            )
        payload = result["files"][0]
        count = sum(result["totals"].values())
        output = Path(payload["path"])
        if (
            payload["bytes"] != count * _ROW.size
            or checkpoint_stamp(output) != payload["stamp"]
        ):
            raise ValueError("privacy exclusion payload disagrees with its completion")
        page = _page(output, count, offset, limit)
        unchanged(deps)
        if checkpoint_stamp(output) != payload["stamp"]:
            raise ValueError("privacy exclusion payload changed during inspection")
    return {
        **request,
        "computation_key": key,
        "findings": count,
        "totals": result["totals"],
        "payload": {field: payload[field] for field in ("path", "bytes", "sha256")},
        "reused": bool(stages.reused),
        "reuse_unavailable_reason": stages.cache.unavailable_reason,
        **page,
    }
