"""Receipted text statistics over one published normalized partition."""

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

_DIGEST = re.compile(r"[0-9a-f]{64}")
_METADATA_CAP = 4 * 1024**2
_RECORD_CAP = 64 * 1024**2
_ANNOTATOR = "text-statistics"
_FIELDS = ("document_id", "utf8_bytes", "codepoints", "lines")
_ROW = struct.Struct("<32sIII")


def _parent(work_root, partition):
    source, key = parse_normalized_partition_id(partition)
    directory = normalized_store_root(work_root) / source / key
    receipt_path = directory / "receipt.json"
    docs = directory / "docs.jsonl"
    if not all(path.is_file() for path in (receipt_path, docs)):
        raise ValueError(f"missing normalized partition: {partition}")
    dependencies = capture([receipt_path, docs])
    with receipt_path.open("rb") as stream:
        raw = stream.read(_METADATA_CAP + 1)
    if len(raw) > _METADATA_CAP:
        raise ValueError("normalized partition receipt exceeds its byte limit")
    try:
        receipt = json.loads(raw)
        identity = receipt["identity"]
        entry = receipt["files"][docs.name]
        count = receipt["counts"]["documents"]
        size, digest = entry["size"], entry["sha256"]
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError("invalid normalized partition receipt") from exc
    if (
        receipt.get("schema_version") != 1
        or receipt.get("key") != key
        or canonical_digest(identity) != key
        or not isinstance(identity, dict)
        or identity.get("artifact_type") != "normalized_source_partition"
        or identity.get("source_id") != source
        or type(count) is not int
        or count < 0
        or type(size) is not int
        or size != docs.stat().st_size
        or not isinstance(digest, str)
        or not _DIGEST.fullmatch(digest)
    ):
        raise ValueError("normalized partition receipt disagrees with its documents")
    return (
        docs,
        count,
        dependencies,
        {
            "partition": partition,
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "documents_sha256": digest,
            "documents_bytes": size,
        },
    )


def _render(output, docs, parent, expected):
    totals = dict.fromkeys(_FIELDS[1:], 0)
    count = 0
    with (
        output.open("xb") as stream,
        progress_phase(
            "corpus_text_statistics", total_work=expected, unit="documents"
        ) as progress,
    ):
        writer = HashingWriter(stream)
        for count, record in enumerate(
            records_from_path(
                docs,
                max_record_bytes=_RECORD_CAP,
                expected_sha256=parent["documents_sha256"],
            ),
            1,
        ):
            row = record.value
            doc_id, text = (
                (row.get("document_id"), row.get("text"))
                if isinstance(row, dict)
                else (None, None)
            )
            if (
                count > expected
                or not isinstance(doc_id, str)
                or not _DIGEST.fullmatch(doc_id)
                or not isinstance(text, str)
            ):
                raise ValueError("invalid document in normalized partition")
            utf8 = len(text.encode("utf-8"))
            points = len(text)
            lines = text.count("\n") + bool(text and not text.endswith("\n"))
            writer.write(_ROW.pack(bytes.fromhex(doc_id), utf8, points, lines))
            totals["utf8_bytes"] += utf8
            totals["codepoints"] += points
            totals["lines"] += lines
            if count % 8192 == 0:
                progress.update(completed_work=count, emit=False)
        if count != expected:
            raise ValueError(
                "normalized partition document count differs from its receipt"
            )
        progress.update(completed_work=count)
        proof = writer.seal(output)
    return {
        "path": str(proof.path),
        "sha256": proof.sha256,
        "bytes": proof.size_bytes,
    }, totals


def _publish(directory, pending, docs, parent, count, dependencies):
    if symlinked(pending):
        raise ValueError("unsafe text statistics staging file")
    pending.unlink(missing_ok=True)
    require_storage([check_storage(directory, projected_bytes=count * _ROW.size)])
    try:
        artifact, totals = _render(pending, docs, parent, count)
        unchanged(dependencies)
        output = directory / "text-statistics.bin"
        rename_noreplace(pending, output)
        sync_directory(directory)
        return {
            "documents": count,
            "totals": totals,
            "files": [{**artifact, "path": str(output)}],
        }
    finally:
        pending.unlink(missing_ok=True)


def _page(path, *, count, offset, limit):
    end = min(count, offset + limit)
    data = b""
    if offset < end:
        with path.open("rb") as stream:
            stream.seek(offset * _ROW.size)
            data = stream.read((end - offset) * _ROW.size)
        if len(data) != (end - offset) * _ROW.size:
            raise ValueError("text statistics page was truncated")
    return {
        "rows": [
            dict(zip(_FIELDS, (doc_id.hex(), *values), strict=True))
            for doc_id, *values in _ROW.iter_unpack(data)
        ],
        "next_offset": end if limit and end < count else None,
    }


def annotate_normalized(work_root: Path, partition: str, *, offset=0, limit=0) -> dict:
    """Produce or reuse aligned typed rows, then return a bounded optional page."""
    if offset < 0 or not 0 <= limit <= 1000:
        raise ValueError(
            "text statistics need a nonnegative offset and a limit from 0 to 1000"
        )
    with normalized_retention_lease(work_root):
        docs, count, dependencies, parent = _parent(work_root, partition)
        request = {
            "artifact_type": _ANNOTATOR,
            "schema_version": 1,
            "producer": implementation_digests(
                "text_statistics.py", "jsonl_records.py"
            ),
            "input": parent,
        }
        key = canonical_digest(request)
        root = (
            work_root / "cache" / "corpus" / "annotations" / _ANNOTATOR / key[:2] / key
        )
        with shared_payload_lease(root):
            stages = SplitStages(
                root,
                {**request, "inputs": dependencies},
                work_root=work_root,
                producer_scoped=True,
            )
            result = stages.phase(
                _ANNOTATOR,
                lambda directory: _publish(
                    directory,
                    stages.root / ".pending.bin",
                    docs,
                    parent,
                    count,
                    dependencies,
                ),
            )
        payload = result["files"][0]
        path = Path(payload["path"])
        if (
            result["documents"] != count
            or payload["bytes"] != count * _ROW.size
            or checkpoint_stamp(path) != payload["stamp"]
        ):
            raise ValueError("text statistics payload disagrees with its completion")
        page = _page(path, count=count, offset=offset, limit=limit)
        unchanged(dependencies)
        if checkpoint_stamp(path) != payload["stamp"]:
            raise ValueError("text statistics changed during inspection")
    return {
        **request,
        "computation_key": key,
        "documents": count,
        "totals": result["totals"],
        "payload": {field: payload[field] for field in ("path", "bytes", "sha256")},
        "reused": bool(stages.reused),
        "reuse_unavailable_reason": stages.cache.unavailable_reason,
        **page,
    }
