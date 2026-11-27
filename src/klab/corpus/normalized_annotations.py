"""Build and page typed sidecars over verified normalized partitions."""

from __future__ import annotations

import hashlib
import json
import re
import struct
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
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


@dataclass(frozen=True)
class NormalizedAnnotator:
    """An installed fixed-width producer and its bounded page decoder."""

    kind: str
    implementation: str
    row: struct.Struct
    render: Callable[[HashingWriter, Iterator[tuple[bytes, dict]], dict], dict]
    decode: Callable[[tuple, dict], dict]
    options: dict | None = None
    prepare: Callable[[Iterator[tuple[bytes, dict]], dict], dict] | None = None


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
        identity.get("source", {}),
        {
            "partition": partition,
            "receipt_sha256": hashlib.sha256(raw).hexdigest(),
            "documents_sha256": digest,
            "documents_bytes": size,
        },
    )


def _render(output, docs, parent, expected, source, annotator):
    profile_rows = expected if annotator.prepare is not None else 0
    with (
        output.open("xb") as stream,
        progress_phase(
            "corpus_" + annotator.kind.replace("-", "_"),
            total_work=expected + profile_rows,
            unit="documents",
        ) as progress,
    ):
        writer = HashingWriter(stream)
        count = 0

        def documents(*, output_rows, completed_before=0):
            nonlocal count
            with closing(
                records_from_path(
                    docs,
                    max_record_bytes=_RECORD_CAP,
                    expected_sha256=parent["documents_sha256"],
                )
            ) as records:
                seen = 0
                for seen, record in enumerate(records, 1):
                    row = record.value
                    doc_id = row.get("document_id") if isinstance(row, dict) else None
                    if (
                        seen > expected
                        or not isinstance(doc_id, str)
                        or not _DIGEST.fullmatch(doc_id)
                    ):
                        raise ValueError("invalid document in normalized partition")
                    if output_rows:
                        count = seen
                    if seen % 8192 == 0:
                        progress.update(
                            completed_work=completed_before + seen, emit=False
                        )
                    yield bytes.fromhex(doc_id), row
                if seen != expected:
                    raise ValueError(
                        "normalized partition document count differs from its receipt"
                    )

        if annotator.prepare is not None:
            with closing(documents(output_rows=False)) as rows:
                source = annotator.prepare(rows, source)
                if next(rows, None) is not None:
                    raise ValueError(
                        "annotation preparation stopped before the end of the partition"
                    )
        with closing(
            documents(output_rows=True, completed_before=profile_rows)
        ) as rows:
            summary = annotator.render(writer, rows, source)
            if next(rows, None) is not None:
                raise ValueError(
                    "annotation producer stopped before the end of the partition"
                )
        progress.update(completed_work=count + profile_rows)
        proof = writer.seal(output)
        if proof.size_bytes != count * annotator.row.size:
            raise ValueError("annotation producer wrote an unexpected payload size")
    return {
        "path": str(proof.path),
        "sha256": proof.sha256,
        "bytes": proof.size_bytes,
    }, summary


def _publish(directory, pending, docs, parent, count, source, dependencies, annotator):
    if symlinked(pending):
        raise ValueError("unsafe annotation staging file")
    pending.unlink(missing_ok=True)
    require_storage(
        [check_storage(directory, projected_bytes=count * annotator.row.size)]
    )
    try:
        artifact, summary = _render(pending, docs, parent, count, source, annotator)
        unchanged(dependencies)
        output = directory / f"{annotator.kind}.bin"
        rename_noreplace(pending, output)
        sync_directory(directory)
        return {
            "documents": count,
            "summary": summary,
            "files": [{**artifact, "path": str(output)}],
        }
    finally:
        pending.unlink(missing_ok=True)


def _page(path, *, count, offset, limit, annotator, summary):
    end = min(count, offset + limit)
    data = b""
    if offset < end:
        width = annotator.row.size
        with path.open("rb") as stream:
            stream.seek(offset * width)
            data = stream.read((end - offset) * width)
        if len(data) != (end - offset) * width:
            raise ValueError("annotation page was truncated")
    return {
        "rows": [
            annotator.decode(values, summary)
            for values in annotator.row.iter_unpack(data)
        ],
        "next_offset": end if limit and end < count else None,
    }


def annotate_partition(
    work_root: Path,
    partition: str,
    annotator: NormalizedAnnotator,
    *,
    offset=0,
    limit=0,
) -> dict:
    """Produce or reuse aligned typed rows, then return a bounded optional page."""
    if offset < 0 or not 0 <= limit <= 1000:
        raise ValueError(
            "annotation needs a nonnegative offset and a limit from 0 to 1000"
        )
    with normalized_retention_lease(work_root):
        docs, count, dependencies, source, parent = _parent(work_root, partition)
        request = {
            "artifact_type": annotator.kind,
            "schema_version": 1,
            "producer": implementation_digests(
                "normalized_annotations.py",
                annotator.implementation,
                "jsonl_records.py",
            ),
            "input": parent,
            **({"options": annotator.options} if annotator.options is not None else {}),
        }
        key = canonical_digest(request)
        root = (
            work_root
            / "cache"
            / "corpus"
            / "annotations"
            / annotator.kind
            / key[:2]
            / key
        )
        with shared_payload_lease(root):
            stages = SplitStages(
                root,
                {**request, "inputs": dependencies},
                work_root=work_root,
                producer_scoped=True,
            )
            result = stages.phase(
                annotator.kind,
                lambda directory: _publish(
                    directory,
                    stages.root / ".pending.bin",
                    docs,
                    parent,
                    count,
                    source,
                    dependencies,
                    annotator,
                ),
            )
        payload = result["files"][0]
        path = Path(payload["path"])
        if (
            result["documents"] != count
            or payload["bytes"] != count * annotator.row.size
            or checkpoint_stamp(path) != payload["stamp"]
        ):
            raise ValueError("annotation payload disagrees with its completion")
        page = _page(
            path,
            count=count,
            offset=offset,
            limit=limit,
            annotator=annotator,
            summary=result["summary"],
        )
        unchanged(dependencies)
        if checkpoint_stamp(path) != payload["stamp"]:
            raise ValueError("annotation changed during inspection")
    return {
        **request,
        "computation_key": key,
        "documents": count,
        **result["summary"],
        "payload": {field: payload[field] for field in ("path", "bytes", "sha256")},
        "reused": bool(stages.reused),
        "reuse_unavailable_reason": stages.cache.unavailable_reason,
        **page,
    }
