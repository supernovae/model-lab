"""A release span stream assembled from immutable native source partitions."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from klab.canonical import canonical_digest
from klab.corpus.project import safe_name
from klab.corpus.shards import range_file_identity
from klab.safe_io import symlinked

SPAN_MEMBER = "spans.jsonl"
NORMALIZED_SPANS = "normalized_spans"
_SOURCE = re.compile(r"[a-z][a-z0-9_-]*")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_SEGMENT_FIELDS = {
    "source_id",
    "path",
    "partition_key",
    "receipt_sha256",
    "size",
}


def capture_span_partitions(
    prepared: list[Path], raw_paths: list[str], receipts: list[dict[str, str]]
) -> list[dict[str, Any]]:
    """Inventory the span partitions in the native build's acquisition order."""
    return [
        {
            "source_id": receipt["source"],
            "path": raw_path,
            "partition_key": directory.name,
            "receipt_sha256": receipt["receipt_sha256"],
            "size": (directory / SPAN_MEMBER).stat().st_size,
        }
        for directory, raw_path, receipt in zip(
            prepared, raw_paths, receipts, strict=True
        )
    ]


@dataclass(frozen=True)
class SpanPartition:
    """A logical source file and the normalized evidence it contributes."""

    directory: Path
    entry: dict[str, Any]

    @property
    def source_file(self) -> tuple[str, str]:
        return self.entry["source_id"], self.entry["path"]

    @property
    def spans(self) -> Path:
        return self.directory / SPAN_MEMBER

    @property
    def receipt(self) -> Path:
        return self.directory / "receipt.json"


def span_partitions(root: Path, files: dict[str, Any]) -> tuple[SpanPartition, ...]:
    """Resolve an ordered span projection through the normalized work area."""
    entry = files.get(SPAN_MEMBER, {})
    if entry.get("storage") != NORMALIZED_SPANS:
        return ()
    from klab.corpus.prepared_shards import normalized_store_root

    if (
        set(entry) != {"storage", "sha256", "size", "partitions"}
        or not isinstance(entry["sha256"], str)
        or not _DIGEST.fullmatch(entry["sha256"])
        or type(entry["size"]) is not int
        or not isinstance(entry["partitions"], list)
        or not entry["partitions"]
        or len(root.parents) < 4
        or root.parent.name != "releases"
        or root.parents[2].name != "corpora"
    ):
        raise ValueError("invalid normalized span projection")
    store = normalized_store_root(root.parents[3])
    if symlinked(store):
        raise ValueError("symlinked normalized span store")
    parts = []
    seen_files = set()
    seen_partitions = set()
    size = 0
    for part in entry["partitions"]:
        if (
            not isinstance(part, dict)
            or set(part) != _SEGMENT_FIELDS
            or not isinstance(part["source_id"], str)
            or not _SOURCE.fullmatch(part["source_id"])
            or not isinstance(part["path"], str)
            or any(
                not isinstance(part[name], str) or not _DIGEST.fullmatch(part[name])
                for name in ("partition_key", "receipt_sha256")
            )
            or type(part["size"]) is not int
            or part["size"] < 0
        ):
            raise ValueError("invalid normalized span partition")
        source_file = part["source_id"], safe_name(part["path"])
        partition = part["source_id"], part["partition_key"]
        if source_file in seen_files or partition in seen_partitions:
            raise ValueError("duplicate normalized span partition")
        seen_files.add(source_file)
        seen_partitions.add(partition)
        size += part["size"]
        parts.append(SpanPartition(store.joinpath(*partition), part))
    if size != entry["size"]:
        raise ValueError("normalized span projection size differs from its partitions")
    return tuple(parts)


def span_dependencies(parts: tuple[SpanPartition, ...]) -> list[Path]:
    """Inspect exact backing paths before using them as verification dependencies."""
    dependencies = []
    for part in parts:
        for path in (part.receipt, part.spans):
            if symlinked(path) or not path.is_file() or path.stat().st_nlink != 1:
                raise ValueError(f"missing or unsafe normalized span input: {path}")
            dependencies.append(path)
    return dependencies


def verify_span_projection(
    parts: tuple[SpanPartition, ...],
    entry: dict[str, Any],
    *,
    on_read: Callable[[int], None] | None = None,
) -> dict[Path, list[int]]:
    """Prove the exact partition receipts and ordered logical span bytes."""
    generations = {path: range_file_identity(path) for path in span_dependencies(parts)}
    whole = hashlib.sha256()
    for part in parts:
        data = part.receipt.read_bytes()
        receipt = json.loads(data)
        identity = receipt["identity"]
        expected = receipt["files"][SPAN_MEMBER]
        if (
            hashlib.sha256(data).hexdigest() != part.entry["receipt_sha256"]
            or receipt["schema_version"] != 1
            or receipt["key"] != part.entry["partition_key"]
            or canonical_digest(identity) != receipt["key"]
            or identity["artifact_type"] != "normalized_source_partition"
            or (identity["source_id"], identity["path"]) != part.source_file
            or expected["size"] != part.entry["size"]
        ):
            raise ValueError(f"normalized span partition receipt changed: {part.spans}")
        digest = hashlib.sha256()
        size, last = 0, b"\n"
        with part.spans.open("rb") as stream:
            for block in iter(lambda: stream.read(2 * 1024**2), b""):
                digest.update(block)
                whole.update(block)
                size += len(block)
                last = block[-1:]
                if on_read is not None:
                    on_read(len(block))
        if (
            last != b"\n"
            or size != expected["size"]
            or digest.hexdigest() != expected["sha256"]
        ):
            raise ValueError(f"normalized span payload changed: {part.spans}")
    if whole.hexdigest() != entry["sha256"]:
        raise ValueError("normalized span projection differs from the rendered view")
    if any(range_file_identity(path) != stamp for path, stamp in generations.items()):
        raise ValueError("normalized span inputs changed during verification")
    return generations
