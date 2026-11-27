"""Native LM TRAIN selection over receipted normalized text statistics."""

from __future__ import annotations

import hashlib
import json
import mmap
import re
import struct
import sys
import tempfile
from array import array
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from klab.corpus.artifact_inventory import parse_normalized_partition_id
from klab.corpus.project import MixturePolicy
from klab.corpus.quota import estimate_supply
from klab.corpus.release_verify import scratch_root
from klab.corpus.text_statistics import annotate_normalized
from klab.workspace_preflight import check_storage, require_storage

MINIMUM = "minimum_train_utf8_bytes"
BELOW_MINIMUM = "below_minimum_utf8_bytes"
# The documented text-statistics v1 sidecar row, independent of the producer.
_STATISTICS = struct.Struct("<32sIII")
_UINT32 = struct.Struct("<I")
_BATCH_ROWS = 8192
_DIGEST = re.compile(r"[0-9a-f]{64}")


def minimum_train_bytes(transforms) -> int | None:
    """Read the named policy from the same recorded transform a build uses."""
    selected = [spec for spec in transforms if MINIMUM in spec["parameters"]]
    if not selected:
        return None
    if len(transforms) != 1 or selected[0]["kind"] != "lm_text":
        raise ValueError(f"{MINIMUM} requires a single LM text transform")
    value = selected[0]["parameters"][MINIMUM]
    if type(value) is not int or not 1 <= value < 2**32:
        raise ValueError(f"{MINIMUM} must be a positive uint32 byte count")
    return value


def _verified_rows(sidecar):
    payload = sidecar["payload"]
    count = sidecar["documents"]
    if (
        sidecar["artifact_type"] != "text-statistics"
        or sidecar["schema_version"] != 1
        or payload["bytes"] != count * _STATISTICS.size
    ):
        raise ValueError("unsupported text-statistics sidecar format")
    digest = hashlib.sha256()
    with Path(payload["path"]).open("rb") as stream:
        for offset in range(0, count, _BATCH_ROWS):
            size = min(_BATCH_ROWS, count - offset) * _STATISTICS.size
            block = stream.read(size)
            if len(block) != size:
                raise ValueError("truncated text-statistics sidecar")
            digest.update(block)
            yield from _STATISTICS.iter_unpack(block)
        if stream.read(1) or digest.hexdigest() != payload["sha256"]:
            raise ValueError("text-statistics sidecar digest differs from its receipt")


class TrainTextSizes:
    """One four-byte size per native index row, addressed by dense row ID."""

    def __init__(self, mapping: mmap.mmap, sidecars: list[dict[str, Any]]):
        self.mapping = mapping
        self.sidecars = sidecars

    def __getitem__(self, rowid: int) -> int:
        return _UINT32.unpack_from(self.mapping, (rowid - 1) * _UINT32.size)[0]


@contextmanager
def open_train_text_sizes(work_root, prepared, connection, progress=None):
    """Join normalized-order sidecars to the existing native index in one scan."""
    # Native document row IDs are dense. The endpoint lookup avoids a second
    # full index scan, and the ordered join below checks every ordinal.
    count = connection.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM documents"
    ).fetchone()[0]
    scratch = scratch_root()
    require_storage([check_storage(scratch, projected_bytes=count * _UINT32.size)])
    indexed = iter(
        connection.execute(
            "SELECT rowid, shard, document_id FROM documents ORDER BY rowid"
        )
    )
    sidecars = []
    observations = set()
    rowid = 0
    with tempfile.TemporaryFile(dir=scratch) as output:
        widths = array("I")
        if widths.itemsize != _UINT32.size:
            raise ValueError("host does not provide four-byte text size entries")
        for shard, partition in enumerate(prepared):
            identifier = f"{partition.parent.name}/{partition.name}"
            sidecar = annotate_normalized(work_root, identifier)
            if sidecar["input"]["partition"] != identifier:
                raise ValueError("text-statistics sidecar names a different partition")
            sidecars.append(
                {
                    **sidecar["input"],
                    "computation_key": sidecar["computation_key"],
                    "payload_sha256": sidecar["payload"]["sha256"],
                    "documents": sidecar["documents"],
                }
            )
            for doc_id, size, _, _ in _verified_rows(sidecar):
                rowid += 1
                if next(indexed, None) != (rowid, shard, doc_id.hex()):
                    raise ValueError(
                        "text-statistics rows differ from the native document index"
                    )
                widths.append(size)
                if len(widths) == _BATCH_ROWS:
                    _write_sizes(output, widths)
            observations.add(
                "reused_text_statistics"
                if sidecar["reused"]
                else "built_text_statistics"
            )
        _write_sizes(output, widths)
        if rowid != count or next(indexed, None) is not None:
            raise ValueError(
                "text-statistics sidecars do not cover the native document index"
            )
        if progress is not None:
            for event in sorted(observations):
                progress.record(event)
        output.flush()
        with mmap.mmap(output.fileno(), 0, access=mmap.ACCESS_READ) as mapping:
            yield TrainTextSizes(mapping, sidecars)


def _write_sizes(output, widths):
    if sys.byteorder != "little":
        widths.byteswap()
    output.write(widths.tobytes())
    del widths[:]


def verify_train_selection_report(
    db,
    report,
    release,
    minimum,
    stage_sources,
    view_sources,
    excluded_by_source,
    documents,
):
    """Verify aggregate reasons and supply against independently checked text."""
    selection = report.get("train_selection")
    expected_exclusions = sum(
        excluded_by_source.get(source, 0) for source in stage_sources
    )
    if (
        not isinstance(selection, dict)
        or selection.get("minimum_utf8_bytes") != minimum
        or selection.get("excluded_by_reason") != {BELOW_MINIMUM: expected_exclusions}
        or not isinstance(selection.get("sidecars"), list)
    ):
        raise ValueError("TRAIN selection report differs from verified documents")
    inventory = set()
    count = 0
    for sidecar in selection["sidecars"]:
        if not isinstance(sidecar, dict) or not isinstance(
            sidecar.get("partition"), str
        ):
            raise ValueError("invalid TRAIN selection sidecar inventory")  # noqa: TRY004 - invalid serialized artifact
        partition = sidecar["partition"]
        parse_normalized_partition_id(partition)
        if (
            partition in inventory
            or any(
                not isinstance(sidecar.get(name), str)
                or not _DIGEST.fullmatch(sidecar[name])
                for name in (
                    "computation_key",
                    "payload_sha256",
                    "documents_sha256",
                    "receipt_sha256",
                )
            )
            or type(sidecar.get("documents")) is not int
            or sidecar["documents"] < 0
        ):
            raise ValueError("invalid TRAIN selection sidecar inventory")
        inventory.add(partition)
        count += sidecar["documents"]
    if count != documents:
        raise ValueError(
            "TRAIN selection sidecar inventory differs from document count"
        )
    available = Counter()
    for source, domains, size in db.execute(
        "SELECT source, domains, SUM(utf8_bytes) FROM docs WHERE dropped IS NULL "
        "AND split='train' AND utf8_bytes>=? GROUP BY source, domains",
        (minimum,),
    ):
        if source in view_sources:
            for domain in json.loads(domains):
                available[domain] += size
    policy = MixturePolicy.model_validate(release["policy"]["mixture"])
    if report.get("mixture_supply") != estimate_supply(
        policy, release["mixture"], available
    ):
        raise ValueError("TRAIN mixture supply differs from verified selection")
