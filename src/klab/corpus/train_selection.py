"""Native LM TRAIN selection over receipted normalized text statistics."""

from __future__ import annotations

import hashlib
import mmap
import re
import struct
import sys
import tempfile
from array import array
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

from klab.corpus.artifact_inventory import parse_normalized_partition_id
from klab.corpus.project import MixturePolicy
from klab.corpus.quota import estimate_supply
from klab.corpus.release_verify import scratch_root
from klab.corpus.source_language import annotate_source_language
from klab.corpus.text_statistics import annotate_normalized
from klab.workspace_preflight import check_storage, require_storage

MINIMUM = "minimum_train_utf8_bytes"
BELOW_MINIMUM = "below_minimum_utf8_bytes"
ROW_LANGUAGE = "train_row_language"
LANGUAGE_EXCLUSIONS = {1: "row_language_not_allowed", 2: "row_language_missing"}
# The documented text-statistics v1 sidecar row, independent of the producer.
_STATISTICS = struct.Struct("<32sIII")
_LANGUAGE = struct.Struct("<32sIBd")
_UINT32 = struct.Struct("<I")
_BATCH_ROWS = 8192
_DIGEST = re.compile(r"[0-9a-f]{64}")
CHECK_ROW_LABEL = 255


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


def _usable_label(value):
    return (
        isinstance(value, str)
        and 0 < len(value) <= 64
        and bool(value.strip())
        and len(value.encode("utf-8")) <= 64
    )


@dataclass(frozen=True)
class TrainRowLanguage:
    allowed: frozenset[str]
    missing: str

    def report(self):
        return {"allowed": sorted(self.allowed), "missing": self.missing}

    def decision(self, label, *, code=False):
        if code:
            return 0
        if not _usable_label(label):
            return 2 if self.missing == "drop" else 0
        return 0 if label in self.allowed else 1

    def document_decision(self, doc):
        metadata = doc.get("metadata") or {}
        return self.decision(
            metadata.get("language"),
            code=(
                doc["document_kind"] == "code"
                or metadata.get("qualification_profile") == "code"
            ),
        )

    def verify_source_decision(self, decision, row, *, code):
        """Check published selection against the row in its pinned source."""
        label = (
            row.get("_klab_source", {}).get("language", row.get("language"))
            if row is not None
            else None
        )
        if self.decision(label, code=code) != decision:
            raise ValueError(
                "document row language decision differs from pinned source"
            )


def source_language_code_context(declaration, document_policy, raw_path):
    """Resolve code context from the pinned declaration and input format."""
    kinds = declaration["document_kinds"]
    suffix = raw_path.suffix.lower()
    if (
        suffix in {".md", ".markdown"}
        and "prose" in kinds
        and document_policy["markdown_kind"] == "prose_when_declared"
    ):
        return False
    qualification = declaration["acquisition"].get("qualification")
    return kinds[0] == "code" or (
        suffix == ".jsonl"
        and bool(qualification)
        and qualification["profile"] == "code"
    )


def train_row_language(transforms) -> TrainRowLanguage | None:
    selected = [spec for spec in transforms if ROW_LANGUAGE in spec["parameters"]]
    if not selected:
        return None
    if len(transforms) != 1 or selected[0]["kind"] != "lm_text":
        raise ValueError(f"{ROW_LANGUAGE} requires a single LM text transform")
    value = selected[0]["parameters"][ROW_LANGUAGE]
    if not isinstance(value, dict) or set(value) != {"allowed", "missing"}:
        raise ValueError(f"{ROW_LANGUAGE} requires allowed and missing")
    labels = value["allowed"]
    if (
        not isinstance(labels, list)
        or not labels
        or not all(_usable_label(label) for label in labels)
        or len(set(labels)) != len(labels)
        or value["missing"] not in ("keep", "drop")
    ):
        raise ValueError(
            f"{ROW_LANGUAGE} requires unique labels and missing: keep or drop"
        )
    return TrainRowLanguage(frozenset(labels), value["missing"])


def _verified_rows(sidecar, kind, row):
    payload = sidecar["payload"]
    count = sidecar["documents"]
    if (
        sidecar["artifact_type"] != kind
        or sidecar["schema_version"] != 1
        or payload["bytes"] != count * row.size
    ):
        raise ValueError(f"unsupported {kind} sidecar format")
    digest = hashlib.sha256()
    with Path(payload["path"]).open("rb") as stream:
        for offset in range(0, count, _BATCH_ROWS):
            size = min(_BATCH_ROWS, count - offset) * row.size
            block = stream.read(size)
            if len(block) != size:
                raise ValueError(f"truncated {kind} sidecar")
            digest.update(block)
            yield from row.iter_unpack(block)
        if stream.read(1) or digest.hexdigest() != payload["sha256"]:
            raise ValueError(f"{kind} sidecar digest differs from its receipt")


class TrainAnnotations:
    """Optional fixed-width selections addressed by the dense native row ID."""

    def __init__(self, sizes, languages, sidecars, language_sidecars):
        self.sizes = sizes
        self.languages = languages
        self.sidecars = sidecars
        self.language_sidecars = language_sidecars

    def __getitem__(self, rowid):
        offset = rowid - 1
        size = (
            _UINT32.unpack_from(self.sizes, offset * _UINT32.size)[0]
            if self.sizes is not None
            else None
        )
        language = self.languages[offset] if self.languages is not None else 0
        return size, language


@contextmanager
def open_train_annotations(
    work_root, prepared, connection, minimum, language, progress=None
):
    """Join normalized-order sidecars to the existing native index in one scan."""
    # Native document row IDs are dense. The endpoint lookup avoids a second
    # full index scan, and the ordered join below checks every ordinal.
    count = connection.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM documents"
    ).fetchone()[0]
    scratch = scratch_root()
    width = _UINT32.size * (minimum is not None) + (language is not None)
    require_storage([check_storage(scratch, projected_bytes=count * width)])
    indexed = iter(
        connection.execute(
            "SELECT rowid, shard, document_id FROM documents ORDER BY rowid"
        )
    )
    sidecars, language_sidecars = [], []
    observations = set()
    rowid = 0

    def sidecar(partition, kind, annotate, layout, inventory):
        evidence = annotate(work_root, partition)
        if evidence["input"]["partition"] != partition:
            raise ValueError(f"{kind} sidecar names a different partition")
        inventory.append(
            {
                **evidence["input"],
                "computation_key": evidence["computation_key"],
                "payload_sha256": evidence["payload"]["sha256"],
                "documents": evidence["documents"],
            }
        )
        action = "reused" if evidence["reused"] else "built"
        observations.add(f"{action}_{kind.replace('-', '_')}")
        return evidence, _verified_rows(evidence, kind, layout)

    with ExitStack() as stack:
        output = (
            stack.enter_context(tempfile.TemporaryFile(dir=scratch))
            if minimum is not None
            else None
        )
        decisions = (
            stack.enter_context(tempfile.TemporaryFile(dir=scratch))
            if language is not None
            else None
        )
        widths = array("I")
        outcomes = bytearray()
        if output is not None and widths.itemsize != _UINT32.size:
            raise ValueError("host does not provide four-byte text size entries")
        for shard, partition in enumerate(prepared):
            identifier = f"{partition.parent.name}/{partition.name}"
            with ExitStack() as inputs:
                primary = aligned_language = None
                if output is not None:
                    _, rows = sidecar(
                        identifier,
                        "text-statistics",
                        annotate_normalized,
                        _STATISTICS,
                        sidecars,
                    )
                    primary = inputs.enter_context(closing(rows))
                if decisions is not None:
                    evidence, rows = sidecar(
                        identifier,
                        "source-language",
                        annotate_source_language,
                        _LANGUAGE,
                        language_sidecars,
                    )
                    language_rows = inputs.enter_context(closing(rows))
                    if primary is None:
                        primary = language_rows
                    else:
                        aligned_language = language_rows
                    labels = evidence["row_labels"]
                    label_decisions = [
                        language.decision(None),
                        *(language.decision(label) for label in labels),
                    ]
                for record in primary:
                    rowid += 1
                    if next(indexed, None) != (rowid, shard, record[0].hex()):
                        kind = (
                            "text-statistics"
                            if output is not None
                            else "source-language"
                        )
                        raise ValueError(
                            f"{kind} rows differ from the native document index"
                        )
                    if output is not None:
                        widths.append(record[1])
                        if len(widths) == _BATCH_ROWS:
                            _write_sizes(output, widths)
                    if decisions is not None:
                        language_record = record
                        if aligned_language is not None:
                            language_record = next(aligned_language, None)
                            if (
                                language_record is None
                                or language_record[0] != record[0]
                            ):
                                raise ValueError(
                                    "source-language rows differ from the native document index"
                                )
                        _, label, code, _ = language_record
                        if code not in (0, 1):
                            raise ValueError("invalid source-language code context")
                        if label == 2**32 - 1:
                            # The v1 sidecar combines malformed labels and
                            # dictionary overflow. The writer already decodes
                            # the original row and can distinguish those cases.
                            result = CHECK_ROW_LABEL
                        elif label < len(label_decisions):
                            result = label_decisions[label]
                        else:
                            raise ValueError("invalid source-language label code")
                        outcomes.append(0 if code else result)
                        if len(outcomes) == _BATCH_ROWS:
                            decisions.write(outcomes)
                            outcomes.clear()
                if (
                    aligned_language is not None
                    and next(aligned_language, None) is not None
                ):
                    raise ValueError(
                        "source-language rows exceed the native document index"
                    )
        if output is not None:
            _write_sizes(output, widths)
        if decisions is not None:
            decisions.write(outcomes)
        if rowid != count or next(indexed, None) is not None:
            raise ValueError("TRAIN sidecars do not cover the native document index")
        if progress is not None:
            for event in sorted(observations):
                progress.record(event)

        def map_rows(stream):
            if stream is None or not count:
                return None
            stream.flush()
            return stack.enter_context(
                mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            )

        yield TrainAnnotations(
            map_rows(output), map_rows(decisions), sidecars, language_sidecars
        )


def _write_sizes(output, widths):
    if sys.byteorder != "little":
        widths.byteswap()
    output.write(widths.tobytes())
    del widths[:]


def verify_train_selection_report(
    report,
    release,
    minimum,
    expected_exclusions,
    available,
    documents,
    *,
    language=None,
    language_exclusions=None,
):
    """Verify aggregate reasons and supply against independently checked text."""
    selection = report.get("train_selection")
    expected = {BELOW_MINIMUM: expected_exclusions} if minimum is not None else {}
    if language is not None:
        expected.update(language_exclusions)
    if (
        not isinstance(selection, dict)
        or selection.get("minimum_utf8_bytes") != minimum
        or selection.get("excluded_by_reason") != expected
        or selection.get("row_language") != (language.report() if language else None)
    ):
        raise ValueError("TRAIN selection report differs from verified documents")
    if minimum is not None:
        _verify_sidecar_inventory(selection.get("sidecars"), documents)
    if language is not None:
        _verify_sidecar_inventory(selection.get("language_sidecars"), documents)
    policy = MixturePolicy.model_validate(release["policy"]["mixture"])
    if report.get("mixture_supply") != estimate_supply(
        policy, release["mixture"], available
    ):
        raise ValueError("TRAIN mixture supply differs from verified selection")


def _verify_sidecar_inventory(sidecars, documents):
    if not isinstance(sidecars, list):
        raise ValueError("invalid TRAIN selection sidecar inventory")  # noqa: TRY004 - invalid serialized artifact
    inventory = set()
    count = 0
    for sidecar in sidecars:
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
