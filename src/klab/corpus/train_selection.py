"""Native LM TRAIN selection over receipted normalized annotations."""

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
from klab.corpus.privacy_candidates import (
    PRIVACY_FIELDS,
    PRIVACY_ROW,
    annotate_privacy_candidates,
    privacy_flags,
)
from klab.corpus.project import MixturePolicy
from klab.corpus.quota import estimate_supply
from klab.corpus.release_verify import scratch_root
from klab.corpus.source_language import annotate_source_language
from klab.corpus.source_quality import (
    QUALITY_FIELDS,
    SOURCE_QUALITY_ROW,
    annotate_source_quality,
    quality_value,
)
from klab.corpus.text_statistics import annotate_normalized
from klab.workspace_preflight import check_storage, require_storage

MINIMUM = "minimum_train_utf8_bytes"
BELOW_MINIMUM = "below_minimum_utf8_bytes"
ROW_LANGUAGE = "train_row_language"
SOURCE_QUALITY = "train_source_quality"
PRIVACY = "train_privacy_candidates"
LANGUAGE_EXCLUSIONS = {1: "row_language_not_allowed", 2: "row_language_missing"}
QUALITY_EXCLUSIONS = {
    3: "below_source_quality_minimum",
    4: "source_quality_unavailable",
}
PRIVACY_MATCH = 5
PRIVACY_EXCLUSIONS = {PRIVACY_MATCH: "privacy_candidate"}
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


def _parameter(transforms, name):
    selected = [spec for spec in transforms if name in spec["parameters"]]
    if not selected:
        return False, None
    if len(transforms) != 1 or selected[0]["kind"] != "lm_text":
        raise ValueError(f"{name} requires a single LM text transform")
    return True, selected[0]["parameters"][name]


def _train_row_language(transforms) -> TrainRowLanguage | None:
    declared, value = _parameter(transforms, ROW_LANGUAGE)
    if not declared:
        return None
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


@dataclass(frozen=True)
class SourceQualityRule:
    field: str
    minimum: int | float
    missing: str

    def decision(self, state, value):
        if state == 1:
            return 3 if value < self.minimum else 0
        if state in (0, 2):
            return 4 if self.missing == "drop" else 0
        raise ValueError("invalid source-quality field state")


@dataclass(frozen=True)
class TrainRowSelection:
    language: TrainRowLanguage | None
    quality: dict[str, SourceQualityRule]
    privacy_mask: int

    def report(self):
        return {
            **({"row_language": self.language.report()} if self.language else {}),
            **(
                {
                    "source_quality": {
                        source: {
                            "field": rule.field,
                            "minimum": rule.minimum,
                            "missing": rule.missing,
                        }
                        for source, rule in sorted(self.quality.items())
                    }
                }
                if self.quality
                else {}
            ),
            **(
                {
                    "privacy_candidates": [
                        name
                        for bit, name in enumerate(PRIVACY_FIELDS)
                        if self.privacy_mask & (1 << bit)
                    ]
                }
                if self.privacy_mask
                else {}
            ),
        }

    def reasons(self):
        return {
            **(LANGUAGE_EXCLUSIONS if self.language else {}),
            **(QUALITY_EXCLUSIONS if self.quality else {}),
            **(PRIVACY_EXCLUSIONS if self.privacy_mask else {}),
        }

    def document_decision(self, doc, *, include_privacy=True):
        reason = self.language.document_decision(doc) if self.language else 0
        if not reason and (rule := self.quality.get(doc["source_id"])) is not None:
            reason = rule.decision(
                *quality_value(rule.field, (doc.get("metadata") or {}).get(rule.field))
            )
        return reason or (
            PRIVACY_MATCH
            if include_privacy
            and self.privacy_mask
            and privacy_flags(doc["text"], self.privacy_mask)
            else 0
        )

    def verify_source_decision(self, decision, row, source, *, code):
        """Independently check row-label/score decisions against the pinned input."""
        if self.language is None and source not in self.quality:
            return
        row = row or {}
        upstream = row.get("_klab_source", {})
        label = upstream.get("language", row.get("language"))
        reason = self.language.decision(label, code=code) if self.language else 0
        expected = 0 if decision == PRIVACY_MATCH else decision
        if reason != (expected if expected in LANGUAGE_EXCLUSIONS else 0):
            raise ValueError(
                "document row language decision differs from pinned source"
            )
        if not reason and (rule := self.quality.get(source)) is not None:
            reason = rule.decision(
                *quality_value(
                    rule.field, upstream.get(rule.field, row.get(rule.field))
                )
            )
        if reason != expected:
            raise ValueError(
                "document source-quality decision differs from pinned source"
            )


def train_row_selection(transforms, sources=None) -> TrainRowSelection | None:
    language = _train_row_language(transforms)
    declared, value = _parameter(transforms, SOURCE_QUALITY)
    quality = {}
    if declared:
        if not isinstance(value, dict) or not value:
            raise ValueError(
                f"{SOURCE_QUALITY} requires a nonempty mapping by source ID"
            )
        for source, rule in value.items():
            if not isinstance(source, str) or (
                sources is not None and source not in sources
            ):
                raise ValueError(f"unknown {SOURCE_QUALITY} source: {source}")
            if (
                not isinstance(rule, dict)
                or set(rule) != {"field", "minimum", "missing"}
                or rule["field"] not in QUALITY_FIELDS
                or quality_value(rule["field"], rule["minimum"])[0] != 1
                or rule["missing"] not in ("keep", "drop")
            ):
                raise ValueError(f"invalid {SOURCE_QUALITY} rule for {source}")
            quality[source] = SourceQualityRule(**rule)
    declared, value = _parameter(transforms, PRIVACY)
    mask = 0
    if declared:
        if (
            not isinstance(value, list)
            or not value
            or any(name not in PRIVACY_FIELDS for name in value)
            or len(value) != len(set(value))
        ):
            raise ValueError(
                f"{PRIVACY} requires unique categories from {PRIVACY_FIELDS}"
            )
        mask = sum(1 << PRIVACY_FIELDS.index(name) for name in value)
    return (
        TrainRowSelection(language, quality, mask)
        if language or quality or mask
        else None
    )


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

    def __init__(self, sizes, decisions, inventories):
        self.sizes = sizes
        self.decisions = decisions
        self.inventories = inventories

    def __getitem__(self, rowid):
        offset = rowid - 1
        size = (
            _UINT32.unpack_from(self.sizes, offset * _UINT32.size)[0]
            if self.sizes is not None
            else None
        )
        decision = self.decisions[offset] if self.decisions is not None else 0
        return size, decision


@contextmanager
def open_train_annotations(
    work_root, prepared, connection, minimum, row_selection, progress=None
):
    """Join normalized-order sidecars to the existing native index in one scan."""
    # Native document row IDs are dense. The endpoint lookup avoids a second
    # full index scan, and the ordered join below checks every ordinal.
    count = connection.execute(
        "SELECT COALESCE(MAX(rowid), 0) FROM documents"
    ).fetchone()[0]
    scratch = scratch_root()
    width = _UINT32.size * (minimum is not None) + (row_selection is not None)
    require_storage([check_storage(scratch, projected_bytes=count * width)])
    indexed = iter(
        connection.execute(
            "SELECT rowid, shard, document_id FROM documents ORDER BY rowid"
        )
    )
    inventories = {}
    language = row_selection.language if row_selection else None
    quality = row_selection.quality if row_selection else {}
    privacy_mask = row_selection.privacy_mask if row_selection else 0
    observations = set()
    rowid = 0
    indexed_row = next(indexed, None)

    def attach(partition, kind, annotate, layout, name, inputs, columns):
        evidence = annotate(work_root, partition)
        if evidence["input"]["partition"] != partition:
            raise ValueError(f"{kind} sidecar names a different partition")
        inventories.setdefault(name, []).append(
            {
                **evidence["input"],
                "computation_key": evidence["computation_key"],
                "payload_sha256": evidence["payload"]["sha256"],
                "documents": evidence["documents"],
            }
        )
        action = "reused" if evidence["reused"] else "built"
        observations.add(f"{action}_{kind.replace('-', '_')}")
        columns[kind] = inputs.enter_context(
            closing(_verified_rows(evidence, kind, layout))
        )
        return evidence

    with ExitStack() as stack:
        output = (
            stack.enter_context(tempfile.TemporaryFile(dir=scratch))
            if minimum is not None
            else None
        )
        decisions = (
            stack.enter_context(tempfile.TemporaryFile(dir=scratch))
            if row_selection is not None
            else None
        )
        widths = array("I")
        outcomes = bytearray()
        if output is not None and widths.itemsize != _UINT32.size:
            raise ValueError("host does not provide four-byte text size entries")
        for shard, partition in enumerate(prepared):
            identifier = f"{partition.parent.name}/{partition.name}"
            with ExitStack() as inputs:
                columns = {}
                if output is not None:
                    attach(
                        identifier,
                        "text-statistics",
                        annotate_normalized,
                        _STATISTICS,
                        "sidecars",
                        inputs,
                        columns,
                    )
                if language is not None:
                    evidence = attach(
                        identifier,
                        "source-language",
                        annotate_source_language,
                        _LANGUAGE,
                        "language_sidecars",
                        inputs,
                        columns,
                    )
                    labels = evidence["row_labels"]
                    label_decisions = [
                        language.decision(None),
                        *(language.decision(label) for label in labels),
                    ]
                rule = quality.get(partition.parent.name)
                if rule is not None:
                    attach(
                        identifier,
                        "source-quality",
                        annotate_source_quality,
                        SOURCE_QUALITY_ROW,
                        "quality_sidecars",
                        inputs,
                        columns,
                    )
                    quality_offset = QUALITY_FIELDS.index(rule.field)
                if privacy_mask:
                    attach(
                        identifier,
                        "privacy-candidates",
                        annotate_privacy_candidates,
                        PRIVACY_ROW,
                        "privacy_sidecars",
                        inputs,
                        columns,
                    )

                # A source without a quality rule contributes ordered zero
                # decisions. It has no quality artifact dependency.
                if not columns:
                    while indexed_row is not None and indexed_row[1] == shard:
                        rowid += 1
                        if indexed_row[0] != rowid:
                            raise ValueError(
                                "native document index row IDs are not dense"
                            )
                        indexed_row = next(indexed, None)
                        outcomes.append(0)
                        if len(outcomes) == _BATCH_ROWS:
                            decisions.write(outcomes)
                            outcomes.clear()
                    continue

                names = tuple(columns)
                language_column = (
                    names.index("source-language") if language is not None else None
                )
                quality_column = (
                    names.index("source-quality") if rule is not None else None
                )
                privacy_column = (
                    names.index("privacy-candidates") if privacy_mask else None
                )
                for records in zip(*columns.values(), strict=True):
                    record = records[0]
                    document_id = record[0]
                    rowid += 1
                    if indexed_row != (rowid, shard, document_id.hex()):
                        raise ValueError(
                            f"{names[0]} rows differ from the native document index"
                        )
                    indexed_row = next(indexed, None)
                    if output is not None:
                        widths.append(record[1])
                        if len(widths) == _BATCH_ROWS:
                            _write_sizes(output, widths)
                    result = 0
                    if language_column is not None:
                        language_id, label, code, _ = records[language_column]
                        if language_id != document_id:
                            raise ValueError(
                                "source-language rows differ from the native document index"
                            )
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
                        if code:
                            result = 0
                    if quality_column is not None:
                        quality_record = records[quality_column]
                        if quality_record[0] != document_id:
                            raise ValueError(
                                "source-quality rows differ from the native document index"
                            )
                        state = (quality_record[-1] >> (2 * quality_offset)) & 3
                        quality_result = rule.decision(
                            state, quality_record[quality_offset + 1]
                        )
                        result = result or quality_result
                    if privacy_column is not None:
                        privacy_id, flags = records[privacy_column]
                        if privacy_id != document_id:
                            raise ValueError(
                                "privacy-candidate rows differ from the native document index"
                            )
                        if flags >= 1 << len(PRIVACY_FIELDS):
                            raise ValueError("invalid privacy candidate flags")
                        result = result or (
                            PRIVACY_MATCH if flags & privacy_mask else 0
                        )
                    if decisions is not None:
                        outcomes.append(result)
                        if len(outcomes) == _BATCH_ROWS:
                            decisions.write(outcomes)
                            outcomes.clear()
        if output is not None:
            _write_sizes(output, widths)
        if decisions is not None:
            decisions.write(outcomes)
        if rowid != count or indexed_row is not None:
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

        if quality:
            inventories.setdefault("quality_sidecars", [])
        yield TrainAnnotations(map_rows(output), map_rows(decisions), inventories)


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
    row_selection=None,
    row_exclusions=None,
    quality_documents=0,
):
    """Verify aggregate reasons and supply against independently checked text."""
    selection = report.get("train_selection")
    expected = {BELOW_MINIMUM: expected_exclusions} if minimum is not None else {}
    if row_selection is not None:
        expected.update(row_exclusions)
    if (
        not isinstance(selection, dict)
        or not train_selection_options_match(selection, minimum, row_selection)
        or selection.get("excluded_by_reason") != expected
    ):
        raise ValueError("TRAIN selection report differs from verified documents")
    if minimum is not None:
        _verify_sidecar_inventory(selection.get("sidecars"), documents)
    if row_selection and row_selection.language:
        _verify_sidecar_inventory(selection.get("language_sidecars"), documents)
    if row_selection and row_selection.quality:
        _verify_sidecar_inventory(
            selection.get("quality_sidecars"), quality_documents, row_selection.quality
        )
    if row_selection and row_selection.privacy_mask:
        _verify_sidecar_inventory(selection.get("privacy_sidecars"), documents)
    policy = MixturePolicy.model_validate(release["policy"]["mixture"])
    if report.get("mixture_supply") != estimate_supply(
        policy, release["mixture"], available
    ):
        raise ValueError("TRAIN mixture supply differs from verified selection")


def train_selection_options_match(report, minimum, row_selection):
    """Compare the recorded options independent of their report provenance."""
    options = row_selection.report() if row_selection is not None else {}
    return report.get("minimum_utf8_bytes") == minimum and all(
        report.get(name) == options.get(name)
        for name in ("row_language", "source_quality", "privacy_candidates")
    )


def _verify_sidecar_inventory(sidecars, documents, sources=None):
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
        source, _ = parse_normalized_partition_id(partition)
        if (
            partition in inventory
            or (sources is not None and source not in sources)
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
