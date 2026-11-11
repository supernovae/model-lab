"""Strict local ingestion for canonical Engram knowledge records."""

from __future__ import annotations

import json
import math
import re
import unicodedata
from collections.abc import Iterator
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from pydantic import (
    StrictInt,
    StrictStr,
    ValidationError,
    field_validator,
    model_validator,
)

from klab.canonical import canonical_json, reject_duplicate_keys
from klab.config.models import StrictModel

MAX_RECORD_BYTES = 1024 * 1024
_RECORD_FIELDS = frozenset(
    {
        "record_version",
        "id",
        "namespace",
        "subject",
        "relation",
        "value",
        "text",
        "query",
        "aliases",
        "hard_negative_ids",
        "source",
        "source_revision",
        "license",
        "created_at",
        "valid_from",
        "valid_until",
        "confidence",
        "priority",
    }
)
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_FRACTION_RE = re.compile(r"[Tt ]\d{2}:\d{2}:\d{2}[.,](\d+)")


def _nonblank(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _normalize_time(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        return value.isoformat()
    elif isinstance(value, str):
        if _DATE_RE.fullmatch(value):
            try:
                return date.fromisoformat(value).isoformat()
            except ValueError as error:
                raise ValueError("invalid ISO-8601 date") from error
        fraction = _FRACTION_RE.search(value)
        if fraction and len(fraction.group(1)) > 6:
            raise ValueError(
                "timestamp precision finer than microseconds is unsupported"
            )
        try:
            normalized_value = value[:-1] + "Z" if value.endswith("z") else value
            parsed = datetime.fromisoformat(normalized_value)
        except ValueError as error:
            raise ValueError("expected an ISO-8601 date or timestamp") from error
    else:
        raise TypeError("expected an ISO-8601 date or timestamp")
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamps must include a timezone")
    utc = parsed.astimezone(UTC)
    fraction = f".{utc.microsecond:06d}" if utc.microsecond else ""
    return f"{utc.strftime('%Y-%m-%dT%H:%M:%S')}{fraction}Z"


def _time_instant(value: str | None) -> datetime | None:
    if value is None:
        return None
    if _DATE_RE.fullmatch(value):
        return datetime.combine(date.fromisoformat(value), datetime.min.time(), UTC)
    return datetime.fromisoformat(value)


def _reject_constant(value: str) -> object:
    raise ValueError(f"nonfinite JSON number: {value}")


class KnowledgeRecord(StrictModel):
    """One identity-stable record, retaining authored bytes and metadata."""

    record_version: Literal[1] = 1
    id: StrictStr
    namespace: StrictStr
    subject: StrictStr | None = None
    relation: StrictStr | None = None
    value: StrictStr | None = None
    text: StrictStr | None = None
    query: StrictStr | None = None
    aliases: tuple[StrictStr, ...] = ()
    hard_negative_ids: tuple[StrictStr, ...] = ()
    source: StrictStr | None = None
    source_revision: StrictStr | None = None
    license: StrictStr
    created_at: StrictStr | None = None
    valid_from: StrictStr | None = None
    valid_until: StrictStr | None = None
    confidence: float | None = None
    priority: StrictInt | None = None

    @field_validator("record_version", mode="before")
    @classmethod
    def _strict_record_version(cls, value: object) -> object:
        if type(value) is not int or value != 1:
            raise ValueError("record_version must be integer 1")
        return value

    @field_validator("confidence", mode="before")
    @classmethod
    def _strict_confidence(cls, value: object) -> object:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("confidence must be a finite JSON number")
        if not 0 <= value <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("confidence must be finite and in [0, 1]")
        return value

    @field_validator("created_at", "valid_from", "valid_until", mode="before")
    @classmethod
    def _canonical_times(cls, value: object) -> str | None:
        return _normalize_time(value)

    @model_validator(mode="after")
    def _validate_record(self) -> KnowledgeRecord:
        if not _nonblank(self.id):
            raise ValueError("id must be nonblank")
        if any(unicodedata.category(character) == "Cc" for character in self.id):
            raise ValueError("id must not contain control characters")
        if not _nonblank(self.namespace):
            raise ValueError("namespace must be nonblank")
        if not _nonblank(self.license):
            raise ValueError("license must be nonblank")
        triple = (self.subject, self.relation, self.value)
        if any(item is not None for item in triple) and not all(
            _nonblank(item) for item in triple
        ):
            raise ValueError(
                "subject, relation and value must be all nonblank or all null"
            )
        if not _nonblank(self.text) and not all(_nonblank(item) for item in triple):
            raise ValueError("record requires a complete triple or nonblank text")
        if self.text is not None and not _nonblank(self.text):
            raise ValueError("text must be nonblank when present")
        for name in ("query", "source", "source_revision"):
            value = getattr(self, name)
            if value is not None and not _nonblank(value):
                raise ValueError(f"{name} must be nonblank when present")
        for field_name in ("aliases", "hard_negative_ids"):
            values = getattr(self, field_name)
            if any(not _nonblank(item) for item in values):
                raise ValueError(f"{field_name} entries must be nonblank")
            if len(set(values)) != len(values):
                raise ValueError(f"{field_name} entries must be unique")
        if self.id in self.hard_negative_ids:
            raise ValueError("record cannot be its own hard negative")
        start, end = _time_instant(self.valid_from), _time_instant(self.valid_until)
        if start is not None and end is not None and end < start:
            raise ValueError("valid_until precedes valid_from")
        return self


def _prepare_record(
    raw: object,
    *,
    namespace: str,
    default_license: str | None,
    source_name: str | None,
    source_revision: str | None,
) -> KnowledgeRecord:
    if not isinstance(raw, dict):
        raise TypeError("record must be a JSON object")
    unknown = set(raw) - _RECORD_FIELDS
    if unknown:
        raise ValueError(f"unknown record fields: {', '.join(sorted(unknown))}")
    values = dict(raw)
    if values.get("namespace") is None:
        values["namespace"] = namespace
    if values.get("source") is None and source_name is not None:
        values["source"] = source_name
    if values.get("source_revision") is None and source_revision is not None:
        source = values.get("source")
        if source_name is not None and source == source_name:
            values["source_revision"] = source_revision
    if values.get("license") is None and default_license is not None:
        values["license"] = default_license
    for list_field in ("aliases", "hard_negative_ids"):
        value = values.get(list_field)
        if value is None:
            values[list_field] = ()
        elif isinstance(value, list):
            values[list_field] = tuple(value)
    try:
        return KnowledgeRecord.model_validate(values, strict=True)
    except ValidationError as error:
        reasons = []
        for issue in error.errors(include_input=False):
            location = ".".join(str(part) for part in issue["loc"])
            reasons.append(f"{location or 'record'}: {issue['msg']}")
        raise ValueError("; ".join(reasons)) from error


def _rows_jsonl(path: Path) -> Iterator[tuple[int, object]]:
    try:
        with path.open("rb") as handle:
            line_number = 0
            while True:
                raw_line = handle.readline(MAX_RECORD_BYTES + 1)
                if not raw_line:
                    break
                line_number += 1
                if len(raw_line) > MAX_RECORD_BYTES:
                    raise ValueError(
                        f"{path}:{line_number}: raw JSONL line exceeds 1 MiB"
                    )
                try:
                    line = raw_line.decode("utf-8", errors="strict")
                    if not line.strip():
                        raise ValueError("blank JSONL line")
                    record = json.loads(
                        line,
                        object_pairs_hook=reject_duplicate_keys,
                        parse_constant=_reject_constant,
                    )
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                yield line_number, record
    except OSError as error:
        raise OSError(f"{path}: {error}") from error


def _rows_parquet(path: Path) -> Iterator[tuple[int, object]]:
    from pyarrow import ArrowException, parquet

    try:
        parquet_file = parquet.ParquetFile(path)
        names = parquet_file.schema_arrow.names
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate Parquet column names")
        unknown = set(names) - _RECORD_FIELDS
        if unknown:
            raise ValueError(
                f"{path}: unknown Parquet columns: {', '.join(sorted(unknown))}"
            )
        row_number = 0
        for batch in parquet_file.iter_batches(batch_size=128):
            for record in batch.to_pylist():
                row_number += 1
                yield row_number, record
    except OSError as error:
        raise OSError(f"{path}: {error}") from error
    except ArrowException as error:
        raise ValueError(f"{path}: {error}") from error


def _rows(path: Path) -> Iterator[tuple[int, object]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        yield from _rows_jsonl(path)
    elif suffix == ".parquet":
        yield from _rows_parquet(path)
    else:
        raise ValueError(
            f"unsupported record input extension: {path.suffix or '<none>'}"
        )


def iter_knowledge_records(
    path: Path,
    *,
    namespace: str,
    default_license: str | None = None,
    source_name: str | None = None,
    source_revision: str | None = None,
) -> Iterator[KnowledgeRecord]:
    """Yield canonical rows after strict schema and cross-reference validation."""
    if source_revision is not None and source_name is None:
        raise ValueError("source_revision requires source_name")
    if not _nonblank(namespace):
        raise ValueError("compile namespace must be nonblank")
    if default_license is not None and not _nonblank(default_license):
        raise ValueError("default license must be nonblank")
    if source_name is not None and not _nonblank(source_name):
        raise ValueError("source_name must be nonblank")
    if source_revision is not None and not _nonblank(source_revision):
        raise ValueError("source_revision must be nonblank")

    ids: set[str] = set()
    count = 0
    for row_number, raw in _rows(path):
        try:
            record = _prepare_record(
                raw,
                namespace=namespace,
                default_license=default_license,
                source_name=source_name,
                source_revision=source_revision,
            )
            size = len(canonical_json(record.model_dump(mode="json")))
            if size > MAX_RECORD_BYTES:
                raise ValueError("canonical record exceeds 1 MiB")
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{row_number}: {error}") from error
        if record.id in ids:
            raise ValueError(f"{path}:{row_number}: duplicate record id {record.id!r}")
        ids.add(record.id)
        count += 1
    if count == 0:
        raise ValueError(f"{path}: input contains no knowledge records")

    for row_number, raw in _rows(path):
        try:
            record = _prepare_record(
                raw,
                namespace=namespace,
                default_license=default_license,
                source_name=source_name,
                source_revision=source_revision,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{row_number}: {error}") from error
        missing = [
            reference for reference in record.hard_negative_ids if reference not in ids
        ]
        if missing:
            raise ValueError(
                f"{path}:{row_number}: record {record.id!r} has dangling hard negatives: "
                f"{', '.join(missing)}"
            )

    for row_number, raw in _rows(path):
        try:
            yield _prepare_record(
                raw,
                namespace=namespace,
                default_license=default_license,
                source_name=source_name,
                source_revision=source_revision,
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"{path}:{row_number}: {error}") from error


def canonical_record_bytes(record: KnowledgeRecord) -> bytes:
    """Return the unique JSON-native representation used in packs."""
    return canonical_json(record.model_dump(mode="json"))
