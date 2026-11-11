from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta, timezone

import pyarrow as pa
import pytest
from pyarrow import parquet

from klab.engram.records import iter_knowledge_records


def _equivalent_rows() -> tuple[list[dict[str, object]], pa.Schema]:
    offset = timezone(timedelta(hours=2))
    json_rows: list[dict[str, object]] = [
        {
            "id": "r1",
            "subject": "Kelmar",
            "relation": "primary fruit",
            "value": "Tupin",
            "text": "Keep this text exactly: cafe\u0301  ",
            "aliases": ["café", "cafe\u0301"],
            "hard_negative_ids": ["r2"],
            "created_at": "2025-02-03T04:05:06.123456+02:00",
            "valid_from": "2025-01-01",
            "valid_until": "2025-12-31T23:59:59Z",
            "confidence": 0.75,
            "priority": -2,
        },
        {
            "id": "r2",
            "text": "A separate fact",
            "source": "catalog.other",
            "source_revision": None,
            "license": "CC-BY-4.0",
        },
    ]
    schema = pa.schema(
        [
            ("id", pa.string()),
            ("namespace", pa.string()),
            ("subject", pa.string()),
            ("relation", pa.string()),
            ("value", pa.string()),
            ("text", pa.string()),
            ("query", pa.string()),
            ("aliases", pa.list_(pa.string())),
            ("hard_negative_ids", pa.list_(pa.string())),
            ("source", pa.string()),
            ("source_revision", pa.string()),
            ("license", pa.string()),
            ("created_at", pa.timestamp("us", tz="+02:00")),
            ("valid_from", pa.date32()),
            ("valid_until", pa.timestamp("us", tz="UTC")),
            ("confidence", pa.float64()),
            ("priority", pa.int64()),
            ("record_version", pa.int64()),
        ]
    )
    parquet_rows = [
        {
            **row,
            "namespace": None,
            "query": None,
            "source": row.get("source"),
            "source_revision": row.get("source_revision"),
            "license": row.get("license"),
            "created_at": datetime(2025, 2, 3, 4, 5, 6, 123456, tzinfo=offset)
            if index == 0
            else None,
            "valid_from": date(2025, 1, 1) if index == 0 else None,
            "valid_until": datetime(2025, 12, 31, 23, 59, 59, tzinfo=UTC)
            if index == 0
            else None,
            "confidence": 0.75 if index == 0 else None,
            "priority": -2 if index == 0 else None,
            "record_version": 1,
        }
        for index, row in enumerate(json_rows)
    ]
    return parquet_rows, schema


def test_jsonl_and_parquet_normalize_to_identical_records(tmp_path) -> None:
    json_rows = [
        {
            "id": "r1",
            "subject": "Kelmar",
            "relation": "primary fruit",
            "value": "Tupin",
            "text": "Keep this text exactly: cafe\u0301  ",
            "aliases": ["café", "cafe\u0301"],
            "hard_negative_ids": ["r2"],
            "created_at": "2025-02-03T04:05:06.123456+02:00",
            "valid_from": "2025-01-01",
            "valid_until": "2025-12-31T23:59:59Z",
            "confidence": 0.75,
            "priority": -2,
        },
        {
            "id": "r2",
            "text": "A separate fact",
            "source": "catalog.other",
            "license": "CC-BY-4.0",
        },
    ]
    json_path = tmp_path / "input.jsonl"
    json_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in json_rows) + "\n",
        encoding="utf-8",
    )
    parquet_path = tmp_path / "input.parquet"
    parquet_rows, schema = _equivalent_rows()
    parquet.write_table(pa.Table.from_pylist(parquet_rows, schema=schema), parquet_path)

    options = {
        "namespace": "synthetic",
        "default_license": "CC0",
        "source_name": "catalog.default",
        "source_revision": "rev-default",
    }
    json_records = list(iter_knowledge_records(json_path, **options))
    parquet_records = list(iter_knowledge_records(parquet_path, **options))

    assert [row.model_dump(mode="json") for row in json_records] == [
        row.model_dump(mode="json") for row in parquet_records
    ]
    assert json_records[0].created_at == "2025-02-03T02:05:06.123456Z"
    assert json_records[0].valid_from == "2025-01-01"
    assert json_records[0].source == "catalog.default"
    assert json_records[0].source_revision == "rev-default"
    assert json_records[1].source == "catalog.other"
    assert json_records[1].source_revision is None
    assert json_records[1].license == "CC-BY-4.0"
    assert json_records[0].text == "Keep this text exactly: cafe\u0301  "
    assert json_records[0].aliases == ("café", "cafe\u0301")
    assert json_records[0].hard_negative_ids == ("r2",)


@pytest.mark.parametrize(
    "raw, reason",
    [
        ('{"id":"x","text":"x","license":"CC0","priority":true}', "priority"),
        ('{"id":"x","text":"x","license":"CC0","confidence":true}', "confidence"),
        ('{"id":"x","text":"x","license":"CC0","confidence":"0.5"}', "confidence"),
        ('{"id":"x","text":"x","license":"CC0","confidence":2}', r"in \[0, 1\]"),
        (
            '{"id":"x","text":"x","license":"CC0","confidence":' + "9" * 400 + "}",
            r"in \[0, 1\]",
        ),
        ('{"id":"x","text":"x","license":"CC0","confidence":NaN}', "nonfinite"),
        ('{"id":"x","text":"x","record_version":true,"license":"CC0"}', "version"),
        ('{"id":"x","text":"x","record_version":2,"license":"CC0"}', "version"),
        ('{"id":"x","text":"x","license":"CC0","unexpected":1}', "unknown"),
        ('{"id":"x","subject":"s","relation":"r","license":"CC0"}', "subject"),
        (
            '{"id":"x","text":"x","license":"CC0","valid_from":"2025-02-01","valid_until":"2025-01-01"}',
            "valid_until",
        ),
        (
            '{"id":"x","text":"x","license":"CC0","created_at":"2025-01-01T00:00:00"}',
            "timezone",
        ),
        (
            '{"id":"x","text":"x","license":"CC0","created_at":"2025-01-01T00:00:00.1234567Z"}',
            "precision",
        ),
        (
            '{"id":"x","text":"x","license":"CC0","hard_negative_ids":["missing"]}',
            "dangling",
        ),
        (
            '{"id":"x","text":"x","license":"CC0","hard_negative_ids":["x"]}',
            "own hard negative",
        ),
        ('{"id":"x","text":"x","license":"CC0","aliases":["a","a"]}', "aliases"),
        ('{"id":"x","text":"x"}', "license"),
        ('{"id":"x","id":"y","text":"x","license":"CC0"}', "duplicate JSON key"),
    ],
)
def test_jsonl_rejects_invalid_records(tmp_path, raw: str, reason: str) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(raw + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=reason):
        list(iter_knowledge_records(path, namespace="n"))


def test_jsonl_rejects_blank_oversized_and_empty_inputs(tmp_path) -> None:
    path = tmp_path / "blank.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"blank.jsonl:1"):
        list(iter_knowledge_records(path, namespace="n"))
    path.write_text(
        json.dumps({"id": "x", "text": "x" * (1024 * 1024), "license": "CC0"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="1 MiB"):
        list(iter_knowledge_records(path, namespace="n"))
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="no knowledge records"):
        list(iter_knowledge_records(path, namespace="n"))


def test_duplicate_ids_and_source_revision_requirements(tmp_path) -> None:
    path = tmp_path / "duplicates.jsonl"
    path.write_text(
        '{"id":"x","text":"a","license":"CC0"}\n'
        '{"id":"x","text":"b","license":"CC0"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate record id"):
        list(iter_knowledge_records(path, namespace="n"))
    with pytest.raises(ValueError, match="requires source_name"):
        list(iter_knowledge_records(path, namespace="n", source_revision="rev"))


def test_unknown_parquet_columns_fail_before_record_conversion(tmp_path) -> None:
    path = tmp_path / "unknown.parquet"
    parquet.write_table(pa.table({"id": ["x"], "secret": [None]}), path)
    with pytest.raises(ValueError, match="unknown Parquet columns"):
        list(iter_knowledge_records(path, namespace="n", default_license="CC0"))
