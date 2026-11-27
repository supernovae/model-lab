"""Score bounded, labeled samples against retained Corpus annotation evidence."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Literal

from pydantic import Field, StrictBool

from klab.corpus.annotation_kinds import ANNOTATORS
from klab.corpus.artifact_inventory import parse_normalized_partition_id
from klab.corpus.near_duplicates import pair_key
from klab.corpus.privacy_candidates import PRIVACY_FIELDS
from klab.corpus.project import StrictModel
from klab.corpus.train_selection import train_row_selection

_LABEL_LIMIT = 2 * 1024**2
_SCREEN_LIMIT = 64 * 1024**2


class LabeledRow(StrictModel):
    document_id: str
    expected: StrictBool | str | list[str] | None


class AnnotationLabels(StrictModel):
    kind: Literal[
        "detected-language", "privacy-candidates", "source-quality", "template-lines"
    ]
    partition: str
    offset: int = Field(default=0, ge=0)
    options: dict = Field(default_factory=dict)
    quality_rule: dict | None = None
    rows: list[LabeledRow] = Field(min_length=1, max_length=1000)


class LabeledPair(StrictModel):
    units: tuple[str, str]
    expected: Literal["candidate", "template_only", "absent"]


class ScreenLabels(StrictModel):
    report: Path
    sha256: str
    pairs: list[LabeledPair] = Field(min_length=1, max_length=1000)


class Labels(StrictModel):
    annotations: list[AnnotationLabels] = Field(default_factory=list)
    screens: list[ScreenLabels] = Field(default_factory=list)


def _read(path, limit):
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"labeled audit byte limit exceeded: {path}")
    return raw


def _score(actual):
    outcomes = Counter(actual)
    tp, fp, fn, tn = (
        outcomes[key]
        for key in ((True, True), (False, True), (True, False), (False, False))
    )
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
    }


def _annotation(work_root, spec):
    allowed_options = {
        "template-lines": {"sample_documents", "min_support"},
        "detected-language": {"memoize_content"},
    }.get(spec.kind, set())
    if set(spec.options) - allowed_options or (
        spec.quality_rule is not None and spec.kind != "source-quality"
    ):
        raise ValueError("audit options do not apply to this annotation kind")
    if spec.kind == "source-quality" and spec.quality_rule is None:
        raise ValueError("source-quality audit requires quality_rule")
    evidence = ANNOTATORS[spec.kind](
        work_root,
        spec.partition,
        offset=spec.offset,
        limit=len(spec.rows),
        **spec.options,
    )
    if [item.document_id for item in spec.rows] != [
        row["document_id"] for row in evidence["rows"]
    ]:
        raise ValueError(
            "annotation labels do not match verified document IDs and order"
        )
    expected = [item.expected for item in spec.rows]
    rows = evidence["rows"]
    if spec.kind == "detected-language":
        if any(value is not None and not isinstance(value, str) for value in expected):
            raise ValueError("language labels must be strings or null")
        states = Counter(row["status"] for row in rows)
        covered = states["detected"]
        matches = sum(
            row["language"] is not None and row["language"] == label
            for label, row in zip(expected, rows, strict=True)
        )
        by_language = defaultdict(Counter)
        for label, row in zip(expected, rows, strict=True):
            by_language[label or "unlabeled"][row["language"] or "abstained"] += 1
        result = {
            "states": dict(states),
            "coverage": covered / len(rows),
            "accuracy_when_detected": matches / covered if covered else None,
            "predictions_by_expected_language": dict(by_language),
        }
    elif spec.kind == "privacy-candidates":
        if any(
            not isinstance(value, list)
            or len(set(value)) != len(value)
            or set(value) - set(PRIVACY_FIELDS)
            for value in expected
        ):
            raise ValueError("privacy labels must list unique candidate categories")
        result = {
            category: _score(
                (category in label, category in row["candidates"])
                for label, row in zip(expected, rows, strict=True)
            )
            for category in PRIVACY_FIELDS
        }
    else:
        if any(type(value) is not bool for value in expected):
            raise ValueError("quality and template labels must be booleans")
        if spec.kind == "template-lines":
            result = _score(
                zip(expected, (row["template_lines"] > 0 for row in rows), strict=True)
            )
        else:
            source, _ = parse_normalized_partition_id(spec.partition)
            policy = train_row_selection(
                [
                    {
                        "kind": "lm_text",
                        "parameters": {
                            "train_source_quality": {source: spec.quality_rule}
                        },
                    }
                ]
            )
            rule = policy.quality[source]
            states = [
                2
                if rule.field in row["invalid_fields"]
                else int(row[rule.field] is not None)
                for row in rows
            ]
            keeps = (
                rule.decision(state, row[rule.field]) == 0
                for state, row in zip(states, rows, strict=True)
            )
            result = _score(zip(expected, keeps, strict=True))
            result["field_states"] = {
                name: states.count(i)
                for i, name in enumerate(("missing", "reported", "invalid"))
            }
    return {
        "kind": spec.kind,
        "partition": spec.partition,
        "computation_key": evidence["computation_key"],
        "payload_sha256": evidence["payload"]["sha256"],
        "offset": spec.offset,
        "labeled_rows": len(rows),
        "options": evidence.get("options", {}),
        **(
            {"quality_rule": spec.quality_rule} if spec.quality_rule is not None else {}
        ),
        "metrics": result,
    }


def _screen(directory, spec):
    raw = _read(directory / spec.report, _SCREEN_LIMIT)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != spec.sha256:
        raise ValueError("labeled audit screen digest differs")
    report = json.loads(raw)
    truth = {}
    for item in spec.pairs:
        key = tuple(sorted(item.units))
        if key[0] == key[1] or key in truth:
            raise ValueError("screen audit labels repeat a pair or its endpoints")
        truth[key] = item.expected
    classified = {}
    for field, outcome in (
        ("candidates", "candidate"),
        ("template_pairs", "template_only"),
    ):
        for row in report.get(field, ()):
            key = pair_key(row)
            if key in truth:
                if key in classified:
                    raise ValueError(
                        "screen report classifies the same labeled pair twice"
                    )
                classified[key] = outcome
    incomplete = report["candidates_truncated"] or (
        report.get("template_only_count", 0) > 0
        and report.get("template_pairs_truncated") is not False
    )
    confusion = defaultdict(Counter)
    for key, expected in truth.items():
        confusion[expected][
            classified.get(key, "unlisted" if incomplete else "absent")
        ] += 1
    metrics = {}
    for name in ("candidate", "template_only"):
        metrics[name] = _score(
            (expected == name, prediction == name)
            for expected, counts in confusion.items()
            for prediction, count in counts.items()
            if prediction != "unlisted"
            for _ in range(count)
        )
    return {
        "report_sha256": digest,
        "release_id": report.get("release_id"),
        "unit": report.get("unit", "document"),
        "parameters": report["parameters"],
        "labeled_pairs": len(truth),
        "predictions_by_expected_class": dict(confusion),
        "unlisted_pairs": sum(counts["unlisted"] for counts in confusion.values()),
        "metrics": metrics,
    }


def audit_evidence(work_root: Path, path: Path) -> dict:
    """Report agreement with pinned annotation and duplicate-screen labels."""
    raw = _read(path, _LABEL_LIMIT)
    labels = Labels.model_validate_json(raw)
    count = sum(len(spec.rows) for spec in labels.annotations) + sum(
        len(spec.pairs) for spec in labels.screens
    )
    if not 1 <= count <= 1000:
        raise ValueError("evidence audit requires 1 to 1000 labeled examples")
    return {
        "labels_sha256": hashlib.sha256(raw).hexdigest(),
        "annotations": [_annotation(work_root, spec) for spec in labels.annotations],
        "screens": [_screen(path.parent, spec) for spec in labels.screens],
    }
