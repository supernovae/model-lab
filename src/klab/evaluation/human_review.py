"""Strict, blinded human-review bundle construction and judgment validation.

This module deliberately only prepares and checks human judgments.  It never ranks
conditions, selects models, or produces training feedback.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence

_BUNDLE_FORMAT = "klab-human-review-bundle"
_REVEAL_FORMAT = "klab-human-review-reveal-map"
_JUDGMENTS_FORMAT = "klab-human-review-judgments"
_VERSION = 1
_MAX_RECORDS = 1_000
_MAX_CASES = 500
_MAX_TEXT = 16_384
_MAX_NOTE = 2_000
_MAX_TOTAL_BYTES = 2 * 1024 * 1024
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite(value: object, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{description} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{description} must be finite")
    return number


def _safe_id(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or not _SAFE_ID.fullmatch(value)
        or value in {".", ".."}
    ):
        raise ValueError(f"{description} must be a safe identifier")
    return value


def _text(value: object, description: str, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(
            f"{description} must be nonblank and at most {maximum} characters"
        )
    return value


def _mapping(value: object, description: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{description} must be a string-keyed object")
    return value


def _exact_keys(
    value: Mapping[str, object], expected: set[str], description: str
) -> None:
    if set(value) != expected:
        raise ValueError(f"{description} must contain exactly {sorted(expected)}")


def _check_json_value(value: object, description: str) -> None:
    """Reject values that cannot appear in strict, finite JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{description} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{description} has a non-string key")
            _check_json_value(child, description)
    elif isinstance(value, (list, tuple)):
        for child in value:
            _check_json_value(child, description)
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise TypeError(f"{description} contains a non-JSON value")


def _criteria(raw: object) -> list[dict[str, object]]:
    """Normalize a compact criterion map or explicit criterion list.

    The public bundle always uses the explicit list form.  The map input is accepted
    solely as a convenient caller input: ``{"quality": {"minimum": 1, ...}}``.
    """
    entries: list[object]
    if isinstance(raw, Mapping):
        entries = []
        for identifier, specification in raw.items():
            entry = _mapping(specification, f"criterion {identifier!r}")
            _exact_keys(
                entry,
                {"description", "minimum", "maximum"},
                f"criterion {identifier!r}",
            )
            entries.append({"id": identifier, **entry})
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        entries = list(raw)
    else:
        raise TypeError("criteria must be an object or a nonempty list")
    if not entries or len(entries) > 16:
        raise ValueError("criteria must contain between one and sixteen entries")

    normalized: list[dict[str, object]] = []
    identifiers: set[str] = set()
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"criterion {index}")
        _exact_keys(
            entry, {"id", "description", "minimum", "maximum"}, f"criterion {index}"
        )
        identifier = _safe_id(entry["id"], f"criterion {index} id")
        if identifier in identifiers:
            raise ValueError(f"duplicate criterion id: {identifier}")
        identifiers.add(identifier)
        description = _text(
            entry["description"], f"criterion {identifier} description", 4_000
        )
        minimum = _finite(entry["minimum"], f"criterion {identifier} minimum")
        maximum = _finite(entry["maximum"], f"criterion {identifier} maximum")
        if minimum >= maximum:
            raise ValueError(f"criterion {identifier} must have minimum below maximum")
        normalized.append(
            {
                "id": identifier,
                "description": description,
                "minimum": minimum,
                "maximum": maximum,
            }
        )
    return normalized


def _candidate(raw: object, index: int) -> dict[str, str]:
    record = _mapping(raw, f"record {index}")
    _exact_keys(
        record,
        {"case_id", "prompt", "condition", "response", "source_id"},
        f"record {index}",
    )
    return {
        "case_id": _safe_id(record["case_id"], f"record {index} case_id"),
        "prompt": _text(record["prompt"], f"record {index} prompt"),
        "condition": _safe_id(record["condition"], f"record {index} condition"),
        "response": _text(record["response"], f"record {index} response"),
        "source_id": _safe_id(record["source_id"], f"record {index} source_id"),
    }


def _seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not -(2**63) <= value < 2**63
    ):
        raise ValueError("seed must be a signed 64-bit integer")
    return value


def create_review_bundle(
    records: Sequence[Mapping[str, object]], criteria: object, seed: int
) -> tuple[dict[str, object], dict[str, object]]:
    """Create a blinded review bundle and a separate, identity-bearing reveal map.

    Each input record is one candidate response.  Exactly two records with the same
    ``case_id`` and prompt form a review pair; their conditions must differ.
    """
    if isinstance(records, (str, bytes, bytearray)) or not isinstance(
        records, Sequence
    ):
        raise TypeError("records must be a sequence of candidate objects")
    if not records or len(records) > _MAX_RECORDS:
        raise ValueError(
            f"records must contain between one and {_MAX_RECORDS} candidates"
        )
    normalized_criteria = _criteria(criteria)
    checked_seed = _seed(seed)
    candidates = [_candidate(record, index) for index, record in enumerate(records)]
    _check_json_value(candidates, "records")
    if len(_canonical_json(candidates).encode("utf-8")) > _MAX_TOTAL_BYTES:
        raise ValueError("records exceed the 2 MiB input limit")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for candidate in candidates:
        grouped[candidate["case_id"]].append(candidate)
    if len(grouped) > _MAX_CASES:
        raise ValueError(f"at most {_MAX_CASES} cases are allowed")

    pairs: list[tuple[str, str, list[dict[str, str]]]] = []
    for case_id, pair in grouped.items():
        if len(pair) != 2:
            raise ValueError(
                f"case {case_id} must have exactly two candidate responses"
            )
        if pair[0]["prompt"] != pair[1]["prompt"]:
            raise ValueError(f"case {case_id} candidates must share the same prompt")
        if pair[0]["condition"] == pair[1]["condition"]:
            raise ValueError(f"case {case_id} candidates must have distinct conditions")
        sensitive_values = (
            case_id,
            *(
                value
                for candidate in pair
                for value in (candidate["condition"], candidate["source_id"])
            ),
        )
        if any(
            sensitive.casefold() in text.casefold()
            for candidate in pair
            for sensitive in sensitive_values
            for text in (candidate["prompt"], candidate["response"])
        ):
            raise ValueError(
                f"case {case_id} leaks an identity into rater-visible text"
            )
        pairs.append(
            (
                case_id,
                pair[0]["prompt"],
                sorted(pair, key=lambda item: item["condition"]),
            )
        )

    # Stable input normalization makes output independent of caller record ordering.
    pairs.sort(key=lambda item: item[0])
    generator = random.Random(checked_seed)
    generator.shuffle(pairs)

    blind_cases: list[dict[str, object]] = []
    reveals: list[dict[str, object]] = []
    for position, (case_id, prompt, pair) in enumerate(pairs, start=1):
        blind_case_id = f"review-{position:04d}"
        if generator.getrandbits(1):
            pair.reverse()
        sides = ("A", "B")
        blind_cases.append(
            {
                "blind_case_id": blind_case_id,
                "prompt": prompt,
                "responses": [
                    {"side": side, "response": candidate["response"]}
                    for side, candidate in zip(sides, pair, strict=True)
                ],
            }
        )
        reveals.append(
            {
                "blind_case_id": blind_case_id,
                "case_id": case_id,
                "sides": [
                    {
                        "side": side,
                        "condition": candidate["condition"],
                        "source_id": candidate["source_id"],
                    }
                    for side, candidate in zip(sides, pair, strict=True)
                ],
            }
        )

    reveal_body: dict[str, object] = {
        "format": _REVEAL_FORMAT,
        "version": _VERSION,
        "bundle_seed": checked_seed,
        "cases": reveals,
    }
    reveal_map = {**reveal_body, "reveal_map_digest": _digest(reveal_body)}
    criteria_digest = _digest(normalized_criteria)
    bundle_body: dict[str, object] = {
        "format": _BUNDLE_FORMAT,
        "version": _VERSION,
        "seed": checked_seed,
        "criteria": normalized_criteria,
        "criteria_digest": criteria_digest,
        "reveal_map_digest": reveal_map["reveal_map_digest"],
        "cases": blind_cases,
    }
    bundle = {**bundle_body, "bundle_digest": _digest(bundle_body)}
    return bundle, reveal_map


def _validate_bundle(
    raw: object,
) -> tuple[dict[str, object], dict[str, tuple[float, float]]]:
    bundle = dict(_mapping(raw, "bundle"))
    _check_json_value(bundle, "bundle")
    _exact_keys(
        bundle,
        {
            "format",
            "version",
            "seed",
            "criteria",
            "criteria_digest",
            "reveal_map_digest",
            "cases",
            "bundle_digest",
        },
        "bundle",
    )
    if (
        bundle["format"] != _BUNDLE_FORMAT
        or type(bundle["version"]) is not int
        or bundle["version"] != _VERSION
    ):
        raise ValueError("unsupported review bundle format")
    _seed(bundle["seed"])
    criteria = _criteria(bundle["criteria"])
    if bundle["criteria"] != criteria or bundle["criteria_digest"] != _digest(criteria):
        raise ValueError("bundle criteria digest mismatch")
    if not isinstance(bundle["reveal_map_digest"], str) or not _SHA256.fullmatch(
        bundle["reveal_map_digest"]
    ):
        raise ValueError("bundle reveal map digest is invalid")
    body = {key: value for key, value in bundle.items() if key != "bundle_digest"}
    if bundle["bundle_digest"] != _digest(body):
        raise ValueError("bundle digest mismatch")
    cases = bundle["cases"]
    if not isinstance(cases, list) or not cases or len(cases) > _MAX_CASES:
        raise ValueError("bundle cases are invalid")
    blind_ids: set[str] = set()
    for index, raw_case in enumerate(cases):
        case = _mapping(raw_case, f"bundle case {index}")
        _exact_keys(
            case, {"blind_case_id", "prompt", "responses"}, f"bundle case {index}"
        )
        blind_id = _safe_id(case["blind_case_id"], f"bundle case {index} blind_case_id")
        if blind_id in blind_ids:
            raise ValueError("bundle has duplicate blind case ids")
        blind_ids.add(blind_id)
        _text(case["prompt"], f"bundle case {index} prompt")
        responses = case["responses"]
        if not isinstance(responses, list) or len(responses) != 2:
            raise ValueError("bundle case must have exactly two responses")
        if [
            item.get("side") if isinstance(item, Mapping) else None
            for item in responses
        ] != ["A", "B"]:
            raise ValueError("bundle responses must be ordered A then B")
        for response in responses:
            response_map = _mapping(response, "bundle response")
            _exact_keys(response_map, {"side", "response"}, "bundle response")
            _text(response_map["response"], "bundle response")
    return bundle, {
        item["id"]: (float(item["minimum"]), float(item["maximum"]))
        for item in criteria
    }


def validate_review_judgments(
    bundle: Mapping[str, object], judgments: Mapping[str, object]
) -> dict[str, object]:
    """Validate a complete, blinded judgment file and return its normalized JSON object."""
    checked_bundle, ranges = _validate_bundle(bundle)
    result = dict(_mapping(judgments, "judgments"))
    _check_json_value(result, "judgments")
    _exact_keys(
        result,
        {"format", "version", "bundle_digest", "criteria_digest", "judgments"},
        "judgments",
    )
    if (
        result["format"] != _JUDGMENTS_FORMAT
        or type(result["version"]) is not int
        or result["version"] != _VERSION
    ):
        raise ValueError("unsupported judgment format")
    if (
        result["bundle_digest"] != checked_bundle["bundle_digest"]
        or result["criteria_digest"] != checked_bundle["criteria_digest"]
    ):
        raise ValueError("judgments are not bound to this bundle and criteria")
    entries = result["judgments"]
    if not isinstance(entries, list):
        raise TypeError("judgments must contain a list")
    expected_ids = {
        case["blind_case_id"]
        for case in checked_bundle["cases"]
        if isinstance(case, Mapping)
    }
    seen: set[str] = set()
    normalized_entries: list[dict[str, object]] = []
    for index, raw_entry in enumerate(entries):
        entry = _mapping(raw_entry, f"judgment {index}")
        allowed = {"blind_case_id", "ratings", "note"}
        if not {"blind_case_id", "ratings"} <= set(entry) or set(entry) - allowed:
            raise ValueError(f"judgment {index} has unsupported or missing fields")
        blind_id = _safe_id(entry["blind_case_id"], f"judgment {index} blind_case_id")
        if blind_id not in expected_ids or blind_id in seen:
            raise ValueError("judgments must use each blinded case exactly once")
        seen.add(blind_id)
        ratings = _mapping(entry["ratings"], f"judgment {index} ratings")
        if set(ratings) != set(ranges):
            raise ValueError(f"judgment {index} must rate every criterion exactly once")
        normalized_ratings: dict[str, dict[str, float]] = {}
        for criterion, bounds in ranges.items():
            side_ratings = _mapping(ratings[criterion], f"judgment {index} {criterion}")
            _exact_keys(side_ratings, {"A", "B"}, f"judgment {index} {criterion}")
            normalized_sides: dict[str, float] = {}
            for side in ("A", "B"):
                rating = _finite(
                    side_ratings[side], f"judgment {index} {criterion} {side}"
                )
                if not bounds[0] <= rating <= bounds[1]:
                    raise ValueError(
                        f"judgment {index} {criterion} {side} is outside the allowed range"
                    )
                normalized_sides[side] = rating
            normalized_ratings[criterion] = normalized_sides
        normalized_entry: dict[str, object] = {
            "blind_case_id": blind_id,
            "ratings": normalized_ratings,
        }
        if "note" in entry:
            normalized_entry["note"] = _text(
                entry["note"], f"judgment {index} note", _MAX_NOTE
            )
        normalized_entries.append(normalized_entry)
    if seen != expected_ids:
        raise ValueError("judgments must be complete for every blinded case")
    return {
        "format": _JUDGMENTS_FORMAT,
        "version": _VERSION,
        "bundle_digest": checked_bundle["bundle_digest"],
        "criteria_digest": checked_bundle["criteria_digest"],
        "judgments": normalized_entries,
    }
