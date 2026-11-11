from __future__ import annotations

import copy

import pytest

from klab.evaluation.human_review import (
    create_review_bundle,
    validate_review_judgments,
)

CRITERIA = [
    {
        "id": "quality",
        "description": "Rate answer quality from poor to excellent.",
        "minimum": 1,
        "maximum": 5,
    }
]
RECORDS = [
    {
        "case_id": "case-alpha",
        "prompt": "Explain alpha.",
        "condition": "baseline",
        "response": "Alpha is the first letter.",
        "source_id": "run-baseline-17",
    },
    {
        "case_id": "case-alpha",
        "prompt": "Explain alpha.",
        "condition": "memory",
        "response": "Alpha is a Greek letter often used first.",
        "source_id": "run-memory-17",
    },
    {
        "case_id": "case-beta",
        "prompt": "Explain beta.",
        "condition": "baseline",
        "response": "Beta is the second letter.",
        "source_id": "run-baseline-17",
    },
    {
        "case_id": "case-beta",
        "prompt": "Explain beta.",
        "condition": "memory",
        "response": "Beta is a Greek letter used for a second version.",
        "source_id": "run-memory-17",
    },
    {
        "case_id": "case-gamma",
        "prompt": "Explain gamma.",
        "condition": "baseline",
        "response": "Gamma is the third letter.",
        "source_id": "run-baseline-17",
    },
    {
        "case_id": "case-gamma",
        "prompt": "Explain gamma.",
        "condition": "memory",
        "response": "Gamma is another Greek letter.",
        "source_id": "run-memory-17",
    },
]


def _judgments(bundle: dict[str, object]) -> dict[str, object]:
    return {
        "format": "klab-human-review-judgments",
        "version": 1,
        "bundle_digest": bundle["bundle_digest"],
        "criteria_digest": bundle["criteria_digest"],
        "judgments": [
            {
                "blind_case_id": case["blind_case_id"],
                "ratings": {"quality": {"A": 4, "B": 4}},
                "note": "Both responses are acceptable.",
            }
            for case in bundle["cases"]
        ],
    }


def test_review_bundle_replays_deterministically_and_keeps_reveals_separate() -> None:
    bundle, reveal_map = create_review_bundle(RECORDS, CRITERIA, seed=17)
    replay_bundle, replay_reveal_map = create_review_bundle(
        list(reversed(RECORDS)), CRITERIA, seed=17
    )

    assert bundle == replay_bundle
    assert reveal_map == replay_reveal_map
    assert bundle["reveal_map_digest"] == reveal_map["reveal_map_digest"]
    serialized_bundle = __import__("json").dumps(bundle, sort_keys=True)
    assert "baseline" not in serialized_bundle
    assert "memory" not in serialized_bundle
    assert "run-baseline-17" not in serialized_bundle
    assert "case-alpha" not in serialized_bundle
    assert "condition" not in serialized_bundle
    assert "source_id" not in serialized_bundle
    assert all(
        set(case) == {"blind_case_id", "prompt", "responses"}
        for case in bundle["cases"]
    )
    assert "baseline" in __import__("json").dumps(reveal_map, sort_keys=True)


def test_seed_changes_blind_case_order_or_side_assignment() -> None:
    first, _ = create_review_bundle(RECORDS, CRITERIA, seed=17)
    second, _ = create_review_bundle(RECORDS, CRITERIA, seed=18)

    assert first["cases"] != second["cases"]


def test_complete_tied_human_judgments_are_valid() -> None:
    bundle, _ = create_review_bundle(RECORDS, CRITERIA, seed=17)

    validated = validate_review_judgments(bundle, _judgments(bundle))

    assert validated["judgments"][0]["ratings"]["quality"] == {"A": 4.0, "B": 4.0}


@pytest.mark.parametrize(
    "mutation",
    [
        lambda bundle, judgments: judgments["judgments"].pop(),
        lambda bundle, judgments: judgments.__setitem__("bundle_digest", "0" * 64),
        lambda bundle, judgments: judgments["judgments"][0].__setitem__(
            "case_id", "case-alpha"
        ),
        lambda bundle, judgments: judgments["judgments"][0]["ratings"][
            "quality"
        ].__setitem__("A", 6),
        lambda bundle, judgments: judgments["judgments"].append(
            copy.deepcopy(judgments["judgments"][0])
        ),
    ],
)
def test_incomplete_forged_or_mismatched_judgments_fail(
    mutation: object,
) -> None:
    bundle, _ = create_review_bundle(RECORDS, CRITERIA, seed=17)
    judgments = _judgments(bundle)

    mutation(bundle, judgments)  # type: ignore[operator]

    with pytest.raises((TypeError, ValueError)):
        validate_review_judgments(bundle, judgments)


def test_invalid_pairs_and_nonfinite_values_fail_closed() -> None:
    mismatched_prompt = copy.deepcopy(RECORDS)
    mismatched_prompt[1]["prompt"] = "Different prompt."
    duplicate_condition = copy.deepcopy(RECORDS)
    duplicate_condition[1]["condition"] = "baseline"

    with pytest.raises(ValueError, match="same prompt"):
        create_review_bundle(mismatched_prompt, CRITERIA, seed=17)
    with pytest.raises(ValueError, match="distinct conditions"):
        create_review_bundle(duplicate_condition, CRITERIA, seed=17)
    leaked_identity = copy.deepcopy(RECORDS)
    leaked_identity[0]["response"] = "The BASELINE response says alpha is first."
    unsafe_identifier = copy.deepcopy(RECORDS)
    unsafe_identifier[0]["source_id"] = "../outside"

    with pytest.raises(ValueError, match="leaks an identity"):
        create_review_bundle(leaked_identity, CRITERIA, seed=17)
    with pytest.raises(ValueError, match="safe identifier"):
        create_review_bundle(unsafe_identifier, CRITERIA, seed=17)
    with pytest.raises(ValueError, match="finite"):
        create_review_bundle(
            RECORDS, [{**CRITERIA[0], "maximum": float("nan")}], seed=17
        )
