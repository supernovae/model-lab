from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from klab.data.chat_recall import cases
from klab.evaluation.capabilities import (
    CapabilityCaseLineage,
    capability_card,
    capability_card_payload,
    check_capability_family_exclusion,
    compare_results,
    describe_capability_card,
    evaluate_capability,
    write_capability_result,
)


def _tokenizer() -> Tokenizer:
    tokenizer = Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    return tokenizer


def _result(
    *,
    score: float,
    memory: str = "none",
    tokens: int = 768,
    seed: int = 42,
    tokenizer_sha: str = "tok",
    path: str = "/a",
) -> dict[str, object]:
    config = {
        "name": "either-name",
        "seed": seed,
        "model": {
            "memory": memory,
            "memory_table_size": 0 if memory == "none" else 31,
            "hidden_dim": 64,
        },
        "dataset": {
            "cache_dir": path,
            "source": "synthetic",
            "synthetic_task": "chat_recall",
        },
        "tokenizer": {"path": f"{path}/tokenizer.json"},
        "training": {"max_steps": 12},
        "logging": {"root_dir": path},
        "checkpoint": {"every_steps": 3},
    }
    passed = score == 1.0
    return {
        "card": "chat-alias-recall-v1",
        "card_digest": "card",
        "valid": True,
        "score": score,
        "results": [{"id": "one", "passed": passed}, {"id": "two", "passed": False}],
        "identity": {
            "checkpoint_sha256": f"checkpoint-{memory}",
            "step": 12,
            "tokens_seen": tokens,
            "config": config,
            "tokenizer_sha256": tokenizer_sha,
            "data_sha256": {"train": "train", "validation": "validation"},
            "source_identity_sha256": "source",
            "runtime": {
                "engine": "pytorch",
                "backend": "cpu",
                "device_name": "CPU",
                "framework_version": "x",
                "os": "Darwin",
                "precision": "fp32",
            },
            "parameter_inventory": {
                "total": 10,
                "trainable": 10,
                "active_per_token": 10,
            },
        },
    }


def test_chat_alias_card_uses_held_out_chat_wording() -> None:
    train = cases("train")
    card = capability_card("chat-alias-recall-v1")

    assert card.version == 1
    assert {item.prompt for item in train}.isdisjoint(
        case.prompt for case in card.cases
    )
    assert {item.expected for item in train} >= {case.expected for case in card.cases}
    assert all("Memory record" not in case.prompt for case in card.cases)
    assert {case.identifier for case in card.cases} == {
        case.identifier for case in cases("test")
    }
    assert all(case.prompt.endswith("Assistant:") for case in card.cases)


def test_described_card_round_trips_and_user_schema_is_strict(tmp_path: Path) -> None:
    payload = describe_capability_card("chat-alias-recall-v1")
    path = tmp_path / "card.json"
    path.write_text(__import__("json").dumps(payload))

    assert capability_card(str(path)).digest == payload["digest"]
    payload["scorer"] = "python:evil"
    payload.pop("digest")
    path.write_text(__import__("json").dumps(payload))
    with pytest.raises(ValueError):
        capability_card(str(path))


def _family_card_payload() -> dict[str, object]:
    payload = describe_capability_card("chat-alias-recall-v1")
    payload.pop("digest")
    payload["format"] = "capability_card_v3"
    payload["version"] = 3
    payload["cases"] = payload["cases"][:2]
    for index, case in enumerate(payload["cases"]):
        case["lineage"] = {
            "split": "test",
            "source_document_family": f"test-doc-{index}",
            "world_id": f"test-world-{index}",
            "template_family": "test-template",
            "parent_content_hashes": [f"{index + 1:064x}"],
        }
    return payload


def test_family_card_round_trips_with_test_lineage(tmp_path: Path) -> None:
    payload = _family_card_payload()
    path = tmp_path / "family-card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    card = capability_card(str(path))
    serialized = capability_card_payload(card)
    assert serialized["format"] == "capability_card_v3"
    assert serialized["digest"] == card.digest
    assert serialized["cases"][0]["lineage"] == payload["cases"][0]["lineage"]
    assert card.cases[0].lineage == CapabilityCaseLineage(
        "test", "test-doc-0", "test-world-0", "test-template", (f"{1:064x}",)
    )
    assert card.scorer == "normalized_full_answer_exact_v1"
    check_capability_family_exclusion(
        card,
        {
            "train": [
                {
                    "split": "train",
                    "source_document_family": "train-doc",
                    "world_id": "train-world",
                    "template_family": "train-template",
                    "parent_content_hashes": [f"{10:064x}"],
                }
            ],
            "validation": [],
        },
    )
    path.write_text(json.dumps(serialized), encoding="utf-8")
    assert capability_card(str(path)) == card


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("split", "train"),
        ("parent_content_hashes", []),
        ("parent_content_hashes", ["not-a-sha256"]),
        ("source_document_family", None),
    ],
)
def test_family_card_rejects_invalid_lineage(
    tmp_path: Path, field: str, value: object
) -> None:
    payload = _family_card_payload()
    payload["cases"][0]["lineage"][field] = value
    if field == "source_document_family":
        payload["cases"][0]["lineage"]["world_id"] = None
    path = tmp_path / "bad-card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        capability_card(str(path))


@pytest.mark.parametrize("field", ["world_id", "parent_content_hashes"])
def test_family_card_rejects_repeated_case_parents(tmp_path: Path, field: str) -> None:
    payload = _family_card_payload()
    payload["cases"][1]["lineage"][field] = payload["cases"][0]["lineage"][field]
    path = tmp_path / "duplicate-card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="world|parent"):
        capability_card(str(path))


@pytest.mark.parametrize(
    "field",
    ["source_document_family", "world_id", "template_family", "parent_content_hashes"],
)
@pytest.mark.parametrize("split", ["train", "validation"])
def test_family_exclusion_rejects_cross_split_identity(
    tmp_path: Path, field: str, split: str
) -> None:
    payload = _family_card_payload()
    path = tmp_path / "family-card.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    card = capability_card(str(path))
    row = {
        "split": split,
        "source_document_family": "other-doc",
        "world_id": "other-world",
        "template_family": "other-template",
        "parent_content_hashes": [f"{10:064x}"],
    }
    row[field] = payload["cases"][0]["lineage"][field]
    with pytest.raises(ValueError, match=split):
        check_capability_family_exclusion(
            card,
            {
                "train": [row] if split == "train" else [],
                "validation": [row] if split == "validation" else [],
            },
        )


def test_v2_card_stays_unmodified_by_v3_serialization() -> None:
    old = capability_card("chat-alias-recall-v1")
    assert capability_card_payload(old) == describe_capability_card(old.name)
    with pytest.raises(ValueError, match="v3"):
        check_capability_family_exclusion(old, {"train": [], "validation": []})
    with pytest.raises(ValueError, match="case schema"):
        capability_card_payload(replace(old, format="capability_card_v3", version=3))


def test_custom_response_budget_cannot_be_silently_replaced(tmp_path: Path) -> None:
    payload = describe_capability_card("chat-alias-recall-v1")
    payload.pop("digest")
    payload["generation"]["max_new_tokens"] = 1000
    path = tmp_path / "long-response.json"
    path.write_text(__import__("json").dumps(payload))
    result = evaluate_capability(
        capability_card(str(path)),
        object(),
        _tokenizer(),
        256,
        __import__("torch").device("cpu"),
    )
    assert not result["valid"]
    assert result["score"] is None


def test_exact_full_answer_scoring_rejects_first_word_junk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = capability_card("chat-alias-recall-v1")
    monkeypatch.setattr(
        "klab.evaluation.capabilities.generate",
        lambda _model, _tokenizer, prompt, *_args, **_kwargs: (
            prompt + " lumen definitely"
        ),
    )
    result = evaluate_capability(
        card, object(), _tokenizer(), 256, __import__("torch").device("cpu")
    )

    assert result["score"] == 0
    assert not any(item["passed"] for item in result["results"])


def test_prompt_overflow_invalidates_evaluation_without_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = capability_card("chat-alias-recall-v1")
    monkeypatch.setattr(
        "klab.evaluation.capabilities.generate",
        lambda *_args, **_kwargs: pytest.fail("generation must not run"),
    )

    result = evaluate_capability(
        card, object(), _tokenizer(), 1, __import__("torch").device("cpu")
    )

    assert result["valid"] is False
    assert result["score"] is None
    assert result["failures"][0]["reason"] == "prompt_and_response_exceed_max_context"


def test_legacy_card_retains_its_explicit_first_word_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    card = capability_card("engram-recall-v1")
    monkeypatch.setattr(
        "klab.evaluation.capabilities.generate",
        lambda _model, _tokenizer, prompt, *_args, **_kwargs: (
            prompt + " lumen trailing text"
        ),
    )

    result = evaluate_capability(
        card, object(), _tokenizer(), 256, __import__("torch").device("cpu")
    )

    assert result["scorer"] == "first_normalized_word_exact_v1"
    assert result["passed"] == 1


def test_compare_requires_matched_identity_but_accepts_relocated_paths() -> None:
    base = _result(score=0.0, path="/old")
    variant = _result(score=1.0, memory="ngram", path="/new")

    comparison = compare_results(base, variant)

    assert comparison["score_delta"] == 1.0
    assert comparison["paired_gains"] == 1
    assert comparison["outcome"] == "supported_observation"
    assert comparison["format"] == "capability_comparison_v2"
    assert "vary_fields" not in comparison
    with pytest.raises(ValueError, match="tokens_seen"):
        compare_results(base, _result(score=1.0, memory="ngram", tokens=767))
    with pytest.raises(ValueError):
        compare_results(base, _result(score=1.0, memory="ngram", seed=7))
    with pytest.raises(ValueError, match="tokenizer_sha256"):
        compare_results(base, _result(score=1.0, memory="ngram", tokenizer_sha="other"))


def test_custom_comparison_requires_exact_declared_model_fields() -> None:
    base = _result(score=0.0)
    variant = _result(score=1.0)
    identity = variant["identity"]
    assert isinstance(identity, dict)
    config = identity["config"]
    assert isinstance(config, dict)
    model = config["model"]
    assert isinstance(model, dict)
    model["hidden_dim"] = 96

    comparison = compare_results(
        base,
        variant,
        vary="custom",
        vary_fields=("model.hidden_dim",),
    )

    assert comparison["format"] == "capability_comparison_v3"
    assert comparison["vary_fields"] == ["model.hidden_dim"]
    with pytest.raises(ValueError, match="outside custom"):
        compare_results(
            base,
            variant,
            vary="custom",
            vary_fields=("model.ffn_dim",),
        )
    with pytest.raises(ValueError, match="include unchanged controls"):
        compare_results(
            base,
            variant,
            vary="custom",
            vary_fields=("model.hidden_dim", "model.ffn_dim"),
        )
    with pytest.raises(ValueError, match="unique dotted"):
        compare_results(
            base,
            variant,
            vary="custom",
            vary_fields=("model..hidden_dim",),
        )
    with pytest.raises(ValueError, match="identical"):
        compare_results(
            base,
            base,
            vary="custom",
            vary_fields=("model.hidden_dim",),
        )
    with pytest.raises(ValueError, match="seed"):
        compare_results(
            base,
            _result(score=1.0, seed=7),
            vary="custom",
            vary_fields=("seed",),
        )


def test_results_are_checkpoint_content_addressed_and_coexist(tmp_path: Path) -> None:
    one = {
        "card": "card",
        "card_digest": "card-digest",
        "checkpoint_sha256": "first",
        "score": 0.0,
    }
    two = {**one, "checkpoint_sha256": "second", "score": 1.0}

    first = write_capability_result(tmp_path, one)
    assert write_capability_result(tmp_path, one) == first
    second = write_capability_result(tmp_path, two)

    assert first != second
    assert first.exists() and second.exists()


def test_memory_comparison_accepts_placement_only_change() -> None:
    base = _result(score=0.0)
    variant = _result(score=0.5)
    for result, placement in ((base, "final"), (variant, "embedding")):
        identity = result["identity"]
        assert isinstance(identity, dict)
        config = identity["config"]
        assert isinstance(config, dict)
        model = config["model"]
        assert isinstance(model, dict)
        model.update(
            {
                "memory": "ngram",
                "memory_table_size": 31,
                "memory_ngram_size": 3,
                "memory_dim": 8,
                "memory_injection": placement,
            }
        )

    comparison = compare_results(base, variant, vary="memory")
    assert set(comparison["differences"]) == {"model.memory_injection"}
    assert (
        comparison["parameter_inventory"]["base"]
        == comparison["parameter_inventory"]["variant"]
    )

    custom = compare_results(
        base,
        variant,
        vary="custom",
        vary_fields=("model.memory_injection",),
    )
    assert custom["format"] == "capability_comparison_v3"
    assert set(custom["differences"]) == {"model.memory_injection"}
