"""Observable sealed-review behavior, including immutable review state."""

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from klab.evaluation.surface_review import (
    complete_surface_review,
    create_surface_bundle,
    open_surface_bundle,
    read_surface_answers,
    record_surface_judgment,
    reveal_surface_review,
    surface_results,
)


def cells(count=36):
    return [
        {
            "candidate_id": f"candidate-{i}",
            "prompt_id": f"prompt-{i // 3}",
            "prompt": f"The fox {i // 3}",
            "category": ("cause_effect", "entity_continuity", "object_continuity")[
                i % 3
            ],
            "decoder": {
                "temperature": 0 if i % 2 else 0.8,
                "top_k": 0,
                "max_new_tokens": 48,
            },
            "rng_seed": i % 2,
            "a": {
                "source": {
                    "kind": "imported_external",
                    "model_uri": "local:model-a",
                    "revision": "r1",
                    "tokenizer_sha256": "tok-a",
                    "output_provenance": "fixture-a",
                },
                "response": f"Alpha {i}",
            },
            "b": {
                "source": {
                    "kind": "imported_external",
                    "model_uri": "local:model-b",
                    "revision": "r2",
                    "tokenizer_sha256": "tok-b",
                    "output_provenance": "fixture-b",
                },
                "response": f"Beta {i}",
            },
        }
        for i in range(count)
    ]


def bundle(tmp_path: Path, count=2):
    path = tmp_path / "sealed"
    create_surface_bundle(
        cells(count),
        profile="full",
        selection_seed=23,
        presentation_seed=42,
        output_dir=path,
    )
    return path


def vote(path: Path, case: dict, choice="A"):
    return record_surface_judgment(
        path,
        case["blind_case_id"],
        {d: choice for d in case["dimensions"]},
        [],
        "2026-09-28T12:00:00Z",
    )


def test_selection_is_balanced_reproducible_and_blind(tmp_path):
    sources = cells()
    manifests = []
    for name, profile, seed in (
        ("a", "quick", 7),
        ("b", "quick", 7),
        ("c", "quick", 8),
        ("d", "standard", 7),
        ("e", "full", 7),
    ):
        path = tmp_path / name
        manifests.append(
            create_surface_bundle(
                sources,
                profile=profile,
                selection_seed=seed,
                presentation_seed=99,
                output_dir=path,
            )
        )
    assert [m["selected_count"] for m in manifests] == [12, 12, 12, 32, 36]
    assert (tmp_path / "a/blind.json").read_bytes() == (
        tmp_path / "b/blind.json"
    ).read_bytes()
    assert (tmp_path / "a/blind.json").read_bytes() != (
        tmp_path / "c/blind.json"
    ).read_bytes()
    assert b"model_uri" not in (tmp_path / "a/blind.json").read_bytes()
    assert "provenance" not in open_surface_bundle(tmp_path / "a")
    with pytest.raises(FileExistsError):
        create_surface_bundle(
            sources,
            profile="quick",
            selection_seed=7,
            presentation_seed=99,
            output_dir=tmp_path / "a",
        )


def test_votes_reload_completion_reveal_and_model_mapping(tmp_path):
    path = bundle(tmp_path)
    cases = open_surface_bundle(path)["blind"]["cases"]
    with pytest.raises(ValueError, match="every case"):
        complete_surface_review(path)
    vote(path, cases[0], "neither")
    assert len(read_surface_answers(path)) == 1
    with pytest.raises(FileExistsError):
        vote(path, cases[0], "B")
    with pytest.raises(ValueError, match="every case"):
        complete_surface_review(path)
    with pytest.raises(ValueError):
        reveal_surface_review(path)
    vote(path, cases[1], "A")
    complete_surface_review(path)
    with pytest.raises(ValueError):
        surface_results(path)
    reveal_surface_review(path)
    results = surface_results(path)
    assert len(results) == 1
    assert sum(sum(v.values()) for v in next(iter(results.values())).values()) == sum(
        len(c["dimensions"]) for c in cases
    )
    with pytest.raises(FileExistsError):
        reveal_surface_review(path)


def test_duplicate_submissions_race_and_tamper(tmp_path):
    path = bundle(tmp_path, 1)
    case = open_surface_bundle(path)["blind"]["cases"][0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [pool.submit(vote, path, case) for _ in range(2)]
    assert sum(f.exception() is None for f in outcomes) == 1
    assert (
        sum(isinstance(f.exception(), (FileExistsError, ValueError)) for f in outcomes)
        == 1
    )
    answer_path = next((path / "answers").iterdir())
    answer = json.loads(answer_path.read_text())
    answer["choices"][case["dimensions"][0]] = "B"
    answer_path.write_text(json.dumps(answer))
    with pytest.raises(ValueError):
        complete_surface_review(path)


def test_changed_sealed_bundle_fails_open(tmp_path):
    path = bundle(tmp_path, 1)
    original = (path / "blind.json").read_bytes()
    (path / "blind.json").write_bytes(original + b" ")
    with pytest.raises(ValueError):
        open_surface_bundle(path)


def test_external_source_without_tokenizer_or_output_provenance_fails(tmp_path):
    candidate = cells(1)[0]
    del candidate["a"]["source"]["tokenizer_sha256"]
    with pytest.raises(ValueError, match="external source requires"):
        create_surface_bundle(
            [candidate],
            profile="full",
            selection_seed=1,
            presentation_seed=2,
            output_dir=tmp_path / "unsealed",
        )
    assert not (tmp_path / "unsealed").exists()


def test_three_sources_yield_all_matched_pairs_and_balanced_quick(tmp_path):
    from collections import Counter
    from itertools import combinations

    candidates = []
    for prompt in range(55):
        category = ("entity_continuity", "object_continuity", "cause_effect")[
            prompt % 3
        ]
        for policy, seed in (("greedy", None), ("sampled", 11), ("sampled", 29)):
            sources = [
                {
                    "kind": "imported_external",
                    "model_uri": f"local:model-{model}",
                    "revision": "r1",
                    "tokenizer_sha256": f"tok-{model}",
                    "output_provenance": "fixture",
                }
                for model in ("old30", "old50", "new30")
            ]
            for left, right in combinations(range(3), 2):
                candidates.append(
                    {
                        "candidate_id": f"{prompt}:{policy}:{seed}:{left}:{right}",
                        "prompt_id": f"prompt-{prompt:02d}",
                        "prompt": f"The fox {prompt}",
                        "category": category,
                        "decoder": {
                            "policy": policy,
                            "temperature": 0 if seed is None else 0.8,
                            "max_new_tokens": 48,
                        },
                        "rng_seed": seed,
                        "a": {
                            "source": sources[left],
                            "response": f"story-{left}-{prompt}-{seed}",
                        },
                        "b": {
                            "source": sources[right],
                            "response": f"story-{right}-{prompt}-{seed}",
                        },
                    }
                )
    assert len(candidates) == 495
    full = tmp_path / "full"
    create_surface_bundle(
        candidates,
        profile="full",
        selection_seed=2026,
        presentation_seed=2027,
        output_dir=full,
    )
    view = open_surface_bundle(full, private=True)
    assert (
        view["manifest"]["eligible_count"] == view["manifest"]["selected_count"] == 495
    )
    assert len({c["candidate_id"] for c in view["provenance"]["cases"]}) == 495
    quick = tmp_path / "quick"
    create_surface_bundle(
        candidates,
        profile="quick",
        selection_seed=2026,
        presentation_seed=2027,
        output_dir=quick,
    )
    selected = open_surface_bundle(quick, private=True)
    categories = Counter(case["category"] for case in selected["blind"]["cases"])
    pairings = Counter(
        tuple(sorted(side["model_uri"] for side in case["sides"].values()))
        for case in selected["provenance"]["cases"]
    )
    assert max(categories.values()) - min(categories.values()) <= 1
    assert max(pairings.values()) - min(pairings.values()) <= 1
    assert all(
        "overall_preference" in case["dimensions"]
        for case in selected["blind"]["cases"]
    )


def test_invalid_dimension_vote_and_changed_answer_are_rejected(tmp_path):
    path = bundle(tmp_path, 1)
    case = open_surface_bundle(path)["blind"]["cases"][0]
    with pytest.raises(ValueError, match="every applicable"):
        record_surface_judgment(
            path,
            case["blind_case_id"],
            {"overall_preference": "A"},
            [],
            "2026-09-28T00:00:00Z",
        )
    with pytest.raises(ValueError, match="every applicable"):
        record_surface_judgment(
            path,
            case["blind_case_id"],
            {d: "uncertain" for d in case["dimensions"]},
            [],
            "2026-09-28T00:00:00Z",
        )
    vote(path, case, "cannot_tell")
    complete_surface_review(path)
    answer = next((path / "answers").glob("*.json"))
    answer.write_bytes(answer.read_bytes() + b" ")
    with pytest.raises(ValueError):
        reveal_surface_review(path)
