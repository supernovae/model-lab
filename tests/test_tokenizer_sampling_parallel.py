"""Tokenizer byte sampling selects the same documents however the scan is split."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from pathlib import Path

import pytest

from klab.canonical import canonical_json
from klab.config.models import TokenizerTrainConfig
from klab.corpus import shards
from klab.corpus.release_verify import verification_settings
from klab.data import tokenizer_sampling

STRATA = {"a": "prose", "b": "code", "c": "prose"}
WEIGHTS = {"prose": 3, "code": 1}


def _release(root: Path, *, ineligible_from: int | None = None) -> Path:
    rng = random.Random(7)
    release = root / "release"
    (release / "lm").mkdir(parents=True)
    shared = "shared duplicate body"
    documents, lineage = [], []
    for number in range(300):
        text = shared if number % 17 == 0 else "x" * rng.randrange(1, 60) + str(number)
        source = rng.choice(sorted(STRATA))
        row = {
            "document_id": f"d{number:04d}",
            "source_id": source,
            "split": rng.choice(["train", "train", "train", "validation"]),
            "drop_reason": "dup" if number % 23 == 5 else None,
            "domains": [STRATA[source]],
            "content_sha256": hashlib.sha256(text.encode()).hexdigest(),
            "rights": {
                "training_eligibility": "ineligible"
                if ineligible_from is not None and number >= ineligible_from
                else "eligible"
            },
            "text": text,
        }
        documents.append(row)
        if number % 11 != 3:
            lineage.append({"record_id": row["document_id"]})
    (release / "documents.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in documents)
    )
    (release / "lm/train.lineage.jsonl").write_bytes(
        b"".join(canonical_json(row) + b"\n" for row in lineage)
    )
    (release / "sources.json").write_text(
        json.dumps(
            [
                {
                    "id": source,
                    "rights_policy": {"training_eligibility": "eligible"},
                    "explicit_training_restriction": "none_found",
                }
                for source in STRATA
            ]
        )
    )
    return release


def _config(
    release: Path,
    max_documents: int = 10**6,
    *,
    max_fit_bytes: int = 1200,
    seed: int = 11,
) -> TokenizerTrainConfig:
    return TokenizerTrainConfig.model_validate(
        {
            "schema_version": 1,
            "vocab_size": 512,
            "min_frequency": 1,
            "max_documents": max_documents,
            "output_dir": str(release.parent / "fit"),
            "dataset": {
                "license": "CC0-1.0",
                "source": "local_text",
                "revision": "a" * 64,
                "cache_dir": str(release.parent / "cache"),
                "corpus_release_path": str(release),
                "corpus_export_path": str(release.parent / "export"),
                "train_path": str(release / "documents.jsonl"),
                "validation_path": str(release / "sources.json"),
                "train_max_documents": 1,
                "validation_max_documents": 1,
                "train_max_tokens": 10**6,
                "validation_max_tokens": 10,
            },
            "sampling": {
                "source_strata": STRATA,
                "byte_weights": WEIGHTS,
                "max_fit_bytes": max_fit_bytes,
                "seed": seed,
                "audit_documents_per_stratum": 3,
            },
        }
    )


def _serial(config: TokenizerTrainConfig) -> dict:
    """The selection rule written out naively: sort every document, fill each quota."""
    caps = tokenizer_sampling.quotas(config)
    release = config.dataset.corpus_release_path
    lineage = {
        json.loads(line)["record_id"]
        for line in (release / "lm/train.lineage.jsonl").read_text().splitlines()
    }
    rows = [
        json.loads(line)
        for line in (release / "documents.jsonl").read_text().splitlines()
    ]
    train = [
        row
        for row in rows
        if row["drop_reason"] is None
        and row["split"] == "train"
        and row["document_id"] in lineage
    ]
    seed = config.sampling.seed
    texts, selected, reserve = [], {}, {}
    for stratum in caps:
        ordered = sorted(
            (row for row in train if STRATA[row["source_id"]] == stratum),
            key=lambda row: (
                int.from_bytes(
                    hashlib.sha256(f"{seed}:{row['content_sha256']}".encode()).digest()[
                        :8
                    ],
                    "big",
                ),
                row["content_sha256"],
                row["document_id"],
            ),
        )
        seen, unique = set(), []
        for row in ordered:
            if row["content_sha256"] not in seen:
                seen.add(row["content_sha256"])
                unique.append(row)
        used = 0
        for index, row in enumerate(unique):
            size = len(row["text"].encode())
            if used + size > caps[stratum]:
                break
            used += size
        else:
            index = len(unique)
        selected[stratum] = [row["document_id"] for row in unique[:index]]
        reserve[stratum] = [row["document_id"] for row in unique[index:]]
        texts += [row["text"] for row in unique[:index]]
    return {"texts": texts, "ids": selected, "reserve": reserve}


def _sampled(config: TokenizerTrainConfig, workers: int) -> dict:
    stats: dict = {}
    with verification_settings(workers=workers):
        texts = list(tokenizer_sampling.documents(config, stats))
        selection = tokenizer_sampling.select_documents(config)
    return {
        "texts": texts,
        "ids": {k: [c.document_id for c in v] for k, v in selection.fit.items()},
        "reserve": {
            k: [c.document_id for c in v] for k, v in selection.reserve.items()
        },
    }


@pytest.fixture
def split_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a few-KB file take the worker pool, in many tiny byte ranges."""
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setattr(shards, "CHUNK_BYTES", 700)


def _expected(config: TokenizerTrainConfig) -> dict:
    expected = _serial(config)
    return {**expected, "reserve": {k: v[:3] for k, v in expected["reserve"].items()}}


def _trim(sampled: dict) -> dict:
    return {**sampled, "reserve": {k: v[:3] for k, v in sampled["reserve"].items()}}


@pytest.mark.parametrize(
    "max_fit_bytes, seed",
    [(1200, 11), (1200, 12), (200, 11), (10**5, 11)],
    ids=["quota", "other-seed", "tight-quota", "quota-above-every-stratum"],
)
def test_parallel_selection_equals_the_naive_sort_and_fill_whatever_the_workers(
    tmp_path: Path, split_scan: None, max_fit_bytes: int, seed: int
) -> None:
    config = _config(_release(tmp_path), max_fit_bytes=max_fit_bytes, seed=seed)
    expected = _expected(config)
    assert all(expected["ids"].values())
    assert _trim(_sampled(config, workers=3)) == expected
    assert _trim(_sampled(config, workers=1)) == expected


def test_a_smaller_quota_selects_a_prefix_of_a_larger_quotas_sample(
    tmp_path: Path, split_scan: None
) -> None:
    release = _release(tmp_path)
    small = _sampled(_config(release, max_fit_bytes=400), workers=2)["ids"]
    large = _sampled(_config(release, max_fit_bytes=1600), workers=2)["ids"]
    for stratum, ids in small.items():
        assert ids and ids == large[stratum][: len(ids)]
        assert len(large[stratum]) > len(ids)


def test_the_sample_fills_each_quota_and_states_how_it_was_chosen(
    tmp_path: Path, split_scan: None
) -> None:
    config = _config(_release(tmp_path))
    stats: dict = {}
    with verification_settings(workers=2):
        list(tokenizer_sampling.documents(config, stats))
    result = stats["sampling"]
    assert (result["order"], result["seed"]) == ("seeded_hash", 11)
    for stratum, cap in result["requested_bytes"].items():
        assert 0 <= result["unfilled_bytes"][stratum] < 60
        assert (
            result["selected_bytes"][stratum] + result["unfilled_bytes"][stratum] == cap
        )
    other: dict = {}
    with verification_settings(workers=2):
        list(
            tokenizer_sampling.documents(
                _config(config.dataset.corpus_release_path, seed=12), other
            )
        )
    assert other["digest"] != stats["digest"]


def test_a_too_tight_threshold_rescans_and_still_selects_exactly(
    tmp_path: Path, split_scan: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(tokenizer_sampling, "OVERSAMPLE", 1)
    config = _config(_release(tmp_path))
    echo = io.StringIO()
    with verification_settings(workers=2, echo=echo):
        sampled = _sampled(config, workers=2)
    assert _trim(sampled) == _expected(config)
    assert echo.getvalue().count("sampling: candidates: 0 shards") == 0
    assert echo.getvalue().count("sampling: candidates:") > 2


def test_a_document_over_the_cap_ends_the_prefix_and_an_ineligible_one_is_refused(
    tmp_path: Path, split_scan: None
) -> None:
    release = _release(tmp_path, ineligible_from=250)
    with pytest.raises(ValueError, match="ineligible TRAIN document"):
        _sampled(_config(release), workers=3)
    with pytest.raises(ValueError, match="needs every TRAIN document"):
        _sampled(_config(_release(tmp_path / "two"), max_documents=60), workers=3)


def test_scan_states_its_workers_and_cores_used(
    tmp_path: Path, split_scan: None
) -> None:
    echo = io.StringIO()
    config = _config(_release(tmp_path))
    with verification_settings(workers=3, echo=echo):
        for _ in tokenizer_sampling.documents(config, {}):
            pass
    assert re.search(r"rank histogram: \d+ shards, up to 3 workers", echo.getvalue())
    assert "cores used" in echo.getvalue()


def test_killed_scan_resumes_from_its_stored_shards_to_the_same_selection(
    tmp_path: Path, split_scan: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(_release(tmp_path))
    expected = _expected(config)
    scratch = tmp_path / "scratch"

    def run(echo: io.StringIO) -> dict:
        with verification_settings(workers=2, scratch_dir=scratch, echo=echo):
            return _trim(_sampled(config, workers=2))

    monkeypatch.setenv("KLAB_FAULT", f"{tokenizer_sampling.CANDIDATE_STAGE}:3")
    with pytest.raises(RuntimeError, match="KLAB_FAULT"):
        run(io.StringIO())
    monkeypatch.delenv("KLAB_FAULT")
    echo = io.StringIO()
    assert run(echo) == expected
    lines = [line for line in echo.getvalue().splitlines() if "candidates" in line]
    assert any("reused 3/" in line for line in lines)
    assert any(
        "rank histogram" in line and "reused 3/" not in line
        for line in echo.getvalue().splitlines()
    )


def test_every_declared_stage_is_shardable_or_gives_a_reason_and_is_documented() -> (
    None
):
    import klab.corpus.declared_family_inventory
    import klab.corpus.export
    import klab.corpus.measurement
    import klab.data.tokenizer  # noqa: F401  (each declares its stage on import)

    with pytest.raises(ValueError, match="say why"):
        shards.Single(" ")
    docs = (Path(__file__).parents[1] / "docs/corpus-preparation.md").read_text()
    for stage in shards.STAGES:
        assert f"| `{stage}` |" in docs, stage
