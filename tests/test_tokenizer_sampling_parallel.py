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


def _config(release: Path, max_documents: int) -> TokenizerTrainConfig:
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
                "max_fit_bytes": 1200,
            },
        }
    )


def _serial(config: TokenizerTrainConfig) -> dict:
    """The selection rule written out as one plain loop over the file."""
    caps = tokenizer_sampling.quotas(config)
    release = config.dataset.corpus_release_path
    lineage = {
        json.loads(line)["record_id"]
        for line in (release / "lm/train.lineage.jsonl").read_text().splitlines()
    }
    used, seen_content, chosen, seen = dict.fromkeys(caps, 0), set(), [], 0
    for line in (release / "documents.jsonl").read_text().splitlines():
        row = json.loads(line)
        if (
            row["drop_reason"] is not None
            or row["split"] != "train"
            or row["document_id"] not in lineage
        ):
            continue
        if seen >= config.max_documents:
            break
        seen += 1
        if row["rights"]["training_eligibility"] != "eligible":
            raise ValueError("ineligible")
        stratum = STRATA[row["source_id"]]
        size = len(row["text"].encode())
        if (
            row["content_sha256"] in seen_content
            or used[stratum] + size > caps[stratum]
        ):
            continue
        seen_content.add(row["content_sha256"])
        used[stratum] += size
        chosen.append(row["text"])
    return {"acquired": seen, "texts": chosen, "bytes": sum(used.values())}


def _sampled(config: TokenizerTrainConfig, workers: int) -> dict:
    stats: dict = {}
    with verification_settings(workers=workers):
        texts = list(tokenizer_sampling.documents(config, stats))
    return {"acquired": stats["acquired"], "texts": texts, "bytes": stats["bytes"]}


@pytest.fixture
def split_scan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make a few-KB file take the worker pool, in many tiny byte ranges."""
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
    monkeypatch.setattr(shards, "CHUNK_BYTES", 700)


@pytest.mark.parametrize("max_documents", [10**6, 90])
def test_parallel_scan_selects_exactly_what_the_serial_loop_selects(
    tmp_path: Path, split_scan: None, max_documents: int
) -> None:
    config = _config(_release(tmp_path), max_documents)
    expected = _serial(config)
    assert len(expected["texts"]) > 20
    assert _sampled(config, workers=3) == expected
    assert _sampled(config, workers=1) == expected


def test_ineligible_document_is_refused_only_when_the_scan_reaches_it(
    tmp_path: Path, split_scan: None
) -> None:
    release = _release(tmp_path, ineligible_from=250)
    with pytest.raises(ValueError, match="ineligible TRAIN document"):
        _sampled(_config(release, 10**6), workers=3)
    assert _sampled(_config(release, 60), workers=3)["acquired"] == 60


def test_scan_states_its_workers_and_cores_used(
    tmp_path: Path, split_scan: None
) -> None:
    echo = io.StringIO()
    config = _config(_release(tmp_path), 10**6)
    with verification_settings(workers=3, echo=echo):
        for _ in tokenizer_sampling.documents(config, {}):
            pass
    assert re.search(r"documents scan: \d+ shards, up to 3 workers", echo.getvalue())
    assert "cores used" in echo.getvalue()


def test_killed_scan_resumes_from_its_stored_shards_to_the_same_selection(
    tmp_path: Path, split_scan: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = _config(_release(tmp_path), 10**6)
    expected = _serial(config)
    scratch = tmp_path / "scratch"
    stage = tokenizer_sampling.SCAN_STAGE

    def run(echo: io.StringIO) -> dict:
        stats: dict = {}
        with verification_settings(workers=2, scratch_dir=scratch, echo=echo):
            texts = list(tokenizer_sampling.documents(config, stats))
        return {"acquired": stats["acquired"], "texts": texts, "bytes": stats["bytes"]}

    monkeypatch.setenv("KLAB_FAULT", f"{stage}:3")
    with pytest.raises(RuntimeError, match="KLAB_FAULT"):
        run(io.StringIO())
    monkeypatch.delenv("KLAB_FAULT")
    echo = io.StringIO()
    assert run(echo) == expected
    reused = [line for line in echo.getvalue().splitlines() if "documents scan" in line]
    assert any("reused 3/" in line for line in reused)


def test_every_declared_stage_is_shardable_or_gives_a_reason_and_is_documented() -> (
    None
):
    import klab.data.tokenizer  # noqa: F401  (declares its stage on import)

    with pytest.raises(ValueError, match="say why"):
        shards.Single(" ")
    docs = (Path(__file__).parents[1] / "docs/corpus-preparation.md").read_text()
    for stage in shards.STAGES:
        assert f"| `{stage}` |" in docs, stage
