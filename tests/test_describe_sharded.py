"""Describe measures view runs in parallel and merges to exactly the serial answer."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path

import pytest
from measurement_oracle import measure_source_rights as serial_source_rights
from measurement_oracle import measure_views as serial_measure_views

from klab.corpus import measurement, shards
from klab.corpus.measurement import (
    measure_source_rights,
    measure_views,
)
from klab.corpus.measurement_inputs import MeasurementTokenizer
from klab.corpus.release import ReleaseMembers
from klab.corpus.release_verify import verification_settings

RELEASE_SPEC = {
    "lm": {"selected": True, "training_splits": ["train", "validation"]},
    "chat": {"selected": True, "training_splits": ["train"]},
}


def _write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _members(root: Path) -> ReleaseMembers:
    return ReleaseMembers(
        root,
        {
            "files": {
                path.relative_to(root).as_posix(): {} for path in root.rglob("*.jsonl")
            }
        },
    )


def _texts(generator: random.Random, count: int) -> list[str]:
    words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    texts = []
    for _ in range(count):
        texts.append(" ".join(generator.choices(words, k=generator.randint(5, 40))))
    # Exact repeats, newline-normalised repeats, and a shared 32-character prefix.
    if count > 8:
        texts[3] = texts[2]
        texts[5] = texts[4].replace(" ", "\r\n", 1)
        texts[7] = "the shared prefix of at least thirty-two characters: one"
        texts[8] = "the shared prefix of at least thirty-two characters: two"
    return texts


@pytest.fixture
def release(tmp_path: Path) -> Path:
    """A release directory with LM and chat views, shared parents and generators."""
    generator = random.Random(11)
    root = tmp_path / "release"
    lineage, ids = [], {}
    for index in range(90):
        record_id = f"record-{index:03d}"
        row = {
            "record_id": record_id,
            "origin": ["primary_source", "synthetic"][index % 2],
            "validation_status": "unverified",
            "shape": {"id": f"shape-{index % 3}", "attributes": {"size": "s"}},
            "domains": ["code", "prose"][: 1 + index % 2],
            "modalities": ["text"],
            "parent_document_ids": [f"parent-{index % 40}"],
            "source_family_ids": [f"family-{index % 4}"],
            "template_family_id": f"template-{index % 2}",
        }
        if index % 7 == 0:
            row["generation_id"] = f"gen-{index}"
        elif index % 7 == 1:
            row["scenario_id"] = f"scenario-{index}"
        elif index % 7 == 2:
            row["semantic_id"] = "semantic"
        if index % 2:
            row["verification"] = {
                "status": "formally_verified" if index == 5 else "schema_validated"
            }
        lineage.append(row)
        ids[index] = record_id
    generator.shuffle(lineage)
    _write(root / "lineage.jsonl", lineage)
    _write(
        root / "generations.jsonl",
        [
            {
                "record_id": f"gen-{i}",
                "generator": {"provider": "p", "model": f"m{i % 2}"},
            }
            for i in range(0, 90, 7)
        ],
    )
    _write(
        root / "scenarios.jsonl",
        [
            {"scenario_id": f"scenario-{i}", "generator_id": "g"}
            for i in range(1, 90, 7)
        ],
    )
    cursor = 0
    _write(root / "lexical/candidates.jsonl", [])
    for view, splits in (
        ("lm", {"train": 40, "validation": 8, "test": 1}),
        ("chat", {"train": 24, "validation": 6, "test": 0}),
    ):
        for split, count in splits.items():
            if split == "test":
                chosen = [ids[81]] if count else []
            else:
                chosen = [ids[cursor + k] for k in range(count)]
                cursor += count
                if (view, split) == ("chat", "train"):
                    chosen[1] = ids[1]
            texts = _texts(generator, count) if count else []
            if view == "lm":
                payload = [{"text": text} for text in texts]
            else:
                payload = [
                    {
                        "messages": [
                            {"role": "user", "content": text},
                            {"role": "assistant", "content": text[::-1] + " ok"},
                        ]
                    }
                    for text in texts
                ]
            _write(root / view / f"{split}.jsonl", payload)
            _write(
                root / view / f"{split}.lineage.jsonl",
                [{"record_id": record_id, "split": split} for record_id in chosen],
            )
    return root


@pytest.fixture(params=["single run", "many runs"])
def runs(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Whole files as one shard, or tiny byte ranges so many shards merge."""
    if request.param == "many runs":
        monkeypatch.setattr(shards, "CHUNK_BYTES", 700)
        monkeypatch.setattr(measurement, "_LINK_STRIDE", 4)
    return request.param


@pytest.fixture(params=["inside the n-gram bound", "beyond the n-gram bound"])
def bound(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param.startswith("beyond"):
        monkeypatch.setattr(measurement, "_NGRAM_CHARS", 900)
    return request.param


@pytest.fixture(scope="module")
def tokenizer(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from tokenizers import Tokenizer, models, pre_tokenizers

    vocab = {
        "[UNK]": 0,
        **{
            word: i + 1
            for i, word in enumerate(
                [
                    "alpha",
                    "beta",
                    "gamma",
                    "delta",
                    "epsilon",
                    "zeta",
                    "eta",
                    "theta",
                    "ok",
                ]
            )
        },
    }
    model = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    model.pre_tokenizer = pre_tokenizers.Whitespace()
    path = tmp_path_factory.mktemp("tokenizer") / "tokenizer.json"
    model.save(str(path))
    return path


@pytest.mark.parametrize("counted", [False, True])
def test_sharded_measurement_equals_the_serial_pass(
    release: Path, runs: str, bound: str, counted: bool, tokenizer: Path
) -> None:
    chosen = tokenizer if counted else None
    expected = serial_measure_views(release, chosen, release_spec=RELEASE_SPEC)
    with verification_settings(workers=1):
        pinned = MeasurementTokenizer.from_path(chosen) if chosen is not None else None
        assert (
            measure_views(_members(release), pinned, release_spec=RELEASE_SPEC)
            == expected
        )
    assert expected["views"]["lm"]["train"]["records"] == 40
    assert bool(
        expected["views"]["chat"]["train"]["concentration"].get("ngram_count_reason")
    ) == bound.startswith("beyond")


def test_worker_processes_give_the_same_answer(
    release: Path, monkeypatch: pytest.MonkeyPatch, tokenizer: Path
) -> None:
    monkeypatch.setattr(shards, "CHUNK_BYTES", 700)
    monkeypatch.setattr(shards, "PARALLEL_MIN_BYTES", 0)
    expected = serial_measure_views(release, tokenizer, release_spec=RELEASE_SPEC)
    with verification_settings(workers=3):
        census = measurement.summarize_release(
            release, release_spec=RELEASE_SPEC, tokenizer=tokenizer
        )
    assert all(census[key] == value for key, value in expected.items())
    inventory = census["inventory"]
    assert inventory["generated_training_records"] == 31
    assert inventory["unverified_generated_training_records"] == 30
    assert inventory["generated_test_shared_parent_ids"] == ["parent-1"]
    assert inventory["generated_heldout_lineage_by_shape"]["shape-1"] == {
        "parent_document_ids": ["parent-1"],
        "source_family_ids": ["family-1"],
        "generator_world_id": [],
        "scenario_family_id": [],
        "template_family_id": ["template-1"],
    }


def test_a_killed_describe_reuses_its_finished_runs(
    release: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(shards, "CHUNK_BYTES", 700)
    cache = tmp_path / "cache"
    expected = serial_measure_views(release, None, release_spec=RELEASE_SPEC)
    scan, measure = measurement._scan_view_links, measurement._measure_shard
    write_range = measurement._write_index_range
    index_ranges_read, links_read, views_read = [], [], []

    def read_index_range(staging, path, start, end):
        index_ranges_read.append((start, end))
        return write_range(staging, path, start, end)

    def read_links(path, *args):
        links_read.append(Path(path))
        return scan(path, *args)

    def read_view(*args):
        views_read.append(args[:5])
        return measure(*args)

    monkeypatch.setattr(measurement, "_write_index_range", read_index_range)
    monkeypatch.setattr(measurement, "_scan_view_links", read_links)
    monkeypatch.setattr(measurement, "_measure_shard", read_view)

    with verification_settings(workers=1):
        with monkeypatch.context() as stop:
            stop.setenv("KLAB_FAULT", f"{measurement.INDEX_STAGE}:2")
            with pytest.raises(RuntimeError, match="KLAB_FAULT"):
                measure_views(_members(release), release_spec=RELEASE_SPEC, cache=cache)
        committed_ranges = set(index_ranges_read)
        assert len(committed_ranges) == 2 and not links_read
        damaged = next(cache.rglob("rows.npy"))
        content = bytearray(damaged.read_bytes())
        content[-1] ^= 1
        damaged.write_bytes(content)
        index_ranges_read.clear()

        with monkeypatch.context() as stop:
            stop.setenv("KLAB_FAULT", f"{measurement.LINK_STAGE}:1")
            with pytest.raises(RuntimeError, match="KLAB_FAULT"):
                measure_views(_members(release), release_spec=RELEASE_SPEC, cache=cache)
        assert len(committed_ranges.intersection(index_ranges_read)) == 1
        assert not list(cache.rglob("rows.npy")), "committed index replaces the ranges"
        index_ranges_read.clear()
        [committed_links] = links_read
        assert not views_read
        links_read.clear()

        with monkeypatch.context() as stop:
            stop.setenv("KLAB_FAULT", f"{measurement.VIEW_STAGE}:3")
            with pytest.raises(RuntimeError, match="KLAB_FAULT"):
                measure_views(_members(release), release_spec=RELEASE_SPEC, cache=cache)
        assert links_read and committed_links not in links_read
        assert len(views_read) == 3
        committed_views = set(views_read)
        links_read.clear()
        views_read.clear()

        assert (
            measure_views(_members(release), release_spec=RELEASE_SPEC, cache=cache)
            == expected
        )
        assert views_read and committed_views.isdisjoint(views_read)
        views_read.clear()
        assert (
            measure_views(_members(release), release_spec=RELEASE_SPEC, cache=cache)
            == expected
        )
        assert not index_ranges_read and not links_read and not views_read

        # A replacement can have the same length and modification time.
        info = committed_links.stat()
        replacement = tmp_path / "replacement.jsonl"
        replacement.write_bytes(committed_links.read_bytes())
        os.utime(replacement, ns=(info.st_atime_ns, info.st_mtime_ns))
        replacement.replace(committed_links)
        assert (
            measure_views(_members(release), release_spec=RELEASE_SPEC, cache=cache)
            == expected
        )
        assert links_read == [committed_links]


def test_a_lineage_link_to_no_record_is_refused(release: Path) -> None:
    links = release / "lm/train.lineage.jsonl"
    rows = [json.loads(line) for line in links.read_text().splitlines()]
    rows[0]["record_id"] = "no-such-record"
    _write(links, rows)
    with (
        verification_settings(workers=1),
        pytest.raises(ValueError, match="reference mismatch"),
    ):
        measure_views(_members(release), release_spec=RELEASE_SPEC)


def test_source_rights_count_each_passage_once_whichever_run_holds_it(
    release: Path, tokenizer: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = random.Random(5)
    rows = []
    for index in range(60):
        words = " ".join(generator.choices(["alpha", "beta", "gamma"], k=6))
        rows.append(
            {
                "split": ["train", "validation"][index % 5 == 0],
                "drop_reason": "duplicate" if index % 11 == 0 else None,
                "rights": {
                    "training_eligibility": ["eligible", "eligible_with_obligations"][
                        index % 3 == 0
                    ]
                },
                "content_sha256": f"{index % 25:064x}",
                "text": words if index % 25 else "alpha beta " + str(index),
            }
        )
    rows[7].pop("rights")
    _write(release / "documents.jsonl", rows)
    monkeypatch.setattr(shards, "CHUNK_BYTES", 900)
    path = release.parent / "local-tokenizer.json"
    path.write_bytes(tokenizer.read_bytes())
    bound = MeasurementTokenizer.from_path(path)
    expected = serial_source_rights(release, path)
    with verification_settings(workers=1):
        assert measure_source_rights(_members(release), bound) == expected
    assert expected["eligible"]["documents"] > 1
    content = json.loads(path.read_bytes())
    content["pre_tokenizer"] = None
    path.write_text(json.dumps(content))
    changed = serial_source_rights(release, path)
    assert changed != expected
    with verification_settings(workers=1):
        assert (
            measure_source_rights(
                _members(release), MeasurementTokenizer.from_path(path)
            )
            == changed
        )
    rows[1]["rights"]["training_eligibility"] = "ineligible"
    rows[1]["split"], rows[1]["drop_reason"] = "train", None
    _write(release / "documents.jsonl", rows)
    with (
        verification_settings(workers=1),
        pytest.raises(ValueError, match="ineligible"),
    ):
        measure_source_rights(
            _members(release), MeasurementTokenizer.from_path(tokenizer)
        )
