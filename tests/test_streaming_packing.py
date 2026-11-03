from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pytest
from test_training import config

from sparselab.data.conversations import RenderedConversation
from sparselab.data.encoding import PreparationEncoder, validate_tokenizer_batch_limits
from sparselab.data.local_stories import LICENSE, REVISION
from sparselab.data.packing import _collect, _collect_streaming, prepare_data
from sparselab.data.tokenizer import load_tokenizer


def _stories(monkeypatch: pytest.MonkeyPatch, texts: list[str]) -> None:
    monkeypatch.setattr(
        "sparselab.data.packing._source_documents",
        lambda _config, _split, **_kwargs: (
            RenderedConversation(text, ((0, len(text)),), "all_tokens")
            for text in texts
        ),
    )


def test_streaming_arrays_match_scalar_ids_and_byte_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = [f"The tiny fox found a blue stone number {index}." for index in range(23)]
    _stories(monkeypatch, texts)
    dataset = run.dataset.model_copy(
        update={"train_max_documents": len(texts), "train_max_tokens": 10_000}
    )
    settings = {"byte_table_size": 1024, "byte_ngram_size": 3}
    expected_ids, _, expected_byte, expected_stats = _collect(
        dataset, tokenizer, "train", **settings
    )
    stats = _collect_streaming(
        dataset,
        tokenizer,
        "train",
        tmp_path,
        selected_documents=len(texts),
        **settings,
    )
    for name, values in (
        ("train.npy", expected_ids),
        ("train_byte_addresses.npy", expected_byte),
    ):
        expected = io.BytesIO()
        np.save(expected, values, allow_pickle=False)
        assert (tmp_path / name).read_bytes() == expected.getvalue()
    assert not (tmp_path / "train_supervision.npy").exists()
    assert stats["retained_documents"] == expected_stats["retained_documents"]
    assert stats["truncated_documents"] == 0
    assert not list(tmp_path.glob("*.raw"))


def test_batched_preparation_matches_scalar_across_fresh_rayon_processes(
    tmp_path: Path,
) -> None:
    run = config(tmp_path)
    run = run.model_copy(
        update={
            "model": run.model.model_copy(
                update={
                    "memory": "byte",
                    "memory_table_size": 1024,
                    "memory_dim": 8,
                    "memory_ngram_size": 3,
                }
            )
        }
    )
    paths = {}
    for split, count in (("train", 71), ("validation", 29)):
        path = tmp_path / f"{split}.jsonl"
        records = [
            {"text": f"{split} {n} fox café 🦊\n```python\nx = {n} * 3\n```"}
            for n in range(count)
        ]
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records),
            encoding="utf-8",
        )
        paths[f"{split}_path"] = path
    dataset = run.dataset.model_copy(
        update={
            "source": "local_text",
            "license": "fixture",
            "train_max_documents": 71,
            "validation_max_documents": 29,
            "train_max_tokens": 50_000,
            "validation_max_tokens": 50_000,
            **paths,
        }
    )
    run = run.model_copy(update={"dataset": dataset})
    config_path = tmp_path / "run.json"
    config_path.write_text(run.model_dump_json(), encoding="utf-8")
    script = """
import hashlib, json, sys
from pathlib import Path
from sparselab.config.models import RunConfig
from sparselab.data.packing import prepare_data
from sparselab.data.tokenizer import load_tokenizer
from sparselab.resource_envelope import ResourceEnvelope
run = RunConfig.model_validate_json(Path(sys.argv[1]).read_text())
run = run.model_copy(update={"dataset": run.dataset.model_copy(
    update={"cache_dir": Path(sys.argv[2])})})
prepared = prepare_data(
    run, load_tokenizer(run.tokenizer.path),
    tokenizer_batch_documents=int(sys.argv[3]),
    tokenizer_batch_source_bytes=int(sys.argv[4]),
    resource_envelope=ResourceEnvelope(
        resource_envelope_version=1, max_workers=int(sys.argv[5]),
        preparation_shard_workers=int(sys.argv[6])
    ),
)
print(json.dumps({
    "manifest": prepared.manifest,
    "arrays": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
               for path in prepared.root.glob("*.npy")},
}))
"""
    reference = None
    coordinates = [
        (workers, documents, 1)
        for workers, documents in [(1, 1), *product((1, 2, 4), (16, 64, 256))]
    ]
    coordinates += [(2, 16, 2), (4, 256, 2), (3, 16, 2), (3, 16, 8), (4, 16, 4)]
    for workers, documents, shards in coordinates:
        cache_dir = tmp_path / f"cache-{workers}-{documents}-{shards}"
        cache_dir.mkdir()
        environment = os.environ.copy()
        environment.update(
            HF_HUB_OFFLINE="1",
            HF_DATASETS_OFFLINE="1",
            UV_OFFLINE="1",
            RAYON_NUM_THREADS=str(workers),
            SPARSELAB_WORK_DIR=str(tmp_path),
        )
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                script,
                str(config_path),
                str(cache_dir),
                str(documents),
                "3000",
                str(workers),
                str(shards),
            ],
            capture_output=True,
            text=True,
            env=environment,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        actual = json.loads(result.stdout)
        if reference is None:
            reference = actual
        else:
            assert actual == reference


@pytest.mark.parametrize(
    ("documents", "source_bytes"),
    [(0, 100), (257, 100), (True, 100), (1.0, 100), (1, 0), (1, 8_388_609), (1, False)],
)
def test_invalid_batch_limits_rejected(documents: object, source_bytes: object) -> None:
    with pytest.raises(ValueError, match="tokenizer_batch_"):
        validate_tokenizer_batch_limits(documents, source_bytes)


def test_batch_byte_boundary_and_oversized_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = ["café", "fox", "𝄞", "tiny"]
    _stories(monkeypatch, texts)
    dataset = run.dataset.model_copy(
        update={"train_max_documents": len(texts), "train_max_tokens": 1000}
    )
    expected, _, byte, _ = _collect(dataset, tokenizer, "train")
    root = tmp_path / "bounded"
    root.mkdir()
    _collect_streaming(
        dataset,
        tokenizer,
        "train",
        root,
        selected_documents=len(texts),
        tokenizer_batch_documents=2,
        tokenizer_batch_source_bytes=5,
    )
    assert np.array_equal(np.load(root / "train.npy"), expected)
    assert not (root / "train_supervision.npy").exists()
    assert byte is None
    oversize = tmp_path / "oversize"
    oversize.mkdir()
    with pytest.raises(ValueError, match="exceeds tokenizer_batch_source_bytes"):
        _collect_streaming(
            dataset,
            tokenizer,
            "train",
            oversize,
            selected_documents=len(texts),
            tokenizer_batch_documents=2,
            tokenizer_batch_source_bytes=4,
        )
    assert not (oversize / "train.npy").exists()


def test_preregistered_large_record_fits_explicit_eight_mib_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    run = config(tmp_path)
    text = "x" * 4_943_884
    # Only the complete record is a known token; truncation or segmentation
    # changes the packed IDs rather than merely changing incidental metadata.
    tokenizer = Tokenizer(
        WordLevel({"<unk>": 0, "<eos>": 1, text: 2}, unk_token="<unk>")
    )
    _stories(monkeypatch, [text])
    dataset = run.dataset.model_copy(
        update={"train_max_documents": 1, "train_max_tokens": len(text) + 1}
    )
    root = tmp_path / "eight-mib"
    root.mkdir()
    stats = _collect_streaming(
        dataset,
        tokenizer,
        "train",
        root,
        selected_documents=1,
        tokenizer_batch_documents=256,
        tokenizer_batch_source_bytes=8_388_608,
    )
    assert np.array_equal(np.load(root / "train.npy"), np.array([2, 1], dtype=np.int32))
    assert stats["retained_documents"] == 1
    assert stats["truncated_documents"] == 0
    assert not (root / "train_supervision.npy").exists()
    _stories(monkeypatch, ["x" * 8_388_609])
    oversized = tmp_path / "over-eight-mib"
    oversized.mkdir()
    with pytest.raises(ValueError, match="exceeds tokenizer_batch_source_bytes"):
        _collect_streaming(
            dataset,
            tokenizer,
            "train",
            oversized,
            selected_documents=1,
            tokenizer_batch_source_bytes=8_388_608,
        )
    assert not (oversized / "train.npy").exists()


def test_batch_boundaries_preserve_output_using_real_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = ["café", "fox", "𝄞", "tiny"]
    _stories(monkeypatch, texts)
    dataset = run.dataset.model_copy(
        update={"train_max_documents": len(texts), "train_max_tokens": 1000}
    )
    observed: list[tuple[int, int]] = []
    real_encode = PreparationEncoder.encode

    def observe(self: PreparationEncoder, documents: list[str]):
        observed.append(
            (len(documents), sum(len(s.encode("utf-8")) for s in documents))
        )
        return real_encode(self, documents)

    monkeypatch.setattr(PreparationEncoder, "encode", observe)
    root = tmp_path / "observed"
    root.mkdir()
    _collect_streaming(
        dataset,
        tokenizer,
        "train",
        root,
        selected_documents=len(texts),
        tokenizer_batch_documents=2,
        tokenizer_batch_source_bytes=8,
    )
    assert observed == [(2, 8), (2, 8)]
    expected, _, _, _ = _collect(dataset, tokenizer, "train")
    assert np.array_equal(np.load(root / "train.npy"), expected)
    assert not (root / "train_supervision.npy").exists()


def test_real_child_uses_scalar_encode_for_single_document(
    tmp_path: Path,
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    audit = tmp_path / "encode-calls.jsonl"
    script = """
import json, sys
from pathlib import Path
import sparselab.preparation_encoder as child
from tokenizers import Tokenizer
real_from_file = Tokenizer.from_file
class Recording:
    def __init__(self, real):
        self.real = real
    def encode(self, text, *, add_special_tokens):
        with Path(sys.argv[2]).open("a") as out:
            out.write(json.dumps(["scalar", text, add_special_tokens]) + "\\n")
        return self.real.encode(text, add_special_tokens=add_special_tokens)
    def encode_batch(self, texts, *, add_special_tokens):
        with Path(sys.argv[2]).open("a") as out:
            out.write(json.dumps(["batch", texts, add_special_tokens]) + "\\n")
        return self.real.encode_batch(texts, add_special_tokens=add_special_tokens)
class Loader:
    from_file = staticmethod(lambda path: Recording(real_from_file(path)))
child.Tokenizer = Loader
child.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(run.tokenizer.path), str(audit)],
        input='["café"]\n["fox", "𝄞"]\n',
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "RAYON_NUM_THREADS": "2",
            "TOKENIZERS_PARALLELISM": "true",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "UV_OFFLINE": "1",
        },
        check=True,
    )
    results = [
        json.loads(line) for line in result.stdout.removeprefix("READY\n").splitlines()
    ]
    assert [result[0] for result in results] == [
        tokenizer.encode("café", add_special_tokens=False).ids,
        tokenizer.encode("fox", add_special_tokens=False).ids,
    ]
    assert json.loads(audit.read_text().splitlines()[0]) == ["scalar", "café", False]
    assert json.loads(audit.read_text().splitlines()[1]) == [
        "batch",
        ["fox", "𝄞"],
        False,
    ]


def test_streaming_rejects_partial_story_and_short_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = ["The fox ran to the river.", "The fox returned home safely."]
    _stories(monkeypatch, texts)
    dataset = run.dataset.model_copy(update={"train_max_tokens": 10_000})
    first_ids = tokenizer.encode(texts[0], add_special_tokens=False).ids
    capped = dataset.model_copy(update={"train_max_tokens": len(first_ids) + 1})
    (tmp_path / "capped").mkdir()
    (tmp_path / "short").mkdir()
    with pytest.raises(ValueError, match="token cap would truncate selected story"):
        _collect_streaming(
            capped, tokenizer, "train", tmp_path / "capped", selected_documents=2
        )
    with pytest.raises(ValueError, match="selected distinct stories; required 3"):
        _collect_streaming(
            dataset, tokenizer, "train", tmp_path / "short", selected_documents=3
        )


def test_token_cap_failure_does_not_acquire_subsequent_source_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = ["fox", "fox code " * 40, "must never be acquired"]
    acquired: list[str] = []

    def documents(_cfg, _split, **_kwargs):
        for text in texts:
            acquired.append(text)
            yield RenderedConversation(text, ((0, len(text)),), "all_tokens")

    monkeypatch.setattr("sparselab.data.packing._source_documents", documents)
    first = len(tokenizer.encode(texts[0], add_special_tokens=False).ids)
    dataset = run.dataset.model_copy(
        update={"train_max_documents": 3, "train_max_tokens": first + 2}
    )
    with pytest.raises(ValueError, match="token cap would truncate selected story 2"):
        _collect_streaming(
            dataset,
            tokenizer,
            "train",
            tmp_path,
            selected_documents=3,
            tokenizer_batch_source_bytes=500,
        )
    assert acquired == texts[:2]
    assert not (tmp_path / "train.npy").exists()


def test_source_verified_before_cached_packing_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    train = [f"The fox explored grove number {n}." for n in range(15)]
    validation = [f"The fox rested in meadow number {n}." for n in range(15)]
    snapshot = {
        "splits": {
            "train": {"count": len(train), "sha256": "original"},
            "validation": {"count": len(validation), "sha256": "original"},
        }
    }
    monkeypatch.setattr("sparselab.data.packing.verify_snapshot", lambda _cfg: snapshot)
    monkeypatch.setattr(
        "sparselab.data.packing._source_documents",
        lambda _cfg, split, **_kwargs: (
            RenderedConversation(text, ((0, len(text)),), "all_tokens")
            for text in (train if split == "train" else validation)
        ),
    )
    dataset = run.dataset.model_copy(
        update={
            "source": "local_stories",
            "train_max_documents": 15,
            "validation_max_documents": 15,
            "train_max_tokens": 10_000,
            "validation_max_tokens": 10_000,
        }
    )
    local = run.model_copy(update={"dataset": dataset})
    first = prepare_data(local, tokenizer)
    assert prepare_data(local, tokenizer).root == first.root
    snapshot["splits"]["train"]["sha256"] = "changed"
    assert prepare_data(local, tokenizer).root != first.root

    def corrupted(_cfg):
        raise ValueError("snapshot content digest mismatch")

    monkeypatch.setattr("sparselab.data.packing.verify_snapshot", corrupted)
    with pytest.raises(ValueError, match="snapshot content digest mismatch"):
        prepare_data(local, tokenizer)


def test_cached_local_stories_rejects_corrupted_saved_source(
    tmp_path: Path,
) -> None:
    run = config(tmp_path)
    source = tmp_path / "snapshot"
    source.mkdir()
    splits: dict[str, dict[str, int | str]] = {}
    for split in ("train", "validation"):
        stories = [
            f"{split} story {number} has a happy ending." for number in range(15)
        ]
        path = source / f"{split}.jsonl"
        path.write_bytes(
            b"".join(
                (json.dumps({"ordinal": number, "text": story}) + "\n").encode("utf-8")
                for number, story in enumerate(stories)
            )
        )
        splits[split] = {
            "path": path.name,
            "count": len(stories),
            "text_bytes": sum(len(story.encode("utf-8")) for story in stories),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "content_sha256": hashlib.sha256(
                b"".join(
                    len(story.encode("utf-8")).to_bytes(8, "big")
                    + story.encode("utf-8")
                    for story in stories
                )
            ).hexdigest(),
            "source_records_consumed": len(stories),
            "duplicates": 0,
            "overlap": 0,
            "empty": 0,
        }
    (source / "excluded.jsonl").write_bytes(b"")
    manifest_path = source / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source": "roneneldan/TinyStories",
                "revision": REVISION,
                "license": LICENSE,
                "splits": splits,
                "excluded": {
                    "path": "excluded.jsonl",
                    "count": 0,
                    "sha256": hashlib.sha256(b"").hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    dataset = run.dataset.model_copy(
        update={
            "source": "local_stories",
            "revision": REVISION,
            "license": LICENSE,
            "train_path": source / "train.jsonl",
            "validation_path": source / "validation.jsonl",
            "source_manifest_path": manifest_path,
            "train_max_documents": 15,
            "validation_max_documents": 15,
            "train_max_tokens": 10_000,
            "validation_max_tokens": 10_000,
        }
    )
    local = run.model_copy(update={"dataset": dataset})
    tokenizer = load_tokenizer(run.tokenizer.path)
    cached = prepare_data(local, tokenizer)
    assert cached.manifest["train"]["retained_documents"] == 15
    assert prepare_data(local, tokenizer).root == cached.root
    train_path = source / "train.jsonl"
    train_path.write_bytes(train_path.read_bytes().replace(b"happy", b"quiet", 1))
    with pytest.raises(ValueError, match="snapshot content or digest mismatch"):
        prepare_data(local, tokenizer)


def test_empty_records_resume_in_order_without_encoding_or_overconsuming(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from sparselab.data.preparation_chunks import PreparationChunks

    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = ["", "café 🦊", "", "", "fox", "", "def f(x): return x + 1", "unused"]
    acquired = []

    def documents(*_args, **_kwargs):
        for text in texts:
            acquired.append(text)
            yield RenderedConversation(text, ((0, len(text)),), "all_tokens")

    monkeypatch.setattr("sparselab.data.packing._source_documents", documents)
    encoded = []
    encode = PreparationEncoder.encode

    def record(self, batch):
        encoded.extend(batch)
        return encode(self, batch)

    monkeypatch.setattr(PreparationEncoder, "encode", record)
    root = tmp_path / "empty-records.tmp"
    owner = PreparationChunks(
        root,
        cache_identity={"fixture": "generated"},
        binding={"fixture": "empty"},
        arrays={"ids": np.dtype(np.int32), "byte_addresses": np.dtype(np.int32)},
        record_limit=3,
    )
    with owner:
        stats = _collect_streaming(
            run.dataset,
            tokenizer,
            "train",
            root,
            selected_documents=3,
            byte_table_size=1024,
            byte_ngram_size=3,
            chunks=owner.open_split("train"),
            tokenizer_batch_documents=16,
        )
    expected_texts = [texts[index] for index in (1, 4, 6)]
    assert acquired == texts[:7]
    assert encoded == expected_texts
    assert {
        name: stats[name]
        for name in (
            "acquired_documents",
            "retained_documents",
            "skipped_documents",
            "truncated_documents",
        )
    } == {
        "acquired_documents": 7,
        "retained_documents": 3,
        "skipped_documents": 4,
        "truncated_documents": 0,
    }
    expected_ids = [
        token
        for text in expected_texts
        for token in [
            *tokenizer.encode(text, add_special_tokens=False).ids,
            tokenizer.token_to_id("<eos>"),
        ]
    ]
    assert np.load(root / "train.npy").tolist() == expected_ids
    resumed = PreparationChunks(
        root,
        cache_identity={"fixture": "generated"},
        binding={"fixture": "empty"},
        arrays={"ids": np.dtype(np.int32), "byte_addresses": np.dtype(np.int32)},
        record_limit=3,
    )
    with resumed:
        _collect_streaming(
            run.dataset,
            tokenizer,
            "train",
            root,
            selected_documents=3,
            byte_table_size=1024,
            byte_ngram_size=3,
            chunks=resumed.open_split("train"),
            tokenizer_batch_documents=16,
        )
    assert acquired == texts[:7] * 2
    assert encoded == expected_texts
    assert np.load(root / "train.npy").tolist() == expected_ids


def test_parallel_shards_join_on_failure_and_resume_verified_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from dataclasses import replace
    from threading import Barrier

    from sparselab.data import encoding, preparation_chunks
    from sparselab.resource_envelope import ResourceEnvelope

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))

    run = config(tmp_path)
    paths = {}
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(
            "".join(
                json.dumps({"text": f"{split} fox {n} café"}) + "\n" for n in range(8)
            )
        )
        paths[f"{split}_path"] = path
    run = run.model_copy(
        update={
            "dataset": run.dataset.model_copy(
                update={
                    "source": "local_text",
                    "license": "fixture",
                    **paths,
                    "train_max_documents": 8,
                    "validation_max_documents": 8,
                    "train_max_tokens": 1000,
                    "validation_max_tokens": 1000,
                }
            )
        }
    )
    tokenizer = load_tokenizer(run.tokenizer.path)
    # Pin a tiny two-worker capacity so this also tests concurrency on CI hosts.
    plan = encoding.tokenizer_work_plan(1)
    monkeypatch.setattr(
        "sparselab.data.preparation_workers.tokenizer_work_plan",
        lambda cap: replace(plan, workers=min(2, cap or 2)),
    )
    initialize = preparation_chunks.PreparationChunks.__init__

    def tiny_chunks(self, *args, **kwargs):
        kwargs["record_limit"] = 4
        initialize(self, *args, **kwargs)

    monkeypatch.setattr(preparation_chunks.PreparationChunks, "__init__", tiny_chunks)
    barrier = Barrier(2)
    real_seal = preparation_chunks._after_sealed_chunk

    def interrupt(split, index, receipt):
        if split == "train" and index == 0:
            if receipt.parent.name.startswith("work-"):
                barrier.wait(timeout=10)
            else:
                raise ValueError("fixture interrupt after committed shard")

    monkeypatch.setattr(preparation_chunks, "_after_sealed_chunk", interrupt)
    with pytest.raises(
        (ValueError, RuntimeError), match="fixture interrupt|sibling shard"
    ):
        prepare_data(
            run,
            tokenizer,
            resource_envelope=ResourceEnvelope(
                resource_envelope_version=1, max_workers=2, preparation_shard_workers=2
            ),
            tokenizer_batch_documents=4,
        )
    staged = [
        p
        for p in run.dataset.cache_dir.rglob("staging.json")
        if not p.parent.name.startswith("work-")
    ]
    assert len(staged) == 1
    assert not (staged[0].parent / "manifest.json").exists()
    assert len(list(staged[0].parent.glob("*.receipt.json"))) == 1
    monkeypatch.setattr(preparation_chunks, "_after_sealed_chunk", real_seal)

    encoded = []
    original_encode = PreparationEncoder.encode

    def record_encoding(self, texts):
        encoded.extend(texts)
        return original_encode(self, texts)

    monkeypatch.setattr(PreparationEncoder, "encode", record_encoding)
    prepared = prepare_data(
        run,
        tokenizer,
        resource_envelope=ResourceEnvelope(
            resource_envelope_version=1, max_workers=1, preparation_shard_workers=1
        ),
    )
    for split in ("train", "validation"):
        expected = [
            token
            for n in range(8)
            for token in [
                *tokenizer.encode(
                    f"{split} fox {n} café", add_special_tokens=False
                ).ids,
                tokenizer.token_to_id("<eos>"),
            ]
        ]
        assert np.load(prepared.root / f"{split}.npy").tolist() == expected
    receipt = json.loads(
        prepared.root.with_name(prepared.root.name + ".preparation.json").read_text()
    )
    assert receipt["shard_workers"] == 1
    assert set(receipt["shards"]) == {"train", "validation"}
    assert sorted(encoded) == sorted(
        [f"train fox {n} café" for n in range(4, 8)]
        + [f"validation fox {n} café" for n in range(8)]
    )
    assert receipt["worker_pool"]["effective_workers"] == 1
    assert not list(prepared.root.glob("work-*"))


def test_parallel_token_cap_stops_before_next_record_and_preserves_frontier(
    tmp_path: Path,
):
    from sparselab.data.preparation_chunks import PreparationChunks
    from sparselab.data.preparation_telemetry import PreparationTelemetry
    from sparselab.data.preparation_workers import (
        PreparationWorkers,
        collect_parallel_stream,
    )
    from sparselab.resource_envelope import ResourceEnvelope

    run = config(tmp_path)
    tokenizer = load_tokenizer(run.tokenizer.path)
    texts = [f"The fox explored a different meadow {n}." for n in range(3)]
    visited = []

    def documents():
        for text in texts:
            visited.append(text)
            yield RenderedConversation(text, ((0, len(text)),), "all_tokens")

    eos = tokenizer.token_to_id("<eos>")
    root = tmp_path / "bounded.tmp"
    owner = PreparationChunks(
        root,
        cache_identity={"fixture": "bounded"},
        binding={"fixture": "bounded"},
        arrays={"ids": np.dtype(np.int32)},
    )
    with (
        owner,
        PreparationWorkers(
            tokenizer,
            root,
            envelope=ResourceEnvelope(
                resource_envelope_version=1, max_workers=2, preparation_shard_workers=2
            ),
            parallel_supported=True,
        ) as workers,
    ):
        split = owner.open_split("train")
        with pytest.raises(ValueError, match="token cap would truncate"):
            collect_parallel_stream(
                documents(),
                split,
                workers,
                selected_documents=3,
                max_tokens=len(tokenizer.encode(texts[0], add_special_tokens=False).ids)
                + 2,
                batch_documents=1,
                batch_source_bytes=1024,
                transform=lambda model, ids, text: {"ids": ids + [eos]},
                telemetry=PreparationTelemetry(root),
            )
    assert visited == texts[:2]
    assert split.state["acquired_documents"] == 1
    assert not (root / "train.npy").exists()
    with PreparationChunks(
        root,
        cache_identity={"fixture": "bounded"},
        binding={"fixture": "bounded"},
        arrays={"ids": np.dtype(np.int32)},
    ) as resumed:
        assert resumed.open_split("train").state == split.state
        assert not list(root.glob("work-*"))


def test_worker_scratch_cleanup_preserves_unexpected_files(tmp_path: Path):
    from sparselab.data.preparation_chunks import PreparationChunks

    root = tmp_path / "owned.tmp"
    settings = {
        "cache_identity": {"fixture": "scratch"},
        "binding": {"fixture": "scratch"},
        "arrays": {"ids": np.dtype(np.int32)},
    }
    with (
        PreparationChunks(root, **settings) as owner,
        owner.work_chunks("train", 0) as work,
    ):
        unexpected = work.root / "keep-me.txt"
        unexpected.write_text("unrelated")
    with pytest.raises(ValueError, match="unexpected preparation work entry"):
        PreparationChunks(root, **settings)
    assert unexpected.read_text() == "unrelated"
    assert (unexpected.parent / "staging.json").exists()


def test_worker_adoption_refuses_changed_raw_bytes(tmp_path: Path):
    from sparselab.data.preparation_chunks import PreparationChunks

    with PreparationChunks(
        tmp_path / "changed.tmp",
        cache_identity={"fixture": "changed"},
        binding={"fixture": "changed"},
        arrays={"ids": np.dtype(np.int32)},
    ) as owner:
        destination = owner.open_split("train")
        with owner.work_chunks("train", 0) as scratch:
            work = scratch.open_split("train")
            work.append_record(1, {"ids": [1, 2, 3]})
            work.seal()
        raw = next(work.root.glob("*.ids.raw"))
        before = raw.stat()
        raw.write_bytes(b"\0" * before.st_size)
        os.utime(raw, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000_000))
        with pytest.raises(ValueError, match="worker raw chunk changed"):
            destination.adopt(work, first_acquired_index=1)
        assert destination.state["acquired_documents"] == 0
        assert not list(owner.root.glob("*.receipt.json"))


@pytest.mark.parametrize("model_change", [{"dropout": 0.1}, {"byte_fallback": True}])
def test_unsafe_parallel_tokenizer_uses_reported_serial_fallback(
    tmp_path: Path, monkeypatch, model_change
):
    from tokenizers import Tokenizer

    from sparselab.resource_envelope import ResourceEnvelope

    monkeypatch.setenv("SPARSELAB_WORK_DIR", str(tmp_path / "work"))
    run = config(tmp_path)
    spec = json.loads(load_tokenizer(run.tokenizer.path).to_str())
    spec["model"].update(model_change)
    tokenizer = Tokenizer.from_str(json.dumps(spec))
    paths = {}
    for split in ("train", "validation"):
        path = tmp_path / f"{split}.jsonl"
        path.write_text(json.dumps({"text": f"{split} fox café " * 32}) + "\n")
        paths[f"{split}_path"] = path
    run = run.model_copy(
        update={
            "dataset": run.dataset.model_copy(
                update={
                    "source": "local_text",
                    "license": "fixture",
                    **paths,
                    "train_max_documents": 1,
                    "validation_max_documents": 1,
                    "train_max_tokens": 1000,
                    "validation_max_tokens": 1000,
                }
            )
        }
    )
    prepared = prepare_data(
        run,
        tokenizer,
        resource_envelope=ResourceEnvelope(
            resource_envelope_version=1, max_workers=2, preparation_shard_workers=2
        ),
    )
    receipt = json.loads(
        prepared.root.with_name(prepared.root.name + ".preparation.json").read_text()
    )
    assert receipt["worker_pool"]["requested_workers"] == 2
    assert receipt["worker_pool"]["effective_workers"] == 1
    assert receipt["worker_pool"]["fallback"] == "tokenizer_not_parallel_eligible"
    assert prepared.manifest["train"]["retained_documents"] == 1
