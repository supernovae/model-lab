from __future__ import annotations

import json
from pathlib import Path

import pytest

from klab.config.models import DatasetConfig, TokenizerTrainConfig
from klab.data import datasets, tokenizer
from klab.data.bakeoff import choose_candidate
from klab.data.local_stories import LICENSE, REVISION, snapshot, verify_snapshot


def _config(
    root: Path, train_count: int = 3, validation_count: int = 2
) -> DatasetConfig:
    return DatasetConfig(
        source="local_stories",
        revision=REVISION,
        license=LICENSE,
        cache_dir=root,
        train_path=root / "train.jsonl",
        validation_path=root / "validation.jsonl",
        source_manifest_path=root / "manifest.json",
        train_max_documents=train_count,
        validation_max_documents=validation_count,
        train_max_tokens=64,
        validation_max_tokens=64,
    )


def test_snapshot_deduplicates_and_replays_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[str] = []
    sources = {
        "train": ["one", "one", "éclair", "third"],
        "validation": ["one", "v one", "v one", "v two"],
    }

    def load_dataset(_name: str, *, split: str, **kwargs: object):
        calls.append(split)
        assert kwargs["revision"] == REVISION
        return ({"text": text} for text in sources[split])

    monkeypatch.setattr("datasets.load_dataset", load_dataset)
    root = tmp_path / "source"
    snapshot(root, train_count=3, validation_count=2)
    config = _config(root)
    manifest = verify_snapshot(config)
    assert calls == ["train", "validation"]
    assert [manifest["splits"][name]["count"] for name in ("train", "validation")] == [
        3,
        2,
    ]
    assert manifest["splits"]["train"]["duplicates"] == 1
    assert manifest["splits"]["validation"]["overlap"] == 1
    assert list(datasets.iter_documents(config, "train")) == ["one", "éclair", "third"]
    assert list(datasets.iter_documents(config, "validation")) == ["v one", "v two"]
    assert calls == ["train", "validation"]
    assert (
        json.loads((root / "train.jsonl").read_text().splitlines()[1])["ordinal"] == 2
    )
    with pytest.raises(FileExistsError):
        snapshot(root, train_count=3, validation_count=2)
    (root / "validation.jsonl").write_text('{"ordinal":1,"text":"one"}\n')
    with pytest.raises(ValueError, match="duplicate or cross-split"):
        list(datasets.iter_documents(config, "train"))


def test_snapshot_never_publishes_shortfall(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *_args, **_kwargs: iter(({"text": "same"}, {"text": "same"})),
    )
    with pytest.raises(ValueError, match="stream exhausted"):
        snapshot(tmp_path / "short", train_count=2, validation_count=1)
    assert not (tmp_path / "short").exists()
    assert not list(tmp_path.glob(".snapshot-*"))


def test_snapshot_rejects_full_digest_collision(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from types import SimpleNamespace

    from klab.data import local_stories

    class CollidingHash:
        def __init__(self, _data: bytes = b"") -> None:
            pass

        def update(self, _data: bytes) -> None:
            pass

        def digest(self) -> bytes:
            return bytes(32)

    monkeypatch.setattr(local_stories, "hashlib", SimpleNamespace(sha256=CollidingHash))
    monkeypatch.setattr(
        "datasets.load_dataset",
        lambda *_args, **_kwargs: iter(({"text": "first"}, {"text": "second"})),
    )
    with pytest.raises(ValueError, match="collision"):
        snapshot(tmp_path / "collision", train_count=2, validation_count=1)
    assert not (tmp_path / "collision").exists()


def test_local_stories_rejects_wrong_revision_and_manifest_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="pinned revision"):
        DatasetConfig.model_validate(
            _config(tmp_path, 1, 1).model_dump() | {"revision": "bad"}
        )
    with pytest.raises(ValueError, match="incomplete or invalid"):
        verify_snapshot(_config(tmp_path))


def test_whole_story_byte_limit_and_replay(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    consumed: list[str] = []

    def documents(_config: DatasetConfig, _split: str):
        for story in ("abc", "éé", "never"):
            consumed.append(story)
            yield story

    monkeypatch.setattr(tokenizer, "iter_documents", documents)
    config = TokenizerTrainConfig(
        schema_version=1,
        vocab_size=260,
        min_frequency=1,
        max_documents=3,
        output_dir=tmp_path / "tok",
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path,
            train_max_documents=3,
            validation_max_documents=1,
            train_max_tokens=6,
            validation_max_tokens=10,
        ),
    )
    stats: dict[str, object] = {}
    assert list(tokenizer._bounded_documents(config, stats, whole_documents=True)) == [
        "abc"
    ]
    assert consumed == ["abc", "éé"]
    assert stats["bytes"] == 3
    consumed.clear()
    assert list(tokenizer._bounded_documents(config, {}, whole_documents=False)) == [
        "abc",
        "é",
    ]


def test_selection_threshold_and_ties() -> None:
    candidates = [
        {"vocab_size": size, "tokens_per_byte": ratio, "valid": True}
        for size, ratio in ((8192, 1.021), (12000, 1.02), (16384, 1.0))
    ]
    assert choose_candidate(candidates) == 12000
    candidates[0]["tokens_per_byte"] = 1.02
    assert choose_candidate(candidates) == 8192
    candidates[0]["valid"] = False
    assert choose_candidate(candidates) == 12000


def test_source_pins_keep_legacy_values_and_exports():
    from klab import local_stories_pins

    assert (
        REVISION
        == local_stories_pins.REVISION
        == "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
    )
    assert LICENSE == local_stories_pins.LICENSE == "CDLA-Sharing-1.0"
