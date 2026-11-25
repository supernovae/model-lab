"""`documents.empty_text` decides between failing the build and skipping one row."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus_fixtures import neutral_corpus

from klab.corpus.document_normalization import EmptyDocumentError
from klab.corpus.prepared_shards import prepare_file
from klab.corpus.rights import resolve_file_rights
from klab.training.manifest import sha256_file

NAME = "sample/10BT/000_00000.parquet.sample.jsonl"


class _Progress:
    def update(self, **_: object) -> None:
        pass

    def record(self, *_: object, **__: object) -> None:
        pass


def _shard(directory: Path) -> Path:
    def row(index: int, text: str) -> dict:
        return {
            "text": text,
            "url": f"https://example.org/page-{index}",
            "id": f"record-{index}",
            "_klab_source": {
                "source_shard_path": "sample/10BT/000_00000.parquet",
                "source_row_index": index,
                "source_row_sha256": f"{index:064x}",
            },
        }

    good = (
        "A public tutorial explains operating system scheduling and reliable "
        "deployments with concrete examples across several paragraphs. " * 2
    )
    path = directory / "shard.jsonl"
    path.write_bytes(
        b"".join(
            json.dumps(item).encode() + b"\n"
            for item in (row(1, good), row(2, ""), row(3, good + " again"))
        )
    )
    return path


def _prepare(tmp_path: Path, empty_text: str) -> Path:
    project = neutral_corpus(tmp_path)
    source = project.sources[0]
    path = _shard(tmp_path)
    return prepare_file(
        build_id="b" * 64,
        prepared_root=tmp_path / "prepared",
        source=source,
        file={"path": NAME, "size": path.stat().st_size, "sha256": sha256_file(path)},
        snapshot_sha="c" * 64,
        source_path=path,
        decision=resolve_file_rights(
            source.rights, NAME, b"", prospective_private_research=True
        ),
        progress=_Progress(),
        normalizer_options=project.release.normalizer,
        document_policy=project.release.policy.documents.model_copy(
            update={"empty_text": empty_text}
        ),
    )


def test_skip_keeps_the_file_and_records_the_empty_row(tmp_path: Path) -> None:
    shard = _prepare(tmp_path, "skip")
    assert len((shard / "docs.jsonl").read_bytes().splitlines()) == 2
    rejected = [json.loads(line) for line in (shard / "rejected.jsonl").open()]
    assert [item["reason"] for item in rejected] == ["empty_or_nontext"]


def test_error_fails_the_build_instead_of_dropping_the_file(tmp_path: Path) -> None:
    with pytest.raises(EmptyDocumentError, match="empty/nontext"):
        _prepare(tmp_path, "error")
