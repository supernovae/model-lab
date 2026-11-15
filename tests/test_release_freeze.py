"""Freezing copies each file once, hashing it as it goes, and never links the build."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path

import pytest
from corpus_fixtures import write_prose_project

from klab.corpus import release as release_module
from klab.corpus.acquisition import acquire
from klab.corpus.pipeline import build
from klab.corpus.project import load_project
from klab.corpus.release import freeze, verify_release
from klab.corpus.release_copy import copy_hashed
from klab.training.manifest import sha256_file

TEXTS = {
    "prose_train": [f"training document {n} with some words\n" for n in range(4)],
    "prose_validation": ["validation document\n"],
    "prose_test": ["test document\n"],
}


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[Path, Path]:
    loaded = load_project(write_prose_project(tmp_path, TEXTS) / "corpus.yaml")
    acquire(loaded, tmp_path / "work")
    return build(loaded, tmp_path / "work", offline=True), tmp_path / "work"


def test_frozen_release_is_an_independent_verified_copy(
    prepared: tuple[Path, Path],
) -> None:
    build_dir, work = prepared
    release = freeze(build_dir, work)
    assert verify_release(release)["release_id"] == release.name
    for name in ("documents.jsonl", "lm/train.jsonl"):
        published, original = release / name, build_dir / name
        assert published.read_bytes() == original.read_bytes()
        assert published.stat().st_ino != original.stat().st_ino
        assert published.stat().st_nlink == 1
    assert freeze(build_dir, work) == release


def test_build_file_changed_after_verification_is_never_published(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir, work = prepared
    verified = release_module.verify_build

    def verify_then_alter(*args, **kwargs):
        result = verified(*args, **kwargs)
        with (build_dir / "lm/train.jsonl").open("ab") as stream:
            stream.write(b'{"text": "late edit"}\n')
        return result

    monkeypatch.setattr(release_module, "verify_build", verify_then_alter)
    with pytest.raises(ValueError, match="tampered artifact: lm/train.jsonl"):
        freeze(build_dir, work)
    releases = work / "corpora/prose-fixture/releases"
    assert not list(releases.glob("*")), "no partial release or staging directory"


@pytest.mark.parametrize("filesystem", ["clones", "copies"])
def test_copy_hashes_the_bytes_it_wrote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filesystem: str
) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(os.urandom(3 * 1024**2 + 17))
    if filesystem == "clones":

        def clone(writer_fd: int, request: int, reader_fd: int) -> None:
            os.sendfile(writer_fd, reader_fd, 0, source.stat().st_size)

        monkeypatch.setattr(fcntl, "ioctl", clone)
    else:

        def refuse(*args) -> None:
            raise OSError("no clone support")

        monkeypatch.setattr(fcntl, "ioctl", refuse)
    digest, size = copy_hashed(source, tmp_path / "deep" / "copy.bin")
    assert (digest, size) == (sha256_file(source), source.stat().st_size)
    assert (tmp_path / "deep" / "copy.bin").read_bytes() == source.read_bytes()


def test_freeze_authenticates_each_snapshot_once(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab.corpus import acquisition

    seen = []
    original = acquisition._verify_snapshot_cold
    monkeypatch.setattr(
        acquisition,
        "_verify_snapshot_cold",
        lambda path, **kwargs: (seen.append(path), original(path, **kwargs))[1],
    )
    freeze(*prepared)
    assert len(seen) == len(set(seen)) == len(TEXTS)
