"""Freezing renames the verified build into the release; it never copies the payload."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from corpus_fixtures import write_prose_project

from klab.corpus import release as release_module
from klab.corpus.acquisition import acquire
from klab.corpus.pipeline import build
from klab.corpus.project import load_project
from klab.corpus.release import consumed_release, freeze, verify_build, verify_release
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


def _payload(root: Path) -> dict[str, tuple[str, int]]:
    """Name, digest and inode of every file except the two identity files."""
    return {
        str(path.relative_to(root)): (sha256_file(path), path.stat().st_ino)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in {"build.json", "manifest.json"}
    }


def test_frozen_release_is_the_build_directory_renamed(
    prepared: tuple[Path, Path],
) -> None:
    build_dir, work = prepared
    before = _payload(build_dir)
    release = freeze(build_dir, work)
    assert verify_release(release)["release_id"] == release.name
    assert _payload(release) == before, "same bytes, same inodes: nothing was copied"
    assert not (release / "build.json").exists()
    assert sorted(path.name for path in build_dir.iterdir()) == [
        "build.json",
        "consumed.json",
    ]
    assert consumed_release(build_dir) == release.name
    assert freeze(build_dir, work) == release
    with pytest.raises(ValueError, match="frozen in place"):
        verify_build(build_dir)
    assert not list(release.parent.glob(".freezing-*"))


def test_build_changed_after_verification_is_never_published(
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
    with pytest.raises(ValueError, match="changed while it was being verified"):
        freeze(build_dir, work)
    releases = work / "corpora/prose-fixture/releases"
    assert not list(releases.glob("*")), "nothing published or staged"
    assert consumed_release(build_dir) is None, "the build was not consumed"


# Each entry makes the n-th durable write or the n-th publishing rename fail.
CRASHES = {
    "after the first rename": ("write", 1),
    "before the consumed marker": ("write", 2),
    "before the manifest": ("write", 3),
    "before the publishing rename": ("rename", 2),
}


@pytest.mark.parametrize("crash", CRASHES)
def test_interrupted_freeze_resumes_to_the_identical_release(
    prepared: tuple[Path, Path],
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
    crash: str,
) -> None:
    from klab import safe_io

    build_dir, work = prepared
    reference_root = tmp_path_factory.mktemp("reference")
    reference = load_project(write_prose_project(reference_root, TEXTS) / "corpus.yaml")
    acquire(reference, reference_root / "work")
    expected = freeze(
        build(reference, reference_root / "work", offline=True), reference_root / "work"
    )
    kind, number = CRASHES[crash]
    calls = {"write": 0, "rename": 0}
    writer, renamer = release_module._write_durably, safe_io.rename_noreplace

    def flaky_write(path, content):
        calls["write"] += 1
        if kind == "write" and calls["write"] == number:
            raise OSError("killed")
        return writer(path, content)

    def flaky_rename(source, target):
        calls["rename"] += 1
        if kind == "rename" and calls["rename"] == number:
            raise OSError("killed")
        return renamer(source, target)

    monkeypatch.setattr(release_module, "_write_durably", flaky_write)
    monkeypatch.setattr(safe_io, "rename_noreplace", flaky_rename)
    with pytest.raises(OSError, match="killed"):
        freeze(build_dir, work)
    monkeypatch.undo()
    releases = work / "corpora/prose-fixture/releases"
    assert [path.name for path in releases.iterdir()] == [f".freezing-{build_dir.name}"]

    release = freeze(build_dir, work)
    assert release.name == expected.name
    assert (release / "manifest.json").read_bytes() == (
        expected / "manifest.json"
    ).read_bytes()
    assert {name: digest for name, (digest, _) in _payload(release).items()} == {
        name: digest for name, (digest, _) in _payload(expected).items()
    }
    assert verify_release(release)["release_id"] == expected.name
    assert consumed_release(build_dir) == release.name


def test_resumed_freeze_verifies_the_staged_files_from_their_bytes(
    prepared: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    build_dir, work = prepared
    writer = release_module._write_durably
    monkeypatch.setattr(
        release_module,
        "_write_durably",
        lambda *args: (_ for _ in ()).throw(OSError("killed")),
    )
    with pytest.raises(OSError, match="killed"):
        freeze(build_dir, work)
    monkeypatch.setattr(release_module, "_write_durably", writer)
    staged = work / "corpora/prose-fixture/releases" / f".freezing-{build_dir.name}"
    with (staged / "lm/train.jsonl").open("ab") as stream:
        stream.write(b'{"text": "edited while stopped"}\n')
    with pytest.raises(ValueError, match="tampered artifact: lm/train.jsonl"):
        freeze(build_dir, work)
    assert not list((work / "corpora/prose-fixture/releases").glob("[0-9a-f]*")), (
        "a tampered stage is never published"
    )


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


def test_release_manifest_names_its_files_and_not_the_build_file(
    prepared: tuple[Path, Path],
) -> None:
    release = freeze(*prepared)
    manifest = json.loads((release / "manifest.json").read_text())
    assert "build.json" not in manifest["files"]
    assert set(manifest["files"]) == {
        str(path.relative_to(release))
        for path in release.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
