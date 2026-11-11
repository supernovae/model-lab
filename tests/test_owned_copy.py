from __future__ import annotations

import errno
import hashlib
import os
from pathlib import Path

import pytest

import klab.owned_copy as copy_module
from klab.data.verification import _owned_file, verify_file


@pytest.fixture
def source(tmp_path: Path) -> Path:
    path = tmp_path / "canonical.bin"
    path.write_bytes(bytes(range(256)) * 10000 + b"final bytes")
    return path


@pytest.fixture
def without_native_clones(monkeypatch: pytest.MonkeyPatch) -> None:
    """Reach transfer fault injection on both reflink and clonefile hosts."""

    def unavailable(*_args: object) -> None:
        raise OSError(errno.EOPNOTSUPP, "native clones disabled for transfer test")

    monkeypatch.setattr(copy_module.fcntl, "ioctl", unavailable)
    monkeypatch.setattr(copy_module, "_clonefile", unavailable)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_private_copy_isolated_and_sealed(source: Path, tmp_path: Path) -> None:
    original = source.stat()
    expected = _digest(source)
    progress: list[int] = []
    target = tmp_path / "worker-private"
    result = copy_module.owned_copy(
        source, target, expected_sha256=expected, progress=progress.append
    )
    assert result.mechanism in {"reflink", "clone", "copy_file_range", "buffered"}
    assert result.logical_bytes == original.st_size
    assert result.proof.path == target.resolve()
    assert result.proof.sha256 == expected
    assert _owned_file(result.proof)
    assert result.proof.fingerprint == copy_module._fingerprint(target)
    assert progress == sorted(set(progress))
    assert progress[-1] == original.st_size
    assert (target.stat().st_dev, target.stat().st_ino) != (
        original.st_dev,
        original.st_ino,
    )
    assert target.stat().st_mode & 0o777 == 0o600
    target.write_bytes(b"worker mutation")
    assert _digest(source) == expected
    assert source.stat().st_mode == original.st_mode
    assert source.stat().st_ino == original.st_ino


def test_existing_destination_and_symlinks_are_rejected(
    source: Path, tmp_path: Path
) -> None:
    existing = tmp_path / "existing"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        copy_module.owned_copy(source, existing)
    assert existing.read_bytes() == b"keep"
    symlink = tmp_path / "link"
    symlink.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        copy_module.owned_copy(source, symlink)
    with pytest.raises(ValueError, match="regular file"):
        copy_module.owned_copy(symlink, tmp_path / "unused")
    parent = tmp_path / "parent-link"
    parent.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="unsafe copy parent"):
        copy_module.owned_copy(source, parent / "unsafe")


def test_creates_private_parent_and_copies_empty_file(tmp_path: Path) -> None:
    source = tmp_path / "empty"
    source.touch(mode=0o644)
    target = tmp_path / "new" / "nested" / "empty"
    result = copy_module.owned_copy(source, target)
    assert result.logical_bytes == 0
    assert result.proof.sha256 == hashlib.sha256(b"").hexdigest()
    assert target.read_bytes() == b""
    assert target.parent.stat().st_mode & 0o777 == 0o700
    assert target.stat().st_mode & 0o777 == 0o600


def test_source_proof_requires_live_fingerprint(source: Path, tmp_path: Path) -> None:
    proof = verify_file(source)
    assert (
        copy_module.owned_copy(source, tmp_path / "valid", proof=proof).proof.sha256
        == proof.sha256
    )
    changed = bytearray(source.read_bytes())
    changed[0] ^= 1
    source.write_bytes(changed)
    with pytest.raises(ValueError, match="current file seal"):
        copy_module.owned_copy(source, tmp_path / "stale", proof=proof)
    assert not (tmp_path / "stale").exists()


@pytest.mark.usefixtures("without_native_clones")
def test_native_partial_failure_resets_before_buffered_fallback(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def broken_native(incoming: int, outgoing: int, count: int) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return os.write(outgoing, os.read(incoming, min(count, 1000)))
        raise OSError(errno.EXDEV, "different file system")

    monkeypatch.setattr(copy_module.os, "copy_file_range", broken_native, raising=False)
    result = copy_module.owned_copy(source, tmp_path / "fallback")
    assert calls == 2
    assert result.mechanism == "buffered"
    assert result.logical_bytes == source.stat().st_size
    assert _digest(result.proof.path) == _digest(source)


@pytest.mark.usefixtures("without_native_clones")
def test_native_copy_file_range_keeps_exact_digest(
    source: Path, tmp_path: Path
) -> None:
    if not hasattr(os, "copy_file_range"):
        pytest.skip("native copy_file_range unavailable")
    result = copy_module.owned_copy(source, tmp_path / "native")
    assert result.mechanism == "copy_file_range"
    assert result.proof.sha256 == _digest(source) == _digest(result.proof.path)


def test_clonefile_private_copy_and_partial_clone_fallback(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    monkeypatch.setattr(copy_module.sys, "platform", "darwin")
    monkeypatch.setattr(copy_module, "_clonefile", shutil.copyfile)
    cloned = copy_module.owned_copy(source, tmp_path / "cloned")
    assert cloned.mechanism == "clone"
    assert cloned.proof.sha256 == _digest(source) == _digest(cloned.proof.path)
    assert cloned.proof.path.stat().st_ino != source.stat().st_ino

    def failed_clone(_source: Path, destination: Path) -> None:
        destination.write_bytes(b"incomplete clone")
        raise OSError(errno.EXDEV, "clone not available")

    monkeypatch.setattr(copy_module, "_clonefile", failed_clone)
    fallback = copy_module.owned_copy(source, tmp_path / "clone-fallback")
    assert fallback.mechanism in {"copy_file_range", "buffered"}
    assert fallback.proof.sha256 == _digest(source) == _digest(fallback.proof.path)


@pytest.mark.usefixtures("without_native_clones")
def test_cross_device_native_failure_uses_buffered_copy(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def cross_device(*_args: object) -> int:
        raise OSError(errno.EXDEV, "cross-device")

    monkeypatch.setattr(copy_module.os, "copy_file_range", cross_device, raising=False)
    result = copy_module.owned_copy(source, tmp_path / "cross-device")
    assert result.mechanism == "buffered"
    assert result.proof.sha256 == _digest(source)


@pytest.mark.usefixtures("without_native_clones")
def test_short_native_result_retries_without_publishing_partial_copy(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        copy_module.os, "copy_file_range", lambda *_args: 0, raising=False
    )
    result = copy_module.owned_copy(source, tmp_path / "retry")
    assert result.mechanism == "buffered"
    assert result.proof.sha256 == _digest(source)


@pytest.mark.usefixtures("without_native_clones")
def test_copy_corruption_and_source_change_never_publish(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(copy_module.os, "copy_file_range", raising=False)
    actual_transfer = copy_module._transfer

    def corrupt(
        incoming: int, outgoing: int, length: int, progress: object, reported: int
    ) -> tuple[str, int]:
        result = actual_transfer(incoming, outgoing, length, progress, reported)
        os.pwrite(outgoing, b"X", 0)
        return result

    monkeypatch.setattr(copy_module, "_transfer", corrupt)

    with pytest.raises(ValueError, match="digest mismatch"):
        copy_module.owned_copy(source, tmp_path / "corrupt")
    assert not (tmp_path / "corrupt").exists()
    monkeypatch.setattr(copy_module, "_transfer", actual_transfer)

    def mutate(
        incoming: int, outgoing: int, length: int, progress: object, reported: int
    ) -> tuple[str, int]:
        result = actual_transfer(incoming, outgoing, length, progress, reported)
        info = source.stat()
        os.utime(source, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
        return result

    monkeypatch.setattr(copy_module, "_transfer", mutate)
    with pytest.raises(ValueError, match="source changed"):
        copy_module.owned_copy(source, tmp_path / "changed")
    assert not (tmp_path / "changed").exists()


@pytest.mark.usefixtures("without_native_clones")
def test_source_truncation_rejects_partial_buffered_copy(
    source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delattr(copy_module.os, "copy_file_range", raising=False)
    actual_transfer = copy_module._transfer

    def truncate_after_authentication(
        incoming: int,
        outgoing: int,
        length: int,
        progress: object,
        reported: int,
    ) -> tuple[str, int]:
        source.write_bytes(b"too short")
        return actual_transfer(incoming, outgoing, length, progress, reported)

    monkeypatch.setattr(copy_module, "_transfer", truncate_after_authentication)

    with pytest.raises(ValueError, match="shortened"):
        copy_module.owned_copy(
            source,
            tmp_path / "partial",
            progress=lambda _: None,
        )
    assert not (tmp_path / "partial").exists()


def test_expected_sha_rejects_wrong_source_before_copy(
    source: Path, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="source digest mismatch"):
        copy_module.owned_copy(source, tmp_path / "wrong", expected_sha256="0" * 64)
    assert not (tmp_path / "wrong").exists()
