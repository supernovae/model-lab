"""Public proof-store key availability preserves filesystem trust semantics."""

import pytest

from sparselab.verification_proofs import ProofStore


def test_key_availability_is_read_only_until_creation_is_requested(
    tmp_path, monkeypatch
):
    config = tmp_path / "settings"
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config))
    store = ProofStore(tmp_path)
    assert store.key_available() is False
    assert not config.exists()
    assert store.key_available(create=True) is True
    assert store.key_path.is_file()
    before = store.key_path.stat()
    assert store.key_available() is True
    after = store.key_path.stat()
    assert (after.st_ino, after.st_mtime_ns, after.st_ctime_ns) == (
        before.st_ino,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    assert not store.directory.exists()


@pytest.mark.parametrize("mode", [0o755, 0o777])
def test_key_readability_does_not_imply_publication_availability(
    tmp_path, monkeypatch, mode
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    store = ProofStore(tmp_path)
    assert store.key_available(create=True) is True
    directory = store.key_path.parent
    directory.chmod(mode)
    # World-writable parents also make reading unsafe; 0755 remains readable.
    assert store.key_available() is (mode == 0o755)
    assert store.key_available(create=True) is False
    assert directory.stat().st_mode & 0o777 == mode
    directory.chmod(0o700)
    assert store.key_available(create=True) is True


def test_key_availability_refuses_a_symlinked_config_root(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "settings"
    link.symlink_to(target, target_is_directory=True)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(link))
    store = ProofStore(tmp_path)
    assert store.key_available() is False
    assert store.key_available(create=True) is False
    assert list(target.iterdir()) == []


@pytest.mark.parametrize("mode", [0o700, 0o755])
def test_inspection_cache_preserves_key_publication_refusal(
    tmp_path, monkeypatch, mode
):
    from sparselab.corpus.admission_inspection_cache import InspectionCache

    work = tmp_path / "work"
    work.mkdir(mode=0o700)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    store = ProofStore(work)
    assert store.key_available(create=True) is True
    store.key_path.parent.chmod(mode)
    cache = InspectionCache(work, proof_store=store)
    assert (cache.proofs is store) is (mode == 0o700)
    assert cache.unavailable_reason == (
        None if mode == 0o700 else "verification_key_unavailable"
    )
    assert store.key_path.parent.stat().st_mode & 0o777 == mode


def test_key_availability_refuses_malformed_existing_key_without_replacing_it(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "settings"))
    store = ProofStore(tmp_path)
    assert store.key_available(create=True) is True
    malformed = b"x" * 31
    store.key_path.write_bytes(malformed)
    assert store.key_available() is False
    assert store.key_available(create=True) is False
    assert store.key_path.read_bytes() == malformed
