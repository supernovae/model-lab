"""A rebuild reuses signed snapshot receipts; a changed snapshot byte defeats them."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from recipe_fixtures import Lab, write_recipe

from klab.corpus import acquisition


def test_rebuild_does_not_rehash_unchanged_snapshots_and_a_changed_byte_is_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe")
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lab.run("corpus", "acquire", project)
    cold = {"count": 0}
    original = acquisition._verify_snapshot_cold

    def counted(*args, **kwargs):
        cold["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(acquisition, "_verify_snapshot_cold", counted)

    def build(*flags: str) -> Path:
        cold["count"] = 0
        return Path(lab.run("corpus", "build", project, "--offline", *flags))

    # Acquisition already wrote the receipts the first build reuses.
    first = build()
    assert cold["count"] == 0
    shutil.rmtree(first)
    second = build()
    assert cold["count"] == 0
    shutil.rmtree(second)
    forced = build("--cold")
    assert cold["count"] > 0
    shutil.rmtree(forced)

    # Flip one byte of one snapshot file, keeping its size.
    victim = next(
        path
        for path in sorted(lab.work.glob("corpora/*/snapshots/*/*/files/**/*"))
        if path.is_file() and path.stat().st_size > 0
    )
    data = bytearray(victim.read_bytes())
    data[0] ^= 1
    victim.chmod(0o600)
    victim.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        build()
    assert cold["count"] > 0
