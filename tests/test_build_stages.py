"""The disk builder resumes from its last verified stage after any failure."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from corpus_fixtures import neutral_corpus
from recipe_fixtures import Lab, write_recipe

from klab.corpus import build_stages, large_build
from klab.training.manifest import sha256_file


def tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class Calls:
    """Count the stage bodies a build runs and fail one on request."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.counts = dict.fromkeys(("index", "dedup", "emit"), 0)
        self.fail: str | None = None
        for stage, name in (
            ("index", "_index_prepared"),
            ("dedup", "_deduplicate"),
            ("emit", "_emit_build"),
        ):
            monkeypatch.setattr(large_build, name, self.counted(stage, name))

    def counted(self, stage: str, name: str):
        original = getattr(large_build, name)

        def run(*args, **kwargs):
            self.counts[stage] += 1
            if self.fail == stage:
                raise OSError(f"injected failure in {stage}")
            return original(*args, **kwargs)

        return run

    def reset(self) -> dict[str, int]:
        seen, self.counts = self.counts, dict.fromkeys(self.counts, 0)
        return seen


def stage_dirs(work: Path) -> list[str]:
    return sorted(
        f"{path.parent.name}/{path.name}"
        for path in work.glob("corpora/*/builds/stages/*/*")
        if not path.name.startswith(".")
    )


def test_a_build_killed_at_any_stage_resumes_without_redoing_earlier_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe")
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lab.run("corpus", "acquire", project)
    calls = Calls(monkeypatch)

    reference = Path(lab.run("corpus", "build", project, "--offline"))
    expected = tree(reference)
    assert calls.reset() == {"index": 1, "dedup": 1, "emit": 1}
    # A successful build prunes its checkpoints.
    assert stage_dirs(lab.work) == []

    for failing, redone, kept in (
        ("index", {"index": 1, "dedup": 1, "emit": 1}, set()),
        ("dedup", {"index": 0, "dedup": 1, "emit": 1}, {"index"}),
        ("emit", {"index": 0, "dedup": 0, "emit": 1}, {"index", "dedup"}),
    ):
        shutil.rmtree(reference)
        calls.fail = failing
        with pytest.raises(OSError, match="injected failure"):
            lab.run("corpus", "build", project, "--offline")
        assert {item.split("/")[0] for item in stage_dirs(lab.work)} == kept
        calls.fail = None
        calls.reset()
        reference = Path(lab.run("corpus", "build", project, "--offline"))
        assert calls.reset() == redone, failing
        assert tree(reference) == expected, failing
        assert stage_dirs(lab.work) == []


def test_a_stage_that_does_not_verify_is_rebuilt_and_leftovers_are_ignored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe")
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lab.run("corpus", "acquire", project)
    calls = Calls(monkeypatch)
    reference = Path(lab.run("corpus", "build", project, "--offline"))
    expected = tree(reference)
    shutil.rmtree(reference)
    calls.reset()

    calls.fail = "emit"
    with pytest.raises(OSError, match="injected failure"):
        lab.run("corpus", "build", project, "--offline")
    dedup = next(lab.work.glob("corpora/*/builds/stages/dedup/*/dedup.sqlite"))
    stages = dedup.parents[2]
    # A same-size flipped byte, an incomplete directory and an unreceipted
    # directory under an arbitrary name must not become inputs.
    data = bytearray(dedup.read_bytes())
    data[len(data) // 2] ^= 1
    dedup.chmod(0o600)
    dedup.write_bytes(bytes(data))
    (stages / "dedup" / ".incomplete-killed").mkdir()
    (stages / "index" / ("0" * 64)).mkdir()
    calls.fail = None
    calls.reset()
    resumed = Path(lab.run("corpus", "build", project, "--offline"))
    assert calls.reset() == {"index": 0, "dedup": 1, "emit": 1}
    assert tree(resumed) == expected

    shutil.rmtree(resumed)
    lab.run("corpus", "build", project, "--offline", "--keep-stages", "1")
    assert len(stage_dirs(lab.work)) == 2


def test_stage_keys_follow_exactly_their_inputs(tmp_path: Path) -> None:
    project = neutral_corpus(tmp_path)
    base = large_build._stage_inputs(project, [])
    policy = project.release.policy
    flipped = "error" if policy.duplicates.rights_conflict == "allow" else "allow"
    changed = project.model_copy(
        update={
            "release": project.release.model_copy(
                update={
                    "policy": policy.model_copy(
                        update={
                            "duplicates": policy.duplicates.model_copy(
                                update={"rights_conflict": flipped}
                            )
                        }
                    )
                }
            )
        }
    )
    after = large_build._stage_inputs(changed, [])
    # Relaxing a duplicate-only option keeps the index and redoes deduplication.
    assert after["index"] == base["index"]
    assert after["dedup"] != base["dedup"]
    store = build_stages.StageStore(tmp_path / "stages")
    assert store.key(base["index"]) != store.key(base["dedup"])
