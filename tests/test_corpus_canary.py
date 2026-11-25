"""`klab corpus build --canary` runs the whole disk pipeline on a sample first."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from recipe_fixtures import Lab, write_recipe

from klab.corpus.document_normalization import EmptyDocumentError


def acquire(tmp_path: Path, monkeypatch, capsys, **recipe_options):
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe", **recipe_options)
    recipe.hub.install(monkeypatch)
    project = str(recipe.project)
    lab.run("corpus", "budget-init", project)
    lab.run("corpus", "acquire", project)
    return lab, project


def builds(lab: Lab) -> list[Path]:
    return [
        path
        for path in lab.work.glob("corpora/*/builds/*")
        if path.name not in {"prepared", "stages"}
    ]


def test_canary_builds_a_bounded_sample_without_touching_the_real_builds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    lab, project = acquire(tmp_path, monkeypatch, capsys)
    result = lab.json(
        "corpus", "build", project, "--offline", "--canary", "--canary-files", "1"
    )
    assert result["canary"] == "passed"
    assert list(lab.work.glob("corpora/*/builds")) == []
    assert Path(result["build"]).is_relative_to(lab.work / "scratch" / "canary")
    ledger = {row["source_id"]: row for row in result["source_ledger"]}
    assert all(row["retained_documents"] > 0 for row in ledger.values())
    # One file per source: fewer documents than the full build keeps.
    sampled = sum(row["documents_built"] for row in ledger.values())
    full = Path(lab.run("corpus", "build", project, "--offline"))
    report = json.loads((full / "report.json").read_text())
    assert sampled < sum(row["documents_built"] for row in report["source_ledger"])
    assert builds(lab) == [full]


def test_canary_surfaces_the_strict_policy_and_empty_document_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    (tmp_path / "strict").mkdir()
    (tmp_path / "empty").mkdir()
    lab, project = acquire(tmp_path / "strict", monkeypatch, capsys, strict_rights=True)
    with pytest.raises(ValueError, match="produced no documents"):
        lab.run("corpus", "build", project, "--offline", "--canary")
    assert builds(lab) == []

    lab, project = acquire(tmp_path / "empty", monkeypatch, capsys, empty_document=True)
    with pytest.raises(
        EmptyDocumentError, match=r"source web_fineweb, file .*row or line .*empty_text"
    ):
        lab.run("corpus", "build", project, "--offline", "--canary")
    assert builds(lab) == []
