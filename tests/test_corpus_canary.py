"""`klab corpus build --canary` runs the whole disk pipeline on a sample first."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from recipe_fixtures import Lab, commit_git_fixture, write_recipe

from klab.corpus.document_normalization import EmptyDocumentError


def acquire(tmp_path: Path, monkeypatch, capsys, *, edit_recipe=None, **recipe_options):
    lab = Lab(tmp_path, monkeypatch, capsys)
    recipe = write_recipe(tmp_path / "recipe", **recipe_options)
    if edit_recipe is not None:
        edit_recipe(recipe)
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
    def add_format(recipe):
        original = recipe.root / "git/ref_mdn"
        files = {
            path.relative_to(original).as_posix(): path.read_text()
            for path in original.glob("files/en-us/*.md")
        }
        files["files/en-us/zz-extra.txt"] = "An independent example in plain text.\n"
        repo = recipe.root / "git/mixed-reference"
        revision = commit_git_fixture(repo, files)

        def edit(source):
            source.update(canonical_uri=str(repo), revision=revision)
            source["acquisition"]["include"].append("files/en-us/*.txt")

        recipe.edit("ref_mdn", edit)

    lab, project = acquire(tmp_path, monkeypatch, capsys, edit_recipe=add_format)
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
    plan_args = (
        "corpus",
        "plan",
        project,
        "--storage-canary",
        result["build"],
        "--json",
    )
    open_file = Path.open

    def no_payload_reads(path, *args, **options):
        if (
            path.is_relative_to(lab.work / "scratch/canary")
            and path.suffix in {".jsonl", ".sqlite"}
            and path.parent.name != "progress"
        ):
            pytest.fail(f"storage plan opened a canary payload: {path}")
        return open_file(path, *args, **options)

    with monkeypatch.context() as check:
        check.setattr(Path, "open", no_payload_reads)
        plan = lab.json(*plan_args)
    assert plan["status"] == "planned", plan
    storage = plan["storage_inspection"]
    calibration = storage["canary_calibration"]
    assert calibration["sampled_documents"] == sampled
    assert calibration["sampled_input_bytes"] < plan["normalization_source_bytes"]
    assert (
        sum(row["pending_source_bytes"] for row in calibration["coverage"])
        == plan["normalization_source_bytes"]
    )
    assert storage["stage_storage_estimates"]["normalize"]["payload_bytes"] is None
    mirror = next(
        row for row in calibration["coverage"] if row["source_id"] == "ref_mdn"
    )
    assert mirror["covered_files"] == mirror["pending_files"] - 1
    assert calibration["modeled_normalization_payload_bytes"] is None
    assert calibration["modeled_named_peak_by_filesystem"] is None

    complete_sample = lab.json(
        "corpus", "build", project, "--offline", "--canary", "--canary-files", "8"
    )
    complete_args = (*plan_args[:4], complete_sample["build"], "--json")
    complete = lab.json(*complete_args)["storage_inspection"]["canary_calibration"]
    assert (
        complete["modeled_normalization_payload_bytes"]
        == sum(complete["modeled_normalization_members"].values())
        > 0
    )
    assert all(
        item["modeled_payload_bytes"] > 0 for item in complete["stages"].values()
    )
    assert (
        sum(complete["modeled_named_peak_by_filesystem"].values())
        > complete["modeled_normalization_payload_bytes"]
    )
    overlay = lab.json(*complete_args, "--reference-normalized-documents")[
        "storage_inspection"
    ]["canary_calibration"]
    assert (
        overlay["modeled_normalization_members"]
        == complete["modeled_normalization_members"]
    )
    assert overlay["stages"]["views"]["modeled_payload_bytes"] is None
    assert overlay["modeled_named_peak_by_filesystem"] is None

    release_path = Path(project).parent / "release.yaml"
    original_release = release_path.read_bytes()
    release = yaml.safe_load(original_release)
    release["lm"]["training_splits"] = ["train"]
    release_path.write_text(yaml.safe_dump(release))
    blocked = lab.json(*complete_args)
    assert blocked["status"] == "blocked" and "pins and policy" in blocked["reason"]
    release_path.write_bytes(original_release)
    assert not builds(lab)

    full = Path(lab.run("corpus", "build", project, "--offline"))
    assert full.name == plan["build_id"]
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
