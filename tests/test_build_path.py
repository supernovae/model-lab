"""The build path: a local corpus becomes a model with only the Build commands."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import yaml

from klab.cli.main import BUILD_COMMANDS, main

EXAMPLE = Path(__file__).resolve().parents[1] / "examples/build-local-text"


def test_local_corpus_builds_a_model_with_only_build_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    shutil.copytree(EXAMPLE, project)
    model = project / "model.yaml"
    config = yaml.safe_load(model.read_text())
    # The shipped declarations name only generic source kinds and settings.
    assert config["dataset"]["source"] == "local_text"
    config["training"].update(max_steps=4, max_tokens=4 * 8 * 64)
    config["optimizer"]["warmup_steps"] = 1
    config["evaluation"]["every_steps"] = 4
    config["checkpoint"]["every_steps"] = 4
    model.write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.chdir(project)

    def klab(*args: str) -> str:
        assert args[0] in BUILD_COMMANDS
        monkeypatch.setattr(sys, "argv", ["klab", *args])
        main()
        return capsys.readouterr().out

    klab("tokenizer", "train", "tokenizer.yaml")
    assert "LIKELY_TO_FIT" in klab("inspect", "model.yaml")
    klab("data", "prepare", "model.yaml")
    klab("train", "model.yaml", "--run-id", "first")
    assert (project / "work/runs/first/checkpoints").is_dir()
    text = klab(
        "generate",
        "first",
        "--runs-dir",
        "work/runs",
        "--prompt",
        "The baker counts",
        "--max-new-tokens",
        "4",
    )
    assert text.startswith("The baker counts")


def test_a_smaller_run_drafts_for_generate_without_changing_its_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    project = tmp_path / "project"
    shutil.copytree(EXAMPLE, project)
    config = yaml.safe_load((project / "model.yaml").read_text())
    config["training"].update(max_steps=2, max_tokens=2 * 8 * 64)
    config["optimizer"]["warmup_steps"] = 1
    config["evaluation"]["every_steps"] = 2
    config["checkpoint"]["every_steps"] = 2
    (project / "model.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    config["model"]["num_layers"] = 1
    (project / "draft.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.chdir(project)

    def klab(*args: str) -> str:
        monkeypatch.setattr(sys, "argv", ["klab", *args])
        main()
        return capsys.readouterr().out

    klab("tokenizer", "train", "tokenizer.yaml")
    klab("train", "model.yaml", "--run-id", "target")
    klab("train", "draft.yaml", "--run-id", "draft")
    request = ["generate", "target", "--runs-dir", "work/runs", "--json"]
    request += ["--prompt", "The baker counts", "--max-new-tokens", "12"]
    plain = json.loads(klab(*request))
    drafted = json.loads(klab(*request, "--draft-run", "draft"))
    assert drafted["response"] == plain["response"]
    report = drafted["speculation"]
    assert report["drafter"] == "draft_model"
    assert report["draft_identity"]["run_id"] == "draft"
    assert report["draft_model_calls"] > 0 and report["wall_seconds"] > 0
    with pytest.raises(ValueError, match="choose one drafter"):
        klab(*request, "--draft-run", "draft", "--prompt-lookup", "2")
