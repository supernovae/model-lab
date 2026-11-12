"""The build path: a local corpus becomes a model with only the Build commands."""

from __future__ import annotations

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
