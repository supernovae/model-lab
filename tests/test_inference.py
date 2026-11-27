from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import torch
from test_training import config

from klab.cli.main import main
from klab.evaluation.inference import load_run
from klab.training.trainer import train


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("inference")
    original = config(root)
    small = original.model_copy(
        update={
            "training": original.training.model_copy(update={"max_steps": 2}),
            "optimizer": original.optimizer.model_copy(update={"warmup_steps": 0}),
        }
    )
    train(small, run_id="original")
    return small


def test_relocated_run_does_not_need_external_tokenizer(trained_run, tmp_path):
    original = load_run("original", trained_run.logging.root_dir)
    shutil.copytree(original.run, tmp_path / "moved")
    external = trained_run.tokenizer.path
    original_bytes = external.read_bytes()
    external.write_text("not a tokenizer")
    try:
        moved = load_run("moved", tmp_path)
        tokens = torch.tensor([moved.tokenizer.encode("hello").ids])
        with torch.no_grad():
            assert torch.equal(original.model(tokens), moved.model(tokens))
        assert (
            original.identity["checkpoint_sha256"]
            == moved.identity["checkpoint_sha256"]
        )
    finally:
        external.write_bytes(original_bytes)


def test_inference_rejects_tampered_tokenizer(trained_run, tmp_path):
    shutil.copytree(trained_run.logging.root_dir / "original", tmp_path / "corrupt")
    (tmp_path / "corrupt/tokenizer.json").write_text("{}")
    with pytest.raises(ValueError):
        load_run("corrupt", tmp_path)


def test_preview_skips_hashing_the_run_data_and_nothing_else(trained_run, tmp_path):
    shutil.copytree(trained_run.logging.root_dir / "original", tmp_path / "preview")
    run = tmp_path / "preview"
    assert load_run("preview", tmp_path).identity["run_data_hashed"] is True
    array = run / "data" / "train.npy"
    intact = array.read_bytes()
    array.write_bytes(intact[:-1] + bytes([intact[-1] ^ 1]))  # same size, one bit

    with pytest.raises(ValueError):  # a normal load hashes it
        load_run("preview", tmp_path)
    preview = load_run("preview", tmp_path, hash_run_data=False)
    assert preview.identity["run_data_hashed"] is False
    with pytest.raises(ValueError, match="without hashing its data"):
        preview.validation_dataset()  # such a load is not evidence for evaluation

    array.write_bytes(intact + b"x")  # a size change is still caught
    with pytest.raises(ValueError):
        load_run("preview", tmp_path, hash_run_data=False)
    array.write_bytes(intact)
    (run / "tokenizer.json").write_text("{}")  # so is everything the preview reads
    with pytest.raises(ValueError):
        load_run("preview", tmp_path, hash_run_data=False)


def test_selected_checkpoint_identity_is_not_latest(trained_run, tmp_path):
    shutil.copytree(trained_run.logging.root_dir / "original", tmp_path / "history")
    generations = sorted((tmp_path / "history/checkpoints").glob("step_*"))
    initial = load_run("history", tmp_path, str(generations[0]))
    final = load_run("history", tmp_path)
    assert initial.identity["step"] == 0
    assert final.identity["step"] == 2
    assert initial.identity["checkpoint_sha256"] != final.identity["checkpoint_sha256"]
    pointer = tmp_path / "history/checkpoints/latest.json"
    stale = json.loads(pointer.read_text())
    stale["relative_path"] = generations[0].name
    pointer.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match="pointer digest"):
        load_run("history", tmp_path)
    # A frozen generation selection remains independently usable.
    again = load_run("history", tmp_path, str(generations[0]))
    assert again.identity == initial.identity


def test_generated_answers_score_only_complete_answer_lines(
    trained_run, tmp_path, monkeypatch, capsys
):
    from klab.evaluation import answers
    from klab.evaluation.generation_request import GenerationResult

    cases = [
        ("signed", "integer", "2", " +002 ", "stop"),
        ("extra", "integer", "2", "2 or 3", "stop"),
        ("truncated", "integer", "2", "2", "length"),
        ("punctuated", "yes_no", "no", " No.", "stop"),
        ("wrong", "yes_no", "no", "yes", "stop"),
        ("label", "text", "Tank A", "Tank A", "stop"),
    ]
    declaration = {
        "schema_version": 1,
        "name": "generated-answers",
        "max_new_tokens": 4,
        "items": [
            {
                "id": name,
                "category": "mechanism",
                "prompt": f"{name}? Answer:",
                "kind": kind,
                "accepted": [key],
            }
            for name, kind, key, _, _ in cases
        ],
    }
    panel = tmp_path / "answers.json"
    panel.write_text(json.dumps(declaration))
    root = trained_run.logging.root_dir
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(
        sys,
        "argv",
        ["klab", "eval", "original", "--runs-dir", str(root), "--answers", str(panel)],
    )
    outputs = {f"{name}? Answer:": (text, finish) for name, _, _, text, finish in cases}

    def generate(model, tokenizer, prompt, context, max_new_tokens, device, **options):
        assert max_new_tokens == 4
        assert options["strict_context"] and options["temperature"] == 0
        assert options["stop_sequences"] == ("\n",)
        text, finish = outputs[prompt]
        return GenerationResult(prompt + text, [], 2, 4, finish, 5)

    with monkeypatch.context() as patch:
        patch.setattr(answers, "generate_result", generate)
        main()
    result = json.loads(capsys.readouterr().out)
    report = result["answers"]
    assert [row["correct"] for row in report["items"]] == [
        True,
        False,
        False,
        True,
        False,
        True,
    ]
    assert report["overall"] == {
        "items": 6,
        "correct": 3,
        "valid_form": 4,
        "length_limited": 1,
        "completion_tokens": 24,
        "forward_input_positions": 30,
    }
    assert report["kinds"]["yes_no"]["valid_form"] == 2
    assert report["kinds"]["yes_no"]["correct"] == 1
    saved = Path(result.pop("output"))
    assert json.loads(saved.read_text()) == result
    pointer = json.loads((root / "original/checkpoints/latest.json").read_text())
    assert result["identity"]["checkpoint_sha256"] == pointer["manifest_sha256"]
    assert result["identity"]["run_data_hashed"] is False

    # The real decoder must refuse to crop a prompt to fit the generation budget.
    declaration["items"][0]["prompt"] = "answer " * 100
    panel.write_text(json.dumps(declaration))
    with pytest.raises(ValueError, match="answer item signed: .*exceed max_seq_len"):
        main()
    assert list(saved.parent.glob("answers-*.json")) == [saved]


def _chat_cli(monkeypatch, capsys, cwd, *options):
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "klab",
            "chat",
            "original",
            "--message",
            "hello",
            "--max-new-tokens",
            "1",
            "--json",
            *options,
        ],
    )
    main()
    return json.loads(capsys.readouterr().out)


def test_chat_uses_selected_store_from_any_checkout_directory(
    trained_run, tmp_path, monkeypatch, capsys
):
    checkout = tmp_path / "checkout"
    source = checkout / "src"
    source.mkdir(parents=True)
    (checkout / "pyproject.toml").write_text("")
    store = tmp_path / "persistent-state"
    shutil.copytree(trained_run.logging.root_dir / "original", store / "runs/original")
    monkeypatch.setenv("KLAB_WORK_DIR", str(store))

    from_root = _chat_cli(monkeypatch, capsys, checkout)
    from_source = _chat_cli(monkeypatch, capsys, source)
    pointer = json.loads((store / "runs/original/checkpoints/latest.json").read_text())
    assert from_source == from_root
    assert from_root["identity"]["checkpoint_sha256"] == pointer["manifest_sha256"]


def test_chat_explicit_run_directory_remains_cwd_relative(
    trained_run, tmp_path, monkeypatch, capsys
):
    (tmp_path / "pyproject.toml").write_text("")
    source = tmp_path / "src"
    source.mkdir()
    shutil.copytree(
        trained_run.logging.root_dir / "original", tmp_path / "runs/original"
    )

    # An explicitly missing directory must not silently select the project run.
    with pytest.raises(SystemExit):
        _chat_cli(monkeypatch, capsys, source, "--runs-dir", "runs")
    response = _chat_cli(monkeypatch, capsys, source, "--runs-dir", "../runs")
    pointer = json.loads(
        (tmp_path / "runs/original/checkpoints/latest.json").read_text()
    )

    assert response["identity"]["checkpoint_sha256"] == pointer["manifest_sha256"]


def _assert_position_buckets(report: dict, panel: dict) -> None:
    buckets = report["position_buckets"]
    assert [(b["start"], b["end"]) for b in buckets] == [(0, 8), (8, 16)]
    targets = sum(b["targets"] for b in buckets)
    assert targets == panel["supervised_targets"] == report["valid_targets"]
    weighted = sum(b["loss"] * b["targets"] for b in buckets) / targets
    assert weighted == pytest.approx(report["loss"], rel=1e-6)


def test_panels_report_loss_by_position_during_and_after_training(
    trained_run, tmp_path, monkeypatch, capsys
):
    from klab.data.packing import TokenBlockDataset, load_prepared_data
    from klab.evaluation.panels import run_panel
    from klab.training.manifest import sha256_file

    source = load_run("original", trained_run.logging.root_dir)
    data = source.run / "data"
    prepared = load_prepared_data(data, byte_enabled=False)
    seq_len = trained_run.training.seq_len
    blocks = TokenBlockDataset(
        prepared.validation, seq_len, supervision=prepared.validation_supervision
    )
    count = min(len(blocks), 6)
    panel = {
        "name": "heldout",
        "prepared_path": str(data),
        "prepared_manifest_sha256": prepared.manifest["manifest_sha256"],
        "tokenizer_file_sha256": sha256_file(trained_run.tokenizer.path),
        "tokenizer_canonical_sha256": prepared.manifest["tokenizer_sha256"],
        "split": "validation",
        "sampling": "first_eligible_blocks",
        "sequence_length": seq_len,
        "blocks": count,
        "supervised_targets": sum(
            int((blocks.numpy_block(i)[1] != -100).sum()) for i in range(count)
        ),
        "batch_size": 2,
        "max_forward_positions": count * seq_len,
    }
    # During training: every evaluation point reports the panel's buckets.
    raw = trained_run.model_dump(mode="json")
    raw["evaluation"]["panels"] = [panel]
    train(type(trained_run).model_validate(raw), run_id="paneled")
    reports = sorted(
        (trained_run.logging.root_dir / "paneled/evaluations").glob("*.json")
    )
    assert reports
    for path in reports:
        report = json.loads(path.read_text())["panels"]["heldout"]
        _assert_position_buckets(report, panel)
    # After training: `klab eval --panel` reports them for one checkpoint.
    declaration = tmp_path / "panel.json"
    declaration.write_text(
        json.dumps(
            {
                **panel,
                "schema_version": 1,
                "purpose": "development",
                "checkpoint_sha256": source.identity["checkpoint_sha256"],
            }
        )
    )
    result = run_panel(
        "original",
        trained_run.logging.root_dir,
        source.identity["checkpoint_relative_path"],
        declaration,
        tmp_path / "result.json",
    )
    assert result["status"] == "complete", result["reason"]
    _assert_position_buckets(result, panel)

    alignment = tmp_path / "alignment.json"
    args = [
        "klab",
        "eval",
        "original",
        "--runs-dir",
        str(trained_run.logging.root_dir),
        "--checkpoint",
        source.identity["checkpoint_relative_path"],
        "--panel",
        str(declaration),
        "--gradient-against",
        str(declaration),
    ]
    monkeypatch.setattr(sys, "argv", [*args, "--output", str(alignment)])
    main()
    capsys.readouterr()
    aligned = json.loads(alignment.read_text())
    assert aligned["evidence_complete"], aligned["reason"]
    assert aligned["cosine"] == pytest.approx(1.0)
    assert aligned["dot_product"] == pytest.approx(aligned["left_norm"] ** 2)
    for scored in aligned["panels"].values():
        assert scored["valid_targets"] == result["valid_targets"]
        assert scored["loss"] == pytest.approx(result["loss"], rel=1e-6)

    limited = tmp_path / "limited.json"
    monkeypatch.setattr(
        sys, "argv", [*args, "--output", str(limited), "--gradient-max-bytes", "1"]
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    denied = json.loads(limited.read_text())
    assert denied["status"] == "failed" and denied["panels"] == {}
    assert "identity" not in denied  # cap refused before loading checkpoint tensors
