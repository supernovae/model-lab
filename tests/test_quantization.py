"""Weight-only quantization: a sealed artifact with both held-out losses."""

from __future__ import annotations

import json
import sys

import pytest
from test_training import config

from klab.cli.main import main
from klab.evaluation.inference import load_run
from klab.evaluation.quantization import MANIFEST, QuantLinear
from klab.training.trainer import train


@pytest.fixture(scope="module")
def trained_run(tmp_path_factory):
    root = tmp_path_factory.mktemp("quantize")
    original = config(root)
    small = original.model_copy(
        update={
            "training": original.training.model_copy(update={"max_steps": 2}),
            "optimizer": original.optimizer.model_copy(update={"warmup_steps": 0}),
            "checkpoint": original.checkpoint.model_copy(update={"every_steps": 1}),
        }
    )
    train(small, run_id="source")
    return small


def _cli(monkeypatch, capsys, *argv):
    monkeypatch.setattr(sys, "argv", ["klab", *argv])
    main()
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize(("bits", "group"), [(8, []), (4, ["--group-size", "8"])])
def test_quantized_checkpoint_records_both_losses_and_generates(
    trained_run, tmp_path, monkeypatch, capsys, bits, group
):
    runs = str(trained_run.logging.root_dir)
    output = tmp_path / f"int{bits}"
    result = _cli(
        monkeypatch,
        capsys,
        *("weights", "quantize", "source", "--runs-dir", runs),
        *("--output", str(output), "--bits", str(bits), *group),
    )
    heldout = result["heldout"]
    source, quantized = heldout["source"]["loss"], heldout["quantized"]["loss"]
    assert source > 0 and quantized > 0
    assert heldout["loss_delta"] == pytest.approx(quantized - source)
    assert heldout["weight_bytes"]["quantized"] < heldout["weight_bytes"]["source"]
    fp32 = load_run("source", trained_run.logging.root_dir)
    assert result["source"]["checkpoint_sha256"] == fp32.identity["checkpoint_sha256"]
    assert heldout["source"]["loss"] == pytest.approx(fp32.evaluate()["loss"])
    loaded = load_run("source", trained_run.logging.root_dir, quantized=output)
    assert all(
        isinstance(loaded.model.get_submodule(name), QuantLinear)
        for name in result["modules"]
    )
    assert loaded.evaluate()["loss"] == pytest.approx(quantized)

    generated = _cli(
        monkeypatch,
        capsys,
        *("generate", "source", "--runs-dir", runs, "--prompt", "The"),
        *("--max-new-tokens", "4", "--quantized", str(output), "--json"),
    )
    assert generated["identity"]["quantization"]["scheme"]["bits"] == bits
    assert generated["text"].startswith("The")


def test_quantized_weights_refuse_another_checkpoint_or_an_edited_manifest(
    trained_run, tmp_path, monkeypatch, capsys
):
    runs = trained_run.logging.root_dir
    output = tmp_path / "int8"
    _cli(
        monkeypatch,
        capsys,
        *("weights", "quantize", "source", "--runs-dir", str(runs)),
        *("--output", str(output), "--bits", "8"),
    )
    earlier = min((runs / "source/checkpoints").glob("step_*"))
    with pytest.raises(ValueError, match="made from checkpoint"):
        load_run("source", runs, str(earlier), quantized=output)
    manifest = json.loads((output / MANIFEST).read_text())
    manifest["heldout"]["quantized"]["loss"] = 0.0
    (output / MANIFEST).write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="digest mismatch"):
        load_run("source", runs, quantized=output)
