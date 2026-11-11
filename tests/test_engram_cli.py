import json
import subprocess
import sys
from pathlib import Path

import yaml
from tokenizers import Tokenizer, models

from klab.canonical import canonical_digest
from klab.config.loading import load_config


def test_native_memory_map_binds_complete_validated_config(tmp_path):
    tokenizer = tmp_path / "tokenizer.json"
    Tokenizer(models.WordLevel({f"token{i}": i for i in range(260)})).save(
        str(tokenizer)
    )
    config = load_config(Path("configs/runtime_smoke_cpu.yaml")).model_dump(mode="json")
    config["tokenizer"]["path"] = str(tokenizer)
    config["model"].update(
        vocab_size=260,
        memory="ngram",
        memory_table_size=31,
        memory_dim=4,
        memory_ngram_size=3,
        memory_variant="context_key",
        memory_addressing="prime_xor_v1",
        memory_table_sizes=[31],
    )
    source = tmp_path / "base.yaml"
    source.write_text(yaml.safe_dump(config))
    before = source.read_bytes(), tokenizer.read_bytes()
    output = tmp_path / "mapping.json"
    bound = tmp_path / "bound.yaml"
    command = [
        sys.executable,
        "-m",
        "klab",
        "tokenizer",
        "memory-map",
        str(tokenizer),
        "--mode",
        "deepseek_v1",
        "--output",
        str(output),
        "--bind-config",
        str(source),
        "--config-output",
        str(bound),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    mapping = json.loads(output.read_text())
    assert (
        load_config(bound).model.memory_token_mapping.model_dump(mode="json") == mapping
    )
    assert bound.with_name("bound.yaml.derivation.json").is_file()
    assert before == (source.read_bytes(), tokenizer.read_bytes())
    again = subprocess.run(command, capture_output=True, text=True, check=False)
    assert again.returncode != 0
    assert json.loads(output.read_text()) == mapping


def test_native_code_work_records_missing_executor_without_loading_model(tmp_path):
    # The unit adapter suite mocks sandbox outcomes; this public path verifies
    # a refused local backend does not try to open a model or run generated code.
    from klab.evaluation.code_work import (
        CodeWorkDeclaration,
        ExecutorUnavailable,
        preflight_executor,
    )

    spec = {
        "version": 1,
        "name": "fixture",
        "checkpoint_sha256": "a" * 64,
        "tokenizer_file_sha256": "b" * 64,
        "image": "fixture/python@sha256:" + "c" * 64,
        "sequence_length": 32,
        "max_new_tokens": 8,
        "seed": 7,
        "timeout_seconds": 1,
        "tasks": [
            {
                "id": "addition",
                "family": "arithmetic",
                "source_revision": "fixture-v1",
                "license": "CC0-1.0",
                "prompt": "Define add(a,b).",
                "tests": "assert add(1,2)==3",
                "tests_sha256": canonical_digest("assert add(1,2)==3"),
            }
        ],
    }
    try:
        preflight_executor(CodeWorkDeclaration.model_validate(spec))
    except ExecutorUnavailable:
        pass
    else:
        return  # This explicitly fake digest must never provision anything.
    declaration = tmp_path / "work.json"
    declaration.write_text(json.dumps(spec))
    output = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "klab",
            "capability",
            "code-work",
            "no-model",
            str(declaration),
            "--checkpoint",
            "no-checkpoint",
            "--runs-dir",
            str(tmp_path / "runs"),
            "--output",
            str(output),
            "--backend",
            "cpu",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(output.read_text())
    assert payload["status"] == "unavailable"
    assert payload["pass_at_1"] is None
    assert not (tmp_path / "runs").exists()
