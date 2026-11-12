import json
import subprocess
import sys
from pathlib import Path

import yaml
from tokenizers import Tokenizer, models

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
