"""External rows become local_chat or local_text JSONL that Lab prepares as-is."""

import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from klab.cli.main import main

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


@pytest.fixture
def klab(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KLAB_WORK_DIR", str(tmp_path / "work"))

    def run(*args: str) -> dict:
        monkeypatch.setattr(sys, "argv", ["klab", *args])
        main()
        out = capsys.readouterr().out.strip().splitlines()
        return json.loads(out[-1]) if "--json" in args else {"out": out}

    return run


def declare(path: Path, **sections) -> Path:
    payload = {
        "license": "CC0-1.0",
        "attribution": "test fixture",
        "limits": {"max_rows": 1000, "max_record_bytes": 65536},
        **sections,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def prepare(root: Path, klab, source: str, converted: Path) -> None:
    """Fit a tokenizer on the converted files, then prepare a run from them."""
    rows = json.loads((converted / "receipt.json").read_text())["outputs"]
    dataset = {
        "source": source,
        "train_path": str(converted / "train.jsonl"),
        "validation_path": str(converted / "validation.jsonl"),
        "license": "CC0-1.0",
        "cache_dir": str(root / "data"),
        "train_max_documents": rows["train"]["rows"],
        "validation_max_documents": rows["validation"]["rows"],
        "train_max_tokens": 100000,
        "validation_max_tokens": 100000,
    }
    tokenizer = yaml.safe_load((CONFIGS / "tokenizer_smoke.yaml").read_text())
    tokenizer.update(
        output_dir=str(root / "tokenizer"), dataset=dataset, vocab_size=272
    )
    (root / "tokenizer.yaml").write_text(yaml.safe_dump(tokenizer))
    klab("tokenizer", "train", str(root / "tokenizer.yaml"))
    config = yaml.safe_load((CONFIGS / "smoke_cpu.yaml").read_text())
    config["model"]["vocab_size"] = 272
    config["tokenizer"]["path"] = str(root / "tokenizer" / "tokenizer.json")
    config["logging"]["root_dir"] = str(root / "runs")
    config["dataset"] = dataset
    run = root / "run.yaml"
    run.write_text(yaml.safe_dump(config, sort_keys=False))
    prepared = Path(klab("data", "prepare", str(run))["out"][-1])
    assert (prepared / "train.npy").is_file()


def chat(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": question, "extra": 1},
            {"role": "assistant", "content": answer},
        ]
    }


def test_chat_rows_convert_verify_and_prepare(tmp_path, klab):
    rows = {
        "train": [chat(f"What is item {n}?", f"Item {n} is a tool.") for n in range(12)]
        + [chat("What is item 0?", "Item 0 is a tool."), chat("Blank?", "  ")],
        "test": [chat(f"Where is box {n}?", f"Box {n} is upstairs.") for n in range(6)],
    }
    for split, items in rows.items():
        pq.write_table(pa.Table.from_pylist(items), tmp_path / f"{split}.parquet")
    declaration = declare(
        tmp_path / "chat.yaml",
        source={
            "kind": "local",
            "files": {"train": "train.parquet", "test": "test.parquet"},
        },
        splits={"mode": "source", "train": "train", "validation": "test"},
        mapping={
            "kind": "chat",
            "messages_field": "messages",
            "loss_mode": "assistant_only",
        },
    )
    out = tmp_path / "converted"
    assert (
        klab("data", "convert", str(declaration), "--output", str(out), "--json")[
            "status"
        ]
        == "complete"
    )
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["reads"]["train"]["excluded"] == {"duplicate": 1, "empty": 1}
    assert receipt["outputs"]["train"]["rows"] == 12
    assert receipt["outputs"]["validation"]["rows"] == 6
    first = json.loads((out / "train.jsonl").read_text().splitlines()[0])
    assert first["format_version"] == 2 and first["loss_mode"] == "assistant_only"
    assert klab("data", "verify-conversion", str(out), "--json")["status"] == "verified"
    prepare(tmp_path, klab, "local_chat", out)


def test_commit_rows_become_diff_text_and_prepare(tmp_path, klab):
    rows = [
        {
            "id": n,
            "path": f"src/mod{n}.py",
            "old_contents": f"def f():\n    return {n}\n",
            "new_contents": f"def f():\n    return {n + 1}\n",
            "message": f"Bump the value in module {n}",
        }
        for n in range(40)
    ]
    rows.append({**rows[0], "id": 99, "new_contents": rows[0]["old_contents"]})
    source = tmp_path / "commits.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in rows))
    declaration = declare(
        tmp_path / "commits.yaml",
        source={"kind": "local", "files": {"train": "commits.jsonl"}},
        splits={
            "mode": "hash",
            "split": "train",
            "key": ["id"],
            "validation_fraction": 0.3,
        },
        mapping={
            "kind": "text",
            "template": "{message}\n\n{unified_diff(old_contents, new_contents, path, path)}",
        },
    )
    out = tmp_path / "converted"
    klab("data", "convert", str(declaration), "--output", str(out), "--json")
    receipt = json.loads((out / "receipt.json").read_text())
    assert receipt["output_kind"] == "local_text"
    assert receipt["reads"]["train"]["excluded"] == {"empty_diff": 1}
    counts = [receipt["outputs"][name]["rows"] for name in ("train", "validation")]
    assert sum(counts) == 40 and min(counts) > 0
    text = json.loads((out / "train.jsonl").read_text().splitlines()[0])["text"]
    assert (
        "--- a/src/mod" in text
        and "\n-    return " in text
        and "\n+    return " in text
    )
    klab("data", "verify-conversion", str(out), "--json")
    prepare(tmp_path, klab, "local_text", out)


def test_cross_split_conversation_is_refused_and_tampering_is_detected(tmp_path, klab):
    shared = {"instruction": "Name the drawer.", "output": "The blue one."}
    (tmp_path / "train.jsonl").write_text(
        json.dumps(shared)
        + "\n"
        + json.dumps({"instruction": "Hi", "output": "Hello"})
        + "\n"
    )
    (tmp_path / "validation.jsonl").write_text(
        json.dumps({"instruction": " Name  the drawer.", "output": "The blue one. "})
        + "\n"
    )
    declaration = declare(
        tmp_path / "pairs.yaml",
        source={
            "kind": "local",
            "files": {"train": "train.jsonl", "validation": "validation.jsonl"},
        },
        splits={"mode": "source", "train": "train", "validation": "validation"},
        mapping={
            "kind": "instruction",
            "prompt": "{instruction}",
            "response": "{output}",
            "loss_mode": "all_tokens",
        },
    )
    out = tmp_path / "converted"
    with pytest.raises(SystemExit):
        klab("data", "convert", str(declaration), "--output", str(out), "--json")
    assert not out.exists() and not (tmp_path / ".converted.work").exists()

    (tmp_path / "validation.jsonl").write_text(
        json.dumps({"instruction": "Name the shelf.", "output": "The top one."}) + "\n"
    )
    klab("data", "convert", str(declaration), "--output", str(out), "--json")
    klab("data", "verify-conversion", str(out), "--json")
    train = out / "train.jsonl"
    train.write_text(train.read_text().replace("blue", "red"))
    with pytest.raises(SystemExit):
        klab("data", "verify-conversion", str(out), "--json")
