"""One explicit command contract from plan to verified Corpus artifact."""

import sys
from pathlib import Path

import pytest
from recipe_fixtures import Lab

from klab.canonical import canonical_json
from klab.corpus.jsonl_records import read_rows
from klab.training.manifest import sha256_file

_DRIVER = """\
import json, os, sys, time
from pathlib import Path

assert "KLAB_TEST_PARENT_SECRET" not in os.environ
request = json.loads(Path(os.environ["KLAB_OPERATOR_REQUEST"]).read_text())
marker = Path(os.environ["INVOCATIONS"])
previous = marker.read_text().count("\\n") if marker.exists() else 0
with marker.open("a") as stream:
    stream.write("ran\\n")
mode = os.environ["MODE"]
if mode == "timeout":
    time.sleep(4)
inputs = request["inputs"]
outputs = request["outputs"]
docs = [json.loads(line)["text"] for line in Path(inputs["text"]).read_text().splitlines()]
ids = [json.loads(line)["document_id"] for line in Path(inputs["ids"]).read_text().splitlines()]

def write(name, rows):
    with Path(outputs[name]).open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + "\\n")

text = [{"text": doc.upper()} for doc in docs]
if mode == "different":
    text[0]["text"] += str(previous)
write("text.jsonl", text)
if mode == "retry" and previous == 0:
    sys.exit(7)
if mode == "misaligned":
    ids.reverse()
if mode == "extra":
    Path("outputs/extra.jsonl").write_text("{}")
write("stats.jsonl", [
    {"document_id": id_, "utf8_bytes": len(doc.encode()), "codepoints": len(doc), "lines": 1}
    for id_, doc in zip(ids, docs, strict=True)
])
write("relations.jsonl", [{
    "subject":"world:first", "predicate":"world:relates",
    "value":{"kind":"entity", "id":"world:heldout" if mode == "cross-split" else "world:second"}
}])
"""


def _make_spec(root, *, mode, determinism="deterministic", timeout=10):
    def pin(path):
        return {
            "path": str(path),
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }

    def rows(name, records, contract):
        path = root / name
        path.write_bytes(b"".join(canonical_json(row) + b"\n" for row in records))
        return {**pin(path), "contract": contract, "max_rows": len(records)}

    driver = root / "transform.py"
    driver.write_text(_DRIVER)
    marker = root / f"{mode}-invocations"
    inputs = {
        "text": rows(
            "input.jsonl", [{"text": "alpha"}, {"text": "beta"}], "local_text"
        ),
        "ids": rows(
            "ids.jsonl",
            [{"document_id": "0" * 64}, {"document_id": "1" * 64}],
            "document_ids",
        ),
        "scope": rows(
            "scope.jsonl",
            [
                {"id": "world:first", "family": "world:train", "split": "train"},
                {"id": "world:second", "family": "world:train", "split": "train"},
                {"id": "world:heldout", "family": "world:heldout", "split": "test"},
            ],
            "relation_scope",
        ),
    }
    limit = {"max_bytes": 8192, "max_rows": 4}
    declaration = root / f"{mode}.json"
    declaration.write_bytes(
        canonical_json(
            {
                "operator": "fixture",
                "determinism": determinism,
                "executable": pin(Path(sys.executable).resolve()),
                "dependencies": {"transform.py": pin(driver)},
                "arguments": ["-I", "-S", "dependencies/transform.py"],
                "environment": {"MODE": mode, "INVOCATIONS": str(marker)},
                "timeout_seconds": timeout,
                "inputs": inputs,
                "outputs": {
                    "text.jsonl": {
                        **limit,
                        "contract": "local_text",
                        "aligned_to": "text",
                    },
                    "stats.jsonl": {
                        **limit,
                        "contract": "text_statistics",
                        "aligned_to": "ids",
                    },
                    "relations.jsonl": {
                        **limit,
                        "contract": "relation_records",
                        "scope": "scope",
                    },
                },
            }
        )
    )
    return declaration, marker


def _command(lab, action, *arguments):
    return lab.json("corpus", "operator", action, *map(str, arguments))


def test_command_plan_authorized_publication_and_replay(tmp_path, monkeypatch, capsys):
    lab = Lab(tmp_path, monkeypatch, capsys)
    monkeypatch.setenv("KLAB_TEST_PARENT_SECRET", "inheritance-guard")
    declaration, marker = _make_spec(tmp_path, mode="retry")
    plan = _command(lab, "plan", declaration)
    assert plan["authority"] == "caller_permissions" and not plan["candidate"]
    with pytest.raises(ValueError, match="requires --allow-command-operators"):
        _command(lab, "run", declaration)
    assert not marker.exists()
    allow = ("--allow-command-operators", plan["authorization"])

    with pytest.raises(ValueError, match="exited with status 7"):
        _command(lab, "run", declaration, *allow)
    assert not _command(lab, "plan", declaration)["candidate"]

    built = _command(lab, "run", declaration, *allow)
    assert built["computation_key"] == plan["computation_key"] == built["key"]
    assert read_rows(Path(built["path"]) / "text.jsonl") == [
        {"text": "ALPHA"},
        {"text": "BETA"},
    ]
    assert read_rows(Path(built["path"]) / "stats.jsonl")[1]["document_id"] == "1" * 64
    assert _command(lab, "inspect", built["key"])["files"] == built["files"]
    assert _command(lab, "run", declaration, *allow)["reused"]
    assert marker.read_text().splitlines() == ["ran", "ran"]

    replay = ("--replay", built["key"])
    replay_plan = _command(lab, "plan", declaration, *replay)
    assert replay_plan["authorization"] != plan["authorization"]
    with pytest.raises(ValueError, match="requires --allow-command-operators"):
        _command(lab, "run", declaration, *replay, *allow)
    replayed = _command(
        lab,
        "run",
        declaration,
        *replay,
        "--allow-command-operators",
        replay_plan["authorization"],
    )
    assert (
        replayed["files"] == built["files"]
        and replayed["replay"]["replay_of"] == built["key"]
    )
    assert marker.read_text().count("\n") == 3

    dependency = tmp_path / "transform.py"
    dependency.write_text(_DRIVER.replace("sys.exit(7)", "sys.exit(8)"))
    with pytest.raises(ValueError, match="file digest or size drift"):
        _command(lab, "run", declaration, *allow)
    assert _command(lab, "inspect", built["key"])["files"] == built["files"]
    assert marker.read_text().count("\n") == 3
    for mode, match in (
        ("misaligned", "does not align"),
        ("cross-split", "cross-split entity"),
        ("extra", "exactly its declared output files"),
        ("timeout", "wall-time limit"),
    ):
        bad, _ = _make_spec(tmp_path, mode=mode, timeout=1 if mode == "timeout" else 10)
        rejected = _command(lab, "plan", bad)
        with pytest.raises(ValueError, match=match):
            _command(
                lab, "run", bad, "--allow-command-operators", rejected["authorization"]
            )
        assert not _command(lab, "plan", bad)["candidate"]


def test_command_publication_keeps_exact_nondeterministic_attempts(
    tmp_path, monkeypatch, capsys
):
    lab = Lab(tmp_path, monkeypatch, capsys)
    declaration, marker = _make_spec(
        tmp_path, mode="different", determinism="nondeterministic"
    )
    plan = _command(lab, "plan", declaration)
    allow = ("--allow-command-operators", plan["authorization"])
    first = _command(lab, "run", declaration, *allow)
    second = _command(lab, "run", declaration, *allow)
    assert first["key"] != second["key"] and first["files"] != second["files"]
    assert (
        first["computation_key"] == second["computation_key"] == plan["computation_key"]
    )
    assert _command(lab, "inspect", first["key"])["files"] == first["files"]
    assert marker.read_text().count("\n") == 2

    deterministic, _ = _make_spec(tmp_path, mode="different")
    approved = _command(lab, "plan", deterministic)
    original = _command(
        lab,
        "run",
        deterministic,
        "--allow-command-operators",
        approved["authorization"],
    )
    replay = ("--replay", original["key"])
    replay_plan = _command(lab, "plan", deterministic, *replay)
    with pytest.raises(ValueError, match="replay changed published output bytes"):
        _command(
            lab,
            "run",
            deterministic,
            *replay,
            "--allow-command-operators",
            replay_plan["authorization"],
        )
    assert _command(lab, "inspect", original["key"])["files"] == original["files"]
