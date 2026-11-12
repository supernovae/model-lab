"""The shared decoder has no schema, path-binding or execution authority."""

import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from klab.documents import read_document


@pytest.mark.parametrize("suffix", ["yaml", "JSON"])
def test_data_parity_and_no_path_interpretation(tmp_path, suffix):
    raw = {"plugin": {"path": "opaque:relative", "future_field": True}, "seed": "7"}
    path = tmp_path / f"input.{suffix}"
    path.write_text(yaml.safe_dump(raw) if suffix == "yaml" else json.dumps(raw))
    assert read_document(path) == raw
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "suffix,text,match",
    [
        ("yaml", "nested: {key: 1, key: 2}", "duplicate YAML key"),
        ("json", '{"nested":{"key":1,"key":2}}', "duplicate JSON key"),
        ("yaml", "nested: {1: value}", "keys must be strings"),
        ("yaml", "nested: [.nan]", "non-finite"),
        ("json", '{"nested":[NaN]}', "non-finite"),
        ("json", '{"nested":[1e999]}', "non-finite"),
        ("yaml", "nested: &cycle [*cycle]", "cyclic"),
        (
            "yaml",
            "nested: !!python/object/apply:os.system [echo]",
            "invalid experiment",
        ),
        ("yaml", "nested: [", "invalid experiment"),
        ("yaml", "<<: {seed: 7}\nseed: 8", "invalid experiment"),
        ("yaml", "[]", "must be a mapping"),
    ],
)
def test_rejected_data_has_source(tmp_path, suffix, text, match):
    path = tmp_path / f"bad.{suffix}"
    path.write_text(text)
    with pytest.raises((ValueError, TypeError), match=match) as error:
        read_document(path)
    assert str(path) in str(error.value)


@pytest.mark.parametrize(
    "suffix,text",
    [
        ("yaml", "fraction: 0.12345678901234567890123456789"),
        ("json", '{"fraction":0.12345678901234567890123456789}'),
    ],
)
def test_exact_numbers_preserve_authored_precision(tmp_path, suffix, text):
    path = tmp_path / f"exact.{suffix}"
    path.write_text(text)
    assert read_document(path, exact_decimals=True)["fraction"] == Decimal(
        "0.12345678901234567890123456789"
    )
    assert isinstance(read_document(path)["fraction"], float)


def test_repeated_alias_is_not_a_cycle(tmp_path):
    path = tmp_path / "aliases.yaml"
    path.write_text("a: &value [1, 2]\nb: *value\n")
    assert read_document(path) == {"a": [1, 2], "b": [1, 2]}


def test_document_import_and_read_do_not_reach_schema_or_runtime(tmp_path):
    path = tmp_path / "input.yaml"
    path.write_text("plugin: {path: relative}\n")
    code = """
import importlib.abc
import sys
from pathlib import Path
class Fence(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'mlx', 'datasets', 'tokenizers', 'pydantic'} or (
            fullname.startswith('klab.') and fullname != 'klab.documents'
        ):
            raise AssertionError('unexpected dependency: ' + fullname)
sys.meta_path.insert(0, Fence())
from klab.documents import read_document
assert read_document(Path(sys.argv[1])) == {'plugin': {'path': 'relative'}}
"""
    subprocess.run([sys.executable, "-c", code, str(path)], check=True, cwd=tmp_path)
    assert list(tmp_path.iterdir()) == [path]


def test_missing_and_invalid_utf8_have_source(tmp_path):
    path = tmp_path / "missing.yaml"
    with pytest.raises(ValueError, match=str(path)):
        read_document(path)
    path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match=str(path)):
        read_document(path)


def test_model_components_do_not_import_workflow_consumers():
    """Fence the computational package, not adapters with known coupling debt."""
    import ast

    import klab

    root = Path(klab.__file__).parent / "model"
    forbidden = (
        "klab.experiments",
        "klab.lab_mode",
        "klab.research",
        "klab.cli",
    )
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom):
                imports = [node.module or ""]
                if node.module == "klab":
                    imports += ["klab." + name.name for name in node.names]
            elif isinstance(node, ast.Import):
                imports = [name.name for name in node.names]
            else:
                continue
            assert not any(
                name == prefix or name.startswith(prefix + ".")
                for name in imports
                for prefix in forbidden
            ), (path, imports)


def test_alias_doubling_visits_shared_leaves_once(tmp_path, monkeypatch):
    levels = 25
    from klab import documents

    path = tmp_path / "doubling.yaml"
    path.write_text(
        "l0: &a0 [1.0, 2.0]\n"
        + "".join(
            f"l{level}: &a{level} [*a{level - 1}, *a{level - 1}]\n"
            for level in range(1, levels + 1)
        )
    )
    finite = documents.math.isfinite
    visits = 0

    def count_leaf(value):
        nonlocal visits
        visits += 1
        # Fail immediately if a shared container is revisited: even the mutated
        # 25-level case stays tiny instead of performing an exponential walk.
        assert visits <= 2, "shared alias leaves were traversed again"
        return finite(value)

    monkeypatch.setattr(documents.math, "isfinite", count_leaf)
    raw = documents.read_document(path)
    assert visits == 2
    for level in range(1, levels + 1):
        assert raw[f"l{level}"][0] is raw[f"l{level - 1}"]
        assert raw[f"l{level}"][1] is raw[f"l{level - 1}"]
