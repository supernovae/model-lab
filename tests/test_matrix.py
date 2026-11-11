from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from klab.experiments.matrix import expand

ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG = ROOT / "configs" / "runtime_smoke_cpu.yaml"


def _matrix(tmp_path: Path, axes: Mapping[str, object], **extra: object) -> Path:
    path = tmp_path / "matrix.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "matrix_version": 1,
                "base_config": str(BASE_CONFIG),
                "axes": axes,
                **extra,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_expand_preserves_axis_and_option_order(tmp_path: Path) -> None:
    path = _matrix(
        tmp_path,
        {
            "seed": [
                {"label": "s17", "set": {"seed": 17}},
                {"label": "s41", "set": {"seed": 41}},
            ],
            "steps": [
                {"label": "short", "set": {"training.max_steps": 3}},
                {"label": "long", "set": {"training.max_steps": 4}},
            ],
        },
        scheduling={
            "preferred_worker": "logical-cpu",
            "requirements": {"backend": ["cpu"]},
        },
    )

    expanded = expand(path)

    assert [item.coordinate for item in expanded] == [
        {"seed": "s17", "steps": "short"},
        {"seed": "s17", "steps": "long"},
        {"seed": "s41", "steps": "short"},
        {"seed": "s41", "steps": "long"},
    ]
    assert [item.config.seed for item in expanded] == [17, 17, 41, 41]
    assert [item.config.training.max_steps for item in expanded] == [3, 4, 3, 4]
    assert {item.preferred_worker for item in expanded} == {"logical-cpu"}
    assert all(item.requirements.backend == ("cpu",) for item in expanded)


def test_expand_resolves_relative_path_axis_values_from_matrix_directory(
    tmp_path: Path,
) -> None:
    path = _matrix(
        tmp_path,
        {
            "tokenizer": [
                {
                    "label": "local",
                    "set": {"tokenizer.path": "assets/tokenizer.json"},
                }
            ]
        },
    )

    expanded = expand(path)

    assert expanded[0].config.tokenizer.path == (
        tmp_path.resolve() / "assets" / "tokenizer.json"
    )


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ("seed", "seed"),
        ("training", "training.max_steps"),
        ("training.max_steps", "training"),
    ],
)
def test_expand_rejects_equal_and_ancestor_cross_axis_conflicts(
    tmp_path: Path, left: str, right: str
) -> None:
    path = _matrix(
        tmp_path,
        {
            "first": [{"label": "one", "set": {left: 17}}],
            "second": [{"label": "two", "set": {right: 18}}],
        },
    )

    with pytest.raises(ValueError, match="overlapping matrix patches"):
        expand(path)


def test_expand_rejects_unknown_path_before_product(tmp_path: Path) -> None:
    path = _matrix(
        tmp_path,
        {"bad": [{"label": "bad", "set": {"model.missing": 1}}]},
    )

    with pytest.raises(ValueError, match="unknown matrix path"):
        expand(path)


def test_expand_checks_product_bound_before_config_validation(tmp_path: Path) -> None:
    path = _matrix(
        tmp_path,
        {
            "first": [
                {"label": "one", "set": {"seed": 1}},
                {"label": "two", "set": {"seed": 2}},
            ],
            "second": [
                {"label": "one", "set": {"training.max_steps": 0}},
                {"label": "two", "set": {"training.max_steps": 0}},
            ],
        },
    )

    with pytest.raises(ValueError, match="exceeds max_runs"):
        expand(path, max_runs=3)


def test_checked_in_matrix_fixture_expands_three_concrete_cpu_configs() -> None:
    matrix_path = ROOT / "tests" / "fixtures" / "runtime-matrix.yaml"
    expanded = expand(matrix_path)

    assert len(expanded) == 3
    assert [item.coordinate["seed"] for item in expanded] == [
        "seed-7",
        "seed-17",
        "seed-41",
    ]
    assert [item.config.runtime.backend for item in expanded] == ["cpu", "cpu", "cpu"]
    assert [item.matrix_sha256 for item in expanded] == [
        item.matrix_sha256 for item in expand(matrix_path)
    ]


def test_matrix_patches_omitted_default_memory_injection(tmp_path: Path) -> None:
    matrix = _matrix(
        tmp_path,
        {
            "architecture": [
                {
                    "label": "embedding",
                    "set": {
                        "model.memory": "ngram",
                        "model.memory_table_size": 31,
                        "model.memory_ngram_size": 3,
                        "model.memory_dim": 8,
                        "model.memory_injection": "embedding",
                    },
                }
            ]
        },
    )

    expanded = expand(matrix)

    assert expanded[0].config.model.memory_injection == "embedding"


def test_matrix_identity_binds_axis_order_and_base_contents(tmp_path: Path) -> None:
    base = yaml.safe_load(BASE_CONFIG.read_text())
    base_path = tmp_path / "base.yaml"
    base_path.write_text(yaml.safe_dump(base))
    axes = {
        "seed": [{"label": "one", "set": {"seed": 17}}],
        "steps": [{"label": "short", "set": {"training.max_steps": 3}}],
    }
    first = expand(_matrix(tmp_path, axes, base_config=str(base_path)))[0]
    reordered = expand(
        _matrix(
            tmp_path, dict(reversed(list(axes.items()))), base_config=str(base_path)
        )
    )[0]
    assert first.config == reordered.config
    assert first.matrix_sha256 != reordered.matrix_sha256
    base["model"]["ffn_dim"] = 48
    base_path.write_text(yaml.safe_dump(base))
    changed = expand(_matrix(tmp_path, axes, base_config=str(base_path)))[0]
    assert changed.config.model.ffn_dim == 48
    assert changed.matrix_sha256 != first.matrix_sha256


def test_matrix_rejects_ancestor_patches_inside_one_option(tmp_path: Path) -> None:
    matrix = _matrix(
        tmp_path,
        {
            "architecture": [
                {
                    "label": "ambiguous",
                    "set": {
                        "model": {"hidden_dim": 16},
                        "model.hidden_dim": 32,
                    },
                }
            ]
        },
    )
    with pytest.raises(ValueError):
        expand(matrix)
