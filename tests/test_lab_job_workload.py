"""Tiny job admission bounds effective evaluation work before model execution."""

import shutil
from pathlib import Path

import pytest
import yaml

from klab.config.loading import load_config
from klab.training.manifest import sha256_file


def _evaluation_config(tmp_path, blocks, batch_size, max_batches):
    root = tmp_path / "inputs"
    example = Path(__file__).resolve().parents[1] / "examples/lab-job-try"
    shutil.copytree(example, root)
    baseline = root / "baseline.yaml"
    config = yaml.safe_load(baseline.read_text())
    config["evaluation"]["max_batches"] = max_batches
    if blocks is None and batch_size is not None:
        config["evaluation"]["batch_size"] = batch_size
    if blocks is not None:
        config["evaluation"]["panels"] = [
            {
                "name": name,
                "prepared_path": "../outputs/data",
                "prepared_manifest_sha256": "0" * 64,
                "tokenizer_file_sha256": "0" * 64,
                "tokenizer_canonical_sha256": "0" * 64,
                "split": "validation",
                "sampling": "first_eligible_blocks",
                "sequence_length": config["training"]["seq_len"],
                "blocks": count,
                "supervised_targets": count * config["training"]["seq_len"],
                "batch_size": batch_size,
                "max_forward_positions": count * config["training"]["seq_len"],
            }
            for name, count in zip(("first", "second"), blocks, strict=True)
        ]
    baseline.write_text(yaml.safe_dump(config))
    return baseline


@pytest.mark.parametrize(
    ("blocks", "batch_size", "max_batches", "expected_batches", "accepted"),
    [
        (None, None, 4, 4, True),
        (None, None, 5, 5, False),
        ((3, 3), 2, 64, 4, True),
        ((3, 5), 2, 1, 5, False),
        ((128, 128), 2, 1, 128, False),
        ((16, 16), 16, 1, 2, True),
        ((16, 17), 17, 1, 2, False),
        ((100_000, 100_000), 100_000, 1, 2, False),
        (None, 8, 4, 4, True),
        (None, 9, 4, 4, False),
    ],
)
def test_tiny_profile_effective_evaluation_limit(
    tmp_path: Path,
    blocks: tuple[int, int] | None,
    batch_size: int | None,
    max_batches: int,
    expected_batches: int,
    accepted: bool,
) -> None:
    from klab.lab_job_workload import TryWorkload, declarations

    baseline = _evaluation_config(tmp_path, blocks, batch_size, max_batches)
    root = baseline.parent
    workload = TryWorkload(
        kind="lab_try",
        profile="tiny-cpu-lab",
        root=root,
        tokenizer="tokenizer.yaml",
        baseline="baseline.yaml",
        candidate="delta.yaml",
        declaration="try.yaml",
        files={path.name: sha256_file(path) for path in root.iterdir()},
    )
    if accepted:
        base, candidate, *_ = declarations(workload, root)
        assert base.evaluation.batch_limit == expected_batches
        assert candidate.evaluation.batch_limit == expected_batches
    else:
        with pytest.raises(
            ValueError,
            match="training/evaluation exceeds the existing tiny smoke profile",
        ):
            declarations(workload, root)
    # Admission reads declarations only; it must not prepare or evaluate panels.
    assert not (tmp_path / "outputs").exists()
    assert not (tmp_path / "native").exists()


@pytest.mark.parametrize(
    ("section", "changes", "message"),
    [
        ("model", {"ffn": "moe", "num_experts": 2}, "dense-FFN PyTorch"),
        (
            "model",
            {
                "memory": "byte",
                "memory_table_size": 16,
                "memory_ngram_size": 2,
                "memory_dim": 8,
            },
            "dense-FFN PyTorch",
        ),
        ("runtime", {"engine": "mlx", "backend": "metal"}, "dense-FFN PyTorch"),
        (
            "dataset",
            {"source": "local_chat", "allocation_manifest_path": "allocation.json"},
            "dense-FFN PyTorch",
        ),
        ("training", {"seq_len": 8}, "sequence length must match"),
    ],
)
def test_fixed_panels_refuse_incompatible_run_config(
    tmp_path, section, changes, message
):
    baseline = _evaluation_config(tmp_path, (3, 3), 2, 64)
    # The same non-divisible selection is valid on an ordinary dense run.
    assert load_config(baseline).evaluation.batch_limit == 4
    values = yaml.safe_load(baseline.read_text())
    values[section].update(changes)
    baseline.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match=message):
        load_config(baseline)
    # Each variant is otherwise valid: the fixed panels are the refusal boundary.
    del values["evaluation"]["panels"]
    baseline.write_text(yaml.safe_dump(values))
    load_config(baseline)


def test_fixed_panels_accept_token_memory(tmp_path):
    baseline = _evaluation_config(tmp_path, (3, 3), 2, 64)
    values = yaml.safe_load(baseline.read_text())
    values["model"].update(
        memory="ngram",
        memory_table_size=31,
        memory_ngram_size=3,
        memory_dim=8,
        memory_variant="context_key",
    )
    baseline.write_text(yaml.safe_dump(values))
    actual = load_config(baseline)
    assert actual.model.memory == "ngram"
    assert actual.evaluation.batch_limit == 4
    assert not (tmp_path / "outputs").exists()
