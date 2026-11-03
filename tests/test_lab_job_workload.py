"""Tiny job admission bounds effective evaluation work before model execution."""

import shutil
from pathlib import Path

import pytest
import yaml

from sparselab.lab_job_workload import TryWorkload, declarations
from sparselab.training.manifest import sha256_file


@pytest.mark.parametrize(
    ("blocks", "max_batches", "expected_batches", "accepted"),
    [
        (None, 4, 4, True),
        (None, 5, 5, False),
        ((3, 3), 64, 4, True),
        ((3, 5), 1, 5, False),
        ((128, 128), 1, 128, False),
    ],
)
def test_tiny_profile_effective_evaluation_limit(
    tmp_path: Path,
    blocks: tuple[int, int] | None,
    max_batches: int,
    expected_batches: int,
    accepted: bool,
) -> None:
    root = tmp_path / "inputs"
    example = Path(__file__).resolve().parents[1] / "examples/lab-job-try"
    shutil.copytree(example, root)
    baseline = root / "baseline.yaml"
    config = yaml.safe_load(baseline.read_text())
    config["evaluation"]["max_batches"] = max_batches
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
                "batch_size": 2,
                "max_forward_positions": count * config["training"]["seq_len"],
            }
            for name, count in zip(("first", "second"), blocks, strict=True)
        ]
    baseline.write_text(yaml.safe_dump(config))
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
