"""Small public-CLI capability run across the shipped PyTorch mechanisms."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

from klab.config.loading import load_config, load_tokenizer_config
from klab.training.manifest import config_sha256, source_identity
from klab.workdir import ensure_work_dir

SMOKE_FAMILIES = {
    "dense": "smoke_cpu.yaml",
    "moe": "smoke_moe_cpu.yaml",
    "block_sparse": "smoke_sparse_cpu.yaml",
    "sliding": "smoke_sliding_cpu.yaml",
    "mla": "smoke_mla_cpu.yaml",
    "token_engram": "smoke_memory_cpu.yaml",
    "byte_engram": "smoke_byte_memory_cpu.yaml",
    "combined": "smoke_combined_cpu.yaml",
}


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


DEFAULT_COMMAND_TIMEOUT_SECONDS = 3600.0


def smoke_readiness(
    configs_root: Path,
    output: Path,
    *,
    families: tuple[str, ...] = (),
    command_timeout: float | None = DEFAULT_COMMAND_TIMEOUT_SECONDS,
) -> Path:
    """Exercise inspect through resumed inference with isolated mutable outputs.

    ``command_timeout`` bounds each child command (seconds); ``None`` disables
    the bound for slow hosts. It is an operational limit, not a quality gate.
    """
    if command_timeout is not None and not (
        isinstance(command_timeout, (int, float))
        and not isinstance(command_timeout, bool)
        and 0 < command_timeout < float("inf")
    ):
        raise ValueError("command_timeout must be positive and finite, or None")
    ensure_work_dir()
    selected = families or tuple(SMOKE_FAMILIES)
    if not selected or any(name not in SMOKE_FAMILIES for name in selected):
        raise ValueError("unknown smoke family")
    if len(set(selected)) != len(selected):
        raise ValueError("duplicate smoke family")
    configs_root = configs_root.expanduser().resolve(strict=True)
    tokenizer_source = load_tokenizer_config(configs_root / "tokenizer_smoke.yaml")
    sources = {
        name: load_config(configs_root / SMOKE_FAMILIES[name]) for name in selected
    }
    for name, source in sources.items():
        if (
            source.runtime.engine != "pytorch"
            or source.runtime.backend != "cpu"
            or source.dataset.source != "synthetic"
            or source.model.vocab_size != tokenizer_source.vocab_size
        ):
            raise ValueError(f"{name} is not a compatible CPU synthetic smoke config")
    output = output.expanduser().absolute()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    derived = output / "configs"
    derived.mkdir()
    runs = output / "runs"
    tokenizer = output / "tokenizer"
    cache = output / "data"
    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "local_cpu_capability_smoke",
        "source_identity_sha256": source_identity()["sha256"],
        "status": "running",
        "command_timeout_seconds": command_timeout,
        "families": {},
        "commands": [],
        "limitations": [
            "tiny synthetic data and short runs establish CLI wiring and recovery only",
            "CUDA, ROCm, XPU, MLX, large-model fit, throughput, and model quality are not tested",
        ],
    }
    report_path = output / "readiness.json"

    def command(*args: str) -> None:
        argv = [
            sys.executable,
            "-m",
            "klab",
            "--work-dir",
            str(output),
            *args,
        ]
        try:
            completed = subprocess.run(
                argv,
                cwd=configs_root.parent,
                text=True,
                capture_output=True,
                timeout=command_timeout,
                check=False,
            )
            row = {
                "argv": argv,
                "exit_code": completed.returncode,
                "stdout_tail": completed.stdout[-4096:],
                "stderr_tail": completed.stderr[-4096:],
            }
        except subprocess.TimeoutExpired as error:
            row = {"argv": argv, "exit_code": None, "error": f"timeout: {error}"}
        report["commands"].append(row)
        _write_json(report_path, report)
        if row.get("exit_code") != 0:
            raise RuntimeError(f"readiness command failed: {' '.join(argv)}")

    tokenizer_payload = tokenizer_source.model_dump(mode="json")
    tokenizer_payload["output_dir"] = str(tokenizer)
    tokenizer_payload["dataset"]["cache_dir"] = str(cache)
    tokenizer_config = derived / "tokenizer.yaml"
    tokenizer_config.write_text(
        yaml.safe_dump(tokenizer_payload, sort_keys=False), encoding="utf-8"
    )
    family_rows = report["families"]
    assert isinstance(family_rows, dict)
    for name in selected:
        source = sources[name]
        payload = source.model_dump(mode="json")
        payload["tokenizer"]["path"] = str(tokenizer / "tokenizer.json")
        payload["dataset"]["cache_dir"] = str(cache)
        payload["logging"]["root_dir"] = str(runs)
        target = derived / f"{name}.yaml"
        target.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        family_rows[name] = {
            "source_config": SMOKE_FAMILIES[name],
            "source_config_sha256": config_sha256(source.model_dump(mode="json")),
            "effective_config_sha256": config_sha256(payload),
            "config": str(target),
            "status": "pending",
        }
    _write_json(report_path, report)
    try:
        command("tokenizer", "train", str(tokenizer_config))
        for name in selected:
            family = family_rows[name]
            family["status"] = "running"
            try:
                path = str(family["config"])
                parent = f"readiness-{name}"
                child = f"{parent}-resumed"
                checkpoint = str(runs / parent / "checkpoints" / "latest.json")
                child_checkpoint = str(runs / child / "checkpoints" / "latest.json")
                command("inspect", path, "--json")
                command("workspace", "preflight", path)
                command("data", "prepare", path)
                command(
                    "stage",
                    path,
                    "--through",
                    "smoke",
                    "--output",
                    str(output / "staging" / name),
                )
                command(
                    "train",
                    path,
                    "--run-id",
                    parent,
                    "--runs-dir",
                    str(runs),
                    "--stop-after-step",
                    "2",
                )
                command("checkpoint", "verify", checkpoint, "--json")
                command("eval", parent, "--runs-dir", str(runs))
                command(
                    "generate",
                    parent,
                    "--runs-dir",
                    str(runs),
                    "--prompt",
                    "Once upon a time",
                    "--max-new-tokens",
                    "4",
                )
                command(
                    "train",
                    path,
                    "--run-id",
                    child,
                    "--runs-dir",
                    str(runs),
                    "--resume",
                    checkpoint,
                    "--stop-after-step",
                    "3",
                )
                command("checkpoint", "verify", child_checkpoint, "--json")
            except (RuntimeError, OSError, ValueError) as error:
                family["status"] = "failed"
                family["failure"] = str(error)
                raise
            else:
                family["status"] = "passed"
                family["parent_checkpoint"] = checkpoint
                family["child_checkpoint"] = child_checkpoint
            finally:
                _write_json(report_path, report)
        report["status"] = "passed"
    except (RuntimeError, OSError, ValueError) as error:
        report["status"] = "failed"
        report["failure"] = str(error)
        raise
    finally:
        _write_json(report_path, report)
    return report_path
