"""Write-new migration of retired dataset declarations; historical readers stay usable.

The CLI integrator calls ``register_parser(data_commands)`` and gates only new
execution with ``require_current_dataset``. Migration never grants compatibility
with a historical tokenizer, prepared artifact, checkpoint, or experiment lock.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import yaml

from klab.config.loading import load_config, load_tokenizer_config
from klab.config.models import DatasetConfig
from klab.data.local_stories import DATASET, LICENSE
from klab.engram.packs import _rename_noreplace
from klab.training.manifest import sha256_file

_RETIRED = {"tinystories", "local_stories"}
_POLICY_WARNING = (
    "Explicit policy migration required: legacy direct loading permits a truncated "
    "final document; snapshot preparation requires whole documents. Generic "
    "acquisition excludes empty/null records and counts retained documents, whereas "
    "legacy direct loading counts source records and rejects null text. Numeric "
    "document/token budgets are preserved, but selected data and exposure may "
    "differ. Review a new scientific protocol; this is not a continuation or an "
    "identity-preserving conversion. Pass --accept-policy-change to acknowledge."
)


def require_current_dataset(dataset: DatasetConfig) -> None:
    """Reject retired sources at new-execution boundaries, never during history reads."""
    if dataset.source in _RETIRED:
        raise ValueError(
            f"dataset.source={dataset.source!r} is retired for new execution; "
            "run `klab data migrate CONFIG --output NEW_DIRECTORY` and use "
            "the emitted source declaration and snapshot config. Historical "
            "verification/replay APIs remain available; existing artifacts must "
            "not be relabeled. Use current snapshot declarations in new scaffolds."
        )


def _absolute(value: Any, base: Path) -> Any:
    if isinstance(value, Path):
        return str((base / value).resolve())
    if isinstance(value, dict):
        return {key: _absolute(item, base) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_absolute(item, base) for item in value]
    return value


def _write_yaml(path: Path, value: dict) -> None:
    with path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(value, handle, sort_keys=False)


def _direct_source(dataset: DatasetConfig, resources: dict | None) -> dict:
    from klab.data.sources import DatasetSource

    if resources is None:
        raise ValueError(
            "direct migration requires explicit acquisition resource bounds; "
            "pass --resources YAML with max_source_records, max_text_bytes, "
            "max_record_bytes, max_work_bytes and min_free_bytes"
        )
    return DatasetSource.model_validate(
        {
            "schema_version": 1,
            "kind": "huggingface",
            "repo_id": DATASET,
            "revision": dataset.revision,
            # The retired loader always used default, ignoring dataset_config.
            "config": "default",
            "splits": {"train": "train", "validation": "validation"},
            "text_field": "text",
            "attribution": DATASET,
            "license": LICENSE,
            "selection": {
                "mode": "bounded",
                "documents": {
                    "train": dataset.train_max_documents,
                    "validation": dataset.validation_max_documents,
                },
            },
            "resources": resources,
            "dedup": {
                "within_split": "keep",
                "overlap": "keep",
                "priority": ["train", "validation"],
            },
        }
    ).model_dump(mode="json")


def migrate(
    input_path: Path,
    output_dir: Path,
    *,
    kind: str = "auto",
    resources: dict | None = None,
    accept_policy_change: bool = False,
) -> Path:
    """Emit source.yaml, a config/binding and receipt in an exclusive directory.

    Direct Hub conversion is offline and emits a *pending* snapshot binding.
    Legacy local snapshots are cold-verified/imported by the generic source API.
    A bare manifest emits only a dataset binding: it cannot infer run budgets.
    """
    input_path = Path(input_path).resolve(strict=True)
    output_dir = Path(output_dir).absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(
            f"migration requires a new output directory: {output_dir}"
        )
    output_dir = output_dir.resolve()
    if not output_dir.parent.is_dir():
        raise FileNotFoundError(
            f"migration parent directory is missing: {output_dir.parent}"
        )
    input_digest = sha256_file(input_path)
    raw = yaml.safe_load(input_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(
            "migration input must be a run/tokenizer YAML or legacy manifest mapping"
        )
    if kind == "auto":
        kind = (
            "snapshot"
            if "splits" in raw and "dataset" not in raw
            else ("run" if "model" in raw else "tokenizer")
        )
    if kind not in {"run", "tokenizer", "snapshot"}:
        raise ValueError("migration kind must be auto, run, tokenizer or snapshot")
    config = None
    dataset = None
    if kind != "snapshot":
        loader = load_config if kind == "run" else load_tokenizer_config
        config = loader(input_path)
        dataset = config.dataset
        if dataset.source not in _RETIRED:
            raise ValueError("migration requires legacy tinystories or local_stories")
    direct = dataset is not None and dataset.source == "tinystories"
    if not direct:
        manifest_path = input_path if dataset is None else dataset.source_manifest_path
        assert manifest_path is not None
        if output_dir.is_relative_to(manifest_path.resolve().parent):
            raise ValueError(
                "migration output must be outside the historical snapshot directory"
            )
    if direct and not accept_policy_change:
        raise ValueError(_POLICY_WARNING)
    declaration = _direct_source(dataset, resources) if direct else None
    staging = Path(tempfile.mkdtemp(prefix=".data-migrate-", dir=output_dir.parent))
    try:
        if direct:
            manifest = None
        else:
            from klab.data.sources import import_legacy_snapshot, verify_snapshot

            imported = import_legacy_snapshot(
                dataset if dataset is not None else input_path, staging / "snapshot"
            )
            manifest = verify_snapshot(imported, verification_mode="cold")
            declaration = manifest["lock"]["source"]
        assert declaration is not None
        binding = {
            "source": "snapshot",
            "revision": declaration["revision"],
            "license": declaration["license"],
            "train_path": str(output_dir / "snapshot" / "train.jsonl"),
            "validation_path": str(output_dir / "snapshot" / "validation.jsonl"),
            "source_manifest_path": str(output_dir / "snapshot" / "manifest.json"),
        }
        _write_yaml(staging / "source.yaml", declaration)
        config_name = "config.yaml" if config is not None else "dataset-binding.yaml"
        if config is not None:
            values = _absolute(config.model_dump(mode="python"), input_path.parent)
            values["dataset"].update(binding)
            # Validate before publishing; retain all original numeric settings.
            type(config).model_validate(values)
        else:
            values = {"dataset": binding}
        _write_yaml(staging / config_name, values)
        receipt = {
            "format": "klab-data-migration-v1",
            "input_path": str(input_path),
            "input_sha256": input_digest,
            "kind": kind,
            "identity_preserved": False,
            "snapshot_status": "pending_acquisition" if direct else "imported",
            "policy_change_accepted": bool(direct and accept_policy_change),
            "warnings": [_POLICY_WARNING] if direct else [],
            "source_path": str(output_dir / "source.yaml"),
            "config_path": str(output_dir / config_name),
            "source_yaml_sha256": sha256_file(staging / "source.yaml"),
            "config_sha256": sha256_file(staging / config_name),
            "snapshot_manifest_sha256": None
            if manifest is None
            else manifest["manifest_sha256"],
        }
        if sha256_file(input_path) != input_digest:
            raise ValueError(
                "migration input changed while reading; retry from stable inputs"
            )
        with (staging / "migration.json").open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2, sort_keys=True)
            handle.write("\n")
        _rename_noreplace(staging, output_dir)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return output_dir / "migration.json"


def _run(args: argparse.Namespace) -> None:
    resources = None
    if args.resources is not None:
        resources = yaml.safe_load(args.resources.read_text(encoding="utf-8"))
        if not isinstance(resources, dict):
            raise ValueError("resources YAML must contain a mapping")
    receipt = migrate(
        args.input,
        args.output,
        kind=args.kind,
        resources=resources,
        accept_policy_change=args.accept_policy_change,
    )
    result = json.loads(receipt.read_text(encoding="utf-8"))
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        print(receipt)
        for warning in result["warnings"]:
            print(f"WARNING: {warning}")
        if result["snapshot_status"] == "pending_acquisition":
            print(
                "Snapshot pending: lock source.yaml and acquire snapshot/ before executing config.yaml."
            )
        print("New identity: verify/rebuild downstream inputs before new execution.")


def register_parser(subparsers: Any) -> None:
    """Register under the existing native ``data`` command group."""
    parser = subparsers.add_parser(
        "migrate", help="write new declarations/configs for retired dataset sources"
    )
    parser.add_argument(
        "input", type=Path, help="run/tokenizer YAML or legacy snapshot manifest"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="new output directory"
    )
    parser.add_argument(
        "--kind", choices=("auto", "run", "tokenizer", "snapshot"), default="auto"
    )
    parser.add_argument(
        "--resources",
        type=Path,
        help="explicit source resource bounds YAML (direct migration)",
    )
    parser.add_argument("--accept-policy-change", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(handler=_run)
