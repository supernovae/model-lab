"""Verified source coverage and fresh-run supervised-target budget proposals.

Passes refer to usable complete TokenBlockDataset blocks, not source records.
No preparation, source acquisition, training, or source-compatibility grant occurs.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from klab.canonical import canonical_digest
from klab.config.loading import load_config
from klab.config.models import RunConfig
from klab.data.packing import (
    _local_chat_identity,
    canonical_tokenizer_sha256,
    load_prepared_data,
    packing_identity,
)
from klab.data.tokenizer import load_tokenizer
from klab.training.manifest import source_identity

_CHUNK_TOKENS = 1 << 20
_LOCATION_KEYS = {
    "cache_dir",
    "train_path",
    "validation_path",
    "source_manifest_path",
    "corpus_release_path",
    "corpus_export_path",
}


def _bind_config(manifest: dict, config: RunConfig) -> dict | None:
    """Match the native packing identity, including live source bindings."""
    identity = manifest["cache_identity"]
    from klab.data.packing import prepared_source_matches

    if not prepared_source_matches(identity, source_identity()["sha256"]):
        raise ValueError("prepared source identity differs from current source")
    tokenizer = load_tokenizer(config.tokenizer.path)
    if tokenizer.get_vocab_size() != config.model.vocab_size:
        raise ValueError("tokenizer vocabulary differs from config")
    expected = {
        "dataset": {
            k: v
            for k, v in config.dataset.model_dump(mode="json").items()
            if k not in _LOCATION_KEYS
        },
        "packing": packing_identity(config),
        "tokenizer_sha256": canonical_tokenizer_sha256(tokenizer),
        "local_chat_source": _local_chat_identity(config.dataset),
    }
    snapshot = None
    if config.dataset.source == "snapshot":
        from klab.data.sources import verify_snapshot

        snapshot = verify_snapshot(config.dataset)
        expected["snapshot_source_sha256"] = canonical_digest(snapshot)
        declared = snapshot["lock"]["source"]
        if (
            config.dataset.revision != declared["revision"]
            or config.dataset.license != declared["license"]
            or config.dataset.dataset_config not in {None, declared["config"]}
            or any(
                getattr(config.dataset, f"{split}_path").resolve()
                != (
                    config.dataset.source_manifest_path.parent
                    / snapshot["splits"][split]["path"]
                ).resolve()
                for split in ("train", "validation")
            )
        ):
            raise ValueError("snapshot source paths/provenance differ from config")
    if config.dataset.corpus_release_path is not None:
        from klab.corpus.export import verify_release_export

        expected["corpus_export"] = verify_release_export(config.dataset)
    if any(identity.get(k) != value for k, value in expected.items()):
        raise ValueError(
            "prepared dataset/tokenizer/packing/source binding differs from config"
        )
    return snapshot


def _block_counts(ids: np.ndarray, mask: np.ndarray | None, seq_len: int) -> dict:
    """Count the exact causal mask slices without allocating a block index list."""
    blocks, tail = divmod(max(0, len(ids) - 1), seq_len)
    usable = targets = 0
    if mask is None:
        usable, targets = blocks, blocks * seq_len
        tail_targets = tail
    else:
        chunk_blocks = max(1, _CHUNK_TOKENS // seq_len)
        for first in range(0, blocks, chunk_blocks):
            end = min(blocks, first + chunk_blocks)
            selected = mask[first * seq_len + 1 : end * seq_len + 1]
            counts = np.count_nonzero(selected.reshape(-1, seq_len), axis=1)
            usable += int(np.count_nonzero(counts))
            targets += int(counts.sum())
        tail_targets = int(np.count_nonzero(mask[blocks * seq_len + 1 :]))
    return {
        "seq_len": seq_len,
        "complete_blocks": blocks,
        "usable_blocks": usable,
        "zero_supervision_blocks": blocks - usable,
        "complete_block_target_positions": blocks * seq_len,
        "supervised_targets": targets,
        "initial_token_without_target": min(1, len(ids)),
        "dropped_tail_tokens": tail,
        "dropped_tail_supervised_targets": tail_targets,
    }


def _source_coverage(snapshot: dict | None, split: str, packed: dict) -> dict:
    # Only the generic, verifier-authenticated exhaustion contract can certify full.
    row = {} if snapshot is None else snapshot["splits"][split]
    acquired = row.get("source_records_consumed")
    retained = row.get("count")
    reasons = {
        key: row.get(key, 0)
        for key in ("duplicate", "overlap", "empty", "null", "text_bytes_limit")
    }
    excluded = (
        sum(reasons.values())
        if row and all(type(n) is int and n >= 0 for n in reasons.values())
        else None
    )
    counts_valid = all(
        type(n) is int and n >= 0 for n in (acquired, retained, excluded)
    )
    exhausted = bool(
        counts_valid
        and snapshot.get("format") == "klab-dataset-snapshot-v1"
        and acquired == retained + excluded
        and row.get("source_exhausted") is True
        and row.get("stop_reason") == "source_exhausted"
    )
    prepared_complete = bool(
        counts_valid
        and packed.get("acquired_documents") == retained
        and packed.get("retained_documents") == retained
        and packed.get("skipped_documents") == 0
        and packed.get("truncated_documents") == 0
    )
    full = exhausted and prepared_complete
    return {
        "full": full,
        "source_exhausted": exhausted if row else None,
        "all_retained_records_prepared": prepared_complete if row else None,
        "status": "full" if full else "unproven_or_bounded",
        "acquired_records": acquired,
        "retained_records": retained,
        "excluded_records": excluded,
        "exclusions": reasons if row else None,
        "stop_reason": row.get("stop_reason"),
        "snapshot_split": row or None,
    }


def coverage(prepared_root: Path, config: RunConfig | Path | None = None) -> dict:
    """Read an authenticated report; block exposure needs a sequence-length config.

    Without a config, historical prepared identities are reported as recorded;
    no assertion of current source compatibility is made. Embedded source coverage
    is reported as recorded and authenticated against the packed source binding.
    """
    root = Path(prepared_root)
    if isinstance(config, (str, Path)):
        config = load_config(Path(config))
    header = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    data = load_prepared_data(
        root, byte_enabled=header.get("byte_addressing") is not None
    )
    manifest = data.manifest
    snapshot = manifest.get("dataset_snapshot")
    if snapshot is not None and (
        not isinstance(snapshot, dict)
        or canonical_digest(snapshot)
        != manifest["cache_identity"].get("snapshot_source_sha256")
    ):
        raise ValueError("embedded snapshot source binding mismatch")
    if config is not None:
        verified_snapshot = _bind_config(manifest, config)
        if snapshot is not None and snapshot != verified_snapshot:
            raise ValueError("embedded snapshot differs from verified source")
        snapshot = verified_snapshot
    splits = {}
    for split in ("train", "validation"):
        ids = getattr(data, split)
        mask = getattr(data, f"{split}_supervision")
        splits[split] = {
            "source_coverage": _source_coverage(snapshot, split, manifest[split]),
            "preparation": manifest[split],
            "packed_tokens": len(ids),
            "blocks": None
            if config is None
            else _block_counts(ids, mask, config.training.seq_len),
        }
    return {
        "format": "klab-dataset-coverage-v1",
        "prepared_root": str(root.absolute()),
        "manifest_sha256": manifest["manifest_sha256"],
        "settings_sha256": manifest["settings_sha256"],
        "source_identity_sha256": manifest["cache_identity"]["source_identity_sha256"],
        "source": manifest["source"],
        "objective": manifest.get("supervision", {"kind": "all_tokens"})["kind"],
        "config_verified": config is not None,
        "splits": splits,
    }


def _absolute_paths(value: Any, base: Path) -> Any:
    # Use typed Path values, including optional paths absent from loader key lists.
    if isinstance(value, Path):
        return str((value if value.is_absolute() else base / value).absolute())
    if isinstance(value, dict):
        return {key: _absolute_paths(item, base) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_absolute_paths(item, base) for item in value]
    return value


def derive_budget(
    config_path: Path,
    prepared_root: Path,
    passes: int,
    output: Path,
    require_full: bool = True,
) -> dict:
    """Exclusively write a validated fresh-run config and return its derivation."""
    if type(passes) is not int or passes <= 0:
        raise ValueError("passes must be a positive integer")
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"budget output already exists: {output}")
    config_path = Path(config_path).absolute()
    config = load_config(config_path)
    if (
        config.dataset.allocation_manifest_path is not None
        or config.training.neural_loss_weight != 1
    ):
        raise ValueError(
            "budget derivation does not support allocation or weighted objectives"
        )
    report = coverage(prepared_root, config)
    if require_full and not all(
        row["source_coverage"]["full"] for row in report["splits"].values()
    ):
        raise ValueError("full source coverage is incomplete, unsupported, or unproven")
    train = report["splits"]["train"]["blocks"]
    validation = report["splits"]["validation"]["blocks"]
    if not train["supervised_targets"] or not validation["supervised_targets"]:
        raise ValueError("training and validation need supervised complete blocks")
    batch = config.training.micro_batch_size * config.training.gradient_accumulation
    blocks = passes * train["usable_blocks"]
    steps = (blocks + batch - 1) // batch
    targets = passes * train["supervised_targets"]
    payload = _absolute_paths(config.model_dump(mode="python"), config_path.parent)
    payload["training"].update(max_steps=steps, max_tokens=targets)
    # Reject incompatible schedules rather than silently changing another setting.
    derived = RunConfig.model_validate(payload)
    content = yaml.safe_dump(derived.model_dump(mode="json"), sort_keys=False)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(content)
    return {
        "format": "klab-dataset-budget-v1",
        "transition": "fresh",
        "config_path": str(config_path),
        "output": str(output.absolute()),
        "passes": passes,
        "require_full": require_full,
        "effective_batch": batch,
        "max_steps": steps,
        "max_tokens": targets,
        "previous_budget": {
            "max_steps": config.training.max_steps,
            "max_tokens": config.training.max_tokens,
        },
        "final_update_blocks": (blocks - 1) % batch + 1,
        "coverage": report,
    }


def _handle(args: argparse.Namespace) -> None:
    try:
        with redirect_stdout(sys.stderr):
            result = (
                coverage(args.prepared_root, args.config)
                if args.coverage_command == "coverage"
                else derive_budget(
                    args.config,
                    args.prepared_root,
                    args.passes,
                    args.output,
                    args.require_full,
                )
            )
    except (OSError, ValueError, TypeError, KeyError) as error:
        if args.json:
            print(json.dumps({"status": "blocked", "error": str(error)}))
        else:
            print(f"Data {args.coverage_command} blocked: {error}", file=sys.stderr)
        raise SystemExit(1) from error
    if args.json:
        print(json.dumps(result, sort_keys=True))
    else:
        report = result.get("coverage", result)
        for split, row in report["splits"].items():
            print(
                f"{split}: source={row['source_coverage']['status']}; packed_tokens={row['packed_tokens']}"
            )
            source = row["source_coverage"]
            print(
                "  source: "
                + ", ".join(
                    f"{k}={source[k]}"
                    for k in (
                        "acquired_records",
                        "retained_records",
                        "excluded_records",
                        "stop_reason",
                    )
                )
            )
            print(
                "  preparation: "
                + ", ".join(
                    f"{k}={row['preparation'].get(k)}"
                    for k in (
                        "acquired_documents",
                        "retained_documents",
                        "skipped_documents",
                        "truncated_documents",
                    )
                )
            )
            if row["blocks"] is not None:
                print("  " + ", ".join(f"{k}={v}" for k, v in row["blocks"].items()))
        if "max_steps" in result:
            print(
                f"Fresh budget: max_steps={result['max_steps']}, max_tokens={result['max_tokens']}; {result['output']}"
            )


def register_parser(data_commands: argparse._SubParsersAction) -> None:
    """Register native data subcommands; caller dispatches args.handler(args)."""
    for name in ("coverage", "budget"):
        command = data_commands.add_parser(name, help=f"Verified dataset {name}")
        command.add_argument("--prepared-root", required=True, type=Path)
        command.add_argument("--config", type=Path, required=name == "budget")
        command.add_argument("--json", action="store_true")
        if name == "budget":
            command.add_argument("--passes", required=True, type=int)
            command.add_argument("--output", required=True, type=Path)
            command.add_argument(
                "--require-full", action=argparse.BooleanOptionalAction, default=True
            )
        command.set_defaults(handler=_handle, coverage_command=name)
