"""Conservative filesystem headroom checks before durable run publication."""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from klab.config.models import RunConfig, TokenizerTrainConfig
from klab.workdir import resolve_work_dir

if TYPE_CHECKING:
    from klab.data.verification import VerifiedPreparedData


_RESERVE_BYTES = 256 * 1024 * 1024
_RESERVE_INODES = 128
# A mixture receipt never exceeds its writer's cap (corpus.mixture_index).
_MAX_MIXTURE_RECEIPT_BYTES = 256 * 1024**2


@dataclass(frozen=True)
class StorageCheck:
    path: str
    filesystem_path: str
    available_bytes: int
    available_inodes: int | None
    projected_bytes: int
    projected_inodes: int
    reserve_bytes: int
    reserve_inodes: int
    status: str


def _existing_ancestor(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        if candidate == candidate.parent:
            raise FileNotFoundError(f"no existing ancestor for {path}")
        candidate = candidate.parent
    if not candidate.is_dir():
        raise NotADirectoryError(candidate)
    return candidate


def check_storage(
    path: Path,
    *,
    projected_bytes: int,
    projected_inodes: int = 64,
    reserve_bytes: int = _RESERVE_BYTES,
    reserve_inodes: int = _RESERVE_INODES,
) -> StorageCheck:
    """Check a path's actual filesystem without creating the destination."""
    if min(projected_bytes, projected_inodes, reserve_bytes, reserve_inodes) < 0:
        raise ValueError("storage projections and reserves must be non-negative")
    ancestor = _existing_ancestor(path)
    stats = os.statvfs(ancestor)
    available_bytes = stats.f_bavail * stats.f_frsize
    available_inodes = stats.f_favail if stats.f_files else None
    enough_bytes = available_bytes >= projected_bytes + reserve_bytes
    enough_inodes = (
        available_inodes is None
        or available_inodes >= projected_inodes + reserve_inodes
    )
    return StorageCheck(
        path=str(path.expanduser().resolve()),
        filesystem_path=str(ancestor),
        available_bytes=available_bytes,
        available_inodes=available_inodes,
        projected_bytes=projected_bytes,
        projected_inodes=projected_inodes,
        reserve_bytes=reserve_bytes,
        reserve_inodes=reserve_inodes,
        status="adequate" if enough_bytes and enough_inodes else "insufficient",
    )


def require_storage(checks: list[StorageCheck]) -> list[dict[str, object]]:
    failures = [item for item in checks if item.status == "insufficient"]
    if failures:
        details = "; ".join(
            f"{item.path}: free {item.available_bytes} bytes / "
            f"{item.available_inodes} inodes, need {item.projected_bytes + item.reserve_bytes} "
            f"bytes / {item.projected_inodes + item.reserve_inodes} inodes"
            for item in failures
        )
        raise OSError(
            f"workspace storage preflight failed before publication: {details}"
        )
    return [asdict(item) for item in checks]


def _train_positions(config: RunConfig) -> int:
    """Packed training positions: a document_start mixture adds masked filler.

    Its receipt records the packed rows, so the allowance counts every row
    rather than only the scheduled targets. Packing verifies the receipt; this
    reads it for sizing only.
    """
    dataset = config.dataset
    receipt = (
        dataset.mixture_output_path / "receipt.json"
        if dataset.source == "local_token_mixture"
        and dataset.mixture_output_path is not None
        else None
    )
    if receipt is None or not receipt.is_file():
        return dataset.train_max_tokens
    with receipt.open("rb") as stream:
        layout = json.loads(stream.read(_MAX_MIXTURE_RECEIPT_BYTES)).get("row_layout")
    if layout is None:
        return dataset.train_max_tokens
    # One BOS precedes the rows.
    return max(dataset.train_max_tokens, layout["rows"] * layout["row_tokens"]) + 1


def projected_data_bytes(config: RunConfig) -> int:
    """Planning allowance for prepared arrays, sized by the arrays a config has.

    Token IDs are int32 and always present. A loss mask is one byte per token
    and is allowed for every source, byte addresses are int32 with byte-addressed
    memory. The sum is doubled because staging and the published arrays
    coexist during finalization, then given a quarter of headroom.
    """
    targets = _train_positions(config) + config.dataset.validation_max_tokens
    per_token = 4 + 1
    if config.model.memory in {"byte", "portable"}:
        per_token += 4
    return math.ceil(1.25 * 2 * targets * per_token) + 32 * 1024 * 1024


def verified_prepared_footprint(receipt: VerifiedPreparedData) -> tuple[int, int]:
    """Measure the complete copied inventory after authenticating sealed data proof."""
    from klab.data.verification import (
        _SEAL,
        VerifiedPreparedData,
        _receipt_from_proofs,
    )

    if (
        not isinstance(receipt, VerifiedPreparedData)
        or getattr(receipt, "_seal", None) is not _SEAL
    ):
        raise TypeError("prepared inventory requires a sealed verification receipt")
    manifest = json.loads((receipt.root / "manifest.json").read_text())
    verified = _receipt_from_proofs(receipt.root, manifest, dict(receipt.proofs))
    if verified.manifest_sha256 != receipt.manifest_sha256:
        raise ValueError("prepared receipt manifest changed")
    total_bytes = 0
    files = 0
    for member in receipt.root.rglob("*"):
        if member.is_symlink():
            raise ValueError("symlink in prepared storage inventory")
        if member.is_file():
            total_bytes += member.stat().st_size
            files += 1
        elif not member.is_dir():
            raise ValueError("nonregular prepared storage inventory member")
    return total_bytes, files


def training_storage_checks(
    config: RunConfig,
    *,
    work_dir: Path | None = None,
    run_dir: Path | None = None,
    checkpoint_generations_upper: int | None = None,
    completed_steps: int = 0,
    completed_tokens: int = 0,
    verified_prepared: VerifiedPreparedData | None = None,
) -> list[StorageCheck]:
    """Reserve incremental growth on each device, not already allocated cache bytes.

    Training copies the prepared arrays into its run even when the cache exists.
    Without in-process proof, budget both a new cache and the eventual run copy.
    """
    if checkpoint_generations_upper is None:
        from klab.experiments.storage import checkpoint_generation_bounds

        checkpoint_generations_upper = checkpoint_generation_bounds(
            config, completed_steps=completed_steps, completed_tokens=completed_tokens
        )["peak_generations"]
    if checkpoint_generations_upper < 1:
        raise ValueError("checkpoint generation count must be positive")
    from klab.model.inspection import inspection_report

    checkpoint = int(inspection_report(config)["estimated_checkpoint_bytes"])
    checkpoint_growth = checkpoint_generations_upper * math.ceil(1.25 * checkpoint)
    policy = config.checkpoint.retention
    if policy is not None:
        checkpoint_growth = max(
            checkpoint_growth, policy.max_peak_bytes + policy.reserve_free_bytes
        )
    if verified_prepared is None:
        run_copy = projected_data_bytes(config)
        cache_growth = run_copy
    else:
        from klab.data.packing import (
            canonical_tokenizer_sha256,
            packing_identity,
            prepared_dataset_identity,
        )
        from klab.data.tokenizer import load_tokenizer
        from klab.data.verification import (
            VerifiedPreparedData,
            _receipt_from_proofs,
        )

        if not isinstance(verified_prepared, VerifiedPreparedData):
            raise TypeError("prepared inventory requires a sealed verification receipt")
        root = verified_prepared.root
        cache_root = config.dataset.cache_dir.resolve()
        if not root.is_relative_to(cache_root):
            raise ValueError("prepared receipt is outside configured cache")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        proof = _receipt_from_proofs(root, manifest, dict(verified_prepared.proofs))
        if proof.manifest_sha256 != verified_prepared.manifest_sha256:
            raise ValueError("prepared receipt manifest changed")
        identity = manifest.get("cache_identity")
        expected_dataset = prepared_dataset_identity(config.dataset)
        expected_packing = packing_identity(config)
        if (
            not isinstance(identity, dict)
            or identity.get("dataset") != expected_dataset
            or identity.get("packing") != expected_packing
            or identity.get("tokenizer_sha256")
            != canonical_tokenizer_sha256(load_tokenizer(config.tokenizer.path))
        ):
            raise ValueError("prepared receipt does not match run configuration")
        run_copy, _ = verified_prepared_footprint(verified_prepared)
        cache_growth = 0
    destinations = (
        (
            run_dir if run_dir is not None else config.logging.root_dir,
            checkpoint_growth + run_copy,
            checkpoint_generations_upper * 16 + 128,
        ),
        (config.dataset.cache_dir, cache_growth, 128 if cache_growth else 0),
        (
            work_dir if work_dir is not None else resolve_work_dir(),
            max(64 * 1024 * 1024, checkpoint // 4),
            128,
        ),
    )
    grouped: dict[int, tuple[Path, int, int]] = {}
    for path, bytes_needed, inodes_needed in destinations:
        ancestor = _existing_ancestor(path)
        device = ancestor.stat().st_dev
        prior = grouped.get(device)
        if prior is None:
            grouped[device] = (path, bytes_needed, inodes_needed)
        else:
            grouped[device] = (
                prior[0],
                prior[1] + bytes_needed,
                prior[2] + inodes_needed,
            )
    return [
        check_storage(
            path, projected_bytes=bytes_needed, projected_inodes=inodes_needed
        )
        for path, bytes_needed, inodes_needed in grouped.values()
    ]


def tokenizer_storage_checks(config: TokenizerTrainConfig) -> list[StorageCheck]:
    # Input text is bounded, but a remote source's download cache is not. Keep
    # that uncertainty visible in documentation; this checks known local growth.
    # Match the trainer's fitting budget: a stratified sample does not fit the
    # whole release merely because its dataset retains the full corpus bound.
    input_bytes = (
        config.sampling.max_fit_bytes
        if config.sampling is not None
        else config.dataset.train_max_tokens
    )
    known_bytes = max(64 * 1024 * 1024, input_bytes * 2)
    if (
        _existing_ancestor(config.output_dir).stat().st_dev
        == _existing_ancestor(config.dataset.cache_dir).stat().st_dev
    ):
        return [check_storage(config.output_dir, projected_bytes=2 * known_bytes)]
    return [
        check_storage(config.output_dir, projected_bytes=known_bytes),
        check_storage(config.dataset.cache_dir, projected_bytes=known_bytes),
    ]
