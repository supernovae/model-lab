"""Export an exact effective RunConfig from an authenticated experiment lock."""

from __future__ import annotations

from pathlib import Path

import yaml

from klab.experiments.lock import _exclusive_bytes, open_lock
from klab.verification_proofs import ProofStore, VerificationMode


def export_effective_config(
    lock: Path,
    cell_id: str,
    output: Path,
    *,
    proof_store: ProofStore | None = None,
    verification_mode: VerificationMode = "cold",
) -> str:
    """Verify lock and inputs, then exclusively write exactly one cell's config."""
    output = output.absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"effective config already exists: {output}")
    locked = open_lock(
        lock.absolute(), proof_store=proof_store, verification_mode=verification_mode
    )
    selected = [cell for cell in locked.cells if cell.id == cell_id]
    if len(selected) != 1:
        raise ValueError(
            f"unknown cell {cell_id!r}; select an exact lock cell ID: "
            f"{[cell.id for cell in locked.cells]}"
        )
    cell = selected[0]
    content = yaml.safe_dump(
        cell.config.model_dump(mode="json"), sort_keys=False
    ).encode("utf-8")
    _exclusive_bytes(output, content)
    return cell.config_sha256
