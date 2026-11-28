"""Inspect mixture lineage pinned by a run's own preparation manifest."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.admission_inspection_cache import capture, unchanged
from klab.training.manifest import read_manifest
from klab.verification_proofs import (
    SMALL_METADATA_LIMIT,
    ProofStore,
    read_trusted_small,
)

_PREPARED = "data/manifest.json"
_KEY = re.compile(r"[0-9a-f]{64}")


def _require_digest(value):
    if not isinstance(value, str) or not _KEY.fullmatch(value):
        raise ValueError("recorded mixture requires full SHA-256 identities")
    return value


def recorded_mixture_lineage(runs_dir: Path, run_id: str) -> dict:
    """Read the pinned recipe and release evidence without opening run payloads."""
    if not run_id or run_id in {".", ".."} or Path(run_id).name != run_id:
        raise ValueError("invalid run ID")
    root = runs_dir.resolve() / run_id
    run_path = root / "manifest.json"
    prepared_path = root / _PREPARED
    if (
        not ProofStore(root, read_only=True).trusted(run_path)
        or run_path.stat().st_size > SMALL_METADATA_LIMIT
        or not run_path.is_file()
    ):
        raise ValueError("missing or unsafe run manifest")
    initial = capture([run_path])
    run = read_manifest(run_path)
    dataset = run["effective_config"]["dataset"]
    if run["run_id"] != run_id or dataset.get("source") != "local_token_mixture":
        raise ValueError("selected run does not pin a Corpus token mixture")
    expected = next(
        (row for row in run["artifacts"] if row["relative_path"] == _PREPARED), None
    )
    if expected is None:
        raise ValueError("run has no pinned preparation manifest")
    if not prepared_path.is_file():
        raise ValueError("run's pinned preparation manifest is missing")
    initial.update(capture([prepared_path]))
    raw = read_trusted_small(prepared_path, root)
    digest = hashlib.sha256(raw).hexdigest()
    if len(raw) != expected["size_bytes"] or digest != expected["sha256"]:
        raise ValueError("run's preparation manifest differs from its recorded pin")
    prepared = json.loads(raw)
    receipt = prepared["token_mixture"]
    release_id = _require_digest(receipt["release_id"])
    if release_id != prepared["revision"] or release_id != dataset.get("revision"):
        raise ValueError(
            "recorded mixture release differs from the prepared data and run"
        )

    binding = receipt.get("additive_release_binding")
    base_id = release_id
    additions = []
    if binding is not None:
        base_id = _require_digest(binding["base_release_id"])
        release_ids = [
            _require_digest(key) for key in binding["additional_release_ids"]
        ]
        evidence = binding["evidence"]
        if (
            not release_ids
            or len({base_id, *release_ids}) != len(release_ids) + 1
            or [row["release_id"] for row in evidence] != release_ids
            or canonical_digest(
                {"base_release_id": base_id, "additional_release_ids": release_ids}
            )
            != release_id
        ):
            raise ValueError("invalid recorded additive mixture binding")
        for row in evidence:
            prior = row["prior"]
            if prior["release_id"] != base_id:
                raise ValueError(
                    "recorded additive release names a different prior release"
                )
            additions.append(
                {
                    "release_id": row["release_id"],
                    "prior_inventory_sha256": _require_digest(
                        prior["inventory_sha256"]
                    ),
                }
            )
    unchanged(initial)
    return {
        "run_id": run_id,
        "run_manifest_sha256": canonical_digest(run),
        "prepared_manifest_sha256": digest,
        "mixture_receipt_sha256": _require_digest(
            prepared["cache_identity"]["token_mixture_receipt_sha256"]
        ),
        "declaration_sha256": _require_digest(receipt["declaration_sha256"]),
        "tokenizer_sha256": _require_digest(receipt["tokenizer_sha256"]),
        "mixture_revision": release_id,
        "base_release_id": base_id,
        "base_manifest_sha256": _require_digest(receipt["release_manifest_sha256"]),
        "additional_releases": additions,
        "scheduled_target_tokens": receipt["actual_target_tokens"],
    }
