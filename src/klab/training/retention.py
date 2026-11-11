"""Finite local full-state keep-sets; no archive, transfer or disk-quota service."""

from __future__ import annotations

import shutil
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING

from klab.config.models import CheckpointRetention
from klab.training.mlx_checkpoints import strict_json

if TYPE_CHECKING:
    from klab.training.checkpoints import CheckpointRecord


def footprint(root: Path) -> int:
    """Observe all checkpoint-root bytes, including interrupted staging, boundedly."""
    total = count = 0
    for path in root.iterdir():
        count += 1
        if count > 65536 or path.is_symlink():
            raise ValueError("unsafe retention path")
        members = path.iterdir() if path.is_dir() else (path,)
        for member in members:
            count += 1
            if count > 65536 or member.is_symlink() or not member.is_file():
                raise ValueError("unsafe or oversized retention inventory")
            total += member.stat().st_size
    return total


def inherited_milestones(
    parent: Path, policy: CheckpointRetention, step: int, tokens: int
) -> list[dict]:
    """Bind reached child boundaries to existing parent selections, without copying."""
    reached = [
        item
        for item in policy.milestones
        if step >= item.step and tokens >= item.committed_targets
    ]
    if not reached:
        return []
    path = parent / "checkpoints/retention.json"
    if path.is_symlink() or not path.is_file():
        raise ValueError("resume lacks verified parent milestone selections")
    evidence = strict_json(path)
    declarations = {item["name"]: item for item in evidence["policy"]["milestones"]}
    selected: dict[str, dict] = {}
    for item in reached:
        role = f"milestone:{item.name}"
        if declarations.get(item.name) != item.model_dump(mode="json"):
            raise ValueError("resumed milestone differs from parent declaration")
        matches = [
            row
            for row in evidence["retained"]
            if role in row["roles"]
            and row["step"] <= step
            and row["tokens_seen"] <= tokens
        ]
        if len(matches) != 1:
            raise ValueError(
                "resume lacks a milestone at or before selected checkpoint"
            )
        row = matches[0]
        location = row["verified_location"]
        target = selected.setdefault(
            location,
            {
                "verified_location": location,
                "manifest_sha256": row["manifest_sha256"],
                "step": row["step"],
                "tokens_seen": row["tokens_seen"],
                "roles": [],
                "local": False,
            },
        )
        target["roles"].append(role)
    return list(selected.values())


def inherited_rows(root: Path) -> list[dict]:
    from klab.training.checkpoints import CheckpointManager

    path = root / "retention-inherited.json"
    if not path.exists():
        return []
    if path.is_symlink():
        raise ValueError("unsafe inherited retention references")
    references = strict_json(path)["references"]
    if len(references) > 128:
        raise ValueError("oversized inherited retention references")
    rows = []
    for reference in references:
        directory = Path(reference["verified_location"])
        manager = CheckpointManager(directory.parent.parent)
        verified = manager.verify(directory)
        manifest = strict_json(directory / "manifest.json")
        if not verified.valid or manifest["sha256"] != reference["manifest_sha256"]:
            raise ValueError("inherited milestone is missing or unverified")
        record = manager._record_from_manifest(directory, manifest)
        if (
            record.step != reference["step"]
            or record.tokens_seen != reference["tokens_seen"]
        ):
            raise ValueError(
                "inherited milestone counters differ from verified checkpoint"
            )
        rows.append(
            {
                **asdict(record),
                **reference,
                "source_identity_sha256": manifest.get("source_identity_sha256"),
                "full_directory_bytes": sum(
                    member.stat().st_size for member in directory.iterdir()
                ),
            }
        )
    return rows


def admit_write(root: Path, policy: CheckpointRetention) -> dict:
    existing = footprint(root) + sum(
        row["full_directory_bytes"] for row in inherited_rows(root)
    )
    required = (
        max(policy.max_generation_bytes, policy.max_peak_bytes - existing)
        + policy.reserve_free_bytes
    )
    available = shutil.disk_usage(root).free
    if existing + policy.max_generation_bytes > policy.max_peak_bytes:
        raise ValueError(
            "checkpoint peak-byte plan cannot fit another staged generation"
        )
    if available < required:
        raise ValueError("insufficient free bytes for checkpoint staging and reserve")
    return {
        "observed_root_bytes": existing,
        "available_bytes": available,
        "required_free_bytes": required,
        "projected_peak_bytes": existing + policy.max_generation_bytes,
    }


def plan(
    root: Path, records: list[CheckpointRecord], policy: CheckpointRetention
) -> dict:
    """Select only manager-verified generations, refusing a missing previous keep-set."""
    records = sorted(records, key=lambda row: (row.step, row.generation_id))
    inherited = inherited_rows(root)
    inherited_identities = {
        row["verified_location"]: row["manifest_sha256"] for row in inherited
    }
    identities = {row.relative_path: row.manifest_sha256 for row in records}
    previous = root / "retention.json"
    if previous.exists():
        if previous.is_symlink():
            raise ValueError("unsafe retention report")
        for row in strict_json(previous)["retained"]:
            observed = (
                identities.get(row["relative_path"])
                if row.get("local", True)
                else inherited_identities.get(row["verified_location"])
            )
            if observed != row["manifest_sha256"]:
                raise ValueError(
                    "previously retained checkpoint is missing or unverified"
                )
    roles: dict[str, list[str]] = {}

    def keep(record: CheckpointRecord, role: str) -> None:
        roles.setdefault(record.relative_path, []).append(role)

    if records:
        keep(records[-1], "latest")
    if len(records) > 1:
        keep(records[-2], "previous")
    finite = [row for row in records if row.validation_loss is not None]
    if finite:
        keep(
            min(finite, key=lambda row: (row.validation_loss, row.generation_id)),
            "best",
        )
    inherited_roles = {role for row in inherited for role in row["roles"]}
    pending = []
    for milestone in policy.milestones:
        if f"milestone:{milestone.name}" in inherited_roles:
            continue
        selected = next(
            (
                row
                for row in records
                if row.step >= milestone.step
                and row.tokens_seen >= milestone.committed_targets
            ),
            None,
        )
        if selected is not None:
            keep(selected, f"milestone:{milestone.name}")
        else:
            pending.append(milestone.model_dump(mode="json"))
    terminal = next(
        (
            row
            for row in records
            if row.step >= policy.terminal_step
            or row.tokens_seen >= policy.terminal_targets
        ),
        None,
    )
    if terminal is not None:
        keep(terminal, "terminal")
    retained = list(inherited)
    if any(
        row["full_directory_bytes"] > policy.max_generation_bytes for row in inherited
    ):
        raise ValueError("inherited checkpoint exceeds declared per-generation bytes")
    for record in records:
        if record.relative_path not in roles:
            continue
        directory = root / record.relative_path
        manifest = strict_json(directory / "manifest.json")
        size = sum(member.stat().st_size for member in directory.iterdir())
        if size > policy.max_generation_bytes:
            raise ValueError("checkpoint exceeds declared per-generation bytes")
        retained.append(
            {
                **asdict(record),
                "local": True,
                "roles": sorted(roles[record.relative_path]),
                "verified_location": str(directory.resolve()),
                "source_identity_sha256": manifest.get("source_identity_sha256"),
                "full_directory_bytes": size,
            }
        )
    total = sum(row["full_directory_bytes"] for row in retained)
    if len(retained) > policy.max_generations or total > policy.max_retained_bytes:
        raise ValueError("required checkpoint keep-set exceeds retention limits")
    observed = footprint(root) + sum(row["full_directory_bytes"] for row in inherited)
    if observed > policy.max_peak_bytes:
        raise ValueError("observed checkpoint bytes exceed peak plan; refusing pruning")
    return {
        "format": "klab-checkpoint-retention-v1",
        "status": "verified",
        "milestones_complete": not pending,
        "pending_milestones": pending,
        "terminal_reached": terminal is not None,
        "policy": policy.model_dump(mode="json"),
        "planned_retained_bytes_upper": policy.retained_upper
        * policy.max_generation_bytes,
        "planned_generation_count_upper": policy.retained_upper,
        "retained": retained,
        "retained_bytes": total,
        "observed_peak_bytes": observed,
        "prune": [
            row.relative_path for row in records if row.relative_path not in roles
        ],
        "storage_semantics": "admission_and_postwrite_checks_not_filesystem_quota",
    }
