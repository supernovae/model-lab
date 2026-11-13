"""One exact pinned-transfer declaration, using the full project's existing ledger."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

from klab.corpus import bounded_metadata
from klab.corpus.project import Project
from klab.corpus.transport_budget import (
    PINNED_FILE_KINDS,
    TransportBudget,
    pinned_transfers,
)

MANIFEST_BYTES = 256 * 1024**2
LEDGER_BYTES = 32 * 1024**2
JOURNAL_BYTES = 64 * 1024**2


def _files(root: Path) -> int:
    """Bounded no-follow metadata walk; no source data is read or copied."""
    pending = [(root, 0)]
    count = total = 0
    while pending:
        directory, depth = pending.pop()
        if depth > 32:
            raise ValueError("acquisition inventory depth exceeded")
        with os.scandir(directory) as entries:
            for entry in entries:
                count += 1
                if count > 100000:
                    raise ValueError("acquisition inventory entry cap exceeded")
                info = entry.stat(follow_symlinks=False)
                if stat.S_ISREG(info.st_mode):
                    total += info.st_size
                elif stat.S_ISDIR(info.st_mode):
                    pending.append((Path(entry.path), depth + 1))
                else:
                    raise ValueError("unsafe acquisition inventory path")
    return total


def prepare(
    project: Project,
    work_root: Path,
    source_id: str,
    *,
    max_new_bytes: int | None,
    min_free_bytes: int,
    proof_store,
    verification_mode,
):
    from klab.corpus import acquisition as native

    if not hasattr(os, "statvfs"):
        raise ValueError(
            "bounded acquisition steps require POSIX filesystem statistics"
        )

    if (
        type(source_id) is not str
        or not source_id
        or source_id not in {s.id for s in project.sources}
    ):
        raise ValueError("exact declared source ID required")
    if type(min_free_bytes) is not int or min_free_bytes < 0:
        raise ValueError("nonnegative explicit free-space floor required")
    if (
        proof_store is not None
        or verification_mode != "cold"
        or bounded_metadata._ACTIVE_STORE.get() is not None
    ):
        raise ValueError(
            "acquisition steps require cold verification without metadata scratch"
        )
    source = next(s for s in project.sources if s.id == source_id)
    if source.redistribution == "rejected":
        raise ValueError("acquisition step source is rejected")
    if source.kind not in PINNED_FILE_KINDS and (
        source.kind != "huggingface_dataset"
        or source.acquisition.bounded_shards is None
        or source.acquisition.retained_subset is not None
    ):
        raise ValueError(
            "acquisition steps require a bounded-HF, archive, PDF or MediaWiki declaration"
        )
    if project.config.transport_budget is None:
        raise ValueError(
            "acquisition step requires the existing full-project transport ledger"
        )
    base = work_root / "corpora" / project.config.id
    native._safe_snapshot_path(base.absolute())
    if not base.is_dir():
        raise ValueError("initialize the full-project transport ledger before a step")
    ledger = base / "transport-budget.sqlite"
    for name, limit in (
        (ledger, LEDGER_BYTES),
        (Path(str(ledger) + "-journal"), JOURNAL_BYTES),
    ):
        if (name.exists() or name.is_symlink()) and (
            name.is_symlink() or not name.is_file() or name.stat().st_size > limit
        ):
            raise ValueError("acquisition step transport metadata cap exceeded")
    if any(
        Path(str(ledger) + suffix).exists() or Path(str(ledger) + suffix).is_symlink()
        for suffix in ("-wal", "-shm")
    ):
        raise ValueError("acquisition step requires a closed rollback-journal ledger")
    journal = Path(str(ledger) + "-journal")
    if journal.exists():
        with journal.open("rb") as stream:
            if any(stream.read(8)):
                raise ValueError("acquisition step requires a clean rollback journal")
    budget = TransportBudget(ledger, project)
    observes_storage = budget.observes_storage
    if observes_storage:
        if max_new_bytes is not None:
            raise ValueError("observational storage does not accept a write allowance")
        if min_free_bytes < budget.receipt()["storage_observation"]["min_free_bytes"]:
            raise ValueError("cannot lower the recorded storage free-space floor")
    elif type(max_new_bytes) is not int or max_new_bytes <= 0:
        raise ValueError("positive remaining native write allowance required")
    if len(json.dumps(budget.receipt()).encode()) > 16 * 1024**2:
        raise ValueError("acquisition step transport metadata cap exceeded")
    effects = {x.source_id: x for x in project.config.source_effects or ()}
    retained = {}
    # Preserve complete origin preflight before any transfer or publication.
    for item in project.sources:
        effect = effects.get(item.id)
        if effect is not None and effect.effect == "reuse_only":
            origin = native._retained_origin(work_root, effect)
            if (origin / "manifest.json").stat().st_size > MANIFEST_BYTES:
                raise ValueError("acquisition step manifest byte cap exceeded")
            retained[item.id] = native._verify_retained(
                project,
                work_root,
                item,
                effect,
                proof_store=None,
                verification_mode="cold",
            )
    existing = None
    parent = base / "snapshots" / source.id
    if parent.exists():
        native._safe_snapshot_path(parent.absolute())
        candidates = []
        for candidate in parent.iterdir():
            candidates.append(candidate)
            if len(candidates) > 128:
                raise ValueError("acquisition snapshot candidate cap exceeded")
        candidates.sort()
        for candidate in candidates:
            if candidate.name.startswith("."):
                continue
            native._safe_snapshot_path(candidate.absolute())
            if (candidate / "manifest.json").stat().st_size > MANIFEST_BYTES:
                raise ValueError("acquisition step manifest byte cap exceeded")
            manifest = native.verify_snapshot(candidate, verification_mode="cold")
            if (
                manifest["declaration_sha256"] == native.declaration_sha256(source)
                and manifest["declaration"] == native.source_declaration_payload(source)
                and native._reusable_immutable_adapter(source, manifest)
                and (source_id not in retained or manifest == retained[source_id][1])
            ):
                existing = candidate
                break
    effect = effects.get(source_id)
    if existing is not None:
        output_bytes = compressed = 0
        manifest_bytes = 0
    elif effect is not None and effect.effect == "reuse_only":
        origin, manifest = retained[source_id]
        output_bytes = sum(row["size"] for row in manifest["files"])
        manifest_bytes = (origin / "manifest.json").stat().st_size
        compressed = 0
    else:
        output_bytes = source.acquisition.max_bytes
        manifest_bytes = MANIFEST_BYTES
        # PDF bodies are kept in the snapshot and so are already output bytes;
        # other bodies occupy one temporary path until they are expanded.
        compressed = (
            0
            if source.kind == "pdf_documents"
            else sum(size for _, size in pinned_transfers(source))
        )
        budget.assert_source_available(source_id)
    # Expanded data + manifest appear at staging and published paths. Each
    # compressed body has one temporary path. Stable capped SQLite/journal paths
    # are additional native overhead, independent of external job logs/receipts.
    path_bytes = (
        2 * (output_bytes + manifest_bytes) + compressed + LEDGER_BYTES + JOURNAL_BYTES
    )
    peak_bytes = (
        output_bytes
        + manifest_bytes
        + max((size for _, size in pinned_transfers(source)), default=0)
        * (compressed != 0)
        + LEDGER_BYTES
        + JOURNAL_BYTES
    )
    current_bytes = _files(base)
    free = os.statvfs(base).f_bavail * os.statvfs(base).f_frsize
    if not observes_storage and path_bytes > max_new_bytes:
        raise ValueError("insufficient remaining native cumulative write allowance")
    if not observes_storage and current_bytes + peak_bytes > budget.spec.max_disk_bytes:
        raise ValueError("transport budget disk allowance exhausted")
    if free < min_free_bytes + (0 if observes_storage else peak_bytes):
        raise ValueError("insufficient acquisition step disk headroom")
    return (
        source,
        budget,
        retained,
        {
            "project_id": project.config.id,
            "project_sha256": native._project_sha(project),
            "source_id": source_id,
            "declaration_sha256": native.declaration_sha256(source),
            "transport_budget_binding": budget.binding,
            "existing_snapshot": str(existing) if existing else None,
            "native_new_path_bytes": path_bytes,
            "native_peak_added_bytes": peak_bytes,
            "existing_project_bytes": current_bytes,
            "free_bytes": free,
            "min_free_bytes": min_free_bytes,
            "max_new_bytes": max_new_bytes,
            "storage_mode": "observational" if observes_storage else "enforced",
            "external_logs_receipts_and_baseline_are_additional": True,
        },
    )
