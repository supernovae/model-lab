"""Bounded queries over verified native duplicate evidence and decisions."""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import ExitStack, closing, contextmanager
from pathlib import Path

from klab.canonical import canonical_digest
from klab.corpus.build_stages import (
    RECEIPT,
    StageStore,
    implementation_digests,
    verify_stage,
)
from klab.corpus.dedup_policy import drops_train_copy, group_policy_error
from klab.corpus.large_build import native_dedup_inputs
from klab.corpus.release_receipt import receipt_first, stat_index
from klab.corpus.release_verify import current_settings
from klab.safe_io import symlinked
from klab.training.manifest import sha256_file

_DIGEST = re.compile(r"[0-9a-f]{64}")
_METHODS = ("raw", "normalized")
_GROUP_COLUMNS = "origin,members,train,validation_count,test,incompatible_rights"


@contextmanager
def _verified_stage(store, name, key):
    directory = store.directory(name, key)
    receipt_path = directory / RECEIPT
    if symlinked(receipt_path) or not receipt_path.is_file():
        raise ValueError(f"missing or unsafe retained Corpus {name} stage: {key}")
    before, _ = stat_index([directory], strict=False)
    receipt = json.loads(receipt_path.read_bytes())
    if (
        receipt["key"] != key
        or receipt["stage"] != name
        or canonical_digest(receipt["inputs"]) != key
    ):
        raise ValueError(f"retained Corpus {name} stage identity differs")
    receipt_sha256 = sha256_file(receipt_path)
    proof_identity = canonical_digest(
        [
            receipt_sha256,
            implementation_digests("duplicate_review.py", "build_stages.py"),
        ]
    )
    settings = current_settings()
    _, verified = receipt_first(
        "duplicate_evidence_stage",
        directory,
        proof_identity,
        [],
        settings.proof_store,
        settings.verification_mode,
        verify=lambda: verify_stage(directory, name, key, receipt["inputs"]),
        reused=lambda: receipt,
    )
    if before != verified:
        raise ValueError(f"Corpus {name} stage changed during verification")
    yield directory, receipt, receipt_sha256
    if stat_index([directory], strict=False)[0] != before:
        raise ValueError(f"Corpus {name} stage changed during inspection")


def _group_summary(row, method, policy):
    key, count, train, validation, test, incompatible = row
    splits = {"train": train, "validation": validation, "test": test}
    present = {split for split, members in splits.items() if members}
    enabled = method in policy.keys
    error = (
        group_policy_error(method, present, incompatible, policy) if enabled else None
    )
    return {
        "sha256": key.hex(),
        "members": count,
        "split_counts": splits,
        "incompatible_rights": bool(incompatible),
        "enabled_by_policy": enabled,
        "policy_error": error,
        "drops_train_copy": enabled
        and error is None
        and drops_train_copy(present, policy),
    }


def review_groups(
    work_root: Path, project, index_key, *, method, group=None, after=None, limit=100
):
    """Inspect a retained exact group page or one group's indexed member page."""
    if (
        method not in _METHODS
        or not _DIGEST.fullmatch(index_key)
        or not 1 <= limit <= 1000
    ):
        raise ValueError(
            "duplicate review requires an exact method, a full index key, and a limit from 1 to 1000"
        )
    if group is not None and not _DIGEST.fullmatch(group):
        raise ValueError("duplicate group requires a full SHA-256")
    if after is not None and group is None and not _DIGEST.fullmatch(after):
        raise ValueError("group cursor requires a full SHA-256")
    if (
        after is not None
        and group is not None
        and not re.fullmatch(r"[0-9]{1,19}", after)
    ):
        raise ValueError("member cursor requires a document ordinal")
    ordinal = int(after) if group is not None and after is not None else 0
    if ordinal >= 2**63:
        raise ValueError("member cursor exceeds the native ordinal range")

    stage_root = work_root / "corpora" / project.config.id / "builds" / "stages"
    store = StageStore(stage_root.resolve())
    policy = project.release.policy.duplicates
    with ExitStack() as stack:
        index_dir, index_receipt, index_sha = stack.enter_context(
            _verified_stage(store, "index", index_key)
        )
        try:
            counts = index_receipt["result"]["duplicate_evidence"][method]
        except KeyError as error:
            raise ValueError(
                "native index predates exact duplicate evidence and must be rebuilt"
            ) from error
        connection = stack.enter_context(
            closing(
                sqlite3.connect(
                    (index_dir / "index.sqlite").as_uri() + "?mode=ro", uri=True
                )
            )
        )
        connection.execute("PRAGMA trusted_schema=OFF")
        result = {
            "index_key": index_key,
            "index_receipt_sha256": index_sha,
            "method": method,
            "counts": counts,
            "policy": policy.model_dump(mode="json"),
        }
        if group is None:
            rows = connection.execute(
                f"SELECT {_GROUP_COLUMNS} FROM duplicate_groups "
                "WHERE method=? AND origin>? ORDER BY origin LIMIT ?",
                (method, bytes.fromhex(after) if after is not None else b"", limit + 1),
            ).fetchall()
            result["groups"] = [
                _group_summary(row, method, policy) for row in rows[:limit]
            ]
            result["next_after"] = (
                rows[limit - 1][0].hex() if len(rows) > limit else None
            )
            return result

        key = bytes.fromhex(group)
        row = connection.execute(
            f"SELECT {_GROUP_COLUMNS} FROM duplicate_groups WHERE method=? AND origin=?",
            (method, key),
        ).fetchone()
        if row is None:
            raise ValueError(f"no {method} duplicate group: {group}")
        result["group"] = _group_summary(row, method, policy)
        dedup_key = store.key(native_dedup_inputs(project, index_sha))
        decision_dir = store.directory("dedup", dedup_key)
        decided = decision_dir.exists() or decision_dir.is_symlink()
        decision_columns = (
            ", selected.representative_id, selected.drop_reason" if decided else ""
        )
        decision_join = ""
        if decided:
            directory, _, _ = stack.enter_context(
                _verified_stage(store, "dedup", dedup_key)
            )
            connection.execute(
                "ATTACH DATABASE ? AS decisions",
                ((directory / "dedup.sqlite").as_uri() + "?mode=ro",),
            )
            decision_join = "LEFT JOIN decisions.dedup AS selected ON selected.document_number=o.document_number AND selected.document_id=d.document_id "
        rows = connection.execute(
            f"SELECT o.document_number,d.document_id,d.source_id,d.split{decision_columns} "
            "FROM origins AS o JOIN documents AS d ON d.rowid=o.document_number "
            f"{decision_join}WHERE o.method=? AND o.origin=? AND o.document_number>? "
            "ORDER BY o.document_number LIMIT ?",
            (method, key, ordinal, limit + 1),
        ).fetchall()
        if decided and any(row[4] is None for row in rows):
            raise ValueError(
                "duplicate decisions do not cover the verified membership page"
            )
        fields = ["ordinal", "document_id", "source_id", "split"]
        if decided:
            fields += ["representative_id", "drop_reason"]
        result["decision_stage"] = dedup_key if decided else None
        result["members"] = [
            dict(zip(fields, row, strict=True)) for row in rows[:limit]
        ]
        result["next_after"] = str(rows[limit - 1][0]) if len(rows) > limit else None
        return result
