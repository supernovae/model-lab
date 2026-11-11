"""Human-bound immutable model-family lifecycle actions."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

from klab.campaign.state import digest, publish_immutable, read_canonical, utc_now
from klab.canonical import canonical_digest, canonical_json
from klab.evaluation.readiness import (
    verify_readiness_result,
    verify_review_receipt,
)
from klab.evaluation.suite import verify_evaluation_index
from klab.family.manifest import family_digest, load_family
from klab.workdir import resolve_work_dir

Action = Literal["promote", "reject", "supersede"]


def decide(
    source: Path,
    action: Action,
    node_id: str,
    readiness: Path,
    evaluation: Path,
    approval: Path,
    note: str,
    *,
    successor: str | None = None,
    work_root: Path | None = None,
) -> dict[str, Any]:
    """Publish one reviewed decision; a conflicting action cannot replace it."""
    source = Path(source).absolute()
    family = load_family(source)
    nodes = {node.id: node for node in family.nodes}
    node = nodes.get(node_id)
    if node is None:
        raise ValueError("unknown family node")
    if action not in {"promote", "reject", "supersede"} or not note.strip():
        raise ValueError("lifecycle action requires an action and nonempty note")
    if (action == "supersede") != (successor is not None):
        raise ValueError("supersede requires a named successor only")
    if successor is not None and (successor not in nodes or successor == node_id):
        raise ValueError("unknown or identical successor node")
    if successor is not None and node_id not in _ancestors(nodes, successor):
        raise ValueError("successor must descend from the superseded node")
    if not node.checkpoint or not node.evaluation_index or not node.readiness_result:
        raise ValueError("node needs pinned checkpoint, evaluation and readiness")
    from klab.family.cli import show

    state = next(
        row
        for row in show(source, work_root=work_root)["nodes"]
        if row["id"] == node_id
    )["availability"]
    if any(
        state[kind] != "PRESENT"
        for kind in (
            "checkpoint",
            "evaluation_index",
            "readiness_result",
        )
    ):
        raise ValueError(
            "model lifecycle action requires present verified model evidence"
        )
    index = verify_evaluation_index(Path(evaluation))
    result = verify_readiness_result(Path(readiness))
    review = verify_review_receipt(Path(approval))
    if (
        index["index_sha256"] != node.evaluation_index.sha256
        or result["result_sha256"] != node.readiness_result.sha256
        or index["checkpoint_sha256"] != node.checkpoint.sha256
        or result["checkpoint_sha256"] != node.checkpoint.sha256
        or result["index_sha256"] != index["index_sha256"]
        or review["index_sha256"] != index["index_sha256"]
        or review["checkpoint_sha256"] != node.checkpoint.sha256
    ):
        raise ValueError("family action scientific binding mismatch")
    _validate_action(action, successor, result["state"], review["decision"])
    root = work_root if work_root is not None else resolve_work_dir(None)
    namespace = root / "family" / family.id / family.identities()[node_id]
    binding = digest(
        "klab-model-lifecycle-binding-v1",
        {
            "node_sha256": family.identities()[node_id],
            "readiness_sha256": result["result_sha256"],
        },
    )
    marker = namespace / f"decision-{binding}.json"
    if marker.exists():
        old = verify_lifecycle_receipt(marker, family_source=source)
        if (
            old.get("action"),
            old.get("successor"),
            old.get("note"),
            old.get("approval_sha256"),
        ) != (action, successor, note, review["receipt_sha256"]):
            raise ValueError("conflicting lifecycle action for bound node/readiness")
        return old
    body = {
        "format": "model-lifecycle-receipt-v1",
        "family_sha256": family_digest(source),
        "family": family.id,
        "node": node_id,
        "node_sha256": family.identities()[node_id],
        "readiness_sha256": result["result_sha256"],
        "evaluation_sha256": index["index_sha256"],
        "checkpoint_sha256": node.checkpoint.sha256,
        "approval_sha256": review["receipt_sha256"],
        "approval_path": str(Path(approval).resolve()),
        "readiness_path": str(Path(readiness).resolve()),
        "evaluation_path": str(Path(evaluation).resolve()),
        "family_path": str(source.resolve()),
        "action": action,
        "successor": successor,
        "note": note,
    }
    identity = digest("klab-model-lifecycle-receipt-v1", _scientific_binding(body))
    receipt = {**body, "receipt_sha256": identity, "issued_at_utc": utc_now()}
    receipt["record_sha256"] = canonical_digest(receipt)
    publish_immutable(marker, receipt)
    return receipt


def _validate_action(
    action: str,
    successor: str | None,
    state: str,
    decision: str,
) -> None:
    if action not in {"promote", "reject", "supersede"}:
        raise ValueError("invalid lifecycle action")
    if (action == "supersede") != (successor is not None):
        raise ValueError("lifecycle successor/action mismatch")
    if action == "reject":
        if decision != "reject" and state != "DO_NOT_ADVANCE":
            raise ValueError("reject requires rejecting review or DO_NOT_ADVANCE")
    elif decision != "approve" or state != "READY_FOR_NEXT_STAGE":
        raise ValueError(
            "promotion/supersession requires ready model and approving review"
        )


def _ancestors(nodes: dict[str, Any], node_id: str) -> list[str]:
    result: list[str] = []
    while nodes[node_id].parent is not None:
        node_id = nodes[node_id].parent
        result.append(node_id)
    return result


def _scientific_binding(record: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key
        not in {
            "approval_path",
            "readiness_path",
            "evaluation_path",
            "family_path",
            "issued_at_utc",
            "receipt_sha256",
            "record_sha256",
        }
    }


def verify_lifecycle_metadata(path: Path) -> dict[str, Any]:
    """Authenticate a compact receipt without requiring its external model bytes."""
    record = read_canonical(path)
    if (
        record.get("format") != "model-lifecycle-receipt-v1"
        or record.get("receipt_sha256")
        != digest("klab-model-lifecycle-receipt-v1", _scientific_binding(record))
        or record.get("record_sha256")
        != hashlib.sha256(
            canonical_json(
                {key: value for key, value in record.items() if key != "record_sha256"}
            )
        ).hexdigest()
    ):
        raise ValueError("invalid lifecycle receipt")
    return record


def verify_lifecycle_receipt(
    path: Path, *, family_source: Path | None = None
) -> dict[str, Any]:
    record = verify_lifecycle_metadata(path)
    readiness = verify_readiness_result(Path(record["readiness_path"]))
    index = verify_evaluation_index(Path(record["evaluation_path"]))
    review = verify_review_receipt(Path(record["approval_path"]))
    if (
        readiness["result_sha256"] != record["readiness_sha256"]
        or readiness["index_sha256"] != record["evaluation_sha256"]
        or readiness["checkpoint_sha256"] != record["checkpoint_sha256"]
        or index["index_sha256"] != record["evaluation_sha256"]
        or review["receipt_sha256"] != record["approval_sha256"]
        or review["index_sha256"] != record["evaluation_sha256"]
        or review["checkpoint_sha256"] != record["checkpoint_sha256"]
        or index["checkpoint_sha256"] != record["checkpoint_sha256"]
    ):
        raise ValueError("lifecycle evidence changed")
    _validate_action(
        record["action"], record["successor"], readiness["state"], review["decision"]
    )
    source = family_source or (
        Path(record["family_path"]) if record.get("family_path") else None
    )
    if source is not None:
        family = load_family(source)
        nodes = {node.id: node for node in family.nodes}
        if (
            family.id != record["family"]
            or record["node"] not in nodes
            or family.identities()[record["node"]] != record["node_sha256"]
        ):
            raise ValueError("lifecycle receipt family node identity mismatch")
        successor = record["successor"]
        if successor is not None and (
            successor not in nodes or record["node"] not in _ancestors(nodes, successor)
        ):
            raise ValueError("lifecycle successor is not a descendant")
    elif record["successor"] is not None:
        raise ValueError(
            "cannot authenticate supersession ancestry without family declaration"
        )
    return record
