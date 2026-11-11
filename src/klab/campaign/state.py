"""Canonical Campaign state, crash-recoverable receipts and approval persistence.

The store authenticates its own records; verification of domain artifacts belongs to
CampaignEngine. Local availability is never an input to scientific identities.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from klab.campaign.plan import CampaignPlan, Stage
from klab.canonical import canonical_json, reject_duplicate_keys
from klab.result_publication import sync_directory

_STATES = frozenset(
    {
        "NOT_STARTED",
        "READY",
        "RUNNING",
        "COMPLETE",
        "BLOCKED",
        "AWAITING_APPROVAL",
        "INTERRUPTED",
        "FAILED",
        "INCONCLUSIVE",
    }
)
_OUTCOMES = frozenset(
    {
        "EXPAND_MORE",
        "DESCRIPTIVE_EVIDENCE",
        "READY_FOR_TOKENIZER",
        "READY_FOR_NEXT_STAGE",
        "DO_NOT_ADVANCE",
        "INCONCLUSIVE",
    }
)
_SHA = re.compile(r"[0-9a-f]{64}\Z")


def _canonical(value: object) -> bytes:
    return canonical_json(value) + b"\n"


def _utc(value: object, field: str) -> None:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise ValueError(f"invalid {field} UTC timestamp")
    try:
        datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"invalid {field} UTC timestamp") from exc


def digest(domain: str, value: object) -> str:
    """Hash canonical JSON in an explicit, disjoint domain."""
    if not domain or "\x00" in domain:
        raise ValueError("invalid digest domain")
    return hashlib.sha256(
        domain.encode("utf-8") + b"\x00" + canonical_json(value)
    ).hexdigest()


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _sha(value: object, field: str) -> None:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ValueError(f"invalid {field} SHA-256")


def _object(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(k, str) for k in value):
        raise ValueError(f"invalid {field} object")
    return value


def read_canonical(path: Path) -> dict[str, Any]:
    """Reject noncanonical bytes, duplicate JSON keys, links and malformed records."""
    if path.is_symlink():
        raise ValueError(f"symlinked campaign record: {path}")
    raw = path.read_bytes()

    def nonfinite(value: str) -> None:
        raise ValueError(f"nonfinite campaign record value: {value}")

    try:
        data = json.loads(
            raw, object_pairs_hook=reject_duplicate_keys, parse_constant=nonfinite
        )
        if raw != _canonical(data):
            raise ValueError(f"noncanonical campaign record: {path}")
    except (UnicodeError, json.JSONDecodeError, TypeError, OverflowError) as exc:
        raise ValueError(f"invalid campaign record: {path}") from exc
    return _object(data, "campaign record")


def _atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        sync_directory(path.parent)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def publish_immutable(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Publish once, preserving existing bytes even when the same input is replayed."""
    data = _canonical(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(name, path)
            sync_directory(path.parent)
        except FileExistsError:
            previous = read_canonical(path)
            if previous != payload:
                raise ValueError(f"conflicting immutable campaign receipt: {path}")
            return previous
    finally:
        os.unlink(name)
    return payload


class CampaignStore:
    """One namespace per authored declaration, with a serialized mutation lock."""

    def __init__(self, plan: CampaignPlan, work_dir: Path):
        self.plan = plan
        self.declaration_sha = digest(
            "klab-campaign-declaration-v1", plan.model_dump(mode="json")
        )
        self.root = Path(work_dir) / "campaigns" / plan.id / self.declaration_sha
        self.index = self.root / "state.json"
        self.state_path = self.index
        self._inventory = [
            {
                "id": stage.id,
                "kind": stage.kind,
                "scope": stage.scope,
                "requires": list(stage.requires),
            }
            for stage in plan.stages
        ]
        self._by_id = {stage.id: stage for stage in plan.stages}
        self._inventory_by_id = {entry["id"]: entry for entry in self._inventory}

    @property
    def declaration_sha256(self) -> str:
        return self.declaration_sha

    def initial(self) -> dict[str, Any]:
        rows = [
            {
                **entry,
                "state": "NOT_STARTED",
                "attempts": [],
                "observations": [],
                "outputs": [],
                "availability": {},
            }
            for entry in self._inventory
        ]
        return {
            "format": "campaign-state-v1",
            "id": self.plan.id,
            "declaration_sha256": self.declaration_sha,
            "stages": rows,
            "next_action": {
                "action": "apply",
                "reason": "campaign not started",
                "identities": {"declaration_sha256": self.declaration_sha},
            },
        }

    @contextmanager
    def locked(self) -> Iterator[None]:
        if self.root.is_symlink():
            raise ValueError("symlinked campaign state namespace")
        self.root.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.root / ".lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _validate_row(self, row: object, inventory: dict[str, Any]) -> dict[str, Any]:
        data = _object(row, "stage row")
        if any(data.get(k) != v for k, v in inventory.items()):
            raise ValueError(f"campaign stage inventory mismatch: {inventory['id']}")
        if data.get("state") not in _STATES:
            raise ValueError(f"invalid campaign state: {inventory['id']}")
        for field in ("attempts", "observations", "outputs"):
            if not isinstance(data.get(field), list):
                raise TypeError(f"invalid campaign stage {field}: {inventory['id']}")
        _object(data.get("availability"), "availability")
        for output in data["outputs"]:
            item = _object(output, "output identity")
            if set(item) != {"kind", "identifier", "sha256"} or not all(
                isinstance(item[k], str) and item[k] for k in ("kind", "identifier")
            ):
                raise ValueError(f"invalid output identity: {inventory['id']}")
            if (
                Path(item["identifier"]).is_absolute()
                or PureWindowsPath(item["identifier"]).is_absolute()
            ):
                raise ValueError(
                    f"machine-local path in output identity: {inventory['id']}"
                )
            _sha(item["sha256"], "output")
        if (
            "outcome" in data
            and data["outcome"] is not None
            and data["outcome"] not in _OUTCOMES
        ):
            raise ValueError(f"invalid campaign outcome: {inventory['id']}")
        if "stage_input_sha256" in data:
            _sha(data["stage_input_sha256"], "stage input")
        if "receipt" in data:
            expected_receipt = (
                f"receipts/{inventory['id']}/{data.get('stage_input_sha256')}.json"
            )
            if data["receipt"] != expected_receipt:
                raise ValueError(
                    f"invalid campaign receipt reference: {inventory['id']}"
                )
        if "verified_at" in data:
            _utc(data["verified_at"], "stage verification")
            if data.get("verification") != "last_committed":
                raise ValueError(
                    f"invalid stage verification marker: {inventory['id']}"
                )
        if "verification" in data and "verified_at" not in data:
            raise ValueError(f"stage verification lacks timestamp: {inventory['id']}")
        return data

    def _validate_index(self, value: dict[str, Any], *, signed: bool) -> dict[str, Any]:
        state = dict(value)
        signature = state.pop("state_sha256", None)
        if set(state) != {
            "format",
            "id",
            "declaration_sha256",
            "stages",
            "next_action",
        }:
            raise ValueError("invalid campaign state index fields")
        if (
            state.get("format") != "campaign-state-v1"
            or state.get("id") != self.plan.id
            or state.get("declaration_sha256") != self.declaration_sha
        ):
            raise ValueError("campaign state declaration mismatch")
        rows = state.get("stages")
        if not isinstance(rows, list) or len(rows) != len(self._inventory):
            raise ValueError("campaign state inventory mismatch")
        for row, inventory in zip(rows, self._inventory, strict=True):
            self._validate_row(row, inventory)
        action = _object(state.get("next_action"), "next_action")
        if not isinstance(action.get("action"), str) or not isinstance(
            action.get("reason"), str
        ):
            raise TypeError("campaign next_action needs typed action and reason")
        if "identities" in action:
            _object(action["identities"], "next_action identities")
        if signed:
            _sha(signature, "state")
            if signature != digest("klab-campaign-state-v1", state):
                raise ValueError("campaign state checksum mismatch")
        elif signature is not None:
            raise ValueError("state_sha256 is store-owned")
        _canonical(state)
        return state

    def read(self) -> dict[str, Any] | None:
        if not self.index.exists() and not self.index.is_symlink():
            return None
        value = read_canonical(self.index)
        return self._validate_index(value, signed=True)

    def save(self, state: dict[str, Any]) -> None:
        data = self._validate_index(state, signed=False)
        signed = {**data, "state_sha256": digest("klab-campaign-state-v1", data)}
        encoded = _canonical(signed)
        if self.index.exists() or self.index.is_symlink():
            if read_canonical(self.index) == signed:
                return
            self.read()  # Never overwrite a corrupt index.
        _atomic(self.index, encoded)

    def input_sha(self, stage: Stage, rows_by_id: dict[str, dict[str, Any]]) -> str:
        requirements = []
        for name in sorted(stage.requires):
            dependency = rows_by_id[name]
            _sha(dependency.get("stage_input_sha256"), "dependency stage input")
            self._validate_row(dependency, self._inventory_by_id[name])
            if dependency["state"] != "COMPLETE":
                raise ValueError(f"dependency is not complete: {name}")
            requirements.append(
                {
                    "id": name,
                    "stage_input_sha256": dependency["stage_input_sha256"],
                    "outputs": dependency["outputs"],
                }
            )
        return digest(
            "klab-campaign-stage-input-v1",
            {
                "declaration_sha256": self.declaration_sha,
                "stage": stage.model_dump(mode="json"),
                "dependencies": requirements,
            },
        )

    def _receipt_path(self, stage_id: str, input_sha: str) -> Path:
        _sha(input_sha, "stage input")
        return self.root / "receipts" / stage_id / f"{input_sha}.json"

    def _read_receipt(self, path: Path, stage: Stage) -> dict[str, Any]:
        receipt = read_canonical(path)
        signature = receipt.get("receipt_sha256")
        _sha(signature, "receipt")
        unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if signature != digest("klab-campaign-stage-receipt-v1", unsigned):
            raise ValueError(f"campaign receipt checksum mismatch: {path}")
        inventory = self._inventory_by_id[stage.id]
        if (
            receipt.get("format") != "campaign-stage-receipt-v1"
            or receipt.get("declaration_sha256") != self.declaration_sha
            or any(receipt.get(k) != v for k, v in inventory.items())
            or receipt.get("stage_input_sha256") != path.stem
        ):
            raise ValueError(f"campaign receipt identity mismatch: {path}")
        if receipt.get("state") not in {
            "COMPLETE",
            "BLOCKED",
            "FAILED",
            "INCONCLUSIVE",
            "INTERRUPTED",
        }:
            raise ValueError(f"invalid committed state: {path}")
        if (
            receipt.get("outcome") not in _OUTCOMES
            or not isinstance(receipt.get("reason"), str)
            or not isinstance(receipt.get("deficits"), list)
            or not isinstance(receipt.get("verified_at"), str)
        ):
            raise ValueError(f"invalid campaign receipt result: {path}")
        _utc(receipt["verified_at"], "receipt verification")
        if set(receipt) != {
            "format",
            "declaration_sha256",
            "id",
            "kind",
            "scope",
            "requires",
            "stage_input_sha256",
            "state",
            "outcome",
            "reason",
            "deficits",
            "outputs",
            "availability",
            "observations",
            "verified_at",
            "data",
            "receipt_sha256",
        }:
            raise ValueError(f"invalid campaign receipt fields: {path}")
        row = {
            **inventory,
            "state": receipt["state"],
            "attempts": [],
            "observations": receipt["observations"],
            "outputs": receipt.get("outputs"),
            "availability": receipt.get("availability"),
            "outcome": receipt["outcome"],
            "reason": receipt["reason"],
            "deficits": receipt["deficits"],
            "stage_input_sha256": path.stem,
            "verified_at": receipt["verified_at"],
            "verification": "last_committed",
            "receipt": str(path.relative_to(self.root)),
        }
        extra = _object(receipt["data"], "receipt data")
        if set(extra) & set(row):
            raise ValueError(f"campaign receipt data overrides identity: {path}")
        row.update(extra)
        self._validate_row(row, inventory)
        return row

    def commit(self, stage: Stage, row: dict[str, Any]) -> dict[str, Any]:
        inventory = self._inventory_by_id[stage.id]
        self._validate_row(row, inventory)
        if row["state"] not in {
            "COMPLETE",
            "BLOCKED",
            "FAILED",
            "INCONCLUSIVE",
            "INTERRUPTED",
        }:
            raise ValueError("only terminal stage results can be committed")
        input_sha = row["stage_input_sha256"]
        path = self._receipt_path(stage.id, input_sha)
        if path.exists() or path.is_symlink():
            previous = self._read_receipt(path, stage)
            self._compare_committed(row, previous)
            return {
                **row,
                "verified_at": previous["verified_at"],
                "verification": "last_committed",
                "receipt": previous["receipt"],
            }
        reserved = set(inventory) | {
            "state",
            "attempts",
            "observations",
            "outputs",
            "availability",
            "outcome",
            "reason",
            "deficits",
            "stage_input_sha256",
            "verified_at",
            "verification",
            "receipt",
        }
        payload = {
            "format": "campaign-stage-receipt-v1",
            "declaration_sha256": self.declaration_sha,
            **inventory,
            "stage_input_sha256": input_sha,
            "state": row["state"],
            "outcome": row.get("outcome"),
            "reason": row.get("reason", ""),
            "deficits": row.get("deficits", []),
            "outputs": row["outputs"],
            "availability": row["availability"],
            "observations": row["observations"],
            "verified_at": utc_now(),
            "data": {key: value for key, value in row.items() if key not in reserved},
        }
        if payload["outcome"] not in _OUTCOMES:
            raise ValueError("terminal result requires a typed outcome")
        payload["receipt_sha256"] = digest("klab-campaign-stage-receipt-v1", payload)
        committed = publish_immutable(path, payload)
        previous = self._read_receipt(path, stage)
        self._compare_committed(row, previous)
        return {
            **row,
            "verified_at": committed["verified_at"],
            "verification": "last_committed",
            "receipt": previous["receipt"],
        }

    @staticmethod
    def _compare_committed(row: dict[str, Any], previous: dict[str, Any]) -> None:
        for key in (
            "state",
            "outcome",
            "reason",
            "deficits",
            "outputs",
            "availability",
            "observations",
        ):
            expected = [] if key == "deficits" else "" if key == "reason" else None
            if row.get(key, expected) != previous[key]:
                raise ValueError(f"conflicting committed campaign {key}")
        reserved = {
            "id",
            "kind",
            "scope",
            "requires",
            "state",
            "attempts",
            "observations",
            "outputs",
            "availability",
            "outcome",
            "reason",
            "deficits",
            "stage_input_sha256",
            "verified_at",
            "verification",
            "receipt",
        }
        if {k: v for k, v in row.items() if k not in reserved} != {
            k: v for k, v in previous.items() if k not in reserved
        }:
            raise ValueError("conflicting committed campaign result data")

    def reconcile(self, state: dict[str, Any]) -> dict[str, Any]:
        self._validate_index(state, signed=False)
        receipts = self.root / "receipts"
        if not receipts.exists():
            for row in state["stages"]:
                if "receipt" in row or (
                    row["state"] in {"COMPLETE", "FAILED", "INCONCLUSIVE"}
                    and "stage_input_sha256" in row
                ):
                    raise ValueError(f"committed stage receipt missing: {row['id']}")
            return state
        if receipts.is_symlink():
            raise ValueError("symlinked campaign receipts")
        discovered: dict[str, dict[str, Any]] = {}
        for path in sorted(receipts.rglob("*")):
            if path.is_dir() and not path.is_symlink():
                continue
            if (
                path.suffix != ".json"
                or path.parent.name not in self._by_id
                or path.is_symlink()
            ):
                raise ValueError(f"unexpected campaign receipt: {path}")
            stage = self._by_id[path.parent.name]
            record = self._read_receipt(path, stage)
            if stage.id in discovered:
                raise ValueError(f"multiple committed receipts for stage: {stage.id}")
            discovered[stage.id] = record
        rows = {row["id"]: row for row in state["stages"]}
        result = dict(state)
        result["stages"] = list(state["stages"])
        for stage in self.plan.ordered_stages():
            existing = rows[stage.id]
            receipt = discovered.get(stage.id)
            if receipt is None:
                if (
                    "receipt" in existing
                    or existing["state"] == "COMPLETE"
                    or (
                        existing["state"] in {"FAILED", "INCONCLUSIVE"}
                        and "stage_input_sha256" in existing
                    )
                ):
                    raise ValueError(f"committed stage receipt missing: {stage.id}")
                continue
            expected_sha = (
                self.input_sha(stage, rows)
                if stage.requires
                else self.input_sha(stage, {})
            )
            if receipt["stage_input_sha256"] != expected_sha:
                raise ValueError(f"campaign receipt input mismatch: {stage.id}")
            if (
                "stage_input_sha256" in existing
                and existing["stage_input_sha256"] != receipt["stage_input_sha256"]
            ):
                raise ValueError(f"campaign index/receipt input mismatch: {stage.id}")
            if "receipt" in existing:
                self._compare_committed(existing, receipt)
                if (
                    existing["receipt"] != receipt["receipt"]
                    or existing.get("verified_at") != receipt["verified_at"]
                ):
                    raise ValueError(f"campaign index/receipt mismatch: {stage.id}")
            elif existing["state"] in {"COMPLETE", "BLOCKED", "FAILED", "INCONCLUSIVE"}:
                self._compare_committed(existing, receipt)
            merged = {**existing, **receipt, "attempts": existing["attempts"]}
            rows[stage.id] = merged
        result["stages"] = [rows[item["id"]] for item in self._inventory]
        return result

    def read_approval(self, stage: Stage, binding: str) -> dict[str, Any] | None:
        _sha(binding, "approval binding")
        path = self.root / "approvals" / stage.id / f"{binding}.json"
        if not path.exists() and not path.is_symlink():
            return None
        receipt = read_canonical(path)
        signature = receipt.get("receipt_sha256")
        _sha(signature, "approval receipt")
        unsigned = {k: v for k, v in receipt.items() if k != "receipt_sha256"}
        if (
            signature != digest("klab-campaign-approval-v1", unsigned)
            or receipt.get("format") != "campaign-approval-receipt-v1"
            or receipt.get("declaration_sha256") != self.declaration_sha
            or receipt.get("id") != stage.id
            or receipt.get("binding_sha256") != binding
            or receipt.get("decision") not in {"approve", "reject"}
            or not isinstance(receipt.get("note"), (str, type(None)))
            or not isinstance(receipt.get("identities"), (dict, list))
            or not isinstance(receipt.get("verified_at"), str)
        ):
            raise ValueError(f"invalid campaign approval receipt: {path}")
        if set(receipt) != {
            "format",
            "declaration_sha256",
            "id",
            "binding_sha256",
            "decision",
            "note",
            "identities",
            "verified_at",
            "receipt_sha256",
        }:
            raise ValueError(f"invalid campaign approval receipt fields: {path}")
        _utc(receipt["verified_at"], "approval verification")
        return receipt

    def approval(
        self,
        stage: Stage,
        binding: str,
        decision: str,
        note: str | None,
        identities: object,
    ) -> dict[str, Any]:
        if (
            stage.kind != "approval"
            or decision not in {"approve", "reject"}
            or not isinstance(note, (str, type(None)))
        ):
            raise ValueError("invalid campaign approval decision")
        _sha(binding, "approval binding")
        old = self.read_approval(stage, binding)
        if old is not None:
            if old["decision"] != decision or old["identities"] != identities:
                raise ValueError("conflicting campaign approval binding or decision")
            return old
        path = self.root / "approvals" / stage.id / f"{binding}.json"
        payload = {
            "format": "campaign-approval-receipt-v1",
            "declaration_sha256": self.declaration_sha,
            "id": stage.id,
            "binding_sha256": binding,
            "decision": decision,
            "note": note,
            "identities": identities,
            "verified_at": utc_now(),
        }
        payload["receipt_sha256"] = digest("klab-campaign-approval-v1", payload)
        publish_immutable(path, payload)
        old = self.read_approval(stage, binding)
        assert old is not None
        if old["decision"] != decision or old["identities"] != identities:
            raise ValueError("conflicting campaign approval binding or decision")
        return old
