"""Durable, conservative response-body reservations for bounded corpus acquisition."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import sqlite3
import time
from contextlib import closing
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from klab.canonical import canonical_digest
from klab.corpus.project import Project

HTTP_OPERATION_TIMEOUT_SECONDS = 30.0


def _valid_time_provenance(owner: Any, reference: Any) -> bool:
    return all(
        isinstance(value, str)
        and 1 <= len(value.strip()) <= maximum
        and value == value.strip()
        and "\0" not in value
        for value, maximum in ((owner, 128), (reference, 512))
    )


PINNED_FILE_KINDS = frozenset({"archive", "pdf_documents", "mediawiki_dump"})


def pinned_transfers(source: Any) -> tuple[tuple[str, int], ...]:
    """Every body transfer a source declares, as ledger path and byte cap."""
    spec = source.acquisition
    if source.kind == "huggingface_dataset" and spec.bounded_shards is not None:
        return tuple(
            (shard.path, shard.max_shard_bytes) for shard in spec.bounded_shards
        )
    if source.kind == "git" and spec.bounded_blobs is not None:
        return tuple((blob.path, blob.max_bytes) for blob in spec.bounded_blobs)
    if source.kind == "archive":
        return ((source_file_name(source.canonical_uri), spec.max_archive_bytes),)
    if source.kind == "mediawiki_dump":
        return ((source_file_name(source.canonical_uri), spec.max_compressed_bytes),)
    if source.kind == "pdf_documents":
        return tuple((item.name, item.max_bytes) for item in spec.documents)
    return ()


def source_file_name(uri: str) -> str:
    """The ledger and staging name of a pinned single-file download."""
    name = PurePosixPath(urlparse(uri).path).name
    return name if name not in {"", ".", ".."} else "download"


def _binding(project: Project) -> str:
    value = {
        "project": project.config.id,
        "budget": project.config.transport_budget.model_dump(mode="json"),
        "sources": [
            {
                "id": source.id,
                "uri": source.canonical_uri,
                "revision": source.revision,
                "shards": [
                    {
                        "path": shard.path,
                        "sha256": shard.expected_sha256.lower(),
                        "max_bytes": shard.max_shard_bytes,
                    }
                    for shard in source.acquisition.bounded_shards
                ],
            }
            for source in project.sources
            if source.kind == "huggingface_dataset"
            and source.acquisition.bounded_shards is not None
        ]
        + [
            {
                "id": source.id,
                "uri": source.canonical_uri,
                "revision": source.revision,
                "tree_oid": source.acquisition.tree_oid,
                "blobs": [
                    blob.model_dump(mode="json")
                    for blob in source.acquisition.bounded_blobs
                ],
            }
            for source in project.sources
            if source.kind == "git" and source.acquisition.bounded_blobs is not None
        ]
        + [
            {
                "id": source.id,
                "kind": source.kind,
                "uri": source.canonical_uri,
                "revision": source.revision,
                "transfers": [
                    {"path": path, "max_bytes": size}
                    for path, size in pinned_transfers(source)
                ],
                "sha256": (
                    [item.expected_sha256 for item in source.acquisition.documents]
                    if source.kind == "pdf_documents"
                    else [source.acquisition.expected_sha256]
                ),
            }
            for source in project.sources
            if source.kind in PINNED_FILE_KINDS
        ],
    }
    if project.config.source_effects is not None:
        value["source_effects"] = [
            effect.model_dump(mode="json") for effect in project.config.source_effects
        ]
    return canonical_digest(value)


def _validate_scope(project: Project) -> None:
    effects = (
        {item.source_id: item for item in project.config.source_effects}
        if project.config.source_effects is not None
        else None
    )
    for source in project.sources:
        if effects is not None and effects[source.id].effect == "reuse_only":
            continue
        if source.redistribution == "rejected" or source.kind in {
            "local",
            "deterministic_generator",
            "inference_generator",
        }:
            continue
        if not pinned_transfers(source):
            raise ValueError(
                "transport budget cannot cover this unaccounted network source"
            )


class _FreeSpaceFloorExhausted(OSError):
    """Recoverable environmental interruption, distinct from invalid source bytes."""


class TransportBudget:
    selected_source_id: str | None = None
    bounded_storage = False

    def __init__(self, path: Path, project: Project):
        spec = project.config.transport_budget
        if spec is None:
            raise ValueError("project has no transport budget")
        _validate_scope(project)
        if not path.is_file() or path.is_symlink():
            raise ValueError(
                "transport budget ledger missing; initialize it explicitly"
            )
        self.path = path
        self.spec = spec
        self.project = project
        self.binding = _binding(project)
        self._read()

    @property
    def _total_body_limit(self) -> int | None:
        return self.spec.max_total_body_bytes

    @property
    def _initial_body_limit(self) -> int | None:
        return self.spec.initial_body_bytes

    def _validate_state(self, state: dict[str, Any]) -> None:
        if state["binding"] != self.binding:
            raise ValueError("transport budget binding changed")
        transfers = state["transfers"]
        expected_deadline = state["created_at"] + self.spec.max_wall_seconds
        mode = state.get("wall_clock_mode", "enforced")
        observation = state.get("wall_clock_observation")
        valid_mode = mode == "enforced" and observation is None
        if mode == "observational" and isinstance(observation, dict):
            valid_mode = (
                set(observation)
                == {"owner", "reference", "selected_at", "planned_deadline"}
                and _valid_time_provenance(
                    observation.get("owner"), observation.get("reference")
                )
                and type(observation.get("selected_at")) in (int, float)
                and math.isfinite(observation["selected_at"])
                and observation["selected_at"] >= state["created_at"]
                and observation["planned_deadline"] == expected_deadline
            )
        storage_mode = state.get("storage_mode", "enforced")
        storage = state.get("storage_observation")
        valid_storage = storage_mode == "enforced" and storage is None
        if storage_mode == "observational" and isinstance(storage, dict):
            valid_storage = (
                type(self) is TransportBudget
                and set(storage)
                == {
                    "owner",
                    "reference",
                    "selected_at",
                    "planned_max_disk_bytes",
                    "min_free_bytes",
                }
                and _valid_time_provenance(
                    storage.get("owner"), storage.get("reference")
                )
                and type(storage.get("selected_at")) in (int, float)
                and math.isfinite(storage["selected_at"])
                and storage["selected_at"] >= state["created_at"]
                and storage["planned_max_disk_bytes"] == self.spec.max_disk_bytes
                and type(storage.get("min_free_bytes")) is int
                and storage["min_free_bytes"] > 0
            )
        network_mode = state.get("network_mode", "enforced")
        network = state.get("network_observation")
        observes_network = network_mode == "observational"
        valid_network = network_mode == "enforced" and network is None
        if observes_network and isinstance(network, dict):
            valid_network = (
                type(self) is TransportBudget
                and set(network)
                == {"owner", "reference", "selected_at", "planned_limits"}
                and _valid_time_provenance(
                    network.get("owner"), network.get("reference")
                )
                and type(network.get("selected_at")) in (int, float)
                and math.isfinite(network["selected_at"])
                and network["selected_at"] >= state["created_at"]
                and type(network.get("planned_limits")) is dict
                and network["planned_limits"] == self._network_limits()
                and all(
                    type(network["planned_limits"][key]) is type(value)
                    for key, value in self._network_limits().items()
                )
            )
        if (
            not valid_network
            or not valid_mode
            or not valid_storage
            or state["deadline"] != expected_deadline
            or not isinstance(transfers, list)
            or (not observes_network and len(transfers) > self.spec.max_transfers)
            or state["source_charged"] != sum(item["reserved"] for item in transfers)
            or state["source_actual"] != sum(item["actual"] for item in transfers)
            or not 0 <= state["source_actual"] <= state["source_charged"]
            or (
                not observes_network
                and state["source_charged"] > self.spec.max_source_body_bytes
            )
            or not 0 <= state["metadata_actual"] <= state["metadata_charged"]
            or (
                not observes_network
                and state["metadata_charged"] > self.spec.max_metadata_body_bytes
            )
            or any(
                item["status"] not in {"started", "interrupted", "failed", "complete"}
                or not 0 <= item["actual"] <= item["reserved"]
                for item in transfers
            )
        ):
            raise ValueError("transport budget ledger counters are invalid")
        charged = state["source_charged"] + state["metadata_charged"]
        if (
            not observes_network
            and self._total_body_limit is not None
            and charged > self._total_body_limit
        ):
            raise ValueError("transport budget cumulative total exceeded")
        if (
            not observes_network
            and self._initial_body_limit is not None
            and "gate_review" not in state
            and charged > self._initial_body_limit
        ):
            raise ValueError("transport budget initial gate exceeded")
        if "gate_review" in state:
            review = state["gate_review"]
            if (
                self._initial_body_limit is None
                or not isinstance(review, dict)
                or set(review) - {"stopped_transfers"}
                != {"receipt_sha256", "report_sha256", "approval_reference"}
                or not self._valid_stop_evidence(review.get("stopped_transfers", {}))
                or any(
                    not isinstance(value, str) or not value.strip()
                    for key, value in review.items()
                    if key != "stopped_transfers"
                )
            ):
                raise ValueError("transport budget gate review is invalid")
        for transfer in transfers:
            if "resources" not in transfer:
                continue  # Historical ledgers have no inferred readings.
            readings = transfer["resources"]
            if (
                not isinstance(readings, dict)
                or set(readings)
                != {"expanded_bytes", "peak_staging_bytes", "peak_staging_inodes"}
                or any(
                    (type(value) is not int or value < 0)
                    for key, value in readings.items()
                    if key != "expanded_bytes" or value is not None
                )
            ):
                raise ValueError("transport budget resource counters are invalid")

    def projected_disk_bytes(self, *, source_id: str | None = None) -> int:
        effects = (
            {item.source_id: item for item in self.project.config.source_effects}
            if self.project.config.source_effects is not None
            else None
        )
        transferring = [
            source
            for source in self.project.sources
            if (effects is None or effects[source.id].effect == "acquire")
            and (source_id is None or source.id == source_id)
        ]
        budgeted = [source for source in transferring if pinned_transfers(source)]
        largest = [size for source in budgeted for _, size in pinned_transfers(source)]
        return (
            sum(source.acquisition.max_bytes for source in budgeted)
            + max(largest, default=0)
            + 65_536  # ledger, manifests and small staging metadata
        )

    @classmethod
    def initialize(cls, path: Path, project: Project) -> TransportBudget:
        spec = project.config.transport_budget
        if spec is None:
            raise ValueError("project has no transport budget")
        _validate_scope(project)
        effects = (
            {item.source_id: item for item in project.config.source_effects}
            if project.config.source_effects is not None
            else None
        )
        if not any(
            (effects is None or effects[source.id].effect == "acquire")
            and pinned_transfers(source)
            for source in project.sources
        ):
            raise ValueError("transport budget requires pinned network transfers")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb"):
                pass
        except FileExistsError as error:
            raise ValueError("transport budget ledger already exists") from error
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("PRAGMA synchronous=FULL")
            db.execute(
                "CREATE TABLE state (id INTEGER PRIMARY KEY CHECK (id=1), body TEXT NOT NULL)"
            )
            now = time.time()
            state = {
                "binding": _binding(project),
                "created_at": now,
                "deadline": now + spec.max_wall_seconds,
                "wall_clock_mode": "enforced",
                "source_charged": 0,
                "source_actual": 0,
                "metadata_charged": 0,
                "metadata_actual": 0,
                "transfers": [],
                "failures": [],
            }
            db.execute("INSERT INTO state VALUES (1, ?)", (json.dumps(state),))
        return cls(path, project)

    def _connection(self, **options):
        db = sqlite3.connect(self.path, **options)
        try:
            if self.bounded_storage:
                # Keep one journal inode through this step. PERSIST retains
                # SQLite rollback/crash semantics without per-update file churn.
                db.execute("PRAGMA journal_mode=PERSIST")
                db.execute("PRAGMA temp_store=MEMORY")
                page_size = db.execute("PRAGMA page_size").fetchone()[0]
                db.execute(f"PRAGMA max_page_count={32 * 1024**2 // page_size}")
            return db
        except BaseException:
            db.close()
            raise

    def _read(self) -> dict[str, Any]:
        try:
            with closing(self._connection()) as db, db:
                row = db.execute("SELECT body FROM state WHERE id=1").fetchone()
            if row is None:
                raise ValueError("transport budget ledger is incomplete")
            state = json.loads(row[0])
            self._validate_state(state)
            if time.time() < state["created_at"]:
                raise ValueError("transport budget clock moved backward")
            return state
        except (sqlite3.Error, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("transport budget ledger is invalid") from error

    def _mutate(self, update: Any, *, check_deadline: bool = True) -> Any:
        with (
            closing(self._connection(timeout=30, isolation_level=None)) as db,
            db,
        ):
            db.execute("PRAGMA synchronous=FULL")
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body FROM state WHERE id=1").fetchone()
            if row is None:
                raise ValueError("transport budget ledger is incomplete")
            state = json.loads(row[0])
            self._validate_state(state)
            now = time.time()
            if now < state["created_at"] or (
                check_deadline
                and state.get("wall_clock_mode", "enforced") == "enforced"
                and now >= state["deadline"]
            ):
                raise ValueError("transport budget deadline exceeded")
            result = update(state)
            body = json.dumps(state)
            if self.bounded_storage and len(body.encode()) > 16 * 1024**2:
                raise ValueError("acquisition step transport metadata cap exceeded")
            db.execute("UPDATE state SET body=? WHERE id=1", (body,))
            db.execute("COMMIT")
            return result

    def remaining_seconds(self) -> float:
        """Finite window for the next operation; observational time has no total expiry."""
        state = self._read()
        if state.get("wall_clock_mode", "enforced") == "observational":
            return HTTP_OPERATION_TIMEOUT_SECONDS
        remaining = state["deadline"] - time.time()
        if remaining <= 0:
            raise ValueError("transport budget deadline exceeded")
        return remaining

    def _network_limits(self) -> dict[str, Any]:
        return {
            name: getattr(self.spec, name)
            for name in (
                "max_source_body_bytes",
                "max_metadata_body_bytes",
                "max_total_body_bytes",
                "initial_body_bytes",
                "max_retries_per_shard",
                "max_transfers",
            )
        }

    def observe_network(self, *, owner: str, reference: str) -> dict[str, Any]:
        """Observe local bytes and explicit retries without a cumulative allowance."""
        if type(self) is not TransportBudget:
            raise ValueError("network observation requires a corpus transport ledger")
        if not _valid_time_provenance(owner, reference):
            raise ValueError("owner and approval reference are required")

        def update(state: dict[str, Any]) -> dict[str, Any]:
            if state.get("network_mode", "enforced") == "observational":
                raise ValueError("transport network is already observational")
            if any(item["status"] == "started" for item in state["transfers"]):
                raise ValueError(
                    "cannot change network policy with an active shard transfer"
                )
            observation = {
                "owner": owner,
                "reference": reference,
                "selected_at": time.time(),
                "planned_limits": self._network_limits(),
            }
            state["network_mode"] = "observational"
            state["network_observation"] = observation
            self._validate_state(state)
            return observation

        return self._mutate(update, check_deadline=False)

    @property
    def observes_storage(self) -> bool:
        return self._read().get("storage_mode", "enforced") == "observational"

    def observe_storage(
        self, *, owner: str, reference: str, min_free_bytes: int
    ) -> dict[str, Any]:
        """Record local storage observation; preserve every network reservation."""
        if type(self) is not TransportBudget:
            raise ValueError(
                "storage observation is supported only for a corpus transport ledger"
            )
        if (
            not _valid_time_provenance(owner, reference)
            or type(min_free_bytes) is not int
            or min_free_bytes <= 0
        ):
            raise ValueError(
                "owner, approval reference and positive free-space floor are required"
            )

        def update(state: dict[str, Any]) -> dict[str, Any]:
            if state.get("storage_mode", "enforced") == "observational":
                raise ValueError("transport storage is already observational")
            if any(item["status"] == "started" for item in state["transfers"]):
                raise ValueError(
                    "cannot change storage policy with an active shard transfer"
                )
            observation = {
                "owner": owner,
                "reference": reference,
                "selected_at": time.time(),
                "planned_max_disk_bytes": self.spec.max_disk_bytes,
                "min_free_bytes": min_free_bytes,
            }
            state["storage_mode"] = "observational"
            state["storage_observation"] = observation
            self._validate_state(state)
            return observation

        return self._mutate(update, check_deadline=False)

    def check_disk(self, root: Path, projected_bytes: int) -> None:
        """Observe the real free-space floor, or retain the default bounded policy."""
        state = self._read()
        free = shutil.disk_usage(root).free
        if state.get("storage_mode", "enforced") == "observational":
            if free < state["storage_observation"]["min_free_bytes"]:
                raise _FreeSpaceFloorExhausted("transport free-space floor exhausted")
        else:
            if projected_bytes > self.spec.max_disk_bytes:
                raise ValueError("transport budget disk allowance exhausted")
            if free < projected_bytes:
                raise ValueError("insufficient disk space for bounded acquisition")

    def observe_time(self, *, owner: str, reference: str) -> dict[str, Any]:
        """Select timing observation once for this corpus ledger, without changing its binding."""
        if type(self) is not TransportBudget:
            raise ValueError(
                "observational time is supported only for a corpus transport ledger"
            )
        if not _valid_time_provenance(owner, reference):
            raise ValueError("owner and approval reference are required")

        def update(state: dict[str, Any]) -> dict[str, Any]:
            if state.get("wall_clock_mode", "enforced") == "observational":
                raise ValueError("transport time is already observational")
            if any(item["status"] == "started" for item in state["transfers"]):
                raise ValueError("cannot change timing with an active shard transfer")
            observation = {
                "owner": owner,
                "reference": reference,
                "selected_at": time.time(),
                "planned_deadline": state["deadline"],
            }
            state["wall_clock_mode"] = "observational"
            state["wall_clock_observation"] = observation
            return observation

        return self._mutate(update, check_deadline=False)

    def reserve_transfer(
        self,
        source_id: str,
        path: str,
        size: int,
        *,
        allow_verified_retry: bool = False,
    ) -> int:
        key = f"{source_id}:{path}"

        def update(state: dict[str, Any]) -> int:
            self._check_transfer(
                state, key, size, allow_verified_retry=allow_verified_retry
            )
            for item in state["transfers"]:
                if item["key"] == key and item["status"] == "started":
                    item["status"] = "interrupted"
            state["source_charged"] += size
            state["transfers"].append(
                {"key": key, "reserved": size, "actual": 0, "status": "started"}
            )
            if state.get("network_mode", "enforced") == "observational":
                state["transfers"][-1]["started_at_unix"] = time.time()
                state["transfers"][-1]["started_monotonic_seconds"] = time.monotonic()
            return len(state["transfers"]) - 1

        return self._mutate(update)

    def _check_transfer(
        self,
        state: dict[str, Any],
        key: str,
        size: int,
        *,
        allow_verified_retry: bool = False,
    ) -> None:
        self._check_combined(state, size)
        prior = [item for item in state["transfers"] if item["key"] == key]
        if any(
            item["status"] == "failed"
            or item["status"] == "complete"
            and (not allow_verified_retry or item.get("sha256_verified") is not True)
            for item in prior
        ):
            raise ValueError("transport budget shard is already complete or terminal")
        observes_network = state.get("network_mode", "enforced") == "observational"
        if not observes_network and len(prior) > self.spec.max_retries_per_shard:
            raise ValueError("transport budget shard retry allowance exhausted")
        if not observes_network and len(state["transfers"]) >= self.spec.max_transfers:
            raise ValueError("transport budget transfer count exhausted")
        if (
            not observes_network
            and size > self.spec.max_source_body_bytes - state["source_charged"]
        ):
            raise ValueError("transport budget source body allowance exhausted")

    def assert_transfer_available(
        self,
        source_id: str,
        path: str,
        size: int,
        *,
        allow_verified_retry: bool = False,
    ) -> None:
        self.remaining_seconds()
        self._check_transfer(
            self._read(),
            f"{source_id}:{path}",
            size,
            allow_verified_retry=allow_verified_retry,
        )

    def assert_source_available(self, source_id: str) -> None:
        """Read-only admission for all shards, including cumulative transfer caps."""
        self.remaining_seconds()
        state = self._read()
        source = next(item for item in self.project.sources if item.id == source_id)
        for path, size in pinned_transfers(source):
            key = f"{source_id}:{path}"
            self._check_transfer(
                state,
                key,
                size,
                allow_verified_retry=source.kind in PINNED_FILE_KINDS,
            )
            state["source_charged"] += size
            state["transfers"].append({"key": key, "status": "started"})

    def charge_transfer_actual(self, index: int, amount: int) -> None:
        def update(state: dict[str, Any]) -> None:
            transfer = state["transfers"][index]
            if (
                transfer["status"] != "started"
                or amount > transfer["reserved"] - transfer["actual"]
            ):
                raise ValueError("transport budget transfer body cap exceeded")
            transfer["actual"] += amount
            if amount and "started_at_unix" in transfer:
                transfer["last_progress_at_unix"] = time.time()
            state["source_actual"] += amount

        self._mutate(update, check_deadline=False)

    def finish_transfer(
        self, index: int, *, success: bool, interrupted: bool = False
    ) -> None:
        def update(state: dict[str, Any]) -> None:
            transfer = state["transfers"][index]
            if transfer["status"] != "started":
                raise ValueError("transport budget transfer already ended")
            transfer["status"] = (
                "complete" if success else "interrupted" if interrupted else "failed"
            )
            transfer["sha256_verified"] = success
            if "started_at_unix" in transfer:
                transfer["ended_at_unix"] = time.time()
                elapsed = max(
                    0.0, time.monotonic() - transfer["started_monotonic_seconds"]
                )
                transfer["elapsed_seconds"] = elapsed
                transfer["body_bytes_per_second"] = (
                    transfer["actual"] / elapsed if elapsed else None
                )

        self._mutate(update, check_deadline=False)

    def record_failure(self, source_id: str, error: BaseException) -> None:
        def update(state: dict[str, Any]) -> None:
            state["failures"].append(
                {
                    "source_id": source_id,
                    "error_type": type(error).__name__,
                    "message": str(error)[:300],
                }
            )

        self._mutate(update, check_deadline=False)

    def reserve_metadata(self, amount: int) -> None:
        def update(state: dict[str, Any]) -> None:
            self._check_combined(state, amount)
            if (
                state.get("network_mode", "enforced") != "observational"
                and amount
                > self.spec.max_metadata_body_bytes - state["metadata_charged"]
            ):
                raise ValueError("transport budget metadata body allowance exhausted")
            state["metadata_charged"] += amount

        self._mutate(update)

    def charge_metadata_actual(self, amount: int) -> None:
        def update(state: dict[str, Any]) -> None:
            state["metadata_actual"] += amount
            if state["metadata_actual"] > state["metadata_charged"]:
                raise ValueError("transport budget metadata accounting invalid")

        self._mutate(update, check_deadline=False)

    def receipt(self) -> dict[str, Any]:
        return self._read()

    def record_resources(self, index: int, readings: dict[str, Any]) -> None:
        """Retain measured counters even after a transfer or parser fails."""

        def update(state: dict[str, Any]) -> None:
            state["transfers"][index]["resources"] = readings
            self._validate_state(state)

        self._mutate(update, check_deadline=False)

    def _check_combined(self, state: dict[str, Any], amount: int) -> None:
        if type(amount) is not int or amount < 0:
            raise ValueError("transport reservation requires nonnegative integer bytes")
        observes_network = state.get("network_mode", "enforced") == "observational"
        charged = state["source_charged"] + state["metadata_charged"] + amount
        if (
            not observes_network
            and self._total_body_limit is not None
            and charged > self._total_body_limit
        ):
            raise ValueError(
                "transport budget cumulative source-plus-metadata allowance exhausted"
            )
        if (
            not observes_network
            and self._initial_body_limit is not None
            and "gate_review" not in state
            and charged > self._initial_body_limit
        ):
            raise ValueError(
                "transport initial feasibility gate requires an owner-reviewed report"
            )

    @staticmethod
    def _valid_stop_evidence(value: Any) -> bool:
        return isinstance(value, dict) and all(
            isinstance(key, str)
            and key.strip()
            and isinstance(reference, str)
            and reference.strip()
            for key, reference in value.items()
        )

    def release_gate(self, report: Path, review: Path) -> dict[str, Any]:
        """Record separately authorized gate review; never resets any counter.

        The caller must have actual owner authority. A reference in this file is
        provenance, not a grant of permission or a bypass of any other hold.
        """
        if any(
            path.is_symlink() or not path.is_file() or path.stat().st_size > 1024 * 1024
            for path in (report, review)
        ):
            raise ValueError("gate report and review require regular files")
        report_bytes = report.read_bytes()
        report_value = json.loads(report_bytes)
        decision = json.loads(review.read_bytes())
        if (
            not isinstance(report_value, dict)
            or set(report_value) != {"ledger_receipt", "findings"}
            or not isinstance(report_value["findings"], dict)
            or report_value["findings"].get("status") != "ready"
        ):
            raise ValueError(
                "gate report requires ledger_receipt and ready feasibility findings"
            )
        if (
            not isinstance(decision, dict)
            or set(decision) - {"stopped_transfers"}
            != {"decision", "report_sha256", "approval_reference"}
            or not self._valid_stop_evidence(decision.get("stopped_transfers", {}))
            or decision["decision"] != "continue"
            or decision["report_sha256"] != hashlib.sha256(report_bytes).hexdigest()
            or not isinstance(decision["approval_reference"], str)
            or not decision["approval_reference"].strip()
        ):
            raise ValueError(
                "gate review requires a report-bound continue decision and actual approval reference"
            )

        def update(state: dict[str, Any]) -> dict[str, Any]:
            if self._initial_body_limit is None or "gate_review" in state:
                raise ValueError("transport initial gate is absent or already released")
            if state != report_value["ledger_receipt"]:
                raise ValueError("gate report is stale or belongs to another ledger")
            started = {
                item["key"]
                for item in state["transfers"]
                if item["status"] == "started"
            }
            stopped = decision.get("stopped_transfers", {})
            if state["source_charged"] <= 0 or set(stopped) != started:
                raise ValueError(
                    "gate report requires pilot reservations and explicit stop evidence for every started transfer"
                )
            # Hash the reviewed state before reconciling abandoned reservations.
            # Stop references attest external cleanup; receipt equality is not
            # evidence of process death. No charge, deadline or retry is reset.
            state["gate_review"] = {
                "receipt_sha256": canonical_digest(state),
                "report_sha256": decision["report_sha256"],
                "approval_reference": decision["approval_reference"],
            }
            if stopped:
                state["gate_review"]["stopped_transfers"] = stopped
                for item in state["transfers"]:
                    if item["status"] == "started":
                        item["status"] = "interrupted"
            return state["gate_review"]

        return self._mutate(update)
