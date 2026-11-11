"""Typed Campaign orchestration over verified Corpus Forge and Experiment APIs."""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any

from klab.campaign.plan import load_campaign, operational_path, safe_path
from klab.campaign.state import CampaignStore, digest
from klab.runtime_profile import RuntimeProfile
from klab.workdir import storage_checks, warn_storage_checks

if TYPE_CHECKING:
    from klab.campaign.preparation_plan import PreparationRequest
    from klab.experiments.preparation_execution import PreparationExecution
    from klab.experiments.workflow_execution import WorkflowExecution


def _verified_operation(function: Callable[..., Any]) -> Callable[..., Any]:
    """Retain only full-verifier proofs within this one Campaign command."""

    @wraps(function)
    def guarded(*args: Any, **kwargs: Any) -> Any:
        from klab.corpus.release import _verification_operation
        from klab.data.sources import snapshot_verification_operation

        engine = args[0]
        previous = getattr(engine, "_verification_options", None)
        if previous is None:
            from klab.verification_proofs import verification_options

            engine._verification_options = verification_options(
                engine.work_dir, cold=engine.cold_verify
            )
        try:
            with (
                _verification_operation(),
                snapshot_verification_operation(
                    mode="cold" if engine.cold_verify else "verified_reuse"
                ),
                (
                    engine.preparation_context.acquisition_scope()
                    if engine.preparation_context is not None
                    else nullcontext()
                ),
            ):
                return function(*args, **kwargs)
        finally:
            if previous is None:
                del engine._verification_options

    return guarded


class CampaignEngine:
    def __init__(
        self,
        source: Path,
        work_dir: Path,
        after_commit: Callable[[str, dict], None] | None = None,
        *,
        runtime_profile: RuntimeProfile | None = None,
        observer: Any | None = None,
        cold_verify: bool = False,
        preparation_context: PreparationExecution | None = None,
        workflow_context: WorkflowExecution | None = None,
    ) -> None:
        self.source = Path(source).resolve()
        self.preparation_context = preparation_context
        self.workflow_context = workflow_context
        if preparation_context is not None and workflow_context is not None:
            raise ValueError("select one portable execution context")
        self.execution_context = preparation_context or workflow_context
        if workflow_context is not None:
            workflow_context.refresh()
            if (
                self.source != workflow_context.campaign_source
                or Path(work_dir).absolute() != workflow_context.work_dir
            ):
                raise ValueError(
                    "workflow engine source/work root differs from context"
                )
        if preparation_context is not None:
            preparation_context.refresh()
            expected = (
                preparation_context.loaded.context.root
                / preparation_context.loaded.descriptor.entrypoint
            )
            if (
                self.source != expected
                or Path(work_dir).absolute() != preparation_context.work_dir
            ):
                raise ValueError(
                    "preparation engine source/work root differs from context"
                )
        self.plan = (
            preparation_context.loaded.plan
            if preparation_context is not None
            else workflow_context.campaign
            if workflow_context is not None
            else load_campaign(self.source)
        )
        from klab.campaign.preparation_plan import compile_preparation

        self.preparation_plan = compile_preparation(self.plan)
        self.store = CampaignStore(self.plan, Path(work_dir))
        self.work_dir = Path(work_dir).resolve()
        self.after_commit = after_commit
        self.runtime_profile = runtime_profile
        self.observer = observer
        self.cold_verify = cold_verify
        self.stages = {stage.id: stage for stage in self.plan.stages}

    def _verification(self) -> dict[str, Any]:
        from klab.verification_proofs import verification_options

        return getattr(self, "_verification_options", None) or verification_options(
            self.work_dir, cold=self.cold_verify
        )

    def _phase(self, name: str, *, host_kind: str | None = None) -> Any:
        return (
            self.observer.phase(name, host_kind=host_kind)
            if self.observer is not None
            else nullcontext()
        )

    def _runtime_binding(
        self,
        rows: dict[str, dict],
        runtime_stage: str,
        lock: Any,
        cell: Any,
        controller: Any,
    ) -> dict | None:
        """Revalidate the accepted cell and source only before new execution."""
        from klab.experiments.binding import open_runtime_binding

        accepted = self._upstream(rows, runtime_stage)
        entry = (
            accepted.get("availability", {}).get("runtime_bindings", {}).get(cell.id)
        )
        declaration = self.stages[runtime_stage]
        if entry is None:
            if (
                declaration.profile_id is not None
                or declaration.worker is not None
                or cell.config.runtime.engine != "pytorch"
                or cell.config.runtime.backend != "cpu"
            ):
                raise ValueError("RUNTIME_REQUIRED: accepted cell binding absent")
            return None
        binding = open_runtime_binding(
            Path(entry["path"]), lock, cell, controller=controller
        )
        if binding["binding_sha256"] != entry["binding_sha256"]:
            raise ValueError("accepted runtime binding digest changed")
        if binding["source_kind"] == "profile":
            if (
                self.runtime_profile is None
                or self.runtime_profile.model_dump(mode="json") != binding["descriptor"]
            ):
                raise ValueError(
                    "RUNTIME_REQUIRED: accepted profile must be supplied unchanged"
                )
        elif declaration.worker != binding["descriptor"]["name"]:
            raise ValueError("accepted worker differs from runtime declaration")
        return binding

    def _panel_runtime_identity(self, stage: Any, rows: dict[str, dict]) -> dict:
        """Bind historical panel evidence without reexecuting a runtime probe."""
        accepted = self._upstream(rows, stage.runtime)
        collect = self.stages[stage.collect]
        cell_id = self.stages[collect.run].cell
        facts = accepted.get("measurements", {})
        if cell_id not in facts.get("cells", []):
            raise ValueError("generation panel cell is absent from accepted runtime")
        return {
            "runtime_acceptance_sha256": accepted["outputs"][0]["sha256"],
            "runtime_binding_sha256": facts.get("runtime_bindings", {}).get(cell_id),
        }

    def _diagnostic_references(self, stage, rows):
        from klab.training.manifest import sha256_file

        references = []
        for name in stage.evaluations:
            evaluation = self.stages[name]
            collect = self.stages[evaluation.collect]
            run = self.stages[collect.run]
            selected = self._upstream(rows, collect.id)["measurements"]
            recorded = self._upstream(rows, name)
            references.append(
                {
                    "stage": name,
                    "cell_id": run.cell,
                    "run_id": self._upstream(rows, run.id)["outputs"][0]["identifier"],
                    "checkpoint": f"checkpoints/{selected['generation']}",
                    "checkpoint_sha256": selected["sha256"],
                    "suite": str(self._path(evaluation.suite)),
                    "suite_sha256": sha256_file(self._path(evaluation.suite)),
                    "index": recorded["availability"]["path"],
                    "index_sha256": recorded["outputs"][0]["sha256"],
                }
            )
        return references

    def _path(self, reference: str) -> Path:
        if self.workflow_context is not None:
            return self.workflow_context.loaded.context.reference(
                self.source, reference
            )
        return safe_path(self.source.parent, reference)

    def _preparation_request(
        self, stage_id: str, rows: dict[str, dict]
    ) -> PreparationRequest:
        from klab.campaign.preparation_plan import (
            BoundPreparationInput,
            PreparationArtifact,
            PreparationRequest,
        )
        from klab.config.loading import load_config, load_tokenizer_config

        operation = self.preparation_plan.operation(stage_id)
        loader = (
            load_tokenizer_config
            if operation.kind == "tokenizer_train"
            else load_config
        )
        config = (
            self.preparation_context.loaded.configs[stage_id].model_copy(deep=True)
            if self.preparation_context is not None
            else loader(self._path(operation.config_reference))
        )
        bindings = []
        for reference in operation.inputs:
            # Existing verification/replay authority authenticates the output
            # before it is copied into an immutable, operation-local request.
            upstream = self._upstream(rows, reference.producer)
            if len(upstream["outputs"]) != 1:
                raise ValueError("preparation input requires one immutable output")
            bindings.append(
                BoundPreparationInput(
                    reference=reference,
                    path=Path(upstream["availability"]["path"]),
                    artifact=PreparationArtifact.model_validate(upstream["outputs"][0]),
                )
            )
        return PreparationRequest(
            operation=operation,
            source=self.source,
            config=config,
            inputs=tuple(bindings),
        )

    def _artifact_declaration(self, stage: Any) -> Any:
        if self.preparation_context is not None:
            return self.preparation_context.artifact(stage)
        return stage.artifact

    def _snapshot_path(self, stage: Any, field: str) -> Path:
        if self.preparation_context is not None:
            return self.preparation_context.loaded.acquisition_paths[stage.id][field]
        return self._operational_path(getattr(stage, field))

    def _snapshot_identity(self, stage: Any, path: Path) -> dict:
        from klab.data.sources import verify_lock, verify_snapshot

        lock = verify_lock(self._snapshot_path(stage, "lock"))
        manifest = verify_snapshot(path)
        if manifest["lock"] != lock:
            raise ValueError("snapshot differs from declared source lock")
        return {
            "kind": "dataset_snapshot",
            "identifier": lock["lock_sha256"],
            "sha256": manifest["manifest_sha256"],
        }

    def _dispatch_snapshot(self, stage: Any, *, resume_acquisition: bool) -> dict:
        from klab.data.sources import snapshot_source

        output = self._snapshot_path(stage, "output")
        # A published snapshot left by an interrupted receipt commit is verified,
        # never acquired again. Partial acquisition requires explicit resume.
        path = output / "manifest.json"
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            journal = output.with_name(f".{output.name}.work")
            try:
                with (
                    self.preparation_context.snapshot_scope(stage)
                    if self.preparation_context is not None
                    else nullcontext()
                ):
                    path = snapshot_source(
                        self._snapshot_path(stage, "lock"),
                        output,
                        self._snapshot_path(stage, "cache_dir"),
                        resume=stage.resume
                        or (resume_acquisition and journal.exists()),
                    )
            except (OSError, RuntimeError) as error:
                if not journal.is_dir() or journal.is_symlink():
                    raise
                return self._result("INTERRUPTED", "DO_NOT_ADVANCE", reason=str(error))
        identity = self._snapshot_identity(stage, path)
        return self._result(outputs=[identity], availability={"path": str(path)})

    def _operational_path(self, reference: str) -> Path:
        return operational_path(self.source.parent, reference)

    def _artifact_dataset(self, stage: Any) -> Any:
        """Supply native tokenizer verification with its declared dataset context."""
        if stage.artifact.kind != "tokenizer":
            return None
        from klab.config.loading import load_config
        from klab.experiments.plan import base_run_config, load_plan

        for consumer in self.plan.stages:
            if consumer.kind == "data_prepare":
                operation = self.preparation_plan.operation(consumer.id)
                if not any(
                    item.port == "tokenizer" and item.producer == stage.id
                    for item in operation.inputs
                ):
                    continue
                if self.preparation_context is not None:
                    return self.preparation_context.loaded.configs[
                        consumer.id
                    ].dataset.model_copy(deep=True)
                path = self._path(operation.config_reference)
                if path.is_file():
                    return load_config(path).dataset
            elif consumer.kind == "experiment_plan" and consumer.tokenizer == stage.id:
                path = self._path(consumer.source)
                if path.is_file():
                    if self.workflow_context is not None:
                        return self.workflow_context.config.dataset.model_copy(
                            deep=True
                        )
                    return base_run_config(load_plan(path), path).dataset
        return None

    @staticmethod
    def _identity(kind: str, identifier: str, sha256: str) -> list[dict[str, str]]:
        return [{"kind": kind, "identifier": identifier, "sha256": sha256}]

    @staticmethod
    def _result(
        state: str = "COMPLETE", outcome: str = "READY_FOR_NEXT_STAGE", **kwargs: Any
    ) -> dict:
        return {
            "state": state,
            "outcome": outcome,
            "reason": kwargs.pop("reason", "verified"),
            "deficits": [],
            "outputs": [],
            "availability": {},
            **kwargs,
        }

    def _provenance_closure(self, provenance: dict) -> list[dict[str, str | None]]:
        """Compare scientific closure bytes, not branch, HEAD or availability paths."""
        entries = provenance.get("declarations") or []
        from klab.training.manifest import sha256_file

        locks = [
            {
                "path": f"dataset-lock:{stage.id}",
                "sha256": sha256_file(self._operational_path(stage.lock))
                if self._operational_path(stage.lock).is_file()
                else None,
            }
            for stage in self.plan.stages
            if stage.kind == "dataset_snapshot"
        ]
        if entries and provenance.get("status") != "UNKNOWN":
            return sorted(
                [
                    {"path": entry["path"], "sha256": entry["sha256"]}
                    for entry in entries
                ]
                + locks,
                key=lambda entry: entry["path"],
            )
        # An explicit UNKNOWN override still needs a byte-bound closure when
        # declarations live outside a Git repository.
        from klab.recovery.provenance import declaration_paths, repository_root

        root = repository_root(self.source) or self.source.parent

        return sorted(
            [
                {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
                for path in declaration_paths(self.source, "campaign")
            ]
            + locks,
            key=lambda entry: entry["path"],
        )

    @staticmethod
    def _require_declaration_files(provenance: dict) -> None:
        closure = provenance["closure"]
        missing = [item["path"] for item in closure if item.get("sha256") is None]
        if not closure or missing:
            raise ValueError(
                "MISSING_DECLARATION_INPUT: "
                + (
                    ", ".join(sorted(missing))
                    if missing
                    else "empty scientific closure"
                )
            )

    def _allow_recovery_publication(self, prior: dict, current: dict) -> bool:
        """Accept only committed, verified output inventory additions after a stage."""
        import hashlib
        import subprocess

        import yaml

        from klab.campaign.state import read_canonical
        from klab.canonical import reject_duplicate_keys
        from klab.documents import _reject_constant, _UniqueLoader
        from klab.recovery.engine import inspect_manifest
        from klab.recovery.evidence import export_evidence
        from klab.recovery.manifest import RecoveryManifest, load_manifest
        from klab.recovery.provenance import _git, repository_root

        if self.plan.recovery is None or current.get("status") != "CLEAN_AND_COMMITTED":
            return False
        root = repository_root(self.source)
        commit = prior.get("source_commit")
        if root is None or not isinstance(commit, str):
            return False
        manifest_path = self._path(self.plan.recovery)
        name = manifest_path.relative_to(root).as_posix()
        old = {item["path"]: item["sha256"] for item in prior["closure"]}
        new = {item["path"]: item["sha256"] for item in current["closure"]}
        if name not in old or name not in new or old[name] == new[name]:
            return False
        try:
            old_bytes = _git(root, "show", f"{commit}:{name}")
            if hashlib.sha256(old_bytes).hexdigest() != old[name]:
                return False  # An uncommitted prior recipe has no trusted old intent.
            text = old_bytes.decode("utf-8")
            payload = (
                json.loads(
                    text,
                    object_pairs_hook=reject_duplicate_keys,
                    parse_constant=_reject_constant,
                )
                if manifest_path.suffix == ".json"
                else yaml.load(text, Loader=_UniqueLoader)
            )
            previous = RecoveryManifest.model_validate(payload)
            updated = load_manifest(manifest_path)
            if previous.model_dump(exclude={"steps", "evidence"}) != updated.model_dump(
                exclude={"steps", "evidence"}
            ):
                return False
            if updated.evidence[: len(previous.evidence)] != previous.evidence:
                return False
            expected_updates: dict[str, str] = {}
            if len(previous.steps) != len(updated.steps):
                return False
            for before, after in zip(previous.steps, updated.steps, strict=True):
                original = before.model_dump()
                revised = after.model_dump()
                for field in tuple(original):
                    if not field.startswith("expected_"):
                        continue
                    old_value, new_value = original[field], revised[field]
                    if old_value is None and isinstance(new_value, str):
                        expected_updates[before.id] = new_value
                        original[field] = new_value
                if original != revised:
                    return False
            appended = updated.evidence[len(previous.evidence) :]
            if not appended and not expected_updates:
                return False
            additions = {
                self._path(self.plan.recovery)
                .parent.joinpath(reference)
                .relative_to(root)
                .as_posix()
                for reference in appended
            }
            if additions & old.keys() or set(new) != set(old) | additions:
                return False
            if any(old[path] != new[path] for path in old if path != name):
                return False
            for reference in appended:
                from klab.recovery.provenance import declaration_reference

                evidence_path = declaration_reference(manifest_path, reference)
                record = read_canonical(evidence_path)
                if record.get("format") != "scientific-evidence-reference-v1":
                    return False
                export_evidence(
                    record["kind"],
                    Path(record["external_location"]),
                    evidence_path,
                    source_commit=record["source_commit"],
                    declaration_hashes=record["declaration_hashes"],
                )
            if expected_updates:
                rows = {
                    item["id"]: item
                    for item in inspect_manifest(manifest_path, self.work_dir)["steps"]
                }
                if any(
                    rows[step]["classification"] != "PRESENT"
                    or rows[step]["actual_sha256"] != expected
                    for step, expected in expected_updates.items()
                ):
                    return False
            return True
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            UnicodeError,
            yaml.YAMLError,
            subprocess.CalledProcessError,
        ):
            return False

    def _assert_declaration_identity(self, state: dict, provenance: dict) -> None:
        current = provenance["closure"]
        for row in state["stages"]:
            if "stage_input_sha256" not in row and not row.get("receipt"):
                continue
            prior = row.get("declaration_provenance")
            if not isinstance(prior, dict) or "closure" not in prior:
                raise ValueError(
                    f"DECLARATION_IDENTITY_UNKNOWN: stage {row['id']} "
                    "has no committed closure binding"
                )
            if prior.get("workflow_context") != provenance.get("workflow_context"):
                raise ValueError("WORKFLOW_CONTEXT_CHANGED: receipt context differs")
            if prior.get("preparation_context") != provenance.get(
                "preparation_context"
            ):
                raise ValueError("PREPARATION_CONTEXT_CHANGED: receipt context differs")
            if prior["closure"] != current and not self._allow_recovery_publication(
                prior, provenance
            ):
                raise ValueError(
                    f"DECLARATION_IDENTITY_CHANGED: stage {row['id']} "
                    "was committed against different scientific declaration bytes"
                )

    def _rows(self, state: dict) -> dict[str, dict]:
        return {row["id"]: row for row in state["stages"]}

    def _upstream(self, rows: dict[str, dict], name: str) -> dict:
        row = rows[name]
        if row["state"] != "COMPLETE" or not row.get("outputs"):
            raise ValueError(f"upstream {name} has no completed immutable output")
        self._verify(self.stages[name], row, rows)
        return row

    def _release(self, rows: dict[str, dict], name: str) -> tuple[Path, str]:
        row = self._upstream(rows, name)
        return Path(row["availability"]["path"]), row["outputs"][0]["sha256"]

    def _lock(self, rows: dict[str, dict], name: str) -> Any:
        from klab.experiments.lock import open_lock

        row = self._upstream(rows, name)
        lock = open_lock(Path(row["availability"]["path"]), **self._verification())
        if lock.plan_sha256 != row["outputs"][0]["sha256"]:
            raise ValueError("experiment lock identity changed")
        return lock

    def _binding(self, stage: Any, rows: dict[str, dict]) -> tuple[str, list[dict]]:
        identities = [
            {
                "id": name,
                "stage_input_sha256": rows[name]["stage_input_sha256"],
                "outputs": rows[name]["outputs"],
            }
            for name in sorted(stage.bind)
        ]
        binding = digest(
            "campaign-approval-binding-v1",
            {
                "declaration_sha256": self.store.declaration_sha,
                "gate": stage.id,
                "gate_declaration": stage.model_dump(mode="json"),
                "identities": identities,
            },
        )
        return binding, identities

    def _projection_verification_key(self, stage: Any, row: dict) -> tuple:
        """Identify a completed stage whose checks passed in this projection."""
        return (
            stage.id,
            row.get("stage_input_sha256"),
            tuple(
                (
                    (output.get("kind"), output.get("identifier"), output.get("sha256"))
                    if isinstance(output, dict)
                    else ("invalid", id(output))
                )
                for output in row.get("outputs", [])
            ),
        )

    def _verify(self, stage: Any, row: dict, rows: dict[str, dict]) -> None:
        """Reopen domain artifacts; no completed-invalid stage is ever dispatched again."""
        cache = getattr(self, "_project_verified", None)
        key = self._projection_verification_key(stage, row)
        if cache is not None and key in cache:
            return
        if row.get("stage_input_sha256") != self.store.input_sha(stage, rows):
            raise ValueError(f"stage input changed: {stage.id}")
        kind = stage.kind
        output = row["outputs"][0] if row.get("outputs") else None
        path = Path(row.get("availability", {}).get("path", ""))
        if kind == "dataset_snapshot":
            path = self._operational_path(row["availability"]["path"])
            if path != self._snapshot_path(stage, "output") / "manifest.json":
                raise ValueError("snapshot output location differs from declaration")
            if row["outputs"] != [self._snapshot_identity(stage, path)]:
                raise ValueError("preparation output identity changed")
        elif kind in {"tokenizer_train", "data_prepare"}:
            from klab.campaign.preparation import verify
            from klab.campaign.preparation_plan import PreparationResult

            if len(row["outputs"]) != 1:
                raise ValueError("preparation output identity changed")
            result = PreparationResult(
                path=self._operational_path(row["availability"]["path"]),
                artifact=row["outputs"][0],
            )
            verify(
                self._preparation_request(stage.id, rows),
                result,
                **self._verification(),
            )
        elif kind in {"artifact_reference", "tokenizer_reference"}:
            from klab.experiments.artifacts import verify_artifact

            artifact = verify_artifact(
                self._artifact_declaration(stage),
                self.source,
                dataset=self._artifact_dataset(stage),
                **self._verification(),
            )
            if output != {
                key: artifact[key] for key in ("kind", "identifier", "sha256")
            }:
                raise ValueError(f"external artifact result changed: {stage.id}")
        elif kind == "corpus_release":
            from klab.corpus.release import verify_release

            manifest = verify_release(path, expected_id=output["sha256"])
            if manifest["release_id"] != output["identifier"]:
                raise ValueError("corpus release identity changed")
        elif kind == "experiment_plan":
            from klab.experiments.lock import open_lock

            lock = open_lock(path, **self._verification())
            if lock.plan_sha256 != output["sha256"] or lock.id != output["identifier"]:
                raise ValueError("experiment plan identity changed")
            self._check_lock(stage, rows, lock)
        elif kind == "experiment_collect":
            from klab.experiments.evidence import read_evidence

            lock = self._lock(rows, stage.plan)
            evidence = read_evidence(
                lock,
                Path(row["availability"]["workspace"]),
                path,
                **self._verification(),
            )
            if stage.evaluations:
                expected = sorted(
                    self._diagnostic_references(stage, rows),
                    key=lambda ref: ref["stage"],
                )
                actual = [
                    item["reference"]
                    for item in evidence.get("diagnostic_evaluations", [])
                ]
                if actual != expected:
                    raise ValueError(
                        "final diagnostic references differ from Campaign dependencies"
                    )
            if evidence["index_sha256"] != output["sha256"]:
                raise ValueError("evidence index changed")
            cell = self._check_collection(stage, evidence, rows)
            if row.get("measurements") != self._selected_collection_checkpoint(cell):
                raise ValueError("collected checkpoint binding changed")
        elif kind == "experiment_run":
            from klab.evaluation.evidence import experiment_evidence
            from klab.workers.controller import Controller

            lock = self._lock(rows, stage.plan)
            controller = Controller(
                Path(row["availability"]["workspace"]) / "controller", read_only=True
            )
            matches = self._attempts(controller, lock, stage, rows)
            if len(matches) != 1 or matches[0]["run_id"] != output["identifier"]:
                raise ValueError("completed run is missing or ambiguous")
            if (
                matches[0]["status"] != "COMPLETE"
                or matches[0]["ingestion_status"] != "COMPLETE"
            ):
                raise ValueError("completed run lost ingestion")
            evidence = experiment_evidence(path, **self._verification())
            if (
                evidence["run_id"] != output["identifier"]
                or digest("campaign-run-v1", self._run_identity(matches[0]))
                != output["sha256"]
            ):
                raise ValueError("completed run receipt identity changed")
        elif kind == "approval":
            binding, identities = self._binding(stage, rows)
            receipt = self.store.read_approval(stage, binding)
            if (
                receipt is None
                or receipt["decision"] != "approve"
                or receipt["identities"] != identities
            ):
                raise ValueError("approval receipt missing or changed")
            if output["sha256"] != binding:
                raise ValueError("approval binding changed")
        elif kind == "evaluation":
            from klab.evaluation.suite import verify_evaluation_index
            from klab.training.manifest import sha256_file

            index = verify_evaluation_index(path, **self._verification())
            collect = self.stages[stage.collect]
            collected = self._upstream(rows, stage.collect)["measurements"]
            run = self._upstream(rows, collect.run)
            if (
                index["index_sha256"] != output["sha256"]
                or index["run_id"] != run["outputs"][0]["identifier"]
                or index["suite_sha256"] != sha256_file(self._path(stage.suite))
                or index["checkpoint_sha256"] != collected["sha256"]
                or index["checkpoint"] != f"checkpoints/{collected['generation']}"
            ):
                raise ValueError(
                    "evaluation suite or collected checkpoint binding changed"
                )
        elif kind == "generation_panel":
            from klab.evaluation.panel import verify_panel_result
            from klab.training.manifest import sha256_file

            result = verify_panel_result(path, **self._verification())
            collect = self.stages[stage.collect]
            collected = self._upstream(rows, stage.collect)["measurements"]
            run = self._upstream(rows, collect.run)
            evaluation = self._upstream(rows, stage.evaluation)
            if (
                result["record_sha256"] != output["sha256"]
                or result["run_id"] != run["outputs"][0]["identifier"]
                or result["checkpoint_sha256"] != collected["sha256"]
                or result["checkpoint"] != f"checkpoints/{collected['generation']}"
                or result["evaluation_index_sha256"]
                != evaluation["outputs"][0]["sha256"]
                or result["panel_sha256"] != sha256_file(self._path(stage.panel))
                or result["runtime_binding"]
                != self._panel_runtime_identity(stage, rows)
            ):
                raise ValueError(
                    "generation panel declaration or evidence binding changed"
                )
        elif kind == "model_readiness":
            from klab.evaluation.readiness import verify_readiness_result
            from klab.training.manifest import sha256_file

            result = verify_readiness_result(path)
            evaluation = self._upstream(rows, stage.evaluation)
            expected_review = (
                str(self._operational_path(stage.review))
                if stage.review is not None
                else None
            )
            expected_scores = (
                str(self._operational_path(stage.reviewed_scores))
                if stage.reviewed_scores is not None
                else None
            )
            if (
                result["result_sha256"] != output["sha256"]
                or result["index_sha256"] != evaluation["outputs"][0]["sha256"]
                or result["policy_sha256"] != sha256_file(self._path(stage.policy))
                or result["review"] != expected_review
                or result.get("reviewed_scores") != expected_scores
            ):
                raise ValueError("model readiness result binding changed")
        elif kind == "corpus_readiness" and stage.measurement_receipt is not None:
            from klab.campaign.policy import measure_readiness

            release, _ = self._release(rows, stage.corpus)
            tokenizer = Path(
                self._upstream(rows, stage.tokenizer)["availability"]["path"]
            )
            measured = measure_readiness(
                release,
                stage.policy,
                tokenizer=tokenizer,
                measurement_receipt=self._operational_path(stage.measurement_receipt),
                measurement_sha256=stage.measurement_sha256,
                _scratch=self.store.root / "scratch" / "corpus-readiness",
            )
            if measured["state"] != "COMPLETE" or measured["measurements"] != row.get(
                "measurements"
            ):
                raise ValueError("completed corpus readiness measurement changed")
            expected = digest(f"campaign-{kind}-v1", self._science(row))
            if output is None or output["sha256"] != expected:
                raise ValueError("corpus readiness result identity changed")
        elif kind in {
            "corpus_readiness",
            "token_measurement",
            "runtime_acceptance",
        }:
            expected = digest(f"campaign-{kind}-v1", self._science(row))
            if output is None or output["sha256"] != expected:
                raise ValueError(f"{kind} result identity changed")
        if cache is not None:
            cache.add(key)

    @staticmethod
    def _science(row: dict) -> dict:
        return {
            key: row[key]
            for key in ("stage_input_sha256", "measurements")
            if key in row
        }

    def _check_lock(self, stage: Any, rows: dict[str, dict], lock: Any) -> None:
        tokenizer = self._upstream(rows, stage.tokenizer)["outputs"][0]
        prepared = self._upstream(rows, stage.prepared)["outputs"][0]
        from klab.evaluation.suite import load_suite
        from klab.training.manifest import sha256_file

        suites = {
            self._path(item.suite)
            for item in self.plan.stages
            if item.kind == "evaluation" and self.stages[item.collect].plan == stage.id
        }
        if suites:
            if len(suites) != 1:
                raise ValueError(
                    "one locked plan cannot bind conflicting evaluation suites"
                )
            suite_path = next(iter(suites))
            suite = load_suite(suite_path)
            if lock.evaluation_suite != {
                "id": suite.id,
                "sha256": sha256_file(suite_path),
            }:
                raise ValueError("Campaign suite differs from locked evaluation suite")
        declared = {
            item.cell
            for item in self.plan.stages
            if item.kind == "experiment_run" and item.plan == stage.id
        }
        if not declared:
            if len(lock.cells) != 1:
                raise ValueError(
                    "ambiguous locked plan without an explicitly selected run cell"
                )
            declared = {lock.cells[0].id}
        for selected in sorted(declared):
            cells = [item for item in lock.cells if item.id == selected]
            if len(cells) != 1:
                raise ValueError(f"declared run cell {selected} is absent or ambiguous")
            cell = cells[0]
            for kind, expected in (
                ("tokenizer", tokenizer),
                ("prepared_data", prepared),
            ):
                actual = [
                    item for item in cell.artifacts.values() if item.get("kind") == kind
                ]
                if len(actual) != 1 or any(
                    actual[0].get(key) != expected[key]
                    for key in ("identifier", "sha256")
                ):
                    raise ValueError(
                        f"locked cell {cell.id} differs from declared {kind}"
                    )
            synthetic = cell.config.dataset.source == "synthetic"
            if synthetic and stage.corpus is not None:
                raise ValueError(
                    "synthetic cell cannot claim the campaign corpus as model input"
                )
            # resolve_plan/open_lock authenticate the complete dataset, tokenizer,
            # prepared-data and executable-source closure. Campaign binds those
            # verified identities; it must not invent a Forge dependency for a
            # direct dataset (including future snapshot-backed sources).
            forge = (
                cell.config.dataset.source == "local_text"
                or cell.config.dataset.corpus_release_path is not None
                or cell.config.dataset.corpus_export_path is not None
                or any(
                    item.get("kind") in {"corpus_release", "corpus_export"}
                    for item in cell.artifacts.values()
                )
            )
            if forge or stage.corpus is not None:
                if stage.corpus is None:
                    raise ValueError("Forge locked cell requires a corpus release")
                release = self._upstream(rows, stage.corpus)["outputs"][0]
                actual = [
                    item
                    for item in cell.artifacts.values()
                    if item.get("kind") == "corpus_release"
                ]
                if len(actual) != 1 or actual[0]["sha256"] != release["sha256"]:
                    raise ValueError(
                        "locked corpus variant differs from upstream campaign release"
                    )

    def _check_collection(
        self, stage: Any, evidence: dict, rows: dict[str, dict]
    ) -> dict:
        run_stage = self.stages[stage.run]
        cell = evidence["cells"][run_stage.cell]
        if (
            cell["status"] != "complete"
            or cell["selected_run_id"] != rows[stage.run]["outputs"][0]["identifier"]
        ):
            raise ValueError("selected locked cell has no complete ingested evidence")
        return cell

    @staticmethod
    def _selected_collection_checkpoint(cell: dict) -> dict:
        """Pin the unique latest verified generation in the immutable collection."""
        complete = [
            attempt
            for attempt in cell["attempts"]
            if attempt["status"] == "complete"
            and attempt["run_id"] == cell["selected_run_id"]
        ]
        if len(complete) != 1 or not complete[0].get("checkpoints"):
            raise ValueError("collection has no unique verified checkpoint generation")
        checkpoints = complete[0]["checkpoints"]
        latest_step = max(checkpoint["step"] for checkpoint in checkpoints)
        latest = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint["step"] == latest_step
        ]
        if len(latest) != 1:
            raise ValueError("ambiguous latest generation in collected evidence")
        selected = latest[0]
        return {
            "generation": selected["path"],
            "sha256": selected["digest"],
            "step": selected["step"],
        }

    def _availability(self, stage: Any) -> str | None:
        if stage.kind == "dataset_snapshot":
            from klab.data.sources import verify_lock

            path = self._operational_path(stage.lock)
            if not path.exists():
                return f"missing declared dataset lock: {path}"
            verify_lock(path)
            return None
        if stage.kind in {"tokenizer_train", "data_prepare"}:
            if self.preparation_context is not None:
                self.preparation_context.refresh()
                return None
            from klab.config.loading import load_config, load_tokenizer_config

            operation = self.preparation_plan.operation(stage.id)
            path = self._path(operation.config_reference)
            if not path.exists():
                return f"missing declared preparation config: {path}"
            loader = (
                load_tokenizer_config
                if operation.kind == "tokenizer_train"
                else load_config
            )
            loader(path)
            return None
        if stage.kind in {"artifact_reference", "tokenizer_reference"}:
            path = self._operational_path(self._artifact_declaration(stage).path)
        elif stage.kind == "corpus_release":
            path = self._path(stage.project)
        elif stage.kind == "experiment_plan":
            path = self._path(stage.source)
            if stage.mode == "reference":
                lock_path = self._operational_path(stage.lock)
                if not lock_path.exists():
                    return f"missing declared local lock: {lock_path}"
        elif stage.kind == "evaluation":
            path = self._path(stage.suite)
        elif stage.kind == "generation_panel":
            path = self._path(stage.panel)
        elif stage.kind == "model_readiness":
            path = self._path(stage.policy)
        else:
            return None
        if not path.exists():
            return f"missing declared local input: {path}"
        if stage.kind in {"artifact_reference", "tokenizer_reference"}:
            from klab.experiments.artifacts import verify_artifact

            verify_artifact(
                self._artifact_declaration(stage),
                self.source,
                dataset=self._artifact_dataset(stage),
                **self._verification(),
            )
        elif stage.kind == "corpus_release":
            from klab.corpus.project import ProjectConfig, load_project
            from klab.documents import read_document

            config = ProjectConfig.model_validate(read_document(path))
            for reference in (
                *config.sources,
                *config.transforms,
                config.splits,
                config.release,
            ):
                member = safe_path(path.parent, reference)
                if not member.is_file():
                    return f"missing declared corpus recipe: {member}"
            project = load_project(path)
            for source in project.sources:
                if source.kind not in {"local", "deterministic_generator"}:
                    raise ValueError(
                        "campaign corpus sources must be local or deterministic_generator"
                    )
                if source.kind == "local":
                    for entry in source.acquisition.files:
                        member = safe_path(project.root, entry.path)
                        if not member.is_file():
                            return f"missing declared corpus source: {member}"
        elif stage.kind == "experiment_plan" and stage.mode == "reference":
            from klab.experiments.lock import open_lock

            open_lock(self._operational_path(stage.lock), **self._verification())
        elif stage.kind == "evaluation":
            from klab.evaluation.suite import load_suite

            load_suite(path)
        elif stage.kind == "generation_panel":
            from klab.evaluation.panel import load_panel

            load_panel(path)
        elif stage.kind == "model_readiness":
            from klab.evaluation.readiness import load_policy

            load_policy(path)
        return None

    def _project(self, state: dict) -> dict:
        """Project once with only traversal-local successful verification reuse."""
        missing = object()
        previous = getattr(self, "_project_verified", missing)
        self._project_verified: set[tuple] = set()
        try:
            return self._project_once(state)
        finally:
            if previous is missing:
                del self._project_verified
            else:
                self._project_verified = previous

    def _project_once(self, state: dict) -> dict:
        rows = self._rows(state)
        projected: list[dict] = []
        next_action: dict = {
            "action": "none",
            "reason": "all stages complete",
            "identities": {},
        }
        for stage in self.plan.ordered_stages():
            row = dict(rows[stage.id])
            prior = [rows[name] for name in stage.requires]
            terminal = row["state"] in {
                "COMPLETE",
                "FAILED",
                "INCONCLUSIVE",
                "BLOCKED",
                "AWAITING_APPROVAL",
                "RUNNING",
                "INTERRUPTED",
            }
            if not terminal or (row["state"] == "BLOCKED" and not row.get("receipt")):
                blocked = [item["id"] for item in prior if item["state"] != "COMPLETE"]
                if blocked:
                    row.update(
                        state="BLOCKED",
                        outcome="DO_NOT_ADVANCE",
                        reason="prerequisite did not complete",
                        blocked_by=blocked,
                    )
                elif all(item["state"] == "COMPLETE" for item in prior):
                    row.pop("blocked_by", None)
                    try:
                        missing = self._availability(stage)
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        row.update(
                            state="FAILED", outcome="DO_NOT_ADVANCE", reason=str(error)
                        )
                    else:
                        if missing:
                            row.update(
                                state="BLOCKED",
                                outcome="DO_NOT_ADVANCE",
                                reason=missing,
                            )
                        else:
                            row.update(
                                state="READY",
                                outcome="READY_FOR_NEXT_STAGE",
                                reason="prerequisites complete",
                            )
            if stage.kind == "approval" and row["state"] == "AWAITING_APPROVAL":
                binding, identities = self._binding(stage, rows)
                receipt = self.store.read_approval(stage, binding)
                if receipt is not None:
                    if receipt["identities"] != identities:
                        raise ValueError("committed approval input identities changed")
                    row.update(
                        state="READY",
                        reason="committed approval awaits gate reconciliation",
                    )
            if row["state"] == "COMPLETE":
                self._verify(stage, row, rows)
            if next_action["action"] == "none" and row["state"] in {
                "READY",
                "RUNNING",
                "INTERRUPTED",
                "AWAITING_APPROVAL",
                "BLOCKED",
                "FAILED",
                "INCONCLUSIVE",
            }:
                action = (
                    "approve"
                    if row["state"] == "AWAITING_APPROVAL"
                    else "resume"
                    if row["state"] in {"RUNNING", "INTERRUPTED"}
                    else "apply"
                    if row["state"] == "READY"
                    else "wait"
                )
                if row["state"] == "INTERRUPTED" and stage.kind == "experiment_run":
                    action = "wait"
                next_action = {
                    "action": action,
                    "stage": stage.id,
                    "reason": row.get("reason", row["state"]),
                    "identities": {"outputs": row.get("outputs", [])},
                }
            projected.append(row)
            rows[stage.id] = row
        return {
            "id": self.plan.id,
            "declaration_sha256": self.store.declaration_sha,
            "stages": projected,
            "next_action": next_action,
        }

    def _observed_checkpoints(
        self, state: dict, manifest_path: Path, recipe_rows: list[dict]
    ) -> list[dict]:
        """Separate verified Campaign generations from logical recovery run names."""
        from klab.recovery.manifest import load_manifest
        from klab.training.checkpoints import CheckpointManager
        from klab.training.mlx_checkpoints import strict_json

        recipe = load_manifest(manifest_path)
        persisted = self._rows(state)
        observed: list[dict] = []
        for collect in self.plan.stages:
            if collect.kind != "experiment_collect":
                continue
            row = persisted[collect.id]
            selected = row.get("measurements")
            if row["state"] != "COMPLETE" or not isinstance(selected, dict):
                continue
            if not {"generation", "sha256"} <= selected.keys():
                continue
            run = self.stages[collect.run]
            run_row = persisted[run.id]
            run_path = Path(run_row.get("availability", {}).get("path", ""))
            generation = selected["generation"]
            checkpoint = run_path / "checkpoints" / generation
            expected = selected["sha256"]
            plan_stage = self.stages[collect.plan]
            plan_row = persisted[collect.plan]
            plan_sha = (
                plan_row["outputs"][0]["sha256"] if plan_row.get("outputs") else None
            )
            entry = {
                "id": f"observed_checkpoint:{collect.id}",
                "kind": "checkpoint",
                "expected_sha256": expected,
                "actual_sha256": None,
                "path": str(checkpoint),
                "run_id": (run_row.get("outputs") or [{}])[0].get("identifier"),
                "generation": generation,
                "plan_sha256": plan_sha,
            }
            try:
                self._verify(run, run_row, persisted)
                self._verify(collect, row, persisted)
                report = CheckpointManager(run_path).verify(checkpoint)
                if not report.valid:
                    raise ValueError(f"collected checkpoint invalid: {report.errors}")
                actual = strict_json(checkpoint / "manifest.json")["sha256"]
                if actual != expected:
                    raise ValueError("collected checkpoint digest differs from receipt")
            except (OSError, ValueError, KeyError, TypeError) as error:
                entry.update(
                    classification=(
                        "ERROR" if checkpoint.exists() else "MISSING_NONRECONSTRUCTABLE"
                    ),
                    reason=f"historical Campaign collection cannot be reverified: {error}",
                )
            else:
                entry.update(
                    classification="PRESENT",
                    actual_sha256=actual,
                    reason="freshly verified Campaign collection and exact generation",
                )
                compatible = [
                    step
                    for step in recipe.steps
                    if step.kind == "experiment_lock"
                    and step.expected_plan_sha256 == plan_sha
                    and self._path(plan_stage.source)
                    == safe_path(manifest_path.parent, step.plan)
                ]
                mapped = [
                    step
                    for step in recipe.steps
                    if step.kind == "checkpoint"
                    and step.expected_sha256 == actual
                    and step.cell
                    in {
                        generation,
                        generation.split("_gen_", 1)[0],
                    }
                ]
                if len(compatible) == len(mapped) == 1:
                    for declared in recipe_rows:
                        if declared["id"] == mapped[0].id:
                            declared.update(
                                classification="PRESENT",
                                actual_sha256=actual,
                                action=None,
                                reason="pinned digest and plan match verified Campaign generation",
                            )
                            break
            observed.append(entry)
        return observed

    def _recoverability(self, state: dict) -> list[dict]:
        """Fresh availability, independent of historical stage outcomes."""
        if self.plan.recovery is not None:
            from klab.recovery.engine import inspect_manifest

            manifest = self._path(self.plan.recovery)
            if not manifest.is_file():
                return [
                    {
                        "id": "declaration",
                        "kind": "recovery",
                        "classification": "MISSING_EXTERNAL",
                        "reason": f"missing recovery manifest: {manifest}",
                    }
                ]
            try:
                from klab.data.sources import snapshot_verification_operation
                from klab.training.manifest import sha256_file

                with snapshot_verification_operation(mode="cold"):
                    recovery_steps = inspect_manifest(manifest, self.work_dir)["steps"]

                manifest_sha = sha256_file(manifest)
                rows = [
                    {
                        "id": "declaration",
                        "kind": "recovery",
                        "classification": "PRESENT",
                        "expected_sha256": manifest_sha,
                        "actual_sha256": manifest_sha,
                        "reason": "recovery declaration available",
                    },
                    *recovery_steps,
                ]
                rows.extend(self._observed_checkpoints(state, manifest, rows))
                return rows
            except (OSError, ValueError, KeyError, TypeError) as error:
                return [
                    {
                        "id": "declaration",
                        "kind": "recovery",
                        "classification": "ERROR",
                        "reason_code": "RECOVERY_INSPECTION_FAILED",
                        "reason": str(error),
                    }
                ]
        rows = self._rows(state)
        result = []
        for stage in self.plan.ordered_stages():
            row = rows[stage.id]
            output = row.get("outputs", [])
            if not output:
                classification = (
                    "NOT_CREATED"
                    if row["state"] in {"NOT_STARTED", "READY", "BLOCKED"}
                    and not row.get("stage_input_sha256")
                    else "MISSING_EXTERNAL"
                )
                reason = (
                    "no committed output"
                    if classification == "NOT_CREATED"
                    else "output availability unknown"
                )
            else:
                try:
                    self._verify(stage, row, rows)
                except (OSError, ValueError, KeyError, TypeError) as error:
                    path = Path(row.get("availability", {}).get("path", ""))
                    if stage.kind == "experiment_run":
                        classification = "MISSING_NONRECONSTRUCTABLE"
                    elif path.exists() and row.get("availability", {}).get("path"):
                        classification = "ERROR"
                    else:
                        classification = "MISSING_EXTERNAL"
                    reason = str(error)
                else:
                    classification, reason = "PRESENT", "verified"
            result.append(
                {
                    "id": stage.id,
                    "kind": stage.kind,
                    "classification": classification,
                    "expected_sha256": output[0]["sha256"] if output else None,
                    "reason": reason,
                }
            )
        return result

    def reconstruct(
        self,
        *,
        allow_network: bool = False,
        allow_uncommitted_declaration: bool = False,
        evidence_output: Path | None = None,
    ) -> dict:
        if self.plan.recovery is None:
            raise ValueError("campaign reconstruct requires a linked recovery manifest")
        from klab.recovery.engine import reconstruct_manifest
        from klab.recovery.provenance import declaration_preflight

        provenance = declaration_preflight(
            self.source, "campaign", allow_uncommitted=allow_uncommitted_declaration
        )
        checks = warn_storage_checks(storage_checks(self.work_dir))
        result = reconstruct_manifest(
            self._path(self.plan.recovery),
            self.work_dir,
            allow_network=allow_network,
            allow_uncommitted_declaration=allow_uncommitted_declaration,
            evidence_output=evidence_output,
        )
        return {
            **self.inspect("status"),
            "reconstruction": result,
            "declaration_provenance": provenance,
            "storage_checks": checks,
        }

    def _submitted_run(self, stage: Any, rows: dict[str, dict]) -> bool:
        from klab.workers.controller import Controller

        lock = self._lock(rows, stage.plan)
        prior = rows[stage.id].get("measurements", {})
        submitted = (
            isinstance(prior, dict)
            and {"experiment_id", "attempt_id", "run_id"} <= prior.keys()
        )
        workspace = Path(rows[stage.plan]["availability"]["workspace"])
        controller_dir = workspace / "controller"
        if not controller_dir.exists():
            if submitted:
                raise ValueError(
                    "LOST_SUBMISSION: controller state vanished after run submission"
                )
            return False
        controller = Controller(controller_dir, read_only=True)
        matches = self._attempts(controller, lock, stage, rows)
        if len(matches) > 1:
            raise ValueError("ambiguous controller attempts for locked cell")
        if submitted and (
            not matches
            or self._run_identity(matches[0])
            != {key: prior[key] for key in ("experiment_id", "attempt_id", "run_id")}
        ):
            raise ValueError(
                "LOST_SUBMISSION: previously submitted attempt is missing or replaced"
            )
        return bool(matches)

    @_verified_operation
    def inspect(self, command: str = "plan") -> dict:
        if command not in {"plan", "next", "status", "explain"}:
            raise ValueError(f"unknown inspection command {command}")
        persisted = self.store.read()
        if self.execution_context is not None:
            provenance = self.execution_context.refresh()
            self._assert_declaration_identity(
                self.store.reconcile(persisted or self.store.initial()), provenance
            )
        elif persisted is not None and any(
            (
                row.get("declaration_provenance", {}).get("preparation_context")
                or row.get("declaration_provenance", {}).get("workflow_context")
            )
            for row in persisted["stages"]
        ):
            raise ValueError(
                "portable preparation receipts require their execution context"
            )
        if command == "status" and persisted is not None:
            rows = self._rows(persisted)
            for stage in self.plan.stages:
                if (
                    stage.kind == "corpus_readiness"
                    and stage.measurement_receipt is not None
                ):
                    row = rows[stage.id]
                    if row["state"] == "COMPLETE":
                        self._verify(stage, row, rows)
            return {
                **{
                    key: persisted[key]
                    for key in ("id", "declaration_sha256", "stages", "next_action")
                },
                "recoverability": self._recoverability(persisted),
            }
        state = persisted or self.store.initial()
        if command != "status":
            state = self.store.reconcile(state)
        if self.execution_context is None and any(
            (
                row.get("declaration_provenance", {}).get("preparation_context")
                or row.get("declaration_provenance", {}).get("workflow_context")
            )
            for row in state["stages"]
        ):
            raise ValueError(
                "portable preparation receipts require their execution context"
            )
        projected = self._project(state)
        if command == "status":
            projected["recoverability"] = self._recoverability(state)
        return projected

    def _record(self, state: dict, row: dict) -> None:
        state["stages"] = [
            row if item["id"] == row["id"] else item for item in state["stages"]
        ]
        projection = self._project(state)
        projected = self._rows(projection)
        state["stages"] = [projected[stage.id] for stage in self.plan.stages]
        state["next_action"] = projection["next_action"]
        self.store.save(state)

    @_verified_operation
    def apply(
        self,
        resume: bool = False,
        max_wait_seconds: float = 120,
        *,
        execute_runs: bool = False,
        allow_uncommitted_declaration: bool = False,
        only_stage: str | None = None,
    ) -> dict:
        from klab.recovery.provenance import declaration_preflight

        if max_wait_seconds < 0 or not math.isfinite(max_wait_seconds):
            raise ValueError("max wait seconds must be finite and nonnegative")
        if only_stage is not None and only_stage not in self.stages:
            raise ValueError("selected Campaign stage does not exist")
        if self.workflow_context is not None and allow_uncommitted_declaration:
            raise ValueError("workflow context does not allow uncommitted declarations")
        if self.preparation_context is not None and (
            allow_uncommitted_declaration or execute_runs
        ):
            raise ValueError(
                "preparation context does not allow overrides or model execution"
            )
        provenance = (
            self.execution_context.refresh()
            if self.execution_context is not None
            else declaration_preflight(
                self.source, "campaign", allow_uncommitted=allow_uncommitted_declaration
            )
        )
        provenance = {
            **provenance,
            "allow_uncommitted_declaration": allow_uncommitted_declaration,
            "closure": self._provenance_closure(provenance),
        }
        self._require_declaration_files(provenance)
        if self.preparation_context is not None:
            preview = self.store.reconcile(self.store.read() or self.store.initial())
            self._assert_declaration_identity(preview, provenance)
            for stage in self.plan.stages:
                row = self._rows(preview)[stage.id]
                if stage.kind == "artifact_reference":
                    missing = self._availability(stage)
                    if missing:
                        raise ValueError(missing)
                if not row.get("attempts"):
                    self.preparation_context.before_dispatch(stage, row)
        checks = warn_storage_checks(storage_checks(self.work_dir))
        with self.store.locked():
            previous = self.store.read()
            if resume and previous is None:
                raise ValueError("cannot resume a campaign without durable state")
            if previous is not None:
                self._assert_declaration_identity(previous, provenance)
            state = self.store.reconcile(previous or self.store.initial())
            self._assert_declaration_identity(state, provenance)
            if previous is None or state != previous:
                self._record(state, state["stages"][0])
            while True:
                projection = self._project(state)
                action = projection["next_action"]
                if action["action"] not in {"apply", "resume"}:
                    return {
                        **projection,
                        "declaration_provenance": provenance,
                        "storage_checks": checks,
                    }
                stage = self.stages[action["stage"]]
                if only_stage is not None and stage.id != only_stage:
                    raise ValueError("Campaign next action differs from selected stage")
                rows = self._rows(state)
                row = rows[stage.id]
                self._require_attempt_contract(stage, rows, only_stage=only_stage)
                if (
                    stage.kind == "dataset_snapshot"
                    and row["state"] in {"RUNNING", "INTERRUPTED"}
                    and not resume
                ):
                    return {
                        **projection,
                        "next_action": {
                            **action,
                            "action": "resume",
                            "reason": "interrupted acquisition requires campaign resume",
                        },
                        "declaration_provenance": provenance,
                        "storage_checks": checks,
                    }
                submitted = False
                dispatch_error: OSError | ValueError | KeyError | TypeError | None = (
                    None
                )
                if stage.kind == "experiment_run":
                    try:
                        submitted = self._submitted_run(stage, rows)
                    except (OSError, ValueError, KeyError, TypeError) as error:
                        dispatch_error = error
                if (
                    stage.kind == "experiment_run"
                    and not execute_runs
                    and not submitted
                    and dispatch_error is None
                ):
                    return {
                        **projection,
                        "next_action": {
                            "action": "execute_run",
                            "stage": stage.id,
                            "reason": "new model execution requires --execute-runs",
                            "identities": {"outputs": row.get("outputs", [])},
                        },
                        "declaration_provenance": provenance,
                        "storage_checks": checks,
                    }
                if row["state"] == "RUNNING" and stage.kind != "experiment_run":
                    reason = "uncommitted interrupted attempt"
                    attempts = list(row["attempts"])
                    if attempts:
                        attempts[-1] = {
                            **attempts[-1],
                            "state": "INTERRUPTED",
                            "reason": reason,
                        }
                    row = {
                        **row,
                        "state": "INTERRUPTED",
                        "reason": reason,
                        "attempts": attempts,
                        "observations": [
                            *row["observations"],
                            {"state": "INTERRUPTED", "reason": reason},
                        ],
                    }
                    self._record(state, row)
                if self.preparation_context is not None:
                    self.preparation_context.before_dispatch(stage, row)
                row = {
                    **row,
                    "state": "RUNNING",
                    "stage_input_sha256": self.store.input_sha(stage, rows),
                    "attempts": [
                        *row.get("attempts", []),
                        {"state": "RUNNING", "reason": "dispatch"},
                    ],
                    "declaration_provenance": provenance,
                    "storage_checks": checks,
                }
                self._record(state, row)
                rows = self._rows(state)
                try:
                    if dispatch_error is not None:
                        raise dispatch_error
                    if self.execution_context is not None:
                        self.execution_context.refresh()
                    result = self.dispatch(
                        stage,
                        rows,
                        max_wait_seconds,
                        execute_runs=execute_runs,
                        **(
                            {"resume_acquisition": resume}
                            if stage.kind == "dataset_snapshot"
                            else {}
                        ),
                    )
                except (OSError, ValueError, KeyError, TypeError) as error:
                    result = self._result("FAILED", "DO_NOT_ADVANCE", reason=str(error))
                from klab.recovery.provenance import (
                    declaration_paths,
                    git_provenance,
                )

                current = (
                    self.execution_context.refresh()
                    if self.execution_context is not None
                    else git_provenance(declaration_paths(self.source, "campaign"))
                )
                current["closure"] = self._provenance_closure(current)
                self._assert_declaration_identity(state, current)
                row.update(result)
                if self.preparation_context is not None and row["state"] == "COMPLETE":
                    self._verify(stage, row, rows)
                row["attempts"][-1] = {"state": row["state"], "reason": row["reason"]}
                if row["state"] in {
                    "COMPLETE",
                    "BLOCKED",
                    "FAILED",
                    "INCONCLUSIVE",
                    "INTERRUPTED",
                } and not (
                    stage.kind == "dataset_snapshot" and row["state"] == "INTERRUPTED"
                ):
                    row = self.store.commit(stage, row)
                self._record(state, row)
                if self.after_commit is not None and row["state"] == "COMPLETE":
                    self.after_commit(stage.id, state)
                if row["state"] != "COMPLETE":
                    return {
                        **self._project(state),
                        "declaration_provenance": provenance,
                        "storage_checks": checks,
                    }
                if only_stage is not None:
                    return {
                        **self._project(state),
                        "declaration_provenance": provenance,
                        "storage_checks": checks,
                    }

    def _require_attempt_contract(
        self, stage: Any, rows: dict[str, dict], *, only_stage: str | None
    ) -> None:
        """Fail before dispatch unless a declared plan uses its charged v2 phase."""
        if stage.kind == "experiment_run":
            plan_name = stage.plan
        elif stage.kind in {"evaluation", "generation_panel"}:
            plan_name = self.stages[stage.collect].plan
        else:
            return
        lock = self._lock(rows, plan_name)
        reference = lock.execution.get("attempt_contract")
        if reference is None:
            return
        if only_stage != stage.id:
            raise ValueError("contracted Campaign execution requires --only-stage")
        from klab.training.attempt_budget import AttemptBudget

        path = os.environ.get("KLAB_ATTEMPT_BUDGET_LEDGER")
        label = os.environ.get("KLAB_ATTEMPT_PHASE_LABEL")
        if not path or not label:
            raise ValueError("contracted Campaign stage lacks an active v2 ledger")
        status = AttemptBudget(Path(path)).status()
        if (
            status.get("version") != 2
            or status.get("contract_sha256") != reference["sha256"]
            or status.get("contract_sha256")
            != os.environ.get("KLAB_ATTEMPT_CONTRACT_SHA256")
            or status.get("content_identity_sha256") != lock.scientific_sha256
            or status.get("content_identity_sha256")
            != os.environ.get("KLAB_ATTEMPT_CONTENT_IDENTITY_SHA256")
        ):
            raise ValueError("Campaign phase differs from pinned attempt contract")
        reservations = [
            item for item in status["reservations"] if item["label"] == label
        ]
        if len(reservations) != 1 or reservations[0]["actual_updates"] is not None:
            raise ValueError("Campaign phase lacks one unused ledger reservation")
        reserved = reservations[0]
        if stage.kind == "experiment_run":
            if self.stages[stage.runtime].worker is not None:
                raise ValueError(
                    "contracted Campaign run requires an owned local runtime profile"
                )
            cells = [cell for cell in lock.cells if cell.id == stage.cell]
            if len(cells) != 1 or (
                reserved["updates"] < cells[0].config.training.max_steps
                or reserved["target_positions"] < cells[0].config.training.max_tokens
                or os.environ.get("KLAB_ATTEMPT_ACTIVITY") != "train"
            ):
                raise ValueError("Campaign run exceeds attempt reservation")
        elif stage.kind == "generation_panel":
            from klab.evaluation.panel import load_panel

            panel = load_panel(self._path(stage.panel))
            if (
                reserved["generation_calls"] < len(panel.prompts)
                or reserved["generated_tokens"]
                < len(panel.prompts) * panel.decoder.max_new_tokens
                or os.environ.get("KLAB_ATTEMPT_ACTIVITY") != "evaluate"
            ):
                raise ValueError("Campaign panel exceeds attempt reservation")
        else:
            from klab.evaluation.suite import load_suite

            suite = load_suite(self._path(stage.suite))
            if (
                any(
                    reserved[key]
                    for key in (
                        "updates",
                        "target_positions",
                        "generation_calls",
                        "generated_tokens",
                    )
                )
                or os.environ.get("KLAB_ATTEMPT_ACTIVITY") != "evaluate"
                or any(
                    item.kind not in {"heldout_lm", "surface_review"}
                    for item in suite.evaluations
                )
            ):
                raise ValueError("Campaign evaluation requires a zero-counter suite")

    @_verified_operation
    def approve(
        self, gate: str, decision: str = "approve", note: str | None = None
    ) -> dict:
        if decision not in {"approve", "reject"}:
            raise ValueError("approval decision must be approve or reject")
        if self.execution_context is not None:
            self.execution_context.refresh()
        with self.store.locked():
            state = self.store.read()
            if state is None:
                raise ValueError("campaign has no pending gate")
            from klab.recovery.provenance import declaration_paths, git_provenance

            provenance = (
                self.execution_context.refresh()
                if self.execution_context is not None
                else git_provenance(declaration_paths(self.source, "campaign"))
            )
            provenance["closure"] = self._provenance_closure(provenance)
            self._require_declaration_files(provenance)
            self._assert_declaration_identity(state, provenance)
            state = self.store.reconcile(state)
            self._assert_declaration_identity(state, provenance)
            self._project(state)
            stage = self.stages.get(gate)
            if stage is None or stage.kind != "approval":
                raise ValueError(f"unknown approval gate {gate}")
            rows = self._rows(state)
            row = rows[gate]
            if row["state"] not in {"AWAITING_APPROVAL", "COMPLETE", "BLOCKED"}:
                raise ValueError(f"gate {gate} is not awaiting approval")
            for name in stage.bind:
                self._upstream(rows, name)
            binding, identities = self._binding(stage, rows)
            receipt = self.store.approval(stage, binding, decision, note, identities)
            outcome = (
                "READY_FOR_NEXT_STAGE" if decision == "approve" else "DO_NOT_ADVANCE"
            )
            result = self._result(
                "COMPLETE" if decision == "approve" else "BLOCKED",
                outcome,
                reason=f"approval {decision}",
                outputs=self._identity("approval", gate, binding),
                bindings=identities,
                approval=receipt,
            )
            row.update(result)
            row["stage_input_sha256"] = self.store.input_sha(stage, rows)
            row = self.store.commit(stage, row)
            self._record(state, row)
            return self._project(state)

    def _attempts(
        self, controller: Any, lock: Any, stage: Any, rows: dict
    ) -> list[dict]:
        from klab.experiments.binding import inspect_runtime_binding
        from klab.experiments.cli import locked_cell_request
        from klab.training.manifest import config_sha256
        from klab.workers.models import ExperimentSpec

        cell = stage.cell

        matches = [
            row
            for row in controller.list_experiments()
            if (row["spec"].get("plan") or {}).get("plan_sha256") == lock.plan_sha256
            and (row["spec"].get("plan") or {}).get("cell_id") == cell
        ]
        selected = next(item for item in lock.cells if item.id == cell)
        entry = (
            self._upstream(rows, stage.runtime)
            .get("availability", {})
            .get("runtime_bindings", {})
            .get(cell)
        )
        runtime_digest = None
        if entry is not None:
            binding = inspect_runtime_binding(Path(entry["path"]), lock, selected)
            runtime_digest = binding["binding_sha256"]
            if runtime_digest != entry["binding_sha256"]:
                raise ValueError("accepted runtime digest differs from receipt")
        elif (
            selected.config.runtime.backend != "cpu"
            or selected.config.runtime.engine != "pytorch"
        ):
            raise ValueError(
                "RUNTIME_REQUIRED: attempted accelerator cell has no accepted binding"
            )
        for row in matches:
            spec = ExperimentSpec.model_validate(row["spec"])
            expected = locked_cell_request(
                lock,
                selected,
                spec.bound_worker,
                controller.root.parent,
                controller,
                read_only=True,
                runtime_binding_sha256=runtime_digest,
                workflow_context=self.workflow_context,
            )
            if (
                spec.config.model_dump(mode="json")
                != selected.config.model_dump(mode="json")
                or spec.config_sha256 != selected.config_sha256
                or config_sha256(spec.config.model_dump(mode="json"))
                != selected.config_sha256
                or spec.plan.model_dump(mode="json") != expected["plan"]
            ):
                raise ValueError(f"controller attempt does not bind locked cell {cell}")
            continuation = spec.continuation
            parent = (
                expected.get("resume")
                or expected.get("extend_budget")
                or expected.get("promote")
            )
            expected_kind = (
                "PROMOTED"
                if "promote" in expected
                else "RESUMED"
                if parent
                else "FRESH"
            )
            if (
                continuation.kind != expected_kind
                or continuation.allow_runtime_drift
                or continuation.budget_extension != ("extend_budget" in expected)
                or continuation.checkpoint_sha256
                != expected["plan"]["parent_checkpoint_sha256"]
            ):
                raise ValueError(
                    f"controller continuation does not bind locked cell {cell}"
                )
            if parent:
                self._check_continuation_parent(parent, continuation)
        return matches

    @staticmethod
    def _check_continuation_parent(parent: Path, continuation: Any) -> None:
        """Bind a previously verified generation to its authenticated run manifest."""
        from klab.canonical import canonical_digest
        from klab.training.manifest import read_manifest

        run_manifest_path = parent.parent.parent / "manifest.json"
        run_manifest = read_manifest(run_manifest_path)
        run_digest = canonical_digest(run_manifest)
        generation = json.loads((parent / "manifest.json").read_text())
        identity = continuation.artifact_identity
        if (
            continuation.parent_run_id != run_manifest["run_id"]
            or identity is None
            or identity.relative_path != "continuation/parent_manifest.json"
            or identity.sha256 != run_digest
            or identity.size_bytes != run_manifest_path.stat().st_size
            or generation.get("manifest_sha256") != run_digest
            or generation.get("sha256") != continuation.checkpoint_sha256
        ):
            raise ValueError("controller parent does not bind locked continuation")

    @staticmethod
    def _run_identity(row: dict) -> dict:
        return {key: row[key] for key in ("experiment_id", "attempt_id", "run_id")}

    @_verified_operation
    def dispatch(
        self,
        stage: Any,
        rows: dict[str, dict],
        max_wait_seconds: float,
        *,
        execute_runs: bool = False,
        resume_acquisition: bool = False,
    ) -> dict:
        """Run one declared adapter; return science identities separately from availability."""
        kind = stage.kind
        input_sha = rows[stage.id]["stage_input_sha256"]
        if kind == "dataset_snapshot":
            return self._dispatch_snapshot(stage, resume_acquisition=resume_acquisition)
        if kind in {"tokenizer_train", "data_prepare"}:
            from klab.campaign.preparation import dispatch

            with (
                self.preparation_context.dispatch_scope()
                if self.preparation_context is not None
                else nullcontext()
            ):
                result = dispatch(
                    self._preparation_request(stage.id, rows),
                    observer=self.observer,
                    **self._verification(),
                )
            return self._result(
                outputs=[result.artifact.model_dump(mode="json")],
                availability={"path": str(result.path)},
            )
        if kind in {"artifact_reference", "tokenizer_reference"}:
            from klab.experiments.artifacts import verify_artifact

            path = self._operational_path(self._artifact_declaration(stage).path)
            if not path.exists():
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason=f"missing declared artifact: {path}",
                )
            verified = verify_artifact(
                self._artifact_declaration(stage),
                self.source,
                dataset=self._artifact_dataset(stage),
                **self._verification(),
            )
            return self._result(
                outputs=[
                    {key: verified[key] for key in ("kind", "identifier", "sha256")}
                ],
                availability={"path": verified["path"]},
            )
        if kind == "corpus_release":
            from klab.corpus.acquisition import acquire
            from klab.corpus.pipeline import build
            from klab.corpus.project import load_project
            from klab.corpus.release import freeze, verify_release

            path = self._path(stage.project)
            if not path.exists():
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason=f"missing corpus project: {path}",
                )
            project = load_project(path)
            if any(
                source.kind not in {"local", "deterministic_generator"}
                for source in project.sources
            ):
                raise ValueError(
                    "campaign corpus sources must be local or deterministic_generator"
                )
            acquire(project, self.store.root, offline=False)
            release = freeze(
                build(project, self.store.root, offline=True), self.store.root
            )
            identity = verify_release(release)["release_id"]
            return self._result(
                outputs=self._identity("corpus_release", identity, identity),
                availability={"path": str(release)},
            )
        if kind == "corpus_readiness":
            from klab.campaign.policy import measure_readiness

            release, _ = self._release(rows, stage.corpus)
            tokenizer = (
                Path(self._upstream(rows, stage.tokenizer)["availability"]["path"])
                if stage.tokenizer
                else None
            )
            measured = measure_readiness(
                release,
                stage.policy,
                tokenizer=tokenizer,
                measurement_receipt=(
                    self._operational_path(stage.measurement_receipt)
                    if stage.measurement_receipt is not None
                    else None
                ),
                measurement_sha256=stage.measurement_sha256,
                _scratch=self.store.root / "scratch" / "corpus-readiness",
            )
            result = self._result(
                measured["state"],
                measured["outcome"],
                reason="corpus policy measured",
                deficits=measured["deficits"],
                measurements=measured["measurements"],
            )
            if result["state"] == "COMPLETE":
                result["outputs"] = self._identity(
                    kind,
                    stage.id,
                    digest(
                        f"campaign-{kind}-v1",
                        {
                            "stage_input_sha256": input_sha,
                            "measurements": result["measurements"],
                        },
                    ),
                )
            return result
        if kind == "token_measurement":
            from klab.corpus.release import describe

            release, release_id = self._release(rows, stage.corpus)
            tokenizer = self._upstream(rows, stage.tokenizer)
            measured = describe(
                release, tokenizer=Path(tokenizer["availability"]["path"])
            )
            facts = {
                "release_id": release_id,
                "tokenizer_sha256": tokenizer["outputs"][0]["sha256"],
                "token_totals": measured["token_totals"],
                "views": measured["measurement"]["views"],
            }
            return self._result(
                measurements=facts,
                outputs=self._identity(
                    kind,
                    stage.id,
                    digest(
                        f"campaign-{kind}-v1",
                        {"stage_input_sha256": input_sha, "measurements": facts},
                    ),
                ),
            )
        if kind == "experiment_plan":
            from klab.canonical import canonical_json
            from klab.experiments.lock import open_lock, publish_lock, resolve_plan
            from klab.experiments.plan import load_plan
            from klab.experiments.prepare import prepare_plan

            source = self._path(stage.source)
            if not source.exists():
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason=f"missing experiment plan: {source}",
                )
            plan = (
                self.workflow_context.loaded.plan
                if self.workflow_context is not None
                else load_plan(source)
            )
            if stage.mode == "bind":
                from klab.experiments.direct_inputs import bind_direct_inputs

                prepared_row = self._upstream(rows, stage.prepared)
                destination = (
                    self.store.root
                    / "experiments"
                    / f"{stage.id}-{plan.id}"
                    / "bound-plan.yaml"
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                if destination.exists():
                    import tempfile

                    with tempfile.TemporaryDirectory(
                        prefix=".bind-verify-", dir=destination.parent
                    ) as temporary:
                        expected = bind_direct_inputs(
                            self._path(self.stages[stage.prepared].config),
                            source,
                            Path(prepared_row["availability"]["path"]),
                            Path(temporary) / "plan.yaml",
                            rebase_suite=True,
                            **self._verification(),
                        )
                    plan = load_plan(destination)
                    if plan != expected:
                        raise ValueError(
                            "existing bound plan differs from declared native inputs"
                        )
                else:
                    plan = bind_direct_inputs(
                        self._path(self.stages[stage.prepared].config),
                        source,
                        Path(prepared_row["availability"]["path"]),
                        destination,
                        rebase_suite=True,
                        **self._verification(),
                    )
                source = destination
            from klab.experiments.plan import base_run_config

            config = (
                self.workflow_context.config
                if self.workflow_context is not None
                else base_run_config(plan, source)
            )
            configured_checks = [
                check
                for kind, target in (
                    ("cache", config.dataset.cache_dir),
                    ("output", config.logging.root_dir),
                )
                if Path(target).is_absolute()
                for check in storage_checks(target, kind=kind)
            ]
            warn_storage_checks(configured_checks)
            workspace = self.store.root / "experiments" / f"{stage.id}-{plan.id}"
            if stage.mode == "reference":
                path = self._operational_path(stage.lock)
                if not path.exists():
                    return self._result(
                        "BLOCKED",
                        "DO_NOT_ADVANCE",
                        reason=f"missing experiment lock: {path}",
                    )
                lock = open_lock(path, **self._verification())
                self._check_lock(stage, rows, lock)
            else:
                prepared = None
                if plan.corpus_variants:
                    record_path = workspace / "preparation.json"
                    if record_path.exists():
                        raw = record_path.read_bytes()
                        prepared = json.loads(raw)
                        if (
                            raw != canonical_json(prepared) + b"\n"
                            or prepared.get("format") != "experiment-preparation-v1"
                            or prepared.get("id") != plan.id
                        ):
                            raise ValueError(
                                "existing immutable preparation receipt is invalid"
                            )
                    else:
                        prepared = prepare_plan(
                            plan, source, workspace, **self._verification()
                        )
                        with record_path.open("xb") as handle:
                            handle.write(canonical_json(prepared) + b"\n")
                            handle.flush()
                            os.fsync(handle.fileno())
                if self.workflow_context is not None:
                    from klab.experiments.workflow_recipe import resolve_workflow

                    self.workflow_context.refresh()
                    lock = resolve_workflow(
                        self.workflow_context.source, self.workflow_context.bindings
                    )
                else:
                    lock = resolve_plan(
                        plan, source, prepared=prepared, **self._verification()
                    )
                from klab.data.legacy import require_current_dataset

                for cell in lock.cells:
                    require_current_dataset(cell.config.dataset)
                self._check_lock(stage, rows, lock)
                path = publish_lock(lock, workspace)
                lock = open_lock(path, **self._verification())
            return self._result(
                outputs=self._identity("experiment_plan", lock.id, lock.plan_sha256),
                availability={"path": str(path), "workspace": str(workspace)},
                storage_checks=[*rows[stage.id]["storage_checks"], *configured_checks],
            )
        if kind == "runtime_acceptance":
            from klab.workspace_preflight import training_storage_checks

            lock = self._lock(rows, stage.plan)
            selected = {
                item.cell
                for item in self.plan.stages
                if item.kind == "experiment_run"
                and item.plan == stage.plan
                and item.runtime == stage.id
            }
            if not selected:
                if len(lock.cells) != 1:
                    raise ValueError("runtime has no unambiguous declared cell")
                selected = {lock.cells[0].id}
            cells = [cell for cell in lock.cells if cell.id in selected]
            if len(cells) != len(selected):
                raise ValueError("runtime references an absent locked cell")
            from klab.data.legacy import require_current_dataset

            for cell in cells:
                require_current_dataset(cell.config.dataset)
            profile = self.runtime_profile if stage.profile_id is not None else None
            if stage.worker is not None and self.runtime_profile is not None:
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="named worker runtime rejects a simultaneous profile",
                )
            if stage.profile_id is not None and (
                profile is None or profile.id != stage.profile_id
            ):
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="RUNTIME_REQUIRED: matching profile_id",
                )
            if (
                profile is None
                and stage.worker is None
                and any(
                    cell.config.runtime.backend != "cpu"
                    or cell.config.runtime.engine != "pytorch"
                    for cell in cells
                )
            ):
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="RUNTIME_REQUIRED: explicit profile or worker",
                )
            workspace = Path(rows[stage.plan]["availability"]["workspace"])
            checks = [
                asdict(check)
                for cell in cells
                for check in training_storage_checks(
                    cell.config, work_dir=workspace, run_dir=workspace / "controller"
                )
            ]
            if any(check["status"] != "adequate" for check in checks):
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="insufficient training storage headroom",
                    availability={"storage_checks": checks},
                )
            from klab.experiments.binding import bind_runtime, open_runtime_binding
            from klab.workers.controller import Controller

            bindings = {}
            if profile is not None or stage.worker is not None:
                controller = Controller(workspace / "controller")
                for cell in cells:
                    try:
                        path = bind_runtime(
                            lock,
                            cell,
                            workspace,
                            profile=profile,
                            worker=stage.worker,
                            controller=controller,
                        )
                        binding = open_runtime_binding(
                            path, lock, cell, controller=controller
                        )
                    except (OSError, ValueError, RuntimeError) as error:
                        return self._result(
                            "BLOCKED", "DO_NOT_ADVANCE", reason=str(error)
                        )
                    bindings[cell.id] = {
                        "path": str(path),
                        "binding_sha256": binding["binding_sha256"],
                        "tested_runtime": binding["tested_runtime"],
                    }
            facts = {
                "headroom": "adequate",
                "cells": sorted(selected),
                "runtimes": {
                    cell.id: cell.config.runtime.model_dump(mode="json")
                    for cell in cells
                },
                "runtime_bindings": {
                    key: value["binding_sha256"] for key, value in bindings.items()
                },
            }
            return self._result(
                measurements=facts,
                availability={"storage_checks": checks, "runtime_bindings": bindings},
                outputs=self._identity(
                    kind,
                    stage.id,
                    digest(
                        f"campaign-{kind}-v1",
                        {"stage_input_sha256": input_sha, "measurements": facts},
                    ),
                ),
            )
        if kind == "approval":
            binding, identities = self._binding(stage, rows)
            approval = self.store.read_approval(stage, binding)
            if approval is None:
                return self._result(
                    "AWAITING_APPROVAL",
                    "DO_NOT_ADVANCE",
                    reason="approval required",
                    bindings=identities,
                    binding_sha256=binding,
                )
            if approval["decision"] == "reject":
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="approval rejected",
                    bindings=identities,
                    binding_sha256=binding,
                )
            return self._result(
                outputs=self._identity("approval", stage.id, binding),
                bindings=identities,
                approval=approval,
            )
        if kind == "experiment_run":
            from klab.experiments.cli import locked_cell_request
            from klab.workers.controller import Controller
            from klab.workers.models import WorkerDefinition

            lock = self._lock(rows, stage.plan)
            cells = [cell for cell in lock.cells if cell.id == stage.cell]
            if len(cells) != 1:
                raise ValueError("experiment_run must select exactly one locked cell")
            cell = cells[0]
            workspace = Path(rows[stage.plan]["availability"]["workspace"])
            if not execute_runs and not self._submitted_run(stage, rows):
                raise ValueError("new model execution requires --execute-runs")
            controller = Controller(
                workspace / "controller",
                observer=self.observer,
                **self._verification(),
            )
            matches = self._attempts(controller, lock, stage, rows)
            if len(matches) > 1:
                raise ValueError("ambiguous controller attempts for locked cell")
            previous = rows[stage.id].get("measurements", {})
            if (
                isinstance(previous, dict)
                and {"experiment_id", "attempt_id", "run_id"} <= previous.keys()
                and (
                    not matches
                    or self._run_identity(matches[0])
                    != {
                        key: previous[key]
                        for key in ("experiment_id", "attempt_id", "run_id")
                    }
                )
            ):
                raise ValueError(
                    "LOST_SUBMISSION: refusing to replace prior run attempt"
                )
            if not matches:
                from klab.data.legacy import require_current_dataset

                require_current_dataset(cell.config.dataset)
                try:
                    binding = self._runtime_binding(
                        rows, stage.runtime, lock, cell, controller
                    )
                except (OSError, ValueError, RuntimeError) as error:
                    return self._result("BLOCKED", "DO_NOT_ADVANCE", reason=str(error))
                worker = f"campaign-{stage.id}-{lock.id}"
                runtime_digest = None
                if binding is not None:
                    runtime_digest = binding["binding_sha256"]
                    if binding["source_kind"] == "worker":
                        worker = binding["descriptor"]["name"]
                    else:
                        controller.register(
                            WorkerDefinition(
                                worker_id=worker,
                                name=worker,
                                transport="local",
                                python=Path(binding["descriptor"]["python"]),
                                root=workspace / "workers" / worker,
                                engine=cell.config.runtime.engine,
                                backend=cell.config.runtime.backend,
                                device_index=cell.config.runtime.device_index,
                            )
                        )
                else:
                    controller.register(
                        WorkerDefinition(
                            worker_id=worker,
                            name=worker,
                            transport="local",
                            python=Path(sys.executable).absolute(),
                            root=workspace / "workers" / worker,
                            engine="pytorch",
                            backend="cpu",
                            device_index=0,
                        )
                    )
                request = locked_cell_request(
                    lock,
                    cell,
                    worker,
                    workspace,
                    controller,
                    runtime_binding_sha256=runtime_digest,
                    workflow_context=self.workflow_context,
                )
                request["declaration_provenance"] = rows[stage.id][
                    "declaration_provenance"
                ]
                request["storage_checks"] = [
                    *rows[stage.id]["storage_checks"],
                    *(
                        check
                        for check in rows[stage.plan].get("storage_checks", [])
                        if check not in rows[stage.id]["storage_checks"]
                    ),
                ]
                controller.submit_many([request])
            deadline = time.monotonic() + max_wait_seconds
            while True:
                matches = self._attempts(controller, lock, stage, rows)
                if len(matches) != 1:
                    raise ValueError("controller submission missing or conflicting")
                attempt = matches[0]
                ids = self._run_identity(attempt)
                availability = {
                    "path": str(controller.root / attempt["run_id"]),
                    "workspace": str(workspace),
                }
                if (
                    attempt["status"] == "COMPLETE"
                    and attempt["ingestion_status"] == "COMPLETE"
                ):
                    from klab.evaluation.evidence import experiment_evidence

                    with self._phase(
                        "campaign.run_evidence", host_kind="verification_bound"
                    ):
                        experiment_evidence(
                            Path(availability["path"]), **self._verification()
                        )
                    return self._result(
                        outputs=self._identity(
                            "experiment_run",
                            attempt["run_id"],
                            digest("campaign-run-v1", ids),
                        ),
                        availability=availability,
                        measurements=ids,
                    )
                if (
                    attempt["status"] in {"FAILED", "CANCELLED", "INTERRUPTED"}
                    or attempt["ingestion_status"] == "ERROR"
                ):
                    return self._result(
                        "FAILED"
                        if attempt["status"] == "FAILED"
                        or attempt["ingestion_status"] == "ERROR"
                        else "INTERRUPTED",
                        "DO_NOT_ADVANCE",
                        reason=f"controller {attempt['status']}, ingestion {attempt['ingestion_status']}",
                        availability=availability,
                        measurements=ids,
                    )
                if time.monotonic() >= deadline:
                    return self._result(
                        "RUNNING",
                        "READY_FOR_NEXT_STAGE",
                        reason="worker active; resume to reconcile",
                        availability=availability,
                        measurements=ids,
                    )
                with self._phase("campaign.controller_tick"):
                    controller.tick(deadline=deadline)
                refreshed = self._attempts(controller, lock, stage, rows)
                if len(refreshed) != 1:
                    raise ValueError("controller submission missing or conflicting")
                # Re-enter the durable gate immediately; only an unchanged attempt
                # consumes idle time, and never beyond the shared deadline.
                if refreshed[0] != attempt:
                    continue
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    with self._phase("campaign.state_idle"):
                        time.sleep(min(0.1, remaining))
        if kind == "experiment_collect":
            from klab.experiments.evidence import collect_evidence, read_evidence

            lock = self._lock(rows, stage.plan)
            workspace = Path(rows[stage.plan]["availability"]["workspace"])
            with self._phase(
                "campaign.checkpoint_collection", host_kind="verification_bound"
            ):
                options = self._verification()
                if stage.evaluations:
                    options = {
                        **options,
                        "diagnostic_references": self._diagnostic_references(
                            stage, rows
                        ),
                    }
                evidence = collect_evidence(lock, workspace, **options)
                verified = read_evidence(
                    lock,
                    workspace,
                    Path(evidence["index_path"]),
                    **self._verification(),
                )
            cell = self._check_collection(stage, verified, rows)
            selected = self._selected_collection_checkpoint(cell)
            return self._result(
                outputs=self._identity(kind, stage.id, verified["index_sha256"]),
                measurements=selected,
                availability={
                    "path": verified["index_path"],
                    "workspace": str(workspace),
                },
            )
        if kind == "evaluation":
            from klab.evaluation.suite import run_suite, verify_evaluation_index

            collect = self.stages[stage.collect]
            lock = self._lock(rows, collect.plan)
            collected = self._upstream(rows, stage.collect)["measurements"]
            run = self._upstream(rows, collect.run)
            run_path = Path(run["availability"]["path"])
            from klab.evaluation.inference import evaluation_config
            from klab.runtime_profile import rederive_authorization
            from klab.workers.controller import Controller

            run_id = run["outputs"][0]["identifier"]
            checkpoint = str(run_path / "checkpoints" / collected["generation"])
            config = evaluation_config(
                run_id,
                run_path.parent,
                checkpoint,
                stage.backend,
                **self._verification(),
            )
            from klab.data.legacy import require_current_dataset

            require_current_dataset(config.dataset)
            authorization = None
            if stage.runtime is not None:
                run_stage = self.stages[collect.run]
                cell = next(cell for cell in lock.cells if cell.id == run_stage.cell)
                workspace = Path(rows[collect.plan]["availability"]["workspace"])
                try:
                    binding = self._runtime_binding(
                        rows,
                        stage.runtime,
                        lock,
                        cell,
                        Controller(workspace / "controller"),
                    )
                    if binding is None:
                        if config.runtime.backend != "cpu":
                            raise ValueError(
                                "RUNTIME_REQUIRED: evaluation binding absent"
                            )
                    else:
                        if (
                            binding["source_kind"] == "worker"
                            and binding["descriptor"]["transport"] != "local"
                        ):
                            raise ValueError(
                                "remote worker evaluation unsupported; declare backend: cpu"
                            )
                        authorization = rederive_authorization(
                            binding["runtime_authorization"], config
                        )
                except (OSError, ValueError, RuntimeError) as error:
                    return self._result("BLOCKED", "DO_NOT_ADVANCE", reason=str(error))
            elif config.runtime.backend != "cpu" or config.runtime.engine != "pytorch":
                return self._result(
                    "BLOCKED",
                    "DO_NOT_ADVANCE",
                    reason="RUNTIME_REQUIRED: accelerator checkpoint evaluation needs runtime",
                )
            with self._phase("campaign.evaluation"):
                index_path = run_suite(
                    self._path(stage.suite),
                    run_id,
                    checkpoint,
                    run_path.parent,
                    stage.backend,
                    authorization=authorization,
                    **self._verification(),
                )
            index = verify_evaluation_index(index_path, **self._verification())
            if (
                index["run_id"] != run["outputs"][0]["identifier"]
                or index["checkpoint_sha256"] != collected["sha256"]
                or index["checkpoint"] != f"checkpoints/{collected['generation']}"
            ):
                raise ValueError("suite index differs from collected run/checkpoint")
            facts = {
                "checkpoint_sha256": index["checkpoint_sha256"],
                "index_sha256": index["index_sha256"],
                "evaluation_runtime": index["evaluation_runtime"],
                "evaluations": [
                    {"id": item["id"], "status": item["status"]}
                    for item in index["evaluations"]
                ],
            }
            return self._result(
                measurements=facts,
                outputs=self._identity(kind, stage.id, index["index_sha256"]),
                availability={"path": str(index_path)},
            )
        if kind == "generation_panel":
            from klab.evaluation.inference import evaluation_config
            from klab.evaluation.panel import run_panel, verify_panel_result
            from klab.runtime_profile import rederive_authorization
            from klab.workers.controller import Controller

            collect = self.stages[stage.collect]
            lock = self._lock(rows, collect.plan)
            collected = self._upstream(rows, stage.collect)["measurements"]
            run = self._upstream(rows, collect.run)
            evaluation = self._upstream(rows, stage.evaluation)
            runtime_identity = self._panel_runtime_identity(stage, rows)
            run_path = Path(run["availability"]["path"])
            run_id = run["outputs"][0]["identifier"]
            checkpoint = str(run_path / "checkpoints" / collected["generation"])
            config = evaluation_config(
                run_id,
                run_path.parent,
                checkpoint,
                stage.backend,
                **self._verification(),
            )
            from klab.data.legacy import require_current_dataset

            require_current_dataset(config.dataset)
            run_stage = self.stages[collect.run]
            cell = next(cell for cell in lock.cells if cell.id == run_stage.cell)
            workspace = Path(rows[collect.plan]["availability"]["workspace"])
            authorization = None
            try:
                binding = self._runtime_binding(
                    rows,
                    stage.runtime,
                    lock,
                    cell,
                    Controller(workspace / "controller"),
                )
                if binding is None:
                    if (
                        config.runtime.backend != "cpu"
                        or config.runtime.engine != "pytorch"
                    ):
                        raise ValueError(
                            "RUNTIME_REQUIRED: generation panel binding absent"
                        )
                else:
                    if (
                        binding["source_kind"] == "worker"
                        and binding["descriptor"]["transport"] != "local"
                    ):
                        raise ValueError("remote worker generation panel unsupported")
                    authorization = rederive_authorization(
                        binding["runtime_authorization"], config
                    )
                    if (
                        binding["binding_sha256"]
                        != runtime_identity["runtime_binding_sha256"]
                    ):
                        raise ValueError(
                            "generation panel accepted runtime binding changed"
                        )
            except (OSError, ValueError, RuntimeError) as error:
                return self._result("BLOCKED", "DO_NOT_ADVANCE", reason=str(error))
            with self._phase("campaign.generation_panel"):
                result_path = run_panel(
                    self._path(stage.panel),
                    Path(evaluation["availability"]["path"]),
                    backend=stage.backend,
                    authorization=authorization,
                    runtime_binding=runtime_identity,
                    **self._verification(),
                )
            result = verify_panel_result(result_path, **self._verification())
            if (
                result["run_id"] != run_id
                or result["checkpoint_sha256"] != collected["sha256"]
                or result["checkpoint"] != f"checkpoints/{collected['generation']}"
                or result["evaluation_index_sha256"]
                != evaluation["outputs"][0]["sha256"]
                or result["runtime_binding"] != runtime_identity
            ):
                raise ValueError(
                    "generation panel differs from collected evidence/runtime"
                )
            # A verified record of negative or failed generations is complete
            # descriptive evidence, never a readiness assessment or promotion.
            return self._result(
                outcome="DESCRIPTIVE_EVIDENCE",
                reason="descriptive generation panel recorded; no readiness assessment",
                measurements={
                    "checkpoint_sha256": result["checkpoint_sha256"],
                    "evaluation_index_sha256": result["evaluation_index_sha256"],
                    "runtime_binding": runtime_identity,
                    "rows": [
                        {"index": index, "status": item["status"]}
                        for index, item in enumerate(result["rows"])
                    ],
                },
                outputs=self._identity(kind, stage.id, result["record_sha256"]),
                availability={"path": str(result_path)},
            )
        if kind == "model_readiness":
            from klab.evaluation.readiness import (
                assess_readiness,
                verify_readiness_result,
            )

            evaluation = self._upstream(rows, stage.evaluation)
            result_path = assess_readiness(
                self._path(stage.policy),
                Path(evaluation["availability"]["path"]),
                self._operational_path(stage.review) if stage.review else None,
                self._operational_path(stage.reviewed_scores)
                if stage.reviewed_scores
                else None,
            )
            assessed = verify_readiness_result(result_path)
            state = assessed["state"]
            facts = {
                "checkpoint_sha256": assessed["checkpoint_sha256"],
                "index_sha256": assessed["index_sha256"],
                "completed_evaluations": assessed["completed_evaluations"],
                "missing_gates": assessed["missing_gates"],
            }
            if stage.reviewed_scores is not None:
                facts["reviewed_score_summary"] = assessed.get("reviewed_score_summary")
            result_state = (
                "COMPLETE"
                if state == "READY_FOR_NEXT_STAGE"
                else "BLOCKED"
                if state == "DO_NOT_ADVANCE"
                else "INCONCLUSIVE"
            )
            outcome = (
                state
                if state in {"READY_FOR_NEXT_STAGE", "DO_NOT_ADVANCE"}
                else "INCONCLUSIVE"
            )
            return self._result(
                result_state,
                outcome,
                reason=f"model policy assessed: {state}",
                measurements=facts,
                outputs=(
                    self._identity(kind, stage.id, assessed["result_sha256"])
                    if state == "READY_FOR_NEXT_STAGE"
                    else []
                ),
                availability={"path": str(result_path)},
            )
        raise ValueError(f"unsupported campaign adapter {kind}")
