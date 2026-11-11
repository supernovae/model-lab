"""Read-only, typed inspection of a declaration with an optional exact parent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from klab.campaign.plan import load_campaign
from klab.config.models import StrictModel
from klab.documents import read_document
from klab.experiments.compiler import describe_differences, flatten_config
from klab.experiments.lock import open_lock, resolve_plan, storage_preview
from klab.experiments.plan import base_run_config, load_plan
from klab.training.checkpoints import CheckpointManager
from klab.verification_proofs import verification_options
from klab.workdir import resolve_work_dir, storage_checks

IterationState = Literal[
    "SAFE_TO_PREPARE",
    "NEEDS_PILOT",
    "NEEDS_APPROVAL",
    "READY_TO_DISPATCH",
    "RUNNING",
    "BLOCKED",
]


class IterationResult(StrictModel):
    """Serialized contract for one read-only iteration observation."""

    format: Literal["klab-iteration-check-v1"] = "klab-iteration-check-v1"
    state: IterationState
    declaration: str
    parent: dict[str, Any]
    declared_delta: list[str]
    delta: list[dict[str, Any]]
    artifacts: dict[str, Any]
    storage: list[dict[str, Any]]
    runtime: dict[str, Any]
    ingestion: list[dict[str, Any]] | None
    campaign: dict[str, Any] | None
    missing_gates: list[dict[str, str]]
    next_command: str
    id: str | None = None
    plan_sha256: str | None = None
    scientific_sha256: str | None = None
    proposed_plan_sha256: str | None = None
    cell: str | None = None
    transition: str | None = None
    verification: dict[str, Any]


def _gate(gates: list[dict[str, str]], code: str, reason: str) -> None:
    gates.append({"code": code, "reason": reason})


def _storage(config: Any, work: Path, retention: dict[str, Any]) -> dict[str, Any]:
    from dataclasses import asdict

    from klab.workspace_preflight import training_storage_checks

    estimate = storage_preview(config, retention)
    checks = [asdict(item) for item in training_storage_checks(config, work_dir=work)]
    return {
        "estimate": estimate,
        "capacity": checks,
        "location_warnings": storage_checks(config.logging.root_dir),
    }


def _parent(path: Path, options: dict[str, Any]) -> tuple[dict[str, Any], Any | None]:
    result: dict[str, Any] = {
        "path": str(path),
        "valid": False,
        "full_state": False,
        "terminal": False,
        "compatible": False,
    }
    if (
        path.name in {"latest.json", "best.json"}
        or not path.name.startswith("step_")
        or "_gen_" not in path.name
    ):
        result["errors"] = [
            {
                "field": "parent",
                "reason": "parent must be an exact generation directory",
            }
        ]
        return result, None
    from klab.canonical import canonical_digest
    from klab.training.manifest import config_sha256, read_manifest

    root = path.parent.parent
    run_manifest = read_manifest(root / "manifest.json")
    manifest_sha256 = canonical_digest(run_manifest)
    manager = CheckpointManager(root, manifest_sha256=manifest_sha256)
    report = manager.verify(path, expected_manifest=manifest_sha256, **options)
    result.update(
        valid=report.valid,
        full_state=report.valid and report.resume_level == "full",
        resume_level=report.resume_level,
        errors=list(report.errors),
        files=list(report.verified_files),
    )
    if not report.valid:
        return result, None
    manifest = json.loads((path / "manifest.json").read_text())
    result.update(
        sha256=manifest["sha256"],
        run_id=manifest.get("run_id"),
        step=manifest.get("step"),
        tokens_seen=manifest.get("tokens_seen"),
        source_sha256=manifest.get("source_identity_sha256"),
    )
    # The manager authenticates and decodes the native full-state snapshot.
    snapshot = manager.load(path, "resume", **options)
    if (
        config_sha256(snapshot.config)
        != config_sha256(run_manifest["effective_config"])
        or snapshot.source_identity_sha256
        != run_manifest.get("source_identity", {}).get("sha256")
        or snapshot.run_id != run_manifest["run_id"]
    ):
        raise ValueError(
            "checkpoint scientific identity differs from its source run manifest"
        )
    return result, snapshot


def _parent_tokenizer(path: Path) -> Path:
    """Authenticate the run-owned tokenizer before accepting a location rebind."""
    from klab.training.manifest import read_manifest, sha256_file

    root = path.parent.parent
    manifest = read_manifest(root / "manifest.json")
    record = next(
        (
            item
            for item in manifest.get("artifacts", [])
            if item.get("relative_path") == "tokenizer.json"
        ),
        None,
    )
    member = root / "tokenizer.json"
    if (
        record is None
        or member.is_symlink()
        or not member.is_file()
        or member.stat().st_size != record["size_bytes"]
        or sha256_file(member) != record["sha256"]
    ):
        raise ValueError(
            "parent run tokenizer is missing or differs from its bound artifact inventory"
        )
    return member


def _source_closure_matches(lock: Any, source: Path) -> bool:
    from klab.recovery.provenance import declaration_paths, repository_root
    from klab.training.manifest import sha256_file

    root = repository_root(source) or source.parent
    actual = {
        item.relative_to(root).as_posix(): sha256_file(item)
        for item in declaration_paths(source, "experiment")
    }
    return actual == lock.availability.get("declaration_hashes")


def _selected_parent_gate(
    run: Path,
    phase: Any,
    supplied: Path,
    digest: str,
    parent_report: dict[str, Any],
    gates: list[dict[str, str]],
) -> None:
    """Select using the same native read-only selector as child dispatch."""
    from klab.experiments.binding import select_generation

    try:
        selected = select_generation(
            run,
            selector=phase.selector,
            at_step=phase.at_step,
            full_state=phase.transition != "promote",
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        _gate(gates, "PARENT_BINDING_REQUIRED", str(error))
        return
    parent_report["selected_generation"] = selected
    if (
        Path(selected["checkpoint_path"]).resolve() != supplied.resolve()
        or selected["checkpoint_sha256"] != digest
    ):
        _gate(
            gates,
            "PARENT_BINDING_MISMATCH",
            "supplied exact generation is not the selected completed, ingested parent run generation",
        )


def _declaration_delta(
    plan: Any, source: Path, parent: Any
) -> tuple[list[str], list[dict[str, Any]], Any, Any]:
    """Project one authored patch through native axis and phase compilers."""
    from klab.config.models import RunConfig
    from klab.experiments.compiler import apply_patch, expand_axes

    expanded = expand_axes(
        base_run_config(plan, source), plan.axes, max_runs=1000, base_dir=source.parent
    )
    if len(expanded) != 1:
        raise ValueError(
            "authored declaration selects multiple cells; publish a lock for exact parent binding"
        )
    old = RunConfig.model_validate(parent.config)
    candidates = [
        phase
        for phase in plan.phases
        if phase.transition != "fresh"
        and (
            phase.checkpoint is None
            or plan.artifacts[phase.checkpoint].sha256 == parent.checkpoint_sha256
        )
        and (phase.at_step is None or phase.at_step == parent.step)
    ]
    if len(candidates) != 1:
        raise ValueError("authored declaration needs one unambiguous parent transition")
    phase = candidates[0]
    candidate = RunConfig.model_validate(
        apply_patch(expanded[0][0], phase.set, source=f"phase {phase.id}")
    )
    before = flatten_config(old, include_derived=True)
    after = flatten_config(candidate, include_derived=True)
    differences = list(
        describe_differences(
            {
                key: {"base": before.get(key), "variant": after.get(key)}
                for key in sorted(before.keys() | after.keys())
                if before.get(key) != after.get(key)
            }
        )
    )
    changed = {row["path"] for row in differences}
    missing = [
        field
        for field in phase.set
        if not any(key == field or key.startswith(field + ".") for key in changed)
    ]
    if missing:
        raise ValueError(
            f"declared interventions unchanged: {', '.join(sorted(missing))}"
        )
    return sorted(phase.set), differences, phase, candidate


def _fresh_delta(
    plan: Any, source: Path, *, context=None
) -> tuple[list[str], list[dict[str, Any]], Any]:
    """Compare one fresh phase's authored patch with its native expanded base."""
    from klab.config.models import RunConfig
    from klab.experiments.compiler import apply_patch, expand_axes
    from klab.experiments.plan import recipe_run_config

    base = (
        base_run_config(plan, source)
        if context is None
        else recipe_run_config(plan, source, context=context)
    )
    expanded = expand_axes(base, plan.axes, max_runs=1000, base_dir=source.parent)
    fresh = [phase for phase in plan.phases if phase.transition == "fresh"]
    if len(expanded) != 1 or len(plan.phases) != 1 or len(fresh) != 1:
        raise ValueError("parent-free inspection requires one unambiguous fresh cell")
    phase = fresh[0]
    before = flatten_config(expanded[0][0], include_derived=True)
    after = flatten_config(
        RunConfig.model_validate(
            apply_patch(expanded[0][0], phase.set, source=f"phase {phase.id}")
        ),
        include_derived=True,
    )
    differences = list(
        describe_differences(
            {
                key: {"base": before.get(key), "variant": after.get(key)}
                for key in sorted(before.keys() | after.keys())
                if before.get(key) != after.get(key)
            }
        )
    )
    return sorted(phase.set), differences, phase


def _classification(
    gates: list[dict[str, str]],
    *,
    campaign: dict[str, Any] | None,
    runtime_needed: bool,
    locked: bool,
    pilot_needed: bool,
) -> tuple[IterationState, str]:
    codes = {gate["code"] for gate in gates}
    hard = codes - {
        "LOCK_REQUIRED",
        "RUNTIME_REQUIRED",
        "PILOT_REQUIRED",
        "APPROVAL_REQUIRED",
    }
    if hard:
        return (
            "BLOCKED",
            "klab campaign explain" if campaign else "klab iteration check",
        )
    if campaign is not None:
        action = campaign.get("next_action", {})
        if any(row.get("state") == "RUNNING" for row in campaign.get("stages", [])):
            return "RUNNING", "klab campaign status"
        if action.get("action") == "approve":
            return "NEEDS_APPROVAL", "klab campaign approve"
        if action.get("action") in {"wait", "resume"}:
            return "BLOCKED", "klab campaign explain"
        next_row = next(
            (
                row
                for row in campaign.get("stages", [])
                if row["id"] == action.get("stage")
            ),
            None,
        )
        if next_row is None or action.get("action") != "apply":
            return "BLOCKED", "klab campaign explain"
        if next_row.get("kind") != "experiment_run":
            return (
                ("NEEDS_PILOT", "klab campaign next")
                if next_row.get("kind") == "runtime_acceptance"
                else ("SAFE_TO_PREPARE", "klab campaign next")
            )
    if not locked:
        return "SAFE_TO_PREPARE", "klab experiment lock"
    if pilot_needed:
        return "NEEDS_PILOT", "klab stage"
    if runtime_needed:
        return "NEEDS_PILOT", "klab experiment bind"
    if "APPROVAL_REQUIRED" in codes:
        return "NEEDS_APPROVAL", "klab campaign approve"
    return (
        "READY_TO_DISPATCH",
        "klab campaign apply" if campaign else "klab experiment run",
    )


def check_iteration(
    source: Path,
    parent: Path | None = None,
    *,
    work_dir: Path | None = None,
    cold_verify: bool = False,
) -> dict[str, Any]:
    """Inspect native evidence without publishing a lock, binding, proof, or Campaign state.

    An omitted parent applies only to fresh phases. Continuations report
    PARENT_REQUIRED until supplied an exact generation; no parent pointer is
    followed and no compatibility is inferred. Proof reuse is lookup-only:
    cold fallback never publishes signed receipts.
    """
    source = Path(source).absolute()
    parent = Path(parent).absolute() if parent is not None else None
    work = resolve_work_dir(work_dir)
    options = verification_options(work, cold=cold_verify, read_only=True)
    store = options["proof_store"]
    gates: list[dict[str, str]] = []
    result: dict[str, Any] = {
        "declaration": str(source),
        "parent": {"path": str(parent)}
        if parent is not None
        else {
            "path": None,
            "status": "NOT_SUPPLIED",
            "valid": None,
            "full_state": None,
            "terminal": None,
            "compatible": None,
        },
        "declared_delta": [],
        "delta": [],
        "artifacts": {},
        "storage": [],
        "runtime": {},
        "ingestion": None,
        "campaign": None,
        "missing_gates": gates,
    }
    result["verification"] = {"mode": options["verification_mode"], "diagnostics": None}
    locked = None
    plan = None
    authored_plan = None
    authored_context = None
    campaign = None
    pilot_needed = True
    runtime_needed = True
    try:
        document = read_document(source)
        if "campaign_version" in document:
            declaration = load_campaign(source)
            from klab.campaign.engine import CampaignEngine

            engine = CampaignEngine(source, work)
            engine._verification_options = options
            campaign = engine.inspect("explain")
            result["campaign"] = campaign
            result["id"] = declaration.id
            plan_stages = [
                stage for stage in declaration.stages if stage.kind == "experiment_plan"
            ]
            if len(plan_stages) == 1:
                stage = plan_stages[0]
                rows = {row["id"]: row for row in campaign.get("stages", [])}
                reference = rows.get(stage.id, {}).get("availability", {}).get("path")
                if reference is None and stage.mode == "reference":
                    from klab.campaign.plan import operational_path

                    reference = str(operational_path(source.parent, stage.lock))
                if reference:
                    try:
                        locked = open_lock(Path(reference), **options)
                        result.update(
                            plan_sha256=locked.plan_sha256,
                            scientific_sha256=locked.scientific_sha256,
                        )
                        from klab.campaign.plan import safe_path

                        authored_path = safe_path(source.parent, stage.source)
                        if "workflow_profile" in locked.availability:
                            from klab.experiments.workflow_recipe import (
                                locked_workflow_context,
                            )

                            workflow = locked_workflow_context(locked, authored_path)
                            authored_plan, authored_context = (
                                workflow.plan,
                                workflow.context,
                            )
                        elif not _source_closure_matches(locked, authored_path):
                            _gate(
                                gates,
                                "DECLARATION_IDENTITY_CHANGED",
                                "Campaign experiment source closure differs from its lock",
                            )
                        else:
                            authored_plan = load_plan(authored_path)
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        _gate(gates, "CAMPAIGN_LOCK_UNAVAILABLE", str(error))
                else:
                    _gate(
                        gates,
                        "LOCK_REQUIRED",
                        "Campaign experiment plan stage has not published a lock",
                    )
            else:
                _gate(
                    gates,
                    "CAMPAIGN_LOCK_AMBIGUOUS",
                    "iteration requires exactly one experiment plan stage",
                )
            action = campaign.get("next_action", {})
            result["ingestion"] = [
                {"stage": row["id"], "state": row["state"], "reason": row.get("reason")}
                for row in campaign.get("stages", [])
                if row.get("kind") in {"experiment_run", "experiment_collect"}
            ]
            result["runtime"] = {
                "status": "RECORDED_ACCEPTANCE"
                if any(
                    row.get("kind") == "runtime_acceptance"
                    and row["state"] == "COMPLETE"
                    for row in campaign.get("stages", [])
                )
                else "UNAVAILABLE",
                "stages": [
                    {
                        "id": row["id"],
                        "state": row["state"],
                        "reason": row.get("reason"),
                        "outputs": row.get("outputs", []),
                    }
                    for row in campaign.get("stages", [])
                    if row.get("kind") == "runtime_acceptance"
                ],
                "probe_executed": False,
            }
            for row in campaign.get("stages", []):
                if row.get("state") in {"FAILED", "INCONCLUSIVE", "INTERRUPTED"} or (
                    row.get("state") == "BLOCKED"
                    and row.get("reason") != "prerequisite did not complete"
                ):
                    _gate(
                        gates,
                        "CAMPAIGN_" + row["state"],
                        f"{row['id']}: {row.get('reason')}",
                    )
            if action.get("action") == "approve":
                _gate(gates, "APPROVAL_REQUIRED", str(action.get("reason")))
            runtime_needed = result["runtime"]["status"] == "UNAVAILABLE"
            pilot_needed = True
        elif "lock_version" in document:
            locked = open_lock(source, **options)
            result.update(
                id=locked.id,
                plan_sha256=locked.plan_sha256,
                scientific_sha256=locked.scientific_sha256,
            )
            authored = locked.availability.get("declaration_source")
            if authored is None:
                _gate(
                    gates,
                    "DECLARATION_SOURCE_UNAVAILABLE",
                    "lock has no authored source binding",
                )
            else:
                declaration_path = Path(authored)
                if "workflow_profile" in locked.availability:
                    from klab.experiments.workflow_recipe import (
                        locked_workflow_context,
                    )

                    workflow = locked_workflow_context(locked, declaration_path)
                    authored_plan, authored_context = workflow.plan, workflow.context
                elif not _source_closure_matches(locked, declaration_path):
                    _gate(
                        gates,
                        "DECLARATION_IDENTITY_CHANGED",
                        "authored declaration/input bytes differ from the lock",
                    )
                else:
                    authored_plan = load_plan(declaration_path)
        else:
            plan = load_plan(source)
            result["id"] = plan.id
            result["artifacts"] = {
                "verification_status": "DECLARED_NOT_RESOLVED",
                "inputs": {
                    name: plan.artifacts[reference].model_dump(mode="json")
                    for name, reference in plan.inputs.items()
                },
                "artifacts": {
                    name: artifact.model_dump(mode="json")
                    for name, artifact in plan.artifacts.items()
                },
            }
            preparation = work / "experiments" / plan.id / "preparation.json"
            prepared = (
                json.loads(preparation.read_text()) if preparation.is_file() else None
            )
            try:
                locked = resolve_plan(plan, source, prepared=prepared, **options)
            except (OSError, ValueError, TypeError, KeyError) as error:
                _gate(gates, "INPUT_OR_SOURCE_UNAVAILABLE", str(error))
                _gate(
                    gates,
                    "LOCK_REQUIRED",
                    "resolve and publish an immutable experiment lock",
                )
            else:
                result.update(
                    scientific_sha256=locked.scientific_sha256,
                    proposed_plan_sha256=locked.plan_sha256,
                )
                _gate(
                    gates,
                    "LOCK_REQUIRED",
                    "publish the verified immutable plan before dispatch",
                )
    except (OSError, ValueError, TypeError, KeyError) as error:
        message = str(error)
        code = (
            "SOURCE_INCOMPATIBLE"
            if "source" in message.lower() and "identity" in message.lower()
            else "DECLARATION_OR_LOCK_INVALID"
        )
        _gate(gates, code, message)

    if locked is not None and "workflow_profile" in locked.availability:
        _gate(
            gates,
            "WORKFLOW_EXECUTION_UNAVAILABLE",
            "workflow inventory integrity is not execution admission; this profile cannot execute",
        )
    if locked is not None:
        result["artifacts"] = {
            "verification_status": "VERIFIED",
            "inputs": locked.inputs,
            "artifacts": locked.artifacts,
            "cells": {cell.id: cell.artifacts for cell in locked.cells},
        }
        try:
            result["storage"] = [
                {"cell": cell.id, **_storage(cell.config, work, locked.retention)}
                for cell in locked.cells
            ]
        except (OSError, ValueError, TypeError, KeyError) as error:
            _gate(gates, "STORAGE_UNAVAILABLE", str(error))
        for entry in result["storage"]:
            for check in entry["capacity"]:
                if check["status"] == "insufficient":
                    _gate(
                        gates,
                        "STORAGE_INSUFFICIENT",
                        f"{entry['cell']}: {check['path']}",
                    )
    elif plan is not None:
        try:
            base = base_run_config(plan, source)
            result["storage"] = [
                {
                    "cell": "base",
                    **_storage(base, work, plan.retention.model_dump(mode="json")),
                }
            ]
        except (OSError, ValueError, TypeError, KeyError) as error:
            _gate(gates, "STORAGE_UNAVAILABLE", str(error))
        for entry in result["storage"]:
            for check in entry["capacity"]:
                if check["status"] == "insufficient":
                    _gate(gates, "STORAGE_INSUFFICIENT", check["path"])
    snapshot = None
    if parent is not None:
        try:
            result["parent"], snapshot = _parent(parent, options)
            if not result["parent"]["valid"]:
                _gate(gates, "PARENT_INVALID", str(result["parent"].get("errors")))
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as error:
            _gate(gates, "PARENT_INVALID", str(error))
            result["parent"]["errors"] = [{"field": "parent", "reason": str(error)}]
    else:
        phases = (
            locked.phases
            if locked is not None
            else (plan.phases if plan is not None else ())
        )
        if any(phase.transition != "fresh" for phase in phases):
            _gate(
                gates,
                "PARENT_REQUIRED",
                "continuation requires an explicit exact parent generation",
            )
        elif phases:
            result["parent"]["status"] = "NOT_APPLICABLE"
            if campaign is None:
                authored = plan or authored_plan
                authored_source = (
                    source
                    if plan is not None
                    else Path(locked.availability["declaration_source"])
                    if authored is not None
                    else None
                )
                if authored is not None and authored_source is not None:
                    try:
                        declared, delta, phase = _fresh_delta(
                            authored, authored_source, context=authored_context
                        )
                        result["declared_delta"], result["delta"] = declared, delta
                        result["transition"] = phase.transition
                        if locked is not None:
                            cells = [
                                cell for cell in locked.cells if cell.phase == phase.id
                            ]
                            if len(cells) != 1 or len(locked.cells) != 1:
                                raise ValueError(
                                    "parent-free inspection requires one fresh lock cell"
                                )
                            result["cell"] = cells[0].id
                    except (OSError, ValueError, TypeError, KeyError) as error:
                        _gate(gates, "DELTA_UNRESOLVED", str(error))
                elif locked is not None:
                    _gate(
                        gates,
                        "DELTA_UNRESOLVED",
                        "authored fresh declaration unavailable",
                    )
        elif locked is not None:
            _gate(gates, "DELTA_UNRESOLVED", "lock has no declared phase")
    if snapshot is not None:
        from klab.config.models import RunConfig
        from klab.experiments.source_compatibility import (
            source_identities_compatible,
        )
        from klab.training.manifest import source_identity

        old = RunConfig.model_validate(snapshot.config)
        result["parent"]["terminal"] = (
            snapshot.step == old.training.max_steps
            and snapshot.tokens_seen == old.training.max_tokens
        )
        if not result["parent"]["terminal"]:
            _gate(
                gates,
                "PARENT_NONTERMINAL",
                "checkpoint has not completed its declared token and update caps",
            )
        current = str(source_identity()["sha256"])
        saved = snapshot.source_identity_sha256
        result["parent"]["source_sha256"] = saved
        result["parent"]["current_source_sha256"] = current
        try:
            authenticated_mapping = bool(
                saved and source_identities_compatible(saved, current)
            )
        except (OSError, ValueError) as error:
            authenticated_mapping = False
            _gate(gates, "SOURCE_INCOMPATIBLE", str(error))
        result["parent"]["authenticated_source_mapping"] = authenticated_mapping
        result["parent"]["source_compatible"] = bool(saved and saved == current)
        if not result["parent"]["source_compatible"] and not any(
            gate["code"] == "SOURCE_INCOMPATIBLE" for gate in gates
        ):
            _gate(
                gates,
                "SOURCE_INCOMPATIBLE",
                "full-state continuation requires the exact parent execution source; "
                "a reviewed lock compatibility mapping does not authorize optimizer source drift",
            )
        if plan is not None and locked is None:
            try:
                declared, delta, phase, candidate = _declaration_delta(
                    plan, source, snapshot
                )
                result["declared_delta"], result["delta"] = declared, delta
                result["transition"] = phase.transition
                from klab.training.continuation import (
                    _resume_settings,
                    validate_budget_extension,
                )
                from klab.training.manifest import architecture_sha256

                try:
                    if phase.transition == "extend_budget":
                        validate_budget_extension(
                            old,
                            candidate,
                            snapshot,
                            parent_tokenizer=_parent_tokenizer(parent),
                            parent_root=_parent_tokenizer(parent).parent,
                        )
                        compatible = True
                    elif phase.transition == "resume":
                        compatible = _resume_settings(old) == _resume_settings(
                            candidate
                        )
                    else:
                        compatible = architecture_sha256(
                            old.model_dump(mode="json")
                        ) == architecture_sha256(candidate.model_dump(mode="json"))
                    result["parent"]["compatible"] = compatible
                    if not compatible:
                        _gate(
                            gates,
                            "CONTINUATION_INCOMPATIBLE",
                            "authored child does not preserve the declared transition settings",
                        )
                except (OSError, ValueError, KeyError) as error:
                    result["parent"]["compatible"] = False
                    _gate(gates, "CONTINUATION_INCOMPATIBLE", str(error))
            except (OSError, ValueError, TypeError, KeyError) as error:
                _gate(gates, "DELTA_UNRESOLVED", str(error))
        if locked is not None:
            from klab.training.continuation import _resume_settings
            from klab.training.manifest import architecture_sha256

            matches = []
            for cell in locked.cells:
                phase = next(item for item in locked.phases if item.id == cell.phase)
                if phase.transition == "fresh":
                    continue
                if (
                    phase.checkpoint is not None
                    and locked.artifacts[phase.checkpoint].get("sha256")
                    != snapshot.checkpoint_sha256
                ):
                    continue
                if (
                    phase.checkpoint is not None
                    and Path(
                        locked.availability["artifacts"][phase.checkpoint]
                    ).resolve()
                    != parent.resolve()
                ):
                    continue
                if phase.at_step is not None and phase.at_step != snapshot.step:
                    continue
                matches.append((cell, phase))
            if campaign is not None:
                stage_by_id = {stage.id: stage for stage in declaration.stages}
                selected = stage_by_id.get(campaign.get("next_action", {}).get("stage"))
                if selected is not None and selected.kind == "experiment_run":
                    matches = [pair for pair in matches if pair[0].id == selected.cell]
            if len(matches) != 1:
                _gate(
                    gates,
                    "PARENT_BINDING_REQUIRED",
                    "exact parent must select one declared continuation cell",
                )
            else:
                cell, phase = matches[0]
                result["cell"] = cell.id
                result["transition"] = phase.transition
                # Resolve actual observed delta; the native transition validator below
                # checks its complete permitted field closure, not only the label.
                before = flatten_config(old, include_derived=True)
                after = flatten_config(cell.config, include_derived=True)
                result["delta"] = list(
                    describe_differences(
                        {
                            key: {"base": before.get(key), "variant": after.get(key)}
                            for key in sorted(before.keys() | after.keys())
                            if before.get(key) != after.get(key)
                        }
                    )
                )
                authored = plan or authored_plan
                if authored is not None:
                    declared_phase = next(
                        (item for item in authored.phases if item.id == phase.id), None
                    )
                    if declared_phase is not None:
                        result["declared_delta"] = sorted(declared_phase.set)
                        changed = {entry["path"] for entry in result["delta"]}
                        for field in result["declared_delta"]:
                            if not any(
                                key == field or key.startswith(field + ".")
                                for key in changed
                            ):
                                _gate(
                                    gates,
                                    "DECLARED_DELTA_UNCHANGED",
                                    f"{field} is unchanged from the authenticated parent",
                                )
                compatible = False
                if phase.transition == "resume":
                    compatible = _resume_settings(old) == _resume_settings(cell.config)
                elif phase.transition == "extend_budget":
                    from klab.training.continuation import (
                        validate_budget_extension,
                    )

                    try:
                        validate_budget_extension(
                            old,
                            cell.config,
                            snapshot,
                            parent_tokenizer=_parent_tokenizer(parent),
                            parent_root=_parent_tokenizer(parent).parent,
                        )
                        compatible = True
                    except (OSError, ValueError, KeyError) as error:
                        _gate(gates, "CONTINUATION_INCOMPATIBLE", str(error))
                elif phase.transition == "promote":
                    compatible = architecture_sha256(
                        old.model_dump(mode="json")
                    ) == architecture_sha256(cell.config.model_dump(mode="json"))
                result["parent"]["compatible"] = compatible
                if not compatible and not any(
                    gate["code"] == "CONTINUATION_INCOMPATIBLE" for gate in gates
                ):
                    _gate(
                        gates,
                        "CONTINUATION_INCOMPATIBLE",
                        "actual child config differs from the declared transition's supported compatibility",
                    )
                if campaign is not None and phase.transition in {
                    "resume",
                    "extend_budget",
                }:
                    selected_run = next(
                        (
                            stage
                            for stage in declaration.stages
                            if stage.id == campaign.get("next_action", {}).get("stage")
                            and stage.kind == "experiment_run"
                            and stage.cell == cell.id
                        ),
                        None,
                    )
                    runtime_row = next(
                        (
                            row
                            for row in campaign["stages"]
                            if selected_run is not None
                            and row["id"] == selected_run.runtime
                        ),
                        None,
                    )
                    facts = runtime_row.get("measurements", {}) if runtime_row else {}
                    same_runtime = (
                        old.runtime.model_dump(mode="json")
                        == cell.config.runtime.model_dump(mode="json")
                        == facts.get("runtimes", {}).get(cell.id)
                    )
                    if (
                        compatible
                        and saved == current
                        and result["parent"]["terminal"]
                        and runtime_row is not None
                        and runtime_row["state"] == "COMPLETE"
                        and cell.id in facts.get("cells", [])
                        and same_runtime
                    ):
                        # This exact-source, full-state parent already trained at
                        # the same model shape on the accepted runtime contract.
                        pilot_needed = False
                # A declared external checkpoint does not establish a pilot.
                # Pilot evidence is a separate accepted runtime gate.
                if campaign is None:
                    bindings = (
                        work
                        / "experiments"
                        / locked.id
                        / "runtime-bindings"
                        / locked.plan_sha256
                        / cell.id
                    )
                    accepted = []
                    if bindings.is_dir():
                        from klab.experiments.binding import (
                            inspect_runtime_binding,
                        )

                        for receipt in sorted(bindings.glob("*.json")):
                            try:
                                accepted.append(
                                    inspect_runtime_binding(receipt, locked, cell)
                                )
                            except OSError, ValueError, TypeError, KeyError:
                                continue
                    result["runtime"] = {
                        "status": "RECEIPT_ONLY" if accepted else "UNAVAILABLE",
                        "bindings": [
                            {
                                "path": str(
                                    bindings / (item["binding_sha256"] + ".json")
                                ),
                                "sha256": item["binding_sha256"],
                                "source_kind": item["source_kind"],
                            }
                            for item in accepted
                        ],
                        "probe_executed": False,
                        "fresh_acceptance": False,
                    }
                if phase.parent is not None and campaign is None:
                    import sqlite3

                    from klab.workers.controller import Controller

                    controller_root = work / "experiments" / locked.id / "controller"
                    parent_id = f"{phase.parent}:{cell.id.split(':', 1)[1]}"
                    try:
                        candidates = (
                            Controller(
                                controller_root, read_only=True, **options
                            ).list_experiments()
                            if (controller_root / "experiments.sqlite3").is_file()
                            else []
                        )
                        rows = [
                            row
                            for row in candidates
                            if (row["spec"].get("plan") or {}).get("plan_sha256")
                            == locked.plan_sha256
                            and (row["spec"].get("plan") or {}).get("cell_id")
                            == parent_id
                        ]
                        result["ingestion"] = [
                            {
                                "run_id": row["run_id"],
                                "status": row["status"],
                                "ingestion_status": row["ingestion_status"],
                            }
                            for row in rows
                        ]
                    except (
                        OSError,
                        ValueError,
                        TypeError,
                        KeyError,
                        sqlite3.Error,
                    ) as error:
                        rows = []
                        _gate(gates, "PARENT_INGESTION_UNAVAILABLE", str(error))
                    completed = [
                        row
                        for row in rows
                        if row["status"] == "COMPLETE"
                        and row["ingestion_status"] == "COMPLETE"
                    ]
                    if len(completed) != 1:
                        _gate(
                            gates,
                            "PARENT_INGESTION_PENDING",
                            "one completed, ingested, lock-bound parent is required",
                        )
                    else:
                        _selected_parent_gate(
                            controller_root / completed[0]["run_id"],
                            phase,
                            parent,
                            snapshot.checkpoint_sha256,
                            result["parent"],
                            gates,
                        )
                elif phase.parent is not None and campaign is not None:
                    parent_id = f"{phase.parent}:{cell.id.split(':', 1)[1]}"
                    parent_rows = [
                        row
                        for stage in declaration.stages
                        if stage.kind == "experiment_run" and stage.cell == parent_id
                        for row in campaign["stages"]
                        if row["id"] == stage.id
                    ]
                    if len(parent_rows) != 1 or parent_rows[0]["state"] != "COMPLETE":
                        _gate(
                            gates,
                            "PARENT_INGESTION_PENDING",
                            "Campaign parent run and ingestion not complete",
                        )
                    else:
                        try:
                            selected_run = Path(parent_rows[0]["availability"]["path"])
                            run_id = parent_rows[0]["measurements"]["run_id"]
                        except (KeyError, TypeError, ValueError) as error:
                            _gate(gates, "PARENT_BINDING_REQUIRED", str(error))
                        else:
                            if run_id != selected_run.name:
                                _gate(
                                    gates,
                                    "PARENT_BINDING_MISMATCH",
                                    "Campaign parent run identity differs from its bound path",
                                )
                            else:
                                _selected_parent_gate(
                                    selected_run,
                                    phase,
                                    parent,
                                    snapshot.checkpoint_sha256,
                                    result["parent"],
                                    gates,
                                )
    if pilot_needed:
        _gate(
            gates,
            "PILOT_REQUIRED",
            "no authenticated full-shape pilot acceptance was supplied",
        )
    if runtime_needed:
        _gate(
            gates,
            "RUNTIME_REQUIRED",
            "runtime receipt inspection is not fresh runtime acceptance",
        )
    if not result["runtime"]:
        result["runtime"] = {
            "status": "UNAVAILABLE",
            "bindings": [],
            "probe_executed": False,
        }
    if store is not None:
        result["verification"]["diagnostics"] = store.diagnostics()
    state, command = _classification(
        gates,
        campaign=campaign,
        runtime_needed=runtime_needed,
        locked=locked is not None and plan is None,
        pilot_needed=pilot_needed,
    )
    from shlex import quote

    if campaign is not None:
        if state == "READY_TO_DISPATCH":
            command = f"klab campaign apply {quote(str(source))} --execute-runs"
        elif state == "RUNNING":
            command = f"klab campaign status {quote(str(source))} --json"
        elif state == "NEEDS_APPROVAL":
            command = f"klab campaign explain {quote(str(source))} --json"
        else:
            command = f"klab campaign next {quote(str(source))} --json"
    elif state == "BLOCKED":
        if parent is not None and any(
            gate["code"] == "PARENT_INVALID" for gate in gates
        ):
            command = f"klab checkpoint verify {quote(str(parent))} --json"
        elif locked is not None:
            command = f"klab experiment explain {quote(str(source))} --json"
        elif plan is not None:
            command = f"klab experiment inspect {quote(str(source))} --json"
        else:
            command = f"klab experiment inspect {quote(str(source))} --json"
    elif state == "SAFE_TO_PREPARE":
        command = f"klab experiment lock {quote(str(source))} --json"
    elif state == "NEEDS_PILOT":
        command = (
            f"klab experiment export-config {quote(str(source))} "
            f"--cell {quote(result['cell'])} "
            f"--output {quote(str(source.parent / 'iteration-effective.yaml'))} --json"
            if locked is not None and plan is None and "cell" in result
            else f"klab experiment inspect {quote(str(source))} --json"
        )
    elif state == "READY_TO_DISPATCH":
        command = f"klab experiment run {quote(str(source))} --cell {quote(result['cell'])} --json"
    command = command.replace("klab ", f"klab --work-dir {quote(str(work))} ", 1)
    result["state"] = state
    result["next_command"] = command
    return IterationResult.model_validate(result).model_dump(mode="json")
