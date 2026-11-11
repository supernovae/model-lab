"""Fresh execution preflight without creating storage or modifying Git."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from klab.documents import read_document
from klab.experiments.plan import base_run_config, load_plan
from klab.recovery.provenance import (
    declaration_paths,
    declaration_reference,
    git_provenance,
)
from klab.runtime_profile import RuntimeProfile
from klab.workdir import resolve_work_dir, storage_checks
from klab.workspace_preflight import training_storage_checks


def snapshot(
    source: Path, work_dir: Path | None = None, *, profile: RuntimeProfile | None = None
) -> dict[str, Any]:
    source = Path(source).absolute()
    root = resolve_work_dir(work_dir)
    raw = read_document(source)
    kinds = [
        (key, kind)
        for key, kind in (
            ("plan_version", "experiment"),
            ("campaign_version", "campaign"),
            ("recovery_version", "recovery"),
        )
        if key in raw
    ]
    if len(kinds) != 1:
        raise ValueError("UNKNOWN_DECLARATION: require exactly one strict version key")
    _, kind = kinds[0]
    provenance = git_provenance(declaration_paths(source, kind))
    reasons: list[str] = []
    if provenance["status"] != "CLEAN_AND_COMMITTED":
        reasons.append(provenance["status"])
    inputs: list[dict[str, Any]] = []
    recovery: dict[str, Any] | None = None
    runtime: dict[str, Any] | None = None
    storage: Any = {"status": "UNAVAILABLE"}
    evaluation: Any = None
    warnings = storage_checks(root)

    def inspect_plan(plan_source: Path) -> None:
        nonlocal runtime, storage, evaluation
        from klab.experiments.artifacts import verify_artifact
        from klab.experiments.lock import _runtime, resolve_plan
        from klab.runtime_profile import authorize_profile

        plan = load_plan(plan_source)
        try:
            config = base_run_config(plan, plan_source)
        except (OSError, ValueError) as error:
            reasons.append("CONFIG_UNAVAILABLE")
            inputs.append(
                {
                    "id": plan.id,
                    "kind": "base_run",
                    "classification": "MISSING_EXTERNAL",
                    "reason": str(error),
                }
            )
            return
        # The base configuration is only a fallback when inputs cannot be resolved:
        # axes and phases can select an entirely different valid runtime.
        try:
            base_runtime, _ = _runtime(config, plan.execution.backend)
            base_error = None
        except ValueError as error:
            base_runtime = None
            base_error = error
        if plan.evaluation_suite:
            from klab.evaluation.suite import load_suite

            suite_source = declaration_reference(plan_source, plan.evaluation_suite)
            try:
                load_suite(suite_source)
                evaluation = {"suite": plan.evaluation_suite}
            except (OSError, ValueError) as error:
                reasons.append("EVALUATION_UNAVAILABLE")
                inputs.append(
                    {
                        "id": plan.evaluation_suite,
                        "kind": "evaluation_suite",
                        "classification": "MISSING_EXTERNAL",
                        "reason": str(error),
                    }
                )
        elif plan.evaluations:
            evaluation = [item.model_dump(mode="json") for item in plan.evaluations]
        else:
            reasons.append("EVALUATION_NOT_DECLARED")
        storage = [
            asdict(check)
            for check in training_storage_checks(
                config, work_dir=root, run_dir=root / "runs"
            )
        ]
        if any(check["status"] != "adequate" for check in storage):
            reasons.append("STORAGE_INADEQUATE")
        warnings.extend(storage_checks(config.dataset.cache_dir, kind="dataset_cache"))
        warnings.extend(storage_checks(config.logging.root_dir, kind="logging_root"))
        for name, artifact in plan.artifacts.items():
            if artifact.from_phase is not None:
                continue
            item = {
                "id": name,
                "kind": artifact.kind,
                "expected_sha256": artifact.sha256,
            }
            try:
                verified = verify_artifact(artifact, plan_source)
                item.update(classification="PRESENT", actual_sha256=verified["sha256"])
            except (OSError, ValueError) as error:
                item.update(classification="MISSING_EXTERNAL", reason=str(error))
                reasons.append("REQUIRED_ARTIFACT_UNVERIFIED")
            inputs.append(item)
        preparation = root / "experiments" / plan.id / "preparation.json"
        try:
            prepared = read_document(preparation) if plan.corpus_variants else None
            resolved = resolve_plan(plan, plan_source, prepared=prepared)
            inputs.append(
                {
                    "id": plan.id,
                    "kind": "experiment_lock",
                    "classification": "PRESENT",
                    "actual_sha256": resolved.plan_sha256,
                    "verification_scope": "resolved inputs; publication not required",
                }
            )
            cell_requirements = [
                {
                    "plan": plan.id,
                    "cell_id": cell.id,
                    "engine": cell.config.runtime.engine,
                    "backend": cell.config.runtime.backend,
                    "device_index": cell.config.runtime.device_index,
                    "precision": cell.config.runtime.precision,
                    "requirements": {},
                }
                for cell in resolved.cells
            ]
            if runtime is not None and "cells" in runtime:
                cell_requirements = [*runtime["cells"], *cell_requirements]
            elif runtime is not None:
                cell_requirements = [runtime, *cell_requirements]
            if len(cell_requirements) == 1:
                runtime = {
                    key: value
                    for key, value in cell_requirements[0].items()
                    if key not in {"plan", "cell_id"}
                }
            else:
                runtime = {"cells": cell_requirements}
            if profile is None and any(
                item["backend"] != "cpu" for item in cell_requirements
            ):
                reasons.append("RUNTIME_REQUIRED")
            if profile is not None:
                for cell in resolved.cells:
                    try:
                        authorize_profile(profile, cell.config)
                    except (ValueError, RuntimeError) as error:
                        reasons.append("RUNTIME_REQUIRED")
                        inputs.append(
                            {
                                "id": cell.id,
                                "kind": "runtime",
                                "classification": "MISSING_EXTERNAL",
                                "reason": str(error),
                            }
                        )
        except (OSError, ValueError, TypeError, KeyError) as error:
            reasons.append("REQUIRED_ARTIFACT_UNVERIFIED")
            inputs.append(
                {
                    "id": plan.id,
                    "kind": "experiment_lock",
                    "classification": "MISSING_RECONSTRUCTABLE",
                    "reason": str(error),
                }
            )
            if base_error is not None:
                reasons.append(
                    "RUNTIME_REQUIREMENT_UNDEFINED"
                    if config.runtime.backend == "auto"
                    and plan.execution.backend is None
                    else "RUNTIME_REQUIREMENT_UNSUPPORTED"
                )
                inputs.append(
                    {
                        "id": plan.id,
                        "kind": "runtime",
                        "classification": "MISSING_EXTERNAL",
                        "reason": str(base_error),
                    }
                )
            elif base_runtime is not None and runtime is None:
                runtime = {
                    "engine": base_runtime.runtime.engine,
                    "backend": base_runtime.runtime.backend,
                    "device_index": base_runtime.runtime.device_index,
                    "precision": base_runtime.runtime.precision,
                    "requirements": {},
                }
            if profile is None and config.runtime.backend not in {"cpu", "auto"}:
                reasons.append("RUNTIME_REQUIRED")

    if kind == "experiment":
        inspect_plan(source)
    elif kind == "campaign":
        from klab.campaign.engine import CampaignEngine

        engine = CampaignEngine(source, root)
        fresh = engine.inspect("plan")
        inputs = fresh["stages"]
        if engine.plan.recovery:
            from klab.recovery.engine import inspect_manifest
            from klab.recovery.manifest import load_manifest

            manifest_source = declaration_reference(source, engine.plan.recovery)
            manifest = load_manifest(manifest_source)
            recovery = inspect_manifest(manifest_source, engine.store.root)
            runtime = (
                manifest.runtime_requirement.model_dump(mode="json")
                if manifest.runtime_requirement
                else None
            )
        for stage in engine.plan.stages:
            if stage.kind == "experiment_plan":
                inspect_plan(declaration_reference(source, stage.source))
            if stage.kind == "evaluation":
                evaluation = {"suite": stage.suite}
        if any(
            row["state"] != "COMPLETE"
            for row in fresh["stages"]
            if row["kind"]
            in {
                "artifact_reference",
                "tokenizer_reference",
                "corpus_release",
                "experiment_plan",
                "runtime_acceptance",
            }
        ):
            reasons.append("REQUIRED_ARTIFACT_UNVERIFIED")
    else:
        from klab.recovery.engine import inspect_manifest
        from klab.recovery.manifest import load_manifest

        manifest = load_manifest(source)
        recovery = inspect_manifest(source, root)
        runtime = (
            manifest.runtime_requirement.model_dump(mode="json")
            if manifest.runtime_requirement
            else None
        )
        evaluation = manifest.evaluation_suite
        for step in manifest.steps:
            if step.kind == "experiment_lock":
                inspect_plan(declaration_reference(source, step.plan))
        rows = recovery.get("steps", recovery.get("artifacts", []))
        if any(
            row.get("classification") not in {"PRESENT", "NOT_CREATED"}
            and row.get("kind") != "optional_cache"
            for row in rows
        ):
            reasons.append("REQUIRED_ARTIFACT_UNVERIFIED")
    if runtime is None:
        reasons.append("RUNTIME_REQUIREMENT_UNDEFINED")
    if not evaluation:
        reasons.append("EVALUATION_NOT_DECLARED")
    if isinstance(storage, dict):
        reasons.append("STORAGE_UNAVAILABLE")
    return {
        "format": "klab-research-snapshot-v1",
        "state": "BLOCKED" if reasons else "READY_FOR_EXECUTION",
        "reason_codes": list(dict.fromkeys(reasons)),
        "source_commit": provenance["source_commit"],
        "provenance": provenance,
        "expected_inputs": inputs,
        "recoverability": recovery,
        "runtime_requirement": runtime,
        "evaluation": evaluation,
        "storage": storage,
        "storage_checks": warnings,
        "work_root": str(root),
    }


def _handle(args: argparse.Namespace) -> None:
    try:
        from klab.runtime_profile import load_runtime_profile

        profile = getattr(args, "runtime_profile_loaded", None)
        if profile is None and args.runtime_profile is not None:
            profile = load_runtime_profile(args.runtime_profile)
        result = snapshot(Path(args.source), args.work_dir, profile=profile)
    except (OSError, ValueError, TypeError, KeyError) as error:
        result = {
            "format": "klab-research-snapshot-v1",
            "state": "BLOCKED",
            "reason_codes": ["INVALID_DECLARATION"],
            "reason": str(error),
        }
    print(
        json.dumps(result, sort_keys=True)
        if args.json
        else json.dumps(result, indent=2, sort_keys=True)
    )
    if result["state"] == "BLOCKED":
        raise SystemExit(1)


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    command = subparsers.add_parser(
        "snapshot", help="Read-only committed declaration and fresh execution preflight"
    )
    command.add_argument("source", type=Path)
    runtime_source = command.add_mutually_exclusive_group()
    runtime_source.add_argument("--runtime-profile", type=Path)
    runtime_source.add_argument("--runtime", metavar="ID")
    command.add_argument("--json", action="store_true")
    command.set_defaults(handler=_handle)
