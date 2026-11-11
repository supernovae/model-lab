"""Read-only graph inspection and factual family comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from klab.family.manifest import FamilyNode, binding_path, load_family
from klab.training.checkpoints import CheckpointManager
from klab.training.manifest import architecture_sha256, read_manifest
from klab.workdir import resolve_work_dir


def validate_run_bindings(
    run: dict[str, Any], node: FamilyNode, *, local_release: bool
) -> None:
    """Check only identities actually observed in the verified run manifest."""
    tokenizer = [
        item["sha256"]
        for item in run["artifacts"]
        if item["relative_path"] == "tokenizer.json"
    ]
    if tokenizer and (len(tokenizer) != 1 or tokenizer[0] != node.tokenizer.sha256):
        raise ValueError("family tokenizer differs from verified run artifact")
    dataset = run["effective_config"]["dataset"]
    location = dataset.get("corpus_release_path")
    if location is not None:
        if dataset.get("revision") != node.corpus.sha256:
            raise ValueError("family corpus release differs from verified run")
        if local_release and Path(location).is_dir():
            from klab.corpus.release import verify_release

            release = verify_release(Path(location), expected_id=node.corpus.sha256)
            if release["corpus_id"] != node.corpus.id:
                raise ValueError("family corpus ID differs from verified release")
    for decision in run["resource_decisions"]:
        if decision.get("kind") != "worker_dispatch" or decision.get("plan") is None:
            continue
        plan = decision["plan"]
        if plan["plan_id"] != node.plan.id or plan["plan_sha256"] != node.plan.sha256:
            raise ValueError("family plan differs from verified worker dispatch")


def _availability(source: Path, node: FamilyNode) -> dict[str, str]:
    from klab.evaluation.readiness import verify_readiness_result
    from klab.evaluation.suite import verify_evaluation_index

    result: dict[str, str] = {}
    for name, binding in (
        ("checkpoint", node.checkpoint),
        ("evaluation_index", node.evaluation_index),
        ("readiness_result", node.readiness_result),
    ):
        if binding is None:
            result[name] = "NOT_CREATED"
            continue
        path = binding_path(source, binding.path)
        if not path.exists():
            result[name] = (
                "MISSING_NONRECONSTRUCTABLE"
                if name == "checkpoint"
                else "MISSING_EXTERNAL"
            )
            continue
        if name == "checkpoint":
            if path.name in {"latest.json", "best.json"} or path.is_symlink():
                raise ValueError(
                    "family checkpoint must pin a generation, not a mutable pointer"
                )
            manager = CheckpointManager(path.parent.parent)
            report = manager.verify(path)
            if not report.valid:
                raise ValueError(f"invalid checkpoint {path}: {report.errors}")
            from klab.training.mlx_checkpoints import strict_json

            manifest = strict_json(path / "manifest.json")
            if manifest["sha256"] != binding.sha256:
                raise ValueError("checkpoint digest mismatch")
            run = read_manifest(path.parent.parent / "manifest.json")
            config = run["effective_config"]
            training = config["training"]
            if (
                run["architecture_sha256"] != node.architecture_sha256
                or manifest["architecture_sha256"] != node.architecture_sha256
                or architecture_sha256(config) != node.architecture_sha256
                or training["max_steps"] != node.budget.max_steps
                or training["max_tokens"] != node.budget.max_tokens
                or node.objective != "next_token"
            ):
                raise ValueError(
                    "family checkpoint architecture/objective/budget differs from verified run"
                )
            validate_run_bindings(run, node, local_release=True)
            if (
                manifest.get("parent_checkpoint_sha256")
                != node.parent_checkpoint_sha256
                or run.get("checkpoint_sha256") != node.parent_checkpoint_sha256
            ):
                raise ValueError("family checkpoint parent lineage mismatch")
        else:
            verifier = (
                verify_evaluation_index
                if name == "evaluation_index"
                else verify_readiness_result
            )
            record = verifier(path)
            identity = record.get(
                "index_sha256" if name == "evaluation_index" else "result_sha256"
            )
            if identity != binding.sha256:
                raise ValueError(f"{name} identity mismatch")
            if not isinstance(record, dict):
                raise ValueError(f"invalid {name} verification")
            if (
                name == "evaluation_index"
                and node.checkpoint
                and record.get("checkpoint_sha256") != node.checkpoint.sha256
            ):
                raise ValueError("evaluation checkpoint binding mismatch")
            if (
                name == "readiness_result"
                and node.evaluation_index
                and record.get("index_sha256") != node.evaluation_index.sha256
            ):
                raise ValueError("readiness evaluation binding mismatch")
        result[name] = "PRESENT"
    return result


def show(source: Path, *, work_root: Path | None = None) -> dict[str, Any]:
    family = load_family(source)
    identities = family.identities()
    rows = []
    for node in family.nodes:
        lock_path = (
            (work_root or resolve_work_dir(None))
            / "experiments"
            / node.plan.id
            / "locks"
            / f"{node.plan.sha256}.json"
        )
        plan_state = "MISSING_EXTERNAL"
        if lock_path.is_file():
            from klab.experiments.lock import open_lock

            lock = open_lock(lock_path, verification_mode="cold")
            if lock.id != node.plan.id or lock.plan_sha256 != node.plan.sha256:
                raise ValueError(f"family plan lock mismatch: {node.id}")
            candidates = [
                cell
                for cell in lock.cells
                if architecture_sha256(cell.config.model_dump(mode="json"))
                == node.architecture_sha256
                and cell.config.training.max_steps == node.budget.max_steps
                and cell.config.training.max_tokens == node.budget.max_tokens
            ]
            if not candidates:
                raise ValueError(
                    f"family architecture/budget differs from pinned plan: {node.id}"
                )
            plan_state = "PRESENT"
        recovery: dict[str, str] = {}
        if node.recovery_manifest is not None:
            from klab.campaign.plan import safe_path
            from klab.recovery.engine import inspect_manifest

            recipe = safe_path(source.parent, node.recovery_manifest)
            for step in inspect_manifest(recipe, work_root or resolve_work_dir(None))[
                "steps"
            ]:
                kind = step["kind"]
                expected = step["expected_sha256"]
                for label, pinned in (
                    ("corpus", node.corpus.sha256),
                    ("tokenizer", node.tokenizer.sha256),
                    ("plan", node.plan.sha256),
                ):
                    kinds = {
                        "corpus": "corpus_release",
                        "tokenizer": "tokenizer_train",
                        "plan": "experiment_lock",
                    }
                    if kind == kinds[label] and expected == pinned:
                        recovery[label] = step["classification"]
                if kind == "prepared_data":
                    recovery["prepared_data"] = step["classification"]
        for label in ("corpus", "tokenizer", "prepared_data"):
            recovery.setdefault(label, "MISSING_EXTERNAL")
        row: dict[str, Any] = {
            "id": node.id,
            "node_sha256": identities[node.id],
            "parent": node.parent,
            "parent_node_sha256": identities.get(node.parent),
            "checkpoint_sha256": node.checkpoint.sha256 if node.checkpoint else None,
            "availability": {
                **_availability(source, node),
                **recovery,
                "plan": plan_state
                if plan_state == "PRESENT"
                else recovery.get("plan", plan_state),
            },
        }
        rows.append(row)
    return {
        "format": "klab-model-family-v1",
        "family": family.id,
        "graph_valid": True,
        "nodes": rows,
    }


def graph(source: Path) -> dict[str, Any]:
    family = load_family(source)
    return {
        "format": "klab-model-family-v1",
        "family": family.id,
        "nodes": list(family.identities()),
        "edges": [
            {"parent": node.parent, "child": node.id}
            for node in family.nodes
            if node.parent
        ],
    }


def compare(source: Path, first: str, second: str) -> dict[str, Any]:
    family = load_family(source)
    nodes = {node.id: node for node in family.nodes}
    if first not in nodes or second not in nodes:
        raise ValueError("unknown family comparison node")

    def ancestors(name: str) -> list[str]:
        result: list[str] = []
        while name := nodes[name].parent or "":
            result.append(name)
        return result

    def facts(node: FamilyNode) -> dict[str, Any]:
        return {
            "ancestors": ancestors(node.id),
            "corpus": node.corpus.model_dump(),
            "tokenizer": node.tokenizer.model_dump(),
            "plan": node.plan.model_dump(),
            "architecture_sha256": node.architecture_sha256,
            "objective": node.objective,
            "budget": node.budget.model_dump(),
            "evaluation_index_sha256": node.evaluation_index.sha256
            if node.evaluation_index
            else None,
            "checkpoint_sha256": node.checkpoint.sha256 if node.checkpoint else None,
        }

    return {
        "format": "klab-model-family-comparison-v1",
        "family": family.id,
        "first": {"id": first, **facts(nodes[first])},
        "second": {"id": second, **facts(nodes[second])},
    }


def _handle(args: argparse.Namespace) -> None:
    try:
        source = Path(args.source).absolute()
        if args.family_command == "show" or args.family_command == "verify":
            result = show(source, work_root=getattr(args, "work_dir", None))
        elif args.family_command == "graph":
            result = graph(source)
        elif args.family_command == "compare":
            result = compare(source, args.first, args.second)
        else:
            from klab.family.receipts import decide

            result = decide(
                source,
                args.family_command,
                args.node,
                Path(args.readiness),
                Path(args.evaluation),
                Path(args.approval),
                args.note,
                successor=getattr(args, "successor", None),
            )
        if args.json:
            print(json.dumps(result, sort_keys=True, allow_nan=False))
        else:
            print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(
            json.dumps(
                {
                    "format": "klab-model-family-v1",
                    "state": "FAILED",
                    "reason": str(error),
                },
                sort_keys=True,
            )
        )
        raise SystemExit(1) from error


def register_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "family", help="Inspect and review model-family lineage"
    )
    verbs = parser.add_subparsers(dest="family_command", required=True)
    for name in (
        "show",
        "graph",
        "compare",
        "verify",
        "promote",
        "reject",
        "supersede",
    ):
        command = verbs.add_parser(name)
        command.add_argument("source")
        if name == "compare":
            command.add_argument("first")
            command.add_argument("second")
        if name in {"promote", "reject", "supersede"}:
            command.add_argument("node")
            command.add_argument("--readiness", required=True)
            command.add_argument("--evaluation", required=True)
            command.add_argument("--approval", required=True)
            command.add_argument("--note", required=True)
            if name == "supersede":
                command.add_argument("--successor", required=True)
        command.add_argument("--json", action="store_true")
        command.set_defaults(handler=_handle)
