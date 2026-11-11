"""Declarative architecture studies over existing concrete run configurations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TypeGuard

import yaml

from klab.canonical import canonical_digest, canonical_json, reject_duplicate_keys
from klab.evaluation.capabilities import (
    CapabilityCard,
    capability_card,
    compare_results,
    evaluate_capability,
    write_capability_result,
)
from klab.evaluation.inference import load_run, write_inference_result
from klab.experiments.compiler import is_dotted_path
from klab.experiments.matrix import ExpandedExperiment, expand
from klab.model.inspection import inspection_report, parameter_inventory
from klab.result_publication import sync_directory as _sync_directory
from klab.result_publication import write_bytes_exclusive
from klab.training.manifest import config_sha256


def study_workspace(name: str) -> Path:
    """Resolve one execution workspace for all coordinates of a named study."""
    from klab.workdir import resolve_work_dir

    identifier = re.sub(r"[^A-Za-z0-9_-]+", "-", name).strip("-")
    if identifier != name or not identifier:
        identifier = (
            f"{identifier or 'study'}-{hashlib.sha256(name.encode()).hexdigest()[:12]}"
        )
    return resolve_work_dir() / "experiments" / identifier


def study_execution_paths(
    study: ArchitectureStudy, *, store: Path | None = None, receipt: Path | None = None
) -> tuple[Path, Path]:
    """Honor explicit paths; otherwise keep runs and receipt in one workspace."""
    if store is None:
        store = (receipt.parent if receipt else study_workspace(study.name)) / "runs"
    if receipt is None:
        receipt = store.parent / "receipt.json"
    return store, receipt


class _StudyLoader(yaml.SafeLoader):
    pass


def _unique_mapping(
    loader: _StudyLoader, node: yaml.MappingNode, deep: bool = False
) -> dict[object, object]:
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in result:
            raise ValueError(f"duplicate study YAML key: {key!r}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_StudyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _unique_mapping
)


@dataclass(frozen=True)
class StudyCard:
    reference: str
    card: CapabilityCard


@dataclass(frozen=True)
class StudyComparison:
    identifier: str
    vary: str
    baseline: dict[str, str]
    variant: dict[str, str]
    vary_fields: tuple[str, ...] | None = None


@dataclass(frozen=True)
class StudyPair:
    comparison: StudyComparison
    baseline_index: int
    variant_index: int
    context: dict[str, str]
    differences: dict[str, dict[str, object]]


@dataclass(frozen=True)
class ArchitectureStudy:
    path: Path
    name: str
    matrix_path: Path
    matrix_sha256: str
    cards: tuple[StudyCard, ...]
    expanded: tuple[ExpandedExperiment, ...]
    comparisons: tuple[StudyComparison, ...]
    pairs: tuple[StudyPair, ...]
    study_sha256: str


def _mapping(value: object, description: str) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{description} must be a string-keyed mapping")
    return value


def _selector(value: object, description: str) -> dict[str, str]:
    raw = _mapping(value, description)
    selector: dict[str, str] = {}
    for key, label in raw.items():
        if not key.strip() or not isinstance(label, str) or not label.strip():
            raise ValueError(f"{description} must contain nonempty axis labels")
        selector[key] = label
    if not selector:
        raise ValueError(f"{description} must select at least one axis")
    return selector


def _study_comparisons(value: object) -> tuple[StudyComparison, ...]:
    if not isinstance(value, list):
        raise TypeError("comparisons must be a list")
    comparisons: list[StudyComparison] = []
    identifiers: set[str] = set()
    allowed_vary = {"memory", "attention", "ffn", "scale", "none", "custom"}
    for raw_item in value:
        item = _mapping(raw_item, "comparison")
        allowed_keys = {"id", "vary", "baseline", "variant", "vary_fields"}
        if set(item) - allowed_keys or not {"id", "vary", "baseline", "variant"} <= set(
            item
        ):
            raise ValueError("comparison requires id, vary, baseline, and variant")
        identifier = item["id"]
        vary = item["vary"]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError("comparison id must be a nonempty string")
        if identifier in identifiers:
            raise ValueError(f"duplicate comparison id: {identifier}")
        identifiers.add(identifier)
        if not isinstance(vary, str) or vary not in allowed_vary:
            raise ValueError(f"comparison {identifier} has unsupported vary axis")
        baseline = _selector(item["baseline"], f"comparison {identifier} baseline")
        variant = _selector(item["variant"], f"comparison {identifier} variant")
        if set(baseline) != set(variant) or baseline == variant:
            raise ValueError(
                f"comparison {identifier} selectors must use the same axes and differ"
            )
        raw_fields = item.get("vary_fields")
        if vary == "custom":
            if not isinstance(raw_fields, list) or not raw_fields:
                raise ValueError(
                    f"comparison {identifier} custom vary requires vary_fields"
                )
            if any(not is_dotted_path(field) for field in raw_fields):
                raise ValueError(
                    f"comparison {identifier} vary_fields must be dotted paths"
                )
            if len(raw_fields) != len(set(raw_fields)):
                raise ValueError(f"comparison {identifier} vary_fields must be unique")
            vary_fields = tuple(raw_fields)
        else:
            if raw_fields is not None:
                raise ValueError(
                    f"comparison {identifier} only custom vary accepts vary_fields"
                )
            vary_fields = None
        comparisons.append(
            StudyComparison(identifier, vary, baseline, variant, vary_fields)
        )
    return tuple(comparisons)


def _resolve_card(reference: str, study_dir: Path) -> StudyCard:
    candidate = Path(reference)
    if not candidate.is_absolute():
        local = study_dir / candidate
        if local.is_file():
            reference = str(local.resolve())
    elif candidate.is_file():
        reference = str(candidate.resolve())
    return StudyCard(reference, capability_card(reference))


def _probe_result(config: Any) -> dict[str, object]:
    inventory = inspection_report(config)
    return {
        "format": "capability_result_v2",
        "card": "architecture-study-plan-probe",
        "card_digest": "0" * 64,
        "evaluation_source_sha256": "1" * 64,
        "valid": True,
        "score": 0.0,
        "results": [],
        "identity": {
            "checkpoint_sha256": "2" * 64,
            "step": 1,
            "tokens_seen": 1,
            "config": config.model_dump(mode="json"),
            "tokenizer_sha256": "3" * 64,
            "data_sha256": {"train": "4" * 64, "validation": "5" * 64},
            "source_identity_sha256": "6" * 64,
            "runtime": {"engine": "study-plan"},
            "parameter_inventory": inventory,
        },
    }


def _pairs(
    expanded: tuple[ExpandedExperiment, ...],
    comparisons: tuple[StudyComparison, ...],
) -> tuple[StudyPair, ...]:
    if not expanded:
        raise ValueError("study matrix has no runs")
    axes = set(expanded[0].coordinate)
    output: list[StudyPair] = []
    for comparison in comparisons:
        selected_axes = set(comparison.baseline)
        unknown = selected_axes - axes
        if unknown:
            raise ValueError(
                f"comparison {comparison.identifier} selects unknown axes: "
                f"{', '.join(sorted(unknown))}"
            )
        available = {
            axis: {item.coordinate[axis] for item in expanded} for axis in axes
        }
        for selector_name, selector in (
            ("baseline", comparison.baseline),
            ("variant", comparison.variant),
        ):
            for axis, label in selector.items():
                if label not in available[axis]:
                    raise ValueError(
                        f"comparison {comparison.identifier} {selector_name} "
                        f"uses unknown label {axis}={label}"
                    )

        def select(
            selector: dict[str, str],
            selected_axes: set[str] = selected_axes,
        ) -> dict[tuple[tuple[str, str], ...], list[int]]:
            groups: dict[tuple[tuple[str, str], ...], list[int]] = defaultdict(list)
            for index, item in enumerate(expanded):
                if all(
                    item.coordinate[axis] == label for axis, label in selector.items()
                ):
                    context = tuple(
                        (axis, item.coordinate[axis])
                        for axis in sorted(axes - selected_axes)
                    )
                    groups[context].append(index)
            return groups

        baseline_groups = select(comparison.baseline)
        variant_groups = select(comparison.variant)
        if not baseline_groups or not variant_groups:
            raise ValueError(f"comparison {comparison.identifier} selects no runs")
        if set(baseline_groups) != set(variant_groups):
            raise ValueError(
                f"comparison {comparison.identifier} has unmatched study coordinates"
            )
        for context in sorted(baseline_groups):
            base_indices = baseline_groups[context]
            variant_indices = variant_groups[context]
            if len(base_indices) != 1 or len(variant_indices) != 1:
                raise ValueError(
                    f"comparison {comparison.identifier} is ambiguous at "
                    f"{dict(context)}; selectors must identify one run per side"
                )
            base_index, variant_index = base_indices[0], variant_indices[0]
            try:
                result = compare_results(
                    _probe_result(expanded[base_index].config),
                    _probe_result(expanded[variant_index].config),
                    vary=comparison.vary,
                    vary_fields=comparison.vary_fields,
                )
            except ValueError as error:
                raise ValueError(
                    f"comparison {comparison.identifier} is not controlled at "
                    f"{dict(context)}: {error}"
                ) from error
            output.append(
                StudyPair(
                    comparison,
                    base_index,
                    variant_index,
                    dict(context),
                    result["differences"],
                )
            )
    return tuple(output)


def plan_study(path: Path, *, max_runs: int = 1000) -> ArchitectureStudy:
    source = path.resolve()
    try:
        raw_value = yaml.load(source.read_text(encoding="utf-8"), Loader=_StudyLoader)
    except OSError as error:
        raise ValueError(f"cannot read architecture study {source}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"invalid architecture study YAML: {error}") from error
    raw = _mapping(raw_value, "architecture study")
    if set(raw) != {"study_version", "name", "matrix", "cards", "comparisons"}:
        raise ValueError(
            "architecture study requires exactly study_version, name, matrix, cards, comparisons"
        )
    if type(raw["study_version"]) is not int or raw["study_version"] != 1:
        raise ValueError("study_version must be integer 1")
    name = raw["name"]
    if not isinstance(name, str) or not name.strip():
        raise ValueError("study name must be a nonempty string")
    matrix_value = raw["matrix"]
    if not isinstance(matrix_value, str) or not matrix_value.strip():
        raise ValueError("study matrix must be a nonempty path")
    matrix_path = Path(matrix_value)
    if not matrix_path.is_absolute():
        matrix_path = source.parent / matrix_path
    matrix_path = matrix_path.resolve()
    references = raw["cards"]
    if (
        not isinstance(references, list)
        or not references
        or any(
            not isinstance(reference, str) or not reference.strip()
            for reference in references
        )
    ):
        raise ValueError("cards must be a nonempty list of card names or paths")
    if len(references) != len(set(references)):
        raise ValueError("study card references must be unique")
    cards = tuple(_resolve_card(reference, source.parent) for reference in references)
    comparisons = _study_comparisons(raw["comparisons"])
    expanded = tuple(expand(matrix_path, max_runs=max_runs))
    pairs = _pairs(expanded, comparisons)
    matrix_sha256 = expanded[0].matrix_sha256
    material = {
        "study_version": 1,
        "name": name,
        "matrix_sha256": matrix_sha256,
        "cards": [
            {"name": item.card.name, "digest": item.card.digest} for item in cards
        ],
        "comparisons": [asdict(item) for item in comparisons],
        "runs": [
            {
                "coordinate": item.coordinate,
                "config_sha256": config_sha256(item.config.model_dump(mode="json")),
            }
            for item in expanded
        ],
    }
    digest = canonical_digest(material)
    return ArchitectureStudy(
        source,
        name,
        matrix_path,
        matrix_sha256,
        cards,
        expanded,
        comparisons,
        pairs,
        digest,
    )


def study_plan_payload(study: ArchitectureStudy) -> dict[str, object]:
    runs = []
    for item in study.expanded:
        estimates = inspection_report(item.config)
        runs.append(
            {
                "coordinate": item.coordinate,
                "config_sha256": config_sha256(item.config.model_dump(mode="json")),
                "parameter_inventory": asdict(parameter_inventory(item.config)),
                "estimated_bytes": {
                    key: estimates[key]
                    for key in (
                        "model_weight_bytes",
                        "optimizer_state_bytes",
                        "estimated_checkpoint_bytes",
                    )
                },
            }
        )
    grouped: list[dict[str, object]] = []
    for comparison in study.comparisons:
        selected = [item for item in study.pairs if item.comparison == comparison]
        grouped.append(
            {
                "id": comparison.identifier,
                "vary": comparison.vary,
                "vary_fields": list(comparison.vary_fields or ()),
                "baseline": comparison.baseline,
                "variant": comparison.variant,
                "pairs": [
                    {
                        "context": pair.context,
                        "baseline": study.expanded[pair.baseline_index].coordinate,
                        "variant": study.expanded[pair.variant_index].coordinate,
                        "differences": pair.differences,
                    }
                    for pair in selected
                ],
            }
        )
    return {
        "format": "klab-architecture-study-plan",
        "study_version": 1,
        "name": study.name,
        "study_sha256": study.study_sha256,
        "matrix_sha256": study.matrix_sha256,
        "cards": [
            {
                "reference": card.reference,
                "name": card.card.name,
                "digest": card.card.digest,
                "hypothesis": card.card.hypothesis,
                "limitations": card.card.limitations,
            }
            for card in study.cards
        ],
        "runs": runs,
        "comparisons": grouped,
        "interpretation": "Plan only; no training, evaluation, or performance claim has run.",
    }


def _write_json_exclusive(path: Path, payload: dict[str, object]) -> None:
    write_bytes_exclusive(
        path, canonical_json(payload) + b"\n", sync_directory=_sync_directory
    )


def study_intent_path(receipt_path: Path) -> Path:
    return receipt_path.with_name(receipt_path.name + ".intent.json")


def _study_receipt_template(study: ArchitectureStudy) -> dict[str, Any]:
    return {
        "format": "klab-architecture-study-receipt",
        "schema_version": 1,
        "study_sha256": study.study_sha256,
        "matrix_sha256": study.matrix_sha256,
        "cards": [
            {
                "reference": item.reference,
                "name": item.card.name,
                "digest": item.card.digest,
            }
            for item in study.cards
        ],
        "runs": [
            {
                "coordinate": item.coordinate,
                "config_sha256": config_sha256(item.config.model_dump(mode="json")),
            }
            for item in study.expanded
        ],
    }


def _submission_binding(intent: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "submission_id": intent["submission_id"],
        "intent_sha256": canonical_digest(intent),
        "study_sha256": intent["receipt"]["study_sha256"],
        "matrix_sha256": intent["receipt"]["matrix_sha256"],
        "cards_sha256": hashlib.sha256(
            canonical_json(intent["receipt"]["cards"])
        ).hexdigest(),
        "index": index,
        **intent["receipt"]["runs"][index],
    }


def submit_study(
    study: ArchitectureStudy,
    receipt_path: Path,
    *,
    store: Path,
    worker: str | None = None,
    stage_bundle: Path | None = None,
) -> dict[str, object]:
    if os.path.lexists(receipt_path):
        raise FileExistsError(f"study receipt already exists: {receipt_path}")
    from klab.workers.controller import Controller

    intent = {
        "format": "klab-study-submission-intent",
        "schema_version": 1,
        "submission_id": uuid.uuid4().hex,
        "store": str(store.resolve()),
        "receipt_path": str(receipt_path.absolute()),
        "receipt": _study_receipt_template(study),
    }
    # Exclusive, durable reservation precedes any bundle preparation/enqueue.
    intent_path = study_intent_path(receipt_path)
    try:
        _write_json_exclusive(intent_path, intent)
    except FileExistsError as error:
        raise FileExistsError(
            f"study submission intent already exists: {intent_path}. Preserve it. "
            "Use study recover if native jobs were enqueued. If the original "
            "submitter has finished and preparation failed before enqueue, fix "
            "the cause and submit with a fresh --receipt in the same directory "
            f"and the same --store {store}. An empty observation alone does not "
            "authorize a retry while another submitter may still be active."
        ) from error
    requests = [
        {
            "config": item.config,
            "worker": worker,
            "stage_bundle": stage_bundle,
            "declaration_provenance": {
                "study_submission": _submission_binding(intent, index)
            },
            "requirements": item.requirements,
            "preferred": item.preferred_worker,
            "matrix": {
                "matrix_sha256": item.matrix_sha256,
                "coordinate": item.coordinate,
            },
        }
        for index, item in enumerate(study.expanded)
    ]
    try:
        submissions = Controller(store).submit_many(requests)
    except BaseException as error:
        error.add_note(
            f"Study intent retained at {intent_path}; enqueue outcome is not "
            "inferred from this exception. Inspect with study recover. If the "
            "original submitter has finished and preparation failed before "
            "enqueue, fix the cause and use a fresh --receipt in the same "
            f"directory with the same --store {store}. Preserve all old evidence."
        )
        raise
    run_entries: list[dict[str, object]] = [
        {
            "coordinate": item.coordinate,
            "config_sha256": config_sha256(item.config.model_dump(mode="json")),
            "experiment_id": submission.experiment_id,
            "attempt_id": submission.attempt_id,
            "run_id": submission.run_id,
        }
        for item, submission in zip(study.expanded, submissions, strict=True)
    ]
    receipt = {**intent["receipt"], "runs": run_entries}
    try:
        _write_json_exclusive(receipt_path, receipt)
    except Exception as error:
        run_ids = ", ".join(str(item["run_id"]) for item in run_entries)
        raise OSError(
            f"study runs were submitted ({run_ids}) but receipt could not be written: {error}"
        ) from error
    return receipt


def recover_study_receipt(
    study: ArchitectureStudy, intent_path: Path, output: Path, *, store: Path
) -> dict[str, object]:
    """Read native evidence only; never instantiate a controller or worker."""
    from klab.config.models import RunConfig
    from klab.workers.models import StudySubmissionMetadata, Submission
    from klab.workers.store import ControllerStore

    if os.path.lexists(output):
        raise FileExistsError(f"recovery output already exists: {output}")
    intent_bytes = intent_path.read_bytes()
    intent = _mapping(
        json.loads(intent_bytes, object_pairs_hook=reject_duplicate_keys),
        "study intent",
    )
    if (
        set(intent)
        != {
            "format",
            "schema_version",
            "submission_id",
            "store",
            "receipt_path",
            "receipt",
        }
        or intent["format"] != "klab-study-submission-intent"
        or type(intent["schema_version"]) is not int
        or intent["schema_version"] != 1
        or intent["store"] != str(store.resolve())
        or intent["receipt"] != _study_receipt_template(study)
    ):
        raise ValueError("study intent identity mismatch")
    if (
        output.resolve() == Path(str(intent["receipt_path"])).resolve()
        or output.resolve() == intent_path.resolve()
    ):
        raise ValueError("recovery requires a distinct unused output")
    expected = [
        _submission_binding(intent, index) for index in range(len(study.expanded))
    ]
    for binding in expected:
        StudySubmissionMetadata.model_validate(binding)
    queue = ControllerStore(store, read_only=True)
    rows = [
        {
            **record,
            "spec": _mapping(
                json.loads(
                    record["specification_json"],
                    object_pairs_hook=reject_duplicate_keys,
                ),
                "native specification",
            ),
        }
        for record in queue.raw_attempt_specifications()
    ]
    matches: dict[int, dict[str, Any]] = {}
    unbound_study_records = False
    for row in rows:
        spec = row["spec"]
        provenance = _mapping(
            spec.get("declaration_provenance") or {}, "native provenance"
        )
        binding = provenance.get("study_submission")
        if binding is None and any(
            spec.get("matrix")
            == {"matrix_sha256": study.matrix_sha256, "coordinate": item.coordinate}
            and spec.get("config_sha256") == expected[index]["config_sha256"]
            for index, item in enumerate(study.expanded)
        ):
            unbound_study_records = True
        if (
            not isinstance(binding, dict)
            or binding.get("submission_id") != intent["submission_id"]
        ):
            continue
        validated = StudySubmissionMetadata.model_validate(binding)
        index = validated.index
        if index >= len(expected) or binding != expected[index] or index in matches:
            raise ValueError("ambiguous or mismatched study submission identity")
        config = RunConfig.model_validate(spec["config"])
        if (
            spec.get("experiment_id") != row["experiment_id"]
            or spec.get("matrix")
            != {
                "matrix_sha256": study.matrix_sha256,
                "coordinate": binding["coordinate"],
            }
            or spec.get("config_sha256") != binding["config_sha256"]
            or config_sha256(config.model_dump(mode="json")) != binding["config_sha256"]
        ):
            raise ValueError("native study coordinate/config identity mismatch")
        matches[index] = row
    if not matches:
        detail = (
            " Legacy/unbound study-like records exist but cannot prove this intent's identity."
            if unbound_study_records
            else ""
        )
        raise ValueError(
            "no native submissions are bound to this intent in the observed store."
            + detail
            + " This observation does not prove that a live submitter has finished. "
            "Preserve the intent; if preparation failed before enqueue and the "
            "original submitter has finished, fix the cause and use a fresh "
            "--receipt in the same directory with the same --store."
        )
    if len(matches) != len(expected):
        raise ValueError(
            f"partial study submission evidence: {len(matches)} of {len(expected)} "
            "coordinates are bound; missing or legacy/unbound identities cannot be recovered"
        )
    runs = []
    for index, item in enumerate(intent["receipt"]["runs"]):
        row = matches[index]
        identities = {
            key: row[key] for key in ("experiment_id", "attempt_id", "run_id")
        }
        Submission.model_validate(identities)
        runs.append({**item, **identities})
    for key in ("experiment_id", "attempt_id", "run_id"):
        if len({item[key] for item in runs}) != len(runs):
            raise ValueError("duplicate native submission identity")
    original = Path(str(intent["receipt_path"]))
    receipt = {
        **intent["receipt"],
        "runs": runs,
        "recovery": {
            "submission_id": intent["submission_id"],
            "intent_path": str(intent_path.absolute()),
            "intent_sha256": canonical_digest(intent),
            "intent_file_sha256": hashlib.sha256(intent_bytes).hexdigest(),
            "original_receipt_path": str(original),
            "original_receipt_sha256": hashlib.sha256(original.read_bytes()).hexdigest()
            if original.is_file()
            else None,
            "store": str(store.resolve()),
        },
    }
    _write_json_exclusive(output, receipt)
    return receipt


def _read_receipt(path: Path) -> tuple[dict[str, object], str]:
    data = path.read_bytes()
    value = json.loads(data, object_pairs_hook=reject_duplicate_keys)
    receipt = _mapping(value, "study receipt")
    if receipt.get("format") != "klab-architecture-study-receipt":
        raise ValueError("unknown architecture study receipt format")
    if type(receipt.get("schema_version")) is not int or receipt["schema_version"] != 1:
        raise ValueError("unsupported architecture study receipt version")
    return receipt, hashlib.sha256(data).hexdigest()


def _path_in_run(run: Path, value: Path) -> str:
    return str(value.resolve().relative_to(run.resolve()))


def _checkpoint_paths(run: Path, selected: str | None) -> list[tuple[str, int | None]]:
    """Return every immutable generation in chronological order.

    An explicit checkpoint remains an explicit single-checkpoint collection.
    """
    if selected is not None:
        return [(selected, None)]
    checkpoints = run / "checkpoints"
    generations: list[tuple[int, int, str]] = []
    for path in checkpoints.glob("step_*_gen_*"):
        if path.is_symlink() or not path.is_dir():
            continue
        manifest = path / "manifest.json"
        if manifest.is_symlink() or not manifest.is_file():
            continue
        try:
            value = json.loads(manifest.read_text(encoding="utf-8"))
            step = value.get("step")
            generation = value.get("generation_id")
            if type(step) is int and type(generation) is int:
                generations.append((step, generation, path.name))
        except OSError, ValueError, TypeError, json.JSONDecodeError:
            continue
    return [(name, step) for step, _, name in sorted(generations)]


def _evaluate_checkpoint_cards(
    loaded: Any, study: ArchitectureStudy
) -> tuple[dict[str, object], dict[str, object]]:
    """Evaluate cards and validation for exactly one verified loaded generation."""
    try:
        validation = loaded.evaluate()
        validation_result = {**validation, "identity": loaded.identity}
        validation_path = write_inference_result(
            loaded.run, "architecture-study-validation", validation_result
        )
        validation_record: dict[str, object] = {
            "loss": validation.get("loss"),
            "perplexity": validation.get("perplexity"),
            "valid_targets": validation.get("valid_targets"),
            "report": _path_in_run(loaded.run, validation_path),
        }
    except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
        validation_record = {"error": str(error)}
    capabilities: dict[str, object] = {}
    for card in study.cards:
        try:
            result = evaluate_capability(
                card.card,
                loaded.model,
                loaded.tokenizer,
                loaded.config.model.max_seq_len,
                loaded.device,
                engine=loaded.engine,
            )
            result["identity"] = loaded.identity
            output_path = write_capability_result(loaded.run, result)
            capabilities[card.reference] = {
                "name": card.card.name,
                "digest": card.card.digest,
                "valid": result.get("valid"),
                "score": result.get("score"),
                "passed": result.get("passed"),
                "case_count": result.get("case_count"),
                "report": _path_in_run(loaded.run, output_path),
                "result": result,
            }
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            capabilities[card.reference] = {
                "name": card.card.name,
                "digest": card.card.digest,
                "valid": False,
                "error": str(error),
            }
    return validation_record, capabilities


def _evaluate_trial(
    item: ExpandedExperiment,
    receipt_item: dict[str, object],
    study: ArchitectureStudy,
    runs_dir: Path,
    checkpoint: str | None,
    backend: str | None,
) -> dict[str, object]:
    record: dict[str, object] = {
        "coordinate": item.coordinate,
        "config_sha256": config_sha256(item.config.model_dump(mode="json")),
        "run_id": receipt_item["run_id"],
        "parameter_inventory": asdict(parameter_inventory(item.config)),
        "estimated_bytes": {
            key: inspection_report(item.config)[key]
            for key in (
                "model_weight_bytes",
                "optimizer_state_bytes",
                "estimated_checkpoint_bytes",
            )
        },
        "status": "unavailable",
        "capabilities": {},
    }
    loaded = load_run(str(receipt_item["run_id"]), runs_dir, checkpoint, backend)
    actual_config_digest = config_sha256(loaded.config.model_dump(mode="json"))
    if actual_config_digest != record["config_sha256"]:
        raise ValueError("run config does not match the submitted study coordinate")
    record["identity"] = loaded.identity
    record["status"] = "evaluated"
    validation, capabilities = _evaluate_checkpoint_cards(loaded, study)
    record["validation"] = validation
    record["capabilities"] = capabilities
    observations: list[dict[str, object]] = [
        {
            "identity": loaded.identity,
            "validation": validation,
            "capabilities": capabilities,
        }
    ]
    selected_identity = loaded.identity
    run_root = loaded.run
    del loaded
    for generation, expected_step in _checkpoint_paths(run_root, checkpoint):
        try:
            observed = load_run(
                str(receipt_item["run_id"]), runs_dir, generation, backend
            )
            try:
                if observed.identity.get("checkpoint_sha256") == selected_identity.get(
                    "checkpoint_sha256"
                ):
                    continue
                checkpoint_validation, checkpoint_capabilities = (
                    _evaluate_checkpoint_cards(observed, study)
                )
                observations.append(
                    {
                        "identity": observed.identity,
                        "validation": checkpoint_validation,
                        "capabilities": checkpoint_capabilities,
                    }
                )
            finally:
                del observed
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            observations.append(
                {
                    "checkpoint_selector": generation,
                    "checkpoint_step": expected_step,
                    "status": "unavailable",
                    "error": str(error),
                    "validation": {"error": "checkpoint could not be loaded"},
                    "capabilities": {},
                }
            )

    def observation_step(observation: dict[str, object]) -> int:
        identity = observation.get("identity")
        step = identity.get("step") if isinstance(identity, dict) else None
        if type(step) is int:
            return step
        fallback = observation.get("checkpoint_step")
        return fallback if type(fallback) is int else 2**63 - 1

    observations.sort(key=observation_step)
    record["checkpoint_observations"] = observations
    return record


def _numeric(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validation_identity_difference(
    base: dict[str, object], variant: dict[str, object]
) -> str | None:
    base_identity = base.get("identity")
    variant_identity = variant.get("identity")
    if not isinstance(base_identity, dict) or not isinstance(variant_identity, dict):
        return "run identity unavailable"
    keys: tuple[str, ...] = (
        "step",
        "tokens_seen",
        "tokenizer_sha256",
        "data_sha256",
        "source_identity_sha256",
        "runtime",
    )
    if "training_runtime" in base_identity or "training_runtime" in variant_identity:
        keys += ("training_runtime",)
    for key in keys:
        if key not in base_identity or key not in variant_identity:
            return f"run identity missing {key}"
        if base_identity[key] != variant_identity[key]:
            return f"run identity differs at {key}"
    return None


def _comparison_report(
    study: ArchitectureStudy,
    pair: StudyPair,
    evaluated: list[dict[str, object]],
) -> dict[str, object]:
    base = evaluated[pair.baseline_index]
    variant = evaluated[pair.variant_index]
    capability_reports: dict[str, object] = {}
    output: dict[str, object] = {
        "id": pair.comparison.identifier,
        "vary": pair.comparison.vary,
        "vary_fields": list(pair.comparison.vary_fields or ()),
        "context": pair.context,
        "baseline": {"coordinate": base["coordinate"], "run_id": base["run_id"]},
        "variant": {
            "coordinate": variant["coordinate"],
            "run_id": variant["run_id"],
        },
        "configuration_differences": pair.differences,
        "capabilities": capability_reports,
    }
    validation_a = base.get("validation")
    validation_b = variant.get("validation")
    identity_difference = _validation_identity_difference(base, variant)
    if (
        identity_difference is None
        and isinstance(validation_a, dict)
        and isinstance(validation_b, dict)
        and _numeric(validation_a.get("loss"))
        and _numeric(validation_b.get("loss"))
    ):
        loss_delta = float(validation_b["loss"]) - float(validation_a["loss"])
        output["validation_loss"] = {
            "baseline": validation_a["loss"],
            "variant": validation_b["loss"],
            "delta_variant_minus_baseline": loss_delta,
            "direction": "lower_is_better",
        }
    else:
        output["validation_loss"] = {
            "status": "inconclusive",
            "reason": identity_difference or "validation metric unavailable",
            "baseline_error": validation_a.get("error")
            if isinstance(validation_a, dict)
            else "run unavailable",
            "variant_error": validation_b.get("error")
            if isinstance(validation_b, dict)
            else "run unavailable",
        }

    for card in study.cards:
        base_values = base.get("capabilities")
        variant_values = variant.get("capabilities")
        base_cap = (
            base_values.get(card.reference) if isinstance(base_values, dict) else None
        )
        variant_cap = (
            variant_values.get(card.reference)
            if isinstance(variant_values, dict)
            else None
        )
        try:
            if not isinstance(base_cap, dict) or not isinstance(variant_cap, dict):
                raise TypeError("capability result unavailable")
            base_result = base_cap.get("result")
            variant_result = variant_cap.get("result")
            if not isinstance(base_result, dict) or not isinstance(
                variant_result, dict
            ):
                raise TypeError(
                    str(
                        base_cap.get("error")
                        or variant_cap.get("error")
                        or "capability result unavailable"
                    )
                )
            capability_reports[card.reference] = compare_results(
                base_result,
                variant_result,
                vary=pair.comparison.vary,
                vary_fields=pair.comparison.vary_fields,
            )
        except (TypeError, ValueError, KeyError) as error:
            capability_reports[card.reference] = {
                "outcome": "inconclusive",
                "reason": str(error),
                "score_delta": None,
            }
    return output


def _summary(
    comparisons: list[dict[str, object]], cards: tuple[StudyCard, ...]
) -> list[dict[str, object]]:
    summary: list[dict[str, object]] = []
    identifiers = sorted({str(item["id"]) for item in comparisons})
    for identifier in identifiers:
        items = [item for item in comparisons if item["id"] == identifier]
        losses: list[float] = []
        for item in items:
            validation = item.get("validation_loss")
            if isinstance(validation, dict):
                value = validation.get("delta_variant_minus_baseline")
                if _numeric(value):
                    losses.append(float(value))
        card_summaries: list[dict[str, object]] = []
        for card in cards:
            deltas: list[float] = []
            for item in items:
                capabilities = item.get("capabilities")
                value = (
                    capabilities.get(card.reference)
                    if isinstance(capabilities, dict)
                    else None
                )
                delta = value.get("score_delta") if isinstance(value, dict) else None
                if _numeric(delta):
                    deltas.append(float(delta))
            card_summaries.append(
                {
                    "card": card.card.name,
                    "card_digest": card.card.digest,
                    "paired_count": len(deltas),
                    "score_deltas": deltas,
                    "mean_score_delta": sum(deltas) / len(deltas) if deltas else None,
                }
            )
        summary.append(
            {
                "comparison": identifier,
                "pair_count": len(items),
                "validation_loss_deltas": losses,
                "mean_validation_loss_delta": sum(losses) / len(losses)
                if losses
                else None,
                "capabilities": card_summaries,
                "interpretation": "Descriptive paired observations only; no significance or universal-benefit claim.",
            }
        )
    return summary


def collect_study(
    study: ArchitectureStudy,
    receipt_path: Path,
    *,
    runs_dir: Path,
    checkpoint: str | None = None,
    backend: str | None = None,
) -> tuple[dict[str, object], Path]:
    receipt, receipt_sha256 = _read_receipt(receipt_path)
    if (
        receipt.get("study_sha256") != study.study_sha256
        or receipt.get("matrix_sha256") != study.matrix_sha256
    ):
        raise ValueError("study specification differs from submission receipt")
    expected_cards = [
        {
            "reference": card.reference,
            "name": card.card.name,
            "digest": card.card.digest,
        }
        for card in study.cards
    ]
    if receipt.get("cards") != expected_cards:
        raise ValueError("study cards differ from submission receipt")
    receipt_runs = receipt.get("runs")
    if not isinstance(receipt_runs, list) or len(receipt_runs) != len(study.expanded):
        raise ValueError("study receipt run inventory does not match the plan")
    for planned, submitted in zip(study.expanded, receipt_runs, strict=True):
        if not isinstance(submitted, dict):
            raise TypeError("study receipt run entry must be an object")
        if (
            submitted.get("coordinate") != planned.coordinate
            or submitted.get("config_sha256")
            != config_sha256(planned.config.model_dump(mode="json"))
            or not isinstance(submitted.get("run_id"), str)
            or not submitted["run_id"]
        ):
            raise ValueError("study receipt coordinate/config identity mismatch")
    evaluated: list[dict[str, object]] = []
    for planned, submitted in zip(study.expanded, receipt_runs, strict=True):
        try:
            record = _evaluate_trial(
                planned,
                submitted,
                study,
                runs_dir,
                checkpoint,
                backend,
            )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError) as error:
            record = {
                "coordinate": planned.coordinate,
                "config_sha256": config_sha256(planned.config.model_dump(mode="json")),
                "run_id": submitted["run_id"],
                "parameter_inventory": asdict(parameter_inventory(planned.config)),
                "status": "unavailable",
                "error": str(error),
                "validation": {"error": "run could not be loaded"},
                "capabilities": {},
            }
        evaluated.append(record)
    comparisons = [_comparison_report(study, pair, evaluated) for pair in study.pairs]
    report: dict[str, object] = {
        "format": "klab-architecture-study-report",
        "report_version": 1,
        "name": study.name,
        "study_sha256": study.study_sha256,
        "matrix_sha256": study.matrix_sha256,
        "receipt_sha256": receipt_sha256,
        "checkpoint": checkpoint or "latest.json",
        "cards": [
            {
                "reference": card.reference,
                "name": card.card.name,
                "digest": card.card.digest,
            }
            for card in study.cards
        ],
        "runs": evaluated,
        "comparisons": comparisons,
        "summary": _summary(comparisons, study.cards),
        "interpretation": "Matched evidence is limited to the declared cards and validation split. Paired summaries are descriptive, not statistical confirmation.",
    }
    digest = canonical_digest(report)
    report_dir = receipt_path.parent / "local-reports"
    report_path = report_dir / f"architecture-{study.study_sha256[:12]}-{digest}.json"
    payload = {**report, "report_sha256": digest}
    encoded = canonical_json(payload) + b"\n"
    try:
        _write_json_exclusive(report_path, payload)
    except FileExistsError:
        if report_path.read_bytes() != encoded:
            raise ValueError(f"conflicting architecture study report: {report_path}")
    return payload, report_path
