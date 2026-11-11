"""Evidence-bound factorial and architecture analyses for study reports."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import cast

from klab.experiments.study import _validation_identity_difference


def _mapping(value: object) -> Mapping[str, object] | None:
    return value if isinstance(value, Mapping) else None


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _metric_definitions(
    metrics: object, cards: Sequence[Mapping[str, object]]
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    outcomes: list[dict[str, object]] = []
    excluded: list[dict[str, str]] = []
    if not isinstance(metrics, list):
        return outcomes, excluded
    for metric in metrics:
        item = _mapping(metric)
        if item is None or not isinstance(item.get("name"), str):
            continue
        name = str(item["name"])
        direction = item.get("direction")
        if direction == "descriptive":
            excluded.append(
                {"metric": name, "reason": "descriptive metrics are not objectives"}
            )
            continue
        if direction not in {"lower", "higher"}:
            excluded.append(
                {"metric": name, "reason": "metric direction is not comparable"}
            )
            continue
        if name == "held-out validation loss":
            outcomes.append(
                {
                    "id": "validation_loss",
                    "metric": name,
                    "source": "validation.loss",
                    "direction": direction,
                    "kind": "validation_loss",
                }
            )
        elif name == "capability-card score":
            for card in cards:
                reference = card.get("reference")
                if isinstance(reference, str):
                    outcomes.append(
                        {
                            "id": f"capability:{reference}",
                            "metric": name,
                            "card_reference": reference,
                            "source": f"capabilities.{reference}.result.score",
                            "direction": direction,
                            "kind": "capability_score",
                        }
                    )
        else:
            excluded.append(
                {"metric": name, "reason": "no report evidence source is defined"}
            )
    return outcomes, excluded


def _observe(
    run: Mapping[str, object] | None, outcome: Mapping[str, object]
) -> tuple[float | None, str | None]:
    if run is None:
        return None, "planned factorial cell is missing"
    endpoint = _mapping(run.get("endpoint_status"))
    if endpoint is None or endpoint.get("status") != "complete":
        return None, "configured endpoint was not reached"
    if outcome.get("kind") == "validation_loss":
        validation = _mapping(run.get("validation"))
        value = validation.get("loss") if validation is not None else None
        if _finite_number(value):
            return float(cast(float | int, value)), None
        reason = validation.get("error") if validation is not None else None
        return None, str(reason or "validation metric unavailable")
    if outcome.get("kind") == "capability_score":
        capabilities = _mapping(run.get("capabilities"))
        reference = outcome.get("card_reference")
        card = (
            capabilities.get(reference)
            if capabilities is not None and isinstance(reference, str)
            else None
        )
        card_data = _mapping(card)
        result = _mapping(card_data.get("result")) if card_data is not None else None
        if result is None and card_data is not None:
            result = card_data
        if result is None or result.get("valid") is not True:
            error = card_data.get("error") if card_data is not None else None
            return None, str(error or "capability result unavailable or invalid")
        value = result.get("score")
        if _finite_number(value):
            return float(cast(float | int, value)), None
        return None, "capability score unavailable"
    return None, "unsupported report outcome source"


def _factor_parts(
    design: Mapping[str, object],
) -> tuple[dict[str, str], dict[str, str]] | None:
    factors = design.get("factors")
    if not isinstance(factors, list) or len(factors) != 2:
        return None
    normalized: list[dict[str, str]] = []
    for factor in factors:
        item = _mapping(factor)
        if item is None or not all(
            isinstance(item.get(field), str)
            for field in ("axis", "control", "treatment")
        ):
            return None
        normalized.append(
            {
                "axis": str(item["axis"]),
                "control": str(item["control"]),
                "treatment": str(item["treatment"]),
            }
        )
    if normalized[0]["axis"] == normalized[1]["axis"]:
        return None
    return normalized[0], normalized[1]


def _factor_groups(
    runs: Sequence[Mapping[str, object]],
    factor_a: Mapping[str, str],
    factor_b: Mapping[str, str],
) -> dict[
    tuple[tuple[str, str], ...], dict[tuple[str, str], list[Mapping[str, object]]]
]:
    groups: dict[
        tuple[tuple[str, str], ...],
        dict[tuple[str, str], list[Mapping[str, object]]],
    ] = {}
    axis_a, axis_b = factor_a["axis"], factor_b["axis"]
    levels_a = {factor_a["control"], factor_a["treatment"]}
    levels_b = {factor_b["control"], factor_b["treatment"]}
    for run in runs:
        coordinate = _mapping(run.get("coordinate"))
        if coordinate is None:
            continue
        value_a, value_b = coordinate.get(axis_a), coordinate.get(axis_b)
        if value_a not in levels_a or value_b not in levels_b:
            continue
        if not all(isinstance(value, str) for value in coordinate.values()):
            continue
        context = tuple(
            sorted(
                (str(axis), str(value))
                for axis, value in coordinate.items()
                if axis not in {axis_a, axis_b}
            )
        )
        cell = (str(value_a), str(value_b))
        groups.setdefault(context, {}).setdefault(cell, []).append(run)
    return groups


def _factorial_analysis(
    runs: Sequence[Mapping[str, object]],
    designs: object,
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {
        "format": "klab-factorial-analysis",
        "version": 1,
        "designs": [],
    }
    results = output["designs"]
    if not isinstance(results, list) or not isinstance(designs, list):
        return output
    for raw_design in designs:
        design = _mapping(raw_design)
        parts = _factor_parts(design) if design is not None else None
        if design is None or parts is None or not isinstance(design.get("id"), str):
            continue
        factor_a, factor_b = parts
        groups = _factor_groups(runs, factor_a, factor_b)
        context_reports: list[dict[str, object]] = []
        design_inconclusive = not groups
        cell_specs = (
            ("y00", factor_a["control"], factor_b["control"]),
            ("y10", factor_a["treatment"], factor_b["control"]),
            ("y01", factor_a["control"], factor_b["treatment"]),
            ("y11", factor_a["treatment"], factor_b["treatment"]),
        )
        for context, cells in sorted(groups.items()):
            context_data = dict(context)
            outcome_reports: list[dict[str, object]] = []
            context_inconclusive = False
            for outcome in outcomes:
                observations: list[dict[str, object]] = []
                reasons: list[str] = []
                selected: dict[str, Mapping[str, object] | None] = {}
                for cell_id, value_a, value_b in cell_specs:
                    matches = cells.get((value_a, value_b), [])
                    run = matches[0] if len(matches) == 1 else None
                    selected[cell_id] = run
                    value, reason = _observe(run, outcome)
                    cell_report: dict[str, object] = {
                        "cell": cell_id,
                        "factor_levels": {
                            factor_a["axis"]: value_a,
                            factor_b["axis"]: value_b,
                        },
                        "run_id": run.get("run_id") if run is not None else None,
                        "status": "complete" if reason is None else "inconclusive",
                        "value": value,
                    }
                    if len(matches) > 1:
                        reason = "multiple planned runs match this factorial cell"
                        cell_report["status"] = "inconclusive"
                        cell_report["value"] = None
                    if reason is not None:
                        cell_report["reason"] = reason
                        reasons.append(f"{cell_id}: {reason}")
                    observations.append(cell_report)
                reference = selected.get("y00")
                for cell_id, _, _ in cell_specs[1:]:
                    candidate = selected.get(cell_id)
                    if reference is not None and candidate is not None:
                        difference = _validation_identity_difference(
                            dict(reference), dict(candidate)
                        )
                        if difference is not None:
                            reasons.append(f"{cell_id}: {difference}")
                values = [item.get("value") for item in observations]
                complete = (
                    len(values) == 4
                    and all(_finite_number(value) for value in values)
                    and not reasons
                )
                outcome_report: dict[str, object] = {
                    "id": outcome["id"],
                    "metric": outcome["metric"],
                    "source": outcome["source"],
                    "direction": outcome["direction"],
                    "status": "complete" if complete else "inconclusive",
                    "cells": observations,
                }
                if outcome.get("card_reference") is not None:
                    outcome_report["card_reference"] = outcome["card_reference"]
                if complete:
                    y00, y10, y01, y11 = (
                        float(cast(float | int, value)) for value in values
                    )
                    factor_a_effect = ((y10 - y00) + (y11 - y01)) / 2
                    factor_b_effect = ((y01 - y00) + (y11 - y10)) / 2
                    outcome_report["factor_a_main_effect"] = factor_a_effect
                    outcome_report["factor_b_main_effect"] = factor_b_effect
                    outcome_report["main_effects"] = {
                        factor_a["axis"]: factor_a_effect,
                        factor_b["axis"]: factor_b_effect,
                    }
                    outcome_report["interaction_delta"] = y11 - y10 - y01 + y00
                else:
                    context_inconclusive = True
                    outcome_report["reasons"] = reasons or [
                        "all four matched endpoint values are required"
                    ]
                outcome_reports.append(outcome_report)
            context_reports.append(
                {
                    "coordinates": context_data,
                    "status": "inconclusive" if context_inconclusive else "complete",
                    "outcomes": outcome_reports,
                }
            )
            design_inconclusive = design_inconclusive or context_inconclusive
        design_report: dict[str, object] = {
            "id": design["id"],
            "factors": [factor_a, factor_b],
            "status": "inconclusive" if design_inconclusive else "complete",
            "contexts": context_reports,
        }
        if not groups:
            design_report["reason"] = (
                "no planned runs contain the declared factorial factor levels"
            )
        results.append(design_report)
    return output


def _evidence_context(
    run: Mapping[str, object], selection: Mapping[str, object]
) -> tuple[dict[str, object] | None, str | None]:
    identity = _mapping(run.get("identity"))
    coordinate = _mapping(run.get("coordinate"))
    if identity is None or coordinate is None:
        return None, "run or coordinate identity unavailable"
    required = (
        "step",
        "tokens_seen",
        "tokenizer_sha256",
        "data_sha256",
        "source_identity_sha256",
        "runtime",
        "training_runtime",
    )
    if any(key not in identity for key in required):
        return None, "comparable endpoint identity is incomplete"
    seed = coordinate.get("seed")
    if seed is None:
        config = _mapping(identity.get("config"))
        seed = config.get("seed") if config is not None else None
    if seed is None:
        return None, "seed identity unavailable"
    return (
        {
            "selection": dict(selection),
            "seed": seed,
            "step": identity["step"],
            "tokens_seen": identity["tokens_seen"],
            "tokenizer_sha256": identity["tokenizer_sha256"],
            "data_sha256": identity["data_sha256"],
            "source_identity_sha256": identity["source_identity_sha256"],
            "runtime": identity["runtime"],
            "training_runtime": identity["training_runtime"],
        },
        None,
    )


def _dominates(
    left: Mapping[str, object],
    right: Mapping[str, object],
    objectives: Sequence[Mapping[str, object]],
) -> bool:
    strict = False
    for objective in objectives:
        key = str(objective["id"])
        left_value = float(cast(float | int, left[key]))
        right_value = float(cast(float | int, right[key]))
        if objective["direction"] == "higher":
            left_value, right_value = -left_value, -right_value
        if left_value > right_value:
            return False
        strict = strict or left_value < right_value
    return strict


def _nondominance_analysis(
    runs: Sequence[Mapping[str, object]],
    selection: Mapping[str, object],
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    objectives = [
        {
            "id": item["id"],
            "metric": item["metric"],
            "source": item["source"],
            "direction": item["direction"],
            **(
                {"card_reference": item["card_reference"]}
                if item.get("card_reference") is not None
                else {}
            ),
        }
        for item in outcomes
    ]
    groups: dict[str, dict[str, object]] = {}
    excluded: list[dict[str, object]] = []
    for run in runs:
        run_id = run.get("run_id")
        context, context_error = _evidence_context(run, selection)
        if context is None:
            excluded.append(
                {"run_id": run_id, "reason": context_error or "identity unavailable"}
            )
            continue
        context_key = json.dumps(
            context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        group = groups.setdefault(
            context_key,
            {
                "context": context,
                "candidates": [],
                "inconclusive_candidates": [],
            },
        )
        candidate: dict[str, object] = {
            "run_id": run_id,
            "coordinate": run.get("coordinate"),
            "values": {},
        }
        values = candidate["values"]
        if not isinstance(values, dict):
            continue
        reasons: list[str] = []
        for outcome in outcomes:
            value, reason = _observe(run, outcome)
            if value is None:
                reasons.append(f"{outcome['id']}: {reason or 'unavailable'}")
            else:
                values[str(outcome["id"])] = value
        candidates = group["candidates"]
        incomplete = group["inconclusive_candidates"]
        if not isinstance(candidates, list) or not isinstance(incomplete, list):
            continue
        if reasons or len(values) != len(objectives) or not objectives:
            candidate["status"] = "inconclusive"
            candidate["reasons"] = reasons or ["no directed objectives are available"]
            incomplete.append(candidate)
        else:
            candidate["status"] = "complete"
            candidates.append(candidate)
    group_reports: list[dict[str, object]] = []
    for context_key, group in sorted(groups.items()):
        candidates = group["candidates"]
        incomplete = group["inconclusive_candidates"]
        if not isinstance(candidates, list) or not isinstance(incomplete, list):
            continue
        frontier = [
            candidate
            for candidate in candidates
            if not any(
                other is not candidate
                and _dominates(other["values"], candidate["values"], objectives)
                for other in candidates
            )
        ]
        ties: dict[tuple[float, ...], list[str]] = {}
        for candidate in frontier:
            values = cast(dict[str, object], candidate["values"])
            key = tuple(
                float(cast(float | int, values[str(item["id"])])) for item in objectives
            )
            ties.setdefault(key, []).append(str(candidate["run_id"]))
        group_reports.append(
            {
                "id": hashlib.sha256(context_key.encode("utf-8")).hexdigest()[:16],
                "context": group["context"],
                "status": "complete" if candidates else "inconclusive",
                "objectives": objectives,
                "candidates": candidates,
                "nondominated_run_ids": [candidate["run_id"] for candidate in frontier],
                "ties": [
                    {
                        "values": {
                            str(item["id"]): value
                            for item, value in zip(objectives, key, strict=True)
                        },
                        "run_ids": sorted(run_ids),
                    }
                    for key, run_ids in sorted(ties.items())
                    if len(run_ids) > 1
                ],
                "inconclusive_candidates": incomplete,
            }
        )
    return {
        "format": "klab-nondominance-analysis",
        "version": 1,
        "status": "complete"
        if group_reports
        and all(group["status"] == "complete" for group in group_reports)
        else "inconclusive",
        "objectives": objectives,
        "groups": group_reports,
        "excluded_runs": excluded,
    }


def _allocation_heatmaps(
    runs: Sequence[Mapping[str, object]],
    designs: object,
    quantities: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    inventory_by_config = {
        item.get("config_sha256"): _mapping(item.get("inventory"))
        for item in quantities
        if isinstance(item.get("config_sha256"), str)
    }
    heatmaps: list[dict[str, object]] = []
    if not isinstance(designs, list):
        designs = []
    for raw_design in designs:
        design = _mapping(raw_design)
        parts = _factor_parts(design) if design is not None else None
        if design is None or parts is None:
            continue
        factor_a, factor_b = parts
        groups = _factor_groups(runs, factor_a, factor_b)
        for context, cells in sorted(groups.items()):
            cells_out: list[dict[str, object]] = []
            for cell_id, value_a, value_b in (
                ("y00", factor_a["control"], factor_b["control"]),
                ("y10", factor_a["treatment"], factor_b["control"]),
                ("y01", factor_a["control"], factor_b["treatment"]),
                ("y11", factor_a["treatment"], factor_b["treatment"]),
            ):
                matches = cells.get((value_a, value_b), [])
                run = matches[0] if len(matches) == 1 else None
                config_sha = run.get("config_sha256") if run is not None else None
                inventory = (
                    inventory_by_config.get(config_sha)
                    if isinstance(config_sha, str)
                    else None
                )
                if run is None or inventory is None or len(matches) != 1:
                    cells_out.append(
                        {
                            "cell": cell_id,
                            "factor_levels": {
                                factor_a["axis"]: value_a,
                                factor_b["axis"]: value_b,
                            },
                            "run_id": run.get("run_id") if run is not None else None,
                            "status": "inconclusive",
                            "reason": "architectural inventory unavailable or cell is ambiguous",
                            "parameters": None,
                        }
                    )
                    continue
                total = inventory.get("total")
                table = inventory.get("memory_table")
                adapter = inventory.get("memory_adapter")
                active = inventory.get("active_per_token")
                if (
                    not isinstance(total, int)
                    or isinstance(total, bool)
                    or total <= 0
                    or not isinstance(table, int)
                    or isinstance(table, bool)
                    or not isinstance(adapter, int)
                    or isinstance(adapter, bool)
                    or not isinstance(active, int)
                    or isinstance(active, bool)
                ):
                    cells_out.append(
                        {
                            "cell": cell_id,
                            "factor_levels": {
                                factor_a["axis"]: value_a,
                                factor_b["axis"]: value_b,
                            },
                            "run_id": run.get("run_id"),
                            "status": "inconclusive",
                            "reason": "architectural parameter inventory is malformed",
                            "parameters": None,
                        }
                    )
                    continue
                memory = int(table) + int(adapter)
                cells_out.append(
                    {
                        "cell": cell_id,
                        "factor_levels": {
                            factor_a["axis"]: value_a,
                            factor_b["axis"]: value_b,
                        },
                        "run_id": run.get("run_id"),
                        "status": "complete",
                        "parameters": {
                            "total": int(total),
                            "active_per_token": int(active),
                            "non_memory": int(total) - memory,
                            "memory_table": int(table),
                            "memory_adapter": int(adapter),
                            "memory_total": memory,
                            "memory_share": memory / int(total),
                        },
                    }
                )
            heatmaps.append(
                {
                    "design_id": design.get("id"),
                    "context": dict(context),
                    "factor_a": factor_a,
                    "factor_b": factor_b,
                    "value": "memory_share",
                    "kind": "architectural_estimate",
                    "status": "complete"
                    if all(item["status"] == "complete" for item in cells_out)
                    else "inconclusive",
                    "cells": cells_out,
                }
            )
    return {
        "format": "klab-allocation-heatmaps",
        "version": 1,
        "kind": "architectural_estimates_not_measured_costs",
        "heatmaps": heatmaps,
    }


def _boundary_sweeps(
    runs: Sequence[Mapping[str, object]],
    axes: object,
    outcomes: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    declared: dict[str, list[str]] = {}
    declared_patches: dict[str, dict[str, object]] = {}
    if isinstance(axes, Mapping):
        for axis, options in axes.items():
            if not isinstance(axis, str) or not isinstance(options, list):
                continue
            for option in options:
                if isinstance(option, str):
                    label, patch = option, None
                else:
                    item = _mapping(option)
                    if item is None:
                        continue
                    label_value = item.get("label")
                    if not isinstance(label_value, str):
                        continue
                    label = label_value
                    patch_value = item.get("set")
                    patch = (
                        dict(patch_value) if isinstance(patch_value, Mapping) else None
                    )
                labels = declared.setdefault(axis, [])
                if label not in labels:
                    labels.append(label)
                if patch is not None:
                    declared_patches.setdefault(axis, {})[label] = patch
    for run in runs:
        coordinate = _mapping(run.get("coordinate"))
        if coordinate is None:
            continue
        for axis, label in coordinate.items():
            if (
                axis == "seed"
                or not isinstance(axis, str)
                or not isinstance(label, str)
            ):
                continue
            labels = declared.setdefault(axis, [])
            if label not in labels:
                labels.append(label)
    axis_reports: list[dict[str, object]] = []
    for axis, labels in declared.items():
        if axis == "seed" or len(labels) < 2:
            continue
        groups: dict[
            tuple[tuple[str, str], ...], dict[str, list[Mapping[str, object]]]
        ] = {}
        for run in runs:
            coordinate = _mapping(run.get("coordinate"))
            if coordinate is None or axis not in coordinate:
                continue
            if not all(isinstance(value, str) for value in coordinate.values()):
                continue
            context = tuple(
                sorted(
                    (str(name), str(value))
                    for name, value in coordinate.items()
                    if name != axis
                )
            )
            groups.setdefault(context, {}).setdefault(str(coordinate[axis]), []).append(
                run
            )
        group_reports: list[dict[str, object]] = []
        for context, cells in sorted(groups.items()):
            selected = {
                label: matches[0]
                for label, matches in cells.items()
                if len(matches) == 1
            }
            anchor = next(
                (selected[label] for label in labels if label in selected), None
            )
            level_reports: list[dict[str, object]] = []
            for label in labels:
                matches = cells.get(label, [])
                run = selected.get(label)
                level: dict[str, object] = {
                    "label": label,
                    "run_id": run.get("run_id") if run is not None else None,
                    "status": "complete" if run is not None else "inconclusive",
                    "outcomes": [],
                }
                patch = declared_patches.get(axis, {}).get(label)
                if patch is not None:
                    level["configuration_patch"] = patch
                reasons: list[str] = []
                if len(matches) > 1:
                    reasons.append("multiple planned runs match this sweep coordinate")
                elif run is None:
                    reasons.append("planned sweep coordinate is missing")
                observations: list[dict[str, object]] = []
                for outcome in outcomes:
                    value, reason = _observe(run, outcome)
                    if anchor is not None and run is not None:
                        difference = _validation_identity_difference(
                            dict(anchor), dict(run)
                        )
                        if difference is not None:
                            reason = difference
                    observation: dict[str, object] = {
                        "id": outcome["id"],
                        "metric": outcome["metric"],
                        "direction": outcome["direction"],
                        "value": value,
                        "status": "complete" if reason is None else "inconclusive",
                    }
                    if reason is not None:
                        observation["reason"] = reason
                        reasons.append(f"{outcome['id']}: {reason}")
                    observations.append(observation)
                level["outcomes"] = observations
                if reasons:
                    level["status"] = "inconclusive"
                    level["reasons"] = reasons
                level_reports.append(level)
            group_reports.append(
                {
                    "context": dict(context),
                    "status": "complete"
                    if all(item["status"] == "complete" for item in level_reports)
                    else "inconclusive",
                    "levels": level_reports,
                }
            )
        axis_reports.append(
            {
                "axis": axis,
                "levels": labels,
                "groups": group_reports,
                "status": "complete"
                if group_reports
                and all(item["status"] == "complete" for item in group_reports)
                else "inconclusive",
            }
        )
    return {
        "format": "klab-boundary-sweeps",
        "version": 1,
        "interpretation": "observed configured levels only; no threshold is inferred",
        "axes": axis_reports,
    }


_ALLOCATION_REGIMES = {
    "default": "iso-neural",
    "iso-neural": "iso-neural",
    "iso-total": "iso-total",
    "iso-active": "iso-active",
    "iso-token": "iso-token",
    "iso-flop": "iso-flop",
}


def _allocation_payload(run: Mapping[str, object]) -> Mapping[str, object] | None:
    """Return the runtime-owned allocation evidence without transforming it."""
    for key in ("allocation", "allocation_accounting", "allocation_metrics"):
        payload = _mapping(run.get(key))
        if payload is not None:
            return payload
    identity = _mapping(run.get("identity"))
    return _mapping(identity.get("allocation")) if identity is not None else None


def _allocation_resource_accounting(
    run: Mapping[str, object], payload: Mapping[str, object] | None
) -> dict[str, object]:
    identity = _mapping(run.get("identity"))
    inventory = _mapping(run.get("parameter_inventory"))
    if inventory is None and identity is not None:
        inventory = _mapping(identity.get("parameter_inventory"))

    def count(*names: str) -> int | None:
        if inventory is None:
            return None
        for name in names:
            value = inventory.get(name)
            if type(value) is int and value >= 0:
                return value
        return None

    runtime_totals = {
        "allocation/raw_tokens": 0.0,
        "allocation/valid_targets": 0.0,
        "allocation/weighted_neural_supervision_mass": 0.0,
    }
    seen_metrics: set[str] = set()
    metric_rows = run.get("allocation_metrics")
    if isinstance(metric_rows, list):
        for row in metric_rows:
            metric = _mapping(row)
            if metric is None:
                continue
            name, value = metric.get("name"), metric.get("value")
            if (
                isinstance(name, str)
                and name in runtime_totals
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
                and value >= 0
            ):
                runtime_totals[name] += float(value)
                seen_metrics.add(name)

    def runtime_total(name: str) -> float | None:
        return runtime_totals[name] if name in seen_metrics else None

    trainable = count("trainable")
    memory_table = count("memory_table", "engram_table")
    memory_adapter = count("memory_adapter", "engram_adapter")
    neural_trainable = (
        trainable - memory_table - memory_adapter
        if trainable is not None
        and memory_table is not None
        and memory_adapter is not None
        and trainable >= memory_table + memory_adapter
        else None
    )
    active = count("active_per_token")
    raw_tokens = runtime_total("allocation/raw_tokens")
    valid_targets = runtime_total("allocation/valid_targets")
    weighted_mass = runtime_total("allocation/weighted_neural_supervision_mass")
    prepared_train: Mapping[str, object] | None = None
    if payload is not None:
        splits = _mapping(payload.get("splits"))
        if splits is not None:
            prepared_train = _mapping(splits.get("train"))
    return {
        "iso-neural": {
            "unit": "trainable_parameters",
            "value": neural_trainable,
            "kind": "architectural_count",
        },
        "iso-total": {
            "unit": "trainable_parameters",
            "value": trainable,
            "kind": "architectural_count",
        },
        "iso-active": {
            "unit": "active_parameters_per_token",
            "value": active,
            "kind": "architectural_count",
        },
        "iso-token": {
            "raw_input_tokens": raw_tokens,
            "valid_supervised_targets": valid_targets,
            "weighted_neural_supervision_mass": weighted_mass,
            "prepared_train_raw_tokens": (
                prepared_train.get("raw_tokens") if prepared_train is not None else None
            ),
            "prepared_train_valid_supervised_targets": (
                prepared_train.get("valid_supervised_targets")
                if prepared_train is not None
                else None
            ),
            "kind": "measured_runtime_counts",
        },
        "iso-flop": {
            "estimated_parameter_proxy_flops": (
                6 * active * raw_tokens
                if active is not None and raw_tokens is not None
                else None
            ),
            "formula": "6 × active parameters per token × measured raw input tokens",
            "kind": "rough_estimate",
            "scope": (
                "parameter-based training proxy; excludes exact attention work, "
                "memory lookup/retrieval, optimizer work, and hardware effects"
            ),
        },
    }


def _allocation_observations(run: Mapping[str, object]) -> list[object]:
    """Preserve every runtime-supplied checkpoint observation in source order."""
    for key in ("allocation_curve", "checkpoint_observations"):
        observations = run.get(key)
        if isinstance(observations, list):
            return list(observations)
    payload = _allocation_payload(run)
    if payload is not None:
        for key in ("curve", "checkpoint_observations", "observations"):
            observations = payload.get(key)
            if isinstance(observations, list):
                return list(observations)
    identity = _mapping(run.get("identity"))
    if identity is None:
        return []
    # Endpoint evidence remains a real checkpoint observation even if a producer
    # did not retain a periodic curve.
    return [
        {
            "checkpoint_sha256": identity.get("checkpoint_sha256"),
            "checkpoint_relative_path": identity.get("checkpoint_relative_path"),
            "step": identity.get("step"),
            "tokens_seen": identity.get("tokens_seen"),
        }
    ]


def _allocation_curve_analysis(
    runs: Sequence[Mapping[str, object]], selection: Mapping[str, object]
) -> dict[str, object] | None:
    """Render allocation evidence as raw task/checkpoint rows, never an estimate.

    The runtime owns the accounting schema.  This consumer intentionally carries
    its records verbatim alongside the checkpoint-bound run identity so that
    new ownership counters do not get silently collapsed into a single score.
    """
    denominators = {
        "iso-neural": "neural-owned trainable parameters",
        "iso-total": "total trainable parameters",
        "iso-active": "active parameters per token",
        "iso-token": "raw input tokens and valid supervised targets",
        "iso-flop": "estimated FLOPs only; never inferred from neural loss weight",
    }
    rows: list[dict[str, object]] = []
    for run in runs:
        coordinate = _mapping(run.get("coordinate"))
        payload = _allocation_payload(run)
        if payload is None and not any(
            isinstance(run.get(key), list)
            for key in ("allocation_curve", "checkpoint_observations")
        ):
            continue
        allocation_label = (
            coordinate.get("allocation") or coordinate.get("ownership")
            if coordinate is not None
            else None
        )
        selected_regime = (
            _ALLOCATION_REGIMES.get(str(selection.get("design")))
            if selection.get("design") is not None
            else None
        )
        resource_regime = (
            payload.get("resource_regime")
            if payload is not None and isinstance(payload.get("resource_regime"), str)
            else selected_regime
        )
        ownership_profile = (
            payload.get("ownership_profile")
            if payload is not None and isinstance(payload.get("ownership_profile"), str)
            else coordinate.get("ownership")
            if coordinate is not None and isinstance(coordinate.get("ownership"), str)
            else None
        )
        if isinstance(allocation_label, str) and allocation_label.count("-") >= 2:
            label_regime, label_profile = allocation_label.rsplit("-", 1)
            resource_regime = resource_regime or label_regime
            ownership_profile = ownership_profile or label_profile
        resource_regime = resource_regime or "unavailable"
        ownership_profile = ownership_profile or "unavailable"
        neural_loss_weight = (
            coordinate.get("neural_loss_weight", "unavailable")
            if coordinate is not None
            else "unavailable"
        )
        identity = _mapping(run.get("identity"))
        runtime_metrics = run.get("allocation_metrics")
        tasks: list[dict[str, object]] = []
        capabilities = _mapping(run.get("capabilities"))
        if capabilities is not None:
            for reference, value in capabilities.items():
                if isinstance(reference, str) and _mapping(value) is not None:
                    tasks.append({"task": reference, "evidence": value})
        curve_id = f"{ownership_profile}:{neural_loss_weight}"
        rows.append(
            {
                "run_id": run.get("run_id"),
                "config_sha256": run.get("config_sha256"),
                "resource_regime": resource_regime,
                "ownership_profile": ownership_profile,
                "neural_loss_weight": neural_loss_weight,
                "curve_id": curve_id,
                "coordinate": dict(coordinate) if coordinate is not None else None,
                "checkpoint_identity": {
                    key: identity.get(key) if identity is not None else None
                    for key in (
                        "checkpoint_sha256",
                        "checkpoint_relative_path",
                        "step",
                        "tokens_seen",
                        "source_identity_sha256",
                    )
                },
                "allocation": dict(payload) if payload is not None else None,
                "resource_accounting": _allocation_resource_accounting(run, payload),
                "runtime_allocation_metrics": list(runtime_metrics)
                if isinstance(runtime_metrics, list)
                else [],
                "checkpoint_observations": _allocation_observations(run),
                "tasks": tasks,
            }
        )
    if not rows:
        return None
    regime_order = ("iso-neural", "iso-total", "iso-active", "iso-token", "iso-flop")
    observed_regimes = {str(row["resource_regime"]) for row in rows}
    resource_regimes: list[dict[str, object]] = []
    for resource_regime in (
        *regime_order,
        *sorted(observed_regimes - set(regime_order)),
    ):
        if resource_regime not in observed_regimes:
            continue
        profiles = sorted(
            {
                str(row["ownership_profile"])
                for row in rows
                if row["resource_regime"] == resource_regime
            }
        )
        resource_regimes.append(
            {
                "label": resource_regime,
                "comparison_denominator": denominators.get(
                    resource_regime, "declared allocation comparison denominator"
                ),
                "cost_kind": (
                    "estimated" if resource_regime == "iso-flop" else "accounting"
                ),
                "ownership_profiles": profiles,
            }
        )
    return {
        "format": "klab-memory-allocation-curve",
        "version": 1,
        "resource_regimes": resource_regimes,
        "regime": next(
            (
                item
                for item in resource_regimes
                if item["label"]
                == _ALLOCATION_REGIMES.get(str(selection.get("design")))
            ),
            resource_regimes[0] if len(resource_regimes) == 1 else None,
        ),
        "accounting": {
            "raw_input_tokens": "measured_count",
            "valid_supervised_targets": "measured_count",
            "weighted_neural_supervision_mass": (
                "measured_weighted_mass: neural_loss_weight × "
                "(neural-owned + hybrid-owned valid supervised targets)"
            ),
            "ownership_labels": {
                "neural": 0,
                "lexical": 1,
                "semantic": 2,
                "hybrid": 3,
            },
        },
        "rows": rows,
        "interpretation": (
            "Each task and every supplied checkpoint/allocation observation is "
            "retained in source order. Resource labels are denominator views, "
            "not independent replications; FLOPs are rough estimates only. "
            "No smoothing, interpolation, ranking, task averaging, or measured "
            "FLOP savings is inferred."
        ),
    }


def build_research_analysis(
    runs: Sequence[Mapping[str, object]],
    *,
    entry: Mapping[str, object],
    selection: Mapping[str, object],
    factorial_designs: object,
    design_axes: object,
    quantities: Sequence[Mapping[str, object]],
    cards: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    """Analyze only validated endpoint rows and recipe-declared directions."""
    outcomes, excluded = _metric_definitions(entry.get("dependent_metrics"), cards)
    allocation_curve = _allocation_curve_analysis(runs, selection)
    output: dict[str, object] = {
        "format": "klab-research-analysis",
        "version": 1,
        "factorial": _factorial_analysis(runs, factorial_designs, outcomes),
        "nondominance": _nondominance_analysis(runs, selection, outcomes),
        "allocation_heatmaps": _allocation_heatmaps(
            runs, factorial_designs, quantities
        ),
        "boundary_sweeps": _boundary_sweeps(runs, design_axes, outcomes),
        "excluded_metrics": excluded,
    }
    if allocation_curve is not None:
        output["allocation_curve"] = allocation_curve
    return output
