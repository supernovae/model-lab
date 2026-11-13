"""Comparable points: one checkpoint's quality and cost, from sealed records.

Shared by ``klab compare`` and the dashboard's Pareto view (no Streamlit
import here). Sources are verified lab records only (``lab_records``):

* try records: each arm's held-out score (with its ``eval_group``) and cost,
  so runs without a probe battery still appear;
* probe records: the battery's target and baseline, including lm-eval;
* code benchmark records (``klab bench``): execution-scored pass@1;
* packaged reference results (``probes/reference_results/*.json``): sealed
  probe and code benchmark records of pinned public checkpoints, so nothing is
  downloaded to plot or compare against them.

A metric value only compares within its group: held-out loss within one
``eval_group`` (validation data, tokenizer, loss mask, protocol), lm-eval
accuracy within one ``benchmark_group`` (tasks, task versions, shots, items),
fact recall within one ``item_group`` (the same rendered prompts, answers and
candidates under one ranking protocol; text-level, so tokenizer-independent),
code pass@1 within one ``benchmark_group`` of ``klab bench`` (task, data,
items, decoding and sandbox).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from klab.lab_records import iter_lab_records, read_lab_record


@dataclass(frozen=True)
class Metric:
    id: str
    label: str
    group_label: str
    higher_is_better: bool
    not_comparable: str
    missing_hint: str


METRICS = {
    "heldout_loss": Metric(
        id="heldout_loss",
        label="held-out loss",
        group_label="eval group",
        higher_is_better=False,
        not_comparable="different eval group (validation data, tokenizer, loss "
        "mask or eval protocol differ)",
        missing_hint="score it on our validation split (`klab try` or `klab probe`)",
    ),
    "lm_eval": Metric(
        id="lm_eval",
        label="lm-eval accuracy",
        group_label="benchmark group",
        higher_is_better=True,
        not_comparable="different benchmark group (tasks, limit, few-shot or "
        "task versions differ)",
        missing_hint="run `klab probe RECORD --config full` (needs the lmeval extra)",
    ),
    "fact_recall": Metric(
        id="fact_recall",
        label="fact recall",
        group_label="item group",
        higher_is_better=True,
        not_comparable="different items (prompts, answers or candidates differ)",
        missing_hint="run `klab probe RECORD --config standard`",
    ),
    "needle": Metric(
        id="needle",
        label="needle retrieval",
        group_label="item group",
        higher_is_better=True,
        not_comparable="different needle items (text, lengths or candidates "
        "differ; results before suite v4 were sized in model tokens)",
        missing_hint="run `klab probe RECORD --config standard`",
    ),
    "long_needle": Metric(
        id="long_needle",
        label="long-context needle retrieval",
        group_label="item group",
        higher_is_better=True,
        not_comparable="different long-needle items (text, lengths or candidates "
        "differ)",
        missing_hint="run `klab probe RECORD --config full`",
    ),
    "code_pass1": Metric(
        id="code_pass1",
        label="code pass@1",
        group_label="code benchmark group",
        higher_is_better=True,
        not_comparable="different code benchmark group (task, data, items, "
        "decoding or sandbox differ)",
        missing_hint="run `klab bench RECORD --task humaneval`",
    ),
}
# Probe rows whose ``details.items`` carry per-item credit (paired evidence).
RANKING_METRICS = ("fact_recall", "needle", "long_needle")

# Cost axes: resident counts every weight held in memory, active counts what
# one token touches (one embedding row, the routed experts and memory rows).
COSTS = {
    "parameters": "resident parameters",
    "active_parameters": "active parameters / token",
    "parameter_bytes": "resident weight bytes",
    "active_parameter_bytes": "active weight bytes / token",
    "tokens_seen": "training tokens",
    "ms_per_token": "scoring latency (ms / token)",
}
IDENTITY_FIELDS = (
    "run_id",
    "checkpoint_sha256",
    "step",
    "tokens_seen",
    *(c for c in COSTS if c not in {"tokens_seen", "ms_per_token"}),
)


def mean_chance(chance: Mapping[str, Any] | None) -> float | None:
    """Mean chance baseline over a group's tasks, or None when any is unknown.

    Custom lm-eval tasks (e.g. winogrande) have no known chance baseline; a
    mean over only the known tasks would draw a misleading line, so the whole
    chance line is omitted instead.
    """
    if not chance:
        return None
    values = list(chance.values())
    if any(
        not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v)
        for v in values
    ):
        return None
    return sum(values) / len(values)


def consistent_chance(values: Iterable[Any]) -> float | None:
    """One chance level shared by every result of a group, else None.

    None when there are no results, any result's chance is unknown, or the
    results disagree (they would not be on one scale).
    """
    known = list(values)
    if not known or len({repr(v) for v in known}) != 1:
        return None
    return mean_chance({"chance": known[0]})


def packaged_reference_records(
    kind: str = "probe",
) -> list[tuple[Path, dict[str, Any]]]:
    """Sealed reference results of KIND shipped with Model Lab (verified on read)."""
    folder = resources.files("klab.probes") / "reference_results"
    out = []
    for item in sorted(folder.iterdir(), key=lambda p: p.name):
        if item.name.endswith(".json"):
            path = Path(str(item))
            found, record = read_lab_record(path)
            if found == kind:
                out.append((path, record))
    return out


def _row(result: Mapping[str, Any], probe: str) -> Mapping[str, Any]:
    for row in result.get("probes") or []:
        if row.get("id") == probe:
            return row
    return {}


def _lm_eval_items(tasks: Mapping[str, Any] | None) -> dict[str, list[float]]:
    return {task: list(row.get("items") or []) for task, row in (tasks or {}).items()}


def is_final(result: Mapping[str, Any]) -> bool:
    """A ``probe --config final`` result: scored on the held-back final split."""
    return (
        bool(result.get("final"))
        or (result.get("suite") or {}).get("verdict_split") == "final"
    )


def probe_points(
    result: Mapping[str, Any],
    *,
    source: str,
    path: Path | None,
    packaged: bool,
    try_record: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Target and baseline of one probe battery as points.

    A battery inside a lab try names its points like the try's arms
    (``TRY_ID:candidate``), so one checkpoint reads the same in every view.
    """
    loss = _row(result, "heldout_loss")
    lm = _row(result, "lm_eval")
    loss_details, lm_details = loss.get("details") or {}, lm.get("details") or {}
    points = []
    for role, who in (
        ("candidate", result.get("target")),
        ("baseline", result.get("baseline")),
    ):
        if not who or not who.get("checkpoint_sha256"):
            continue
        candidate = role == "candidate"
        metrics: dict[str, dict[str, Any]] = {}
        value = loss.get("value" if candidate else "baseline_value")
        if value is not None and loss.get("status") not in {"error", "not_comparable"}:
            metrics["heldout_loss"] = {
                "value": value,
                "group": who.get("eval_group"),
                "window_sums": loss_details.get("window_sums") if candidate else None,
                "window_counts": loss_details.get("window_counts")
                if candidate
                else None,
            }
        value = lm.get("value" if candidate else "baseline_value")
        complete = lm.get("status") not in {"error", "unavailable", "skipped"}
        if value is not None and complete and lm_details.get("benchmark_group"):
            tasks = lm_details.get("tasks" if candidate else "baseline_tasks")
            metrics["lm_eval"] = {
                "value": value,
                # A baseline row only exists when both sides shared the group.
                "group": lm_details["benchmark_group"],
                "tasks": {t: r.get("acc") for t, r in (tasks or {}).items()},
                "items": _lm_eval_items(tasks),
                "chance": lm_details.get("chance"),
            }
        for metric_id in RANKING_METRICS:
            row = _row(result, metric_id)
            details = row.get("details") or {}
            value = row.get("value" if candidate else "baseline_value")
            if (
                value is None
                or row.get("status")
                in {"error", "unavailable", "skipped", "not_comparable"}
                or not details.get("item_group")
            ):
                continue
            key = "credit" if candidate else "baseline_credit"
            credits = [item.get(key) for item in details.get("items") or []]
            metrics[metric_id] = {
                "value": value,
                "group": details["item_group"],
                "items": {metric_id: credits}
                if credits and None not in credits
                else None,
                "chance": {metric_id: details.get("chance")},
            }
        reference = who.get("reference")
        label = f"{who.get('run_id')}@{who.get('step')}"
        if reference:
            label = reference["name"]
        elif try_record is not None:
            label = f"{try_record.get('try_id')}:{role}"
        points.append(
            {
                **{k: who.get(k) for k in IDENTITY_FIELDS},
                "label": label,
                "kind": "reference"
                if reference
                else "try"
                if try_record is not None
                else "probe",
                "role": role,
                "reference": reference,
                "packaged": packaged,
                "ms_per_token": loss_details.get("ms_per_token") if candidate else None,
                "verdict": (result.get("verdict") or {}).get("status")
                if candidate
                else "baseline",
                # Final-split provenance: kept so ordinary views can refuse it.
                "final": is_final(result),
                "verdict_split": "final" if is_final(result) else "heldout",
                "source": source,
                "path": str(path) if path else None,
                "created_at": str(
                    (try_record or result).get("created_at")  # type: ignore[union-attr]
                ),
                "metrics": metrics,
            }
        )
    return points


def try_points(record: Mapping[str, Any], path: Path) -> list[dict[str, Any]]:
    """Each scored arm of a lab try (whether or not it was probed)."""
    points = []
    for role, arm in (record.get("arms") or {}).items():
        heldout = arm.get("heldout") or {}
        if not arm.get("checkpoint_sha256") or heldout.get("loss") is None:
            continue
        points.append(
            {
                **{k: arm.get(k) for k in IDENTITY_FIELDS},
                "label": f"{record.get('try_id')}:{role}",
                "kind": "try",
                "role": role,
                "reference": None,
                "packaged": False,
                "ms_per_token": heldout.get("ms_per_token"),
                "verdict": (record.get("comparison") or {}).get("verdict"),
                "final": False,
                "verdict_split": "heldout",
                "source": str(record.get("try_id")),
                "path": str(path),
                "created_at": str(record.get("created_at")),
                "question": record.get("question"),
                "metrics": {
                    "heldout_loss": {
                        "value": heldout["loss"],
                        # Older records carry no group: shown, never compared.
                        "group": arm.get("eval_group"),
                        "window_sums": heldout.get("window_sums"),
                        "window_counts": heldout.get("window_counts"),
                    }
                },
            }
        )
    return points


def bench_points(
    record: Mapping[str, Any], path: Path, *, packaged: bool = False
) -> list[dict[str, Any]]:
    """The scored checkpoint of one complete code benchmark (else nothing)."""
    who = record.get("target") or {}
    code = record.get("code_pass1") or {}
    if (
        not who.get("checkpoint_sha256")
        or record.get("status") != "complete"
        or code.get("value") is None
        or not code.get("benchmark_group")
    ):
        return []
    task = record["task"]["name"]
    reference = who.get("reference")
    return [
        {
            **{k: who.get(k) for k in IDENTITY_FIELDS},
            "label": reference["name"]
            if reference
            else f"{who.get('run_id')}@{who.get('step')}",
            "kind": "reference" if reference else "bench",
            "role": "candidate",
            "reference": reference,
            "packaged": packaged,
            "ms_per_token": None,
            "verdict": None,
            "final": False,
            "verdict_split": "heldout",
            "source": str(record.get("bench_id")),
            "path": str(path),
            "created_at": str(record.get("created_at")),
            "metrics": {
                "code_pass1": {
                    "value": code["value"],
                    "group": code["benchmark_group"],
                    "tasks": {task: code["value"]},
                    "items": {
                        task: [
                            1.0 if p["status"] == "passed" else 0.0
                            for p in record["problems"]
                        ]
                    },
                    "chance": None,
                }
            },
        }
    ]


def points_from_record(
    kind: str, record: Mapping[str, Any], path: Path, *, packaged: bool = False
) -> list[dict[str, Any]]:
    if kind == "bench":
        return bench_points(record, path, packaged=packaged)
    if kind == "try":
        out = try_points(record, path)
        probe = record.get("probe")
        if isinstance(probe, dict) and "format" in probe:
            out += probe_points(
                probe,
                source=str(record.get("try_id")),
                path=path,
                packaged=False,
                try_record=record,
            )
        return out
    return probe_points(
        record,
        source=str(record.get("probe_id") or path.parent.name),
        path=path,
        packaged=packaged,
    )


def collect_points(
    lab_dir: Path | None,
    *,
    references: bool = True,
    limit: int = 500,
    final: bool = False,
) -> list[dict[str, Any]]:
    """All points, newest first (packaged references last, i.e. oldest).

    Ordinary views (compare, Pareto, the Models page, enrichment) see only
    selection evidence: points from ``probe --config final`` results are excluded, so
    held-back scores can never fill in or rank an ordinary comparison. FINAL
    returns only those final points (explicit final reporting); packaged
    references have no final results.
    """
    points: list[dict[str, Any]] = []
    if lab_dir is not None and lab_dir.is_dir():
        accepted, _ = iter_lab_records(lab_dir, limit, ("try", "probe", "bench"))
        for kind, path, record in accepted:
            points += points_from_record(kind, record, path)
    points.sort(key=lambda p: p["created_at"], reverse=True)
    if references and not final:
        for kind in ("probe", "bench"):
            for path, record in packaged_reference_records(kind):
                points += points_from_record(kind, record, path, packaged=True)
    return [p for p in points if bool(p.get("final")) == final]


# Fields merged per (group, checkpoint): the measurement and every cost. A
# newer record that lacks a field (e.g. a checkpoint used as a baseline, which
# is not timed) never erases an older record's measurement of it.
MERGED_FIELDS = (
    "value",
    "window_sums",
    "window_counts",
    "items",
    "tasks",
    "chance",
    *COSTS,
    "step",
)


def metric_points(points: Iterable[Mapping[str, Any]], metric: str) -> list[dict]:
    """Flat points for one metric, one per (group, checkpoint).

    Pass points newest first. The same checkpoint measured again in the same
    group is one point: each field in MERGED_FIELDS takes the newest record
    that has it, and ``evidence`` names the record each field came from.
    Scored under another group, it is a separate point that never shares a
    chart or a comparison with this one.
    """
    merged: dict[tuple[Any, Any], dict[str, Any]] = {}
    for point in points:
        value = (point.get("metrics") or {}).get(metric)
        if not value or value.get("value") is None:
            continue
        key = (value.get("group"), point.get("checkpoint_sha256"))
        flat = {k: v for k, v in point.items() if k != "metrics"}
        flat = {**flat, **value, "metric": metric}
        current = merged.get(key)
        if current is None:
            current = merged[key] = {**flat, "evidence": {}}
            for field in MERGED_FIELDS:
                if flat.get(field) is not None:
                    current["evidence"][field] = flat["source"]
            continue
        for field in MERGED_FIELDS:
            if current.get(field) is None and flat.get(field) is not None:
                current[field] = flat[field]
                current["evidence"][field] = flat["source"]
    return list(merged.values())


def with_checkpoint_evidence(
    subject: Mapping[str, Any], pool: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """SUBJECT plus compatible evidence for its checkpoint from verified records.

    Other records that measured the same checkpoint digest (e.g. ``klab
    probe RUN --config full`` after a try) fill what the subject lacks: a metric
    the subject never measured comes from the newest record that did; a
    metric it has is only completed from records in the *same* group. Every
    metric keeps ``evidence`` (field -> record) so the provenance is shown,
    and ``by_group`` holds the checkpoint's measurement in every group it was
    scored in, so a comparison can use the one that matches the other side.
    """
    sha = subject.get("checkpoint_sha256")
    same = [p for p in pool if sha and p.get("checkpoint_sha256") == sha]
    out: dict[str, Any] = {**subject, "metrics": {}, "by_group": {}}
    for metric in METRICS:
        merged = metric_points([subject, *same], metric)
        own = (subject.get("metrics") or {}).get(metric)
        if own is not None:
            pick = next(m for m in merged if m["group"] == own.get("group"))
        elif merged:
            pick = merged[0]  # newest group measured for this checkpoint
        else:
            continue
        out["metrics"][metric] = pick
        out["by_group"][metric] = {m["group"]: m for m in merged if m["group"]}
        for cost in COSTS:
            if out.get(cost) is None and pick.get(cost) is not None:
                out[cost] = pick[cost]
    return out


def pareto_frontier(
    points: Sequence[tuple[float, float]], *, maximize: bool = False
) -> list[int]:
    """Indices of points not dominated (lower cost, better quality).

    Cost (x) is always minimized; quality (y) is minimized unless MAXIMIZE.
    """
    sign = -1.0 if maximize else 1.0
    order = sorted(
        range(len(points)), key=lambda i: (points[i][0], sign * points[i][1])
    )
    frontier: list[int] = []
    best = float("inf")
    for index in order:
        if sign * points[index][1] < best:
            frontier.append(index)
            best = sign * points[index][1]
    return frontier
