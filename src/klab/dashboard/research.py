"""Read-only catalog and static-report views for the dashboard.

This module deliberately consumes only the packaged catalog and validated report
bundles.  It neither opens run storage nor renders bundle-provided HTML/Markdown.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from klab.canonical import canonical_digest
from klab.experiments.reporting import load_report_bundle
from klab.research.catalog import list_lessons, list_research

_BUNDLE_NAME = re.compile(r"^[0-9a-f]{64}$")
_MAX_BUNDLES = 200


def _plain(value: object) -> object:
    """Convert frozen Pydantic records without relying on a Pydantic version."""
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(mode="json")
    return value


def _mapping(value: object) -> dict[str, object]:
    value = _plain(value)
    return value if isinstance(value, dict) else {}


def _rows(value: object) -> list[dict[str, object]]:
    value = _plain(value)
    if isinstance(value, dict):
        return [value]
    if not isinstance(value, (list, tuple)):
        return []
    rows = []
    for item in value:
        row = _mapping(item)
        if row:
            rows.append(row)
    return rows


def _text_list(value: object) -> str:
    if not isinstance(value, (list, tuple)):
        return ""
    return "; ".join(str(item) for item in value)


def _projection_rows(value: object) -> list[dict[str, object]]:
    """Flatten a projection's list or named stage groups for safe tables."""
    rows = _rows(value)
    if rows:
        return rows
    mapping = _mapping(value)
    flattened: list[dict[str, object]] = []
    for group, members in mapping.items():
        for member in _rows(members):
            flattened.append({"group": group, **member})
    return flattened


def _projection_table(title: str, value: object) -> None:
    """Render a lifecycle section even when its declared projection is empty."""
    st.subheader(title)
    rows = _projection_rows(value)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("No declared rows.")


def _lesson_commands(identifier: str, scale: str, data: str) -> list[str]:
    root = f"experiments/learn-{identifier}"
    return [
        f"klab learn scaffold {identifier} --scale {scale} --data {data} --backend cpu --output {root}",
        f"klab tokenizer train {root}/tokenizer.yaml",
        f"klab data prepare {root}/model.yaml",
        f"klab inspect {root}/model.yaml",
        f'klab learn probe {root}/model.yaml --prompt "Once upon a time, a small bird found a key."',
        f"klab train {root}/model.yaml --run-id learn-{identifier} --stop-after-step 2",
    ]


def learn_page() -> None:
    """Render packaged lessons without requiring a run database."""
    st.header("Learn")
    st.write(
        "Choose one mechanism, inspect its shapes, then copy the standalone commands. No campaign or run store is required to browse these lessons."
    )
    controls = st.columns(2)
    scale = controls[0].selectbox(
        "Scale", ("smoke", "nano", "micro", "tiny"), key="lesson_scale"
    )
    data = controls[1].selectbox("Data", ("offline", "tinystories"), key="lesson_data")
    search = st.text_input("Search lessons", key="lesson_search").strip().lower()

    try:
        lessons = tuple(_plain(lesson) for lesson in list_lessons())
    except (OSError, ValueError) as error:
        st.error(f"Packaged lessons could not be read: {error}")
        return

    matches = []
    for lesson in lessons:
        record = _mapping(lesson)
        searchable = " ".join(
            str(record.get(key, ""))
            for key in ("id", "title", "summary", "mechanism_kind")
        ).lower()
        if not search or search in searchable:
            matches.append(record)
    if not matches:
        st.info("No lessons match this search.")
        return

    for lesson in matches:
        identifier = str(lesson.get("id", "lesson"))
        title = str(lesson.get("title", identifier))
        with st.expander(title, expanded=bool(search)):
            st.text(str(lesson.get("summary", "")))
            st.caption(
                f"Kind: {lesson.get('mechanism_kind', 'unavailable')} · "
                f"Scope: {_text_list(lesson.get('benefit_scope'))}"
            )
            steps = _rows(lesson.get("steps"))
            if steps:
                st.subheader("Shape walkthrough")
                st.dataframe(
                    [
                        {
                            "explanation": step.get("explanation", ""),
                            "shapes": step.get("shapes", ""),
                            "observe": step.get("observe", ""),
                            "source": f"{step.get('source_path', '')}:{step.get('source_symbol', '')}",
                        }
                        for step in steps
                    ],
                    hide_index=True,
                    width="stretch",
                )
            sources = [
                f"{step.get('source_path', '')}:{step.get('source_symbol', '')}"
                for step in steps
                if step.get("source_path")
            ]
            if sources:
                st.text("Implementation sources\n" + "\n".join(sources))
            if lesson.get("try_changes"):
                st.text("Try changing\n" + _text_list(lesson["try_changes"]))
            if lesson.get("limits"):
                st.text("Limits\n" + _text_list(lesson["limits"]))
            st.text(
                "Copyable standalone commands\n"
                + "\n".join(_lesson_commands(identifier, scale, data))
            )


def discover_reports(
    reports_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    """Load at most 200 direct, content-addressed report bundles.

    The shared loader validates the manifest and all bundle contents. We do not
    inspect report-controlled links, HTML, or Markdown here.
    """
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, str]] = []
    try:
        children = sorted(reports_dir.iterdir(), key=lambda item: item.name)[
            :_MAX_BUNDLES
        ]
    except FileNotFoundError:
        return accepted, rejected
    except OSError as error:
        return accepted, [{"bundle": str(reports_dir), "reason": str(error)}]

    for child in children:
        if not child.is_dir() or child.is_symlink():
            continue
        if not _BUNDLE_NAME.fullmatch(child.name):
            rejected.append(
                {
                    "bundle": child.name,
                    "reason": "not a content-addressed bundle directory",
                }
            )
            continue
        manifest = child / "manifest.json"
        if not manifest.is_file() or manifest.is_symlink():
            rejected.append(
                {"bundle": child.name, "reason": "missing safe manifest.json"}
            )
            continue
        try:
            bundle = _mapping(load_report_bundle(child))
            if not bundle:
                raise ValueError("bundle reader returned no report")
            accepted.append({"directory": child.name, "bundle": bundle})
        except (OSError, ValueError, TypeError) as error:
            rejected.append({"bundle": child.name, "reason": str(error)})
    return accepted, rejected


def _report(record: dict[str, object]) -> dict[str, object]:
    bundle = _mapping(record.get("bundle"))
    return _mapping(bundle.get("report", bundle.get("report.json", bundle)))


def _research_entry(report: dict[str, object]) -> dict[str, object]:
    research = _mapping(report.get("research"))
    metadata = _mapping(research.get("metadata", research))
    entry = _mapping(metadata.get("entry", metadata))
    if entry:
        return entry
    experiment = _mapping(report.get("portability")).get("experiment")
    if isinstance(experiment, str):
        return next(
            (
                _mapping(_plain(candidate))
                for candidate in list_research()
                if candidate.id == experiment
            ),
            {},
        )
    return {}


def _report_title(report: dict[str, object]) -> str:
    entry = _research_entry(report)
    return str(entry.get("title") or entry.get("id") or "Unlinked historical evidence")


def _table(title: str, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if materialized:
        st.subheader(title)
        st.dataframe(pd.DataFrame(materialized), hide_index=True, width="stretch")


def _comparison_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for comparison in _rows(report.get("comparisons")):
        capabilities = _mapping(comparison.get("capabilities"))
        for card, result in capabilities.items():
            detail = _mapping(result)
            delta = detail.get("score_delta", detail.get("delta"))
            if isinstance(delta, dict):
                delta = delta.get("value")
            rows.append(
                {
                    "comparison": comparison.get(
                        "id", comparison.get("name", "comparison")
                    ),
                    "card": card,
                    "seed": detail.get("seed", comparison.get("seed", "")),
                    "observed_delta": delta,
                    "status": detail.get("status", comparison.get("status", "")),
                    "reason": detail.get("reason", comparison.get("reason", "")),
                }
            )
    return rows


def _capability_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for run in _rows(report.get("runs")):
        run_id = run.get("run_id", "")
        endpoint = _mapping(run.get("endpoint_status"))
        for card, capability in _mapping(run.get("capabilities")).items():
            detail = _mapping(capability)
            outcome = _mapping(detail.get("result", detail))
            cases = _rows(outcome.get("results"))
            if not cases:
                rows.append(
                    {
                        "run_id": run_id,
                        "endpoint_status": endpoint.get("status", "unavailable"),
                        "card": card,
                        "valid": outcome.get("valid", False),
                        "error": outcome.get("error", "case evidence unavailable"),
                    }
                )
            for case in cases:
                rows.append(
                    {
                        "run_id": run_id,
                        "endpoint_status": endpoint.get("status", "unavailable"),
                        "card": card,
                        **case,
                    }
                )
    return rows


def _card_score_rows(report: dict[str, object]) -> list[dict[str, object]]:
    rows = []
    for run in _rows(report.get("runs")):
        for card, capability in _mapping(run.get("capabilities")).items():
            detail = _mapping(capability)
            outcome = _mapping(detail.get("result", detail))
            score = outcome.get("score")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                rows.append(
                    {"run_id": run.get("run_id", ""), "card": card, "score": score}
                )
    return rows


def _portability_contrast_rows(report: dict[str, object]) -> list[dict[str, object]]:
    portability = _mapping(report.get("portability"))
    conclusions = _mapping(portability.get("conclusions"))
    rows: list[dict[str, object]] = []
    for name, raw_summary in conclusions.items():
        summary = _mapping(raw_summary)
        contrasts = _rows(summary.get("contrasts"))
        if not contrasts:
            rows.append(
                {
                    "conclusion": name,
                    "status": summary.get("status", ""),
                    "reason": summary.get("reason", ""),
                }
            )
            continue
        for contrast in contrasts:
            common = {
                "conclusion": name,
                "status": summary.get("status", ""),
                "summary_reason": summary.get("reason", ""),
                "contrast": contrast.get("name", ""),
                "metric": contrast.get("metric", ""),
                "partition": contrast.get("partition", ""),
                "matched_pairs": contrast.get("matched_pairs", ""),
                "mean_delta": contrast.get("mean_delta"),
                "contrast_status": contrast.get("status", ""),
            }
            per_seed = _rows(contrast.get("per_seed"))
            if not per_seed:
                rows.append({**common, "mean_delta": contrast.get("mean_delta")})
                continue
            for observation in per_seed:
                rows.append(
                    {
                        **common,
                        "recipient": observation.get("recipient", ""),
                        "seed": observation.get("seed", ""),
                        "left": observation.get("left"),
                        "right": observation.get("right"),
                        "delta": observation.get("delta"),
                    }
                )
    return rows


def _render_bundle(record: dict[str, object]) -> None:
    report = _report(record)
    st.subheader(_report_title(report))
    st.caption(f"Bundle: {record['directory']}")
    entry = _research_entry(report)
    if not entry:
        st.warning(
            "Unlinked historical evidence: hypothesis metadata was not recorded."
        )
    else:
        for field, heading in (
            ("question", "Question"),
            ("hypothesis", "Hypothesis"),
            ("failure_interpretation", "Failure interpretation"),
        ):
            if entry.get(field):
                st.text(f"{heading}\n{entry[field]}")
        _table(
            "Controls and varied fields",
            [
                {
                    "fixed controls": _text_list(entry.get("controls")),
                    "varied fields": _text_list(entry.get("independent_variables")),
                }
            ],
        )
        _table("Cards", _rows(entry.get("cards")))
    portability = _mapping(report.get("portability"))
    if portability:
        st.subheader("Portability campaign")
        st.json(
            {
                "experiment": portability.get("experiment"),
                "campaign_id": portability.get("campaign_id"),
                "scale": portability.get("scale"),
                "outcome_counts": portability.get("outcome_counts"),
            }
        )
        _table(
            "Portability conclusions and matched contrasts",
            _portability_contrast_rows(report),
        )
        _table("Source gates", _rows(portability.get("source_gates")))
        _table(
            "Recipient preparation gates", _rows(portability.get("preparation_gates"))
        )
        st.subheader("Campaign resource accounting")
        st.json(portability.get("costs", {}))

    research = _mapping(report.get("research"))
    if research.get("dataset_profile") is not None:
        st.subheader("Dataset and resource provenance")
        st.json(research["dataset_profile"])
    for heading, key in (
        ("Architectural quantities and cache templates", "architectural_quantities"),
        ("Implementation telemetry", "implementation_quantities"),
        ("Measured costs and unavailable values", "costs"),
        ("Local evidence validation", "local_validation"),
    ):
        value = report.get(key)
        if value:
            st.subheader(heading)
            st.json(value)
    comparisons = _comparison_rows(report)
    _table("Observed pair deltas", comparisons)
    numeric = [
        row
        for row in comparisons
        if isinstance(row.get("observed_delta"), (int, float))
        and not isinstance(row.get("observed_delta"), bool)
    ]
    if numeric:
        st.plotly_chart(
            px.bar(
                pd.DataFrame(numeric),
                x="comparison",
                y="observed_delta",
                color="card",
                hover_data=["seed", "status"],
            ),
            width="stretch",
            key=f"{record['directory']}-comparison-deltas",
        )
    capabilities = _capability_rows(report)
    _table("Raw card outcomes", capabilities)
    score_rows = _card_score_rows(report)
    if score_rows:
        st.plotly_chart(
            px.strip(
                pd.DataFrame(score_rows),
                x="card",
                y="score",
                color="card",
                hover_data=["run_id"],
            ),
            width="stretch",
            key=f"{record['directory']}-capability-scores",
        )
    _table(
        "Endpoint status",
        [
            {"run_id": run.get("run_id", ""), **_mapping(run.get("endpoint_status"))}
            for run in _rows(report.get("runs"))
        ],
    )
    _table("Missing or rejected evidence", _rows(report.get("missing_or_rejected")))
    limitations = report.get("limitations")
    _table(
        "Limitations",
        [{"limitation": item} for item in limitations]
        if isinstance(limitations, list)
        else [],
    )


def _learning_observations(
    directory: Path,
) -> tuple[list[dict[str, object]], list[str]]:
    root = directory / "learning-observations"
    accepted: list[dict[str, object]] = []
    rejected: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return accepted, rejected
    for path in sorted(root.glob("*.json"))[:_MAX_BUNDLES]:
        if (
            path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != root.resolve()
            or path.stat().st_size > 2 * 1024 * 1024
        ):
            rejected.append(f"{path.name}: unsafe path or oversized observation")
            continue
        try:
            raw = path.read_bytes()
            payload = json.loads(raw)
            if (
                not isinstance(payload, dict)
                or payload.get("format") != "klab_learning_observation_v1"
                or path.stem != hashlib.sha256(raw.rstrip(b"\n")).hexdigest()
            ):
                raise ValueError("observation format or content address mismatch")
            identity = payload.get("identity")
            content = {
                key: value for key, value in payload.items() if key != "identity"
            }
            if identity != canonical_digest(content):
                raise ValueError("observation identity mismatch")
            accepted.append(payload)
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            rejected.append(f"{path.name}: {error}")
    return accepted, rejected


def _render_learning_observations(directory: Path) -> None:
    observations, rejected = _learning_observations(directory)
    st.subheader("Learning observations")
    if rejected:
        st.warning("Some learning observations failed content-address validation.")
        _table(
            "Rejected learning observations", [{"reason": item} for item in rejected]
        )
    if not observations:
        st.info("No checkpoint-bound learning observations are available.")
        return
    rows = []
    prompt_rows = []
    status_rows = []
    for observation in observations:
        checkpoint = _mapping(observation.get("checkpoint"))
        groups = _mapping(observation.get("groups"))
        validation = _mapping(_mapping(groups.get("held_out_lm")).get("details")).get(
            "validation"
        )
        validation = _mapping(validation)
        rows.append(
            {
                "run_id": observation.get("run_id"),
                "step": checkpoint.get("step"),
                "tokens_seen": checkpoint.get("tokens_seen"),
                "validation_loss": validation.get("loss"),
                "validation_targets": validation.get("valid_targets"),
                "identity": observation.get("identity"),
            }
        )
        for group_name, result in groups.items():
            status_rows.append(
                {
                    "run_id": observation.get("run_id"),
                    "step": checkpoint.get("step"),
                    "group": group_name,
                    "status": _mapping(result).get("status"),
                }
            )
        cases = _mapping(_mapping(groups.get("fixed_prompt_panel")).get("details")).get(
            "cases"
        )
        if isinstance(cases, list):
            for case in cases:
                item = _mapping(case)
                prompt_rows.append(
                    {
                        "run_id": observation.get("run_id"),
                        "step": checkpoint.get("step"),
                        "prompt_id": item.get("id"),
                        "prompt": item.get("prompt"),
                        "output": item.get("output"),
                    }
                )
    _table("Learning curve and measured coordinates", rows)
    curve = [
        row for row in rows if isinstance(row.get("validation_loss"), (int, float))
    ]
    if curve:
        st.plotly_chart(
            px.line(
                pd.DataFrame(curve),
                x="tokens_seen",
                y="validation_loss",
                color="run_id",
            ),
            width="stretch",
            key="learning-observation-curve",
        )
    _table("Exercise group status", status_rows)
    _table("Fixed prompt trajectories", prompt_rows)
    _table(
        "Out-of-domain card outcomes and resource measurements",
        [
            {
                "run_id": observation.get("run_id"),
                "step": _mapping(observation.get("checkpoint")).get("step"),
                "capability_cards": _mapping(observation.get("groups")).get(
                    "capability_cards"
                ),
                "resource_capture": _mapping(observation.get("groups")).get(
                    "resource_capture"
                ),
            }
            for observation in observations
        ],
    )


def research_page(
    reports_dir: Path, lifecycle: Path | None = None, evidence_root: Path = Path(".")
) -> None:
    """Render read-only catalog, lifecycle declarations, and immutable reports."""
    st.header("Research")
    st.write(
        "Browse declared hypotheses, reviewed lifecycle decisions, and immutable evidence bundles. These views do not open run storage or mutate metadata."
    )
    try:
        from klab.research.lifecycle import (
            load_lifecycle,
            next_experiments,
            research_status,
        )

        registry = load_lifecycle(lifecycle)
        status = _mapping(research_status(registry, evidence_root=evidence_root))
        next_view = _mapping(next_experiments(registry, evidence_root=evidence_root))
        baseline_rows = _rows(status.get("baselines"))
        _projection_table("Reference baselines", baseline_rows)
        if baseline_rows:
            identifiers = [str(row.get("id")) for row in baseline_rows]
            selected_id = st.selectbox(
                "Baseline details", identifiers, key="research_baseline"
            )
            selected = next(
                (row for row in baseline_rows if str(row.get("id")) == selected_id),
                None,
            )
            if selected is not None:
                st.json(selected)
        funnel = _mapping(status.get("funnel"))
        st.subheader("Research funnel")
        _table(
            "Declared entries by stage",
            [
                {"stage": stage, "entries": funnel.get(stage, 0)}
                for stage in (
                    "mechanism",
                    "micro",
                    "replicate",
                    "scale",
                    "confirm",
                    "promote",
                )
            ]
            + [{"stage": "unassessed", "entries": funnel.get("unassessed", 0)}],
        )
        _projection_table("Research entry maturity", status.get("entries"))
        _projection_table("Findings", status.get("findings"))
        availability = _mapping(status.get("availability"))
        _table(
            "Evidence availability",
            [
                {"id": identity, **_mapping(value)}
                for identity, value in availability.items()
            ],
        )
        _projection_table("Scale-ready", next_view.get("scale_ready"))
        _projection_table("Blocked work", next_view.get("blocked"))
        _projection_table(
            "Prior evidence warnings",
            status.get("research_diagnostics", status.get("warnings")),
        )
        diagnostics = _projection_rows(status.get("diagnostics"))
        if diagnostics:
            if any(row.get("severity") == "error" for row in diagnostics):
                st.error(
                    "Lifecycle metadata or evidence is invalid; declarations remain visible."
                )
            else:
                st.warning(
                    "Lifecycle metadata or evidence has warnings; declarations remain visible."
                )
            _table("Lifecycle diagnostics", diagnostics)
    except (OSError, TypeError, ValueError) as error:
        st.error(f"Lifecycle metadata could not be read: {error}")
    try:
        entries = [_mapping(_plain(entry)) for entry in list_research()]
    except (OSError, ValueError) as error:
        st.error(f"Packaged research catalog could not be read: {error}")
        entries = []
    for entry in entries:
        with st.expander(str(entry.get("title", entry.get("id", "Research entry")))):
            for field, title in (
                ("question", "Question"),
                ("hypothesis", "Hypothesis"),
                ("failure_interpretation", "Failure interpretation"),
            ):
                if entry.get(field):
                    st.text(f"{title}\n{entry[field]}")
            _table(
                "Controls and varied fields",
                [
                    {
                        "fixed controls": _text_list(entry.get("controls")),
                        "varied fields": _text_list(entry.get("independent_variables")),
                    }
                ],
            )
            _table("Dataset/card applicability", _rows(entry.get("cards")))
            st.text(
                f"Recommended scale: {entry.get('recommended_scale', 'unavailable')}\nResource class: {entry.get('runtime_class', 'unavailable')}"
            )
            for paper in _rows(entry.get("papers")):
                url = paper.get("url")
                if isinstance(url, str) and url.startswith("https://"):
                    st.link_button(str(paper.get("title", url)), url)

    st.subheader("Completed reports")
    bundles, rejected = discover_reports(reports_dir)
    st.caption(
        f"Report root: {reports_dir} (direct children only; maximum {_MAX_BUNDLES})"
    )
    if rejected:
        st.warning("Some report bundles were rejected and were not rendered.")
        st.dataframe(rejected, hide_index=True, width="stretch")
    if not bundles:
        st.info("No validated report bundles are available.")
    for bundle in bundles:
        with st.expander(_report_title(_report(bundle))):
            _render_bundle(bundle)
    _render_learning_observations(reports_dir)
