"""Local, read-only Streamlit dashboard: the lab pages plus run telemetry.

The app intentionally opens SQLite through ``dashboard.queries`` rather than the
writer-side ExperimentStore, so loading a dashboard can never migrate a run store.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from importlib.resources import files
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import plotly.express as px
import streamlit as st

from klab.dashboard import generation, inspection, lab_pages, training_charts, ui
from klab.dashboard.queries import DashboardSnapshot, RunRecord, runs, snapshot
from klab.dashboard.research import learn_page, research_page
from klab.dashboard.stores import (
    available_stores,
    discovery_roots,
    parse_run_store,
)
from klab.evaluation.post_train_triage import read_triage, triage_summary
from klab.evaluation.surface_overlay import surface_review_status
from klab.training.checkpoints import CheckpointManager, _safe_member
from klab.training.metric_registry import metric_spec

_HELP_PACKAGE = "klab.dashboard"
_PILOT_PURPOSES = frozenset({"smoke", "warmup"})
_MEMORY_NATIVE = {
    "memory/device_allocated_bytes",
    "memory/device_reserved_bytes",
    "memory/device_peak_allocated_bytes",
    "memory/device_peak_reserved_bytes",
    "memory/driver_allocated_bytes",
}
_MEMORY_MLX = {
    "memory/device_allocated_bytes",
    "memory/device_peak_allocated_bytes",
    "memory/device_cache_bytes",
}
_MEMORY_SAMPLED = {
    "memory/device_sampled_peak_bytes",
    "memory/process_rss_bytes",
    "memory/process_peak_rss_bytes",
    "memory/system_available_bytes",
}
_ESTIMATE_BUCKETS = (
    ("resident_weights_bytes", "resident weights"),
    ("runtime_buffers_bytes", "runtime buffers"),
    ("gradients_bytes", "gradients"),
    ("optimizer_bytes", "optimizer state"),
    ("activations_bytes", "retained activations"),
    ("attention_working_bytes", "attention working"),
    ("workspace_bytes", "temporary workspace"),
    ("headroom_bytes", "headroom"),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--runs-dir", default="klab-work/runs")
    parser.add_argument(
        "--run-store", action="append", type=parse_run_store, default=[]
    )
    parser.add_argument("--reports-dir", default="artifacts/research-reports")
    parser.add_argument("--lifecycle")
    parser.add_argument("--evidence-root", default=".")
    parser.add_argument("--surface-dir")
    parser.add_argument("--lab-dir")
    args, _ = parser.parse_known_args()
    return args


def metric_help(slug: str) -> str:
    try:
        resource = files(_HELP_PACKAGE).joinpath("help", f"{slug}.md")
        return resource.read_text(encoding="utf-8")
    except FileNotFoundError, OSError:
        return "No packaged guide is available for this metric."


def _as_object(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _nested(value: object, *keys: str) -> object | None:
    current = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _format_bytes(value: object) -> str:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return "unavailable"
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    number = float(value)
    for unit in units:
        if abs(number) < 1024 or unit == units[-1]:
            return f"{number:.2f} {unit}"
        number /= 1024
    return "unavailable"


def _observation_age(observed_at: datetime) -> str:
    seconds = max(0, int((datetime.now(UTC) - observed_at).total_seconds()))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    return f"{seconds // 3600}h"


def _recorded_age(value: object) -> str:
    if not isinstance(value, str):
        return "unavailable"
    try:
        observed_at = datetime.fromisoformat(value)
    except ValueError:
        return "unavailable"
    if observed_at.tzinfo is None:
        # ExperimentStore's SQLite datetime('now') values are UTC.
        observed_at = observed_at.replace(tzinfo=UTC)
    return _observation_age(observed_at.astimezone(UTC))


def _filtered_runs(view: DashboardSnapshot, include_pilots: bool) -> list[RunRecord]:
    return [
        record
        for record in view.runs
        if include_pilots or record.purpose not in _PILOT_PURPOSES
    ]


def selected(view: DashboardSnapshot) -> tuple[list[RunRecord], list[str]]:
    records = _filtered_runs(view, st.session_state.get("include_pilot_runs", False))
    ids = {record.run_id for record in records}
    selected_ids = [
        run_id for run_id in st.session_state.get("selected_runs", []) if run_id in ids
    ]
    return records, selected_ids


def _selection_controls(root: Path) -> None:
    """Entrypoint-owned widgets retain identity across Streamlit pages."""
    key = f"sidebar_records:{root.resolve()}"
    try:
        records = runs(root)
        st.session_state[key] = records
    except (OSError, sqlite3.Error, ValueError) as error:
        records = st.session_state.get(key, [])
        st.sidebar.warning(
            f"Run list unavailable; cached selection may be stale: {error}"
        )
    include_pilots = st.sidebar.checkbox(
        "Include smoke and warmup pilots", value=False, key="include_pilot_runs"
    )
    ids = [
        record.run_id
        for record in records
        if include_pilots or record.purpose not in _PILOT_PURPOSES
    ]
    active = [
        record.run_id
        for record in records
        if record.status.lower() == "running" and record.run_id in ids
    ]
    requested = st.query_params.get("run")
    request_key = (str(root), requested)
    if requested and st.session_state.get("dashboard_link_applied") != request_key:
        if requested in ids:
            st.session_state["selected_runs"] = [requested]
            st.session_state["dashboard_link_applied"] = request_key
        else:
            st.sidebar.warning(
                "Linked run is unavailable or hidden in this store. No substitute selected."
            )
            st.session_state["selected_runs"] = []
    if active:
        if st.sidebar.button("Focus recorded running runs"):
            st.session_state["selected_runs"] = active
            st.query_params.pop("run", None)
    else:
        st.sidebar.info("No runs recorded as running in this store.")
    previous = st.session_state.get("selected_runs", active or ids[:1])
    st.session_state["selected_runs"] = [run_id for run_id in previous if run_id in ids]
    names = {
        record.run_id: f"{record.name} · {record.status} · {record.run_id}"
        for record in records
    }
    st.sidebar.multiselect("Runs", ids, key="selected_runs", format_func=names.get)
    for record in records:
        if record.run_id in st.session_state["selected_runs"]:
            st.sidebar.caption(
                f"{record.name} · Recorded {record.status} · Updated {record.updated_at} UTC"
            )
            st.sidebar.markdown(
                f"[Open this run](training?{urlencode({'store': str(root), 'run': record.run_id})})"
            )
    st.sidebar.caption(
        "Stored status is not a process heartbeat. Verify activity before calling a run live."
    )


def _store_controls(primary: Path, extra: list[tuple[str, str]]) -> Path:
    """Switch whole read-only projections, keeping selections scoped by directory."""
    stores, warnings = available_stores(primary, extra)
    previous_label = st.session_state.get("dashboard_run_store")
    previous_path = st.session_state.get("dashboard_active_store")
    if previous_label and previous_label not in stores and previous_path:
        stores[previous_label] = Path(previous_path)
        st.sidebar.warning(
            "Selected store is no longer discoverable; selection retained."
        )
    for warning in warnings:
        st.sidebar.warning(warning)
    requested_store = st.query_params.get("store")
    if (
        requested_store
        and st.session_state.get("dashboard_store_link_applied") != requested_store
    ):
        match = next(
            (label for label, path in stores.items() if str(path) == requested_store),
            None,
        )
        if match is None:
            st.error(
                "Linked run store is unavailable. Configure or restore that store; no substitute is shown."
            )
            st.stop()
        st.session_state["dashboard_run_store"] = match
        previous_label = match
        st.session_state["dashboard_store_link_applied"] = requested_store
    labels = list(stores)
    label = (
        st.sidebar.selectbox(
            "Run store",
            labels,
            index=labels.index(previous_label) if previous_label in labels else 0,
            key="dashboard_run_store",
        )
        if len(stores) > 1
        else next(iter(stores))
    )
    root = stores[label]
    identity = str(root)
    previous = st.session_state.get("dashboard_active_store")
    if previous != identity:
        selections = st.session_state.setdefault("dashboard_store_selections", {})
        if previous is not None and "selected_runs" in st.session_state:
            selections[previous] = list(st.session_state["selected_runs"])
        if identity in selections:
            st.session_state["selected_runs"] = selections[identity]
        else:
            st.session_state.pop("selected_runs", None)
        # These widgets name values/generations in one store, not global identities.
        for key in ("training_metrics", "inspect_run", "inspect_checkpoint"):
            st.session_state.pop(key, None)
        st.session_state["dashboard_active_store"] = identity
    if len(stores) > 1:
        st.sidebar.caption(
            "Telemetry uses the selected store. Lab records remain separate."
        )
    return root


def _rows_for(
    rows: tuple[dict[str, object], ...], selected_ids: list[str]
) -> list[dict[str, object]]:
    wanted = set(selected_ids)
    return [row for row in rows if row["run_id"] in wanted]


def architecture(record: RunRecord) -> dict[str, str]:
    return {
        "attention": str(_nested(record.config, "attention", "kind") or "unavailable"),
        "dataset": str(_nested(record.config, "dataset", "source") or "unavailable"),
        "ffn": str(_nested(record.config, "model", "ffn") or "unavailable"),
        "memory": str(_nested(record.config, "model", "memory") or "unavailable"),
        "purpose": record.purpose or "unavailable",
    }


def _runtime(record: RunRecord, view: DashboardSnapshot) -> dict[str, object]:
    manifest = view.manifests.get(record.run_id, {})
    return _as_object(record.metadata.get("runtime")) or _as_object(
        manifest.get("runtime")
    )


def _requested_precision(record: RunRecord) -> object:
    return _nested(record.config, "runtime", "precision") or "unavailable"


def _effective_precision(record: RunRecord, runtime: dict[str, object]) -> object:
    return runtime.get("precision") or _requested_precision(record)


def _optimizer_semantics(record: RunRecord) -> str:
    optimizer = _nested(record.config, "optimizer", "name")
    if optimizer == "adafactor":
        return "Adafactor relative learning-rate cap; not numerically AdamW LR"
    if optimizer == "adamw":
        return "AdamW update learning rate"
    return "unavailable"


def _mlx_run_ids(view: DashboardSnapshot) -> set[str]:
    return {
        record.run_id
        for record in view.runs
        if _runtime(record, view).get("engine") == "mlx"
    }


def _estimate(record: RunRecord) -> dict[str, object]:
    return _as_object(record.metadata.get("memory_estimate"))


def _comparison_rows(
    records: list[RunRecord], view: DashboardSnapshot
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for record in records:
        runtime = _runtime(record, view)
        rows.append(
            {
                "run_id": record.run_id,
                "purpose": record.purpose or "unavailable",
                "engine": runtime.get("engine")
                or _nested(record.config, "runtime", "engine")
                or "unavailable",
                "backend": runtime.get("backend")
                or _nested(record.config, "runtime", "backend")
                or "unavailable",
                "requested_precision": _requested_precision(record),
                "effective_precision": _effective_precision(record, runtime),
                "optimizer": _nested(record.config, "optimizer", "name")
                or "unavailable",
                "optimizer_semantics": _optimizer_semantics(record),
                "sequence_length": _nested(record.config, "training", "seq_len")
                or "unavailable",
                "micro_batch": _nested(record.config, "training", "micro_batch_size")
                or "unavailable",
                "accumulation": _nested(
                    record.config, "training", "gradient_accumulation"
                )
                or "unavailable",
                "dataset": _nested(record.config, "dataset", "source") or "unavailable",
                "tokenizer": _nested(record.config, "tokenizer", "path")
                or "unavailable",
            }
        )
    return rows


def comparison(records: list[RunRecord], view: DashboardSnapshot) -> None:
    if len(records) < 2:
        return
    rows = _comparison_rows(records, view)
    changed = [
        key
        for key in rows[0]
        if key != "run_id" and len({str(row[key]) for row in rows}) > 1
    ]
    st.subheader("Comparison conditions")
    st.caption(
        "Comparisons are descriptive unless these recorded scientific and runtime conditions match. "
        "No values are interpolated or normalized."
    )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    if changed:
        st.warning("Different conditions: " + ", ".join(changed))
    else:
        st.success(
            "Displayed recorded conditions match; compare only observed budget points."
        )


def overview(view: DashboardSnapshot, root: Path | None = None) -> None:
    st.header("Overview")
    records, selected_ids = selected(view)
    if root is not None:
        generation.checkpoint_summary(root, selected_ids, view)
    if not records:
        st.info("No eligible runs yet. Smoke and warmup pilots are hidden by default.")
        return
    st.dataframe(
        [
            {
                "run_id": item.run_id,
                "name": item.name,
                "status": item.status,
                "parent": item.parent_run_id or "unavailable",
                "updated_at": item.updated_at,
                **architecture(item),
            }
            for item in records
        ],
        width="stretch",
        hide_index=True,
    )
    comparison([record for record in records if record.run_id in selected_ids], view)
    selected_events = _rows_for(view.events, selected_ids)
    if selected_events:
        st.subheader("Events")
        st.dataframe(selected_events, width="stretch", hide_index=True)


def _runtime_forecast_panel(view: DashboardSnapshot, selected_ids: list[str]) -> None:
    progress_rows = _rows_for(view.runtime_progress, selected_ids)
    final_by_run: dict[str, dict[str, object]] = {}
    for event in _rows_for(view.events, selected_ids):
        if event.get("kind") != "runtime_final_observation":
            continue
        try:
            decoded = json.loads(str(event.get("payload_json", "{}")))
        except json.JSONDecodeError:
            continue
        if isinstance(decoded, dict):
            final_by_run[str(event["run_id"])] = decoded
    if not progress_rows and not final_by_run:
        return

    latest_by_run = {str(row["run_id"]): row for row in progress_rows}
    run_ids = sorted(set(latest_by_run) | set(final_by_run))
    summaries: list[dict[str, object]] = []
    detail: dict[str, dict[str, object]] = {}
    for run_id in run_ids:
        row = latest_by_run.get(run_id)
        payload: dict[str, object] = {}
        if row is not None:
            try:
                decoded = json.loads(str(row.get("payload_json", "{}")))
            except json.JSONDecodeError:
                decoded = {}
            if isinstance(decoded, dict):
                payload = decoded
        derived = _as_object(payload.get("derived"))
        live = _as_object(derived.get("live"))
        fraction = live.get("progress_fraction")
        summaries.append(
            {
                "run_id": run_id,
                "state": live.get("state", payload.get("state", "unavailable")),
                "phase": payload.get("phase", "unavailable"),
                "completed_targets": live.get("completed_targets"),
                "total_targets": live.get("total_targets"),
                "progress_percent": (
                    round(float(fraction) * 100, 1)
                    if isinstance(fraction, (int, float))
                    else None
                ),
                "recent_targets_per_second": live.get("recent_targets_per_second"),
                "long_targets_per_second": live.get("long_targets_per_second"),
                "optimizer_only_eta_low_seconds": live.get("eta_low_seconds"),
                "optimizer_only_eta_high_seconds": live.get("eta_high_seconds"),
                "optimizer_only_eta_status": live.get("eta_status", "unavailable"),
            }
        )
        detail[run_id] = {
            "latest_progress": payload or None,
            "final_observed": final_by_run.get(run_id),
        }

    st.subheader("Runtime forecast")
    st.dataframe(summaries, width="stretch", hide_index=True)
    st.caption(
        "ETA is optimizer-only and remains unavailable without compatible timing telemetry. "
        "Final phase observations keep preparation, validation, checkpointing, and reporting separate."
    )
    with st.expander("Runtime record details"):
        st.json(detail, expanded=False)


def _triage_panel(
    root: Path, selected_ids: list[str], surface_dir: Path | None = None
) -> None:
    st.subheader("Post-train triage (read-only)")
    for run_id in selected_ids:
        try:
            report = read_triage(run_id, root)
        except (OSError, ValueError) as error:
            st.warning(f"{run_id}: UNKNOWN — invalid triage artifact: {error}")
            continue
        if report is None:
            st.info(f"{run_id}: UNKNOWN — no verified post-train triage artifact")
            continue
        st.write(f"{run_id}: {triage_summary(report)}")
        if surface_dir is not None:
            overlay = surface_review_status(run_id, root, surface_dir)
            st.write(
                f"{run_id}: independent subjective quality — {overlay['independent_subjective_quality']}"
            )
            for bundle in overlay["bundles"]:
                st.caption(f"{bundle['status']}: {bundle['path']}")


def training(
    view: DashboardSnapshot, root: Path | None = None, surface_dir: Path | None = None
) -> None:
    ui.page_header("Training", "How is the current run progressing?")
    records, selected_ids = selected(view)
    inspection.learning_summary(view, selected_ids)
    inspection.engram_summary(view, selected_ids, compact=True)
    _runtime_forecast_panel(view, selected_ids)
    comparison([record for record in records if record.run_id in selected_ids], view)
    if root is not None:
        _triage_panel(root, selected_ids, surface_dir)
    points = _rows_for(view.metrics, selected_ids)
    if not points:
        st.info("Select a run with recorded metrics.")
        return
    comparison_condition = st.selectbox(
        "Comparison condition",
        ("descriptive", "equal_tokens", "equal_steps", "equal_wall_time"),
        help="This labels the comparison claim; it does not interpolate observations.",
        key="training_comparison_condition",
    )
    st.caption(
        "Descriptive comparison only."
        if comparison_condition == "descriptive"
        else f"Showing observed points for `{comparison_condition}`; no controlled conclusion is inferred."
    )
    axis = st.radio(
        "X axis",
        ("step", "tokens_seen", "wall_time"),
        horizontal=True,
        key="training_axis",
    )
    names = sorted({str(point["name"]) for point in points})
    selected_names = st.multiselect(
        "Metrics",
        names,
        default=[
            name
            for name in (
                "train/loss",
                "validation/loss",
                "performance/tokens_per_second",
            )
            if name in names
        ],
        key="training_metrics",
        help="Start with loss and throughput. Add any recorded metric for deeper inspection.",
    )
    for name in selected_names:
        subset = [point for point in points if point["name"] == name]
        if not subset:
            continue
        st.subheader(name.replace("/", " / ").replace("_", " "))
        training_charts.render(subset, name, axis, str(root), selected_ids)
        spec = metric_spec(name)
        if spec is not None:
            with st.expander(f"More about {name}"):
                st.markdown(metric_help(spec.help_slug))


def architecture_diagnostics(view: DashboardSnapshot) -> None:
    st.header("Architecture diagnostics")
    st.caption("For per-port memory values, use Engram memory in the navigation.")
    _, selected_ids = selected(view)
    points = _rows_for(view.metrics, selected_ids)
    for title, prefix in {
        "MoE": "moe/",
        "Engram": "engram/",
        "Attention": "attention/",
    }.items():
        subset = [point for point in points if str(point["name"]).startswith(prefix)]
        st.subheader(title)
        if not subset:
            st.info(f"No persisted {title} diagnostics for the selected runs.")
            continue
        st.plotly_chart(
            px.line(
                subset,
                x="step",
                y="value",
                color="run_id",
                facet_row="name",
                markers=True,
            )
            .update_yaxes(matches=None)
            .update_layout(
                height=max(350, 220 * len({point["name"] for point in subset}))
            ),
            width="stretch",
        )


def evaluation(view: DashboardSnapshot) -> None:
    st.header("Evaluation")
    _, selected_ids = selected(view)
    points = [
        point
        for point in _rows_for(view.metrics, selected_ids)
        if str(point["name"]).startswith("validation/")
    ]
    if not points:
        st.info("No persisted evaluation metrics are available for the selected runs.")
        return
    st.plotly_chart(
        px.line(
            points, x="step", y="value", color="run_id", facet_row="name", markers=True
        )
        .update_yaxes(matches=None)
        .update_layout(height=max(350, 220 * len({point["name"] for point in points}))),
        width="stretch",
    )


def runtime_view(view: DashboardSnapshot) -> None:
    st.header("Runtime")
    records, selected_ids = selected(view)
    _runtime_forecast_panel(view, selected_ids)
    rows = []
    for record in records:
        if record.run_id not in selected_ids:
            continue
        runtime = _runtime(record, view)
        rows.append(
            {
                "run_id": record.run_id,
                "worker": view.manifests.get(record.run_id, {}).get(
                    "worker_id", "unavailable"
                ),
                "engine": runtime.get("engine", "unavailable"),
                "backend": runtime.get("backend", "unavailable"),
                "device": runtime.get("device_name")
                or runtime.get("torch_device")
                or "unavailable",
                "physical_device_id": runtime.get("physical_device_id", "unavailable"),
                "os": runtime.get("os", "unavailable"),
                "requested_precision": _requested_precision(record),
                "effective_precision": _effective_precision(record, runtime),
                "optimizer": _nested(record.config, "optimizer", "name")
                or "unavailable",
                "optimizer_semantics": _optimizer_semantics(record),
                "declared_precisions": runtime.get(
                    "precision_capabilities", "unavailable"
                ),
                "tested_precisions": runtime.get("tested_precisions", "unavailable"),
                "tested_features": runtime.get("tested_features", "unavailable"),
                "validated_at": runtime.get("validated_at", "unavailable"),
                "framework": runtime.get("framework_version")
                or runtime.get("torch_version")
                or "unavailable",
                "system_total": _format_bytes(runtime.get("system_total_bytes")),
                "system_available": _format_bytes(
                    runtime.get("system_available_bytes")
                ),
                "device_total": _format_bytes(runtime.get("device_total_bytes")),
                "device_free": _format_bytes(runtime.get("device_free_bytes")),
                "device_recommended": _format_bytes(
                    runtime.get("device_recommended_bytes")
                ),
                "measurement_source": runtime.get("measurement_source", "unavailable"),
            }
        )
    if not rows:
        st.info(
            "Runtime metadata is unavailable for these legacy or not-yet-registered runs."
        )
        return
    st.dataframe(rows, width="stretch", hide_index=True)
    st.caption(
        "Unified-memory device capacity and system RAM are reported separately and are not added together."
    )
    with st.expander("Runtime and precision guide"):
        st.markdown(metric_help("runtime"))


def memory_view(view: DashboardSnapshot) -> None:
    st.header("Memory")
    records, selected_ids = selected(view)
    values = [
        point
        for point in _rows_for(view.metrics, selected_ids)
        if str(point["name"]).startswith("memory/")
    ]
    mlx_run_ids = _mlx_run_ids(view)
    mlx_native = [
        point
        for point in values
        if point["run_id"] in mlx_run_ids and point["name"] in _MEMORY_MLX
    ]
    native = [
        point
        for point in values
        if point["name"] in _MEMORY_NATIVE
        and not (point["run_id"] in mlx_run_ids and point["name"] in _MEMORY_MLX)
    ]
    sampled = [point for point in values if point["name"] in _MEMORY_SAMPLED]
    if native:
        st.subheader("Native allocator readings")
        st.plotly_chart(
            px.line(native, x="step", y="value", color="run_id", facet_row="name")
            .update_yaxes(matches=None)
            .update_layout(
                height=max(350, 220 * len({point["name"] for point in native}))
            ),
            width="stretch",
        )
    else:
        st.info(
            "Native allocator readings are unavailable for the selected non-MLX runs."
        )
    if mlx_native:
        st.subheader("MLX Metal memory")
        st.plotly_chart(
            px.line(mlx_native, x="step", y="value", color="run_id", facet_row="name")
            .update_yaxes(matches=None)
            .update_layout(
                height=max(350, 220 * len({point["name"] for point in mlx_native}))
            ),
            width="stretch",
        )
        st.caption(
            "MLX active, peak, and cache memory are Metal framework measurements. "
            "Cache memory is neither allocator-reserved nor driver-allocated memory."
        )
    else:
        st.info("MLX Metal memory is unavailable for the selected runs.")
    if sampled:
        st.subheader("Sampled and host readings")
        st.plotly_chart(
            px.line(sampled, x="step", y="value", color="run_id", facet_row="name")
            .update_yaxes(matches=None)
            .update_layout(
                height=max(350, 220 * len({point["name"] for point in sampled}))
            ),
            width="stretch",
        )
        st.caption(
            "Sampled device peaks are lower bounds; host RSS is not added to device capacity."
        )
    else:
        st.info("Sampled memory readings are unavailable for the selected runtime.")
    estimates = []
    for record in records:
        if record.run_id not in selected_ids:
            continue
        estimate = _estimate(record)
        if estimate:
            estimates.append(
                {
                    "run_id": record.run_id,
                    **{
                        label: _format_bytes(estimate.get(key))
                        for key, label in _ESTIMATE_BUCKETS
                    },
                    "result": estimate.get("result", "unavailable"),
                }
            )
    st.subheader("Disjoint estimated memory")
    if estimates:
        st.dataframe(estimates, width="stretch", hide_index=True)
    else:
        st.info("No persisted memory estimate is available for these runs.")
    with st.expander("Memory guide"):
        st.markdown(metric_help("memory"))


def _current_stages(
    rows: list[dict[str, object]], records: list[RunRecord]
) -> list[dict[str, object]]:
    latest: dict[str, dict[str, object]] = {}
    for row in rows:
        run_id = str(row["run_id"])
        if run_id not in latest or int(row["sequence"]) > int(
            latest[run_id]["sequence"]
        ):
            latest[run_id] = row
    return [
        {
            "run_id": record.run_id,
            "run_status": record.status,
            "current_stage": latest.get(record.run_id, {}).get("stage", "unavailable"),
            "stage_status": latest.get(record.run_id, {}).get("status", "unavailable"),
            "step": latest.get(record.run_id, {}).get("step", "unavailable"),
            "tokens_seen": latest.get(record.run_id, {}).get(
                "tokens_seen", "unavailable"
            ),
            "started_at": latest.get(record.run_id, {}).get(
                "started_at", "unavailable"
            ),
            "last_run_update": record.updated_at,
            "last_run_update_age": _recorded_age(record.updated_at),
        }
        for record in records
    ]


def stage_view(view: DashboardSnapshot) -> None:
    st.header("Stages")
    records, selected_ids = selected(view)
    selected_records = [record for record in records if record.run_id in selected_ids]
    rows = _rows_for(view.stages, selected_ids)
    current = _current_stages(rows, selected_records)
    st.subheader("Current stage")
    if current:
        st.dataframe(current, width="stretch", hide_index=True)
        st.caption(
            "A silent RUNNING row remains RUNNING; the dashboard never infers completion "
            "from its age or from the presence of a checkpoint."
        )
    else:
        st.info("Select a run to show its recorded stage status.")
    st.subheader("Append-only stage intervals")
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.info("No stage intervals are persisted for the selected runs.")


def _checkpoint_directory(root: Path, row: dict[str, object]) -> Path:
    run_id, relative = row["run_id"], row["relative_path"]
    if (
        not isinstance(run_id, str)
        or Path(run_id).name != run_id
        or not isinstance(relative, str)
        or Path(relative).name != relative
    ):
        raise ValueError("unsafe projected checkpoint path")
    run = _safe_member(root, run_id)
    checkpoints = _safe_member(run, "checkpoints") if run is not None else None
    directory = _safe_member(checkpoints, relative) if checkpoints is not None else None
    if directory is None:
        raise ValueError("unsafe projected checkpoint path")
    return directory


def checkpoint_view(view: DashboardSnapshot, root: Path) -> None:
    st.header("Checkpoints")
    records, selected_ids = selected(view)
    generation.checkpoint_summary(root, selected_ids, view)
    rows = _rows_for(view.checkpoints, selected_ids)
    st.caption(
        "Stored verification describes save time, not current disk integrity. "
        "Local verification is read-only and timestamped; recheck after files change."
    )
    key = f"checkpoint_verification:{root.resolve()}"
    verified = st.session_state.setdefault(key, {})
    if rows and st.button("Verify selected checkpoint files"):
        with st.spinner("Verifying local checkpoint files"):
            for row in rows:
                identity = (row["run_id"], row["checkpoint_id"], row["digest"])
                try:
                    directory = _checkpoint_directory(root, row)
                    result = CheckpointManager(directory.parent.parent).verify(
                        directory,
                        require_training_state=row["resume_level"] != "weights_only",
                    )
                    valid, errors = result.valid, list(result.errors)
                    if valid:
                        manifest = json.loads((directory / "manifest.json").read_text())
                        if manifest.get("sha256") != row["digest"]:
                            valid = False
                            errors.append(
                                {
                                    "field": "projection",
                                    "reason": "checkpoint digest mismatch",
                                }
                            )
                except (OSError, ValueError, TypeError) as error:
                    valid, errors = False, [{"field": "path", "reason": str(error)}]
                verified[identity] = {
                    "session_verification": "verified" if valid else "invalid",
                    "checked_at": datetime.now(UTC).isoformat(),
                    "verification_errors": json.dumps(errors) if errors else None,
                }
    rows = [
        {
            "run_id": row["run_id"],
            "step": row["step"],
            **verified.get(
                (row["run_id"], row["checkpoint_id"], row["digest"]),
                {
                    "session_verification": "not checked",
                    "checked_at": None,
                    "verification_errors": None,
                },
            ),
            **row,
        }
        for row in rows
    ]
    if rows:
        checked = sum(row["session_verification"] != "not checked" for row in rows)
        invalid = sum(row["session_verification"] == "invalid" for row in rows)
        st.caption(
            f"Local file checks: {checked}/{len(rows)} checked; "
            f"{checked - invalid} verified; {invalid} invalid."
        )
        if invalid:
            st.error(f"{invalid} checkpoint generation(s) failed local verification.")
        st.dataframe(
            rows,
            width="stretch",
            hide_index=True,
            column_config={
                "checkpoint_id": st.column_config.NumberColumn(
                    "Save sequence (within run)",
                    help="Ordinal save number, not a checkpoint count.",
                )
            },
        )
        local_best: list[dict[str, object]] = []
        for run_id in selected_ids:
            candidates = [
                row
                for row in rows
                if row["run_id"] == run_id
                and row["verification_status"] == "verified"
                and row["session_verification"] == "verified"
                and isinstance(row["validation_loss"], (int, float))
            ]
            if candidates:
                best = min(
                    candidates,
                    key=lambda row: (
                        float(row["validation_loss"]),
                        int(row["step"]),
                        str(row["checkpoint_id"]),
                    ),
                )
                local_best.append(
                    {
                        "run_id": run_id,
                        "local_best_checkpoint": best["checkpoint_id"],
                        "validation_loss": best["validation_loss"],
                        "available_locally": True,
                        "verified_locally_at": best["checked_at"],
                    }
                )
            else:
                local_best.append(
                    {
                        "run_id": run_id,
                        "local_best_checkpoint": None,
                        "validation_loss": None,
                        "available_locally": None,
                        "verified_locally_at": None,
                    }
                )
        st.subheader("Local best")
        st.caption(
            "Only generations verified in this session are eligible; blank means no verified candidate."
        )
        st.dataframe(local_best, width="stretch", hide_index=True)
    else:
        st.info(
            "No checkpoint projection is available for these legacy or not-yet-checkpointed runs."
        )
    parents = []
    metadata_by_run = {record.run_id: record.metadata for record in records}
    for run_id in selected_ids:
        manifest = view.manifests.get(run_id, {})
        parents.append(
            {
                "run_id": run_id,
                "continuation_kind": manifest.get("continuation_kind", "unavailable"),
                "parent_run_id": manifest.get("parent_run_id"),
                "parent_checkpoint_sha256": manifest.get("checkpoint_sha256")
                or manifest.get("parent_checkpoint_sha256")
                or _as_object(metadata_by_run.get(run_id)).get("checkpoint_sha256"),
            }
        )
    st.subheader("Parent checkpoint identity")
    st.dataframe(parents, width="stretch", hide_index=True)
    lineage = []
    for record in records:
        if record.run_id not in selected_ids:
            continue
        value = record.metadata.get("lineage_best")
        if isinstance(value, dict):
            lineage.append(
                {
                    "run_id": record.run_id,
                    **value,
                    "availability": "ancestor metadata; local generation may be unavailable",
                }
            )
    st.subheader("Ancestor lineage best")
    if lineage:
        st.dataframe(lineage, width="stretch", hide_index=True)
    else:
        st.info("No ancestor lineage-best metadata is recorded.")
    with st.expander("Checkpoint guide"):
        st.markdown(metric_help("checkpoints"))


def learn() -> None:
    learn_page()
    st.subheader("Runtime guides")
    guides = {
        "Runtime and precision": "runtime",
        "Training memory": "memory",
        "Activation recomputation": "recomputation",
        "Optimizer and learning rate": "learning-rate",
        "Gradient accumulation": "accumulation",
        "Disk checkpoints and lineage": "checkpoints",
        "Activation offload": "offload",
        "Independent workers boundary": "independent-workers",
    }
    search = (
        st.text_input("Search runtime guides", key="learn_guide_search").strip().lower()
    )
    for title, slug in guides.items():
        body = metric_help(slug)
        if search and search not in title.lower() and search not in body.lower():
            continue
        with st.expander(title, expanded=bool(search)):
            st.markdown(body)


def _read_snapshot(
    root: Path,
) -> tuple[DashboardSnapshot | None, datetime | None, str | None]:
    run_ids = tuple(st.session_state.get("selected_runs", []))
    key = f"dashboard_snapshot:{root.resolve()}:{run_ids}"
    try:
        fresh = snapshot(root, run_ids)
    except (OSError, sqlite3.Error, ValueError) as error:
        previous = st.session_state.get(key)
        if previous is None:
            return None, None, str(error)
        view, observed_at = previous
        return view, observed_at, str(error)
    observed_at = datetime.now(UTC)
    st.session_state[key] = (fresh, observed_at)
    return fresh, observed_at, None


def render(root: Path, page: str, surface_dir: Path | None = None) -> None:
    if page == "generate":
        generation.generation_page(None, root, [])
        return
    st.button(
        "Refresh now",
        key="dashboard_refresh",
        help="Read the current recorded values without reloading the page.",
    )
    view, observed_at, error = _read_snapshot(root)
    if view is None:
        st.error(f"Dashboard could not read the run store: {error}")
        return
    assert observed_at is not None
    if error:
        st.error(f"Showing stale data ({_observation_age(observed_at)} old): {error}")
    else:
        st.caption(
            f"Observed at {observed_at:%H:%M:%S} UTC · "
            + (
                f"Refreshes every {st.session_state.get('refresh_seconds', 10)} seconds."
                if page != "generate" and st.session_state.get("refresh_seconds", 10)
                else "Automatic refresh paused."
            )
        )
    _, focused_ids = selected(view)
    for record in view.runs:
        if record.run_id in focused_ids:
            with st.container(border=True):
                st.markdown(f"**{record.name}** · Recorded status: **{record.status}**")
                st.caption(f"Run {record.run_id} · Updated {record.updated_at} UTC")
                if record.status.lower() == "running":
                    st.caption(
                        "Activity unverified: stored running status is not a live process heartbeat."
                    )
    pages = {
        "generate": lambda value: generation.generation_page(
            value, root, selected(value)[1]
        ),
        "overview": lambda value: overview(value, root),
        "training": lambda value: training(value, root, surface_dir),
        "evaluation": evaluation,
        "architecture": architecture_diagnostics,
        "engram": lambda value: inspection.engram_summary(value, selected(value)[1]),
        "snapshots": lambda value: inspection.snapshot_inspector(
            value, selected(value)[1]
        ),
        "runtime": runtime_view,
        "memory": memory_view,
        "checkpoints": lambda value: checkpoint_view(value, root),
        "stages": stage_view,
    }
    pages[page](view)


def _run_page(root: Path, page: str, surface_dir: Path | None = None) -> None:
    """Keep run views unavailable until a read-only projection exists."""
    run_ids = tuple(st.session_state.get("selected_runs", []))
    key = f"dashboard_snapshot:{root.resolve()}:{run_ids}"
    if not (root / "experiments.sqlite3").is_file() and key not in st.session_state:
        st.info("No runs yet. Run `klab train CONFIG --run-id NAME`.")
        return
    if page == "training":
        training_charts.load_plotly()
    if page == "generate":
        render(root, page, surface_dir)
        return
    interval = st.session_state.get("refresh_seconds", 10)
    st.fragment(run_every=interval or None)(render)(root, page, surface_dir)


def main() -> None:
    st.set_page_config(page_title="Kybern Model Lab", page_icon="🔬", layout="wide")
    ui.style()
    args = arguments()
    try:
        root = _store_controls(Path(args.runs_dir), args.run_store)
    except ValueError as error:
        st.error(str(error))
        st.stop()
    if discovery_roots():

        @st.fragment(run_every=60)
        def watch_stores():
            stores, warnings = available_stores(Path(args.runs_dir), args.run_store)
            signature = (tuple(stores.items()), warnings)
            previous = st.session_state.get("store_discovery_signature")
            st.session_state["store_discovery_signature"] = signature
            st.caption("New run stores are checked every minute.")
            if previous is not None and previous != signature:
                st.rerun()

        with st.sidebar:
            watch_stores()
    reports_dir = Path(args.reports_dir)
    lifecycle = Path(args.lifecycle) if args.lifecycle else None
    evidence_root = Path(args.evidence_root)
    surface_dir = Path(args.surface_dir) if args.surface_dir else None
    lab_dir = (
        Path(args.lab_dir)
        if args.lab_dir
        else Path(args.runs_dir).expanduser().resolve().parent / "lab"
    )
    runs_dirs = [root, lab_dir / "runs"]
    pages: dict[str, Any] = {}
    lab = [
        st.Page(
            lambda: lab_pages.home_page(lab_dir, runs_dirs, pages),
            title="Home",
            icon=":material/home:",
            url_path="home",
            default=True,
        ),
        st.Page(
            lambda: lab_pages.experiments_page(lab_dir),
            title="Experiments",
            icon=":material/science:",
            url_path="experiments",
        ),
        st.Page(
            lambda: lab_pages.models_page(lab_dir, runs_dirs),
            title="Models",
            icon=":material/scatter_plot:",
            url_path="models",
        ),
        st.Page(
            lambda: lab_pages.behaviors_page(lab_dir),
            title="Behaviors",
            icon=":material/chat:",
            url_path="behaviors",
        ),
        st.Page(
            lambda: lab_pages.explorer_page(lab_dir, runs_dirs),
            title="Explorer",
            icon=":material/account_tree:",
            url_path="explorer",
        ),
    ]
    pages.update({p.url_path: p for p in lab})
    run_pages = [
        st.Page(
            (lambda name=name: _run_page(root, name, surface_dir))
            if name == "training"
            else (lambda name=name: _run_page(root, name)),
            title={
                "engram": "Engram memory",
                "snapshots": "Snapshot inspector",
                "memory": "Device memory",
            }.get(name, name.title()),
            icon={
                "training": ":material/monitoring:",
                "engram": ":material/memory:",
                "snapshots": ":material/manage_search:",
                "checkpoints": ":material/save:",
            }.get(name, ":material/analytics:"),
            url_path=name,
        )
        for name in (
            "overview",
            "training",
            "generate",
            "engram",
            "snapshots",
            "evaluation",
            "architecture",
            "runtime",
            "memory",
            "checkpoints",
            "stages",
        )
    ]
    page = st.navigation(
        {
            "Lab": lab,
            "Run telemetry": run_pages,
            "Learn": [
                st.Page(lambda: learn(), title="Learn", url_path="learn"),
                st.Page(
                    lambda: research_page(reports_dir, lifecycle, evidence_root),
                    title="Research",
                    url_path="research",
                ),
            ],
        }
    )
    st.sidebar.markdown("### 🔬 Kybern Model Lab")
    st.sidebar.caption("Observe training · Inspect evidence · Compare models")
    with st.sidebar.expander("Data locations"):
        st.caption(f"lab {lab_dir}  \nruns {root}")
    _selection_controls(root)
    if page.url_path in {item.url_path for item in run_pages}:
        st.sidebar.selectbox(
            "Automatic refresh",
            [10, 2, 30, 0],
            format_func=lambda value: f"Every {value}s" if value else "Paused",
            key="refresh_seconds",
        )
        with st.container(horizontal=True):
            for destination in (
                "training",
                "generate",
                "engram",
                "snapshots",
                "checkpoints",
                "explorer",
            ):
                target = next(
                    (item for item in run_pages + lab if item.url_path == destination),
                    None,
                )
                if target is not None:
                    st.page_link(target, label=target.title)
    page.run()


if __name__ == "__main__":
    main()
