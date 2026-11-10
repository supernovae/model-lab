"""Recorded training observations only: no model loads or artifact writes."""

from __future__ import annotations

import math

import streamlit as st

from sparselab.dashboard.queries import DashboardSnapshot
from sparselab.training.metric_registry import metric_spec


def latest_metrics(view: DashboardSnapshot, selected_ids: list[str]) -> list[dict]:
    """Keep each series' own observation step; never join asynchronous observations."""
    series: dict[tuple[str, str], list[dict]] = {}
    for row in view.metrics:
        if row["run_id"] in selected_ids:
            series.setdefault((str(row["run_id"]), str(row["name"])), []).append(row)
    result = []
    for (run_id, name), points in sorted(series.items()):
        points = sorted(points, key=lambda row: (row["step"], row["wall_time"]))
        last = points[-1]
        value = last["value"]
        valid = isinstance(value, (int, float)) and math.isfinite(value)
        previous = points[-2] if len(points) > 1 else None
        prior = previous["value"] if previous else None
        delta = (
            value - prior
            if valid and isinstance(prior, (int, float)) and math.isfinite(prior)
            else None
        )
        spec = metric_spec(name)
        result.append(
            {
                "run_id": run_id,
                "metric": name,
                "value": value if valid else None,
                "availability": "recorded" if valid else "non-finite or missing",
                "step": last["step"],
                "tokens_seen": last["tokens_seen"],
                "previous_step": previous["step"] if previous else None,
                "change": delta,
                "unit": spec.unit if spec else "unavailable",
            }
        )
    return result


def engram_summary(
    view: DashboardSnapshot, selected_ids: list[str], *, compact: bool = False
) -> None:
    st.subheader("Engram · n-gram memory")
    st.caption(
        "Recorded diagnostics, target-weighted across microbatches in each update. "
        "Counts may be fractional. Address reuse includes repeated input; it is not "
        "proof of hash collisions, retrieval accuracy, or improved model quality."
    )
    rows = [
        row
        for row in latest_metrics(view, selected_ids)
        if row["metric"].startswith("engram/")
    ]
    if not selected_ids:
        st.info("Select a run in the sidebar to inspect its memory observations.")
        return
    for run_id in selected_ids:
        own = [row for row in rows if row["run_id"] == run_id]
        with st.container(border=True):
            st.markdown(f"**{run_id}**")
            if not own:
                st.info(
                    "Unavailable: this run has no persisted Engram diagnostics. Missing values are not zero."
                )
                continue
            ports = sorted(
                {
                    row["metric"].split("/")[2]
                    for row in own
                    if row["metric"].startswith("engram/port/")
                }
            )
            scopes = [("Aggregate", "engram/")] + [
                (f"Port {port}", f"engram/port/{port}/") for port in ports
            ]
            by_name = {row["metric"]: row for row in own}
            if compact:
                columns = st.columns(3)
                for column, metric, label in zip(
                    columns,
                    ("gate_mean", "value_norm", "table_utilization"),
                    ("Mean gate", "Value norm", "Addressed-row fraction"),
                    strict=True,
                ):
                    row = by_name.get(f"engram/{metric}")
                    value = row["value"] if row else None
                    column.metric(
                        label,
                        (
                            f"{value:.2%}"
                            if metric == "table_utilization"
                            else f"{value:.4g}"
                        )
                        if value is not None
                        else "Unavailable",
                    )
                steps = [row["step"] for row in own]
                st.caption(
                    f"Recorded steps {min(steps)}–{max(steps)} · {len(ports)} per-port series groups · Open Engram memory for details."
                )
                continue
            summary = []
            for label, prefix in scopes:
                for metric in (
                    "gate_mean",
                    "value_norm",
                    "hidden_norm",
                    "bucket_reuse_rate",
                    "table_utilization",
                ):
                    row = by_name.get(prefix + metric)
                    summary.append(
                        {
                            "scope": label,
                            "metric": metric,
                            "value": row["value"] if row else None,
                            "step": row["step"] if row else None,
                            "availability": row["availability"]
                            if row
                            else "not recorded",
                        }
                    )
            st.dataframe(summary, hide_index=True, width="stretch")
    if compact:
        return
    st.info(
        "Utilization is a length-weighted per-document unique addressed-row fraction, "
        "not the fraction of the whole table learned. Reported collision counts include "
        "repeated inputs; they are not true hash-collision rates. Gates and norms measure "
        "memory activity, not useful knowledge or causal impact."
    )
    with st.expander("How to read these values", expanded=True):
        names = (
            "gate_mean",
            "value_norm",
            "hidden_norm",
            "bucket_reuse_rate",
            "table_utilization",
            "collision_count",
            "maximum_address_fraction",
        )
        for name in names:
            spec = metric_spec(f"engram/{name}")
            st.write(f"**{name.replace('_', ' ').capitalize()}** — {spec.summary}")
        st.caption(
            "Aggregate and per-port values come from the producer. No gradients or quality gains are inferred."
        )
    with st.expander("All memory values and changes since the previous observation"):
        st.dataframe(rows, hide_index=True, width="stretch")
        st.caption(
            "A change compares successive observations of the same metric, not checkpoints or model quality."
        )


def snapshot_inspector(view: DashboardSnapshot, selected_ids: list[str]) -> None:
    st.header("Snapshot inspector")
    st.caption(
        "Inspect recorded state without loading weights, running inference, or changing training."
    )
    records = [record for record in view.runs if record.run_id in selected_ids]
    if not records:
        st.info("Select a run in the sidebar.")
        return
    run_id = st.selectbox(
        "Inspect run", [record.run_id for record in records], key="inspect_run"
    )
    record = next(record for record in records if record.run_id == run_id)
    st.caption(
        f"Recorded status: {record.status} · Run updated: {record.updated_at} UTC"
    )
    section = st.radio(
        "Inspect",
        ["Latest values", "Configuration", "Checkpoints", "Recent events"],
        horizontal=True,
        key="inspect_section",
    )
    if section == "Latest values":
        search = st.text_input("Filter metric names", key="inspect_metric_filter")
        rows = [
            row
            for row in latest_metrics(view, [run_id])
            if search.lower() in row["metric"].lower()
        ]
        st.dataframe(rows, width="stretch", hide_index=True)
        st.caption(
            "Each metric retains its own step and target count. Blank change means no finite prior observation."
        )
    elif section == "Configuration":
        st.json(record.config, expanded=False)
        with st.expander("Recorded runtime and run metadata"):
            st.json(record.metadata, expanded=False)
    elif section == "Checkpoints":
        rows = sorted(
            (row for row in view.checkpoints if row["run_id"] == run_id),
            key=lambda row: (row["step"], row["created_at"]),
            reverse=True,
        )
        if not rows:
            st.info("No checkpoint metadata recorded for this run.")
        else:
            row = st.selectbox(
                "Saved checkpoint (save sequence within run)",
                rows,
                format_func=lambda row: (
                    f"Step {row['step']:,} · Save {row['checkpoint_id']}"
                ),
                key="inspect_checkpoint",
            )
            columns = st.columns(3)
            columns[0].metric("Saved step", row["step"])
            columns[1].metric("Recorded targets", f"{row['tokens_seen']:,}")
            columns[2].metric(
                "Size",
                f"{row['size_bytes'] / 1024**3:.2f} GiB"
                if isinstance(row["size_bytes"], (int, float))
                else "Unavailable",
            )
            st.write(f"Generation: `{row['relative_path']}`")
            st.caption(
                f"Backend: {row['backend']} · Resume level: {row['resume_level']} · Save-time verification: {row['verification_status']}"
            )
            with st.expander("Identity and recorded checkpoint metadata"):
                st.json(row)
            st.caption(
                "Metadata describes save-time evidence. Use Checkpoints for explicit local file verification."
            )
        st.info(
            "Explorer supports CPU inspection of supported checkpoints up to 60M parameters. Larger models have architecture-only inspection. This page never loads a model."
        )
    else:
        rows = [row for row in view.events if row["run_id"] == run_id][-50:]
        if rows:
            st.dataframe(list(reversed(rows)), width="stretch", hide_index=True)
        else:
            st.info("No recorded events for this run.")


def learning_summary(view: DashboardSnapshot, selected_ids: list[str]) -> None:
    """Explain observed learning separately from checkpoint capability evidence."""
    st.subheader("Learning evidence")
    st.caption(
        "Lower training loss means better prediction of training targets. Lower validation "
        "loss is evidence on the recorded held-out data; neither establishes reasoning, "
        "coding skill, or the benefit of Engram without a controlled comparison."
    )
    rows = latest_metrics(view, selected_ids)
    for run_id in selected_ids:
        st.markdown(f"**{run_id}**")
        columns = st.columns(2)
        for col, name, label in zip(
            columns,
            ("train/loss", "validation/loss"),
            ("Training prediction loss", "Held-out prediction loss"),
            strict=True,
        ):
            row = next(
                (
                    row
                    for row in rows
                    if row["run_id"] == run_id and row["metric"] == name
                ),
                None,
            )
            value = row["value"] if row else None
            col.metric(
                label,
                f"{value:.4f}" if value is not None else "Not recorded",
                delta=f"{row['change']:+.4f}"
                if row and row["change"] is not None
                else None,
                delta_color="inverse",
            )
            if row:
                col.caption(
                    f"Step {row['step']} · change from step {row['previous_step'] if row['previous_step'] is not None else 'unavailable'}"
                )
    st.info(
        "Demonstrated capabilities: not established by this telemetry view. Open Experiments "
        "for verified probe verdicts and Behaviors for saved examples. Check the exact "
        "checkpoint and comparison group: an older probe does not score the current training model. "
        "Impact requires a comparable baseline; missing evidence is unknown, not failure."
    )
