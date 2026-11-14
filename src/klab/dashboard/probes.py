"""Probe battery components shared by the lab pages: verdict, table, charts, Pareto."""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from klab.dashboard.probe_data import (
    history_rows,
    live_batteries,
    probe_row,
    result_group,
)
from klab.dashboard.ui import (
    ACTION_TEXT,
    SERIES,
    STATUS_COLOR,
    STATUS_ICON,
    chip,
    num,
)
from klab.probes.points import (
    COSTS,
    METRICS,
    mean_chance,
    metric_points,
    pareto_frontier,
)
from klab.probes.suite import NEEDLE_PROBES, PROBES

_chip = chip
_num = num


def _meter(row: dict[str, Any]) -> str:
    regression = row.get("regression")
    scale = (row.get("thresholds") or {}).get("fail") or (
        row.get("thresholds") or {}
    ).get("warn")
    if regression is None or not scale:
        return ""
    share = min(1.0, abs(regression) / scale) * 50
    if row.get("within_noise"):
        share = min(share, 6)
    if regression < 0:
        style = f"right:50%;width:{share:.1f}%;background:{STATUS_COLOR['pass']}"
    else:
        tone = STATUS_COLOR.get(row.get("status", ""), "#b58100")
        if row.get("status") == "pass":
            tone = "#8c959f"
        style = f"left:50%;width:{share:.1f}%;background:{tone}"
    return (
        f'<div class="pb-meter" title="left = better, right = worse; '
        f'the full half is the fail threshold"><div class="bar" style="{style}"></div>'
        f'<div class="mid"></div></div>'
    )


def _verdict_banner(result: dict[str, Any], sweep_id: str | None = None) -> None:
    verdict = result.get("verdict") or {}
    status = str(verdict.get("status", "info"))
    action = ACTION_TEXT.get(verdict.get("action", ""), verdict.get("action", ""))
    suggestion = str(verdict.get("suggestion", ""))
    if sweep_id and status in {"warn", "fail"}:
        # One point on a sweep's curve: the screen's "abandon" or "tweak" judges a
        # single idea, but a sweep measures how a setting moves the result.
        suggestion = (
            f"This try is one point of sweep {sweep_id}. The probe screen's "
            f"{status.upper()} says the change did not beat the baseline here, "
            f"which is a reading on the curve, not a failed run. {suggestion} "
            f"Read the whole curve with `klab report {sweep_id}`."
        )
        status = "info"
        action = "Read the whole sweep curve"
    next_tier = verdict.get("next_tier")
    guard = result.get("guard") or {}
    guard_text = (
        "⚠ dev-split gain not shared by the held-out split (possible probe overfit)"
        if guard.get("overfit_suspected")
        else "held-out split decides; no dev/held-out divergence"
    )
    stop = result.get("stop") or {}
    stop_text = (
        f"<p>⚡ {html.escape(str(stop['reason']))}</p>" if stop.get("stopped") else ""
    )
    missing = verdict.get("missing") or []
    if missing:
        stop_text += (
            "<p><b>Missing evidence:</b> "
            + "; ".join(
                html.escape(f"{m['id']} ({m['status']}): {m.get('note') or ''}")
                for m in missing
            )
            + "</p>"
        )
    st.markdown(
        f'<div class="pb-banner" style="background:{STATUS_COLOR.get(status, "#0969da")}">'
        f"<h3>{STATUS_ICON.get(status, '?')} {html.escape(status.upper())} · "
        f"next: {html.escape(str(action))}"
        + (f" (--config {html.escape(next_tier)})" if next_tier else "")
        + "</h3>"
        f"<p>{html.escape(suggestion)}</p>"
        f"<p style='font-size:.85rem'>Guard: {html.escape(guard_text)} · "
        f"tiers run: {html.escape(' → '.join(result.get('tiers_run') or []))} · "
        f"{result.get('seconds', 0):.1f}s</p>{stop_text}</div>",
        unsafe_allow_html=True,
    )


def _metrics(result: dict[str, Any]) -> None:
    rows = {row["id"]: row for row in result.get("probes") or []}
    picks = [
        ("heldout_loss", "Held-out loss", "inverse"),
        ("fact_recall", "Fact recall", "normal"),
        ("needle", "Needle", "normal"),
        ("repetition", "Repetition (seq-rep-4)", "inverse"),
    ]
    columns = st.columns(len(picks))
    for column, (probe_id, label, mode) in zip(columns, picks, strict=True):
        row = rows.get(probe_id) or {}
        value = row.get("value")
        delta = row.get("delta")
        column.metric(
            label,
            ("not run" if not row else "–") if value is None else f"{value:.3f}",
            None if delta is None else f"{delta:+.3f} vs baseline",
            delta_color=mode,
            help=(next((p.explains for p in PROBES if p.id == probe_id), None)),
        )


def _detail(row: dict[str, Any]) -> str:
    details = row.get("details") or {}
    if row["id"] == "heldout_loss" and details.get("perplexity"):
        return (
            f"ppl {details['perplexity']:.1f} · {details.get('valid_targets')} targets"
        )
    if row["id"] == "token_agreement" and "js" in details:
        text = f"JS {details['js']:.3f} · KL {details['kl_base_to_candidate']:.3f}"
        if details.get("informative") is False:
            text += " · near-identical predictions"
        return text
    if row["id"] == "repetition" and details.get("distinct_2") is not None:
        return f"distinct-1 {details['distinct_1']:.2f} · distinct-2 {details['distinct_2']:.2f}"
    if row["id"] == "parametric_recall" and details.get("chance") is not None:
        return (
            f"chance {details['chance']:.2f} · never-trained control "
            f"{num(details.get('control_accuracy'), 2)} · n={details.get('n')}"
        )
    if (
        row["id"] in {"fact_recall", *NEEDLE_PROBES}
        and details.get("chance") is not None
    ):
        return f"chance {details['chance']:.2f} · n={details.get('n')}"
    if row["id"] == "calibration" and details.get("reliability"):
        return f"{len(details['reliability'])} confidence bins"
    if row["id"] == "lm_eval" and details.get("tasks"):
        return " · ".join(
            f"{task} {_num((v or {}).get('acc'), 2)}"
            for task, v in details["tasks"].items()
        )
    return ""


def _probe_table(result: dict[str, Any]) -> None:
    body = []
    for row in result.get("probes") or []:
        status = str(row.get("status"))
        value = _num(row.get("value"))
        if row.get("baseline_value") is not None:
            value += (
                f" <span class='pb-hint'>vs {_num(row.get('baseline_value'))}</span>"
            )
        delta = "" if row.get("delta") is None else f"{row['delta']:+.3f}"
        if row.get("delta_se") is not None:
            label = " within-run eval noise" if row["id"] == "heldout_loss" else ""
            delta += f" <span class='pb-hint'>±{row['delta_se']:.3f}{label}</span>"
        hint = ""
        if status in {"warn", "fail"}:
            hint = html.escape(row.get("suggests") or "")
        elif row.get("note"):
            hint = html.escape(str(row["note"]))
        body.append(
            "<tr>"
            f"<td>{_chip(status)}</td>"
            f"<td><b>{html.escape(row['title'])}</b><br>"
            f"<span class='pb-hint'>{html.escape(row['tier'])} tier · cost {row['cost']}"
            + (" · hard" if row.get("hard") else "")
            + "</span></td>"
            f"<td>{value}</td><td>{delta}{_meter(row)}</td>"
            f"<td>{html.escape(_detail(row))}<br><span class='pb-hint'>{hint}</span></td>"
            "</tr>"
        )
    st.markdown(
        "<table class='pb-table'><tr><th>Status</th><th>Probe</th>"
        "<th>Candidate</th><th>Δ vs baseline</th><th>Notes</th></tr>"
        + "".join(body)
        + "</table>",
        unsafe_allow_html=True,
    )


def _comparison_chart(result: dict[str, Any]) -> None:
    rows = [
        row
        for row in result.get("probes") or []
        if row.get("value") is not None and row.get("baseline_value") is not None
    ]
    if not rows:
        st.caption("Bars appear when a baseline is part of the result.")
        return
    figure = make_subplots(
        rows=1, cols=len(rows), subplot_titles=[row["title"] for row in rows]
    )
    for index, row in enumerate(rows, start=1):
        colors = ["#8c959f", STATUS_COLOR.get(row["status"], "#0969da")]
        figure.add_trace(
            go.Bar(
                x=["baseline", "candidate"],
                y=[row["baseline_value"], row["value"]],
                marker_color=colors,
                showlegend=False,
                hovertemplate="%{x}: %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=index,
        )
    figure.update_layout(height=260, margin={"l": 10, "r": 10, "t": 40, "b": 10})
    figure.update_annotations(font_size=11)
    st.plotly_chart(figure, use_container_width=True)
    for needle in (r for r in rows if r["id"] in NEEDLE_PROBES):
        by_length = needle.get("details", {}).get("by_length") or {}
        if not by_length:
            continue
        frame = pd.DataFrame(
            [
                {
                    "context length": v.get("chars") or v.get("tokens"),
                    "arm": arm,
                    "accuracy": v[key],
                }
                for v in by_length.values()
                for arm, key in (
                    ("candidate", "accuracy"),
                    ("baseline", "baseline_accuracy"),
                )
                if v.get(key) is not None
            ]
        )
        chart = px.line(
            frame,
            x="context length",
            y="accuracy",
            color="arm",
            markers=True,
            title=f"{needle['title']}: retrieval by context length",
            color_discrete_map={
                "candidate": SERIES["candidate"],
                "baseline": SERIES["baseline"],
            },
        )
        chance = needle["details"].get("chance")
        if chance:
            chart.add_hline(y=chance, line_dash="dot", annotation_text="chance")
        chart.update_layout(
            height=280,
            margin={"l": 10, "r": 10, "t": 40, "b": 10},
            yaxis_range=[0, 1.05],
        )
        st.plotly_chart(chart, use_container_width=True)


def _explain() -> None:
    st.markdown(
        "Each probe is cheap and answers one question. They run cheapest first; a "
        "**hard** failure stops the battery (fast-fail) and the battery only escalates "
        "to the next tier when nothing failed. Verdicts use the **held-out** item "
        "split only; the **dev** split is free to look at, and a dev gain the "
        "held-out split does not share raises the overfit guard. Probes are "
        "screening signals, not proof that an idea is useful: selecting on them "
        "again and again turns the held-out items into development data, so keep "
        "a separate, untouched final evaluation. A missing check (optional tier "
        "not installed, a probe error) makes the battery **incomplete**, never a "
        "pass."
    )
    for spec in PROBES:
        fail = "—" if spec.fail is None else f"{spec.fail:g}"
        how = {
            "delta_rel": "relative change vs baseline",
            "delta_abs": "absolute change vs baseline",
            "value_min": "absolute value (minimum)",
        }[spec.mode]
        with st.expander(
            f"{spec.title} · {spec.tier} tier · cost {spec.cost}",
            expanded=spec is PROBES[0],
        ):
            st.markdown(spec.explains)
            st.markdown(
                f"**Judged on** {how}: warn beyond {spec.warn:g}, fail beyond {fail}"
                + (" (hard: stops the battery)" if spec.hard else "")
                + ". Changes within 2 standard errors count as noise (for held-out "
                "loss this is within-run eval noise; seed-to-seed spread is "
                "larger).\n\n"
                f"**If it fails:** {spec.suggests}\n\n"
                f"*Reference:* {spec.reference}"
            )


def _history(entries: list[Any]) -> None:
    frame = pd.DataFrame(history_rows(entries))
    if frame.empty:
        st.info("No probe results yet.")
        return
    frame["when"] = pd.to_datetime(frame["when"], errors="coerce", utc=True)
    shown = frame[
        [
            "when",
            "id",
            "idea",
            "verdict",
            "action",
            "tier",
            "loss",
            "loss_delta",
            "recall",
            "needle",
            "suite",
        ]
    ]
    st.dataframe(shown, use_container_width=True, hide_index=True)
    plotted = frame.dropna(subset=["loss_delta"])
    if not plotted.empty:
        chart = px.scatter(
            plotted,
            x="when",
            y="loss_delta",
            color="verdict",
            symbol="action",
            hover_data=["id", "idea"],
            color_discrete_map=STATUS_COLOR,
            title="Held-out loss change vs baseline over time (below 0 = better)",
        )
        chart.add_hline(y=0, line_color="#8c959f")
        chart.update_traces(marker_size=11)
        chart.update_layout(height=320, margin={"l": 10, "r": 10, "t": 40, "b": 10})
        st.plotly_chart(chart, use_container_width=True)
    if frame["suite"].nunique() > 1:
        st.warning(
            "History mixes probe suite versions; only compare rows with the same suite digest."
        )


PARETO_EXPLAINER = """
**Reading the chart.** Each dot is one checkpoint: up or down is quality,
left or right is what it cost. The dotted line is the **Pareto frontier**:
checkpoints no other checkpoint beats on *both* cost and quality. Ideas worth
keeping move the frontier; a dot behind it is dominated.

**Resident vs. active parameters.** *Resident* counts every weight held in
memory. *Active* counts what one token actually touches: one row of the
input embedding table, only the routed experts of a mixture-of-experts layer,
only the looked-up rows of a memory table. Sparse ideas trade resident memory
for fewer active parameters, so look at both axes. The grey bar spans a
checkpoint's active → resident count.

**Comparison groups.** A number only means something next to numbers measured
the same way. Held-out loss compares within one *eval group* (same validation
data, tokenizer, loss mask and eval protocol); lm-eval accuracy within one
*benchmark group* (same tasks, task versions, shots, scoring protocol, and the
same prompts and targets for every item). Other groups are hidden, never mixed
in. One checkpoint measured by several records is one point; each value says
which record it came from.

**Reference points (◆).** Pinned public checkpoints (Pythia, SmolLM2) scored
by the same lm-eval path as our runs, so a result sits on a known curve. They
were trained on hundreds of billions to trillions of tokens with their own
tokenizers, so they only appear on lm-eval accuracy, never on our held-out
loss. At tiny scale most tasks sit near chance (dashed line); use
`klab compare RESULT --references` for paired deltas with standard errors.
"""
PARAMETER_SPANS = {
    "parameters": ("active_parameters", "parameters"),
    "active_parameters": ("active_parameters", "parameters"),
    "parameter_bytes": ("active_parameter_bytes", "parameter_bytes"),
    "active_parameter_bytes": ("active_parameter_bytes", "parameter_bytes"),
}
KIND_STYLE = {
    "reference": ("diamond", "#8250df", "reference model"),
    "try": ("circle", "#0969da", "lab try arm"),
    "probe": ("square", "#1a7f37", "probed checkpoint"),
}


def own_group(result: dict[str, Any] | None, metric_id: str) -> str | None:
    """The comparison group a result's own value was measured in."""
    if not result:
        return None
    return result_group(result, probe_row(result, metric_id) or {"id": metric_id})


def _pareto(raw: list[dict[str, Any]], selected: Any | None) -> None:
    controls = st.columns([2, 3])
    metric_id = controls[0].radio(
        "Quality",
        list(METRICS),
        horizontal=True,
        key="probe_pareto_metric",
        format_func=lambda m: METRICS[m].label,
        help="Held-out loss: our validation split (lab runs only). lm-eval "
        "accuracy: public tasks, shared with the reference models. Fact recall: "
        "the in-context fact items, text-level, so references can share it.",
    )
    metric = METRICS[metric_id]
    axis = controls[1].radio(
        "Cost axis",
        list(COSTS),
        horizontal=True,
        key="probe_pareto_axis",
        format_func=lambda name: COSTS[name],
    )
    points = pd.DataFrame(metric_points(raw, metric_id))
    if points.empty:
        st.info(
            f"No checkpoint has {metric.label} yet. To get it, {metric.missing_hint}."
        )
        _pareto_explainer()
        return
    ungrouped = int(points["group"].isna().sum())
    points = points.dropna(subset=["group"])
    counts = points["group"].value_counts()
    groups = list(counts.index)
    own = own_group(selected.result if selected is not None else None, metric_id)
    group = st.selectbox(
        f"Comparison group ({metric.group_label})",
        groups,
        index=groups.index(own) if own in groups else 0,
        key=f"probe_pareto_group_{metric_id}",
        format_func=lambda g: (
            f"{g[:12]} · {int(counts[g])} checkpoint(s), "
            f"{int(((points['group'] == g) & (points['kind'] == 'reference')).sum())}"
            " reference(s)"
        ),
        help="Only checkpoints measured the same way share a scale; see the "
        "explainer below.",
    )
    data = points[points["group"] == group]
    missing_axis = int(data[axis].isna().sum())
    data = data.dropna(subset=[axis]).reset_index(drop=True)
    has_refs = bool((data["kind"] == "reference").any())
    log_x = st.toggle(
        "Log cost axis",
        value=has_refs,
        key=f"probe_pareto_log_{metric_id}",
        help="References are 10–100× larger than lab runs; a log axis keeps both "
        "readable.",
    )
    notes = []
    if len(groups) > 1:
        notes.append(f"{len(groups) - 1} other {metric.group_label}(s) hidden")
    if missing_axis:
        notes.append(
            f"{missing_axis} checkpoint(s) without {COSTS[axis]}"
            + (" (references are not timed)" if axis == "ms_per_token" else "")
        )
    if ungrouped:
        notes.append(
            f"{ungrouped} older result(s) recorded no {metric.group_label}; "
            "re-score them to place them"
        )
    if notes:
        st.caption(" · ".join(notes))
    if data.empty:
        st.info(f"No checkpoint in this group records {COSTS[axis]}.")
        _pareto_explainer()
        return
    front = pareto_frontier(
        list(zip(data[axis], data["value"], strict=True)),
        maximize=metric.higher_is_better,
    )
    data["frontier"] = data.index.isin(front)
    st.plotly_chart(_pareto_figure(data, axis, metric, log_x), use_container_width=True)
    table = data[
        [
            "label",
            "kind",
            "parameters",
            "active_parameters",
            "tokens_seen",
            "value",
            "frontier",
        ]
    ].rename(columns={"value": metric.label, "kind": "source"})
    st.dataframe(
        table.sort_values(metric.label, ascending=not metric.higher_is_better),
        use_container_width=True,
        hide_index=True,
        column_config={
            "label": st.column_config.TextColumn("checkpoint", width="large"),
            "parameters": st.column_config.NumberColumn(
                "resident params", format="compact"
            ),
            "active_parameters": st.column_config.NumberColumn(
                "active / token", format="compact"
            ),
            "tokens_seen": st.column_config.NumberColumn(
                "training tokens", format="compact"
            ),
            metric.label: st.column_config.NumberColumn(format="%.4f"),
            "frontier": st.column_config.CheckboxColumn("on frontier"),
        },
    )
    _pareto_explainer()


def _pareto_figure(data: pd.DataFrame, axis: str, metric: Any, log_x: bool) -> Any:
    figure = go.Figure()
    span = PARAMETER_SPANS.get(axis)
    if span is not None:
        low, high = span
        xs: list[Any] = []
        ys: list[Any] = []
        for _, row in data.dropna(subset=[low, high]).iterrows():
            if row[low] != row[high]:
                xs += [row[low], row[high], None]
                ys += [row["value"], row["value"], None]
        if xs:
            figure.add_trace(
                go.Scatter(
                    x=xs,
                    y=ys,
                    mode="lines",
                    line={"color": "#afb8c1", "width": 5},
                    name="active → resident",
                    hoverinfo="skip",
                )
            )
    frontier = data[data["frontier"]].sort_values(axis)
    figure.add_trace(
        go.Scatter(
            x=frontier[axis],
            y=frontier["value"],
            mode="lines",
            line={"color": "#0969da", "dash": "dot", "shape": "hv"},
            name="Pareto frontier",
        )
    )
    for kind, (symbol, color, name) in KIND_STYLE.items():
        part = data[data["kind"] == kind]
        if part.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=part[axis],
                y=part["value"],
                mode="markers+text" if kind == "reference" else "markers",
                text=part["label"] if kind == "reference" else None,
                textposition="top center",
                marker={
                    "symbol": symbol,
                    "size": 14 if kind == "reference" else 12,
                    "color": color,
                    "line": {
                        "width": [3 if f else 1 for f in part["frontier"]],
                        "color": "#24292f",
                    },
                },
                name=name,
                customdata=part[
                    ["label", "parameters", "active_parameters", "tokens_seen"]
                ],
                hovertemplate="<b>%{customdata[0]}</b><br>"
                + f"{metric.label} %{{y:.4f}}<br>"
                + "resident %{customdata[1]:,} · active %{customdata[2]:,}<br>"
                + "tokens %{customdata[3]:,}<extra></extra>",
            )
        )
    # One group per figure; any unknown task baseline omits the line.
    chance = [
        mean_chance(c)
        for c in data.get("chance", pd.Series(dtype=object)).dropna()
        if isinstance(c, dict)
    ]
    if chance and chance[0] is not None:
        figure.add_hline(
            y=chance[0],
            line_dash="dash",
            line_color="#8c959f",
            annotation_text="chance (task mean)",
            annotation_position="bottom right",
        )
    better = "up" if metric.higher_is_better else "down"
    figure.update_layout(
        title=f"{metric.label} vs {COSTS[axis]} · better is {better} and left",
        xaxis_title=COSTS[axis],
        yaxis_title=metric.label,
        xaxis_type="log" if log_x else "linear",
        height=440,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        legend={"orientation": "h", "y": -0.2},
    )
    return figure


def _pareto_explainer() -> None:
    with st.expander("How to read the Pareto view", expanded=False):
        st.markdown(PARETO_EXPLAINER)


@st.fragment(run_every=2)
def _live(lab_dir: Path) -> None:
    for state in live_batteries(lab_dir):
        total = int(state.get("total") or 0)
        done = len(state.get("done") or [])
        chips = " ".join(_chip(s) for s in (state.get("statuses") or {}).values())
        st.markdown(
            f"<div class='pb-live'>⏳ <b>{html.escape(state['key'])}</b> · tier "
            f"{html.escape(str(state.get('tier')))} · running "
            f"<code>{html.escape(str(state.get('current') or state.get('state')))}</code>"
            f"<br>{chips}</div>",
            unsafe_allow_html=True,
        )
        if total:
            st.progress(done / total, text=f"{done}/{total} probes")
