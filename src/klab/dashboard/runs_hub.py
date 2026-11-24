"""The Runs page: what is training now, and how every earlier run went.

Runs are shown as lineages (a run and the recoveries that continued it), each with a
stitched loss curve and a verdict per attempt. Everything comes from the read-only
queries in :mod:`klab.dashboard.queries`; activity is claimed only from a fresh
heartbeat, never from a stored ``running`` status.
"""

from __future__ import annotations

import html
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlencode

import plotly.graph_objects as go
import streamlit as st

from klab.dashboard import ui
from klab.dashboard.lineage import Lineage, heartbeat_age, lineages, stitch
from klab.dashboard.queries import metric_series, run_facts, runs

_PILOT_PURPOSES = frozenset({"smoke", "warmup"})
_LOSS_SERIES = ("train/loss", "validation/loss")

VERDICT_GUIDE = """
A verdict is read from what the trainer recorded for the attempt. Nothing is guessed.

| Verdict | Recorded facts |
|---|---|
| **Training** | Status `running` and a heartbeat in the last 150 seconds. |
| **Not verified** | Status `running` but no recent heartbeat: the process most likely died without recording its end. `klab workspace settle` marks such runs interrupted. |
| **Completed** | The run reached the end of its token or step budget. |
| **Paused cleanly** | Stopped on request (a signal or the `CANCEL` file) and left a resumable checkpoint at its last step, so nothing is repeated. |
| **Interrupted** | Stopped some other way, or without a resumable checkpoint. |
| **Memory spill**, **Out of memory**, **Non-finite loss**, **Disk full**, **Corrupt restore** | The run failed and recorded that reason. The full recorded reason is shown beside it. |
| **No progress** | The attempt stopped before it completed one update. |

A good verdict is lowered to a warning when the attempt trained more than 15% slower than
the run it resumed. **Steps redone** counts the updates after the attempt's last verified
checkpoint: the next recovery starts from that checkpoint and repeats them.
"""


def _duration(seconds: float | None) -> str:
    if seconds is None or seconds <= 0:
        return "–"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} days"


def _loss(value: float | None) -> str:
    return "–" if value is None else f"{value:.3f}"


def _run_link(root: Path, run_id: str) -> str:
    return "run?" + urlencode({"store": str(root), "run": run_id})


def _progress_bar(fraction: float | None, color: str) -> str:
    if fraction is None:
        return ""
    return (
        "<div class='rh-bar'><div class='rh-fill' "
        f"style='width:{fraction * 100:.1f}%;background:{color}'></div></div>"
    )


def _tiles(items: list[tuple[str, str, str]]) -> str:
    """Label, value and a line of help for each tile."""
    return (
        "<div class='rh-tiles'>"
        + "".join(
            f"<div class='rh-tile'><div class='l'>{html.escape(label)}</div>"
            f"<div class='v'>{html.escape(value)}</div>"
            f"<div class='h'>{html.escape(hint)}</div></div>"
            for label, value, hint in items
        )
        + "</div>"
    )


def _summary_tiles(lineage: Lineage, now: datetime) -> str:
    tip = lineage.tip
    rate = tip.facts.tokens_per_second
    remaining = (
        lineage.target_tokens - lineage.tokens
        if lineage.target_tokens and lineage.tokens is not None
        else None
    )
    eta = remaining / rate if remaining and rate else None
    latest = tip.facts.latest or {}
    progress = (
        f"{lineage.progress:.1%} of {ui.compact(lineage.target_tokens)} tokens"
        if lineage.progress is not None
        else "no token budget recorded"
    )
    step = (
        f"step {tip.last_step:,}"
        + (f" of {lineage.target_steps:,}" if lineage.target_steps else "")
        if tip.last_step is not None
        else "no update recorded"
    )
    best = min(
        (
            a.facts.best_validation_loss
            for a in lineage.attempts
            if a.facts.best_validation_loss is not None
        ),
        default=None,
    )
    items = [
        ("Tokens trained", ui.compact(lineage.tokens), progress),
        ("Position", step, "the furthest update any attempt reached"),
        (
            "Validation loss",
            _loss(latest.get("validation/loss")),
            f"best {_loss(best)} · held-out data, lower is better",
        ),
        (
            "Training loss",
            _loss(latest.get("train/loss")),
            "last logged update of the newest attempt",
        ),
        (
            "Throughput",
            f"{rate:,.0f} tok/s" if rate else "–",
            (
                f"best attempt {lineage.best_rate:,.0f} tok/s"
                if lineage.best_rate
                else "mean of the newest attempt"
            ),
        ),
        (
            "Time left" if lineage.live else "Time to finish",
            _duration(eta),
            "remaining tokens at the newest attempt's rate"
            if eta
            else "needs a budget and a rate",
        ),
    ]
    return _tiles(items)


def _attempt_rows(lineage: Lineage, root: Path) -> str:
    # Attempts of one run share a long prefix; show what tells them apart.
    ids = [attempt.run_id for attempt in lineage.attempts]
    prefix = os.path.commonprefix(ids) if len(ids) > 1 else ""
    prefix = prefix[: prefix.rfind("-") + 1]
    rows = []
    for attempt in reversed(lineage.attempts):
        facts = attempt.facts
        steps = (
            f"{facts.first_step:,} → {attempt.last_step:,}"
            if facts.first_step is not None and attempt.last_step is not None
            else "–"
        )
        rate = (
            f"{facts.tokens_per_second:,.0f}"
            if facts.tokens_per_second and facts.updates >= 5
            else "–"
        )
        duration = (
            _duration((facts.last_wall_time or 0) - (facts.first_wall_time or 0))
            if facts.first_wall_time is not None
            else "–"
        )
        lost = attempt.steps_lost
        notes = "".join(
            f"<div class='pb-hint'>{html.escape(note)}</div>" for note in attempt.notes
        )
        branch = (
            "" if attempt.on_main_line else " <span class='pb-hint'>(dead end)</span>"
        )
        rows.append(
            "<tr>"
            f"<td class='nw'><a href='{html.escape(_run_link(root, attempt.run_id))}' "
            f"target='_self' title='{html.escape(attempt.run_id)}'>"
            f"{html.escape(attempt.run_id[len(prefix) :] or attempt.run_id)}</a>{branch}</td>"
            f"<td class='nw'>{ui.chip(attempt.verdict.level, attempt.verdict.label)}</td>"
            f"<td class='nw'>{steps}</td><td>{ui.compact(attempt.last_tokens)}</td>"
            f"<td>{rate}</td><td class='nw'>{duration}</td>"
            f"<td>{'–' if lost is None else f'{lost:,}'}</td>"
            f"<td>{html.escape(attempt.verdict.why)}{notes}</td>"
            "</tr>"
        )
    return (
        "<table class='pb-table'><tr><th>Attempt</th><th>Outcome</th><th>Steps</th>"
        "<th>Tokens</th><th>Tok/s</th><th>Ran for</th><th>Steps redone</th>"
        "<th>What happened</th></tr>" + "".join(rows) + "</table>"
    )


def loss_figure(lineage: Lineage, rows: list[dict[str, object]]) -> go.Figure:
    """Training and validation loss along the main line, with each attempt's end marked."""
    figure = go.Figure()
    for name, label, color, mode in (
        ("train/loss", "Training loss", ui.SERIES["baseline"], "lines"),
        ("validation/loss", "Validation loss", ui.SERIES["candidate"], "lines+markers"),
    ):
        points = [row for row in rows if row["name"] == name]
        if points:
            figure.add_trace(
                go.Scatter(
                    x=[float(row["tokens_seen"]) / 1e9 for row in points],  # type: ignore[arg-type]
                    y=[row["value"] for row in points],
                    name=label,
                    mode=mode,
                    line={
                        "color": color,
                        "width": 1.2 if name == "train/loss" else 2.4,
                    },
                    opacity=0.55 if name == "train/loss" else 1.0,
                    hovertemplate="%{x:.3f}B tokens<br>"
                    + label
                    + " %{y:.4f}<extra></extra>",
                )
            )
    floor = min((float(row["value"]) for row in rows), default=0.0)  # type: ignore[arg-type]
    ended = [a for a in lineage.main_line if a.last_tokens is not None]
    if ended:
        figure.add_trace(
            go.Scatter(
                x=[a.last_tokens / 1e9 for a in ended],  # type: ignore[operator]
                y=[floor] * len(ended),
                name="Attempt ended",
                mode="markers",
                marker={
                    "size": 11,
                    "symbol": "diamond",
                    "color": [
                        ui.STATUS_COLOR.get(a.verdict.level, "#6e7781") for a in ended
                    ],
                    "line": {"width": 1, "color": "white"},
                },
                text=[f"{a.run_id}<br>{a.verdict.label}" for a in ended],
                hovertemplate="%{text}<br>at %{x:.3f}B tokens<extra></extra>",
            )
        )
    figure.update_xaxes(title_text="Tokens trained (billions)")
    # A log axis: the first updates fall from ln(vocabulary) and would flatten the rest.
    figure.update_yaxes(title_text="Loss (nats per token, log scale)", type="log")
    return ui.figure_layout(figure, height=340, title="Loss across every attempt")


def _lineage_block(
    lineage: Lineage, root: Path, now: datetime, *, expanded: bool
) -> None:
    tip = lineage.tip
    counts = {
        level: sum(a.verdict.level == level for a in lineage.attempts)
        for level in ("pass", "warn", "fail")
    }
    tally = " · ".join(
        f"{count} {word}"
        for level, word in (("pass", "good"), ("warn", "warning"), ("fail", "failed"))
        if (count := counts[level])
    )
    with st.container(border=True):
        st.markdown(
            f"<div class='rh-head'><span class='n'>{html.escape(lineage.name)}</span>"
            f"{ui.chip(tip.verdict.level, tip.verdict.label)}"
            f"<span class='pb-hint'>{html.escape(lineage.root_id)} · "
            f"{len(lineage.attempts)} attempt(s)"
            + (f" ({tally})" if tally else "")
            + f" · trained for {_duration(lineage.trained_seconds)}</span></div>"
            + _progress_bar(
                lineage.progress, ui.STATUS_COLOR.get(tip.verdict.level, "#6e7781")
            )
            + _summary_tiles(lineage, now),
            unsafe_allow_html=True,
        )
        st.caption(tip.verdict.why)
        with st.expander("Loss curve and attempts", expanded=expanded):
            try:
                rows = stitch(
                    metric_series(
                        root,
                        tuple(a.run_id for a in lineage.main_line),
                        _LOSS_SERIES,
                        1500,
                    ),
                    lineage.main_line,
                )
            except (OSError, sqlite3.Error, ValueError) as error:
                rows = []
                st.warning(f"Loss series unavailable: {error}")
            if rows:
                st.plotly_chart(
                    loss_figure(lineage, rows),
                    width="stretch",
                    key=f"runs-hub-loss-{lineage.root_id}",
                )
                st.caption(
                    "One curve stitched from the attempts on the path to the newest "
                    "one. Updates that a recovery repeated are shown once. Diamonds mark "
                    "where an attempt ended, coloured by its outcome."
                )
            else:
                st.info("No loss has been logged for this run yet.")
            st.markdown(_attempt_rows(lineage, root), unsafe_allow_html=True)


def runs_page(root: Path) -> None:
    ui.page_header(
        "Runs",
        "What is training right now, and how did every earlier run go?",
        f"Run store {root}",
    )
    if not (root / "experiments.sqlite3").is_file():
        ui.empty_state(
            "No runs in this store yet",
            "A run appears here as soon as `klab train` starts it.",
            ["klab train CONFIG.yaml --run-id NAME"],
        )
        return
    interval = st.session_state.get("refresh_seconds", 10)
    st.fragment(run_every=interval or None)(_body)(root)
    ui.explainer("How a verdict is decided", VERDICT_GUIDE)


def _body(root: Path) -> None:
    now = datetime.now(UTC)
    try:
        records = runs(root)
        facts = run_facts(root)
    except (OSError, sqlite3.Error, ValueError) as error:
        st.error(f"Dashboard could not read the run store: {error}")
        return
    include_pilots = st.session_state.get("include_pilot_runs", False)
    records = [r for r in records if include_pilots or r.purpose not in _PILOT_PURPOSES]
    groups = lineages(records, facts, now)
    live = [group for group in groups if group.live]
    st.subheader("Active now")
    if live:
        for group in live:
            age = heartbeat_age(group.tip.facts, now)
            st.markdown(
                "<div class='rh-live'>Verified by a heartbeat "
                f"{age:.0f} s ago, observed at {now:%H:%M:%S} UTC.</div>"
                if age is not None
                else "",
                unsafe_allow_html=True,
            )
            _lineage_block(group, root, now, expanded=True)
    else:
        latest = groups[0].tip if groups else None
        st.info(
            "Nothing is verified to be training: no run in this store has written a "
            "heartbeat in the last 150 seconds."
            + (
                f" The most recent activity was {latest.run_id} "
                f"({latest.verdict.label.lower()}), last recorded "
                f"{datetime.fromtimestamp(groups[0].last_activity, UTC):%Y-%m-%d %H:%M} UTC."
                if latest
                else ""
            )
        )
    history = [group for group in groups if not group.live]
    st.subheader(f"History · {len(history)} run(s)")
    if not history:
        st.caption("Earlier runs will be listed here.")
    for index, group in enumerate(history):
        _lineage_block(group, root, now, expanded=index == 0 and not live)
