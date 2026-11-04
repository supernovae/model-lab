"""Shared look and feel for the lab pages: palette, chips, headers, empty states.

One palette for every page. Status colors pass WCAG AA contrast for white text
and always come with an icon, so color is never the only signal.
"""

from __future__ import annotations

import html
from collections.abc import Sequence
from typing import Any

import streamlit as st

STATUS_COLOR = {
    "pass": "#1a7f37",
    "warn": "#9a6700",
    "fail": "#cf222e",
    "info": "#0969da",
    "skipped": "#6e7781",
    "not_comparable": "#8250df",
    "error": "#cf222e",
    "unavailable": "#9a6700",
    "incomplete": "#8250df",
    "baseline": "#6e7781",
}
STATUS_ICON = {
    "pass": "✔",
    "warn": "▲",
    "fail": "✖",
    "info": "•",
    "skipped": "⊘",
    "not_comparable": "≠",
    "error": "!",
    "unavailable": "○",
    "incomplete": "…",
}
ACTION_TEXT = {
    "abandon": "Abandon this idea",
    "tweak": "Tweak and retry",
    "rerun": "Fix the missing checks and re-run",
    "escalate": "Escalate to the next tier",
    "longer_run": "Schedule a longer run",
    "compare": "Compare against a baseline",
    "report": "Final verdict: report it (terminal)",
}
# Candidate / baseline / reference series, colorblind-safe (Okabe-Ito based).
SERIES = {
    "candidate": "#0072b2",
    "baseline": "#999999",
    "reference": "#8250df",
    "control": "#e69f00",
    "accent": "#009e73",
}
# Sequential scale for heatmaps (loss, attention, routing): perceptually uniform.
SEQUENTIAL = "Viridis"

CSS = """
<style>
.block-container {padding-top: 4.5rem; max-width: 1400px;}
.pb-banner {border-radius: 14px; padding: 18px 22px; margin: 6px 0 14px 0;
  color: white; box-shadow: 0 2px 10px rgba(0,0,0,.08);}
.pb-banner h3 {margin: 0 0 4px 0; color: white;}
.pb-banner p {margin: 2px 0; opacity: .95;}
.pb-chip {display: inline-block; padding: 1px 10px; border-radius: 999px;
  color: white; font-size: .78rem; font-weight: 600; letter-spacing: .02em;
  white-space: nowrap;}
.pb-table {width: 100%; border-collapse: collapse; font-size: .92rem;}
.pb-table td, .pb-table th {padding: 7px 8px; border-bottom: 1px solid rgba(128,128,128,.18);
  vertical-align: top; text-align: left;}
.pb-table th {font-weight: 600; opacity: .7; font-size: .8rem; text-transform: uppercase;}
.pb-meter {position: relative; width: 120px; height: 10px; border-radius: 6px;
  background: rgba(128,128,128,.15); margin-top: 5px;}
.pb-meter .mid {position: absolute; left: 50%; top: -2px; width: 2px; height: 14px;
  background: rgba(128,128,128,.6);}
.pb-meter .bar {position: absolute; top: 0; height: 10px; border-radius: 6px;}
.pb-hint {opacity: .75; font-size: .82rem;}
.pb-live {border: 1px dashed rgba(128,128,128,.5); border-radius: 12px; padding: 10px 14px;
  margin-bottom: 10px;}
.lab-q {font-size: 1.05rem; opacity: .8; margin: -6px 0 14px 0;}
.lab-card {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
  padding: 14px 16px; margin-bottom: 10px; background: rgba(128,128,128,.04);}
.lab-card h4 {margin: 0 0 4px 0; font-size: 1rem;}
.lab-card p {margin: 0 0 6px 0; font-size: .9rem; opacity: .85;}
.lab-card code {font-size: .85rem;}
.lab-empty {border: 2px dashed rgba(128,128,128,.35); border-radius: 14px;
  padding: 20px 22px; margin: 8px 0 16px 0;}
.lab-empty h4 {margin: 0 0 6px 0;}
.lab-tok {display: inline-block; padding: 2px 3px; margin: 1px; border-radius: 4px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85rem;
  white-space: pre;}
.lab-arch {display: flex; flex-direction: column; align-items: stretch; gap: 6px;}
.lab-block {border-radius: 10px; padding: 8px 12px; border: 1px solid rgba(128,128,128,.3);}
.lab-block .t {font-weight: 600; font-size: .9rem;}
.lab-block .s {font-size: .8rem; opacity: .75;}
.lab-row {display: flex; gap: 6px; flex-wrap: wrap;}
.lab-row > .lab-block {flex: 1 1 220px;}
.lab-unit {display: inline-block; width: 16px; height: 16px; border-radius: 4px;
  margin: 2px 2px 0 0;}
.lab-arrow {text-align: center; opacity: .45; font-size: .8rem; line-height: .8;}
</style>
"""


def style() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def chip(status: str, label: str | None = None) -> str:
    """A colored pill with an icon; LABEL overrides the status word."""
    color = STATUS_COLOR.get(status, "#6e7781")
    label = html.escape((label or status).replace("_", " ").upper())
    return (
        f'<span class="pb-chip" style="background:{color}">'
        f"{STATUS_ICON.get(status, '?')} {label}</span>"
    )


def num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return html.escape(str(value))


def compact(value: Any) -> str:
    """12,345,678 → 12.3M; None → –."""
    if value is None:
        return "–"
    value = float(value)
    for unit, size in (("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= size:
            return f"{value / size:.1f}{unit}"
    return f"{value:g}"


def page_header(title: str, question: str, caption: str | None = None) -> None:
    """Every lab page opens with the question it answers."""
    style()
    st.header(title)
    st.markdown(f"<p class='lab-q'>{html.escape(question)}</p>", unsafe_allow_html=True)
    if caption:
        st.caption(caption)


def empty_state(title: str, why: str, commands: Sequence[str]) -> None:
    """A friendly dead end that names the command that fills the page."""
    st.markdown(
        f"<div class='lab-empty'><h4>{html.escape(title)}</h4>"
        f"<p>{html.escape(why)}</p></div>",
        unsafe_allow_html=True,
    )
    for command in commands:
        st.code(command, language="bash")


def step_card(title: str, why: str, command: str | None = None) -> None:
    st.markdown(
        f"<div class='lab-card'><h4>{html.escape(title)}</h4>"
        f"<p>{html.escape(why)}</p></div>",
        unsafe_allow_html=True,
    )
    if command:
        st.code(command, language="bash")


def explainer(label: str, text: str, expanded: bool = False) -> None:
    with st.expander(f"ⓘ {label}", expanded=expanded):
        st.markdown(text)


def figure_layout(figure: Any, height: int = 320, **extra: Any) -> Any:
    figure.update_layout(
        height=height,
        margin={"l": 10, "r": 10, "t": 44, "b": 10},
        font={"size": 12},
        legend={"orientation": "h", "y": -0.18},
        **extra,
    )
    return figure
