"""The lab pages: Home, Experiments, Models, Behaviors, Explorer.

Each page answers one researcher question, opens verdict-first, explains its
numbers in plain words and, when empty, names the command that fills it.
Data comes from :mod:`klab.dashboard.lab_data` (verified records only).
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from klab.dashboard import explorer_view as xv
from klab.dashboard import lab_data
from klab.dashboard.probe_data import ProbeEntry, probe_row
from klab.dashboard.probes import (
    _comparison_chart,
    _explain,
    _history,
    _live,
    _metrics,
    _pareto,
    _probe_table,
    _verdict_banner,
)
from klab.dashboard.ui import (
    SERIES,
    STATUS_COLOR,
    chip,
    compact,
    empty_state,
    explainer,
    figure_layout,
    num,
    page_header,
    step_card,
)
from klab.probes.points import METRICS, consistent_chance
from klab.probes.suite import suite_identity
from klab.reference_models import REFERENCES

PageLinks = Mapping[str, Any]


def _snapshot(lab_dir: Path) -> lab_data.LabSnapshot:
    return lab_data.snapshot(lab_dir)


def _when(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value).astimezone()
    except ValueError:
        return value
    return moment.strftime("%b %d %H:%M %Z")


def _rejected(snap: lab_data.LabSnapshot) -> None:
    if snap.rejected:
        with st.expander(
            f"⚠ {len(snap.rejected)} record(s) rejected (edited or unreadable)"
        ):
            st.caption(
                "Rejected records failed the seal check and are never shown as "
                "results. Re-run the command that produced them."
            )
            for path, reason in snap.rejected:
                st.markdown(f"`{html.escape(str(path))}`: {html.escape(reason)}")


def _result_picker(entries: Sequence[ProbeEntry], key: str) -> ProbeEntry:
    index = st.selectbox(
        "Result",
        range(len(entries)),
        format_func=lambda i: entries[i].label,
        key=key,
        help="Newest first. Each result is one probe battery: a lab try's arms "
        "or a `klab probe` run.",
    )
    return entries[index]


def _activity_table(rows: list[dict[str, Any]]) -> None:
    body = []
    for row in rows:
        verdict = (
            chip(str(row["verdict"]))
            if row.get("verdict")
            else "<span class='pb-hint'>no probes</span>"
        )
        delta = row.get("loss_delta")
        delta_text = "" if delta is None else f"Δ loss {delta:+.4f}"
        body.append(
            "<tr>"
            f"<td>{verdict}</td>"
            f"<td><b>{html.escape(str(row['what']))}</b><br>"
            f"<span class='pb-hint'>{html.escape(row['kind'])} · "
            f"{html.escape(row['id'])}</span></td>"
            f"<td>{html.escape(str(row.get('outcome') or ''))}<br>"
            f"<span class='pb-hint'>{delta_text}</span></td>"
            f"<td class='pb-hint'>{html.escape(_when(row['when']))}</td>"
            "</tr>"
        )
    st.markdown(
        "<table class='pb-table'><tr><th>Verdict</th><th>What</th><th>Outcome</th>"
        "<th>When</th></tr>" + "".join(body) + "</table>",
        unsafe_allow_html=True,
    )


# --- Home -------------------------------------------------------------------


def home_page(lab_dir: Path, runs_dirs: Sequence[Path], pages: PageLinks) -> None:
    page_header(
        "Lab home",
        "What happened, and what should I do next?",
        f"lab root {lab_dir}",
    )
    _live(lab_dir)
    snap = _snapshot(lab_dir)
    _rejected(snap)
    if snap.empty:
        empty_state(
            "Your lab is empty",
            "A lab try trains a baseline and one change side by side, scores both "
            "on the same held-out split and runs the fast probe battery. The "
            "reference models below are already scored, so the Models page works "
            "before you train anything.",
            [
                "klab try DELTA.yaml --vs BASELINE.yaml",
                "klab compare --list-references",
            ],
        )
        _page_links(pages)
        return
    top = st.columns([3, 2])
    with top[0]:
        if snap.entries:
            latest = snap.entries[0]
            st.subheader("Latest verdict")
            st.caption(latest.label)
            _verdict_banner(latest.result)
        else:
            st.subheader("Latest try")
            _activity_table(lab_data.activity(snap, 1))
    with top[1]:
        st.subheader("Next steps")
        for step in lab_data.next_steps(snap, runs_dirs):
            step_card(step["title"], step["why"], step["command"] or None)
            if step.get("plan"):
                st.code(step["plan"], language="json")
    references = {p["run_id"] for p in snap.points if p["kind"] == "reference"}
    checkpoints = {p.get("checkpoint_sha256") for p in snap.points}
    cols = st.columns(5)
    cols[0].metric("Lab tries", len(snap.tries), help="`klab try` records")
    cols[1].metric(
        "Probe batteries",
        len(snap.entries),
        help="Probe results (from tries and `klab probe`)",
    )
    cols[2].metric(
        "Scored checkpoints",
        len(checkpoints - {None}),
        help="Checkpoints with sealed lab scores, including public references. Not the count of available training snapshots.",
    )
    cols[3].metric(
        "Reference models",
        len(references),
        help="Pinned public checkpoints with sealed scores",
    )
    cols[4].metric(
        "Explorations", len(snap.explorations), help="Cached `klab explore` views"
    )
    st.subheader("Recent activity")
    _activity_table(lab_data.activity(snap))
    _page_links(pages)


def _page_links(pages: PageLinks) -> None:
    if not pages:
        return
    st.subheader("Where to look")
    blurbs = {
        "experiments": "Did my change help? Verdicts, probe results, history, compare.",
        "models": "Where does each checkpoint sit? Catalog, Pareto frontier, references.",
        "behaviors": "What does it actually do? Generations, recall misses, needle, calibration.",
        "explorer": "What is inside? Architecture, weights, attention, routing, memory.",
    }
    cols = st.columns(len(blurbs))
    for col, (name, blurb) in zip(cols, blurbs.items(), strict=True):
        if name in pages:
            with col:
                st.page_link(pages[name], label=pages[name].title, icon=None)
                st.caption(blurb)


# --- Experiments ------------------------------------------------------------


def experiments_page(lab_dir: Path) -> None:
    identity = suite_identity()
    page_header(
        "Experiments",
        "Did my change help, and is the evidence good enough to act on?",
        f"{identity['name']} v{identity['version']} · suite "
        f"{identity['sha256'][:12]} · verdicts use the held-out item split only",
    )
    _live(lab_dir)
    snap = _snapshot(lab_dir)
    _rejected(snap)
    if not snap.entries:
        empty_state(
            "No probe results yet",
            "Run a lab try (the fast probe tier runs on both arms automatically) "
            "or probe an existing run against a baseline.",
            [
                "klab try DELTA.yaml --vs BASELINE.yaml",
                "klab probe RUN --vs BASELINE_RUN --config standard",
            ],
        )
        _unprobed_tries(snap)
        _explain()
        return
    entry = _result_picker(snap.entries, "experiments_result")
    result = entry.result
    target = result.get("target") or {}
    baseline = result.get("baseline") or {}
    st.caption(
        f"candidate **{target.get('run_id')}** (step {target.get('step')}) · "
        + (
            f"baseline **{baseline.get('run_id')}** · "
            if baseline
            else "no baseline · "
        )
        + f"suite {result['suite']['sha256'][:12]}"
    )
    if entry.delta:
        st.caption(
            "change: "
            + ", ".join(
                f"`{k}` {v.get('base')!r} → {v.get('variant')!r}"
                for k, v in entry.delta.items()
            )
            + (
                f" · try verdict: {lab_data.TRY_VERDICT_TEXT.get(str(entry.comparison.get('verdict')), entry.comparison.get('verdict'))}"
                if entry.comparison
                else ""
            )
        )
    _verdict_banner(result)
    _metrics(result)
    results_tab, trends_tab, compare_tab, explain_tab = st.tabs(
        ["Results", "History & trends", "Compare", "What the probes mean"]
    )
    with results_tab:
        _probe_table(result)
        _comparison_chart(result)
    with trends_tab:
        _trends(snap.entries)
        st.markdown("**All results**")
        _history(snap.entries)
        _unprobed_tries(snap)
    with compare_tab:
        _compare(entry, snap)
    with explain_tab:
        _explain()


def _unprobed_tries(snap: lab_data.LabSnapshot) -> None:
    rows = lab_data.tries_without_probes(snap)
    if rows:
        with st.expander(f"{len(rows)} lab try(s) without a probe battery"):
            st.caption(
                "These tries recorded held-out loss only (`--config no-probes` or an "
                "older version). Probe the candidate to get a verdict:"
            )
            for row in rows:
                st.markdown(f"**{html.escape(str(row['what']))}** · {row['id']}")
                st.code(row["command"], language="bash")


def _trends(entries: Sequence[ProbeEntry]) -> None:
    frame = pd.DataFrame(lab_data.probe_trends(entries))
    if frame.empty:
        st.info("No probe values yet.")
        return
    titles = dict(zip(frame["probe"], frame["title"], strict=False))
    probe = st.selectbox(
        "Probe",
        list(titles),
        format_func=lambda p: titles[p],
        key="experiments_trend_probe",
    )
    part = frame[frame["probe"] == probe].copy()
    part["when"] = pd.to_datetime(part["when"], errors="coerce", utc=True)
    # Only results measured the same way (suite + eval/benchmark/item group)
    # share a scale; a trend never joins points across groups.
    ungrouped = int(part["trend_group"].isna().sum())
    part = part.dropna(subset=["trend_group"]).sort_values("when")
    if part.empty:
        st.info(
            "No result of this probe records its comparison group; re-score to "
            "plot a trend."
        )
        return
    counts = part["trend_group"].value_counts()
    groups = list(dict.fromkeys(reversed(part["trend_group"].tolist())))
    label = METRICS[probe].group_label if probe in METRICS else "comparison group"
    group = st.selectbox(
        f"Comparison group (suite · {label})",
        groups,
        index=0,
        key=f"experiments_trend_group_{probe}",
        format_func=lambda g: (
            f"suite {g[:8]} · {g.split(':', 1)[1][:12]} · {int(counts[g])} result(s)"
        ),
        help="Results from different probe suites, datasets/tokenizers, "
        "benchmark tasks/limits or item sets are not on one scale; pick one.",
    )
    notes = []
    if len(groups) > 1:
        notes.append(f"{len(groups) - 1} other comparison group(s) hidden")
    if ungrouped:
        notes.append(f"{ungrouped} older result(s) recorded no group")
    if notes:
        st.caption(" · ".join(notes))
    part = part[part["trend_group"] == group]
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=part["when"],
            y=part["value"],
            mode="markers+lines",
            name="candidate",
            line={"color": SERIES["candidate"]},
            marker={
                "size": 12,
                "color": [STATUS_COLOR.get(s, "#6e7781") for s in part["status"]],
                "line": {"width": 1, "color": "#24292f"},
            },
            customdata=part[["id", "status", "suite"]],
            hovertemplate="%{customdata[0]}<br>%{y:.4f} · %{customdata[1]} · suite "
            "%{customdata[2]}<extra></extra>",
        )
    )
    based = part.dropna(subset=["baseline_value"])
    if not based.empty:
        figure.add_trace(
            go.Scatter(
                x=based["when"],
                y=based["baseline_value"],
                mode="markers",
                name="baseline",
                marker={"size": 9, "symbol": "x", "color": SERIES["baseline"]},
            )
        )
    better = "higher" if part["higher_is_better"].iloc[0] else "lower"
    st.plotly_chart(
        figure_layout(
            figure,
            height=320,
            title=f"{titles[probe]} per result ({better} is better; dot color = verdict)",
        ),
        use_container_width=True,
    )


def _compare(entry: ProbeEntry, snap: lab_data.LabSnapshot) -> None:
    from klab.probes import compare as compare_mod

    st.caption(
        "Paired deltas with standard errors, one metric at a time and only "
        "inside one comparison group; anything else is shown as not comparable, "
        "never mixed in. Same engine as `klab compare`."
    )
    others = [e.key for e in snap.entries if e.key != entry.key]
    cols = st.columns([3, 1])
    chosen = cols[0].multiselect(
        "Compare against", others, default=others[:2], key="experiments_compare_with"
    )
    references = cols[1].toggle(
        "Reference models", value=True, key="experiments_compare_refs"
    )
    if not chosen and not references:
        st.info("Pick results or switch on the reference models.")
        return
    try:
        report = compare_mod.compare(
            entry.key, chosen, lab_dir=snap.lab_dir, references=references
        )
    except (ValueError, KeyError) as error:
        st.warning(f"Cannot compare: {error}")
        return
    body = []
    for comparison in report["comparisons"]:
        cells = []
        for item in comparison["pairs"]:
            word = compare_mod._verdict_word(item["status"], item["metric"])
            tone = {
                "better": "pass",
                "worse": "fail",
                "within_noise": "info",
                "not_comparable": "not_comparable",
                "missing_evidence": "skipped",
            }.get(word, "info")
            delta = (
                ""
                if item.get("delta") is None
                else f"{item['delta']:+.4f}"
                + ("" if item.get("se") is None else f" ±{item['se']:.4f}")
            )
            cells.append(
                f"<td>{chip(tone, word)}"
                f"<br>{delta}<br><span class='pb-hint'>"
                f"{html.escape(str(item.get('note') or ''))}</span></td>"
            )
        body.append(
            f"<tr><td><b>{html.escape(comparison['other'])}</b></td>{''.join(cells)}</tr>"
        )
    header = "".join(f"<th>{html.escape(m.label)}</th>" for m in METRICS.values())
    st.markdown(
        f"<p><b>{html.escape(report['subject']['label'])}</b> vs …</p>"
        f"<table class='pb-table'><tr><th>Other</th>{header}</tr>{''.join(body)}</table>",
        unsafe_allow_html=True,
    )
    explainer(
        "How to read this",
        "**better / worse** means the difference is larger than two paired "
        "standard errors. **within noise** means it is not. **not comparable** "
        "means the two values were measured on different data, tokenizers or "
        "items (their comparison group differs), so no delta is shown. "
        "**missing evidence** names the command that would measure it.",
    )


# --- Models -----------------------------------------------------------------


def models_page(lab_dir: Path, runs_dirs: Sequence[Path]) -> None:
    page_header(
        "Models",
        "Where does each checkpoint sit on cost vs quality, next to known models?",
        "every verified checkpoint in the lab, plus pinned public references",
    )
    snap = _snapshot(lab_dir)
    _rejected(snap)
    catalog_tab, pareto_tab, refs_tab, unscored_tab = st.tabs(
        ["Catalog", "Pareto frontier", "Reference models", "Not yet scored"]
    )
    with catalog_tab:
        rows = lab_data.checkpoint_catalog(snap.points)
        lab_rows = [r for r in rows if r["kind"] != "reference"]
        if not lab_rows:
            empty_state(
                "No lab checkpoints yet",
                "Checkpoints appear here once a try or a probe scores them.",
                ["klab try DELTA.yaml --vs BASELINE.yaml"],
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            st.dataframe(
                frame[
                    [
                        "label",
                        "kind",
                        "parameters",
                        "active_parameters",
                        "tokens_seen",
                        "heldout_loss",
                        "fact_recall",
                        "lm_eval",
                        "checkpoint",
                        "sources",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
                height=min(36 * (len(frame) + 1) + 4, 640),
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
                    "heldout_loss": st.column_config.NumberColumn(
                        "held-out loss ↓", format="%.4f"
                    ),
                    "fact_recall": st.column_config.ProgressColumn(
                        "fact recall ↑", min_value=0.0, max_value=1.0, format="%.2f"
                    ),
                    "lm_eval": st.column_config.ProgressColumn(
                        "lm-eval acc ↑", min_value=0.0, max_value=1.0, format="%.3f"
                    ),
                    "checkpoint": st.column_config.TextColumn("sha256"),
                },
            )
            st.caption(
                "Newest value per metric. Values in one column are only comparable "
                "inside one comparison group: use the Pareto tab or "
                "`klab compare` for that."
            )
    with pareto_tab:
        entry = snap.entries[0] if snap.entries else None
        _pareto(snap.points, entry)
    with refs_tab:
        _references(snap)
    with unscored_tab:
        runs = lab_data.unscored_runs(runs_dirs, snap.points)
        if not runs:
            st.success(
                "Every run in the run directories has been scored by a lab record."
            )
        else:
            st.caption(
                "Plain `klab train` runs that no try or probe has scored, so "
                "they are missing from the catalog and the Pareto view. Probe one "
                "to place it (add `--config full` for lm-eval accuracy next to the "
                "references):"
            )
            for run in runs:
                st.markdown(f"**{html.escape(run['run_id'])}** · `{run['path']}`")
                st.code(run["command"], language="bash")


def _references(snap: lab_data.LabSnapshot) -> None:
    by_label: dict[str, dict[str, Any]] = {}
    for point in snap.points:
        if point["kind"] == "reference":
            by_label.setdefault(str(point.get("run_id")), point)
    rows = []
    for name, ref in REFERENCES.items():
        point = by_label.get(f"ref:{name}") or {}
        metrics = point.get("metrics") or {}
        rows.append(
            {
                "reference": f"ref:{name}",
                "family": ref.family,
                "parameters": point.get("parameters"),
                "training tokens": ref.training_tokens,
                "lm-eval acc": (metrics.get("lm_eval") or {}).get("value"),
                "fact recall": (metrics.get("fact_recall") or {}).get("value"),
                "revision": ref.revision[:10],
                "scored by": "packaged result"
                if point.get("packaged")
                else point.get("source"),
            }
        )
    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
        column_config={
            "parameters": st.column_config.NumberColumn(format="compact"),
            "training tokens": st.column_config.NumberColumn(format="compact"),
            "lm-eval acc": st.column_config.NumberColumn(format="%.3f"),
            "fact recall": st.column_config.NumberColumn(format="%.3f"),
        },
    )
    explainer(
        "Why references, and what they can and cannot tell you",
        "Pinned public checkpoints, scored by the same lm-eval and text-level "
        "fact-recall paths as lab runs and shipped as sealed records, so a lab "
        "result sits on a known curve without downloading anything. They were "
        "trained on 10⁵–10⁶× more tokens with their own tokenizers, so they "
        "never appear on our held-out loss. The needle probe is text-level "
        "(fixed character lengths) from suite v4 on, so a re-scored reference "
        "is comparable on needle too. Re-score one yourself "
        "with `klab probe ref:NAME --config full` (reference + lmeval extras).",
        expanded=False,
    )


# --- Behaviors --------------------------------------------------------------


def behaviors_page(lab_dir: Path) -> None:
    page_header(
        "Behaviors",
        "What does the model actually do, and where does it fail?",
        "outputs and per-item outcomes recorded by the probe battery",
    )
    snap = _snapshot(lab_dir)
    _rejected(snap)
    if not snap.entries:
        empty_state(
            "No behavior evidence yet",
            "Generations, recall misses and calibration come from the standard "
            "probe tier.",
            ["klab probe RUN --config standard"],
        )
        return
    entry = _result_picker(snap.entries, "behaviors_result")
    gen_tab, recall_tab, needle_tab, calib_tab = st.tabs(
        ["Generations", "Fact recall", "Needle in a haystack", "Calibration"]
    )
    with gen_tab:
        _generations(entry)
    with recall_tab:
        _recall(entry)
    with needle_tab:
        _needle(snap.entries)
    with calib_tab:
        _calibration(entry)


def _generations(entry: ProbeEntry) -> None:
    samples = lab_data.generations(entry)
    row = probe_row(entry.result, "repetition") or {}
    if not samples:
        empty_state(
            "No generations in this result",
            "The degeneration probe (fast tier) records greedy continuations.",
            [f"klab probe {(entry.result.get('target') or {}).get('run_id')}"],
        )
        return
    details = row.get("details") or {}
    cols = st.columns(3)
    cols[0].metric(
        "Repeated 4-grams (seq-rep-4)",
        num(row.get("value")),
        None if row.get("delta") is None else f"{row['delta']:+.3f} vs baseline",
        delta_color="inverse",
        help="Share of 4-grams in the continuation that already appeared in it. "
        "Near 1 means the model loops.",
    )
    cols[1].metric("Distinct unigrams", num(details.get("distinct_1"), 2))
    cols[2].metric("Distinct bigrams", num(details.get("distinct_2"), 2))
    if (row.get("value") or 0) > 0.5:
        st.warning(
            "▲ Degeneration: more than half of the 4-grams repeat. Typical for "
            "tiny or under-trained models; check longer training before judging "
            "the idea."
        )
    has_base = any(s["baseline"] for s in samples)
    for sample in samples:
        st.markdown(f"**Prompt:** {html.escape(sample['prompt'])}")
        cols = st.columns(2 if has_base else 1)
        cols[0].markdown(
            f"<div class='lab-card'><p class='pb-hint'>candidate</p>"
            f"{html.escape(sample['candidate'])}</div>",
            unsafe_allow_html=True,
        )
        if has_base:
            cols[1].markdown(
                f"<div class='lab-card'><p class='pb-hint'>baseline</p>"
                f"{html.escape(sample['baseline'] or '–')}</div>",
                unsafe_allow_html=True,
            )


def _items_table(items: list[dict[str, Any]], show_prompt: bool) -> None:
    body = []
    for item in items:
        credit = item.get("credit")
        tone = "pass" if credit == 1 else "fail" if credit == 0 else "warn"
        base = item.get("baseline_credit")
        base_cell = (
            ""
            if base is None
            else f"<td>{html.escape(str(item.get('baseline_picked')))} "
            f"<span class='pb-hint'>({base:.2f})</span></td>"
        )
        body.append(
            "<tr>"
            f"<td>{chip(tone, 'hit' if credit == 1 else 'miss' if credit == 0 else 'tie')}</td>"
            + (
                f"<td>{html.escape(str(item.get('prompt', '')))}</td>"
                if show_prompt
                else ""
            )
            + f"<td><b>{html.escape(str(item.get('answer')))}</b></td>"
            f"<td>{html.escape(str(item.get('picked')))} "
            f"<span class='pb-hint'>({num(credit, 2)})</span></td>{base_cell}</tr>"
        )
    has_base = any(i.get("baseline_credit") is not None for i in items)
    st.markdown(
        "<table class='pb-table'><tr><th>Outcome</th>"
        + ("<th>Asked</th>" if show_prompt else "")
        + "<th>Expected</th><th>Candidate picked</th>"
        + ("<th>Baseline picked</th>" if has_base else "")
        + "</tr>"
        + "".join(body)
        + "</table>",
        unsafe_allow_html=True,
    )


def _recall(entry: ProbeEntry) -> None:
    shown = False
    for probe_id, title, explain in (
        (
            "fact_recall",
            "In-context fact recall",
            (
                "The fact is stated in the prompt, then asked about after a "
                "distractor. Tests whether the model can use its context."
            ),
        ),
        (
            "parametric_recall",
            "Closed-book fact recall (from weights)",
            (
                "The run's own withheld-facts training facts (its "
                "`dataset.synthetic_seed` manifest), asked with no context; the "
                "never-trained control facts should stay at chance. Only runs "
                "trained on `withheld_facts` are scored."
            ),
        ),
    ):
        row = probe_row(entry.result, probe_id)
        if not row:
            continue
        if row.get("value") is None:
            if row.get("note"):
                # Inapplicable (skipped) or missing evidence: say why.
                st.caption(f"{title}: {row.get('status')} · {row['note']}")
            continue
        shown = True
        details = row.get("details") or {}
        st.markdown(f"#### {title}")
        st.caption(explain)
        cols = st.columns(4)
        cols[0].metric(
            "Accuracy (held-out)", num(row["value"]), help="Fractional credit for ties."
        )
        cols[1].metric("Chance", num(details.get("chance"), 2))
        cols[2].metric("Dev split", num(details.get("dev_accuracy")))
        if probe_id == "parametric_recall":
            cols[3].metric(
                "Never-trained control", num(details.get("control_accuracy"))
            )
        else:
            cols[3].metric("Items", details.get("n", "–"))
        if details.get("constant_pick") or details.get("baseline_constant_pick"):
            st.caption(
                "▲ Same pick for every question in each candidate set: the "
                "model ranks answers by a context-free prior, not by the "
                "prompt (typical of untrained models). Accuracy is chance by "
                "construction; candidate order plays no part."
            )
        items = lab_data.recall_items(entry, probe_id)
        if items:
            misses = [i for i in items if (i.get("credit") or 0) < 1]
            only = st.toggle(
                f"Only misses ({len(misses)} of {len(items)})",
                value=bool(misses),
                key=f"behaviors_misses_{probe_id}",
            )
            _items_table(misses if only else items, show_prompt=True)
        else:
            st.caption(
                "This result predates per-item records; re-probe to see each item."
            )
    if not shown:
        target = (entry.result.get("target") or {}).get("run_id")
        empty_state(
            "No fact-recall evidence in this result",
            "Fact recall runs in the standard tier.",
            [f"klab probe {target} --config standard"],
        )


def _needle(entries: Sequence[ProbeEntry]) -> None:
    frame = pd.DataFrame(lab_data.needle_curves(entries))
    if frame.empty:
        empty_state(
            "No needle results yet",
            "The needle probe (standard tier) hides a fact at growing distances "
            "in the context and asks for it at the end.",
            ["klab probe RUN --config standard"],
        )
        return
    # Only one item group (same items and lengths) is drawn at a time.
    ungrouped = int(frame["group"].isna().sum())
    frame = frame.dropna(subset=["group"])
    if frame.empty:
        st.info(
            "No needle result records its item group; re-probe to plot "
            "retrieval by length."
        )
        return
    frame = frame.sort_values("when", ascending=False)
    groups = list(dict.fromkeys(frame["group"]))
    counts = frame.groupby("group")["checkpoint_sha256"].nunique()
    group = st.selectbox(
        "Comparison group (item group)",
        groups,
        index=0,
        key="behaviors_needle_group",
        format_func=lambda g: f"{g[:12]} · {int(counts[g])} checkpoint(s)",
        help="Needle items have fixed character lengths (older results: each "
        "model's own tokens); only results with the same items and lengths "
        "share a scale.",
    )
    notes = []
    if len(groups) > 1:
        notes.append(f"{len(groups) - 1} other item group(s) hidden")
    if ungrouped:
        notes.append(f"{ungrouped} older point(s) recorded no item group")
    st.caption(
        "Retrieval accuracy by context length, one line per checkpoint (newest "
        "result of each) within one item group."
        + (" " + " · ".join(notes) if notes else "")
    )
    frame = frame[frame["group"] == group].sort_values(["checkpoint", "length"])
    unit = str(frame["unit"].iloc[0]) if len(frame) else "characters"
    figure = px.line(
        frame,
        x="length",
        y="accuracy",
        color="checkpoint",
        markers=True,
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    chance = consistent_chance(
        frame.drop_duplicates("checkpoint_sha256")["chance"].tolist()
    )
    if chance is not None:
        figure.add_hline(y=chance, line_dash="dot", annotation_text="chance")
    st.plotly_chart(
        figure_layout(
            figure,
            height=380,
            title="Needle retrieval vs context length",
            yaxis_range=[0, 1.05],
            xaxis_title=f"context {unit}",
        ),
        use_container_width=True,
    )


def _calibration(entry: ProbeEntry) -> None:
    rows = lab_data.calibration_curves(entry)
    row = probe_row(entry.result, "calibration") or {}
    if not rows:
        empty_state(
            "No reliability bins in this result",
            "Calibration (fast tier) records them from suite v3 on; re-probe to "
            "draw the diagram.",
            [f"klab probe {(entry.result.get('target') or {}).get('run_id')}"],
        )
        return
    frame = pd.DataFrame(rows)
    st.metric(
        "Expected calibration error",
        num(row.get("value")),
        None if row.get("delta") is None else f"{row['delta']:+.3f} vs baseline",
        delta_color="inverse",
        help="Average gap between how confident the top guess is and how often "
        "it is right, weighted by how many tokens fall in each bin.",
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dot", "color": "#8c959f"},
            name="perfectly calibrated",
        )
    )
    for arm in ("baseline", "candidate"):
        part = frame[frame["arm"] == arm]
        if part.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=part["confidence"],
                y=part["accuracy"],
                mode="lines+markers",
                name=arm,
                line={"color": SERIES[arm]},
                marker={"size": 6 + 30 * part["share"]},
                customdata=part[["share", "low", "high"]],
                hovertemplate="confidence %{x:.2f} → accuracy %{y:.2f}<br>"
                "%{customdata[0]:.1%} of tokens in [%{customdata[1]:.2f}, "
                "%{customdata[2]:.2f})<extra></extra>",
            )
        )
    st.plotly_chart(
        figure_layout(
            figure,
            height=420,
            title="Reliability diagram (marker size = share of tokens)",
            xaxis_title="confidence of the top guess",
            yaxis_title="how often it was right",
            xaxis_range=[0, 1],
            yaxis_range=[0, 1],
        ),
        use_container_width=True,
    )
    st.caption(
        "Below the diagonal: over-confident. Above: under-confident. Tiny models "
        "often sit in the low-confidence corner."
    )


# --- Explorer ---------------------------------------------------------------


def _generation_step(gen_dir: Path) -> Any:
    try:
        metadata = json.loads((gen_dir / "manifest.json").read_text(encoding="utf-8"))
        if isinstance(metadata, Mapping) and metadata.get("step") is not None:
            return metadata["step"]
    except OSError, json.JSONDecodeError:
        pass
    parts = gen_dir.name.split("_")
    if len(parts) >= 2 and parts[0] == "step" and parts[1].isdigit():
        return int(parts[1])
    return "–"


def _explorable_runs(
    snap: lab_data.LabSnapshot, runs_dirs: Sequence[Path]
) -> list[dict[str, str]]:
    choices: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in (snap.lab_dir / "runs", *runs_dirs):
        if not root or not root.is_dir():
            continue
        for run in sorted(root.iterdir()):
            checkpoints = run / "checkpoints"
            if not run.is_dir() or not checkpoints.is_dir():
                continue
            for gen in sorted(checkpoints.glob("step_*_gen_*")):
                if not gen.is_dir() or not (gen / "manifest.json").is_file():
                    continue
                spec = str(gen.resolve())
                if spec in seen:
                    continue
                seen.add(spec)
                step = _generation_step(gen)
                choices.append(
                    {
                        "spec": spec,
                        "label": f"{run.name} · {gen.name} · step {step}",
                    }
                )
    for path, record in snap.tries:
        spec = str(path.resolve())
        if spec in seen:
            continue
        cand = (record.get("arms") or {}).get("candidate") or {}
        if cand.get("reference") or str(cand.get("run_id") or "").startswith("ref:"):
            continue
        seen.add(spec)
        run_id = cand.get("run_id") or "unknown-run"
        try_id = record.get("try_id") or path.parent.name
        step = cand.get("step") if cand.get("step") is not None else "–"
        ckpt = cand.get("checkpoint")
        gen_part = f" ({Path(str(ckpt)).name})" if ckpt else ""
        choices.append(
            {
                "spec": spec,
                "label": f"{run_id} · try {try_id}{gen_part} · step {step}",
            }
        )
    for path, record in snap.probes:
        spec = str(path.resolve())
        if spec in seen:
            continue
        target = record.get("target") or {}
        if target.get("reference") or str(target.get("run_id") or "").startswith(
            "ref:"
        ):
            continue
        seen.add(spec)
        run_id = target.get("run_id") or "unknown-run"
        probe_id = record.get("probe_id") or path.parent.name
        step = target.get("step") if target.get("step") is not None else "–"
        ckpt = target.get("checkpoint")
        gen_part = f" ({Path(str(ckpt)).name})" if ckpt else ""
        choices.append(
            {
                "spec": spec,
                "label": f"{run_id} · probe {probe_id}{gen_part} · step {step}",
            }
        )
    return choices


def explorer_page(lab_dir: Path, runs_dirs: Sequence[Path]) -> None:
    from klab import explorer as ex

    page_header(
        "Model explorer",
        "What is inside this model, and how does it process a sentence?",
        f"small checkpoints only (≤ {ex.MAX_PARAMETERS / 1e6:.0f}M parameters, "
        f"≤ {ex.MAX_TOKENS} tokens) · computed through the verified checkpoint "
        "loader (runtime shown per exploration) and cached under LAB/explorer",
    )
    st.markdown(
        "For text completion from the available larger snapshot, open [Generate](generate). Explorer is for small-model internal diagnostics."
    )
    snap = _snapshot(lab_dir)
    cached = snap.explorations
    choices = _explorable_runs(snap, runs_dirs)
    if not cached and not choices:
        empty_state(
            "Nothing to explore yet",
            "Train or try a model first; then explore it here or from the CLI.",
            ["klab try DELTA.yaml --vs BASELINE.yaml", "klab explore RUN"],
        )
        return
    source = st.radio(
        "Show",
        ["Cached exploration", "Explore a run", "Compare checkpoints"],
        horizontal=True,
        key="explorer_source",
        index=0 if cached else 1,
    )
    label_map = {c["spec"]: c["label"] for c in choices}
    specs = [c["spec"] for c in choices]
    exploration: dict[str, Any] | None = None
    if source == "Cached exploration":
        if not cached:
            st.info("No cached explorations yet; pick “Explore a run”.")
            return
        index = st.selectbox(
            "Exploration",
            range(len(cached)),
            format_func=lambda i: (
                f"{(cached[i].get('target') or {}).get('run_id')} · "
                f"“{cached[i]['text'][:48]}” · {_when(cached[i]['created_at'])} · "
                f"{ex.runtime_label(cached[i])}"
                + (" · ⚠ unverified" if cached[i].get("cache_issue") else "")
            ),
            key="explorer_cached",
        )
        exploration = cached[index]
        render_exploration(exploration)
        return
    if source == "Compare checkpoints":
        if not specs:
            st.info("No local checkpoint generations or pinned records to compare.")
            return
        cols = st.columns([2, 2, 3])
        left_spec = cols[0].selectbox(
            "Left checkpoint",
            specs,
            index=0,
            format_func=lambda s: label_map.get(s, s),
            key="explorer_compare_left",
        )
        right_spec = cols[1].selectbox(
            "Right checkpoint",
            specs,
            index=1 if len(specs) > 1 else 0,
            format_func=lambda s: label_map.get(s, s),
            key="explorer_compare_right",
        )
        text = cols[2].text_input(
            "Sample text",
            value="",
            key="explorer_compare_text",
            max_chars=600,
        )
        saved_compare = st.session_state.get("explorer_compare_last")
        if saved_compare is not None and (
            saved_compare.get("left_spec") != left_spec
            or saved_compare.get("right_spec") != right_spec
            or saved_compare.get("text") != text
        ):
            st.session_state.pop("explorer_compare_last", None)
            saved_compare = None
        if st.button("Compare", type="primary", key="explorer_compare_go"):
            st.session_state.pop("explorer_compare_last", None)
            saved_compare = None
            if left_spec == right_spec:
                st.warning("Pick two distinct checkpoints to compare.")
                return
            with st.spinner(
                "Loading both checkpoints on CPU and running paired forward passes…"
            ):
                try:
                    left_exp, _ = ex.explore(
                        left_spec, lab_dir=lab_dir, text=text, backend="cpu"
                    )
                    right_exp, _ = ex.explore(
                        right_spec, lab_dir=lab_dir, text=text, backend="cpu"
                    )
                    if left_exp.get("unavailable") or right_exp.get("unavailable"):
                        reason = left_exp.get("unavailable") or right_exp.get(
                            "unavailable"
                        )
                        st.warning(f"⊘ Cannot compare: {reason}")
                        return
                    comparison = ex.compare_explorations(left_exp, right_exp)
                except ex.ExplorerUnavailable as error:
                    st.warning(str(error))
                    return
                except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
                    st.error(f"Could not compare checkpoints: {error}")
                    return
            saved_compare = {
                "left_spec": left_spec,
                "right_spec": right_spec,
                "text": text,
                "comparison": comparison,
            }
            st.session_state["explorer_compare_last"] = saved_compare
        if saved_compare is None:
            st.caption(
                "Pick two distinct pinned checkpoints and a shared sentence, "
                "then press Compare."
            )
            return
        render_comparison(saved_compare["comparison"])
        return
    if not specs:
        st.info("No local checkpoint generations or pinned records to explore.")
        return
    cols = st.columns([2, 3])
    run = cols[0].selectbox(
        "Run",
        specs,
        format_func=lambda s: label_map.get(s, s),
        key="explorer_run",
    )
    text = cols[1].text_input(
        "Sample text", value="", key="explorer_text", max_chars=600
    )
    saved_single = st.session_state.get("explorer_last")
    if isinstance(saved_single, Mapping) and "exploration" in saved_single:
        if saved_single.get("spec") == run and saved_single.get("text") == text:
            exploration = saved_single["exploration"]
        else:
            st.session_state.pop("explorer_last", None)
    if st.button("Explore", type="primary", key="explorer_go"):
        st.session_state.pop("explorer_last", None)
        exploration = None
        with st.spinner("Loading the checkpoint and running one forward pass…"):
            try:
                exploration, _ = ex.explore(
                    run, lab_dir=lab_dir, text=text, backend="cpu"
                )
            except ex.ExplorerUnavailable as error:
                st.warning(str(error))
                return
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
                st.error(f"Could not explore {run}: {error}")
                return
        st.session_state["explorer_last"] = {
            "spec": run,
            "text": text,
            "exploration": exploration,
        }
    if exploration is None:
        st.caption(
            "Pick a checkpoint and a sample sentence, then press Explore. Same as "
            f"`klab explore {run} --config explore.yaml` "
            "with the sentence in the declaration’s `text` field."
        )
        return
    render_exploration(exploration)


def render_exploration(exploration: Mapping[str, Any]) -> None:
    from klab.explorer import runtime_label

    arch = exploration["architecture"]
    target = exploration.get("target") or {}
    inv = arch.get("inventory") or {}
    if exploration.get("unavailable"):
        st.warning(f"⊘ Too big to explore here: {exploration['unavailable']}")
        st.markdown(xv.architecture_html(arch), unsafe_allow_html=True)
        return
    st.caption(f"Diagnostics runtime: {runtime_label(exploration)}")
    if exploration.get("cache_issue"):
        st.warning(f"Cache provenance unverified: {exploration['cache_issue']}")
    tokens = exploration["tokens"]
    losses = [t["loss"] for t in tokens if t["loss"] is not None]
    cols = st.columns(5)
    cols[0].metric("Resident parameters", compact(inv.get("total")))
    cols[1].metric("Active per token", compact(inv.get("active_per_token")))
    cols[2].metric("Layers", len(arch["layers"]))
    cols[3].metric(
        "Mean loss on the text",
        f"{sum(losses) / len(losses):.2f}" if losses else "–",
        help="Average next-token loss (nats) over the sample text.",
    )
    cols[4].metric("Tokens", len(tokens))
    st.caption(
        f"{target.get('run_id')} · step {target.get('step')} · checkpoint "
        f"{str(target.get('checkpoint_sha256'))[:12]} · computed in "
        f"{exploration.get('seconds', 0):.2f}s"
        + (" · text truncated" if exploration.get("truncated") else "")
    )
    tabs = ["Architecture", "Tokens", "Attention", "Weights"]
    if exploration.get("activation"):
        tabs.append("Activations")
    if exploration.get("routing"):
        tabs.append("Expert routing")
    if exploration.get("memory"):
        tabs.append("Memory lookups")
    shown = dict(zip(tabs, st.tabs(tabs), strict=True))
    with shown["Architecture"]:
        cols = st.columns([3, 2])
        with cols[0]:
            st.markdown(xv.architecture_html(arch), unsafe_allow_html=True)
        with cols[1]:
            st.plotly_chart(xv.parameter_breakdown(arch), use_container_width=True)
    with shown["Tokens"]:
        st.markdown(xv.token_strip_html(tokens), unsafe_allow_html=True)
        st.plotly_chart(xv.token_loss_figure(tokens), use_container_width=True)
        position = st.slider(
            "Next-token guesses after position",
            0,
            len(tokens) - 1,
            min(3, len(tokens) - 1),
            key="explorer_position",
        )
        token = tokens[position]
        st.markdown(
            f"After **{html.escape(''.join(t['text'] for t in tokens[1 : position + 1])[-80:])}** "
            "the model's top guesses were:"
        )
        st.dataframe(
            pd.DataFrame(token["next_top"])[["text", "p", "id"]],
            hide_index=True,
            column_config={
                "text": "token",
                "p": st.column_config.ProgressColumn(
                    "probability", min_value=0.0, max_value=1.0, format="%.3f"
                ),
            },
        )
    with shown["Attention"]:
        _attention(exploration, tokens)
    with shown["Weights"]:
        weights = exploration["weights"]
        st.plotly_chart(xv.weight_norms(weights), use_container_width=True)
        name = st.selectbox(
            "Histogram of", [w["name"] for w in weights], key="explorer_weight"
        )
        st.plotly_chart(
            xv.weight_histogram(next(w for w in weights if w["name"] == name)),
            use_container_width=True,
        )
    if "Activations" in shown:
        with shown["Activations"]:
            _activations(exploration, tokens)
    if "Expert routing" in shown:
        with shown["Expert routing"]:
            _routing(exploration, tokens)
    if "Memory lookups" in shown:
        with shown["Memory lookups"]:
            _memory(exploration, tokens)


def _attention(exploration: Mapping[str, Any], tokens: list[dict[str, Any]]) -> None:
    attention = exploration["attention"]
    if attention.get("note"):
        st.caption(f"ⓘ {attention['note']}")
    layers = attention["layers"]
    if not layers:
        st.info("No layer exposes attention weights.")
        return
    cols = st.columns(2)
    layer = cols[0].selectbox(
        "Layer", sorted(layers, key=int), key="explorer_attention_layer"
    )
    heads = layers[layer]
    head = cols[1].selectbox("Head", range(len(heads)), key="explorer_attention_head")
    summary = xv.head_summary(heads[head])
    stats = st.columns(4)
    stats[0].metric(
        "Looks back (mean tokens)",
        f"{summary['mean_distance']:.1f}",
        help="Average distance between a token and what it attends to.",
    )
    stats[1].metric(
        "On the previous token",
        f"{summary['previous_token_share']:.0%}",
        help="High values suggest a previous-token head.",
    )
    stats[2].metric(
        "On the first token",
        f"{summary['first_token_share']:.0%}",
        help="Many heads park unused attention on the first token (an attention sink).",
    )
    stats[3].metric(
        "Spread (entropy, nats)",
        f"{summary['mean_entropy']:.2f}",
        help="0 = all weight on one key; higher = spread over many.",
    )
    st.plotly_chart(
        xv.attention_figure(heads[head], tokens, f"Layer {layer} · head {head}"),
        use_container_width=True,
    )


def _routing(exploration: Mapping[str, Any], tokens: list[dict[str, Any]]) -> None:
    arch = exploration["architecture"]
    routing = exploration["routing"]
    st.caption(
        "Which expert the router sent each token to, and with what gate weight. "
        "Balanced load means every expert is used; collapse means one takes all."
    )
    for key in sorted(routing, key=int):
        layer = routing[key]
        experts = arch["layers"][int(key)]["ffn"]["experts"]
        load = xv.expert_load(layer)
        cols = st.columns([3, 1])
        cols[0].plotly_chart(
            xv.routing_figure(layer, tokens, experts, f"Layer {key} routing"),
            use_container_width=True,
        )
        with cols[1]:
            st.metric("Router entropy", f"{layer['entropy']:.2f}")
            st.dataframe(
                pd.DataFrame(
                    {"expert": list(range(len(load))), "share of tokens": load}
                ),
                hide_index=True,
                column_config={
                    "share of tokens": st.column_config.ProgressColumn(
                        min_value=0.0, max_value=1.0, format="%.2f"
                    )
                },
            )


def _memory(exploration: Mapping[str, Any], tokens: list[dict[str, Any]]) -> None:
    memory = exploration["memory"]
    st.caption(f"ⓘ {memory['note']}")
    for stream in memory["streams"]:
        offset = stream.get("single_token_offset")
        if offset:  # offset 0 is a legitimate order-1 (unigram) table
            st.warning(
                f"▲ Table `{stream['table']}` reads the row of a single token "
                f"({offset} back), not an "
                "n-gram: every address equals that token id modulo the table "
                "size. The hash may be degenerate for this table size (e.g. a "
                "multiplier that is a multiple of the row count)."
            )
    st.plotly_chart(xv.memory_address_figure(memory, tokens), use_container_width=True)
    st.caption(
        "Each dot is a recorded lookup. Repeated rows may reflect repeated n-grams or hash "
        "collisions. Row IDs are addresses, not concepts or neurons. The gate below shows "
        "mixing strength; these records do not identify neuron connections or causal influence."
    )
    reuse = xv.memory_reuse(memory)
    cols = st.columns(max(1, len(reuse)))
    for col, (table, share) in zip(cols, reuse.items(), strict=False):
        col.metric(
            f"Row reuse · {table}",
            f"{share:.0%}",
            help="Share of positions that read a row an earlier position read.",
        )
    figure = xv.gate_figure(memory, tokens)
    if figure is not None:
        st.plotly_chart(figure, use_container_width=True)
    st.dataframe(
        pd.DataFrame(xv.memory_rows(memory, tokens)),
        hide_index=True,
        use_container_width=True,
    )


def _activations(exploration: Mapping[str, Any], tokens: list[dict[str, Any]]) -> None:
    activation = exploration["activation"]
    sites = list(
        activation.get("sites") or ("residual_pre", "residual_mid", "residual_post")
    )
    layers = sorted(activation["layers"], key=int)
    st.caption(
        "FP32-derived root-mean-square (RMS) residual-stream magnitudes per token "
        "and layer (`residual_pre` before `norm1`, `residual_mid` before `norm2`, "
        "`residual_post` after the FFN residual). Descriptive activation magnitudes only, "
        "never wall-clock or step timing."
    )
    cols = st.columns(3)
    site = cols[0].selectbox(
        "Residual site",
        sites,
        format_func=lambda s: xv.ACTIVATION_SITE_LABEL.get(s, s),
        key="explorer_activation_site",
    )
    layer = cols[1].selectbox("Layer", layers, key="explorer_activation_layer")
    position = cols[2].selectbox(
        "Token position",
        range(len(tokens)),
        format_func=lambda i: f"#{i} {tokens[i]['text']!r}",
        key="explorer_activation_position",
    )
    layer_sites = activation["layers"][str(layer)]
    mcols = st.columns(len(sites))
    for col, s in zip(mcols, sites, strict=True):
        col.metric(
            f"L{layer} · {s} (#{position})",
            f"{layer_sites[s][position]:.4f}",
        )
    st.plotly_chart(
        xv.activation_heatmap(activation, tokens, site),
        use_container_width=True,
    )
    st.plotly_chart(
        xv.activation_line_figure(activation, tokens, layer=layer, position=position),
        use_container_width=True,
    )


def render_comparison(comparison: Mapping[str, Any]) -> None:
    from klab.explorer import runtime_label

    left = comparison["left"]
    right = comparison["right"]
    left_target = left.get("target") or {}
    right_target = right.get("target") or {}
    tokens = left["tokens"]
    common = comparison["common_layers"]
    only_left = comparison["only_left"]
    only_right = comparison["only_right"]

    st.caption(
        f"Diagnostics runtime: {runtime_label(left)} · tokenizer "
        f"{str(left_target.get('tokenizer_sha256') or '')[:12]} · "
        f"{len(tokens)} tokens · descriptive activation magnitudes and "
        "checkpoint-local routing (not inference timing)"
    )
    pcols = st.columns(2)
    pcols[0].markdown(
        f"**Left:** `{html.escape(str(left_target.get('run_id')))}` · "
        f"step `{left_target.get('step')}` · "
        f"checkpoint `{str(left_target.get('checkpoint_sha256'))[:12]}` "
        f"(`{html.escape(str(left_target.get('checkpoint') or 'pinned'))}`)"
    )
    pcols[1].markdown(
        f"**Right:** `{html.escape(str(right_target.get('run_id')))}` · "
        f"step `{right_target.get('step')}` · "
        f"checkpoint `{str(right_target.get('checkpoint_sha256'))[:12]}` "
        f"(`{html.escape(str(right_target.get('checkpoint') or 'pinned'))}`)"
    )
    token_labels = ", ".join(f"#{t['position']}:{t['text']}" for t in tokens)
    st.caption(
        f"Shared sentence: “{left['text']}” · Tokens: {token_labels} · "
        f"Matched layers: {common}"
        + (f" · Only left: {only_left}" if only_left else "")
        + (f" · Only right: {only_right}" if only_right else "")
    )

    sites = list(
        left["activation"].get("sites")
        or ("residual_pre", "residual_mid", "residual_post")
    )
    all_layers = sorted(set(common) | set(only_left) | set(only_right))
    scols = st.columns(2)
    site = scols[0].selectbox(
        "Residual site",
        sites,
        format_func=lambda s: xv.ACTIVATION_SITE_LABEL.get(s, s),
        key="explorer_compare_site",
    )
    layer = scols[1].selectbox("Layer", all_layers, key="explorer_compare_layer")
    left_fig, right_fig = xv.paired_activation_heatmaps(comparison, site)
    hcols = st.columns(2)
    hcols[0].plotly_chart(left_fig, use_container_width=True)
    hcols[1].plotly_chart(right_fig, use_container_width=True)
    delta_fig = xv.activation_delta_heatmap(comparison, site)
    if delta_fig is not None:
        st.plotly_chart(delta_fig, use_container_width=True)
    else:
        st.info(
            "Architectures differ (`architecture_matches=False`): showing the two "
            "measured RMS series independently without signed deltas."
        )
    st.plotly_chart(
        xv.paired_activation_line_figure(comparison, site, int(layer)),
        use_container_width=True,
    )

    st.subheader("MoE routing (checkpoint-local expert IDs)")
    st.caption(
        "Expert IDs belong to each checkpoint independently even when indices "
        "coincide; per-expert loads are never subtracted across runs."
    )
    rcols = st.columns(2)
    for side_label, side, col in (
        ("Left", left, rcols[0]),
        ("Right", right, rcols[1]),
    ):
        with col:
            routing = side.get("routing")
            if not routing:
                st.info(f"{side_label}: no MoE router (dense FFN)")
                continue
            for key in sorted(routing, key=int):
                r_layer = routing[key]
                experts = side["architecture"]["layers"][int(key)]["ffn"]["experts"]
                load = xv.expert_load(r_layer)
                st.plotly_chart(
                    xv.routing_figure(
                        r_layer,
                        side["tokens"],
                        experts,
                        f"{side_label} · Layer {key} routing",
                    ),
                    use_container_width=True,
                )
                st.metric(
                    f"{side_label} L{key} router entropy",
                    f"{r_layer['entropy']:.2f}",
                )
                st.dataframe(
                    pd.DataFrame(
                        {"expert": list(range(len(load))), "share of tokens": load}
                    ),
                    hide_index=True,
                    column_config={
                        "share of tokens": st.column_config.ProgressColumn(
                            min_value=0.0, max_value=1.0, format="%.2f"
                        )
                    },
                )
