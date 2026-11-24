"""The Engram tab: how a run's n-gram memory is built and what its recorded values say.

Everything shown is either the run's saved configuration or a value the trainer
recorded. One figure is derived and labelled as such: the memory's push, which is
the gate times the value norm over the hidden norm. Nothing here measures whether the
memory improves predictions; only a controlled comparison does.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from klab.dashboard import ui
from klab.dashboard.inspection import latest_metrics
from klab.dashboard.queries import DashboardSnapshot, RunRecord

ENGRAM_HELP = (Path(__file__).parent / "help" / "engram.md").read_text(encoding="utf-8")

PRIMER = """
An **Engram** is a lookup table that the model learns alongside its other weights. At
every position the model hashes the last few tokens (an *n-gram*) into a row number,
reads that row, and adds it to its hidden state through a **gate**:

`new hidden = hidden + gate × value`

- The **address** is the row number. It depends only on tokens already seen, never on
  the token being predicted.
- The **value** is the row that was read, projected to the model's width.
- The **gate** is a number between 0 and 1 that the model computes by comparing the
  row with its current hidden state: near 0 it ignores what it found, near 1 it adds
  all of it.

A run can look up several n-gram lengths, hash each one more than once (*hash heads*)
so that two n-grams colliding in one table are unlikely to collide in the other, and
do all of this at more than one layer (a *site*). Each combination has its own table.
Training fills the tables; adding documents does not.
"""

WATCH = """
- **A gate falling toward 0** at a site means the model is learning to ignore the
  memory there. A gate that settles at a small steady value is normal: the memory is
  one input among many.
- **Value norm rising while the gate falls** is the model trading one for the other;
  watch the **push**, which combines them.
- **Sites differ.** An early site sees raw token statistics and usually keeps a larger
  gate than a deep one, where the hidden state already carries context.
- **Reuse and hottest row** mostly describe the input, not the model: boilerplate and
  repeated code raise both. A sudden jump usually means the data changed.
- **Rows read per document** stays small by construction: one document cannot read
  more rows than it has positions.

None of these show that the memory makes predictions better. For that, compare
against the same run without it: `klab try` with a delta that sets `model.memory`.
"""


@dataclass(frozen=True)
class Table:
    site: str
    order: int | None
    head: int | None
    rows: int
    width: int | None

    @property
    def parameters(self) -> int | None:
        return self.rows * self.width if self.width else None


@dataclass(frozen=True)
class Reading:
    """One recorded value of a site, with where it started in this run."""

    value: float | None
    step: int | None = None
    first: float | None = None
    first_step: int | None = None


def tables(config: dict[str, object]) -> list[Table]:
    """The run's memory tables in the order the config lists them: site, order, head.

    When the listed sizes do not match sites × orders × heads the tables are returned
    without labels rather than guessed.
    """
    model = config.get("model")
    if not isinstance(model, dict) or model.get("memory") in (None, "none", False):
        return []
    width = model.get("memory_dim")
    width = width if isinstance(width, int) else None
    injection = model.get("memory_injection") or "final"
    blocks = model.get("memory_block_indices")
    if injection == "after_block":
        if not isinstance(blocks, list) or not blocks:
            blocks = [model.get("memory_block_index")]
        sites = [f"after block {block}" for block in blocks]
    else:
        sites = [
            "on the embeddings"
            if injection == "embedding"
            else "before the output head"
        ]
    orders = model.get("memory_ngram_orders")
    if not isinstance(orders, list) or not orders:
        orders = [model.get("memory_ngram_size")]
    heads = model.get("memory_hash_heads")
    heads = heads if isinstance(heads, int) and heads > 0 else 1
    sizes = model.get("memory_table_sizes")
    expected = len(sites) * len(orders) * heads
    if not isinstance(sizes, list) or not sizes:
        size = model.get("memory_table_size")
        sizes = [size] * expected if isinstance(size, int) else []
    sizes = [size for size in sizes if isinstance(size, int)]
    if len(sizes) != expected:
        return [Table("unlabelled", None, None, size, width) for size in sizes]
    labels = [
        (site, order if isinstance(order, int) else None, head)
        for site in sites
        for order in orders
        for head in range(heads)
    ]
    return [
        Table(site, order, head, size, width)
        for (site, order, head), size in zip(labels, sizes, strict=True)
    ]


def scopes(names: set[str]) -> list[tuple[str, str]]:
    """Label and metric prefix of each site that recorded values, else the aggregate."""
    ports = sorted(
        {name.split("/")[2] for name in names if name.startswith("engram/port/")},
        key=lambda port: (not port.isdigit(), int(port) if port.isdigit() else 0, port),
    )
    if ports:
        return [(f"Site after block {port}", f"engram/port/{port}/") for port in ports]
    return (
        [("All sites", "engram/")]
        if any(n.startswith("engram/") for n in names)
        else []
    )


def _finite(value: object) -> float | None:
    return (
        float(value)
        if isinstance(value, (int, float)) and math.isfinite(value)
        else None
    )


def push(gate: float | None, value: float | None, hidden: float | None) -> float | None:
    """Approximate size of what the memory adds, relative to the hidden state."""
    if gate is None or value is None or not hidden:
        return None
    return gate * value / hidden


def readings(
    series: dict[str, list[tuple[int, float | None]]], prefix: str
) -> dict[str, Reading]:
    """Latest and first value of each metric of one site, plus the derived push."""
    result: dict[str, Reading] = {}
    for name, points in series.items():
        if not name.startswith(prefix) or "/" in name[len(prefix) :]:
            continue
        known = [(step, value) for step, value in points if value is not None]
        if not known:
            result[name[len(prefix) :]] = Reading(None)
            continue
        result[name[len(prefix) :]] = Reading(
            known[-1][1], known[-1][0], known[0][1], known[0][0]
        )

    def joined(first: bool) -> float | None:
        """The push at one step; values observed at different steps are never mixed."""
        parts = [
            result.get(name) for name in ("gate_mean", "value_norm", "hidden_norm")
        ]
        if any(part is None for part in parts):
            return None
        steps = {part.first_step if first else part.step for part in parts}  # type: ignore[union-attr]
        if len(steps) != 1:
            return None
        return push(*(part.first if first else part.value for part in parts))  # type: ignore[union-attr]

    gate = result.get("gate_mean")
    result["push"] = Reading(
        joined(False),
        gate.step if gate else None,
        joined(True),
        gate.first_step if gate else None,
    )
    return result


def _series(
    view: DashboardSnapshot, run_id: str
) -> dict[str, list[tuple[int, float | None]]]:
    series: dict[str, list[tuple[int, float | None]]] = {}
    for row in view.metrics:
        name = str(row["name"])
        if row["run_id"] == run_id and name.startswith("engram/"):
            series.setdefault(name, []).append(
                (int(row["step"]), _finite(row["value"]))
            )  # type: ignore[call-overload]
    for points in series.values():
        points.sort()
    return series


def _since(reading: Reading, form: str) -> str:
    if (
        reading.first is None
        or reading.first_step is None
        or reading.first_step == reading.step
    ):
        return ""
    return f" Was {format(reading.first, form)} at step {reading.first_step:,}."


def _site_tiles(values: dict[str, Reading]) -> str:
    def get(name: str) -> Reading:
        return values.get(name, Reading(None))

    def show(reading: Reading, form: str) -> str:
        return "Unavailable" if reading.value is None else format(reading.value, form)

    gate, value, hidden = get("gate_mean"), get("value_norm"), get("hidden_norm")
    pushed, reuse, hottest = (
        get("push"),
        get("bucket_reuse_rate"),
        get("maximum_address_fraction"),
    )
    unique, lookups, used = (
        get("unique_addresses"),
        get("lookup_count"),
        get("table_utilization"),
    )
    distinct = (
        f"{unique.value:,.0f} of {lookups.value:,.0f}"
        if unique.value is not None and lookups.value
        else "Unavailable"
    )
    return ui.tiles(
        [
            (
                "Gate",
                show(gate, ".3f"),
                "How much of the row it read the model adds, from 0 (ignore) to 1 (add all)."
                + _since(gate, ".3f"),
            ),
            (
                "Push (derived)",
                show(pushed, ".2%"),
                "Size of what the memory adds, relative to the hidden state: gate × value "
                "norm ÷ hidden norm. Approximate." + _since(pushed, ".2%"),
            ),
            (
                "Value norm",
                show(value, ".2f"),
                "Length of the vector read from the tables, before the gate."
                + _since(value, ".2f"),
            ),
            (
                "Hidden norm",
                show(hidden, ".1f"),
                "Length of the hidden state the memory is added to."
                + _since(hidden, ".1f"),
            ),
            (
                "Distinct rows read",
                distinct,
                "Different rows read, of all lookups in one forward pass of this site.",
            ),
            (
                "Reuse",
                show(reuse, ".1%"),
                (
                    "Lookups that hit a row already read in the same document: repeated "
                    "n-grams, or different n-grams sharing a row. The two cannot be told apart here."
                ),
            ),
            (
                "Rows read per document",
                show(used, ".2%"),
                (
                    "Share of a table's rows one document reads. Small by construction; "
                    "not how much of the table has been trained."
                ),
            ),
            (
                "Hottest row",
                show(hottest, ".2%"),
                (
                    "Share of lookups that land on the single most-read row. High means one "
                    "n-gram dominates the input."
                ),
            ),
        ],
        minimum=230,
    )


_CHARTS = (
    ("gate_mean", "Gate", ".3f"),
    ("push", "Push (derived): gate × value ÷ hidden", ".2%"),
    ("value_norm", "Value norm", ".2f"),
    ("hidden_norm", "Hidden norm", ".1f"),
    ("bucket_reuse_rate", "Reuse", ".1%"),
    ("maximum_address_fraction", "Hottest row", ".2%"),
)
_SITE_COLORS = (
    ui.SERIES["candidate"],
    ui.SERIES["control"],
    ui.SERIES["accent"],
    ui.SERIES["reference"],
)


def history_figure(
    series: dict[str, list[tuple[int, float | None]]], sites: list[tuple[str, str]]
) -> go.Figure:
    """Each value over the run's steps, one line per site."""
    figure = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=[title for _, title, _ in _CHARTS],
        vertical_spacing=0.11,
    )
    for index, (metric, title, form) in enumerate(_CHARTS):
        row, column = index // 2 + 1, index % 2 + 1
        for color, (label, prefix) in zip(_SITE_COLORS * 4, sites, strict=False):
            if metric == "push":
                parts = [
                    dict(series.get(prefix + n, []))
                    for n in ("gate_mean", "value_norm", "hidden_norm")
                ]
                points = [
                    (step, push(parts[0][step], parts[1].get(step), parts[2].get(step)))
                    for step in sorted(parts[0])
                ]
            else:
                points = series.get(prefix + metric, [])
            points = [(step, value) for step, value in points if value is not None]
            if not points:
                continue
            figure.add_trace(
                go.Scatter(
                    x=[step for step, _ in points],
                    y=[value for _, value in points],
                    name=label,
                    legendgroup=label,
                    showlegend=index == 0,
                    mode="lines",
                    line={"color": color, "width": 1.6},
                    hovertemplate=f"step %{{x:,}}<br>{label}: %{{y:{form}}}<extra></extra>",
                ),
                row=row,
                col=column,
            )
        figure.update_yaxes(tickformat=form, row=row, col=column)
    figure.update_xaxes(title_text="Optimizer step", row=3)
    return ui.figure_layout(figure, height=760)


def _layout_table(layout: list[Table]) -> None:
    rows = sum(table.rows for table in layout)
    parameters = sum(table.parameters or 0 for table in layout)
    st.markdown(
        f"**{len(layout)} tables**, {rows:,} rows"
        + (f", {parameters / 1e6:,.1f}M parameters" if parameters else "")
        + ". Table sizes are distinct primes so that two tables never wrap the same way."
    )
    st.dataframe(
        [
            {
                "Site": table.site,
                "N-gram length": table.order,
                "Hash head": table.head,
                "Rows": table.rows,
                "Row width": table.width,
                "Parameters": table.parameters,
            }
            for table in layout
        ],
        hide_index=True,
        width="stretch",
    )
    st.caption(
        "Recorded values are per site, not per table: the trainer combines a site's "
        "tables before it records them."
    )


def engram_page(
    view: DashboardSnapshot, selected_ids: list[str], records: list[RunRecord]
) -> None:
    st.header("Engram memory")
    ui.explainer("What an Engram is", PRIMER)
    if not selected_ids:
        st.info("Select a run in the sidebar to see its memory.")
        return
    by_id = {record.run_id: record for record in records}
    for run_id in selected_ids:
        record = by_id.get(run_id)
        layout = tables(record.config) if record else []
        series = _series(view, run_id)
        sites = scopes(set(series))
        with st.container(border=True):
            st.subheader(run_id)
            if not layout and not sites:
                st.info(
                    "This run has no Engram memory: its configuration sets none and it "
                    "recorded no memory values."
                )
                continue
            if layout:
                st.markdown("##### How this memory is built")
                _layout_table(layout)
            if not sites:
                st.info(
                    "No memory values are recorded yet. Missing values are not zero."
                )
                continue
            st.markdown("##### What it is doing now")
            for label, prefix in sites:
                values = readings(series, prefix)
                step = values.get("gate_mean", Reading(None)).step
                st.markdown(
                    f"**{label}**"
                    + (f" · recorded at step {step:,}" if step is not None else "")
                )
                st.markdown(
                    _site_tiles(values) + "<div style='height:10px'></div>",
                    unsafe_allow_html=True,
                )
            st.markdown("##### How it changed during this run")
            st.plotly_chart(
                history_figure(series, sites),
                width="stretch",
                key=f"engram-history-{run_id}",
            )
            st.caption(
                "Values are averages over the positions of each logged update, so they "
                "wobble with the documents in that update. Read the trend, not a point."
            )
    ui.explainer("What to look for", WATCH, expanded=True)
    rows = [
        row
        for row in latest_metrics(view, selected_ids)
        if row["metric"].startswith("engram/")
    ]
    if rows:
        with st.expander(
            "Every recorded memory value, and its change since the previous one"
        ):
            st.dataframe(rows, hide_index=True, width="stretch")
            st.markdown(ENGRAM_HELP)
