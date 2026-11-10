"""Figures and HTML for the model explorer (pure functions, no Streamlit).

Input is one exploration from :mod:`sparselab.explorer` (the sealed cache
record); nothing here touches a model.
"""

from __future__ import annotations

import html
from collections.abc import Mapping
from typing import Any

import numpy as np
import plotly.graph_objects as go

from sparselab.dashboard.ui import SEQUENTIAL, SERIES, compact, figure_layout

ATTENTION_KIND = {
    "dense": "dense attention",
    "mla": "latent (MLA) attention",
    "block_sparse": "block-sparse attention",
}

ACTIVATION_SITE_LABEL = {
    "residual_pre": "residual_pre (input to norm1)",
    "residual_mid": "residual_mid (input to norm2, after attention)",
    "residual_post": "residual_post (after FFN residual)",
}
DIVERGING = "RdBu_r"


def _unit_row(count: int, active: int, color: str, title: str) -> str:
    units = "".join(
        f"<span class='lab-unit' title='{html.escape(title)} {i}' "
        f"style='background:{color};opacity:{1 if i < active else 0.28}'></span>"
        for i in range(count)
    )
    return f"<div>{units}</div>"


def architecture_html(arch: Mapping[str, Any]) -> str:
    """A top-to-bottom block diagram: embedding → layers → memory → output."""
    inv = arch.get("inventory") or {}
    parts = [
        "<div class='lab-arch'>",
        (
            "<div class='lab-block' style='background:rgba(0,114,178,.10)'>"
            f"<div class='t'>Token embedding</div><div class='s'>{arch['vocab_size']:,} "
            f"tokens × {arch['hidden_dim']} dims · "
            f"{compact(arch['embedding_parameters'])} params</div></div>"
        ),
    ]
    memory = arch.get("memory")
    if memory and memory.get("injection") == "embedding":
        parts.append(_memory_block(memory))
    for layer in arch["layers"]:
        att, ffn = layer["attention"], layer["ffn"]
        kv = (
            f" · {att['kv_heads']} shared KV head(s)"
            if att["kv_heads"] != att["heads"]
            else ""
        )
        window = f" · window {att['window']}" if att.get("window") else ""
        if ffn["kind"] == "moe":
            ffn_units = _unit_row(
                ffn["experts"], ffn["experts_per_token"], SERIES["accent"], "expert"
            )
            ffn_text = (
                f"{ffn['experts']} experts, {ffn['experts_per_token']} active per "
                f"token{' + shared expert' if ffn['shared_expert'] else ''} · "
                f"{compact(ffn['per_expert'])} params each"
            )
            ffn_title = "Mixture of experts"
        else:
            ffn_units = ""
            ffn_text = f"{arch['ffn_dim']} hidden units"
            ffn_title = "Feed-forward (dense)"
        parts.append("<div class='lab-arrow'>▼</div>")
        parts.append(
            "<div class='lab-row'>"
            "<div class='lab-block' style='background:rgba(128,128,128,.06);"
            "flex:0 0 74px'><div class='t'>Layer "
            f"{layer['index']}</div><div class='s'>norms "
            f"{compact(layer['norm_parameters'])}</div></div>"
            "<div class='lab-block' style='background:rgba(230,159,0,.10)'>"
            f"<div class='t'>{ATTENTION_KIND.get(att['kind'], att['kind'])}</div>"
            f"{_unit_row(att['heads'], att['heads'], SERIES['control'], 'head')}"
            f"<div class='s'>{att['heads']} heads × {att['head_dim']} dims{kv}{window}"
            f" · {compact(att['parameters'])} params</div></div>"
            "<div class='lab-block' style='background:rgba(0,158,115,.10)'>"
            f"<div class='t'>{ffn_title}</div>{ffn_units}<div class='s'>{ffn_text}"
            f" · {compact(ffn['parameters'])} params</div></div></div>"
        )
    if memory and memory.get("injection") != "embedding":
        parts.append("<div class='lab-arrow'>▼</div>")
        parts.append(_memory_block(memory))
    output = (
        "tied to the embedding (no extra params)"
        if arch.get("tie_embeddings")
        else f"{compact(arch.get('output_parameters'))} params"
    )
    parts.append("<div class='lab-arrow'>▼</div>")
    parts.append(
        "<div class='lab-block' style='background:rgba(0,114,178,.10)'>"
        f"<div class='t'>Next-token output</div><div class='s'>{output}</div></div>"
    )
    parts.append("</div>")
    total = inv.get("total")
    active = inv.get("active_per_token")
    if total:
        parts.append(
            f"<p class='pb-hint' style='margin-top:8px'>{int(total):,} resident "
            f"parameters · {int(active or total):,} active per token</p>"
        )
    return "".join(parts)


def _memory_block(memory: Mapping[str, Any]) -> str:
    orders = (
        f" · n-gram orders {', '.join(map(str, memory['ngram_orders']))}"
        if memory.get("ngram_orders")
        else ""
    )
    return (
        "<div class='lab-block' style='background:rgba(130,80,223,.10)'>"
        f"<div class='t'>Memory · {html.escape(str(memory['kind']))} "
        f"({html.escape(str(memory['injection']))} injection)</div>"
        f"<div class='s'>{memory['tables']} table(s) × {memory['rows']:,} rows × "
        f"{memory['dim']} dims{orders} · {compact(memory['parameters'])} params</div>"
        "</div>"
    )


def parameter_breakdown(arch: Mapping[str, Any]) -> go.Figure:
    """Where the parameters live, stacked per component."""
    labels, values, colors = (
        ["embedding"],
        [arch["embedding_parameters"]],
        [SERIES["candidate"]],
    )
    for layer in arch["layers"]:
        i = layer["index"]
        labels += [f"L{i} attention", f"L{i} {layer['ffn']['kind']}", f"L{i} norms"]
        values += [
            layer["attention"]["parameters"],
            layer["ffn"]["parameters"],
            layer["norm_parameters"],
        ]
        colors += [SERIES["control"], SERIES["accent"], SERIES["baseline"]]
    if arch.get("memory"):
        labels.append("memory")
        values.append(arch["memory"]["parameters"])
        colors.append(SERIES["reference"])
    if arch.get("output_parameters"):
        labels.append("output")
        values.append(arch["output_parameters"])
        colors.append(SERIES["candidate"])
    figure = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            hovertemplate="%{y}: %{x:,} params<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed")
    return figure_layout(
        figure,
        height=max(220, 26 * len(labels) + 60),
        title="Parameters per component",
        xaxis_title="parameters",
    )


def weight_norms(weights: list[Mapping[str, Any]]) -> go.Figure:
    names = [w["name"] for w in weights]
    figure = go.Figure(
        go.Bar(
            x=[w["rms"] for w in weights],
            y=names,
            orientation="h",
            marker_color=SERIES["candidate"],
            customdata=[[w["norm"], w["abs_max"], str(w["shape"])] for w in weights],
            hovertemplate="%{y}<br>RMS %{x:.4f} · ‖W‖ %{customdata[0]:.3f}"
            "<br>max |w| %{customdata[1]:.3f} · shape %{customdata[2]}<extra></extra>",
        )
    )
    figure.update_yaxes(autorange="reversed", tickfont={"size": 10})
    return figure_layout(
        figure,
        height=max(260, 18 * len(names) + 60),
        title="Weight RMS per tensor (hover for norm, max and shape)",
        xaxis_title="root-mean-square weight",
    )


def weight_histogram(weight: Mapping[str, Any]) -> go.Figure:
    hist = weight["histogram"]
    edges = np.asarray(hist["edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    figure = go.Figure(
        go.Bar(
            x=centers,
            y=hist["counts"],
            width=np.diff(edges),
            marker_color=SERIES["accent"],
            hovertemplate="%{x:.4f}: %{y}<extra></extra>",
        )
    )
    sampled = (
        f" · every {hist['sampled_every']}th value"
        if hist.get("sampled_every", 1) > 1
        else ""
    )
    return figure_layout(
        figure,
        height=260,
        title=f"{weight['name']} · mean {weight['mean']:+.4f} · std "
        f"{weight['std']:.4f}{sampled}",
        xaxis_title="weight value",
        yaxis_title="count",
        bargap=0,
    )


def _shade(loss: float | None, top: float) -> str:
    if loss is None:
        return "rgba(128,128,128,.12)"
    share = min(1.0, loss / top) if top > 0 else 0.0
    # Low loss = pale blue, high loss = strong orange: readable in both themes.
    if share < 0.5:
        return f"rgba(0,114,178,{0.12 + 0.5 * (0.5 - share):.2f})"
    return f"rgba(230,159,0,{0.25 + 0.75 * (share - 0.5) * 2:.2f})"


def token_strip_html(tokens: list[Mapping[str, Any]]) -> str:
    """The sample text, each token shaded by how surprised the model was."""
    losses = [t["loss"] for t in tokens if t["loss"] is not None]
    top = float(np.percentile(losses, 95)) if losses else 1.0
    spans = []
    for token in tokens:
        loss = token["loss"]
        guess = ", ".join(f"{g['text']!r} {g['p']:.2f}" for g in token["next_top"][:3])
        title = (
            f"#{token['position']} id {token['id']}"
            + ("" if loss is None else f" · loss {loss:.2f} · rank {token['rank']}")
            + f" · next: {guess}"
        )
        spans.append(
            f"<span class='lab-tok' style='background:{_shade(loss, top)}' "
            f"title='{html.escape(title, quote=True)}'>"
            f"{html.escape(token['text'])}</span>"
        )
    return (
        "<div style='line-height:2'>"
        + "".join(spans)
        + "</div><p class='pb-hint'>blue = predicted well · orange = surprising "
        "(hover a token for its loss, rank and the model's next guesses)</p>"
    )


def token_loss_figure(tokens: list[Mapping[str, Any]]) -> go.Figure:
    scored = [t for t in tokens if t["loss"] is not None]
    figure = go.Figure(
        go.Bar(
            x=[t["position"] for t in scored],
            y=[t["loss"] for t in scored],
            marker_color=SERIES["control"],
            customdata=[[t["text"], t["rank"]] for t in scored],
            hovertemplate="#%{x} %{customdata[0]}<br>loss %{y:.3f} · rank "
            "%{customdata[1]}<extra></extra>",
        )
    )
    mean = float(np.mean([t["loss"] for t in scored])) if scored else 0.0
    figure.add_hline(
        y=mean, line_dash="dot", annotation_text=f"mean {mean:.2f}", line_color="#666"
    )
    return figure_layout(
        figure,
        height=260,
        title="Per-token loss (nats)",
        xaxis_title="position",
        yaxis_title="loss",
    )


def _labels(tokens: list[Mapping[str, Any]]) -> list[str]:
    # Unique labels keep plotly from merging repeated tokens on an axis.
    return [f"{t['position']}:{t['text'][:10]}" for t in tokens]


def attention_figure(
    weights: list[list[float]], tokens: list[Mapping[str, Any]], title: str
) -> go.Figure:
    labels = _labels(tokens)[: len(weights)]
    # Row 0 can only attend to itself (weight 1); scale to the other rows so
    # the patterns that carry information stay visible.
    rest = np.asarray(weights, dtype=np.float64)[1:]
    top = float(rest.max()) if rest.size else 1.0
    figure = go.Figure(
        go.Heatmap(
            z=weights,
            x=labels,
            y=labels,
            colorscale=SEQUENTIAL,
            zmin=0,
            zmax=max(top, 1e-6),
            hovertemplate="query %{y}<br>key %{x}<br>weight %{z:.3f}<extra></extra>",
            colorbar={"title": "weight"},
        )
    )
    figure.update_yaxes(autorange="reversed", tickfont={"size": 9})
    figure.update_xaxes(tickfont={"size": 9}, tickangle=-60)
    return figure_layout(
        figure,
        height=560,
        title=f"{title} · color scale tops out at {top:.2f}",
        xaxis_title="attends to (key)",
        yaxis_title="token (query)",
    )


def head_summary(weights: list[list[float]]) -> dict[str, float]:
    """Plain numbers describing one head: how peaked and how far back it looks."""
    matrix = np.asarray(weights, dtype=np.float64)
    positions = np.arange(matrix.shape[0])
    distance = (matrix * (positions[:, None] - positions[None, :])).sum(axis=1)
    entropy = -(matrix * np.log(np.clip(matrix, 1e-12, None))).sum(axis=1)
    return {
        "mean_distance": float(distance[1:].mean()) if len(distance) > 1 else 0.0,
        "mean_entropy": float(entropy[1:].mean()) if len(entropy) > 1 else 0.0,
        "first_token_share": float(matrix[1:, 0].mean()) if len(matrix) > 1 else 1.0,
        "previous_token_share": float(np.diag(matrix, -1).mean())
        if len(matrix) > 1
        else 0.0,
    }


def routing_figure(
    layer: Mapping[str, Any], tokens: list[Mapping[str, Any]], experts: int, title: str
) -> go.Figure:
    """Token × expert gate weights: which expert each token was sent to."""
    selected = np.asarray(layer["selected"])
    weights = np.asarray(layer["weights"])
    # selected/weights are [batch*time, k] or [batch, time, k]; flatten to [T, k].
    selected = selected.reshape(-1, selected.shape[-1])
    weights = weights.reshape(-1, weights.shape[-1])
    matrix = np.zeros((experts, selected.shape[0]))
    for position in range(selected.shape[0]):
        for k in range(selected.shape[1]):
            matrix[int(selected[position, k]), position] += float(weights[position, k])
    labels = _labels(tokens)[: selected.shape[0]]
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=[f"expert {i}" for i in range(experts)],
            colorscale=SEQUENTIAL,
            zmin=0,
            hovertemplate="%{x} → %{y}<br>gate %{z:.3f}<extra></extra>",
            colorbar={"title": "gate"},
        )
    )
    figure.update_xaxes(tickfont={"size": 9}, tickangle=-60)
    return figure_layout(figure, height=200 + 26 * experts, title=title)


def expert_load(layer: Mapping[str, Any]) -> list[float]:
    counts = np.asarray(layer.get("counts") or [], dtype=np.float64).reshape(-1)
    total = counts.sum()
    return (counts / total).tolist() if total else counts.tolist()


def memory_rows(
    memory: Mapping[str, Any], tokens: list[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Per position: the token, its gate, and each table's row (↺ = reused)."""
    rows = []
    for position, token in enumerate(tokens):
        row: dict[str, Any] = {"position": position, "token": token["text"]}
        if memory.get("gate"):
            row["gate"] = memory["gate"][position]
        for stream in memory["streams"]:
            if position >= len(stream["addresses"]):
                continue
            address = stream["addresses"][position]
            again = stream["first_seen_at"][position]
            row[stream["table"]] = (
                f"{address} ↺{again}" if again is not None else str(address)
            )
        rows.append(row)
    return rows


def memory_reuse(memory: Mapping[str, Any]) -> dict[str, float]:
    """Share of positions that re-read a row an earlier position read."""
    out = {}
    for stream in memory["streams"]:
        seen = stream["first_seen_at"]
        out[stream["table"]] = (
            sum(s is not None for s in seen) / len(seen) if seen else 0.0
        )
    return out


def gate_figure(memory: Mapping[str, Any], tokens: list[Mapping[str, Any]]) -> Any:
    if not memory.get("gate"):
        return None
    labels = _labels(tokens)[: len(memory["gate"])]
    figure = go.Figure(
        go.Bar(
            x=labels,
            y=memory["gate"],
            marker_color=SERIES["reference"],
            hovertemplate="%{x}<br>gate %{y:.3f}<extra></extra>",
        )
    )
    figure.update_xaxes(tickfont={"size": 9}, tickangle=-60)
    return figure_layout(
        figure,
        height=260,
        title="How much each token mixes in its memory read (gate, 0–1)",
        yaxis_range=[0, 1],
    )


def activation_heatmap(
    activation: Mapping[str, Any],
    tokens: list[Mapping[str, Any]],
    site: str,
    *,
    title: str | None = None,
    zmin: float = 0.0,
    zmax: float | None = None,
) -> go.Figure:
    """Layer × token RMS heatmap for one residual stream site."""
    layers = sorted(activation["layers"], key=int)
    matrix = [activation["layers"][k][site] for k in layers]
    labels = _labels(tokens)[: len(matrix[0]) if matrix else 0]
    arr = np.asarray(matrix, dtype=np.float64)
    top = float(arr.max()) if arr.size else 1.0
    scale_max = max(float(zmax if zmax is not None else top), 1e-6)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=[f"layer {k}" for k in layers],
            colorscale=SEQUENTIAL,
            zmin=zmin,
            zmax=scale_max,
            hovertemplate="%{y} · %{x}<br>RMS %{z:.4f}<extra></extra>",
            colorbar={"title": "RMS"},
        )
    )
    figure.update_xaxes(tickfont={"size": 9}, tickangle=-60)
    site_label = ACTIVATION_SITE_LABEL.get(site, site)
    return figure_layout(
        figure,
        height=max(220, 42 * len(layers) + 120),
        title=title or f"Residual RMS · {site_label} (descriptive magnitude)",
        xaxis_title="token position",
        yaxis_title="layer",
    )


def activation_line_figure(
    activation: Mapping[str, Any],
    tokens: list[Mapping[str, Any]],
    *,
    layer: str | int,
    position: int | None = None,
) -> go.Figure:
    """Per-token RMS across all three residual sites within a single layer."""
    layer_key = str(layer)
    layer_sites = activation["layers"][layer_key]
    sites = list(
        activation.get("sites") or ("residual_pre", "residual_mid", "residual_post")
    )
    labels = _labels(tokens)
    palette = {
        "residual_pre": SERIES["baseline"],
        "residual_mid": SERIES["control"],
        "residual_post": SERIES["candidate"],
    }
    figure = go.Figure()
    for site in sites:
        values = layer_sites[site]
        figure.add_trace(
            go.Scatter(
                x=labels[: len(values)],
                y=values,
                mode="lines+markers",
                name=site,
                line={"color": palette.get(site, SERIES["accent"])},
                hovertemplate=f"layer {layer_key} · {site}<br>%{{x}}: RMS %{{y:.4f}}<extra></extra>",
            )
        )
    if position is not None and 0 <= position < len(labels):
        figure.add_vline(
            x=labels[position],
            line_dash="dot",
            line_color="#666",
        )
    return figure_layout(
        figure,
        height=280,
        title=f"Layer {layer_key} residual stream RMS by token",
        xaxis_title="token position",
        yaxis_title="RMS",
    )


def paired_activation_range(
    left_activation: Mapping[str, Any],
    right_activation: Mapping[str, Any],
    site: str | None = None,
) -> tuple[float, float]:
    """Shared [0, zmax] RMS range so paired heatmaps sit on one color scale."""
    values: list[float] = []
    for act in (left_activation, right_activation):
        sites = [site] if site is not None else list(act.get("sites") or ())
        for layer_sites in (act.get("layers") or {}).values():
            for s in sites:
                if s in layer_sites:
                    values.extend(float(v) for v in layer_sites[s])
    top = max(values) if values else 1.0
    return 0.0, max(top, 1e-6)


def _target_short_label(exploration: Mapping[str, Any], side: str) -> str:
    target = exploration.get("target") or {}
    run_id = target.get("run_id") or side
    step = target.get("step")
    sha = str(target.get("checkpoint_sha256") or "")[:12]
    step_part = f" · step {step}" if step is not None else ""
    return f"{side}: {run_id}{step_part} ({sha})"


def paired_activation_heatmaps(
    comparison: Mapping[str, Any], site: str
) -> tuple[go.Figure, go.Figure]:
    """Left and right Layer × Token RMS heatmaps on a shared color range."""
    left = comparison["left"]
    right = comparison["right"]
    zmin, zmax = paired_activation_range(left["activation"], right["activation"], site)
    site_label = ACTIVATION_SITE_LABEL.get(site, site)
    left_fig = activation_heatmap(
        left["activation"],
        left["tokens"],
        site,
        title=f"{_target_short_label(left, 'Left')} · {site_label}",
        zmin=zmin,
        zmax=zmax,
    )
    right_fig = activation_heatmap(
        right["activation"],
        right["tokens"],
        site,
        title=f"{_target_short_label(right, 'Right')} · {site_label}",
        zmin=zmin,
        zmax=zmax,
    )
    return left_fig, right_fig


def activation_delta_heatmap(
    comparison: Mapping[str, Any], site: str
) -> go.Figure | None:
    """Signed (right − left) RMS heatmap with centered-zero diverging scale.

    Returns None when architectures differ or delta is not permitted.
    """
    delta = comparison.get("delta")
    if not comparison.get("architecture_matches") or not delta:
        return None
    common = [str(i) for i in comparison.get("common_layers") or [] if str(i) in delta]
    if not common:
        return None
    matrix = [delta[k][site] for k in common]
    tokens = comparison["left"]["tokens"]
    labels = _labels(tokens)[: len(matrix[0])]
    arr = np.asarray(matrix, dtype=np.float64)
    bound = max(float(np.abs(arr).max()), 1e-6) if arr.size else 1.0
    site_label = ACTIVATION_SITE_LABEL.get(site, site)
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=labels,
            y=[f"layer {k}" for k in common],
            colorscale=DIVERGING,
            zmin=-bound,
            zmax=bound,
            zmid=0.0,
            hovertemplate="%{y} · %{x}<br>Δ RMS (right − left) %{z:+.4f}<extra></extra>",
            colorbar={"title": "Δ RMS"},
        )
    )
    figure.update_xaxes(tickfont={"size": 9}, tickangle=-60)
    return figure_layout(
        figure,
        height=max(220, 42 * len(common) + 120),
        title=f"Signed Δ RMS (right − left) · {site_label}",
        xaxis_title="token position",
        yaxis_title="layer",
    )


def paired_activation_line_figure(
    comparison: Mapping[str, Any], site: str, layer: int
) -> go.Figure:
    """Measured per-token RMS series for left and right at one layer and site."""
    left = comparison["left"]
    right = comparison["right"]
    tokens = left["tokens"]
    labels = _labels(tokens)
    key = str(layer)
    figure = go.Figure()
    left_layers = left["activation"]["layers"]
    right_layers = right["activation"]["layers"]
    if key in left_layers:
        vals = left_layers[key][site]
        figure.add_trace(
            go.Scatter(
                x=labels[: len(vals)],
                y=vals,
                mode="lines+markers",
                name=_target_short_label(left, "left"),
                line={"color": SERIES["baseline"]},
                hovertemplate=f"left L{key} · {site}<br>%{{x}}: RMS %{{y:.4f}}<extra></extra>",
            )
        )
    if key in right_layers:
        vals = right_layers[key][site]
        figure.add_trace(
            go.Scatter(
                x=labels[: len(vals)],
                y=vals,
                mode="lines+markers",
                name=_target_short_label(right, "right"),
                line={"color": SERIES["candidate"]},
                hovertemplate=f"right L{key} · {site}<br>%{{x}}: RMS %{{y:.4f}}<extra></extra>",
            )
        )
    return figure_layout(
        figure,
        height=280,
        title=f"Layer {key} · {ACTIVATION_SITE_LABEL.get(site, site)} RMS per token",
        xaxis_title="token position",
        yaxis_title="RMS",
    )


def memory_address_figure(
    memory: Mapping[str, Any], tokens: list[Mapping[str, Any]]
) -> go.Figure:
    """Observed token-to-address edges; table identity keeps equal row IDs distinct."""
    figure = go.Figure()
    for stream in memory["streams"]:
        addresses = stream["addresses"][: len(tokens)]
        figure.add_trace(
            go.Scatter(
                x=list(range(len(addresses))),
                y=[f"{stream['table']} · row {address}" for address in addresses],
                text=[str(token["text"]) for token in tokens[: len(addresses)]],
                mode="markers",
                name=str(stream["table"]),
                hovertemplate="Position %{x} · %{text}<br>%{y}<extra></extra>",
            )
        )
    figure.update_yaxes(type="category")
    return figure_layout(
        figure,
        height=420,
        title="Observed token → memory row lookups",
        xaxis_title="Token position",
        yaxis_title="Table and hashed row",
    )
