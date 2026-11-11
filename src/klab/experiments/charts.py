"""Deterministic, dependency-free SVG charts for static study reports."""

from __future__ import annotations

import html
import math
from collections.abc import Iterable


def _context_label(value: object) -> str:
    if not isinstance(value, dict) or not value:
        return str(value)
    return ", ".join(f"{key}={value[key]}" for key in sorted(value))


def _chart_label(value: str) -> str:
    return value if len(value) <= 14 else f"{value[:13]}…"


def _svg(title: str, points: Iterable[tuple[str, float]], *, y_label: str) -> str:
    values = list(points)
    width, height, left, bottom = 760, 300, 58, 45
    right = 90
    plot_w, plot_h = width - left - right, height - 30 - bottom
    finite = [(label, value) for label, value in values if math.isfinite(value)]
    if not finite:
        body = '<text x="20" y="55">No observed values available.</text>'
    else:
        low, high = min(value for _, value in finite), max(value for _, value in finite)
        if low == high:
            low, high = low - 0.5, high + 0.5
        marks: list[str] = []
        for index, (label, value) in enumerate(finite):
            x = left + (plot_w * index / max(1, len(finite) - 1))
            y = 30 + plot_h * (1 - (value - low) / (high - low))
            marks.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="4"/>')
            marks.append(
                f'<text x="{x:.2f}" y="{height - 18}" text-anchor="middle">{html.escape(_chart_label(label))}</text>'
            )
            marks.append(
                f'<text x="{x:.2f}" y="{y - 8:.2f}" text-anchor="middle">{value:.6g}</text>'
            )
        body = (
            f'<line x1="{left}" y1="30" x2="{left}" y2="{height - bottom}"/>'
            f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}"/>'
            f'<text x="5" y="20">{html.escape(y_label)} ({low:.6g} … {high:.6g})</text>'
            + "".join(marks)
        )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f"<title>{html.escape(title)}</title><style>text{{font:12px sans-serif}}line{{stroke:#555}}circle{{fill:#1769aa}}</style>"
        f"{body}</svg>\n"
    )


def _allocation_svg(heatmap: dict[str, object]) -> str:
    factor_a = heatmap.get("factor_a")
    factor_b = heatmap.get("factor_b")
    factor_a = factor_a if isinstance(factor_a, dict) else {}
    factor_b = factor_b if isinstance(factor_b, dict) else {}
    axis_a = str(factor_a.get("axis", "factor A"))
    axis_b = str(factor_b.get("axis", "factor B"))
    levels_a = [factor_a.get("control"), factor_a.get("treatment")]
    levels_b = [factor_b.get("control"), factor_b.get("treatment")]
    cells = heatmap.get("cells")
    cells = cells if isinstance(cells, list) else []
    indexed: dict[tuple[object, object], dict[str, object]] = {}
    shares: list[float] = []
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        coordinates = cell.get("factor_levels")
        parameters = cell.get("parameters")
        if not isinstance(coordinates, dict) or not isinstance(parameters, dict):
            continue
        share = parameters.get("memory_share")
        if (
            isinstance(share, (int, float))
            and not isinstance(share, bool)
            and math.isfinite(float(share))
        ):
            shares.append(float(share))
            indexed[(coordinates.get(axis_a), coordinates.get(axis_b))] = cell
    low, high = (min(shares), max(shares)) if shares else (0.0, 0.0)
    width, height = 760, 390
    left, top, cell_w, cell_h = 150, 95, 275, 122
    marks: list[str] = []
    marks.append(
        f'<text x="{left + cell_w}" y="30" text-anchor="middle">{html.escape(axis_a)} levels</text>'
    )
    marks.append(
        f'<text x="18" y="{top + cell_h}" transform="rotate(-90 18 {top + cell_h})" text-anchor="middle">{html.escape(axis_b)} levels</text>'
    )
    for column, value_a in enumerate(levels_a):
        x = left + column * cell_w
        marks.append(
            f'<text x="{x + cell_w / 2}" y="70" text-anchor="middle">{html.escape(str(value_a))}</text>'
        )
    for row, value_b in enumerate(levels_b):
        y = top + row * cell_h
        marks.append(
            f'<text x="{left - 12}" y="{y + cell_h / 2}" text-anchor="end">{html.escape(str(value_b))}</text>'
        )
        for column, value_a in enumerate(levels_a):
            cell = indexed.get((value_a, value_b))
            x = left + column * cell_w
            parameters = cell.get("parameters") if isinstance(cell, dict) else None
            parameters = parameters if isinstance(parameters, dict) else {}
            share = parameters.get("memory_share")
            if isinstance(share, (int, float)) and not isinstance(share, bool):
                ratio = 0.5 if high == low else (float(share) - low) / (high - low)
                red, green, blue = (
                    round(239 - ratio * 87),
                    round(246 - ratio * 128),
                    round(255 - ratio * 175),
                )
                fill = f"#{red:02x}{green:02x}{blue:02x}"
                text_color = "#fff" if ratio > 0.55 else "#111"
                details = (
                    f"memory share {float(share) * 100:.3f}%",
                    f"total {int(parameters['total']):,}; active/token {int(parameters['active_per_token']):,}",
                    f"non-memory {int(parameters['non_memory']):,}",
                    f"table {int(parameters['memory_table']):,}; adapter {int(parameters['memory_adapter']):,}",
                )
            else:
                fill, text_color = "#eee", "#333"
                details = ("Inventory unavailable", "", "", "")
            y = top + row * cell_h
            marks.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 8}" height="{cell_h - 8}" rx="4" fill="{fill}" stroke="#555"/>'
            )
            for offset, detail in enumerate(details):
                if detail:
                    marks.append(
                        f'<text x="{x + 10}" y="{y + 26 + offset * 23}" fill="{text_color}">{html.escape(detail)}</text>'
                    )
    title = f"Architectural parameter allocation: {heatmap.get('design_id', 'factorial design')}"
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f"<title>{html.escape(title)}</title>"
        "<style>text{font:12px sans-serif}</style>"
        f'<text x="20" y="370">Configuration-derived counts; memory share colors cells. Not measured runtime cost. Context: {html.escape(_context_label(heatmap.get("context", {})))}</text>'
        + "".join(marks)
        + "</svg>\n"
    )


def _learned_curve_svg(title: str, curves: list[dict[str, object]]) -> str:
    width, height = 900, 430
    left, top, right, bottom = 72, 36, 235, 72
    plot_w, plot_h = width - left - right, height - top - bottom
    all_steps = [
        int(point["step"])
        for curve in curves
        for point in curve.get("points", [])
        if isinstance(point, dict) and type(point.get("step")) is int
    ]
    max_step = max(all_steps, default=1)
    denominator = math.log1p(max_step) or 1.0
    palette = (
        "#1769aa",
        "#d1495b",
        "#2a9d8f",
        "#e09f3e",
        "#7b2cbf",
        "#4d908e",
        "#f15bb5",
        "#577590",
        "#6a994e",
        "#bc4749",
    )
    marks: list[str] = []
    for index in range(5):
        ratio = index / 4
        y = top + plot_h * ratio
        accuracy = 1.0 - ratio
        marks.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#ddd"/>'
            f'<text x="{left - 10}" y="{y + 4:.2f}" text-anchor="end">{accuracy:.2f}</text>'
        )
    tick_steps = sorted(set(all_steps))
    if len(tick_steps) > 9:
        tick_steps = [0, 8, 32, 128, 512, 2048, 8192]
        tick_steps = [step for step in tick_steps if step <= max_step]
        if max_step not in tick_steps:
            tick_steps.append(max_step)
    for step in tick_steps:
        x = left + plot_w * math.log1p(step) / denominator
        marks.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#eee"/>'
            f'<text x="{x:.2f}" y="{height - bottom + 20}" text-anchor="middle">{step}</text>'
        )
    for index, curve in enumerate(curves):
        label = str(curve.get("label", f"series-{index + 1}"))
        color = palette[index % len(palette)]
        points = curve.get("points")
        points = points if isinstance(points, list) else []
        segments: list[list[tuple[float, float]]] = []
        current: list[tuple[float, float]] = []
        for point in points:
            if not isinstance(point, dict) or type(point.get("step")) is not int:
                if current:
                    segments.append(current)
                    current = []
                continue
            step, value = point["step"], point.get("accuracy")
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                if current:
                    segments.append(current)
                    current = []
                continue
            x = left + plot_w * math.log1p(step) / denominator
            y = top + plot_h * (1.0 - float(value))
            current.append((x, y))
        if current:
            segments.append(current)
        for segment in segments:
            if len(segment) >= 2:
                coordinates = " ".join(f"{x:.2f},{y:.2f}" for x, y in segment)
                marks.append(
                    f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="1.6"/>'
                )
            for x, y in segment:
                marks.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3" fill="{color}"/>'
                )
        legend_y = top + 14 + (index % 21) * 17
        if index < 42:
            legend_x = width - right + (index // 21) * 106
            marks.append(
                f'<line x1="{legend_x}" y1="{legend_y - 4}" x2="{legend_x + 14}" y2="{legend_y - 4}" stroke="{color}" stroke-width="2"/>'
                f'<text x="{legend_x + 19}" y="{legend_y}">{html.escape(_chart_label(label))}</text>'
            )
    marks.extend(
        [
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#555"/>',
            f'<line x1="{left}" y1="{top + plot_h}" x2="{width - right}" y2="{top + plot_h}" stroke="#555"/>',
            f'<text x="{left + plot_w / 2:.2f}" y="{height - 10}" text-anchor="middle">Training updates (log1p scale)</text>',
            f'<text x="16" y="{top + plot_h / 2:.2f}" transform="rotate(-90 16 {top + plot_h / 2:.2f})" text-anchor="middle">Accuracy</text>',
        ]
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img">'
        f"<title>{html.escape(title)}</title>"
        "<style>text{font:11px sans-serif}</style>" + "".join(marks) + "</svg>\n"
    )


def _learned_portability_charts(
    portability: dict[str, object],
) -> dict[str, str]:
    curves = portability.get("curves")
    curves = curves if isinstance(curves, list) else []
    groups = (
        ("source-monitor", "source_monitor", "source_monitor"),
        ("preparation", "preparation_validation", "preparation_copy"),
        ("recipient-calibration", "calibration", "adapter_return"),
        ("recipient-final-report", "held_out", "final_report"),
        ("recipient-final-state", "held_out", "final_state"),
    )
    charts: dict[str, str] = {}
    for name, partition, wording in groups:
        selected = [
            curve
            for curve in curves
            if isinstance(curve, dict)
            and curve.get("partition") == partition
            and curve.get("wording") == wording
        ]
        charts[f"learned-{name}.svg"] = _learned_curve_svg(
            f"Learned portability: {name.replace('-', ' ')}", selected
        )
    return charts


def render_charts(report: dict[str, object]) -> dict[str, str]:
    """Return stable SVG bytes keyed by safe fixed chart names.

    Charts intentionally render only endpoint observations supplied in the report;
    they never smooth, rank, or infer missing values.
    """
    portability = report.get("portability")
    if isinstance(portability, dict):
        return _learned_portability_charts(portability)
    charts: dict[str, str] = {}
    runs = report.get("runs")
    cards = report.get("cards")
    if isinstance(runs, list) and isinstance(cards, list):
        for card in cards:
            if not isinstance(card, dict) or not isinstance(card.get("name"), str):
                continue
            name = card["name"]
            reference = card.get("reference")
            values: list[tuple[str, float]] = []
            for run in runs:
                if not isinstance(run, dict):
                    continue
                caps = run.get("capabilities")
                cap = (
                    (
                        caps.get(reference)
                        if isinstance(reference, str)
                        else caps.get(name)
                    )
                    if isinstance(caps, dict)
                    else None
                )
                result = cap.get("result", cap) if isinstance(cap, dict) else None
                score = result.get("score") if isinstance(result, dict) else None
                if (
                    isinstance(score, (int, float))
                    and not isinstance(score, bool)
                    and math.isfinite(float(score))
                ):
                    values.append((str(run.get("run_id", "run"))[:12], float(score)))
            if values:
                charts[f"card-{len(charts) + 1}.svg"] = _svg(
                    name, values, y_label="endpoint score"
                )
    comparisons = report.get("comparisons")
    if isinstance(comparisons, list):
        values = []
        for item in comparisons:
            if not isinstance(item, dict):
                continue
            caps = item.get("capabilities")
            if not isinstance(caps, dict):
                continue
            for cap in caps.values():
                if (
                    isinstance(cap, dict)
                    and isinstance(cap.get("score_delta"), (int, float))
                    and not isinstance(cap.get("score_delta"), bool)
                    and math.isfinite(float(cap["score_delta"]))
                ):
                    values.append(
                        (str(item.get("id", "pair")), float(cap["score_delta"]))
                    )
        charts["pair-deltas.svg"] = _svg(
            "Observed per-seed pair deltas", values, y_label="variant − baseline"
        )
    analysis = report.get("research_analysis")
    if isinstance(analysis, dict):
        factorial = analysis.get("factorial")
        designs = factorial.get("designs") if isinstance(factorial, dict) else None
        if isinstance(designs, list):
            interaction_index = 0
            for design in designs:
                if not isinstance(design, dict):
                    continue
                contexts = design.get("contexts")
                if not isinstance(contexts, list):
                    continue
                grouped: dict[
                    tuple[str, str], tuple[str, str | None, list[tuple[str, float]]]
                ] = {}
                for context in contexts:
                    if not isinstance(context, dict):
                        continue
                    coordinates = context.get("coordinates", {})
                    outcomes = context.get("outcomes")
                    if not isinstance(outcomes, list):
                        continue
                    label = _context_label(coordinates)
                    for outcome in outcomes:
                        if not isinstance(outcome, dict):
                            continue
                        delta = outcome.get("interaction_delta")
                        if (
                            not isinstance(delta, (int, float))
                            or isinstance(delta, bool)
                            or not math.isfinite(float(delta))
                        ):
                            continue
                        card = outcome.get("card_reference")
                        key = (str(outcome.get("id", "outcome")), str(card or ""))
                        if key not in grouped:
                            grouped[key] = (
                                str(outcome.get("metric", "outcome")),
                                card if isinstance(card, str) else None,
                                [],
                            )
                        grouped[key][2].append((label, float(delta)))
                for metric, card, points in grouped.values():
                    interaction_index += 1
                    title = f"Factorial interaction delta: {metric}"
                    if card is not None:
                        title += f" ({card})"
                    title += f" [{design.get('id', 'design')}]"
                    charts[f"factorial-{interaction_index}.svg"] = _svg(
                        title,
                        points,
                        y_label="y11 − y10 − y01 + y00",
                    )
        allocations = analysis.get("allocation_heatmaps")
        heatmaps = (
            allocations.get("heatmaps") if isinstance(allocations, dict) else None
        )
        if isinstance(heatmaps, list):
            for heatmap in heatmaps:
                if isinstance(heatmap, dict):
                    charts[f"allocation-{len(charts) + 1}.svg"] = _allocation_svg(
                        heatmap
                    )
        allocation_curve = analysis.get("allocation_curve")
        curve_rows = (
            allocation_curve.get("rows") if isinstance(allocation_curve, dict) else None
        )
        if isinstance(curve_rows, list):
            curve_groups: dict[str, list[tuple[str, float]]] = {}
            for row in curve_rows:
                if not isinstance(row, dict):
                    continue
                coordinate = row.get("coordinate")
                ownership_label = row.get("ownership_profile")
                if not isinstance(ownership_label, str):
                    ownership_label = (
                        coordinate.get("allocation")
                        if isinstance(coordinate, dict)
                        else None
                    )
                regime_label = row.get("resource_regime")
                if not isinstance(regime_label, str):
                    regime_label = "regime"
                weight_label = row.get("neural_loss_weight")
                if not isinstance(weight_label, str):
                    weight_label = (
                        coordinate.get("neural_loss_weight")
                        if isinstance(coordinate, dict)
                        else None
                    )
                observations = row.get("checkpoint_observations")
                if not isinstance(observations, list):
                    continue
                for observation in observations:
                    if not isinstance(observation, dict):
                        continue
                    identity = observation.get("identity")
                    step = identity.get("step") if isinstance(identity, dict) else None
                    label = (
                        f"{regime_label}:{ownership_label or row.get('run_id', 'run')}/"
                        f"{weight_label or 'weight'}@{step}"
                    )
                    validation = observation.get("validation")
                    loss = (
                        validation.get("loss") if isinstance(validation, dict) else None
                    )
                    if (
                        isinstance(loss, (int, float))
                        and not isinstance(loss, bool)
                        and math.isfinite(float(loss))
                    ):
                        curve_groups.setdefault("held-out validation loss", []).append(
                            (label, float(loss))
                        )
                    capabilities = observation.get("capabilities")
                    if isinstance(capabilities, dict):
                        for task, capability in capabilities.items():
                            result = (
                                capability.get("result", capability)
                                if isinstance(capability, dict)
                                else None
                            )
                            score = (
                                result.get("score")
                                if isinstance(result, dict)
                                else None
                            )
                            if (
                                isinstance(score, (int, float))
                                and not isinstance(score, bool)
                                and math.isfinite(float(score))
                            ):
                                curve_groups.setdefault(f"task: {task}", []).append(
                                    (label, float(score))
                                )
            for metric, points in curve_groups.items():
                charts[f"allocation-curve-{len(charts) + 1}.svg"] = _svg(
                    f"Raw allocation checkpoint observations: {metric}",
                    points,
                    y_label=metric,
                )
        sweeps = analysis.get("boundary_sweeps")
        axes = sweeps.get("axes") if isinstance(sweeps, dict) else None
        if isinstance(axes, list):
            sweep_index = 0
            for axis in axes:
                if not isinstance(axis, dict):
                    continue
                groups = axis.get("groups")
                if not isinstance(groups, list):
                    continue
                for group in groups:
                    if not isinstance(group, dict):
                        continue
                    levels = group.get("levels")
                    if not isinstance(levels, list):
                        continue
                    points: list[tuple[str, float]] = []
                    for level in levels:
                        if not isinstance(level, dict):
                            continue
                        outcomes = level.get("outcomes")
                        if not isinstance(outcomes, list):
                            continue
                        validation = next(
                            (
                                item
                                for item in outcomes
                                if isinstance(item, dict)
                                and item.get("id") == "validation_loss"
                                and item.get("status") == "complete"
                            ),
                            None,
                        )
                        value = (
                            validation.get("value")
                            if isinstance(validation, dict)
                            else None
                        )
                        if isinstance(value, (int, float)) and not isinstance(
                            value, bool
                        ):
                            points.append(
                                (str(level.get("label", "level")), float(value))
                            )
                    if points:
                        sweep_index += 1
                        context_label = _context_label(group.get("context", {}))
                        charts[f"sweep-{sweep_index}.svg"] = _svg(
                            f"Observed configured sweep: {axis.get('axis')} / {context_label}",
                            points,
                            y_label="held-out validation loss",
                        )
    return charts
