"""Readable terminal rendering of a probe result (``--json`` is the contract).

Plain ANSI, no extra dependency: colors are used only for a TTY (or
``FORCE_COLOR``) and never when ``NO_COLOR`` is set.
"""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any, TextIO

GLYPH = {
    "pass": "✔",
    "warn": "▲",
    "fail": "✖",
    "info": "•",
    "skipped": "⊘",
    "unavailable": "○",
    "not_comparable": "≠",
    "error": "!",
    "incomplete": "…",
}
COLOR = {
    "pass": "32",
    "warn": "33",
    "fail": "31",
    "info": "36",
    "skipped": "2",
    "unavailable": "33",
    "not_comparable": "35",
    "error": "31;1",
    "incomplete": "33",
}
ACTION_TEXT = {
    "abandon": "ABANDON",
    "tweak": "TWEAK",
    "rerun": "RERUN (missing evidence)",
    "escalate": "ESCALATE",
    "longer_run": "LONGER RUN",
    "compare": "COMPARE",
    "report": "REPORT (final verdict)",
}
SPARK = "▁▂▃▄▅▆▇█"


def use_color(stream: TextIO | None = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("FORCE_COLOR"):
        return True
    stream = stream or sys.stdout
    return bool(getattr(stream, "isatty", lambda: False)())


def _paint(text: str, code: str, color: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if color else text


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "–"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        if value != 0 and abs(value) < 10**-digits:
            return f"{value:.1e}"
        return f"{value:.{digits}f}"
    return str(value)


def _compact(value: Any) -> str:
    if not isinstance(value, (int, float)) or value is None:
        return "–"
    for unit, size in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(value) >= size:
            return f"{value / size:.1f}{unit}"
    return str(int(value))


def params_text(who: Mapping[str, Any]) -> str:
    """``135M params`` or ``162M params (106M active)`` when they differ."""
    total, active = who.get("parameters"), who.get("active_parameters")
    text = f"{_compact(total)} params"
    if active is not None and total is not None and active != total:
        text += f" ({_compact(active)} active)"
    return text


def sparkline(values: Sequence[float | None]) -> str:
    cells = []
    for value in values:
        if value is None:
            cells.append("·")
            continue
        index = min(len(SPARK) - 1, max(0, round(value * (len(SPARK) - 1))))
        cells.append(SPARK[index])
    return "".join(cells)


def meter(row: Mapping[str, Any], color: bool, width: int = 5) -> str:
    """Centered meter: left = better than baseline, right = worse.

    The full half-width corresponds to the probe's fail threshold.
    """
    regression = row.get("regression")
    thresholds = row.get("thresholds") or {}
    scale = thresholds.get("fail") or thresholds.get("warn")
    if regression is None or not scale:
        return " " * (2 * width + 1)
    filled = min(width, max(0, round(abs(regression) / scale * width)))
    if row.get("within_noise"):
        filled = min(filled, 1)
    left = "·" * width
    right = "·" * width
    if regression < 0:
        left = "·" * (width - filled) + "◀" * filled
        left = _paint(left, "32", color) if filled else left
    elif regression > 0:
        right = "▶" * filled + "·" * (width - filled)
        code = {"fail": "31", "warn": "33"}.get(str(row.get("status")))
        right = _paint(right, code, color) if filled and code else right
    return f"{left}│{right}"


def _delta(row: Mapping[str, Any]) -> str:
    delta = row.get("delta")
    if delta is None:
        return ""
    text = f"Δ {delta:+.3f}"
    if row.get("thresholds", {}).get("mode") == "delta_rel" and row.get(
        "baseline_value"
    ):
        text += f" ({delta / abs(row['baseline_value']):+.1%})"
    if row.get("delta_se") is not None:
        text += f" ±{row['delta_se']:.3f}"
        if row["id"] == "heldout_loss":
            text += " (eval noise, 1 seed)"
    return text


def _extra(row: Mapping[str, Any]) -> str:
    details = row.get("details") or {}
    if row["id"] == "heldout_loss" and details.get("perplexity") is not None:
        return f"ppl {details['perplexity']:.1f}"
    if row["id"] == "token_agreement" and "js" in details:
        text = f"JS {details['js']:.3f} · KL {details['kl_base_to_candidate']:.3f}"
        if details.get("informative") is False:
            text += " (near-identical: agreement uninformative)"
        return text
    if row["id"] == "repetition" and details.get("distinct_2") is not None:
        return f"distinct-2 {details['distinct_2']:.2f}"
    if row["id"] == "needle" and details.get("by_length"):
        lengths = details["by_length"]
        accs = [lengths[k]["accuracy"] for k in lengths]
        if all(lengths[k].get("chars") is not None for k in lengths):
            chars = "/".join(str(lengths[k]["chars"]) for k in lengths)
            return f"by length {sparkline(accs)} ({chars} chars)"
        tokens = "/".join(str(lengths[k]["tokens"]) for k in lengths)
        return f"by length {sparkline(accs)} ({tokens} tok)"
    if row["id"] == "fact_recall" and details.get("chance") is not None:
        return f"chance {details['chance']:.2f}"
    if row["id"] == "parametric_recall" and details.get("control_accuracy") is not None:
        return (
            f"never-trained control {details['control_accuracy']:.2f} "
            f"(chance {details['chance']:.2f})"
        )
    if row["id"] == "lm_eval" and details.get("tasks"):
        return " ".join(
            f"{task.split('_')[0]} {_num(v.get('acc'), 2)}"
            for task, v in details["tasks"].items()
        )
    return ""


def render(result: Mapping[str, Any], *, color: bool = False) -> str:
    from klab.lab_records import is_setup_record
    from klab.lab_results import baseline_text

    if is_setup_record(result):
        return "\n".join(
            [
                f"PROBE {result['probe_id']}  status: {result['status']}",
                baseline_text(result),
                f"  setup: {result.get('failure', 'incomplete; no scoring evidence')}",
            ]
        )
    suite = result["suite"]
    target = result["target"]
    baseline = result.get("baseline")
    tiers = " → ".join(result.get("tiers_run") or []) or "none"
    final = bool(result.get("final")) or suite.get("verdict_split") == "final"
    lines = [
        _paint("PROBE BATTERY", "1", color)
        + (_paint("  FINAL (held-back split)", "1", color) if final else "")
        + f"  {suite['name']} v{suite['version']} · suite {suite['sha256'][:8]}"
        + f" · tier {result['tier']} (ran {tiers}) · {result['seconds']:.1f}s",
    ]
    if "probe_id" in result:
        lines.append(baseline_text(result))
    for label, who in (("candidate", target), ("baseline ", baseline)):
        if who is None:
            continue
        lines.append(
            f"  {label} {who['run_id']}  step {_num(who.get('step'))} · "
            f"{_compact(who.get('tokens_seen'))} tokens · "
            f"{params_text(who)}"
        )
        ref = who.get("reference")
        if ref:
            lines.append(
                _paint(
                    f"            {ref['repo_id']}@{ref['revision'][:12]} · "
                    f"{ref['license']} · weights {who['checkpoint_sha256'][:12]}",
                    "2",
                    color,
                )
            )
    lines.append("")
    title_width = max(len(row["title"]) for row in result["probes"])
    for row in result["probes"]:
        status = row["status"]
        badge = _paint(
            f"{GLYPH.get(status, '?')} {status.upper().replace('_', ' '):<11}",
            COLOR.get(status, "0"),
            color,
        )
        if status == "skipped" or row.get("value") is None:
            note = row.get("note") or ""
            lines.append(
                f"  {badge} {row['title']:<{title_width}}  {_paint(note, '2', color)}"
            )
            continue
        value = f"{_num(row['value'])}"
        if row.get("baseline_value") is not None:
            value += f" vs {_num(row['baseline_value'])}"
        lines.append(
            f"  {badge} {row['title']:<{title_width}}  {value:<17} "
            f"{meter(row, color)}  {_delta(row):<26} {_extra(row)}"
        )
        if status in {"warn", "fail"}:
            lines.append(
                " " * 13 + _paint(f"↳ {row['suggests']}", COLOR[status], color)
            )
        elif row.get("note") and status in {"not_comparable", "error", "info"}:
            lines.append(" " * 13 + _paint(f"↳ {row['note']}", "2", color))
    stop = result.get("stop") or {}
    if stop.get("stopped"):
        lines.append(_paint(f"\n  ⚡ {stop['reason']}", "33", color))
    verdict = result["verdict"]
    status = verdict["status"]
    action = ACTION_TEXT.get(verdict["action"], verdict["action"])
    lines.append("")
    lines.append(
        "  verdict "
        + _paint(
            f"{GLYPH.get(status, '?')} {status.upper()}",
            COLOR.get(status, "0") + ";1",
            color,
        )
        + "  next → "
        + _paint(action, "1", color)
        + (f" (--config {verdict['next_tier']})" if verdict.get("next_tier") else "")
    )
    lines.append(f"  {verdict['suggestion']}")
    for gap in verdict.get("missing") or []:
        lines.append(
            _paint("  missing ", "33", color)
            + f"{gap['id']} ({gap['status']}): {gap.get('note') or ''}".rstrip(": ")
        )
    guard = result.get("guard") or {}
    if guard.get("probes"):
        parts = [
            f"{name} dev {v['dev_gain']:+.2f} / held-out {v['heldout_gain']:+.2f}"
            + (" ⚠" if v["flag"] else "")
            for name, v in guard["probes"].items()
        ]
        mark = (
            _paint("overfit suspected", "31", color)
            if guard.get("overfit_suspected")
            else _paint("ok", "32", color)
        )
        lines.append(
            _paint("  guard   ", "2", color)
            + (
                "FINAL verdict on the held-back final split"
                if guard.get("verdict_split") == "final"
                else "verdicts use the held-out split"
            )
            + f" · {mark} · "
            + "; ".join(parts)
        )
    lines.append(
        _paint("  legend  meter ◀ better │ worse ▶ (full = fail threshold)", "2", color)
    )
    return "\n".join(lines)
