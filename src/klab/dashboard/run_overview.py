"""Plain-language facts and events of a run, from its saved config and event log."""

from __future__ import annotations

import json

from klab.dashboard import ui
from klab.dashboard.queries import RunRecord


def _get(value: object, *keys: str) -> object | None:
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def facts(record: RunRecord) -> list[tuple[str, str, str]]:
    """Label, value and explanation of what was trained, on what, and how.

    A fact the run did not record is left out; nothing is filled in with a default.
    """
    config, tiles = record.config, []
    total = _int(_get(record.metadata, "inspection", "total"))
    layers = _int(_get(config, "model", "num_layers"))
    hidden = _int(_get(config, "model", "hidden_dim"))
    if total or layers:
        shape = " · ".join(
            part
            for part in (
                f"{layers} layers" if layers else "",
                f"width {hidden:,}" if hidden else "",
            )
            if part
        )
        tiles.append(
            (
                "Model",
                f"{ui.compact(total)} parameters" if total else shape,
                (shape + ". " if total and shape else "")
                + "Parameters are the numbers training adjusts.",
            )
        )
    table = _int(_get(record.metadata, "inspection", "memory_table"))
    memory = _get(config, "model", "memory")
    if memory not in (None, "none", False):
        tiles.append(
            (
                "Engram memory",
                f"{ui.compact(table)} parameters" if table else str(memory),
                (
                    f"A learned {memory} lookup table, counted in the model's total. "
                    "See the Engram memory tab."
                ),
            )
        )
    sequence = _int(_get(config, "training", "seq_len"))
    micro = _int(_get(config, "training", "micro_batch_size"))
    accumulation = _int(_get(config, "training", "gradient_accumulation"))
    if sequence and micro and accumulation:
        tiles.append(
            (
                "One update",
                f"{ui.compact(sequence * micro * accumulation)} tokens",
                (
                    f"{accumulation} × {micro} sequence(s) of {sequence:,} tokens are read "
                    "before the weights change once (gradient accumulation)."
                ),
            )
        )
    budget = _int(_get(config, "training", "max_tokens"))
    steps = _int(_get(config, "training", "max_steps"))
    if budget or steps:
        tiles.append(
            (
                "Budget",
                f"{ui.compact(budget)} tokens" if budget else f"{steps:,} updates",
                (f"{steps:,} updates. " if budget and steps else "")
                + "Training stops when this much has been read.",
            )
        )
    peak = _get(config, "optimizer", "peak")
    if isinstance(peak, float):
        warmup = _int(_get(config, "optimizer", "warmup_steps"))
        tiles.append(
            (
                "Learning rate",
                f"{peak:g} peak",
                f"{_get(config, 'optimizer', 'name') or 'optimizer'}"
                + (f", rising over the first {warmup:,} updates" if warmup else "")
                + ", then decaying. How far each update moves the weights.",
            )
        )
    backend = _get(config, "runtime", "backend")
    if backend:
        secondary = _get(config, "runtime", "secondary_device_indices")
        devices = 1 + (len(secondary) if isinstance(secondary, list) else 0)
        mode = _get(config, "runtime", "parallel", "mode")
        name = _get(record.metadata, "runtime", "device_name")
        tiles.append(
            (
                "Hardware",
                f"{devices} × {backend}" + (f" · {mode}" if mode else ""),
                (f"{name}. " if name else "")
                + f"Numbers are computed in {_get(config, 'runtime', 'precision') or 'default'} precision.",
            )
        )
    source = _get(config, "dataset", "source")
    if source:
        documents = _int(_get(config, "dataset", "train_max_documents"))
        held_out = _int(_get(config, "dataset", "validation_max_tokens"))
        tiles.append(
            (
                "Training data",
                f"{ui.compact(documents)} documents" if documents else str(source),
                f"Source {source}"
                + (
                    f"; {ui.compact(held_out)} held-out tokens for validation"
                    if held_out
                    else ""
                )
                + ".",
            )
        )
    return tiles


# Event kind -> (label, level for the chip). Kinds not listed are routine bookkeeping.
_EVENTS = {
    "recovery": ("Resumed from a checkpoint", "info"),
    "runtime_optimization_transition": ("Runtime layout carried over", "info"),
    "runtime_drift": ("Resumed under a different Lab or runtime", "warn"),
    "checkpoint_saved": ("Checkpoint saved", "pass"),
    "checkpoint_background_skipped": ("Checkpoint copy skipped", "warn"),
    "run_failed": ("Run failed", "fail"),
    "run_interrupted": ("Run stopped", "warn"),
    "resumable": ("Resumable checkpoint recorded", "pass"),
    "memory_measurements_unavailable": ("A memory reading is unavailable", "skipped"),
}


def _detail(kind: str, payload: dict[str, object]) -> str:
    if kind == "checkpoint_saved":
        size = payload.get("bytes")
        return str(payload.get("relative_path") or "") + (
            f" · {size / 2**30:.1f} GiB" if isinstance(size, int) else ""
        )
    if kind == "runtime_optimization_transition":
        changes = payload.get("changes")
        if isinstance(changes, list) and changes:
            return "; ".join(
                f"{c.get('field')}: {c.get('previous')!r} → {c.get('requested')!r}"
                for c in changes
                if isinstance(c, dict)
            )
        return "no setting changed"
    if kind == "resumable":
        return str(payload.get("checkpoint") or "")
    if kind == "memory_measurements_unavailable":
        return "; ".join(f"{name}: {why}" for name, why in payload.items())
    reason = payload.get("reason")
    return str(reason) if isinstance(reason, str) else ""


def timeline(events: list[dict[str, object]]) -> list[dict[str, object]]:
    """The events that changed the run's course, newest first, in plain words."""
    rows = []
    for event in events:
        kind = str(event.get("kind"))
        if kind not in _EVENTS:
            continue
        try:
            payload = json.loads(str(event.get("payload_json") or "{}"))
        except ValueError:
            payload = {}
        label, level = _EVENTS[kind]
        rows.append(
            {
                "run_id": event.get("run_id"),
                "step": event.get("step"),
                "tokens_seen": event.get("tokens_seen"),
                "kind": kind,
                "label": label,
                "level": level,
                "detail": _detail(kind, payload if isinstance(payload, dict) else {}),
            }
        )
    rows.reverse()
    return rows
