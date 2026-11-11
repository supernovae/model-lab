"""Read-only probe history for the dashboard (no Streamlit import here).

Sources under the lab root: ``tries/*/try.json`` (the ``probe`` block of a lab
try) and ``probes/*/probe.json`` (``klab probe``). Live progress comes from
``tries/*/probe-progress.json`` and ``probes/*/progress.json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klab.lab_records import is_setup_record, iter_lab_records
from klab.probes.points import is_final
from klab.probes.verdict import respect_try_comparison

LIVE_STATES = {"loading", "running"}
STALE_SECONDS = 600


@dataclass(frozen=True)
class ProbeEntry:
    key: str
    source: str  # "try" or "probe"
    created_at: str
    path: Path
    result: dict[str, Any]
    question: str | None
    delta: dict[str, Any]
    comparison: dict[str, Any] | None

    @property
    def label(self) -> str:
        verdict = self.result.get("verdict") or {}
        title = (
            self.question
            or ", ".join(self.delta)
            or (self.result.get("target") or {}).get("run_id")
        )
        return (
            f"{self.key} · {title} · {str(verdict.get('status', '?')).upper()} → "
            f"{verdict.get('action', '?')}"
        )


def _read(path: Path) -> dict[str, Any] | None:
    """Unsealed live progress files only; records go through the verified reader."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError, ValueError:
        return None
    return value if isinstance(value, dict) else None


def load_history(
    lab_dir: Path, limit: int = 500
) -> tuple[list[ProbeEntry], list[tuple[Path, str]]]:
    """Verified probe results, newest first, plus rejected records and why.

    Uses the same sealed-record reader as ``klab report``: an edited
    try.json or probe.json is listed as rejected, never shown as a result.
    ``probe --config final`` results are not history: they never feed trends,
    behaviors, catalogs, pickers or next steps (the Home activity list names
    them as final verdicts without their items).
    """
    accepted, rejected = iter_lab_records(lab_dir, limit)
    entries: list[ProbeEntry] = []
    for kind, path, record in accepted:
        if is_setup_record(record):
            continue
        if kind == "try":
            probe = record.get("probe")
            if not isinstance(probe, dict) or "format" not in probe:
                continue
            entries.append(
                ProbeEntry(
                    key=str(record.get("try_id")),
                    source="try",
                    created_at=str(record.get("created_at")),
                    path=path,
                    # Older records: a NOT_COMPARABLE try never shows a
                    # promoting probe verdict.
                    result={
                        **probe,
                        "verdict": respect_try_comparison(
                            probe.get("verdict"), record.get("comparison")
                        ),
                    },
                    question=record.get("question"),
                    delta=record.get("delta") or {},
                    comparison=record.get("comparison"),
                )
            )
        elif not is_final(record):
            entries.append(
                ProbeEntry(
                    key=str(record.get("probe_id") or path.parent.name),
                    source="probe",
                    created_at=str(record.get("created_at")),
                    path=path,
                    result=record,
                    question=None,
                    delta={},
                    comparison=None,
                )
            )
    entries.sort(key=lambda entry: entry.created_at, reverse=True)
    return entries, rejected


def load_entries(lab_dir: Path, limit: int = 500) -> list[ProbeEntry]:
    return load_history(lab_dir, limit)[0]


def live_batteries(lab_dir: Path, now: datetime | None = None) -> list[dict[str, Any]]:
    """Batteries whose progress file says running and was updated recently."""
    now = now or datetime.now(UTC)
    live = []
    files = [
        *(lab_dir / "tries").glob("*/probe-progress.json"),
        *(lab_dir / "probes").glob("*/progress.json"),
    ]
    for path in files:
        state = _read(path)
        if not state or state.get("state") not in LIVE_STATES:
            continue
        try:
            updated = datetime.fromisoformat(str(state.get("updated_at")))
        except ValueError:
            updated = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        if (now - updated).total_seconds() > STALE_SECONDS:
            continue
        live.append({**state, "key": path.parent.name})
    return live


def probe_row(result: dict[str, Any], probe_id: str) -> dict[str, Any] | None:
    for row in result.get("probes") or []:
        if row.get("id") == probe_id:
            return row
    return None


# Probes measured on their own item set: their scale is the benchmark/item
# group in the row's details, never the run's eval group.
OWN_GROUP_PROBES = frozenset({"lm_eval", "fact_recall", "parametric_recall", "needle"})


def result_group(result: dict[str, Any], row: dict[str, Any]) -> str | None:
    """The comparison group one probe row of a result was measured in.

    lm-eval rows carry their benchmark group and the ranking probes their
    item group; every other probe is measured on the run's held-out data, so
    its eval group (dataset + tokenizer + windows) applies. ``None`` when the
    record predates the group (never comparable with anything).
    """
    details = row.get("details") or {}
    own = details.get("benchmark_group") or details.get("item_group")
    if own or row.get("id") in OWN_GROUP_PROBES:
        return own or None
    return (result.get("target") or {}).get("eval_group")


def history_rows(entries: Iterable[ProbeEntry]) -> list[dict[str, Any]]:
    rows = []
    for entry in entries:
        result = entry.result
        verdict = result.get("verdict") or {}
        loss = probe_row(result, "heldout_loss") or {}
        recall = probe_row(result, "fact_recall") or {}
        needle = probe_row(result, "needle") or {}
        target = result.get("target") or {}
        details = loss.get("details") or {}
        rows.append(
            {
                "when": entry.created_at,
                "id": entry.key,
                "source": entry.source,
                "idea": entry.question
                or ", ".join(entry.delta)
                or target.get("run_id"),
                "verdict": verdict.get("status"),
                "action": verdict.get("action"),
                "tier": result.get("tier"),
                "suite": (result.get("suite") or {}).get("sha256", "")[:8],
                "target": target.get("run_id"),
                "checkpoint": target.get("checkpoint_sha256"),
                "eval_group": target.get("eval_group"),
                "loss": loss.get("value"),
                "loss_delta": loss.get("delta"),
                "recall": recall.get("value"),
                "needle": needle.get("value"),
                "parameters": target.get("parameters"),
                "parameter_bytes": target.get("parameter_bytes"),
                "tokens_seen": target.get("tokens_seen"),
                "ms_per_token": details.get("ms_per_token"),
                "overfit_suspected": (result.get("guard") or {}).get(
                    "overfit_suspected"
                ),
            }
        )
    return rows
