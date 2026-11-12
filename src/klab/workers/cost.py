"""Attributed cost of queued work on priced workers.

A rented machine bills for every hour it exists. Lab sees only the time an
attempt runs, from its receipt's start to its finish, so the cost here is the
attempt's wall time at the worker's hourly rate: a lower bound on the bill that
excludes setup, idle time and storage.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

ACTIVE = frozenset({"ASSIGNED", "RUNNING", "CANCEL_REQUESTED"})


def _time(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        moment = datetime.fromisoformat(value)
    except ValueError:
        return None
    return moment if moment.tzinfo else moment.replace(tzinfo=UTC)


def attempt_seconds(attempt: Mapping[str, Any], now: datetime) -> float | None:
    """Seconds the attempt has run: to its finish, or to now while active."""
    receipt = attempt.get("terminal_receipt") or attempt.get("receipt") or {}
    started = _time(receipt.get("started_at"))
    if started is None:
        return None
    finished = _time(receipt.get("finished_at"))
    if finished is None:
        if attempt.get("status") not in ACTIVE:
            return None
        finished = now
    return max(0.0, (finished - started).total_seconds())


def attempt_cost(
    attempt: Mapping[str, Any], pricing: Any, now: datetime
) -> dict[str, Any]:
    seconds = attempt_seconds(attempt, now)
    return {
        "currency": pricing.currency,
        "hourly_rate": pricing.hourly_rate,
        "seconds": seconds,
        "amount": None if seconds is None else seconds / 3600 * pricing.hourly_rate,
        "final": attempt.get("status") not in ACTIVE,
    }


def worker_spend(
    attempts: Iterable[Mapping[str, Any]], worker_id: str, pricing: Any, now: datetime
) -> float:
    """Attributed spend of every attempt this worker has run in the store."""
    total = 0.0
    for attempt in attempts:
        if attempt.get("worker_id") != worker_id:
            continue
        amount = attempt_cost(attempt, pricing, now)["amount"]
        total += amount or 0.0
    return total
