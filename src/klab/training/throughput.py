"""Pure, deterministic training throughput and ETA summaries."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from math import isfinite
from statistics import median


def summarize_training_progress(
    points: Sequence[tuple[float, int]],
    *,
    total_targets: int | None,
    now: float,
    recent_window_seconds: float = 60.0,
    long_window_seconds: float = 600.0,
    stalled_after_seconds: float = 300.0,
) -> dict[str, object]:
    """Summarize cumulative target observations without reading a clock.

    Rates are medians of positive per-interval rates whose ending observation
    lies in each trailing window.  This makes the result deterministic while
    resisting a single unusually slow or fast interval.
    """
    _validate_arguments(
        points,
        total_targets=total_targets,
        now=now,
        recent_window_seconds=recent_window_seconds,
        long_window_seconds=long_window_seconds,
        stalled_after_seconds=stalled_after_seconds,
    )

    completed = points[-1][1] if points else 0
    last_progress_at = _last_progress_at(points)
    last_progress_age_seconds = (
        None if last_progress_at is None else now - last_progress_at
    )
    recent_rate = _median_rate(points, now - recent_window_seconds)
    long_rate = _median_rate(points, now - long_window_seconds)
    progress_fraction = (
        None
        if total_targets is None or total_targets <= 0
        else min(completed / total_targets, 1.0)
    )

    result: dict[str, object] = {
        "completed_targets": completed,
        "total_targets": total_targets,
        "progress_fraction": progress_fraction,
        "recent_targets_per_second": recent_rate,
        "long_targets_per_second": long_rate,
        "last_progress_at": last_progress_at,
        "last_progress_age_seconds": last_progress_age_seconds,
    }

    if total_targets is not None and completed >= total_targets:
        result["state"] = "COMPLETE"
        _set_eta(result, 0.0, 0.0, "complete", "complete")
        return result

    if last_progress_at is None or (
        last_progress_age_seconds is not None
        and last_progress_age_seconds >= stalled_after_seconds
    ):
        result["state"] = "NO_PROGRESS"
        _set_eta(result, None, None, "suspended", None)
        return result

    if recent_rate is None and long_rate is None:
        result["state"] = "RUNNING"
        _set_eta(result, None, None, "unavailable", None)
        return result

    remaining = None if total_targets is None else max(total_targets - completed, 0)
    if remaining is None:
        result["state"] = "RUNNING"
        _set_eta(result, None, None, "unavailable", None)
        return result

    rates = [rate for rate in (recent_rate, long_rate) if rate is not None]
    assert rates
    fastest_rate = max(rates)
    slowest_rate = min(rates)
    state = "RUNNING"
    if (
        recent_rate is not None
        and long_rate is not None
        and fastest_rate / slowest_rate > 2
    ):
        state = "RUNNING_UNSTABLE"
    if remaining <= max(total_targets * 0.01, 1):
        state = "COMPLETING"

    result["state"] = state
    basis = (
        "recent_and_long"
        if len(rates) == 2
        else "recent"
        if recent_rate is not None
        else "long"
    )
    _set_eta(
        result,
        remaining / fastest_rate,
        remaining / slowest_rate,
        "available",
        basis,
    )
    return result


def _set_eta(
    result: dict[str, object],
    low_seconds: float | None,
    high_seconds: float | None,
    status: str,
    basis: str | None,
) -> None:
    result.update(
        eta_low_seconds=low_seconds,
        eta_high_seconds=high_seconds,
        eta_status=status,
        eta_basis=basis,
        optimizer_only_eta={
            "low_seconds": low_seconds,
            "high_seconds": high_seconds,
            "status": status,
            "basis": basis,
        },
    )


def _median_rate(
    points: Sequence[tuple[float, int]], window_start: float
) -> float | None:
    rates = [
        (completed - previous_completed) / (elapsed - previous_elapsed)
        for (previous_elapsed, previous_completed), (elapsed, completed) in pairwise(
            points
        )
        if elapsed >= window_start
        and elapsed > previous_elapsed
        and completed > previous_completed
    ]
    return float(median(rates)) if rates else None


def _last_progress_at(points: Sequence[tuple[float, int]]) -> float | None:
    for (previous_elapsed, previous_completed), (elapsed, completed) in zip(
        reversed(points[:-1]), reversed(points[1:])
    ):
        if completed > previous_completed:
            return elapsed
    return None


def _validate_arguments(
    points: Sequence[tuple[float, int]],
    *,
    total_targets: int | None,
    now: float,
    recent_window_seconds: float,
    long_window_seconds: float,
    stalled_after_seconds: float,
) -> None:
    if total_targets is not None and (
        isinstance(total_targets, bool)
        or not isinstance(total_targets, int)
        or total_targets < 0
    ):
        raise ValueError("total_targets must be a non-negative integer or None")
    _validate_finite_number(now, "now")
    for name, value in (
        ("recent_window_seconds", recent_window_seconds),
        ("long_window_seconds", long_window_seconds),
        ("stalled_after_seconds", stalled_after_seconds),
    ):
        _validate_finite_number(value, name)
        if value < 0:
            raise ValueError(f"{name} must be non-negative")

    previous_elapsed: float | None = None
    previous_completed: int | None = None
    for index, point in enumerate(points):
        if not isinstance(point, tuple) or len(point) != 2:
            raise ValueError(
                "points must contain (elapsed_seconds, completed_targets) tuples"
            )
        elapsed, completed = point
        _validate_finite_number(elapsed, f"points[{index}][0]")
        if (
            isinstance(completed, bool)
            or not isinstance(completed, int)
            or completed < 0
        ):
            raise ValueError("completed target counts must be non-negative integers")
        if previous_elapsed is not None and elapsed < previous_elapsed:
            raise ValueError("point elapsed seconds must be monotonic")
        if previous_completed is not None and completed < previous_completed:
            raise ValueError("completed target counts must be cumulative")
        previous_elapsed = elapsed
        previous_completed = completed

    if previous_elapsed is not None and now < previous_elapsed:
        raise ValueError("now must not precede the latest observation")


def _validate_finite_number(value: object, name: str) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite number")
