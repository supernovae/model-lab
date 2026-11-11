import pytest

from klab.training.throughput import summarize_training_progress


def test_summarizes_exact_rates_and_eta() -> None:
    summary = summarize_training_progress(
        [(0.0, 0), (10.0, 20), (20.0, 40)],
        total_targets=100,
        now=20.0,
    )

    assert summary == {
        "completed_targets": 40,
        "total_targets": 100,
        "progress_fraction": 0.4,
        "recent_targets_per_second": 2.0,
        "long_targets_per_second": 2.0,
        "last_progress_at": 20.0,
        "last_progress_age_seconds": 0.0,
        "state": "RUNNING",
        "eta_low_seconds": 30.0,
        "eta_high_seconds": 30.0,
        "eta_status": "available",
        "optimizer_only_eta": {
            "low_seconds": 30.0,
            "high_seconds": 30.0,
            "status": "available",
            "basis": "recent_and_long",
        },
        "eta_basis": "recent_and_long",
    }


def test_returns_null_rates_before_two_observations() -> None:
    summary = summarize_training_progress([(0.0, 0)], total_targets=100, now=0.0)

    assert summary["recent_targets_per_second"] is None
    assert summary["long_targets_per_second"] is None
    assert summary["eta_low_seconds"] is None
    assert summary["eta_high_seconds"] is None
    assert summary["eta_status"] == "suspended"
    assert summary["optimizer_only_eta"]["status"] == "suspended"
    assert summary["state"] == "NO_PROGRESS"


def test_zero_rate_is_unavailable_and_no_progress() -> None:
    summary = summarize_training_progress(
        [(0.0, 0), (10.0, 0)], total_targets=100, now=10.0
    )

    assert summary["recent_targets_per_second"] is None
    assert summary["long_targets_per_second"] is None
    assert summary["last_progress_at"] is None
    assert summary["state"] == "NO_PROGRESS"
    assert summary["eta_status"] == "suspended"
    assert summary["eta_low_seconds"] is None
    assert summary["eta_high_seconds"] is None


def test_completed_budget_has_exact_zero_eta() -> None:
    summary = summarize_training_progress(
        [(0.0, 0), (10.0, 100)], total_targets=100, now=10.0
    )

    assert summary["progress_fraction"] == 1.0
    assert summary["state"] == "COMPLETE"
    assert summary["eta_low_seconds"] == 0.0
    assert summary["eta_high_seconds"] == 0.0
    assert summary["optimizer_only_eta"]["low_seconds"] == 0.0
    assert summary["optimizer_only_eta"]["high_seconds"] == 0.0
    assert summary["eta_status"] == "complete"


def test_marks_divergent_recent_and_long_rates_unstable() -> None:
    summary = summarize_training_progress(
        [(0.0, 0), (50.0, 50), (100.0, 100), (110.0, 200)],
        total_targets=300,
        now=110.0,
        recent_window_seconds=10.0,
        long_window_seconds=600.0,
    )

    assert summary["recent_targets_per_second"] == 5.5
    assert summary["long_targets_per_second"] == 1.0
    assert summary["state"] == "RUNNING_UNSTABLE"


def test_stalled_progress_suspends_an_otherwise_available_eta() -> None:
    summary = summarize_training_progress(
        [(0.0, 0), (10.0, 10)],
        total_targets=100,
        now=310.0,
        stalled_after_seconds=300.0,
    )

    assert summary["last_progress_age_seconds"] == 300.0
    assert summary["state"] == "NO_PROGRESS"
    assert summary["eta_status"] == "suspended"
    assert summary["eta_low_seconds"] is None
    assert summary["eta_high_seconds"] is None
    assert summary["optimizer_only_eta"]["status"] == "suspended"


@pytest.mark.parametrize(
    ("points", "total_targets", "now"),
    [
        ([(1.0, 1), (0.0, 2)], 10, 1.0),
        ([(0.0, 2), (1.0, 1)], 10, 1.0),
        ([(0.0, 1.5)], 10, 0.0),
        ([(0.0, 0)], -1, 0.0),
        ([(0.0, 0)], 10, -1.0),
    ],
)
def test_rejects_malformed_or_non_monotonic_input(
    points: list[tuple[float, int]], total_targets: int, now: float
) -> None:
    with pytest.raises(ValueError):
        summarize_training_progress(points, total_targets=total_targets, now=now)
