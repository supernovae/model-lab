"""Attributed cost and spend caps for priced workers."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

from klab.workers.controller import Controller
from klab.workers.cost import attempt_cost, worker_spend
from klab.workers.models import WorkerDefinition, WorkerPricing

NOW = datetime(2026, 11, 12, 12, 0, tzinfo=UTC)


def _attempt(status: str, started: str | None, finished: str | None = None) -> dict:
    receipt = {"started_at": started, "finished_at": finished}
    return {
        "worker_id": "rented",
        "status": status,
        "terminal_receipt": receipt if finished else None,
        "receipt": receipt,
    }


def test_cost_is_run_time_at_the_hourly_rate() -> None:
    pricing = WorkerPricing(hourly_rate=0.6)
    done = attempt_cost(
        _attempt("COMPLETE", "2026-11-12T10:00:00Z", "2026-11-12T10:30:00Z"),
        pricing,
        NOW,
    )
    assert done == {
        "currency": "USD",
        "hourly_rate": 0.6,
        "seconds": 1800.0,
        "amount": pytest.approx(0.3),
        "final": True,
    }
    running = attempt_cost(_attempt("RUNNING", "2026-11-12T11:00:00Z"), pricing, NOW)
    assert running["amount"] == pytest.approx(0.6) and not running["final"]
    queued = attempt_cost(_attempt("QUEUED", None), pricing, NOW)
    assert queued["amount"] is None


def test_spend_counts_only_this_worker() -> None:
    pricing = WorkerPricing(hourly_rate=1.0)
    attempts = [
        _attempt("COMPLETE", "2026-11-12T10:00:00Z", "2026-11-12T11:00:00Z"),
        {
            **_attempt("COMPLETE", "2026-11-12T09:00:00Z", "2026-11-12T11:00:00Z"),
            "worker_id": "other",
        },
    ]
    assert worker_spend(attempts, "rented", pricing, NOW) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "fields",
    [
        {"hourly_rate": 0},
        {"hourly_rate": 1, "currency": "usd"},
        {"hourly_rate": 1, "max_spend": -5},
    ],
)
def test_invalid_pricing_is_refused(fields: dict) -> None:
    with pytest.raises(ValueError):
        WorkerPricing(**fields)


def _priced_store(
    tmp_path: Path, max_spend: float | None
) -> tuple[Controller, WorkerDefinition]:
    from klab.workers.execution import _caps
    from klab.workers.models import WorkerCapabilities

    definition = WorkerDefinition(
        worker_id="rented",
        name="rented",
        transport="ssh",
        host="gpu-host",
        python=Path(sys.executable).absolute(),
        root=tmp_path / "worker",
        engine="pytorch",
        backend="cpu",
        device_index=0,
        pricing=WorkerPricing(hourly_rate=2.0, max_spend=max_spend),
    )
    capability = WorkerCapabilities.model_validate(_caps(definition))
    controller = Controller(tmp_path / "controller")
    controller.store.save_worker(
        "rented",
        {
            "definition": definition.model_dump(mode="json"),
            "capabilities": capability.model_dump(mode="json"),
        },
    )
    controller.store.enqueue_many(
        [
            {
                "experiment_id": "experiment-1",
                "attempt_id": "attempt-1",
                "run_id": "run-1",
                "spec": {"schema_version": 1, "experiment_id": "experiment-1"},
            }
        ]
    )
    controller.store.assign("attempt-1", "rented")
    controller.store.set_receipt(
        "attempt-1",
        {
            "state": "COMPLETE",
            "started_at": "2026-11-12T10:00:00Z",
            "finished_at": "2026-11-12T11:30:00Z",
        },
        "COMPLETE",
    )
    return controller, definition


def test_listing_and_status_report_attributed_cost(tmp_path: Path) -> None:
    controller, definition = _priced_store(tmp_path, max_spend=None)
    (row,) = controller.list_experiments()
    assert row["cost"]["amount"] == pytest.approx(3.0)
    spend = controller.spend(definition)
    assert spend == {
        "currency": "USD",
        "hourly_rate": 2.0,
        "attributed": pytest.approx(3.0),
        "max_spend": None,
    }
    assert controller.spend_cap_reason(definition) is None


def test_a_worker_at_its_cap_takes_no_more_work(tmp_path: Path) -> None:
    controller, definition = _priced_store(tmp_path, max_spend=3.0)
    assert controller.spend_cap_reason(definition) == (
        "spend cap reached (3.00 of 3.00 USD)"
    )
    roomy = definition.model_copy(
        update={"pricing": WorkerPricing(hourly_rate=2.0, max_spend=10.0)}
    )
    assert controller.spend_cap_reason(roomy) is None


def test_unpriced_definitions_serialize_unchanged() -> None:
    plain = WorkerDefinition(
        worker_id="w",
        name="w",
        transport="local",
        python=Path(sys.executable).absolute(),
        root=Path("/tmp/w"),
        engine="pytorch",
        backend="cpu",
        device_index=0,
    )
    assert "pricing" not in plain.model_dump(mode="json")
