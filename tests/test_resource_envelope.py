from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from klab import resource_envelope as resources
from klab.resource_envelope import (
    ResourceEnvelope,
    check_envelope,
    load_resource_envelope,
)
from klab.workspace_preflight import StorageCheck


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("resource_envelope_version", 2),
        ("resource_envelope_version", True),
        ("max_rss_bytes", True),
        ("max_host_memory_fraction", 1.1),
        ("min_disk_bytes", -1),
        ("spill_to_disk", "false"),
        ("unexpected_field", 1),
    ],
)
def test_rejects_invalid_schema_fields(field: str, value: object) -> None:
    with pytest.raises(ValidationError):
        ResourceEnvelope.model_validate({"resource_envelope_version": 1, field: value})


def test_yaml_load_errors_and_preserves_absence_of_output(tmp_path: Path) -> None:
    path = tmp_path / "envelope.yaml"
    with pytest.raises(ValueError, match="cannot read resource envelope"):
        load_resource_envelope(path)
    path.write_text("[broken", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid YAML"):
        load_resource_envelope(path)
    path.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="YAML mapping"):
        load_resource_envelope(path)
    path.write_text("resource_envelope_version: 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid resource envelope"):
        load_resource_envelope(path)
    path.write_text(
        "resource_envelope_version: 1\nmax_workers: 2\nspill_to_disk: false\n",
        encoding="utf-8",
    )
    assert load_resource_envelope(path) == ResourceEnvelope(
        resource_envelope_version=1, max_workers=2, spill_to_disk=False
    )
    assert not (tmp_path / "output").exists()


@pytest.fixture
def measured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        resources.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=1000, available=400),
    )
    monkeypatch.setattr(
        resources.psutil, "swap_memory", lambda: SimpleNamespace(free=200)
    )
    monkeypatch.setattr(
        resources,
        "check_storage",
        lambda path, **kwargs: StorageCheck(
            path=str(path),
            filesystem_path=str(path.parent),
            available_bytes=300,
            available_inodes=20,
            projected_bytes=kwargs["projected_bytes"],
            projected_inodes=kwargs["projected_inodes"],
            reserve_bytes=kwargs["reserve_bytes"],
            reserve_inodes=kwargs["reserve_inodes"],
            status="adequate",
        ),
    )


@pytest.mark.usefixtures("measured")
def test_observational_measurements_and_exact_boundaries(tmp_path: Path) -> None:
    target = tmp_path / "does-not-exist" / "spool"
    measurements = check_envelope(
        None, workspace=target, rss_bytes=500, pending_workers=2, queue_depth=3
    )
    assert measurements == {
        "rss_bytes": 500,
        "host_total_ram_bytes": 1000,
        "host_available_ram_bytes": 400,
        "host_memory_fraction": 0.5,
        "swap_free_bytes": 200,
        "disk_free_bytes": 300,
        "disk_free_inodes": 20,
        "pending_workers": 2,
        "queue_depth": 3,
    }
    envelope = ResourceEnvelope(
        resource_envelope_version=1,
        max_rss_bytes=500,
        max_host_memory_fraction=0.5,
        min_available_ram_bytes=400,
        min_swap_bytes=200,
        min_disk_bytes=300,
        min_inodes=20,
        max_workers=2,
        max_queue_depth=3,
    )
    assert (
        check_envelope(
            envelope, workspace=target, rss_bytes=500, pending_workers=2, queue_depth=3
        )
        == measurements
    )
    assert not target.exists()


@pytest.mark.usefixtures("measured")
@pytest.mark.parametrize(
    ("field", "limit", "rss", "workers", "queue"),
    [
        ("max_rss_bytes", 499, 500, 0, 0),
        ("min_disk_bytes", 301, 0, 0, 0),
    ],
)
def test_every_limit_rejects_violation_before_creating_output(
    tmp_path: Path, field: str, limit: float, rss: int, workers: int, queue: int
) -> None:
    output = tmp_path / "new" / "cache"
    envelope = ResourceEnvelope.model_validate(
        {"resource_envelope_version": 1, field: limit}
    )
    with pytest.raises(ValueError, match=field):
        check_envelope(
            envelope,
            workspace=output,
            rss_bytes=rss,
            pending_workers=workers,
            queue_depth=queue,
        )
    assert not output.exists()


@pytest.mark.parametrize(
    ("unavailable", "fields"),
    [
        ("virtual_memory", ("min_available_ram_bytes", "max_host_memory_fraction")),
        ("swap_memory", ("min_swap_bytes",)),
        ("check_storage", ("min_disk_bytes", "min_inodes")),
    ],
)
def test_unavailable_measurements_fail_closed_only_if_requested(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    measured: None,
    unavailable: str,
    fields: tuple[str, ...],
) -> None:
    def unavailable_measurement(*args: object, **kwargs: object) -> None:
        raise OSError("no measurement")

    target = resources if unavailable == "check_storage" else resources.psutil
    monkeypatch.setattr(target, unavailable, unavailable_measurement)
    observed = check_envelope(None, workspace=tmp_path, rss_bytes=50)
    assert any(value is None for value in observed.values())
    for field in fields:
        with pytest.raises(ValueError, match=f"{field}.*unavailable"):
            check_envelope(
                ResourceEnvelope.model_validate(
                    {"resource_envelope_version": 1, field: 1}
                ),
                workspace=tmp_path,
                rss_bytes=50,
            )


@pytest.mark.usefixtures("measured")
def test_filesystem_with_no_inode_accounting_requires_inode_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def no_inode_accounting(path: Path, **kwargs: int) -> StorageCheck:
        return StorageCheck(
            path=str(path),
            filesystem_path=str(path.parent),
            available_bytes=300,
            available_inodes=None,
            projected_bytes=0,
            projected_inodes=0,
            reserve_bytes=0,
            reserve_inodes=0,
            status="adequate",
        )

    monkeypatch.setattr(resources, "check_storage", no_inode_accounting)
    assert (
        check_envelope(
            ResourceEnvelope(resource_envelope_version=1, min_disk_bytes=300),
            workspace=tmp_path,
            rss_bytes=None,
        )["disk_free_inodes"]
        is None
    )
    with pytest.raises(ValueError, match="min_inodes.*unavailable"):
        check_envelope(
            ResourceEnvelope(resource_envelope_version=1, min_inodes=0),
            workspace=tmp_path,
            rss_bytes=None,
        )


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("rss_bytes", -1),
        ("pending_workers", True),
    ],
)
def test_invalid_operational_counts_rejected(
    tmp_path: Path, name: str, value: object
) -> None:
    arguments: dict[str, object] = {
        "rss_bytes": 0,
        "pending_workers": 0,
        "queue_depth": 0,
    }
    arguments[name] = value
    with pytest.raises(ValueError, match=name):
        check_envelope(None, workspace=tmp_path, **arguments)


def test_checks_disk_on_existing_ancestor_without_creating_output(
    tmp_path: Path,
) -> None:
    target = tmp_path / "new" / "nested"
    result = check_envelope(None, workspace=target, rss_bytes=None)
    assert result["disk_free_bytes"] is not None
    assert result["disk_free_inodes"] is None or result["disk_free_inodes"] >= 0
    assert not target.exists()


def test_scientific_config_identity_excludes_operational_policy() -> None:
    from klab.config.loading import load_config
    from klab.config.models import RunConfig

    config = load_config(
        Path(__file__).resolve().parents[1] / "configs" / "runtime_smoke_cpu.yaml"
    )
    scientific = config.model_dump(mode="json")
    assert "resource_envelope" not in scientific
    with pytest.raises(ValidationError, match="resource_envelope"):
        RunConfig.model_validate(
            {**scientific, "resource_envelope": {"resource_envelope_version": 1}}
        )


def test_rss_permission_failure_is_unknown(monkeypatch, tmp_path):
    def denied():
        raise resources.psutil.AccessDenied(pid=1)

    monkeypatch.setattr(resources.psutil, "Process", denied)
    rss = resources.current_process_rss_bytes()
    assert rss is None
    with pytest.raises(ValueError, match="max_rss_bytes.*unavailable"):
        check_envelope(
            ResourceEnvelope(resource_envelope_version=1, max_rss_bytes=1024),
            workspace=tmp_path,
            rss_bytes=rss,
        )
