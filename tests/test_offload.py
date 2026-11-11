"""Behavioral tests for synchronous saved-tensor activation offload.

CPU tests establish that the feature is not falsely advertised there.  The hook
path itself is deliberately exercised only on actual accelerator hardware.
"""

from __future__ import annotations

import gc

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from klab.training.offload import (
    ActivationOffload,
    probe_activation_offload,
)

MPS_AVAILABLE = bool(torch.backends.mps.is_available())
requires_mps = pytest.mark.skipif(not MPS_AVAILABLE, reason="requires actual MPS")


def test_cpu_offload_rejects_and_probe_does_not_claim_support() -> None:
    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="meaningless on CPU"):
        ActivationOffload(torch.device("cpu"), model)

    probe = probe_activation_offload(torch.device("cpu"))
    assert not probe.supported
    assert probe.measurement is None
    assert probe.estimated_device_capacity_savings_bytes == 0
    assert "meaningless" in (probe.reason or "")


@requires_mps
@pytest.mark.mps
def test_parameter_and_buffer_aliases_are_not_copied_after_updates() -> None:
    model = torch.nn.Linear(4, 4, bias=False).to("mps")
    model.register_buffer("weight_view", model.weight.detach().view(2, 8))
    offload = ActivationOffload(torch.device("mps"), model)
    for _ in range(2):
        offload.pack(model.weight.view(2, 8))
        offload.pack(model.weight_view)
        with torch.no_grad():
            model.weight.add_(1)
    assert offload.metrics.bytes_to_cpu == 0
    assert offload.metrics.live_host_bytes == 0


@requires_mps
@pytest.mark.mps
def test_mps_probe_roundtrips_without_claiming_extra_capacity() -> None:
    probe = probe_activation_offload(torch.device("mps"))
    assert probe.supported, probe.reason
    assert probe.unified_memory
    assert probe.estimated_device_capacity_savings_bytes == 0
    assert probe.measurement is not None
    assert probe.measurement.bytes_to_cpu > 0
    assert probe.measurement.bytes_to_device > 0
    assert probe.measurement.transfer_seconds >= 0


@requires_mps
@pytest.mark.mps
def test_mps_preserves_gradients_deduplicates_views_and_releases_storage() -> None:
    device = torch.device("mps")
    model = torch.nn.Linear(4, 4, bias=False).to(device)
    reference = torch.nn.Linear(4, 4, bias=False).to(device)
    reference.load_state_dict(model.state_dict())
    offload = ActivationOffload(device, model)

    # Parameter storage and its aliases remain in place even when not leaves.
    assert isinstance(offload.pack(model.weight.view(2, 8)), torch.Tensor)

    base = torch.randn(4, 4, device=device, requires_grad=True)
    first = offload.pack(base[:, :3])
    second = offload.pack(base[:, 1:])
    assert not isinstance(first, torch.Tensor)
    assert not isinstance(second, torch.Tensor)
    copied_once = offload.metrics.bytes_to_cpu
    assert copied_once == base.untyped_storage().nbytes()
    first_restored = offload.unpack(first)
    second_restored = offload.unpack(second)
    assert torch.equal(first_restored, base[:, :3])
    assert torch.equal(second_restored, base[:, 1:])
    assert offload.metrics.bytes_to_device == copied_once
    del first, second, first_restored, second_restored
    gc.collect()
    assert offload.metrics.live_host_bytes == 0
    assert offload.reset()

    x = torch.randn(3, 4, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    with offload.hooks():
        loss = model(x).square().mean()
    reference_loss = reference(reference_x).square().mean()
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(x.grad, reference_x.grad, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        model.weight.grad, reference.weight.grad, rtol=1e-4, atol=1e-5
    )
    del loss, reference_loss
    gc.collect()
    assert offload.metrics.live_host_bytes == 0


@requires_mps
@pytest.mark.mps
def test_mps_offload_composes_with_nonreentrant_recomputation() -> None:
    device = torch.device("mps")
    block = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4)
    ).to(device)
    baseline = torch.nn.Sequential(
        torch.nn.Linear(4, 8), torch.nn.SiLU(), torch.nn.Linear(8, 4)
    ).to(device)
    baseline.load_state_dict(block.state_dict())
    offload = ActivationOffload(device, block)
    x = torch.randn(2, 4, device=device, requires_grad=True)
    baseline_x = x.detach().clone().requires_grad_()

    with offload.hooks():
        loss = checkpoint(block, x, use_reentrant=False).square().sum()
    baseline_loss = checkpoint(baseline, baseline_x, use_reentrant=False).square().sum()
    loss.backward()
    baseline_loss.backward()

    torch.testing.assert_close(x.grad, baseline_x.grad, rtol=1e-4, atol=1e-5)
    for observed, expected in zip(
        block.parameters(), baseline.parameters(), strict=True
    ):
        torch.testing.assert_close(observed.grad, expected.grad, rtol=1e-4, atol=1e-5)
    assert offload.metrics.bytes_to_cpu > 0
    assert offload.metrics.bytes_to_device > 0
    assert offload.metrics.transfer_seconds >= 0


@requires_mps
@pytest.mark.mps
def test_saved_storage_versions_preserve_distinct_snapshots() -> None:
    model = torch.nn.Linear(4, 4).to("mps")
    offload = ActivationOffload(torch.device("mps"), model)
    original = torch.arange(16.0).reshape(4, 4)
    value = original.to("mps")
    first = offload.pack(value)
    value.add_(100)
    second = offload.pack(value)
    torch.testing.assert_close(offload.unpack(first).cpu(), original)
    torch.testing.assert_close(offload.unpack(second).cpu(), original + 100)
    del first, second
    assert offload.metrics.live_host_bytes == 0


@requires_mps
@pytest.mark.mps
def test_packed_snapshot_does_not_pin_original_device_allocation() -> None:
    model = torch.nn.Linear(4, 4).to("mps")
    offload = ActivationOffload(torch.device("mps"), model)
    value = torch.ones(1024 * 1024, device="mps")
    packed = offload.pack(value)
    torch.mps.synchronize()
    allocated = torch.mps.current_allocated_memory()
    size = value.numel() * value.element_size()
    del value
    torch.mps.synchronize()
    assert allocated - torch.mps.current_allocated_memory() >= size
    assert torch.all(offload.unpack(packed) == 1)
    del packed
    assert offload.metrics.live_host_bytes == 0
