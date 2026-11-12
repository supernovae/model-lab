from __future__ import annotations

import copy
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from klab.config.models import (
    ActivationOffloadConfig,
    AdamWConfig,
    DatasetConfig,
    LoggingConfig,
    MemoryConfig,
    ModelConfig,
    RunConfig,
    RuntimeConfig,
    TokenizerConfig,
    TrainingConfig,
)
from klab.engines.base import Microbatch
from klab.engines.pytorch import PyTorchEngine
from klab.workers.execution import _acquire_config_lease
from klab.workers.leases import (
    MultiDeviceLease,
    acquire_lease,
    acquire_multi_device_leases,
)


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_run_config(
    tmp_path: Path,
    *,
    secondary_device_indices: tuple[int, ...] = (),
) -> RunConfig:
    return RunConfig(
        schema_version=2,
        name="multi-gpu-test",
        seed=42,
        runtime=RuntimeConfig(
            engine="pytorch",
            backend="cpu",
            device_index=0,
            secondary_device_indices=secondary_device_indices,
            precision="fp32",
        ),
        model=ModelConfig(
            vocab_size=260,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            ffn_dim=64,
            max_seq_len=16,
            memory="ngram",
            memory_table_size=64,
            memory_ngram_size=3,
            memory_dim=16,
            memory_ngram_orders=(2, 3),
            memory_hash_heads=2,
        ),
        tokenizer=TokenizerConfig(path=tmp_path / "tokenizer.json"),
        dataset=DatasetConfig(
            source="synthetic",
            cache_dir=tmp_path / "cache",
            train_max_documents=32,
            validation_max_documents=8,
            train_max_tokens=1024,
            validation_max_tokens=256,
        ),
        training=TrainingConfig(
            micro_batch_size=2,
            gradient_accumulation=4,
            seq_len=16,
            max_steps=10,
            max_tokens=1280,
            deterministic=True,
        ),
        optimizer=AdamWConfig(
            peak=1e-3,
            floor=1e-4,
            warmup_steps=2,
            weight_decay=0.1,
        ),
        logging=LoggingConfig(
            root_dir=tmp_path / "runs",
            every_steps=1,
            architecture_diagnostics="full",
        ),
    )


def _make_microbatches(
    seed: int,
    *,
    count: int = 4,
    batch_size: int = 2,
    seq_len: int = 16,
    vocab_size: int = 128,
) -> tuple[list[Microbatch], int]:
    rng = np.random.default_rng(seed)
    batches: list[Microbatch] = []
    for idx in range(count):
        inputs = rng.integers(0, vocab_size, size=(batch_size, seq_len), dtype=np.int64)
        targets = rng.integers(
            0, vocab_size, size=(batch_size, seq_len), dtype=np.int64
        )
        if idx % 2 == 1:
            targets[:, -2:] = -100
        batches.append(Microbatch(inputs=inputs, targets=targets))
    valid_targets = sum(
        int(np.count_nonzero(batch.targets != -100)) for batch in batches
    )
    return batches, valid_targets


def _named_optimizer_slots(engine: PyTorchEngine) -> dict[str, dict[str, torch.Tensor]]:
    assert engine.model is not None and engine.optimizer is not None
    names = {id(parameter): name for name, parameter in engine.model.named_parameters()}
    return {
        names[id(parameter)]: {
            key: value.detach().clone()
            for key, value in values.items()
            if isinstance(value, torch.Tensor)
        }
        for parameter, values in engine.optimizer.state.items()
    }


def test_runtime_config_secondary_device_indices_validation(tmp_path: Path) -> None:
    default_cfg = RuntimeConfig()
    assert default_cfg.secondary_device_indices == ()
    assert "secondary_device_indices" not in default_cfg.model_dump(mode="python")
    assert "secondary_device_indices" not in default_cfg.model_dump(mode="json")

    cuda_cfg = RuntimeConfig(
        engine="pytorch",
        backend="cuda",
        device_index=0,
        secondary_device_indices=(1,),
    )
    assert cuda_cfg.secondary_device_indices == (1,)
    assert cuda_cfg.model_dump(mode="python")["secondary_device_indices"] == (1,)
    assert cuda_cfg.model_dump(mode="json")["secondary_device_indices"] == [1]

    xpu_cfg = RuntimeConfig(
        engine="pytorch",
        backend="xpu",
        device_index=0,
        secondary_device_indices=(1,),
    )
    assert xpu_cfg.secondary_device_indices == (1,)

    cpu_cfg = RuntimeConfig(
        engine="pytorch",
        backend="cpu",
        device_index=0,
        secondary_device_indices=(0,),
    )
    assert cpu_cfg.secondary_device_indices == (0,)

    with pytest.raises(ValidationError, match="cannot be combined with fp16"):
        RuntimeConfig(
            engine="pytorch",
            backend="cuda",
            device_index=0,
            secondary_device_indices=(1,),
            precision="fp16",
        )

    with pytest.raises(ValidationError, match="must not contain duplicates"):
        RuntimeConfig(
            engine="pytorch",
            backend="cuda",
            device_index=0,
            secondary_device_indices=(1, 1),
        )

    with pytest.raises(ValidationError, match="must be non-negative"):
        RuntimeConfig(
            engine="pytorch",
            backend="cuda",
            device_index=0,
            secondary_device_indices=(-1,),
        )

    for backend in ("xpu", "cuda"):
        with pytest.raises(
            ValidationError, match="must not include the primary device_index"
        ):
            RuntimeConfig(
                engine="pytorch",
                backend=backend,
                device_index=0,
                secondary_device_indices=(0,),
            )

    for engine, backend in (
        ("pytorch", "mps"),
        ("pytorch", "metal"),
        ("mlx", "metal"),
    ):
        with pytest.raises(
            ValidationError,
            match="requires PyTorch on cpu, cuda, rocm, or xpu",
        ):
            RuntimeConfig(
                engine=engine,
                backend=backend,
                device_index=0,
                secondary_device_indices=(1,),
            )

    with pytest.raises(
        ValidationError, match="cannot be combined with activation offload"
    ):
        RuntimeConfig(
            engine="pytorch",
            backend="cuda",
            device_index=0,
            secondary_device_indices=(1,),
            memory=MemoryConfig(
                activation_offload=ActivationOffloadConfig(enabled=True)
            ),
        )


def test_acquire_multi_device_leases_and_contention_rollback(tmp_path: Path) -> None:
    state_dir = tmp_path / "leases"
    dev_a = "phys-b70-e300"
    dev_b = "phys-b70-2300"

    with acquire_multi_device_leases(
        worker_id="trainer",
        backend="xpu",
        physical_device_ids=(dev_a, dev_b),
        state_dir=state_dir,
    ) as multi:
        assert multi is not None
        assert len(multi.leases) == 2
        assert [lease.key for lease in multi.leases] == [
            f"physical-{dev_a}",
            f"physical-{dev_b}",
        ]
        assert multi.leases[0].metadata["worker_id"] == "trainer"
        assert multi.leases[1].metadata["worker_id"] == "trainer-dev1"
        assert len(multi.inherited_fds) == 6
        for lease in multi.leases:
            assert lease.path.is_file()

        before_heartbeats = [item["heartbeat_at"] for item in multi.metadata]
        time.sleep(0.01)
        multi.heartbeat()
        after_heartbeats = [item["heartbeat_at"] for item in multi.metadata]
        assert after_heartbeats[0] > before_heartbeats[0]
        assert after_heartbeats[1] > before_heartbeats[1]
        for idx, lease in enumerate(multi.leases):
            persisted = json.loads(lease.path.read_text(encoding="utf-8"))
            assert persisted["heartbeat_at"] == pytest.approx(after_heartbeats[idx])

    with acquire_multi_device_leases(
        worker_id="trainer-rev",
        backend="xpu",
        physical_device_ids=(dev_b, dev_a),
        state_dir=state_dir,
    ) as multi_rev:
        assert multi_rev is not None
        assert [lease.key for lease in multi_rev.leases] == [
            f"physical-{dev_b}",
            f"physical-{dev_a}",
        ]
        assert multi_rev.leases[0].metadata["worker_id"] == "trainer-rev"
        assert multi_rev.leases[1].metadata["worker_id"] == "trainer-rev-dev1"

    # In-process contention is valid because leases.py calls os.open() per acquire,
    # and fcntl.flock() locks bind to the kernel open file description, not the PID.
    for held_dev, free_dev in ((dev_b, dev_a), (dev_a, dev_b)):
        held_lease = acquire_lease(
            worker_id="external-holder",
            backend="xpu",
            physical_device_id=held_dev,
            state_dir=state_dir,
        )
        assert held_lease is not None
        try:
            contended = acquire_multi_device_leases(
                worker_id="contended-trainer",
                backend="xpu",
                physical_device_ids=(dev_a, dev_b),
                state_dir=state_dir,
            )
            assert contended is None

            free_lease = acquire_lease(
                worker_id="probe-free",
                backend="xpu",
                physical_device_id=free_dev,
                state_dir=state_dir,
            )
            assert free_lease is not None
            free_lease.close()
        finally:
            held_lease.close()


def test_pytorch_engine_multi_device_microbatch_parallel_equivalence(
    tmp_path: Path,
) -> None:
    cfg_single = _make_run_config(tmp_path / "single", secondary_device_indices=())
    cfg_multi = _make_run_config(tmp_path / "multi", secondary_device_indices=(0,))

    engine_single = PyTorchEngine()
    engine_multi = PyTorchEngine()
    try:
        validated_runtime = engine_single.validate(cfg_single)
        engine_multi.runtime = validated_runtime
        _seed_all(cfg_single.seed)
        engine_single.initialize(cfg_single)

        _seed_all(cfg_multi.seed)
        engine_multi.initialize(cfg_multi)
        assert engine_single.model is not None and engine_multi.model is not None
        for (name_s, param_s), (name_m, param_m) in zip(
            engine_single.model.named_parameters(),
            engine_multi.model.named_parameters(),
            strict=True,
        ):
            assert name_s == name_m
            assert torch.equal(param_s, param_m), f"initial mismatch on {name_s}"

        for step in (1, 2):
            microbatches, valid_targets = _make_microbatches(100 + step)
            res_single = engine_single.train_update(microbatches, step, valid_targets)
            res_multi = engine_multi.train_update(microbatches, step, valid_targets)

            assert res_single.outcome == "APPLIED"
            assert res_multi.outcome == "APPLIED"
            assert res_single.committed_targets == valid_targets
            assert res_multi.committed_targets == valid_targets
            assert res_single.metrics is not None and res_multi.metrics is not None
            assert res_multi.metrics["batch/devices_used"] == 2.0
            assert "batch/devices_used" not in res_single.metrics
            # Per-device timings appear only when replicas ran, and are sane.
            for name in (
                "performance/primary_compute_seconds",
                "performance/replica_compute_seconds",
                "performance/device_wait_seconds",
                "performance/gradient_merge_seconds",
            ):
                assert res_multi.metrics[name] >= 0.0
                assert name not in res_single.metrics
            assert (
                res_multi.metrics["performance/primary_compute_seconds"]
                <= res_multi.metrics["performance/step_seconds"]
            )

            for metric_key in (
                "train/loss",
                "optimizer/grad_norm",
                "engram/lookup_count",
                "engram/unique_addresses",
                "engram/table_utilization",
            ):
                assert res_multi.metrics[metric_key] == pytest.approx(
                    res_single.metrics[metric_key], rel=1e-6, abs=1e-6
                ), f"step {step} metric {metric_key} mismatch"

            for (name_s, param_s), (name_m, param_m) in zip(
                engine_single.model.named_parameters(),
                engine_multi.model.named_parameters(),
                strict=True,
            ):
                assert name_s == name_m
                assert torch.allclose(param_s, param_m, atol=1e-6, rtol=1e-5), (
                    f"step {step} parameter {name_s} mismatch"
                )
                assert param_s.grad is not None and param_m.grad is not None
                assert torch.allclose(
                    param_s.grad, param_m.grad, atol=1e-6, rtol=1e-5
                ), f"step {step} grad {name_s} mismatch"

            slots_single = _named_optimizer_slots(engine_single)
            slots_multi = _named_optimizer_slots(engine_multi)
            assert slots_single.keys() == slots_multi.keys()
            for name in slots_single:
                for state_key in ("exp_avg", "exp_avg_sq", "step"):
                    assert torch.allclose(
                        slots_single[name][state_key],
                        slots_multi[name][state_key],
                        atol=1e-6,
                        rtol=1e-5,
                    ), f"step {step} optimizer {name}.{state_key} mismatch"

        state_single = engine_single.export_training_state()
        state_multi = engine_multi.export_training_state()
        assert (
            state_single.optimizer_parameter_names
            == state_multi.optimizer_parameter_names
        )
        assert torch.equal(
            state_single.rng["torch"],  # type: ignore[arg-type]
            state_multi.rng["torch"],  # type: ignore[arg-type]
        )

        restored_multi = PyTorchEngine()
        try:
            restored_multi.runtime = validated_runtime
            _seed_all(999)
            restored_multi.initialize(
                cfg_multi,
                initial_weights={
                    k: v.clone() for k, v in engine_single.model.state_dict().items()
                },
            )
            restored_multi.restore_training_state(copy.deepcopy(state_single))

            microbatches_3, valid_targets_3 = _make_microbatches(103)
            res_single_3 = engine_single.train_update(
                microbatches_3, 3, valid_targets_3
            )
            res_restored_3 = restored_multi.train_update(
                microbatches_3, 3, valid_targets_3
            )
            assert res_single_3.metrics is not None
            assert res_restored_3.metrics is not None
            assert res_restored_3.metrics["train/loss"] == pytest.approx(
                res_single_3.metrics["train/loss"], rel=1e-6, abs=1e-6
            )
            assert res_restored_3.metrics["optimizer/grad_norm"] == pytest.approx(
                res_single_3.metrics["optimizer/grad_norm"], rel=1e-6, abs=1e-6
            )
            assert restored_multi.model is not None
            for (name_s, param_s), (name_r, param_r) in zip(
                engine_single.model.named_parameters(),
                restored_multi.model.named_parameters(),
                strict=True,
            ):
                assert name_s == name_r
                assert torch.allclose(param_s, param_r, atol=1e-6, rtol=1e-5)
        finally:
            restored_multi.close()
    finally:
        engine_single.close()
        engine_multi.close()


def test_worker_config_lease_acquires_primary_and_secondary_devices(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lease_dir = tmp_path / "leases"
    cfg = _make_run_config(tmp_path / "worker", secondary_device_indices=(1,))
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "backend": "xpu",
                    "device_index": 0,
                    "secondary_device_indices": (1,),
                }
            )
        }
    )
    definition = {
        "worker_id": "worker-b70-dual",
        "engine": "pytorch",
        "backend": "xpu",
        "device_index": 0,
    }
    passive_gpu0 = SimpleNamespace(
        engine="pytorch",
        backend="xpu",
        device_index=0,
        physical_device_id="phys-b70-e300",
    )
    passive_gpu1 = SimpleNamespace(
        engine="pytorch",
        backend="xpu",
        device_index=1,
        physical_device_id="phys-b70-2300",
    )
    monkeypatch.setattr(
        "klab.workers.execution.discover_runtimes",
        lambda: [passive_gpu0, passive_gpu1],
    )

    lease = _acquire_config_lease(definition, passive_gpu0, cfg, state_dir=lease_dir)
    assert isinstance(lease, MultiDeviceLease)
    try:
        assert [item.key for item in lease.leases] == [
            "physical-phys-b70-e300",
            "physical-phys-b70-2300",
        ]
        # While MultiDeviceLease is held, neither GPU 0 nor GPU 1 can be leased by another caller.
        l0 = acquire_lease(
            worker_id="other-0",
            backend="xpu",
            physical_device_id="phys-b70-e300",
            state_dir=lease_dir,
        )
        l1 = acquire_lease(
            worker_id="other-1",
            backend="xpu",
            physical_device_id="phys-b70-2300",
            state_dir=lease_dir,
        )
        assert l0 is None and l1 is None
    finally:
        lease.close()

    # Missing secondary device index in passive discovery raises ValueError.
    monkeypatch.setattr(
        "klab.workers.execution.discover_runtimes",
        lambda: [passive_gpu0],
    )
    with pytest.raises(ValueError, match="secondary_device_indices index 1 is absent"):
        _acquire_config_lease(definition, passive_gpu0, cfg, state_dir=lease_dir)

    # Blank physical_device_id on secondary (or primary) raises ValueError instead of dropping physical leases.
    passive_gpu1_blank = SimpleNamespace(
        engine="pytorch",
        backend="xpu",
        device_index=1,
        physical_device_id="",
    )
    monkeypatch.setattr(
        "klab.workers.execution.discover_runtimes",
        lambda: [passive_gpu0, passive_gpu1_blank],
    )
    with pytest.raises(
        ValueError,
        match="secondary_device_indices index 1 lacks a reported physical_device_id",
    ):
        _acquire_config_lease(definition, passive_gpu0, cfg, state_dir=lease_dir)

    passive_gpu0_blank = SimpleNamespace(
        engine="pytorch",
        backend="xpu",
        device_index=0,
        physical_device_id=None,
    )
    with pytest.raises(
        ValueError,
        match="secondary_device_indices requires a reported physical_device_id on the primary device",
    ):
        _acquire_config_lease(definition, passive_gpu0_blank, cfg, state_dir=lease_dir)


def test_discover_runtimes_enumerates_multi_xpu_indices_and_physical_uuids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from klab import runtime

    uuid0_bytes = bytes.fromhex("868023e200000000e300000000000000")
    uuid1_bytes = bytes.fromhex("868023e2000000002300000000000000")
    props_by_index = {
        0: SimpleNamespace(
            name="Intel(R) Arc(TM) Pro B70 Graphics",
            total_memory=32 * 1024**3,
            uuid=uuid0_bytes,
        ),
        1: SimpleNamespace(
            name="Intel(R) Arc(TM) Pro B70 Graphics",
            total_memory=32 * 1024**3,
            uuid=uuid1_bytes,
        ),
    }
    fake_xpu = SimpleNamespace(
        is_available=lambda: True,
        device_count=lambda: 2,
        get_device_name=lambda i=0: props_by_index[i].name,
        get_device_properties=lambda i=0: props_by_index[i],
        mem_get_info=lambda i=0: (30 * 1024**3, 32 * 1024**3),
    )
    monkeypatch.setattr(torch, "xpu", fake_xpu)
    monkeypatch.setattr(runtime, "_backend_available", lambda b: b in {"cpu", "xpu"})

    infos = [info for info in runtime.discover_runtimes() if info.backend == "xpu"]
    assert [
        (i.device_index, i.torch_device, i.device_name, i.physical_device_id)
        for i in infos
    ] == [
        (
            0,
            "xpu:0",
            "Intel(R) Arc(TM) Pro B70 Graphics",
            "868023e2-0000-0000-e300-000000000000",
        ),
        (
            1,
            "xpu:1",
            "Intel(R) Arc(TM) Pro B70 Graphics",
            "868023e2-0000-0000-2300-000000000000",
        ),
    ]
    assert "device memory deferred to isolated validation" in infos[0].limitations

    measured1 = next(
        info
        for info in runtime.discover_runtimes(measurement_device=torch.device("xpu:1"))
        if info.backend == "xpu"
    )
    assert measured1.device_index == 1
    assert measured1.physical_device_id == "868023e2-0000-0000-2300-000000000000"
    assert (
        "physical device identity unavailable through safe PyTorch APIs"
        not in measured1.limitations
    )

    lease_dir = tmp_path / "leases"
    cfg = _make_run_config(tmp_path / "worker", secondary_device_indices=(1,))
    cfg = cfg.model_copy(
        update={
            "runtime": cfg.runtime.model_copy(
                update={
                    "backend": "xpu",
                    "device_index": 0,
                    "secondary_device_indices": (1,),
                }
            )
        }
    )
    definition = {
        "worker_id": "worker-b70-dual-real-discovery",
        "engine": "pytorch",
        "backend": "xpu",
        "device_index": 0,
    }
    lease = _acquire_config_lease(definition, infos[0], cfg, state_dir=lease_dir)
    assert isinstance(lease, MultiDeviceLease)
    try:
        assert [item.key for item in lease.leases] == [
            "physical-868023e2-0000-0000-e300-000000000000",
            "physical-868023e2-0000-0000-2300-000000000000",
        ]
    finally:
        lease.close()

    from klab.staging import inspect_runtime

    cfg_idx1 = cfg.model_copy(
        update={"runtime": cfg.runtime.model_copy(update={"device_index": 1})}
    )
    inspected1 = inspect_runtime(cfg_idx1)
    assert inspected1.device_index == 1
    assert inspected1.torch_device == "xpu:1"
    assert inspected1.physical_device_id == "868023e2-0000-0000-2300-000000000000"
    assert (
        "passive memory readings for this device index are unavailable"
        not in inspected1.limitations
    )


def test_gradient_reduction_applies_replica_pushes_in_replica_order() -> None:
    import threading

    from klab.engines.multi_device import GradientReduction, SecondaryReplica

    def replica(scale: float) -> SecondaryReplica:
        model = torch.nn.Linear(3, 2)
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, scale)
        return SecondaryReplica(device=torch.device("cpu"), model=model)

    primary = torch.nn.Linear(3, 2)
    primary.weight.grad = torch.ones_like(primary.weight)
    parameters = list(primary.parameters())
    reduction = GradientReduction(parameters, torch.device("cpu"), pushes=2)
    first, second = replica(2.0), replica(4.0)

    # The later replica finishes first but must wait for its turn.
    late = threading.Thread(target=reduction.push, args=(1, second))
    late.start()
    late.join(timeout=0.2)
    assert late.is_alive()
    assert all(collected is None for collected in reduction.collected)
    reduction.push(0, first)
    late.join(timeout=10)
    assert not late.is_alive()

    reduction.merge()
    # Weight had its own gradient; bias had none and takes the collected sum.
    assert torch.equal(primary.weight.grad, torch.full_like(primary.weight, 7.0))
    assert torch.equal(primary.bias.grad, torch.full_like(primary.bias, 6.0))
    assert first.model.weight.grad is None and second.model.weight.grad is None


def test_failed_replica_slice_surfaces_without_blocking_other_replicas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from concurrent.futures import ThreadPoolExecutor

    from klab.engines import multi_device

    replicas = [
        multi_device.SecondaryReplica(
            device=torch.device("cpu"), model=torch.nn.Linear(2, 2)
        )
        for _ in range(2)
    ]
    primary = torch.nn.Linear(2, 2)

    def failing(replica, config, pairs, valid_targets, loss, **options):
        if replica is replicas[0]:
            raise RuntimeError("replica slice failed")
        options["reduction"].push(options["turn"], replica)
        return {"compute_seconds": 0.0, "finished_at": 0.0}

    monkeypatch.setattr(multi_device, "execute_replica_microbatches", failing)
    with (
        ThreadPoolExecutor(max_workers=2) as executor,
        pytest.raises(RuntimeError, match="replica slice failed"),
    ):
        multi_device.dispatch_replicas_and_primary(
            replicas=replicas,
            config=None,
            active_pairs=[("a", 1), ("b", 1), ("c", 1)],
            valid_targets=3,
            output_head_loss_fn=None,
            primary_fn=lambda pairs: None,
            primary_params=list(primary.parameters()),
            primary_device=torch.device("cpu"),
            executor=executor,
        )
