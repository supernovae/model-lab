# Invalid persisted audit state remains a ValueError/RuntimeError contract.
"""PyTorch implementation of the engine-neutral training boundary."""

from __future__ import annotations

import random
import re
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext

import numpy as np
import torch
from torch.nn import functional

from klab.config.models import AdamWConfig, RunConfig
from klab.engines.base import (
    CanonicalTensor,
    EngineNonFiniteError,
    EngineOutOfMemory,
    EngineState,
    EvaluationResult,
    Microbatch,
    UpdateResult,
    WeightSource,
)
from klab.engines.multi_device import (
    SecondaryReplica,
    accumulate_microbatch_diagnostics,
    diagnostic_is_count,
    dispatch_replicas_and_primary,
    microbatch_position_mask,
)
from klab.engines.output_head import chunked_output_head_cross_entropy
from klab.memory import DeviceMemorySpill, MemoryMonitor, SpillLimits
from klab.model.inspection import architecture_metrics
from klab.model.transformer import DenseLM
from klab.runtime import (
    make_grad_scaler,
    precision_context,
    synchronize,
    torch_device_for,
    validate_runtime,
)
from klab.runtime_profile import RuntimeAuthorization, require_authorization
from klab.training.objectives import Batch as ObjectiveBatch
from klab.training.objectives import Frozen, TrainingObjective, load_frozen_model
from klab.training.offload import ActivationOffload
from klab.training.optimizer import (
    OptimizerStateOffload,
    learning_rate_for_step,
    make_adafactor,
    make_optimizer,
    set_optimizer_learning_rate,
)
from klab.training.pilot_progress import current_pilot_progress, pilot_phase
from klab.training.trainable import apply_trainable_parameter_filter


def _rng_state(device: torch.device, backend: str) -> dict[str, object]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    if backend in {"cuda", "rocm"}:
        device_rng = torch.cuda.get_rng_state(device)
    elif backend == "mps":
        device_rng = torch.mps.get_rng_state()
    elif backend == "xpu":
        device_rng = torch.xpu.get_rng_state(device)
    else:
        device_rng = None
    return {
        "python": python_state,
        "numpy_kind": numpy_state[0],
        "numpy_keys": torch.from_numpy(numpy_state[1].copy()),
        "numpy_pos": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch": torch.get_rng_state(),
        "device_type": backend,
        "device_index": device.index or 0,
        "device_rng": device_rng,
    }


def _restore_rng(state: Mapping[str, object], device: torch.device) -> None:
    random.setstate(state["python"])  # type: ignore[arg-type]
    keys = state["numpy_keys"]
    np.random.set_state(
        (
            str(state["numpy_kind"]),
            keys.numpy(),  # type: ignore[union-attr]
            int(state["numpy_pos"]),
            int(state["numpy_has_gauss"]),
            float(state["numpy_cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch"])  # type: ignore[arg-type]
    backend = state.get("device_type", "cpu")
    if backend in {"cuda", "rocm"}:
        torch.cuda.set_rng_state(state["device_rng"], device)  # type: ignore[arg-type]
    elif backend == "mps":
        torch.mps.set_rng_state(state["device_rng"])  # type: ignore[arg-type]
    elif backend == "xpu":
        torch.xpu.set_rng_state(state["device_rng"], device)  # type: ignore[arg-type]


def _is_allocation_failure(error: BaseException) -> bool:
    """Recognize only PyTorch/backend allocation failures, preserving other errors."""
    if isinstance(error, (torch.OutOfMemoryError, MemoryError)):
        return True
    return isinstance(error, RuntimeError) and bool(
        re.search(
            r"\b(out of memory|cannot allocate memory|can't allocate memory|"
            r"allocation (?:failed|failure))\b",
            str(error),
            re.IGNORECASE,
        )
    )


@contextmanager
def _translate_allocation_failures(
    spill_limits: SpillLimits | None = None,
) -> Iterator[None]:
    try:
        yield
    except EngineOutOfMemory:
        raise
    except (RuntimeError, MemoryError) as error:
        if not _is_allocation_failure(error):
            raise
        if spill_limits is not None:
            # The allocator itself is capped at the spill guard's device limit.
            raise DeviceMemorySpill(
                "device memory spill: an allocation would exceed "
                f"{spill_limits.device_fraction:g} of device memory "
                f"(runtime.memory.max_device_memory_fraction): {error}"
            ) from error
        raise EngineOutOfMemory(f"PyTorch allocation failed: {error}") from error


def _apply_recomputation_strategy(model: DenseLM, config: RunConfig) -> None:
    """Recompute attention chunks inside blocks where the strategy asks for it."""
    recomputation = config.runtime.memory.activation_checkpointing
    chunks = (
        recomputation.enabled
        and recomputation.strategy == "transformer_block_and_attention"
    )
    for module in model.modules():
        if hasattr(module, "recompute_chunks"):
            module.recompute_chunks = chunks


def _cap_allocator(device: torch.device, spill_limits: SpillLimits | None) -> bool:
    """Make the caching allocator refuse growth past the spill guard's limit."""
    api = getattr(torch, device.type, None)
    cap = getattr(api, "set_per_process_memory_fraction", None)
    if spill_limits is None or device.type not in {"cuda", "xpu"} or cap is None:
        return False
    cap(spill_limits.device_fraction, device)
    return True


def _named_optimizer_state(
    model: DenseLM,
    optimizer: torch.optim.Optimizer,
    saved_state: dict[str, object],
    saved_parameter_names: dict[int, str] | list[list[str]],
) -> dict[str, object]:
    """Validate and remap saved slots by canonical parameter name before loading.

    ``Optimizer.load_state_dict`` otherwise binds state positionally.  Reordered
    registration is permissible under recorded runtime drift, but equal-shaped
    AdamW/Adafactor slots must never silently follow that positional order.
    """
    if not isinstance(saved_parameter_names, dict):
        raise TypeError("PyTorch optimizer parameter names must be an ID-to-name map")
    saved_names: dict[int, str] = {}
    for saved_id, name in saved_parameter_names.items():
        if isinstance(saved_id, bool) or not isinstance(saved_id, (int, str)):
            raise TypeError("saved optimizer parameter ID is invalid")
        try:
            normalized_id = int(saved_id)
        except ValueError as error:
            raise ValueError("saved optimizer parameter ID is invalid") from error
        if normalized_id in saved_names or not isinstance(name, str) or not name:
            raise ValueError("saved optimizer parameter names are invalid")
        saved_names[normalized_id] = name

    if len(set(saved_names.values())) != len(saved_names):
        raise ValueError("saved optimizer parameter names are not unique")
    saved_groups = saved_state.get("param_groups")
    saved_slots = saved_state.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_slots, dict):
        raise TypeError("saved optimizer state is malformed")
    live_state = optimizer.state_dict()
    live_groups = live_state.get("param_groups")
    if not isinstance(live_groups, list) or len(saved_groups) != len(live_groups):
        raise ValueError("saved optimizer parameter group count differs")

    parameter_names = {
        id(parameter): name for name, parameter in model.named_parameters()
    }
    live_ids_to_names: dict[int, str] = {}
    live_names_to_ids: dict[str, int] = {}
    live_names_to_parameters: dict[str, torch.nn.Parameter] = {}
    for serialized_group, live_group in zip(
        live_groups, optimizer.param_groups, strict=True
    ):
        serialized_ids = serialized_group.get("params")
        parameters = live_group.get("params")
        if not isinstance(serialized_ids, list) or not isinstance(parameters, list):
            raise TypeError("live optimizer parameter group is malformed")
        if len(serialized_ids) != len(parameters):
            raise ValueError("live optimizer parameter group is inconsistent")
        for serialized_id, parameter in zip(serialized_ids, parameters, strict=True):
            name = parameter_names.get(id(parameter))
            if name is None:
                raise ValueError("optimizer includes an unnamed parameter")
            live_ids_to_names[int(serialized_id)] = name
            live_names_to_ids[name] = int(serialized_id)
            live_names_to_parameters[name] = parameter

    if set(saved_names.values()) != set(live_ids_to_names.values()):
        raise ValueError("saved optimizer parameter names differ from the live model")
    saved_ids_in_groups: set[int] = set()
    saved_group_names: list[set[str]] = []
    for group in saved_groups:
        if not isinstance(group, dict) or not isinstance(group.get("params"), list):
            raise TypeError("saved optimizer parameter group is malformed")
        group_ids = group["params"]
        if any(not isinstance(value, int) for value in group_ids):
            raise TypeError("saved optimizer parameter group IDs are invalid")
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("saved optimizer parameter group repeats a parameter")
        saved_ids_in_groups.update(group_ids)
        try:
            saved_group_names.append({saved_names[value] for value in group_ids})
        except KeyError as error:
            raise ValueError("saved optimizer group lacks a parameter name") from error
    if saved_ids_in_groups != set(saved_names):
        raise ValueError("saved optimizer names and groups disagree")

    unused_live_groups = set(range(len(live_groups)))
    saved_to_live_group: list[int] = []
    for names in saved_group_names:
        matches = [
            index
            for index in unused_live_groups
            if {live_ids_to_names[int(value)] for value in live_groups[index]["params"]}
            == names
        ]
        if len(matches) != 1:
            raise ValueError("saved optimizer parameter group membership differs")
        match = matches[0]
        unused_live_groups.remove(match)
        saved_to_live_group.append(match)

    remapped_slots: dict[int, object] = {}
    for saved_id, slots in saved_slots.items():
        if not isinstance(saved_id, int) or saved_id not in saved_names:
            raise ValueError("saved optimizer state has an unknown parameter")
        if not isinstance(slots, dict):
            raise TypeError("saved optimizer parameter state is malformed")
        name = saved_names[saved_id]
        parameter = live_names_to_parameters[name]
        factor_shapes = {}
        if parameter.ndim >= 2:
            factor_shapes = {
                "row_var": (*parameter.shape[:-1], 1),
                "col_var": (*parameter.shape[:-2], 1, parameter.shape[-1]),
            }
        for slot_name, value in slots.items():
            if not isinstance(slot_name, str):
                raise TypeError("saved optimizer state key is invalid")
            if not isinstance(value, torch.Tensor) or value.ndim == 0:
                continue
            shape = tuple(value.shape)
            if shape == tuple(parameter.shape):
                continue
            if factor_shapes.get(slot_name) == shape:
                continue
            raise ValueError(
                f"optimizer state {slot_name!r} shape {shape} does not match {name!r}"
            )
        live_id = live_names_to_ids[name]
        remapped_slots[live_id] = slots

    remapped_groups: list[dict[str, object]] = []
    for live_group_index in range(len(live_groups)):
        saved_group_index = saved_to_live_group.index(live_group_index)
        remapped_group = dict(saved_groups[saved_group_index])
        remapped_group["params"] = list(live_groups[live_group_index]["params"])
        remapped_groups.append(remapped_group)
    return {"state": remapped_slots, "param_groups": remapped_groups}


class _PyTorchWeights(WeightSource):
    def __init__(self, model: DenseLM) -> None:
        self._model = model
        self._trainable = {
            name: parameter.requires_grad
            for name, parameter in model.named_parameters(remove_duplicate=False)
        }
        aliases: dict[str, str] = {}
        seen: dict[tuple[int, int, tuple[int, ...], torch.dtype], str] = {}
        for name, tensor in model.state_dict().items():
            key = (
                tensor.data_ptr(),
                tensor.storage_offset(),
                tuple(tensor.shape),
                tensor.dtype,
            )
            if key in seen:
                aliases[name] = seen[key]
            else:
                seen[key] = name
        self._aliases = aliases

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._aliases

    def tensors(self) -> Iterator[CanonicalTensor]:
        for name, tensor in self._model.state_dict().items():
            if name in self._aliases:
                continue
            array = tensor.detach().cpu().contiguous().numpy()
            yield CanonicalTensor(name, array, self._trainable.get(name, False))

    def detached(self) -> WeightSource:
        """Copy every tensor to host memory now, so training can continue."""
        return _DetachedWeights(
            self._aliases,
            [
                CanonicalTensor(
                    name,
                    tensor.detach().to("cpu", copy=True).contiguous().numpy(),
                    self._trainable.get(name, False),
                )
                for name, tensor in self._model.state_dict().items()
                if name not in self._aliases
            ],
        )


class _DetachedWeights(WeightSource):
    """A host-memory copy of the weights, independent of the live model."""

    def __init__(
        self, aliases: Mapping[str, str], tensors: list[CanonicalTensor]
    ) -> None:
        self._aliases = aliases
        self._tensors = tensors

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._aliases

    def tensors(self) -> Iterator[CanonicalTensor]:
        # Release each copy once it is written.
        while self._tensors:
            yield self._tensors.pop(0)


def output_projection_cross_entropy(
    hidden: torch.Tensor, targets: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor:
    """Project in BF16 and accumulate cross-entropy in FP32, as in training."""
    with torch.autocast(device_type=hidden.device.type, dtype=torch.bfloat16):
        logits = functional.linear(hidden, weight)
    return functional.cross_entropy(
        logits.float().flatten(0, 1),
        targets.flatten(),
        ignore_index=-100,
        reduction="sum",
    )


class PyTorchEngine:
    """The former trainer compute path, without lifecycle or checkpoint ownership."""

    def __init__(self) -> None:
        self.config: RunConfig | None = None
        self.runtime = None
        self.device: torch.device | None = None
        self.model: DenseLM | None = None
        self.optimizer: torch.optim.Optimizer | None = None
        self.scaler: torch.amp.GradScaler | None = None
        self.monitor: MemoryMonitor | None = None
        self.offload: ActivationOffload | None = None
        self.state_offload: OptimizerStateOffload | None = None
        self._allocator_cap: SpillLimits | None = None
        self.objective: TrainingObjective | None = None
        self.frozen = Frozen()
        self.objective_identity: dict[str, object] | None = None
        self._last_gradient_parameter_names: tuple[str, ...] = ()
        self._last_update_parameter_names: tuple[str, ...] = ()
        self._secondary_replicas: list[SecondaryReplica] = []
        self._replica_executor: ThreadPoolExecutor | None = None

    def validate(
        self, config: RunConfig, *, authorization: RuntimeAuthorization | None = None
    ):
        self.runtime = validate_runtime(config, authorization=authorization)
        self.authorization = authorization
        return self.runtime

    def initialize(
        self,
        config: RunConfig,
        initial_weights: Mapping[str, object] | None = None,
    ) -> None:
        require_authorization(config, getattr(self, "authorization", None))
        runtime = self.runtime if self.runtime is not None else self.validate(config)
        device = torch_device_for(runtime.backend, config.runtime.device_index)
        with pilot_phase("model_initialization"):
            model = DenseLM(config.model, config.attention).to(device)
        if initial_weights is not None:
            model.load_state_dict(initial_weights)  # type: ignore[arg-type]
        trainable_parameters = apply_trainable_parameter_filter(model, config)
        _apply_recomputation_strategy(model, config)
        with pilot_phase("optimizer_initialization"):
            optimizer = (
                make_optimizer(
                    model,
                    config.optimizer.peak,
                    config.optimizer.weight_decay,
                    config.optimizer.betas,
                    config.optimizer.eps,
                    memory_table_lr_multiplier=config.optimizer.memory_table_lr_multiplier,
                    memory_table_weight_decay=config.optimizer.memory_table_weight_decay,
                )
                if config.optimizer.name == "adamw"
                else make_adafactor(
                    model,
                    config.optimizer.peak,
                    config.optimizer.weight_decay,
                    config.optimizer.beta2_decay,
                    config.optimizer.eps,
                    config.optimizer.d,
                )
            )
        selected_ids = {id(parameter) for _, parameter in trainable_parameters}
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != selected_ids:
            raise RuntimeError(
                "optimizer parameters differ from the explicit trainable inventory"
            )
        self.state_offload = (
            OptimizerStateOffload(optimizer, device)
            if config.optimizer.state_offload
            else None
        )
        self.config, self.runtime, self.device = config, runtime, device
        self.model, self.optimizer = model, optimizer
        self._initialize_objective(config, device, runtime.backend)
        if config.runtime.compile_output_head_loss:
            self._compiled_output_head_loss = torch.compile(
                output_projection_cross_entropy,
                dynamic=config.runtime.compile_dynamic_shapes,
                fullgraph=config.runtime.compile_dynamic_shapes,
            )
        if config.runtime.compile_transformer_blocks:
            if config.runtime.compile_dynamic_shapes:
                from klab.model.rope import RoPE

                for module in model.modules():
                    if isinstance(module, RoPE):
                        module.reserve(config.model.max_seq_len, device)
            for block in model.blocks:
                block.forward_with_aux = torch.compile(
                    block.forward_with_aux,
                    dynamic=config.runtime.compile_dynamic_shapes,
                    fullgraph=config.runtime.compile_dynamic_shapes,
                )
        self._last_gradient_parameter_names = ()
        self._last_update_parameter_names = ()
        self.scaler = make_grad_scaler(config, device)
        self.monitor = MemoryMonitor(
            device, runtime_info=runtime, spill_limits=SpillLimits.from_config(config)
        )
        self._allocator_cap = None
        if self.monitor.spill_limits is not None:
            # A spilling update slows inside one forward; check between blocks.
            model.before_block = self.monitor.check_spill
            if _cap_allocator(device, self.monitor.spill_limits):
                self._allocator_cap = self.monitor.spill_limits
        self.offload = (
            ActivationOffload(device, model)
            if config.runtime.memory.activation_offload.enabled
            else None
        )
        if self._replica_executor is not None:
            self._replica_executor.shutdown(wait=False)
            self._replica_executor = None
        self._secondary_replicas = []
        for replica in self._secondary_replicas:
            replica.stale = True
        if config.runtime.secondary_device_indices:
            saved_replica_rng = _rng_state(device, runtime.backend)
            try:
                primary_state = model.state_dict()
                for sec_idx in config.runtime.secondary_device_indices:
                    sec_device = torch_device_for(runtime.backend, sec_idx)
                    replica_model = DenseLM(config.model, config.attention).to(
                        sec_device
                    )
                    replica_model.load_state_dict(primary_state)
                    apply_trainable_parameter_filter(replica_model, config)
                    _apply_recomputation_strategy(replica_model, config)
                    compiled_loss = None
                    if config.runtime.compile_output_head_loss:
                        compiled_loss = torch.compile(
                            output_projection_cross_entropy,
                            dynamic=config.runtime.compile_dynamic_shapes,
                            fullgraph=config.runtime.compile_dynamic_shapes,
                        )
                    if config.runtime.compile_transformer_blocks:
                        if config.runtime.compile_dynamic_shapes:
                            for module in replica_model.modules():
                                if isinstance(module, RoPE):
                                    module.reserve(config.model.max_seq_len, sec_device)
                        for block in replica_model.blocks:
                            block.forward_with_aux = torch.compile(
                                block.forward_with_aux,
                                dynamic=config.runtime.compile_dynamic_shapes,
                                fullgraph=config.runtime.compile_dynamic_shapes,
                            )
                    self._secondary_replicas.append(
                        SecondaryReplica(
                            device=sec_device,
                            model=replica_model,
                            compiled_output_head_loss=compiled_loss,
                        )
                    )
                self._replica_executor = ThreadPoolExecutor(
                    max_workers=len(self._secondary_replicas),
                    thread_name_prefix="klab-replica",
                )
            finally:
                _restore_rng(saved_replica_rng, device)

    def _initialize_objective(
        self, config: RunConfig, device: torch.device, backend: str
    ) -> None:
        """Build the run's objective and load the frozen models it compares against.

        Plain next-token training keeps the engine's own loss path, so
        ``self.objective`` stays ``None`` for it.
        """
        from klab.objectives import DEFAULT_KIND, build, objective

        self.objective = None
        self.frozen = Frozen()
        self.objective_identity = None
        settings = config.training.objective
        registered = objective(settings.kind)
        if registered.kind == DEFAULT_KIND:
            return
        identity: dict[str, object] = {
            "kind": registered.kind,
            "source": registered.source,
            "params": dict(settings.params),
        }
        # Constructing a model draws from the RNG; the run's own stream must
        # not depend on whether an objective loads one.
        rng = _rng_state(device, backend)
        try:
            frozen: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {}
            for role, path in (
                (
                    "teacher",
                    settings.teacher_checkpoint if registered.teacher else None,
                ),
                (
                    "reference",
                    config.training.parent_checkpoint if registered.reference else None,
                ),
            ):
                if path is None:
                    continue
                module, identity[role] = load_frozen_model(
                    path, config, device, role=role
                )

                def run(
                    ids: torch.Tensor, module: torch.nn.Module = module
                ) -> torch.Tensor:
                    with torch.no_grad(), precision_context(config, device):
                        return module(ids)

                frozen[role] = run
        finally:
            _restore_rng(rng, device)
        self.frozen = Frozen(**frozen)
        self.objective = build(registered.kind, settings.params)
        self.objective_identity = identity

    def _ready(
        self,
    ) -> tuple[RunConfig, torch.device, DenseLM, torch.optim.Optimizer, MemoryMonitor]:
        if (
            self.config is None
            or self.device is None
            or self.model is None
            or self.optimizer is None
            or self.monitor is None
        ):
            raise RuntimeError("PyTorchEngine is not initialized")
        return self.config, self.device, self.model, self.optimizer, self.monitor

    def train_update(
        self, microbatches: list[Microbatch], update_index: int, valid_targets: int
    ) -> UpdateResult:
        # Strict dynamic compilation must never turn cache exhaustion or a
        # compiler failure into an apparently successful eager training run.
        strict_compile = (
            self.config is not None and self.config.runtime.compile_dynamic_shapes
        )
        with (
            _translate_allocation_failures(self._allocator_cap),
            torch._dynamo.config.patch(
                suppress_errors=False, fail_on_recompile_limit_hit=True
            )
            if strict_compile
            else nullcontext(),
        ):
            return self._train_update(microbatches, update_index, valid_targets)

    def _train_update(
        self, microbatches: list[Microbatch], update_index: int, valid_targets: int
    ) -> UpdateResult:
        config, device, model, optimizer, monitor = self._ready()
        head_chunk = config.runtime.output_head_chunk_tokens
        chunk_counts = [
            int(np.count_nonzero(batch.targets != -100)) for batch in microbatches
        ]
        observed_targets = sum(chunk_counts)
        if observed_targets <= 0:
            raise ValueError("update needs at least one supervised target")
        if valid_targets != observed_targets:
            raise ValueError(
                "update valid-target count differs from the supplied microbatches"
            )
        window_rng = _rng_state(device, config.runtime.backend) if self.scaler else None
        prior_lrs = [group["lr"] for group in optimizer.param_groups]
        lr = learning_rate_for_step(
            update_index,
            config.training.max_steps,
            config.optimizer.warmup_steps,
            config.optimizer.peak,
            config.optimizer.floor,
            stable_steps=config.training.stable_steps,
            decay_steps=(
                config.optimizer.decay_steps
                if isinstance(config.optimizer, AdamWConfig)
                else None
            ),
        )
        synchronize(device)
        started = time.perf_counter()
        monitor.begin_update()
        if self.offload is not None and not self.offload.reset():
            raise RuntimeError("previous update retained offloaded activation storage")
        optimizer.zero_grad(set_to_none=True)
        language_sum = torch.zeros((), device=device)
        aux_sum = torch.zeros((), device=device)
        diagnostic_sums: dict[str, torch.Tensor] = {}
        diagnostic_weights: dict[str, int] = {}
        losses_finite = torch.ones((), device=device, dtype=torch.bool)
        recomputed = 0.0
        executed_microbatches = 0
        pilot_progress = current_pilot_progress()
        active_pairs = [
            (batch, chunk_valid)
            for batch, chunk_valid in zip(microbatches, chunk_counts, strict=True)
            if chunk_valid or batch.position_mask is not None
        ]
        devices_used = 1
        replica_timings: dict[str, float] = {}
        objective_sums: dict[str, torch.Tensor | float] = {}
        update_rows = sum(batch.inputs.shape[0] for batch, _ in active_pairs)
        if (
            self._secondary_replicas
            and len(microbatches) >= 2
            and len(active_pairs) >= 2
            and self.scaler is None
            and self.offload is None
        ):
            devices_used = min(len(active_pairs), 1 + len(self._secondary_replicas))
            trainable_inputs = [
                parameter for parameter in model.parameters() if parameter.requires_grad
            ]

            def _run_primary_slice(
                primary_pairs: list[tuple[Microbatch, int]],
            ) -> tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                float,
                int,
                dict[str, torch.Tensor],
                dict[str, int],
            ]:
                p_lang = torch.zeros((), device=device)
                p_aux = torch.zeros((), device=device)
                p_diag: dict[str, torch.Tensor] = {}
                p_weights: dict[str, int] = {}
                p_finite = torch.ones((), device=device, dtype=torch.bool)
                p_recomp = 0.0
                p_exec = 0
                for batch, chunk_valid in primary_pairs:
                    p_exec += 1
                    x, y = (
                        torch.from_numpy(batch.inputs).to(device),
                        torch.from_numpy(batch.targets).to(device),
                    )
                    addresses = (
                        None
                        if batch.byte_addresses is None
                        else torch.from_numpy(batch.byte_addresses).to(device)
                    )
                    position_mask, position_count = microbatch_position_mask(
                        batch, device, chunk_valid
                    )
                    valid_target_mask = y != -100
                    with precision_context(config, device):
                        logits, auxiliary = model.forward_with_aux(
                            x,
                            byte_addresses=addresses,
                            valid_target_mask=valid_target_mask,
                            activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                            diagnostics=config.logging.architecture_diagnostics,
                            return_hidden=config.runtime.compile_output_head_loss
                            or head_chunk is not None,
                            position_mask=position_mask,
                        )
                    diagnostics = architecture_metrics(model)
                    monitor.sample("forward")
                    if config.runtime.compile_output_head_loss:
                        ce_sum = self._compiled_output_head_loss(
                            logits, y, model.output.weight
                        )
                    elif head_chunk is not None:
                        ce_sum = chunked_output_head_cross_entropy(
                            logits,
                            y,
                            model.output.weight,
                            head_chunk,
                            lambda: precision_context(config, device),
                        )
                    else:
                        ce_sum = functional.cross_entropy(
                            logits.float().flatten(0, 1),
                            y.flatten(),
                            ignore_index=-100,
                            reduction="sum",
                        )
                    auxiliary = auxiliary.float()
                    p_finite = (
                        p_finite & torch.isfinite(ce_sum) & torch.isfinite(auxiliary)
                    )
                    backward_loss = (ce_sum + auxiliary * chunk_valid) / valid_targets
                    torch.autograd.backward(backward_loss, inputs=trainable_inputs)
                    monitor.sample("backward")
                    p_recomp += model.recomputed_block_call_ratio
                    p_lang = p_lang + ce_sum.detach()
                    p_aux = p_aux + auxiliary.detach() * chunk_valid
                    accumulate_microbatch_diagnostics(
                        p_diag,
                        p_weights,
                        diagnostics,
                        batch,
                        chunk_valid,
                        position_count,
                    )
                return p_lang, p_aux, p_finite, p_recomp, p_exec, p_diag, p_weights

            (
                (
                    language_sum,
                    aux_sum,
                    losses_finite,
                    recomputed,
                    executed_microbatches,
                    diagnostic_sums,
                    diagnostic_weights,
                ),
                replica_outs,
                replica_timings,
            ) = dispatch_replicas_and_primary(
                replicas=self._secondary_replicas,
                config=config,
                active_pairs=active_pairs,
                valid_targets=valid_targets,
                output_head_loss_fn=output_projection_cross_entropy,
                primary_fn=_run_primary_slice,
                primary_params=trainable_inputs,
                primary_device=device,
                executor=self._replica_executor,
            )
            active_replicas = [
                replica
                for rep_idx, replica in enumerate(self._secondary_replicas)
                if active_pairs[rep_idx + 1 :: 1 + len(self._secondary_replicas)]
            ]
            for replica, rep_out in zip(active_replicas, replica_outs, strict=True):
                language_sum = language_sum + rep_out["language_sum"].to(device)
                aux_sum = aux_sum + rep_out["aux_sum"].to(device)
                losses_finite = losses_finite & rep_out["losses_finite"].to(device)
                recomputed += float(rep_out["recomputed"])
                executed_microbatches += int(rep_out["executed_microbatches"])
                for name, value in rep_out["diagnostic_sums"].items():
                    val_dev = value.to(device)
                    diagnostic_sums[name] = (
                        diagnostic_sums.get(name, val_dev.new_zeros(())) + val_dev
                    )
                    weight = rep_out["diagnostic_weights"][name]
                    if diagnostic_is_count(name, microbatches):
                        diagnostic_weights[name] = 1
                    else:
                        diagnostic_weights[name] = (
                            diagnostic_weights.get(name, 0) + weight
                        )
            if pilot_progress is not None:
                pilot_progress.emit(
                    "progress",
                    "optimizer_update",
                    current_step=update_index,
                    counter="items",
                    value=executed_microbatches,
                    total=len(microbatches),
                    subject=f"update_{update_index}_backward_dispatches",
                )
        else:
            for batch, chunk_valid in zip(microbatches, chunk_counts, strict=True):
                if not chunk_valid and batch.position_mask is None:
                    continue
                executed_microbatches += 1
                x, y = (
                    torch.from_numpy(batch.inputs).to(device),
                    torch.from_numpy(batch.targets).to(device),
                )
                addresses = (
                    None
                    if batch.byte_addresses is None
                    else torch.from_numpy(batch.byte_addresses).to(device)
                )
                position_mask, position_count = microbatch_position_mask(
                    batch, device, chunk_valid
                )
                valid_target_mask = y != -100
                if self.objective is not None:
                    # The objective owns the forward: it may reshape the batch
                    # or compare against a frozen model.
                    auxiliary_terms: list[torch.Tensor] = []

                    def objective_forward(
                        ids: torch.Tensor,
                        targets: torch.Tensor = y,
                        terms: list[torch.Tensor] = auxiliary_terms,
                    ) -> torch.Tensor:
                        with (
                            self.offload.hooks()
                            if self.offload is not None
                            else nullcontext(),
                            precision_context(config, device),
                        ):
                            output, term = model.forward_with_aux(
                                ids,
                                valid_target_mask=(targets != -100).view(ids.shape)
                                if targets.numel() == ids.numel()
                                else None,
                                activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                                diagnostics=config.logging.architecture_diagnostics,
                            )
                        terms.append(term.float())
                        return output

                    objective_loss = self.objective.loss(
                        objective_forward,
                        ObjectiveBatch(x, y, valid_targets, update_rows),
                        self.frozen,
                    )
                    if not auxiliary_terms:
                        raise RuntimeError(
                            "training objective never ran the model being trained"
                        )
                    ce_sum = objective_loss.total.float()
                    auxiliary = torch.stack(auxiliary_terms).sum()
                    for name, value in objective_loss.metrics.items():
                        objective_sums[name] = (
                            objective_sums.get(name, 0.0)
                            + value.detach().float() * chunk_valid
                        )
                    diagnostics = architecture_metrics(model)
                    monitor.sample("forward")
                else:
                    with (
                        self.offload.hooks()
                        if self.offload is not None
                        else nullcontext(),
                        precision_context(config, device),
                    ):
                        logits, auxiliary = model.forward_with_aux(
                            x,
                            byte_addresses=addresses,
                            valid_target_mask=valid_target_mask,
                            activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                            diagnostics=config.logging.architecture_diagnostics,
                            return_hidden=config.runtime.compile_output_head_loss
                            or head_chunk is not None,
                            position_mask=position_mask,
                        )
                    diagnostics = architecture_metrics(model)
                    monitor.sample("forward")
                    if config.runtime.compile_output_head_loss:
                        ce_sum = self._compiled_output_head_loss(
                            logits, y, model.output.weight
                        )
                    elif head_chunk is not None:
                        ce_sum = chunked_output_head_cross_entropy(
                            logits,
                            y,
                            model.output.weight,
                            head_chunk,
                            lambda: precision_context(config, device),
                        )
                    else:
                        ce_sum = functional.cross_entropy(
                            logits.float().flatten(0, 1),
                            y.flatten(),
                            ignore_index=-100,
                            reduction="sum",
                        )
                auxiliary = auxiliary.float()
                losses_finite = (
                    losses_finite & torch.isfinite(ce_sum) & torch.isfinite(auxiliary)
                )
                backward_loss = (ce_sum + auxiliary * chunk_valid) / valid_targets
                if self.scaler is not None:
                    backward_loss = self.scaler.scale(backward_loss)
                torch.autograd.backward(
                    backward_loss,
                    inputs=[
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                )
                monitor.sample("backward")
                if pilot_progress is not None:
                    # Dispatch completion is not GPU synchronization or a committed update.
                    pilot_progress.emit(
                        "progress",
                        "optimizer_update",
                        current_step=update_index,
                        counter="items",
                        value=executed_microbatches,
                        total=len(microbatches),
                        subject=f"update_{update_index}_backward_dispatches",
                    )
                recomputed += model.recomputed_block_call_ratio
                language_sum = language_sum + ce_sum.detach()
                aux_sum = aux_sum + auxiliary.detach() * chunk_valid
                accumulate_microbatch_diagnostics(
                    diagnostic_sums,
                    diagnostic_weights,
                    diagnostics,
                    batch,
                    chunk_valid,
                    position_count,
                )
        if self.scaler is not None:
            self.scaler.unscale_(optimizer)
        named_parameters = dict(model.named_parameters())
        gradient_names = tuple(
            name
            for name, parameter in named_parameters.items()
            if parameter.grad is not None
        )
        if config.training.trainable_parameters is not None:
            allowed_names = set(config.training.trainable_parameters)
            unexpected_gradients = set(gradient_names) - allowed_names
            if unexpected_gradients:
                optimizer.zero_grad(set_to_none=True)
                raise RuntimeError(
                    "gradients escaped the explicit trainable inventory: "
                    f"{sorted(unexpected_gradients)}"
                )
        self._last_gradient_parameter_names = gradient_names
        norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            config.training.grad_clip_norm,
            error_if_nonfinite=False,
            foreach=False,
        )
        gradients_finite = torch.isfinite(norm) if self.scaler is None else None
        # One pre-step boundary checks the accumulated loss and gradients before
        # optimizer mutation. No diagnostic scalar is read per microbatch.
        synchronize(device)
        if not bool(losses_finite):
            optimizer.zero_grad(set_to_none=True)
            raise EngineNonFiniteError("nonfinite training loss")
        if gradients_finite is not None and not bool(gradients_finite):
            optimizer.zero_grad(set_to_none=True)
            raise EngineNonFiniteError("nonfinite unscaled gradients")
        set_optimizer_learning_rate(optimizer, lr)
        overflow = False
        if self.scaler is None:
            optimizer.step()
        else:
            old_scale = self.scaler.get_scale()
            self.scaler.step(optimizer)
            self.scaler.update()
            overflow = self.scaler.get_scale() < old_scale
        monitor.sample("optimizer")
        if not overflow:
            for replica in self._secondary_replicas:
                replica.stale = True
        synchronize(device)
        memory_metrics = monitor.end_update()
        elapsed = time.perf_counter() - started
        if overflow:
            optimizer.zero_grad(set_to_none=True)
            for group, old_lr in zip(optimizer.param_groups, prior_lrs, strict=True):
                group["lr"] = old_lr
            assert window_rng is not None
            _restore_rng(window_rng, device)
            return UpdateResult("OVERFLOW", None, valid_targets, 0, elapsed)
        self._last_update_parameter_names = gradient_names
        metrics = {
            "train/loss": float(language_sum) / valid_targets,
            "moe/router_auxiliary_loss": float(aux_sum) / valid_targets,
            "optimizer/learning_rate": lr,
            "optimizer/grad_norm": float(norm),
            "performance/step_seconds": elapsed,
            "performance/tokens_per_second": valid_targets / max(elapsed, 1e-9),
            "batch/micro_batch_size": float(config.training.micro_batch_size),
            "batch/accumulation_steps": float(len(microbatches)),
            **(
                {"batch/devices_used": float(devices_used)}
                if self._secondary_replicas
                else {}
            ),
            **replica_timings,
            "batch/effective_batch_size": float(
                sum(batch.inputs.shape[0] for batch in microbatches)
            ),
            "batch/effective_tokens_per_update": float(valid_targets),
            "batch/compute_slots": float(
                sum(batch.inputs.size for batch in microbatches)
            ),
            "batch/real_positions": float(
                sum(
                    batch.inputs.size
                    if batch.position_mask is None
                    else batch.position_mask.sum()
                    for batch in microbatches
                )
            ),
            "batch/document_diagnostics_version": float(
                2
                if any(batch.position_mask is not None for batch in microbatches)
                else 1
            ),
            "recompute/block_call_ratio": recomputed / executed_microbatches,
            **{
                name: float(value) / diagnostic_weights[name]
                for name, value in diagnostic_sums.items()
            },
            **{
                f"objective/{name}": float(value) / valid_targets
                for name, value in objective_sums.items()
            },
            **memory_metrics,
        }
        if self.scaler is not None:
            metrics["optimizer/loss_scale"] = self.scaler.get_scale()
        if self.offload is not None:
            metrics.update(self.offload.metrics.as_metrics(elapsed))
        return UpdateResult(
            "APPLIED",
            float(language_sum),
            valid_targets,
            valid_targets,
            elapsed,
            metrics,
        )

    def evaluate(self, batches: Iterable[Microbatch]) -> EvaluationResult:
        with _translate_allocation_failures(self._allocator_cap):
            return self._evaluate(batches)

    def _evaluate(self, batches: Iterable[Microbatch]) -> EvaluationResult:
        config, device, model, _, _ = self._ready()
        was_training = model.training
        rng = _rng_state(device, config.runtime.backend)
        total_loss, total_targets, count = 0.0, 0, 0
        try:
            model.eval()
            with torch.no_grad(), precision_context(config, device):
                for batch in batches:
                    x = torch.from_numpy(batch.inputs).to(device)
                    y = torch.from_numpy(batch.targets).to(device)
                    addresses = (
                        None
                        if batch.byte_addresses is None
                        else torch.from_numpy(batch.byte_addresses).to(device)
                    )
                    if self.objective is not None:
                        targets_in_batch = int(np.count_nonzero(batch.targets != -100))
                        total_loss += float(
                            self.objective.evaluation_loss(
                                model,
                                ObjectiveBatch(x, y, targets_in_batch, x.shape[0]),
                                self.frozen,
                            )
                        )
                        total_targets += targets_in_batch
                        count += 1
                        continue
                    if config.runtime.output_head_chunk_tokens is not None:
                        hidden, _ = model.forward_with_aux(
                            x, byte_addresses=addresses, return_hidden=True
                        )
                        total_loss += float(
                            chunked_output_head_cross_entropy(
                                hidden,
                                y,
                                model.output.weight,
                                config.runtime.output_head_chunk_tokens,
                                lambda: precision_context(config, device),
                            )
                        )
                    else:
                        logits = model(x, byte_addresses=addresses)
                        total_loss += float(
                            functional.cross_entropy(
                                logits.float().flatten(0, 1),
                                y.flatten(),
                                ignore_index=-100,
                                reduction="sum",
                            )
                        )
                    total_targets += int(np.count_nonzero(batch.targets != -100))
                    count += 1
        finally:
            model.train(was_training)
            _restore_rng(rng, device)
        return EvaluationResult(total_loss, total_targets, count)

    def export_weights(self) -> WeightSource:
        _, _, model, _, _ = self._ready()
        return _PyTorchWeights(model)

    def export_training_state(self) -> EngineState:
        _, device, model, optimizer, _ = self._ready()
        names = {id(parameter): name for name, parameter in model.named_parameters()}
        optimizer_state = optimizer.state_dict()
        parameter_names = {
            stored_id: names[id(parameter)]
            for stored_group, live_group in zip(
                optimizer_state["param_groups"], optimizer.param_groups, strict=True
            )
            for stored_id, parameter in zip(
                stored_group["params"], live_group["params"], strict=True
            )
        }
        return EngineState(
            optimizer_state,
            _rng_state(device, self.config.runtime.backend),
            self.scaler.state_dict() if self.scaler is not None else None,
            parameter_names,  # type: ignore[arg-type]
        )

    def restore_training_state(self, state: EngineState) -> None:
        _, device, model, optimizer, _ = self._ready()
        from klab.training.checkpoints import restore_optimizer_state

        remapped_optimizer_state = _named_optimizer_state(
            model,
            optimizer,
            state.optimizer,
            state.optimizer_parameter_names,
        )
        restore_optimizer_state(optimizer, remapped_optimizer_state)
        if self.state_offload is not None:
            # Loading places state beside each parameter; return it to the host.
            self.state_offload.to_host()
        if self.scaler is not None:
            if state.scaler is None:
                raise ValueError("fp16 resume requires saved GradScaler state")
            self.scaler.load_state_dict(state.scaler)
        _restore_rng(state.rng, device)
        for replica in self._secondary_replicas:
            replica.stale = True

    def memory_snapshot(self) -> dict[str, float]:
        _, _, _, _, monitor = self._ready()
        return {
            key: float(value)
            for key, value in monitor.sample("snapshot").items()
            if key.startswith("memory/") and isinstance(value, (int, float))
        }

    def synchronize(self) -> None:
        _, device, _, _, _ = self._ready()
        synchronize(device)

    @property
    def unavailable_memory_reasons(self) -> dict[str, str]:
        return {} if self.monitor is None else self.monitor.unavailable_reasons

    def close(self) -> None:
        if self._replica_executor is not None:
            self._replica_executor.shutdown(wait=True)
            self._replica_executor = None
        self._secondary_replicas.clear()
        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None
