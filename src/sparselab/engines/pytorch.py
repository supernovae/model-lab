# Invalid persisted audit state remains a ValueError/RuntimeError contract.
# ruff: noqa: TRY004
"""PyTorch implementation of the engine-neutral training boundary."""

from __future__ import annotations

import hashlib
import json
import random
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch.nn import functional

from sparselab.config.models import AdamWConfig, RunConfig
from sparselab.data.allocation import (
    OWNER_HYBRID,
    OWNER_LEXICAL,
    OWNER_NEURAL,
    OWNER_SEMANTIC,
    load_allocation_manifest,
    load_semantic_retriever,
)
from sparselab.data.packing import canonical_tokenizer_sha256
from sparselab.data.tokenizer import load_tokenizer
from sparselab.engines.base import (
    CanonicalTensor,
    EngineNonFiniteError,
    EngineOutOfMemory,
    EngineState,
    EvaluationResult,
    Microbatch,
    UpdateResult,
    WeightSource,
)
from sparselab.engram.semantic import SemanticQueryBatch
from sparselab.memory import MemoryMonitor
from sparselab.model.inspection import architecture_metrics
from sparselab.model.norm import RMSNorm
from sparselab.model.transformer import DenseLM
from sparselab.research.portability import (
    PortabilityRun,
    apply_trainable_parameter_filter,
    initialize_memory_artifact,
    initialize_recipient_backbone,
    load_portability_manifest,
    verify_portability_assets_unchanged,
)
from sparselab.runtime import (
    make_grad_scaler,
    precision_context,
    synchronize,
    torch_device_for,
    validate_runtime,
)
from sparselab.runtime_profile import RuntimeAuthorization, require_authorization
from sparselab.training.manifest import source_identity
from sparselab.training.offload import ActivationOffload
from sparselab.training.optimizer import (
    learning_rate_for_step,
    make_adafactor,
    make_optimizer,
    set_optimizer_learning_rate,
)
from sparselab.training.pilot_progress import current_pilot_progress, pilot_phase


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
def _translate_allocation_failures() -> Iterator[None]:
    try:
        yield
    except (RuntimeError, MemoryError) as error:
        if _is_allocation_failure(error):
            raise EngineOutOfMemory(f"PyTorch allocation failed: {error}") from error
        raise


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


def _parameter_sha256(parameter: torch.nn.Parameter) -> str:
    raw = parameter.detach().to(device="cpu").contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw).cast("B")).hexdigest()


def _initialize_learned_parameters(
    model: DenseLM, run: PortabilityRun, *, interface_only: bool
) -> None:
    identity = run.payload["initialization"]
    coordinate = run.payload["coordinate"]
    pair_seed = int(identity["pair_seed"])
    width = int(identity["width"])
    timing = identity["purpose"] == "timing"
    role = coordinate["role"]
    condition = coordinate["condition"]
    if role == "source":
        role_families = {
            "source-backbone": "source-backbone",
            "source-memory": "source-memory",
        }
    elif role == "preparation":
        role_families = {"preparation-backbone": "preparation-backbone"}
    elif role == "native":
        role_families = {"native-memory": "native-memory"}
    elif role == "recipient":
        role_families = {"recipient-adapter": "recipient-adapter"}
    else:
        role_families = {
            "source-byte-128": {
                "source-backbone": "source-backbone",
                "source-memory": "source-memory",
            },
            "dense-128": {"source-backbone": "source-backbone"},
            "native-64": {"native-memory": "native-memory"},
            "preparation-64": {"preparation-backbone": "preparation-backbone"},
            "adapter-64": {"recipient-adapter": "recipient-adapter"},
            "adapter-128": {"recipient-adapter": "recipient-adapter"},
        }
    initialized = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if role == "native" and not name.startswith("memory."):
            continue
        if interface_only and name not in {
            "memory.output.weight",
            "memory.gate.weight",
        }:
            continue
        if role == "source" or role == "calibration" and condition == "source-byte-128":
            family = (
                "source-memory" if name.startswith("memory.") else "source-backbone"
            )
        elif role == "calibration":
            family = next(iter(role_families[condition].values()))
        else:
            family = next(iter(role_families.values()))
        module_name, _, parameter_name = name.rpartition(".")
        module = model.get_submodule(module_name) if module_name else model
        with torch.no_grad():
            if parameter_name == "weight" and isinstance(module, RMSNorm):
                parameter.fill_(1.0)
            elif parameter_name == "bias":
                parameter.zero_()
            else:
                derivation = (
                    ("timing|" if timing else "")
                    + "learned-engram-portability-v1|init-v1|"
                    + f"{pair_seed}|{family}|{width}|{name}"
                )
                tensor_seed = int.from_bytes(
                    hashlib.sha256(derivation.encode("ascii")).digest()[:8], "big"
                ) % (1 << 63)
                generator = torch.Generator(device="cpu").manual_seed(tensor_seed)
                values = torch.empty(parameter.shape, dtype=torch.float32, device="cpu")
                torch.nn.init.normal_(values, mean=0.0, std=0.02, generator=generator)
                parameter.copy_(
                    values.to(device=parameter.device, dtype=parameter.dtype)
                )
        initialized += 1
    if interface_only and initialized != 2:
        raise ValueError(
            "portable recipient initialization lacks its output/gate interface"
        )


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
        self.semantic_encoder = None
        self.portability_run: PortabilityRun | None = None
        self._portability_parameters: tuple[tuple[str, torch.nn.Parameter], ...] = ()
        self._last_gradient_parameter_names: tuple[str, ...] = ()
        self._last_update_parameter_names: tuple[str, ...] = ()
        self._last_portability_row_gradient_evidence: dict[str, object] | None = None
        self._portability_initial_digests: dict[str, str] = {}
        self._portability_update_history: list[dict[str, object]] = []
        self._learned_fact_targets: dict[int, int] = {}
        self._learned_fact_ids_by_row: dict[int, str] = {}
        self._learned_initial_table_rows: dict[int, bytes] = {}
        self._learned_fact_exposures: dict[int, int] = {}
        self._learned_audit_parameter: torch.nn.Parameter | None = None
        self._learned_address_collision_targets = 0

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
        *,
        resume_portability: bool = False,
    ) -> None:
        require_authorization(config, getattr(self, "authorization", None))
        runtime = self.runtime if self.runtime is not None else self.validate(config)
        portability_run = (
            load_portability_manifest(config)
            if config.training.portability_manifest_path is not None
            else None
        )
        learned_v2 = (
            portability_run is not None
            and type(portability_run.payload.get("version")) is int
            and portability_run.payload["version"] == 2
        )
        if (
            portability_run is not None
            and initial_weights is not None
            and not (learned_v2 and resume_portability)
        ):
            raise ValueError(
                "portability initialization must come from its verified manifest"
            )
        if resume_portability and not (learned_v2 and initial_weights is not None):
            raise ValueError(
                "learned portability resume requires a verified v2 snapshot"
            )
        device = torch_device_for(runtime.backend, config.runtime.device_index)
        initialization_rng = (
            _rng_state(device, runtime.backend)
            if learned_v2 and not resume_portability
            else None
        )
        with pilot_phase("model_initialization"):
            model = DenseLM(config.model, config.attention).to(device)
        self.semantic_encoder = None
        if config.dataset.allocation_manifest_path is not None:
            allocation = load_allocation_manifest(
                config.dataset.allocation_manifest_path,
                source_identity_sha256=cast(str, source_identity()["sha256"]),
                tokenizer_sha256=canonical_tokenizer_sha256(
                    load_tokenizer(config.tokenizer.path)
                ),
            )
            retriever = load_semantic_retriever(allocation)
            if (retriever is None) != (config.model.semantic_memory_dim is None):
                raise ValueError(
                    "model.semantic_memory_dim must match the verified allocation pack"
                )
            if retriever is not None:
                if retriever.memory_dim != config.model.semantic_memory_dim:
                    raise ValueError(
                        "model.semantic_memory_dim differs from semantic pack value width"
                    )
                model.add_semantic_memory(
                    "allocation",
                    retriever,
                    site="final",
                    min_score=3.5 if portability_run is not None else None,
                )
                self.semantic_encoder = retriever.key_encoder
        if learned_v2:
            assert portability_run is not None
            if resume_portability:
                assert initial_weights is not None
                model.load_state_dict(initial_weights)  # type: ignore[arg-type]
                initialize_memory_artifact(model, config, portability_run)
            else:
                initialize_recipient_backbone(model, config, portability_run)
                initialize_memory_artifact(model, config, portability_run)
                role = portability_run.payload["coordinate"]["role"]
                if role == "recipient":
                    if config.model.memory == "portable":
                        _initialize_learned_parameters(
                            model, portability_run, interface_only=True
                        )
                else:
                    _initialize_learned_parameters(
                        model, portability_run, interface_only=False
                    )
        elif portability_run is not None:
            initialize_recipient_backbone(model, config, portability_run)
            initialize_memory_artifact(model, config, portability_run)
        elif initial_weights is not None:
            model.load_state_dict(initial_weights)  # type: ignore[arg-type]
        if initialization_rng is not None:
            _restore_rng(initialization_rng, device)
        trainable_parameters = apply_trainable_parameter_filter(
            model, config, portability_run
        )
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
        self.config, self.runtime, self.device = config, runtime, device
        self.model, self.optimizer = model, optimizer
        if config.runtime.compile_output_head_loss:
            self._compiled_output_head_loss = torch.compile(
                output_projection_cross_entropy,
                dynamic=config.runtime.compile_dynamic_shapes,
                fullgraph=config.runtime.compile_dynamic_shapes,
            )
        if config.runtime.compile_transformer_blocks:
            if config.runtime.compile_dynamic_shapes:
                from sparselab.model.rope import RoPE

                for module in model.modules():
                    if isinstance(module, RoPE):
                        module.reserve(config.model.max_seq_len, device)
            for block in model.blocks:
                block.forward_with_aux = torch.compile(
                    block.forward_with_aux,
                    dynamic=config.runtime.compile_dynamic_shapes,
                    fullgraph=config.runtime.compile_dynamic_shapes,
                )
        self.portability_run = portability_run
        self._portability_parameters = trainable_parameters
        self._last_gradient_parameter_names = ()
        self._last_update_parameter_names = ()
        self._last_portability_row_gradient_evidence = None
        self._portability_update_history = []
        self._portability_initial_digests = (
            {
                name: _parameter_sha256(parameter)
                for name, parameter in model.named_parameters()
            }
            if portability_run is not None
            else {}
        )
        self._initialize_learned_fact_audit(model, config, portability_run)
        self.scaler = make_grad_scaler(config, device)
        self.monitor = MemoryMonitor(device, runtime_info=runtime)
        self.offload = (
            ActivationOffload(device, model)
            if config.runtime.memory.activation_offload.enabled
            else None
        )

    def _initialize_learned_fact_audit(
        self,
        model: DenseLM,
        config: RunConfig,
        run: PortabilityRun | None,
    ) -> None:
        """Bind v2 factual targets to exact initialized local-table bytes."""
        self._learned_fact_targets = {}
        self._learned_fact_ids_by_row = {}
        self._learned_initial_table_rows = {}
        self._learned_fact_exposures = {}
        self._learned_address_collision_targets = 0
        if (
            run is None
            or run.payload.get("version") != 2
            or run.payload["coordinate"]["condition"] not in {"source-real", "native"}
        ):
            return
        descriptor = run.payload["world_manifest"]
        if not isinstance(descriptor, dict) or not isinstance(
            descriptor.get("path"), str
        ):
            raise ValueError("learned portability manifest lacks a world descriptor")
        data_path = run.root / descriptor["path"]
        try:
            data = json.loads(data_path.read_text(encoding="utf-8"))
            facts_descriptor = data["facts"]
            facts_path = data_path.parent / facts_descriptor["path"]
            facts = [
                json.loads(line)
                for line in facts_path.read_text(encoding="utf-8").splitlines()
            ]
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise ValueError(
                "verified learned portability facts cannot be read"
            ) from error
        training_ids = set(run.payload["training_fact_ids"])
        tokenizer = load_tokenizer(config.tokenizer.path)
        for fact in facts:
            if not isinstance(fact, dict) or fact.get("fact_id") not in training_ids:
                continue
            row, symbol, fact_id = (
                fact.get("target_row"),
                fact.get("assigned_symbol"),
                fact.get("fact_id"),
            )
            token_id = (
                tokenizer.token_to_id(symbol) if isinstance(symbol, str) else None
            )
            if (
                type(row) is not int
                or not isinstance(fact_id, str)
                or token_id is None
                or tokenizer.encode(symbol, add_special_tokens=False).ids != [token_id]
                or row in self._learned_fact_targets
            ):
                raise ValueError(
                    "learned portability factual target contract is invalid"
                )
            self._learned_fact_targets[row] = token_id
            self._learned_fact_ids_by_row[row] = fact_id
            self._learned_fact_exposures[row] = 0
        if set(self._learned_fact_ids_by_row.values()) != training_ids:
            raise ValueError("learned portability factual target inventory differs")
        memory = model.memory
        if memory is None or not hasattr(memory, "table"):
            raise TypeError("learned source/native run lacks a local byte table")
        with torch.no_grad():
            for row in self._learned_fact_targets:
                value = memory.table.weight[row].detach().to("cpu").contiguous()
                self._learned_initial_table_rows[row] = bytes(
                    value.view(torch.uint8).numpy()
                )
        self._learned_audit_parameter = memory.table.weight

    def _learned_audit_state(self) -> dict[str, object]:
        if self._learned_audit_parameter is None or self.optimizer is None:
            raise RuntimeError("learned audit state has no local table parameter")
        value = self.optimizer.state[self._learned_audit_parameter].get(
            "sparselab_learned_audit_v1"
        )
        if not isinstance(value, dict):
            raise RuntimeError("learned audit state is absent")
        return cast(dict[str, object], value)

    def _write_learned_audit_state(self) -> None:
        if self._learned_audit_parameter is None or self.optimizer is None:
            return
        rows = sorted(self._learned_fact_targets)
        initial = np.stack(
            [
                np.frombuffer(self._learned_initial_table_rows[row], dtype=np.uint8)
                for row in rows
            ]
        )
        run = self.portability_run
        if run is None:
            raise RuntimeError("learned audit state lacks a portability run")
        coordinate = json.dumps(run.coordinate, sort_keys=True, separators=(",", ":"))
        self.optimizer.state[self._learned_audit_parameter][
            "sparselab_learned_audit_v1"
        ] = {
            "coordinate_sha256": hashlib.sha256(coordinate.encode("utf-8")).hexdigest(),
            "data_manifest_sha256": run.payload["world_manifest"]["sha256"],
            "max_steps": self.config.training.max_steps,
            "rows": torch.tensor(rows, dtype=torch.int64),
            "tokens": torch.tensor(
                [self._learned_fact_targets[row] for row in rows], dtype=torch.int64
            ),
            "initial_rows": torch.from_numpy(initial.copy()),
            "exposures": torch.zeros(len(rows), dtype=torch.int64),
            "gradient_seen": torch.zeros(len(rows), dtype=torch.bool),
            "applied_steps": torch.zeros(
                self.config.training.max_steps, dtype=torch.bool
            ),
            # Keep the v2 resume key stable; this counter records hash-row collisions.
            "invalid_targets": torch.zeros((), dtype=torch.int64),
        }

    def _restore_learned_audit_state(self) -> None:
        if not self._learned_fact_targets:
            return
        if self._learned_audit_parameter is None or self.optimizer is None:
            raise RuntimeError("learned audit state has no local table parameter")
        table_state = self.optimizer.state[self._learned_audit_parameter]
        if "sparselab_learned_audit_v1" not in table_state:
            if any(bool(values) for values in self.optimizer.state.values()):
                raise ValueError(
                    "learned audit state is absent after an optimizer update"
                )
            # A verified step-zero checkpoint has no optimizer state. Its restored
            # model table is the deterministic initial table captured at initialize.
            self._write_learned_audit_state()
        state = self._learned_audit_state()
        required = {
            "coordinate_sha256",
            "data_manifest_sha256",
            "max_steps",
            "rows",
            "tokens",
            "initial_rows",
            "exposures",
            "gradient_seen",
            "applied_steps",
            "invalid_targets",
        }
        if set(state) != required or not all(
            isinstance(state[key], torch.Tensor)
            for key in required
            - {"coordinate_sha256", "data_manifest_sha256", "max_steps"}
        ):
            raise ValueError("learned audit optimizer state is malformed")
        run = self.portability_run
        if run is None:
            raise RuntimeError("learned audit state lacks a portability run")
        coordinate = json.dumps(run.coordinate, sort_keys=True, separators=(",", ":"))
        if (
            state["coordinate_sha256"]
            != hashlib.sha256(coordinate.encode("utf-8")).hexdigest()
            or state["data_manifest_sha256"] != run.payload["world_manifest"]["sha256"]
            or state["max_steps"] != self.config.training.max_steps
        ):
            raise ValueError("learned audit optimizer state is bound to another run")
        rows = cast(torch.Tensor, state["rows"]).cpu().tolist()
        tokens = cast(torch.Tensor, state["tokens"]).cpu().tolist()
        if rows != sorted(self._learned_fact_targets) or tokens != [
            self._learned_fact_targets[row] for row in rows
        ]:
            raise ValueError("learned audit optimizer state factual inventory differs")
        initial = cast(torch.Tensor, state["initial_rows"]).cpu()
        expected_width = len(next(iter(self._learned_initial_table_rows.values())))
        if initial.dtype != torch.uint8 or tuple(initial.shape) != (
            len(rows),
            expected_width,
        ):
            raise ValueError("learned audit optimizer state baseline is malformed")
        self._learned_initial_table_rows = {
            row: bytes(initial[index].contiguous().numpy())
            for index, row in enumerate(rows)
        }
        exposures = cast(torch.Tensor, state["exposures"]).cpu().tolist()
        self._learned_fact_exposures = dict(zip(rows, exposures, strict=True))
        self._learned_address_collision_targets = int(
            cast(torch.Tensor, state["invalid_targets"]).item()
        )

    def _record_applied_factual_exposure(self, microbatches: list[Microbatch]) -> None:
        """Count exact factual targets and track conflicting occupants of their hash rows."""
        pending: dict[int, int] = {}
        collisions = 0
        for batch in microbatches:
            if batch.byte_addresses is None:
                raise ValueError(
                    "learned source/native update lacks byte-address sidecars"
                )
            if batch.byte_addresses.shape != batch.targets.shape:
                raise ValueError("learned source/native byte-address sidecars misalign")
            for row, target in zip(
                batch.byte_addresses.reshape(-1), batch.targets.reshape(-1), strict=True
            ):
                expected = self._learned_fact_targets.get(int(row))
                if expected is None or int(target) == -100:
                    continue
                if int(target) != expected:
                    collisions += 1
                    continue
                pending[int(row)] = pending.get(int(row), 0) + 1
        for row, count in pending.items():
            self._learned_fact_exposures[row] += count
        self._learned_address_collision_targets += collisions
        state = self._learned_audit_state()
        rows = sorted(self._learned_fact_targets)
        exposure_tensor = cast(torch.Tensor, state["exposures"])
        for index, row in enumerate(rows):
            exposure_tensor[index] = self._learned_fact_exposures[row]
        cast(torch.Tensor, state["invalid_targets"]).fill_(
            self._learned_address_collision_targets
        )

    def _record_applied_gradient_rows(self, evidence: dict[str, object]) -> None:
        state = self._learned_audit_state()
        index_by_row = {
            row: index for index, row in enumerate(sorted(self._learned_fact_targets))
        }
        seen = cast(torch.Tensor, state["gradient_seen"])
        for row in evidence["nonzero_row_indices"]:
            if row in index_by_row:
                seen[index_by_row[row]] = True

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

    def portability_audit(self, *, completed: bool = False) -> dict[str, object]:
        if self.portability_run is None or self.model is None:
            raise RuntimeError("no portability run is initialized")
        final_digests = {
            name: _parameter_sha256(parameter)
            for name, parameter in self.model.named_parameters()
        }
        selected = {name for name, _ in self._portability_parameters}
        changed = {
            name
            for name, digest in final_digests.items()
            if self._portability_initial_digests.get(name) != digest
        }
        frozen_changes = sorted(changed - selected)
        if frozen_changes:
            raise RuntimeError(
                f"frozen parameters changed during portability run: {frozen_changes}"
            )
        backbone_names = sorted(
            name
            for name in self._portability_initial_digests
            if not name.startswith(("memory.", "semantic_memories."))
        )
        backbone_unchanged = not (changed & set(backbone_names))
        verify_portability_assets_unchanged(self.portability_run)
        audit: dict[str, object] = {
            "trainable_parameter_names": sorted(selected),
            "gradient_update_history": self._portability_update_history.copy(),
            "updated_parameter_names": sorted(changed),
            "frozen_parameters_unchanged": True,
            "backbone_unchanged": backbone_unchanged,
            "initial_parameter_sha256": self._portability_initial_digests.copy(),
            "final_parameter_sha256": final_digests,
            "assets_unchanged": True,
        }
        if self.portability_run.payload.get("version") == 2:
            audit.update(self._learned_byte_audit(completed=completed))
        return audit

    def _learned_byte_audit(self, *, completed: bool) -> dict[str, object]:
        """Audit source/native learning from factual target rows, never step counts."""
        assert self.config is not None and self.model is not None
        coordinate = self.portability_run.payload["coordinate"]
        condition = coordinate["condition"]
        finite_tensors = all(
            bool(torch.isfinite(value).all())
            for value in self.model.state_dict().values()
            if isinstance(value, torch.Tensor)
            and (value.is_floating_point() or value.is_complex())
        )
        unavailable = {
            "full_fact_exposure": False,
            "valid_update_audit": False,
            "finite_tensors": finite_tensors,
            "nonzero_table_delta": False,
            "gradient_changed_row_fraction": 0.0,
            "source_byte_audit_available": False,
            "source_byte_audit_reason": "condition_has_no_trainable_local_byte_table",
        }
        if condition not in {"source-real", "native"}:
            return unavailable
        memory = self.model.memory
        if memory is None or not hasattr(memory, "table"):
            raise TypeError("learned source/native audit lacks a local byte table")
        expected_steps = self.config.training.max_steps
        expected_rows = set(self._learned_fact_targets)
        state = self._learned_audit_state()
        applied_steps = cast(torch.Tensor, state["applied_steps"]).detach().cpu()
        if applied_steps.dtype != torch.bool or applied_steps.numel() != expected_steps:
            raise ValueError("learned audit optimizer update bitmap is malformed")
        history_valid = bool(applied_steps.all()) and all(
            isinstance(entry.get("gradient_parameter_names"), list)
            and isinstance(entry.get("update_parameter_names"), list)
            and entry["gradient_parameter_names"] == entry["update_parameter_names"]
            and set(entry["gradient_parameter_names"]).issubset(
                {name for name, _ in self._portability_parameters}
            )
            and isinstance(entry.get("row_gradient_evidence"), dict)
            for entry in self._portability_update_history
        )
        gradient_seen = cast(torch.Tensor, state["gradient_seen"]).detach().cpu()
        if gradient_seen.dtype != torch.bool or gradient_seen.numel() != len(
            expected_rows
        ):
            raise ValueError("learned audit optimizer gradient coverage is malformed")
        gradient_rows = {
            row
            for row, seen in zip(
                sorted(expected_rows), gradient_seen.tolist(), strict=True
            )
            if seen
        }
        changed_rows: set[int] = set()
        with torch.no_grad():
            for row, initial in self._learned_initial_table_rows.items():
                final = memory.table.weight[row].detach().to("cpu").contiguous()
                if bytes(final.view(torch.uint8).numpy()) != initial:
                    changed_rows.add(row)
            table_changed = bool(changed_rows)
        gradient_changed = expected_rows & gradient_rows & changed_rows
        expected_fact_samples = expected_steps * self.config.training.micro_batch_size
        expected_presentations, remainder = (
            divmod(expected_fact_samples, len(expected_rows))
            if expected_rows
            else (0, 1)
        )
        exposure_complete = (
            bool(expected_rows)
            and remainder == 0
            and expected_presentations > 0
            and all(
                self._learned_fact_exposures[row] >= expected_presentations
                for row in expected_rows
            )
        )
        result: dict[str, object] = {
            "full_fact_exposure": bool(completed and exposure_complete),
            "valid_update_audit": bool(completed and history_valid),
            "finite_tensors": finite_tensors,
            "nonzero_table_delta": bool(table_changed),
            "gradient_changed_row_fraction": len(gradient_changed) / len(expected_rows),
            "source_byte_audit_available": True,
            "expected_presentations_per_fact": expected_presentations,
            "factual_target_row_count": len(expected_rows),
            "factual_target_exposure_count": sum(self._learned_fact_exposures.values()),
            "factual_target_exposures": {
                self._learned_fact_ids_by_row[row]: self._learned_fact_exposures[row]
                for row in sorted(expected_rows)
            },
            "address_collision_target_count": self._learned_address_collision_targets,
            "applied_update_count": int(applied_steps.sum().item()),
            "expected_update_count": expected_steps,
            "applied_update_steps": (
                torch.nonzero(applied_steps, as_tuple=False).flatten().add(1).tolist()
            ),
            "gradient_observed_factual_row_count": len(expected_rows & gradient_rows),
            "gradient_observed_factual_rows": [
                self._learned_fact_ids_by_row[row]
                for row in sorted(expected_rows & gradient_rows)
            ],
            "changed_factual_row_count": len(changed_rows),
            "changed_factual_rows": [
                self._learned_fact_ids_by_row[row] for row in sorted(changed_rows)
            ],
            "gradient_changed_factual_row_count": len(gradient_changed),
            "gradient_changed_row_threshold": 0.9,
            "completed_training": completed,
        }
        if completed and condition == "source-real":
            result.update(self._source_memory_ablation())
        return result

    def _source_memory_ablation(self) -> dict[str, float]:
        """Measure source-monitor memory dependence without retaining state changes."""
        assert (
            self.config is not None
            and self.device is not None
            and self.model is not None
        )
        from sparselab.evaluation.inference import InferenceRun
        from sparselab.evaluation.learned_portability import (
            evaluate_learned_portability,
        )

        descriptor = self.portability_run.payload["world_manifest"]
        data_path = self.portability_run.root / descriptor["path"]
        data = json.loads(data_path.read_text(encoding="utf-8"))
        tokenizer = load_tokenizer(self.config.tokenizer.path)
        loaded = InferenceRun(
            Path("."),
            self.config,
            self.model,
            tokenizer,
            self.device,
            {"tokenizer_sha256": data["tokenizer"]["sha256"]},
        )
        enabled = evaluate_learned_portability(
            loaded,
            data_path,
            partitions=("source_monitor",),
            wording="source_monitor",
        )["metrics"]
        disabled = evaluate_learned_portability(
            loaded,
            data_path,
            partitions=("source_monitor",),
            wording="source_monitor",
            memory_enabled=False,
        )["metrics"]
        if not isinstance(enabled, dict) or not isinstance(disabled, dict):
            raise RuntimeError("source memory ablation lacks metrics")
        enabled_accuracy = float(enabled["accuracy"])
        disabled_accuracy = float(disabled["accuracy"])
        enabled_nll = float(enabled["answer_nll"])
        disabled_nll = float(disabled["answer_nll"])
        return {
            "enabled_minus_disabled_accuracy": enabled_accuracy - disabled_accuracy,
            "disabled_minus_enabled_nll": disabled_nll - enabled_nll,
            "source_monitor_enabled_accuracy": enabled_accuracy,
            "source_monitor_disabled_accuracy": disabled_accuracy,
            "source_monitor_enabled_answer_nll": enabled_nll,
            "source_monitor_disabled_answer_nll": disabled_nll,
        }

    def portability_update_evidence(self) -> dict[str, object] | None:
        evidence = self._last_portability_row_gradient_evidence
        if evidence is None:
            return None
        return {
            **evidence,
            "nonzero_row_indices": list(evidence["nonzero_row_indices"]),
            "nonzero_row_gradient_norms": list(evidence["nonzero_row_gradient_norms"]),
        }

    def train_update(
        self, microbatches: list[Microbatch], update_index: int, valid_targets: int
    ) -> UpdateResult:
        # Strict dynamic compilation must never turn cache exhaustion or a
        # compiler failure into an apparently successful eager training run.
        strict_compile = (
            self.config is not None and self.config.runtime.compile_dynamic_shapes
        )
        with (
            _translate_allocation_failures(),
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
        self._last_portability_row_gradient_evidence = None
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
        allocation_enabled = config.dataset.allocation_manifest_path is not None
        portability_enabled = self.portability_run is not None
        allocation_raw_counts = {
            code: 0
            for code in (OWNER_NEURAL, OWNER_LEXICAL, OWNER_SEMANTIC, OWNER_HYBRID)
        }
        allocation_target_counts = dict(allocation_raw_counts)
        allocation_groups: dict[int, list[torch.nn.Parameter]] | None = None
        if allocation_enabled:
            neural, lexical, semantic = [], [], []
            seen: set[int] = set()
            for name, parameter in model.named_parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                if not parameter.requires_grad:
                    continue
                if name.startswith("memory."):
                    lexical.append(parameter)
                elif name.startswith("semantic_memories."):
                    semantic.append(parameter)
                else:
                    neural.append(parameter)
            allocation_groups = {
                OWNER_NEURAL: neural,
                OWNER_LEXICAL: lexical,
                OWNER_SEMANTIC: semantic,
            }
        for batch, chunk_valid in zip(microbatches, chunk_counts, strict=True):
            if allocation_enabled:
                if (
                    batch.owner_ids is None
                    or batch.owner_ids.shape != batch.targets.shape
                    or batch.owner_ids.dtype != np.uint8
                    or np.any(batch.owner_ids > OWNER_HYBRID)
                ):
                    raise ValueError("owner sidecar shape, dtype, or codes are invalid")
                for owner in allocation_raw_counts:
                    allocation_raw_counts[owner] += int(
                        np.count_nonzero(batch.owner_ids == owner)
                    )
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
            owners = None
            owner_tensor = None
            memory_mask = None
            semantic_queries = None
            if allocation_enabled:
                if batch.owner_ids is None:
                    raise ValueError("allocation training requires owner sidecars")
                owners = batch.owner_ids
                if (
                    owners.shape != batch.targets.shape
                    or owners.dtype != np.uint8
                    or np.any(owners > OWNER_HYBRID)
                ):
                    raise ValueError("owner sidecar shape, dtype, or codes are invalid")
                owner_tensor = torch.from_numpy(owners).to(device)
                for owner in allocation_target_counts:
                    allocation_target_counts[owner] += int(
                        np.count_nonzero((owners == owner) & (batch.targets != -100))
                    )
                semantic_width = (
                    next(iter(model.semantic_memories.values())).retriever.key_dim
                    if model.semantic_memories
                    else None
                )
                assert allocation_groups is not None
                if (
                    allocation_target_counts[OWNER_LEXICAL]
                    and not allocation_groups[OWNER_LEXICAL]
                ):
                    raise ValueError(
                        "lexical ownership requires a lexical memory module"
                    )
                if (
                    allocation_target_counts[OWNER_SEMANTIC]
                    and not allocation_groups[OWNER_SEMANTIC]
                ):
                    raise ValueError(
                        "semantic ownership requires a verified semantic memory"
                    )
                if allocation_target_counts[OWNER_HYBRID] and (
                    not allocation_groups[OWNER_LEXICAL]
                    or not allocation_groups[OWNER_SEMANTIC]
                ):
                    raise ValueError(
                        "hybrid ownership requires lexical and semantic memory"
                    )
                memory_mask = (owner_tensor == OWNER_LEXICAL) | (
                    owner_tensor == OWNER_HYBRID
                )
                if (
                    batch.semantic_queries is not None
                    or batch.semantic_mask is not None
                ):
                    if (
                        self.semantic_encoder is None
                        or batch.semantic_queries is None
                        or batch.semantic_mask is None
                        or batch.semantic_queries.ndim != 3
                        or batch.semantic_queries.shape
                        != (*batch.inputs.shape, semantic_width)
                        or batch.semantic_queries.dtype != np.float32
                        or batch.semantic_mask.shape != batch.inputs.shape
                        or batch.semantic_mask.dtype != bool
                        or not np.isfinite(batch.semantic_queries).all()
                    ):
                        raise ValueError("semantic allocation sidecars are invalid")
                    semantic_mask = torch.from_numpy(batch.semantic_mask).to(device)
                    allowed = (owner_tensor == OWNER_SEMANTIC) | (
                        owner_tensor == OWNER_HYBRID
                    )
                    if bool(torch.any(semantic_mask & ~allowed)):
                        raise ValueError(
                            "semantic query mask may select only semantic or hybrid ownership"
                        )
                    semantic_queries = SemanticQueryBatch(
                        self.semantic_encoder,
                        torch.from_numpy(batch.semantic_queries).to(device),
                        semantic_mask,
                    )
                elif allocation_groups[OWNER_SEMANTIC]:
                    raise ValueError("semantic ownership requires query sidecars")
            elif batch.semantic_queries is not None:
                if self.semantic_encoder is None or batch.semantic_mask is None:
                    raise ValueError(
                        "semantic query sidecars require a verified attached pack"
                    )
                semantic_queries = SemanticQueryBatch(
                    self.semantic_encoder,
                    torch.from_numpy(batch.semantic_queries).to(device),
                    torch.from_numpy(batch.semantic_mask).to(device),
                )
            position_mask = None
            position_count = chunk_valid
            if batch.position_mask is not None:
                mask = batch.position_mask
                if mask.dtype != np.bool_ or mask.shape != batch.inputs.shape:
                    raise ValueError("position mask must be boolean and match inputs")
                if (
                    np.any((batch.targets != -100) & ~mask)
                    or np.any(mask[:, 1:] & ~mask[:, :-1])
                    or np.any(~mask[:, 0])
                ):
                    raise ValueError(
                        "position mask must be nonempty right-padded prefixes covering targets"
                    )
                position_mask = torch.from_numpy(mask).to(device)
                position_count = int(mask.sum())
            valid_target_mask = y != -100
            neural_aux_targets = 0
            if allocation_enabled:
                assert owner_tensor is not None
                if not portability_enabled:
                    neural_target_mask = (owner_tensor == OWNER_NEURAL) | (
                        owner_tensor == OWNER_HYBRID
                    )
                    valid_target_mask = valid_target_mask & neural_target_mask
                neural_aux_targets = int(valid_target_mask.sum())
            with (
                self.offload.hooks() if self.offload is not None else nullcontext(),
                precision_context(config, device),
            ):
                logits, auxiliary = model.forward_with_aux(
                    x,
                    byte_addresses=addresses,
                    semantic_queries=semantic_queries,
                    memory_mask=memory_mask,
                    valid_target_mask=valid_target_mask,
                    activation_checkpointing=config.runtime.memory.activation_checkpointing.enabled,
                    diagnostics=config.logging.architecture_diagnostics,
                    return_hidden=config.runtime.compile_output_head_loss,
                    position_mask=position_mask,
                )
            diagnostics = architecture_metrics(model)
            monitor.sample("forward")
            if config.runtime.compile_output_head_loss:
                ce_sum = self._compiled_output_head_loss(logits, y, model.output.weight)
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
            if allocation_enabled and not portability_enabled:
                assert owner_tensor is not None and allocation_groups is not None
                terms: list[tuple[torch.Tensor, list[torch.nn.Parameter]]] = []
                for owner, parameters in allocation_groups.items():
                    mask = (owner_tensor == owner) | (owner_tensor == OWNER_HYBRID)
                    count = int(mask.logical_and(y != -100).sum())
                    weight = (
                        config.training.neural_loss_weight
                        if owner == OWNER_NEURAL
                        else 1.0
                    )
                    if count and parameters and weight:
                        labels = y.masked_fill(~mask, -100)
                        component = functional.cross_entropy(
                            logits.float().flatten(0, 1),
                            labels.flatten(),
                            ignore_index=-100,
                            reduction="sum",
                        )
                        terms.append((component * (weight / valid_targets), parameters))
                if (
                    config.training.neural_loss_weight
                    and neural_aux_targets
                    and auxiliary.requires_grad
                ):
                    terms.append(
                        (
                            auxiliary
                            * neural_aux_targets
                            * config.training.neural_loss_weight
                            / valid_targets,
                            allocation_groups[OWNER_NEURAL],
                        )
                    )
                for index, (term, parameters) in enumerate(terms):
                    if self.scaler is not None:
                        term = self.scaler.scale(term)
                    torch.autograd.backward(
                        term,
                        inputs=parameters,
                        retain_graph=index + 1 < len(terms),
                    )
            else:
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
            if allocation_enabled and not portability_enabled:
                aux_sum = (
                    aux_sum
                    + auxiliary.detach()
                    * neural_aux_targets
                    * config.training.neural_loss_weight
                )
            else:
                aux_sum = aux_sum + auxiliary.detach() * chunk_valid
            for name, value in diagnostics.items():
                padded_engram = batch.position_mask is not None and name.startswith(
                    "engram/"
                )
                count_metric = padded_engram and name.rsplit("/", 1)[-1] in {
                    "lookup_count",
                    "unique_addresses",
                    "collision_count",
                }
                weight = (
                    1
                    if count_metric
                    else position_count
                    if padded_engram
                    else chunk_valid
                )
                diagnostic_sums[name] = (
                    diagnostic_sums.get(name, value.new_zeros(())) + value * weight
                )
                diagnostic_weights[name] = (
                    1 if count_metric else diagnostic_weights.get(name, 0) + weight
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
        row_gradient_evidence: dict[str, object] | None = None
        if (
            self.portability_run is not None
            and self.portability_run.payload.get("version") == 2
            and config.model.memory == "byte"
            and self.portability_run.payload["coordinate"]["condition"]
            in {"source-real", "native"}
        ):
            memory = model.memory
            if memory is None or not hasattr(memory, "table"):
                raise TypeError(
                    "learned source/native run lacks a trainable byte table"
                )
            if any(batch.byte_addresses is None for batch in microbatches):
                raise ValueError(
                    "learned source/native update lacks byte-address sidecars"
                )
            input_rows = np.unique(
                np.concatenate(
                    [
                        cast(np.ndarray, batch.byte_addresses).reshape(-1)
                        for batch in microbatches
                    ]
                )
            )
            row_indices = torch.as_tensor(input_rows, dtype=torch.long, device=device)
            table_gradient = memory.table.weight.grad
            if table_gradient is None:
                nonzero_indices: list[int] = []
                nonzero_norms: list[float] = []
                aggregate_norm = 0.0
            else:
                selected_gradients = table_gradient.index_select(0, row_indices).float()
                row_norms = torch.linalg.vector_norm(selected_gradients, dim=1)
                nonzero = row_norms > 0
                nonzero_indices = row_indices[nonzero].detach().cpu().tolist()
                nonzero_norms = row_norms[nonzero].detach().cpu().tolist()
                aggregate_norm = float(
                    torch.linalg.vector_norm(selected_gradients[nonzero]).detach().cpu()
                )
            row_gradient_evidence = {
                "step": update_index,
                "distinct_input_rows": len(input_rows),
                "nonzero_row_indices": [int(value) for value in nonzero_indices],
                "nonzero_row_gradient_norms": [float(value) for value in nonzero_norms],
                "nonzero_row_gradient_count": len(nonzero_indices),
                "nonzero_row_gradient_l2_norm": aggregate_norm,
            }
        if portability_enabled:
            norm = torch.nn.utils.clip_grad_norm_(
                [parameter for _, parameter in self._portability_parameters],
                config.training.grad_clip_norm,
                error_if_nonfinite=False,
                foreach=False,
            )
        elif allocation_enabled:
            assert allocation_groups is not None
            norms = [
                torch.nn.utils.clip_grad_norm_(
                    parameters,
                    config.training.grad_clip_norm,
                    error_if_nonfinite=False,
                    foreach=False,
                )
                for parameters in allocation_groups.values()
                if parameters
            ]
            norm = torch.linalg.vector_norm(
                torch.stack([value.float() for value in norms])
            )
        else:
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
        self._last_portability_row_gradient_evidence = row_gradient_evidence
        if self._learned_fact_targets:
            if (
                self._learned_audit_parameter is None
                or "sparselab_learned_audit_v1"
                not in self.optimizer.state[self._learned_audit_parameter]
            ):
                self._write_learned_audit_state()
            self._record_applied_factual_exposure(microbatches)
            state = self._learned_audit_state()
            steps = cast(torch.Tensor, state["applied_steps"])
            if (
                update_index < 1
                or update_index > steps.numel()
                or bool(steps[update_index - 1])
            ):
                raise RuntimeError("learned audit observed an invalid applied update")
            steps[update_index - 1] = True
            if row_gradient_evidence is None:
                raise RuntimeError("learned source/native update lacks row gradients")
            self._record_applied_gradient_rows(row_gradient_evidence)
        if portability_enabled:
            update_evidence: dict[str, object] = {
                "step": update_index,
                "gradient_parameter_names": list(gradient_names),
                "update_parameter_names": list(gradient_names),
            }
            if row_gradient_evidence is not None:
                update_evidence["row_gradient_evidence"] = row_gradient_evidence
            self._portability_update_history.append(update_evidence)
        metrics = {
            "train/loss": float(language_sum) / valid_targets,
            "moe/router_auxiliary_loss": float(aux_sum) / valid_targets,
            "optimizer/learning_rate": lr,
            "optimizer/grad_norm": float(norm),
            "performance/step_seconds": elapsed,
            "performance/tokens_per_second": valid_targets / max(elapsed, 1e-9),
            "batch/micro_batch_size": float(config.training.micro_batch_size),
            "batch/accumulation_steps": float(len(microbatches)),
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
            **memory_metrics,
        }
        if allocation_enabled:
            neural_targets = (
                allocation_target_counts[OWNER_NEURAL]
                + allocation_target_counts[OWNER_HYBRID]
            )
            metrics.update(
                {
                    "allocation/raw_tokens": float(sum(allocation_raw_counts.values())),
                    "allocation/valid_targets": float(
                        sum(allocation_target_counts.values())
                    ),
                    "allocation/weighted_neural_supervision_mass": (
                        config.training.neural_loss_weight * neural_targets
                    ),
                    "allocation/owner/neural_raw_tokens": float(
                        allocation_raw_counts[OWNER_NEURAL]
                    ),
                    "allocation/owner/lexical_raw_tokens": float(
                        allocation_raw_counts[OWNER_LEXICAL]
                    ),
                    "allocation/owner/semantic_raw_tokens": float(
                        allocation_raw_counts[OWNER_SEMANTIC]
                    ),
                    "allocation/owner/hybrid_raw_tokens": float(
                        allocation_raw_counts[OWNER_HYBRID]
                    ),
                    "allocation/owner/neural_targets": float(
                        allocation_target_counts[OWNER_NEURAL]
                    ),
                    "allocation/owner/lexical_targets": float(
                        allocation_target_counts[OWNER_LEXICAL]
                    ),
                    "allocation/owner/semantic_targets": float(
                        allocation_target_counts[OWNER_SEMANTIC]
                    ),
                    "allocation/owner/hybrid_targets": float(
                        allocation_target_counts[OWNER_HYBRID]
                    ),
                }
            )
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
        with _translate_allocation_failures():
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
                    semantic_queries = None
                    memory_mask = None
                    if config.dataset.allocation_manifest_path is not None:
                        if (
                            batch.owner_ids is None
                            or batch.owner_ids.shape != batch.targets.shape
                            or batch.owner_ids.dtype != np.uint8
                            or np.any(batch.owner_ids > OWNER_HYBRID)
                        ):
                            raise ValueError(
                                "allocation evaluation requires valid owner sidecars"
                            )
                        owners = torch.from_numpy(batch.owner_ids).to(device)
                        memory_mask = (owners == OWNER_LEXICAL) | (
                            owners == OWNER_HYBRID
                        )
                        semantic_width = (
                            next(
                                iter(model.semantic_memories.values())
                            ).retriever.key_dim
                            if model.semantic_memories
                            else None
                        )
                        if (
                            batch.semantic_queries is not None
                            or batch.semantic_mask is not None
                        ):
                            if (
                                self.semantic_encoder is None
                                or batch.semantic_queries is None
                                or batch.semantic_mask is None
                                or batch.semantic_queries.ndim != 3
                                or batch.semantic_queries.shape
                                != (*batch.inputs.shape, semantic_width)
                                or batch.semantic_queries.dtype != np.float32
                                or batch.semantic_mask.shape != batch.inputs.shape
                                or batch.semantic_mask.dtype != bool
                                or not np.isfinite(batch.semantic_queries).all()
                            ):
                                raise ValueError(
                                    "semantic allocation sidecars are invalid"
                                )
                            semantic_mask = torch.from_numpy(batch.semantic_mask).to(
                                device
                            )
                            allowed = (owners == OWNER_SEMANTIC) | (
                                owners == OWNER_HYBRID
                            )
                            if bool(torch.any(semantic_mask & ~allowed)):
                                raise ValueError(
                                    "semantic query mask may select only semantic or hybrid ownership"
                                )
                            semantic_queries = SemanticQueryBatch(
                                self.semantic_encoder,
                                torch.from_numpy(batch.semantic_queries).to(device),
                                semantic_mask,
                            )
                    elif batch.semantic_queries is not None:
                        if self.semantic_encoder is None or batch.semantic_mask is None:
                            raise ValueError(
                                "semantic query sidecars require a verified attached pack"
                            )
                        semantic_queries = SemanticQueryBatch(
                            self.semantic_encoder,
                            torch.from_numpy(batch.semantic_queries).to(device),
                            torch.from_numpy(batch.semantic_mask).to(device),
                        )
                    logits = model(
                        x,
                        byte_addresses=addresses,
                        semantic_queries=semantic_queries,
                        memory_mask=memory_mask,
                    )
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
        from sparselab.training.checkpoints import restore_optimizer_state

        remapped_optimizer_state = _named_optimizer_state(
            model,
            optimizer,
            state.optimizer,
            state.optimizer_parameter_names,
        )
        restore_optimizer_state(optimizer, remapped_optimizer_state)
        self._restore_learned_audit_state()
        if self.scaler is not None:
            if state.scaler is None:
                raise ValueError("fp16 resume requires saved GradScaler state")
            self.scaler.load_state_dict(state.scaler)
        _restore_rng(state.rng, device)

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
        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None
