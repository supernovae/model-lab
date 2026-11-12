"""Optional native MLX execution engine.

MLX is imported only when this engine is initialized.  Checkpoint codecs consume the
plain NumPy trees returned through :class:`EngineState`; offline consumers therefore
never need an MLX installation.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np

from klab.config.models import RunConfig
from klab.engines.base import (
    CanonicalTensor,
    EngineCapabilityError,
    EngineNonFiniteError,
    EngineState,
    EvaluationResult,
    Microbatch,
    UpdateResult,
    WeightSource,
)
from klab.engines.mlx_policy import validate as validate_mlx
from klab.memory import MemoryMonitor
from klab.runtime import RuntimeInfo
from klab.runtime_profile import RuntimeAuthorization, require_authorization
from klab.training.optimizer import learning_rate_for_step


def _capture_rng_state(mx: Any) -> dict[str, object]:
    """Capture the codec-compatible generators affected by MLX initialization."""
    numpy_state = np.random.get_state()
    key = np.asarray(mx.random.state[0], dtype=np.uint32).copy()
    return {
        "python": random.getstate(),
        "numpy_kind": str(numpy_state[0]),
        "numpy_keys": np.asarray(numpy_state[1], dtype=np.uint32).copy(),
        "numpy_pos": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "mlx": key,
    }


def _restore_rng_state(mx: Any, rng: Mapping[str, object]) -> None:
    """Restore the exact envelope used by native MLX checkpoint state."""
    key = np.asarray(rng["mlx"], dtype=np.uint32)
    random.setstate(rng["python"])  # type: ignore[arg-type]
    np.random.set_state(
        (
            rng["numpy_kind"],
            np.asarray(rng["numpy_keys"], dtype=np.uint32),
            rng["numpy_pos"],
            rng["numpy_has_gauss"],
            rng["numpy_cached_gaussian"],
        )
    )
    # ``mx.random.state`` is read-only in MLX 0.32.2; re-seeding is the
    # SDK-supported restoration path.
    mx.random.seed((int(key[0]) << 32) | int(key[1]))


@contextmanager
def preserve_rng_state() -> Iterator[None]:
    """Preserve Python, NumPy, and MLX RNG around inference-only setup."""
    import mlx.core as mx

    state = _capture_rng_state(mx)
    try:
        yield
    finally:
        _restore_rng_state(mx, state)


def _flatten(tree: object) -> dict[str, object]:
    from mlx import nn

    return dict(nn.utils.tree_flatten(tree))


def _unflatten(tree: Mapping[str, object]) -> object:
    from mlx import nn

    return nn.utils.tree_unflatten(tree)


def _mlx_tree(tree: Mapping[str, object], mx: Any) -> object:
    return _unflatten({name: mx.array(value) for name, value in tree.items()})


class _MLXWeightSource:
    def __init__(self, engine: MLXEngine) -> None:
        self._engine = engine

    @property
    def aliases(self) -> Mapping[str, str]:
        return self._engine._canonical_aliases()

    def tensors(self) -> Iterator[CanonicalTensor]:
        # Conversion is deliberately per parameter, which bounds CPU staging to one
        # canonical tensor while the checkpoint writer emits shards.
        for name, value in _flatten(self._engine.model.parameters()).items():
            array = np.ascontiguousarray(np.asarray(value))
            yield CanonicalTensor(name=name, array=array, trainable=True)
            del array


class MLXEngine:
    """Actual MLX fp32 engine; trainer-owned counters never live here."""

    STATE_CODEC = "mlx_native"
    STATE_CODEC_VERSION = 1

    def __init__(self) -> None:
        self.config: RunConfig | None = None
        self.runtime: RuntimeInfo | None = None
        self.model: Any | None = None
        self.optimizer: Any | None = None
        self.monitor: MemoryMonitor | None = None
        self._mx: Any | None = None
        self._nn: Any | None = None
        self._value_and_grad: Any | None = None
        self._optimizer_parameter_names: list[list[str]] = []

    def validate(
        self, config: RunConfig, *, authorization: RuntimeAuthorization | None = None
    ) -> RuntimeInfo:
        from klab.runtime import validate_runtime

        self.runtime = validate_runtime(config, authorization=authorization)
        self.authorization = authorization
        return self.runtime

    def initialize(
        self, config: RunConfig, initial_weights: Mapping[str, object] | None = None
    ) -> None:
        require_authorization(config, getattr(self, "authorization", None))
        validate_mlx(config)
        runtime = self.runtime if self.runtime is not None else self.validate(config)
        import mlx.core as mx
        import mlx.optimizers as optim
        from mlx import nn

        from klab.model.mlx_dense import MLXDenseLM

        if not mx.metal.is_available():
            raise EngineCapabilityError("MLX Metal execution is unavailable")
        mx.set_default_device(mx.gpu)
        random.seed(config.seed)
        np.random.seed(config.seed)
        # Module constructors use MLX's default stream. Derive that stream from an
        # explicit key split instead of relying on unrelated Torch/Python state.
        init_key, _next_key = mx.random.split(mx.random.key(config.seed))
        mx.eval(init_key)
        init_words = np.asarray(init_key, dtype=np.uint32)
        mx.random.seed((int(init_words[0]) << 32) | int(init_words[1]))
        self.config, self.runtime, self._mx, self._nn = config, runtime, mx, nn
        self.model = MLXDenseLM(
            config.model,
            config.attention,
            recompute_blocks=config.runtime.memory.activation_checkpointing.enabled,
        )
        flat = _flatten(self.model.trainable_parameters())
        decay = [name for name, value in flat.items() if len(value.shape) >= 2]
        no_decay = [name for name, value in flat.items() if len(value.shape) < 2]
        self._optimizer_parameter_names = [decay, no_decay]
        self.optimizer = optim.MultiOptimizer(
            [
                optim.AdamW(
                    learning_rate=config.optimizer.peak,
                    betas=list(config.optimizer.betas),
                    eps=config.optimizer.eps,
                    weight_decay=config.optimizer.weight_decay,
                    bias_correction=True,
                ),
                optim.AdamW(
                    learning_rate=config.optimizer.peak,
                    betas=list(config.optimizer.betas),
                    eps=config.optimizer.eps,
                    weight_decay=0.0,
                    bias_correction=True,
                ),
            ],
            filters=[lambda _name, value: len(value.shape) >= 2],
        )
        if initial_weights is not None:
            self.load_canonical_weights(initial_weights)
        # AdamW initializes its slots on its first update.  Keeping that lazy state
        # avoids allocating two full moment trees for inference and step-zero runs.
        self.monitor = MemoryMonitor("metal", runtime_info=runtime)
        self._value_and_grad = nn.value_and_grad(self.model, self._loss_sum)

    def _require_initialized(self) -> tuple[Any, Any, Any]:
        if (
            self.config is None
            or self.model is None
            or self.optimizer is None
            or self._mx is None
        ):
            raise RuntimeError("MLXEngine.initialize() must be called first")
        return self.config, self.model, self.optimizer

    def _loss_sum(self, input_ids: Any, targets: Any) -> tuple[Any, Any]:
        """Summed masked CE; the caller applies the window denominator once."""
        _, model, _ = self._require_initialized()
        mx = self._mx
        assert mx is not None
        logits = model(input_ids).astype(mx.float32)
        valid = targets != -100
        labels = mx.maximum(targets, 0).astype(mx.int32)
        selected = mx.take_along_axis(logits, labels[..., None], axis=-1)[..., 0]
        per_token = mx.logsumexp(logits, axis=-1) - selected
        return mx.sum(per_token * valid.astype(mx.float32)), mx.sum(
            valid.astype(mx.int32)
        )

    def logits(self, input_ids: Any) -> Any:
        """Native logits API retained for MLX inference integrations."""
        _, model, _ = self._require_initialized()
        return model(input_ids)

    def _canonical_aliases(self) -> dict[str, str]:
        config, _, _ = self._require_initialized()
        return (
            {"output.weight": "embedding.weight"} if config.model.tie_embeddings else {}
        )

    def _finite_tree(self, tree: object) -> bool:
        mx = self._mx
        assert mx is not None
        checks = [mx.all(mx.isfinite(value)) for value in _flatten(tree).values()]
        mx.eval(checks)
        return all(bool(value) for value in checks)

    def _rng(self) -> dict[str, object]:
        mx = self._mx
        assert mx is not None
        return _capture_rng_state(mx)

    def _restore_rng(self, rng: Mapping[str, object]) -> None:
        """Validate all generators before changing any process-global generator."""
        from klab.training.mlx_checkpoints import validate_rng

        mx = self._mx
        assert mx is not None
        validate_rng(rng)
        _restore_rng_state(mx, rng)

    def train_update(
        self, microbatches: list[Microbatch], update_index: int, valid_targets: int
    ) -> UpdateResult:
        config, model, optimizer = self._require_initialized()
        mx, value_and_grad, monitor = self._mx, self._value_and_grad, self.monitor
        assert mx is not None and value_and_grad is not None and monitor is not None
        if not microbatches or type(update_index) is not int or update_index <= 0:
            raise ValueError(
                "MLX update requires microbatches and a positive update index"
            )
        if type(valid_targets) is not int or valid_targets <= 0:
            raise ValueError("MLX update needs a positive valid target count")
        from mlx.optimizers import clip_grad_norm
        from mlx.utils import tree_map

        mx.synchronize()
        started = time.perf_counter()
        monitor.begin_update()
        before_rng = self._rng()
        model.train()
        gradients: object | None = None
        loss_sum, observed_targets = 0.0, 0
        for batch in microbatches:
            chunk_valid = int(np.count_nonzero(batch.targets != -100))
            if not chunk_valid:
                continue
            inputs = mx.array(np.asarray(batch.inputs, dtype=np.int32))
            targets = mx.array(np.asarray(batch.targets, dtype=np.int32))
            (loss, count), grads = value_and_grad(inputs, targets)
            mx.eval(loss, count, grads)
            count_value, loss_value = int(count), float(loss)
            observed_targets += count_value
            loss_sum += loss_value
            gradients = (
                grads
                if gradients is None
                else tree_map(lambda left, right: left + right, gradients, grads)
            )
            monitor.sample("forward-backward")
        if observed_targets != valid_targets:
            raise ValueError(
                f"MLX valid target mismatch: expected {valid_targets}, observed {observed_targets}"
            )
        assert gradients is not None
        if not np.isfinite(loss_sum) or not self._finite_tree(gradients):
            self._restore_rng(before_rng)
            raise EngineNonFiniteError("nonfinite MLX loss or gradients")
        gradients = tree_map(lambda value: value / valid_targets, gradients)
        gradients, norm = clip_grad_norm(gradients, config.training.grad_clip_norm)
        mx.eval(norm, gradients)
        gradient_norm = float(norm)
        if not np.isfinite(gradient_norm) or not self._finite_tree(gradients):
            self._restore_rng(before_rng)
            raise EngineNonFiniteError("nonfinite MLX clipped gradients")
        lr = learning_rate_for_step(
            update_index,
            config.training.max_steps,
            config.optimizer.warmup_steps,
            config.optimizer.peak,
            config.optimizer.floor,
        )
        optimizer.learning_rate = lr
        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state)
        monitor.sample("optimizer")
        if not self._finite_tree(model.parameters()) or not self._finite_tree(
            optimizer.state
        ):
            raise EngineNonFiniteError(
                "MLX optimizer produced non-finite state after commit"
            )
        memory_metrics = monitor.end_update()
        mx.synchronize()
        elapsed = time.perf_counter() - started
        metrics = {
            "train/loss": loss_sum / valid_targets,
            "optimizer/learning_rate": lr,
            "optimizer/grad_norm": gradient_norm,
            "performance/step_seconds": elapsed,
            "performance/tokens_per_second": valid_targets / max(elapsed, 1e-9),
            "batch/micro_batch_size": float(config.training.micro_batch_size),
            "batch/accumulation_steps": float(len(microbatches)),
            "batch/effective_batch_size": float(
                sum(batch.inputs.shape[0] for batch in microbatches)
            ),
            "batch/effective_tokens_per_update": float(valid_targets),
            **memory_metrics,
        }
        return UpdateResult(
            "APPLIED", loss_sum, valid_targets, valid_targets, elapsed, metrics
        )

    def evaluate(self, batches: Iterable[Microbatch]) -> EvaluationResult:
        _, model, _ = self._require_initialized()
        mx = self._mx
        assert mx is not None
        was_training, state = model.training, self._rng()
        loss_sum, valid_targets, count = 0.0, 0, 0
        try:
            model.eval()
            for batch in batches:
                loss, targets = self._loss_sum(
                    mx.array(np.asarray(batch.inputs, dtype=np.int32)),
                    mx.array(np.asarray(batch.targets, dtype=np.int32)),
                )
                mx.eval(loss, targets)
                loss_sum += float(loss)
                valid_targets += int(targets)
                count += 1
        finally:
            model.train(was_training)
            self._restore_rng(state)
        if valid_targets <= 0 or count <= 0:
            raise ValueError("MLX evaluation has no valid targets")
        if not np.isfinite(loss_sum):
            raise EngineNonFiniteError("non-finite MLX evaluation loss")
        return EvaluationResult(loss_sum, valid_targets, count)

    def export_weights(self) -> WeightSource:
        self._require_initialized()
        return _MLXWeightSource(self)

    def load_canonical_weights(
        self, weights: Mapping[str, object], aliases: Mapping[str, str] | None = None
    ) -> None:
        """Validate canonical CPU tensors before a single native model mutation."""
        _, model, _ = self._require_initialized()
        mx = self._mx
        assert mx is not None
        expected_aliases = self._canonical_aliases()
        if aliases is not None and dict(aliases) != expected_aliases:
            raise ValueError("canonical aliases differ from the model architecture")
        expected = _flatten(model.parameters())
        canonical = set(expected)
        supplied = set(weights)
        missing = canonical - supplied
        extra = supplied - canonical - set(expected_aliases)
        if missing or extra:
            raise ValueError(
                f"canonical tensor names differ; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        converted: dict[str, object] = {}
        for name in canonical:
            array = np.asarray(weights[name])
            expected_value = expected[name]
            if (
                array.shape != tuple(expected_value.shape)
                or array.dtype != np.dtype(np.float32)
                or not np.isfinite(array).all()
            ):
                raise ValueError(
                    f"canonical tensor shape, dtype, or values differ: {name}"
                )
            converted[name] = mx.array(array)
        for alias, target in expected_aliases.items():
            if alias in weights:
                alias_value = np.asarray(weights[alias])
                if (
                    alias_value.shape != np.asarray(weights[target]).shape
                    or alias_value.dtype != np.dtype(np.float32)
                    or not np.array_equal(alias_value, np.asarray(weights[target]))
                ):
                    raise ValueError(f"canonical alias values differ: {alias}")
        model.update(_unflatten(converted))
        mx.eval(model.parameters())

    def export_training_state(self) -> EngineState:
        _, _, optimizer = self._require_initialized()
        return EngineState(
            optimizer={"state": _flatten(optimizer.state)},
            rng=self._rng(),
            scaler=None,
            optimizer_parameter_names=[
                names.copy() for names in self._optimizer_parameter_names
            ],
        )

    def restore_training_state(self, state: EngineState) -> None:
        """Validate a complete native state before mutating MLX or global RNG."""
        from klab.training.mlx_checkpoints import (
            validate_optimizer_state,
            validate_rng,
        )

        config, model, optimizer = self._require_initialized()
        mx = self._mx
        assert mx is not None
        if state.scaler is not None:
            raise ValueError("MLX fp32 resume cannot restore a gradient scaler")
        if set(state.optimizer) != {"state"} or not isinstance(
            state.optimizer["state"], Mapping
        ):
            raise ValueError(
                "MLX optimizer state must contain its flattened state tree"
            )
        if not isinstance(state.optimizer_parameter_names, list):
            raise TypeError("MLX optimizer parameter groups must be lists")
        shapes = {
            name: tuple(value.shape)
            for name, value in _flatten(model.trainable_parameters()).items()
        }
        exported = {
            name: np.asarray(value) for name, value in state.optimizer["state"].items()
        }
        completed_step = validate_optimizer_state(
            {"state": exported},
            state.optimizer_parameter_names,
            shapes,
        )
        if completed_step > config.training.max_steps:
            raise ValueError("MLX optimizer step exceeds the configured schedule")
        expected_learning_rate = (
            config.optimizer.peak
            if completed_step == 0
            else learning_rate_for_step(
                completed_step,
                config.training.max_steps,
                config.optimizer.warmup_steps,
                config.optimizer.peak,
                config.optimizer.floor,
            )
        )
        if float(exported["states.0.learning_rate"]) != float(
            np.float32(expected_learning_rate)
        ):
            raise ValueError("MLX optimizer learning rate differs from saved schedule")
        validate_rng(state.rng)
        native = _mlx_tree(exported, mx)
        # The codec has already inspected NumPy dtypes and finiteness; only now
        # materialize MLX state, after every reversible validation has succeeded.
        mx.eval(native)
        optimizer.state = native
        self._restore_rng(state.rng)

    def memory_snapshot(self) -> dict[str, float]:
        self._require_initialized()
        assert self.monitor is not None
        return {
            key: float(value)
            for key, value in self.monitor.sample("snapshot").items()
            if key.startswith("memory/") and isinstance(value, (int, float))
        }

    def synchronize(self) -> None:
        _, model, optimizer = self._require_initialized()
        mx = self._mx
        assert mx is not None
        mx.eval(model.parameters(), optimizer.state)
        mx.synchronize()

    def close(self) -> None:
        if self.monitor is not None:
            self.monitor.close()
            self.monitor = None
        self.model = self.optimizer = self._value_and_grad = None
