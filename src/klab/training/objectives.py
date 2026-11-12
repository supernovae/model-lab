"""Training objectives and the frozen models they compare against.

An objective turns one microbatch into the loss an update minimizes. It never
touches the optimizer, the data cursor or checkpoints: the engine hands it a
``forward`` for the model being trained, the batch, and any frozen models the
run declared, and backpropagates what it returns.

Losses are reported per supervised target. ``Loss.total`` is therefore in
"summed over this microbatch's targets" units: the engine adds the totals of an
update and divides by the update's target count. An objective whose natural
unit is something else (a pair, a document) scales its mean accordingly; see
:class:`Preference`.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor
from torch.nn import functional

if TYPE_CHECKING:
    from klab.config.models import RunConfig

Forward = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class Batch:
    """One microbatch on the training device.

    ``inputs`` and ``targets`` are ``[rows, tokens]``, or ``[pairs, 2, tokens]``
    for an objective registered with ``pairs=True`` (chosen reply first).
    Targets are ``-100`` where a position is not supervised.
    """

    inputs: Tensor
    targets: Tensor
    # Totals across every microbatch of the optimizer update this batch is in.
    targets_in_update: int
    rows_in_update: int

    @property
    def target_count(self) -> Tensor:
        return (self.targets != -100).sum()


@dataclass(frozen=True)
class Frozen:
    """Frozen models, each a ``forward(input_ids) -> logits`` that tracks no gradients."""

    teacher: Forward | None = None
    reference: Forward | None = None


@dataclass(frozen=True)
class Loss:
    """``total`` is backpropagated; ``metrics`` are microbatch means to log."""

    total: Tensor
    metrics: dict[str, Tensor] = field(default_factory=dict)


def cross_entropy_sum(logits: Tensor, targets: Tensor) -> Tensor:
    return functional.cross_entropy(
        logits.float().flatten(0, -2),
        targets.flatten(),
        ignore_index=-100,
        reduction="sum",
    )


class TrainingObjective:
    """Base class: next-token evaluation, and a place to hang ``loss``."""

    def loss(self, forward: Forward, batch: Batch, frozen: Frozen) -> Loss:
        raise NotImplementedError

    def evaluation_loss(self, forward: Forward, batch: Batch, frozen: Frozen) -> Tensor:
        """Held-out loss in the same per-target-sum units; no gradients are tracked."""
        return cross_entropy_sum(forward(batch.inputs), batch.targets)


class NextToken(TrainingObjective):
    def loss(self, forward: Forward, batch: Batch, frozen: Frozen) -> Loss:
        return Loss(cross_entropy_sum(forward(batch.inputs), batch.targets))


class Distill(TrainingObjective):
    """Cross-entropy blended with KL toward a frozen teacher (Hinton et al., 2015)."""

    def __init__(self, weight: float, temperature: float) -> None:
        self.weight, self.temperature = weight, temperature

    def loss(self, forward: Forward, batch: Batch, frozen: Frozen) -> Loss:
        assert frozen.teacher is not None
        supervised = batch.targets != -100
        student = forward(batch.inputs).float()[supervised]
        teacher = frozen.teacher(batch.inputs).float()[supervised]
        labels = batch.targets[supervised]
        hard = functional.cross_entropy(student, labels, reduction="sum")
        # The temperature-squared factor keeps the soft term's gradient scale
        # independent of the temperature.
        soft = functional.kl_div(
            functional.log_softmax(student / self.temperature, dim=-1),
            functional.softmax(teacher / self.temperature, dim=-1),
            reduction="sum",
        ) * (self.temperature**2)
        count = supervised.sum().clamp(min=1)
        return Loss(
            (1 - self.weight) * hard + self.weight * soft,
            {
                "next_token_loss": hard.detach() / count,
                "teacher_kl": soft.detach() / count,
                "teacher_agreement": (student.argmax(-1) == teacher.argmax(-1))
                .float()
                .mean(),
            },
        )


def reply_log_likelihood(logits: Tensor, targets: Tensor) -> Tensor:
    """Summed log-probability of each row's supervised targets."""
    token = -functional.cross_entropy(
        logits.float().flatten(0, -2),
        targets.flatten(),
        ignore_index=-100,
        reduction="none",
    )
    return token.view(targets.shape).sum(-1)


class Preference(TrainingObjective):
    """Direct preference optimization (Rafailov et al., 2023).

    Raises the chosen reply's likelihood relative to the rejected one, measured
    against the frozen reference so the policy is not rewarded for drifting.
    """

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def _pairs(
        self, forward: Forward, batch: Batch, frozen: Frozen
    ) -> tuple[Tensor, Tensor, Tensor]:
        assert frozen.reference is not None
        if batch.inputs.ndim != 3 or batch.inputs.shape[1] != 2:
            raise ValueError("preference objective needs [pairs, 2, tokens] batches")
        inputs, targets = batch.inputs.flatten(0, 1), batch.targets.flatten(0, 1)
        policy = reply_log_likelihood(forward(inputs), targets).view(-1, 2)
        reference = reply_log_likelihood(frozen.reference(inputs), targets).view(-1, 2)
        gain = policy - reference
        margin = gain[:, 0] - gain[:, 1]
        return -functional.logsigmoid(self.beta * margin), margin, policy

    def loss(self, forward: Forward, batch: Batch, frozen: Frozen) -> Loss:
        pair_loss, margin, policy = self._pairs(forward, batch, frozen)
        # The update's loss is the mean over its pairs, in per-target-sum units.
        scale = batch.targets_in_update / max(batch.rows_in_update, 1)
        return Loss(
            pair_loss.sum() * scale,
            {
                "preference_loss": pair_loss.detach().mean(),
                "preference_margin": margin.detach().mean(),
                "preference_accuracy": (policy[:, 0] > policy[:, 1]).float().mean(),
                "reward_accuracy": (margin > 0).float().mean(),
            },
        )

    def evaluation_loss(self, forward: Forward, batch: Batch, frozen: Frozen) -> Tensor:
        pair_loss, _, _ = self._pairs(forward, batch, frozen)
        return pair_loss.mean() * batch.target_count


def read_checkpoint(path: Path, *, role: str) -> tuple[RunConfig, Any, dict[str, Any]]:
    """Verify a checkpoint and return its run config, weights snapshot and manifest."""
    from klab.config.models import RunConfig as _RunConfig
    from klab.training.checkpoints import CheckpointManager
    from klab.training.manifest import (
        canonical_digest,
        checkpoint_selection,
        read_manifest,
    )

    selected = checkpoint_selection(Path(path))
    root = selected.parent.parent.resolve()
    if not (root / "manifest.json").is_file():
        raise ValueError(
            f"{role} checkpoint {path} is not a run directory or a checkpoint "
            "inside a run's checkpoints directory"
        )
    manifest = read_manifest(root / "manifest.json")
    source = _RunConfig.model_validate(manifest["effective_config"])
    manager = CheckpointManager(root, manifest_sha256=canonical_digest(manifest))
    with manager.writer_lease():
        snapshot = manager.load(selected, "promote")
    return source, snapshot, {**manifest, "run_root": str(root)}


def load_frozen_model(
    path: Path, config: RunConfig, device: torch.device, *, role: str
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Load a verified checkpoint as a frozen model that shares the run's tokenizer.

    Returns the model in evaluation mode and the identity recorded for the run.
    The architecture may differ from the model being trained; the vocabulary,
    the tokenizer and the context it must read may not.
    """
    from klab.model.transformer import DenseLM
    from klab.training.manifest import sha256_file

    source, snapshot, manifest = read_checkpoint(path, role=role)
    root = Path(manifest["run_root"])
    if source.runtime.engine != "pytorch":
        raise ValueError(f"{role} checkpoint must come from the PyTorch engine")
    if sha256_file(root / "tokenizer.json") != sha256_file(config.tokenizer.path):
        raise ValueError(f"{role} checkpoint was trained with a different tokenizer")
    if source.model.vocab_size != config.model.vocab_size:
        raise ValueError(f"{role} vocabulary size differs from the run's")
    if source.model.max_seq_len < config.training.seq_len:
        raise ValueError(
            f"{role} context ({source.model.max_seq_len}) is shorter than "
            f"training.seq_len ({config.training.seq_len})"
        )
    if source.model.memory in {"byte", "portable"}:
        raise ValueError(
            f"{role} checkpoint needs byte addresses, "
            "which an objective's frozen model does not receive"
        )
    model = DenseLM(source.model, source.attention)
    model.load_state_dict(snapshot.model)  # type: ignore[arg-type]
    model.requires_grad_(False)
    model.eval()
    return model.to(device), {
        "run_id": manifest["run_id"],
        "checkpoint_sha256": snapshot.checkpoint_sha256,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
    }
