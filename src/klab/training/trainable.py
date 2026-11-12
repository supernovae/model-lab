"""Select which model storages an optimizer updates."""

from __future__ import annotations

import torch

from klab.config.models import RunConfig
from klab.model.inspection import named_tensor_inventory


def apply_trainable_parameter_filter(
    model: torch.nn.Module, config: RunConfig
) -> tuple[tuple[str, torch.nn.Parameter], ...]:
    """Freeze every nonselected storage and return optimizer parameters by name."""
    names = config.training.trainable_parameters
    if names is None:
        return tuple(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )
    inventory = named_tensor_inventory(
        config.model, config.attention, trainable_parameters=names
    )
    parameters = dict(model.named_parameters())
    selected: list[tuple[str, torch.nn.Parameter]] = []
    for name in names:
        if name not in parameters or not inventory[name].trainable:
            raise ValueError(
                f"configured trainable parameter is not a canonical model storage: {name}"
            )
        selected.append((name, parameters[name]))
    selected_names = set(names)
    for name, parameter in parameters.items():
        parameter.requires_grad_(name in selected_names)
    return tuple(selected)
