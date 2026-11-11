"""Explicit byte-memory adapter transfer between compatible checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

from torch import Tensor

from klab.model.memory import ByteAddressMemory
from klab.model.transformer import DenseLM


def transfer_byte_memory(source_state: Mapping[str, Tensor], target: DenseLM) -> None:
    """Load only byte-memory parameters into a compatible target model."""
    if not isinstance(target.memory, ByteAddressMemory):
        raise TypeError("target model does not use byte memory")
    prefix = "memory."
    adapter = {
        key.removeprefix(prefix): value
        for key, value in source_state.items()
        if key.startswith(prefix)
    }
    required = {"table.weight", "output.weight", "gate.weight"}
    if adapter.keys() != required:
        raise ValueError("source checkpoint lacks an exact byte-memory adapter")
    target.memory.load_state_dict(adapter)
