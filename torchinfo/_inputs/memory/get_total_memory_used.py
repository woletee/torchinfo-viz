from __future__ import annotations
from collections.abc import Mapping
from torchinfo._constants.input_types import CORRECTED_INPUT_DATA_TYPE
from torchinfo._inputs.memory.get_tensor_memory import get_tensor_memory
from torchinfo._inputs.traverse_input_data import traverse_input_data
from typing import cast


def get_total_memory_used(data: CORRECTED_INPUT_DATA_TYPE) -> int:
    """Calculates the total memory of all tensors stored in data."""
    result = traverse_input_data(
        data,
        action_fn=get_tensor_memory,
        aggregate_fn=lambda data: (
            (lambda d: sum((v for v in d.values() if isinstance(v, int))))
            if isinstance(data, Mapping)
            else sum
        ),
    )
    return cast("int", result)
