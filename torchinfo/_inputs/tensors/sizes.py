from __future__ import annotations
from collections.abc import Iterator
from typing import Any
from torchinfo._constants.input_types import INPUT_SIZE_TYPE
from torchinfo._inputs.traverse_input_data import traverse_input_data


def flatten(nested_array: INPUT_SIZE_TYPE) -> Iterator[Any]:
    """Flattens a nested array."""
    for item in nested_array:
        if isinstance(item, (list, tuple)):
            yield from flatten(item)
        else:
            yield item


def get_input_data_sizes(data: Any) -> Any:
    """Converts input data to an equivalent data structure of torch.Sizes"""
    return traverse_input_data(
        data, action_fn=lambda data: data.size(), aggregate_fn=type
    )
