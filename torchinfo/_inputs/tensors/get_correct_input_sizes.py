from __future__ import annotations
from torchinfo._constants.input_types import CORRECTED_INPUT_SIZE_TYPE
from torchinfo._constants.input_types import INPUT_SIZE_TYPE
from torchinfo._inputs.tensors.sizes import flatten


def get_correct_input_sizes(input_size: INPUT_SIZE_TYPE) -> CORRECTED_INPUT_SIZE_TYPE:
    """Convert input_size to the correct form, which is a list of tuples."""
    if not isinstance(input_size, (list, tuple)):
        raise TypeError(
            "Input_size is not a recognized type. Please ensure input_size is valid.\nFor multiple inputs to the network, ensure input_size is a list of tuple sizes. If you are having trouble here, please submit a GitHub issue."
        )
    if not input_size or any((size <= 0 for size in flatten(input_size))):
        raise ValueError("Input_data is invalid, or negative size found in input_data.")
    if isinstance(input_size, list) and isinstance(input_size[0], int):
        return [tuple(input_size)]
    if isinstance(input_size, list):
        return input_size
    if isinstance(input_size, tuple) and isinstance(input_size[0], tuple):
        return list(input_size)
    return [input_size]
