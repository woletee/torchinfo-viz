from __future__ import annotations
from torchinfo._constants.input_types import DETECTED_INPUT_OUTPUT_TYPES
from torchinfo._layers.shapes.is_packed_input import is_packed_input
from torchinfo._layers.shapes.unpacked_size import unpacked_size


def calculate_size(
    inputs: DETECTED_INPUT_OUTPUT_TYPES | None, batch_dim: int | None
) -> tuple[list[int], int]:
    """Return the corrected shape and element size for a layer input or output."""
    if inputs is None:
        return ([], 0)
    if is_packed_input(inputs):
        size = list(inputs[0].data.size())
        if batch_dim is not None:
            size = [*size[:batch_dim], 1, *size[batch_dim + 1 :]]
        return (size, inputs[0].data.element_size())
    return unpacked_size(inputs, batch_dim)
