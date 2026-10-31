from __future__ import annotations
from collections.abc import Mapping
from torchinfo._layers.shapes.nested_list_size import nested_list_size
import numpy as np
import torch


def unpacked_size(inputs, batch_dim):
    if isinstance(inputs, Mapping):
        size, elem_bytes = nested_list_size(list(inputs.values())[-1])
        if batch_dim is not None:
            size = [[*size[:batch_dim], 1, *size[batch_dim + 1 :]]]
        return (size, elem_bytes)
    if isinstance(inputs, torch.Tensor):
        return (list(inputs.size()), inputs.element_size())
    if isinstance(inputs, np.ndarray):
        tensor = torch.from_numpy(inputs)
        return (list(tensor.size()), tensor.element_size())
    if isinstance(inputs, (list, tuple)):
        size, elem_bytes = nested_list_size(inputs)
        if batch_dim is not None and batch_dim < len(size):
            size[batch_dim] = 1
        return (size, elem_bytes)
    if isinstance(inputs, torch.distributions.Distribution):
        return (list(inputs.batch_shape + inputs.event_shape), 0)
    raise TypeError(
        f"Model contains a layer with an unsupported input or output type: {inputs}, type: {type(inputs)}"
    )
