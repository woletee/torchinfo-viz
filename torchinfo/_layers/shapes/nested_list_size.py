from __future__ import annotations
from collections.abc import Mapping
from torchinfo._constants.input_types import DETECTED_INPUT_OUTPUT_TYPES
import numpy as np
import torch


def nested_list_size(inputs: DETECTED_INPUT_OUTPUT_TYPES) -> tuple[list[int], int]:
    """Flattens nested list size."""
    if hasattr(inputs, "tensors"):
        return nested_list_size(inputs.tensors)
    if isinstance(inputs, np.ndarray):
        inputs = torch.from_numpy(inputs)
    if isinstance(inputs, torch.Tensor):
        return (list(inputs.size()), inputs.element_size())
    if not hasattr(inputs, "__getitem__") or not inputs:
        return ([], 0)
    if isinstance(inputs, Mapping):
        return nested_list_size(list(inputs.values()))
    if callable(getattr(inputs, "size", None)) and callable(
        getattr(inputs, "element_size", None)
    ):
        return (list(inputs.size()), inputs.element_size())
    if isinstance(inputs, (list, tuple)):
        return nested_list_size(inputs[0])
    return ([], 0)
