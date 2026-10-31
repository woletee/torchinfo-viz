from __future__ import annotations
from torchinfo._constants.input_types import CORRECTED_INPUT_DATA_TYPE
from torchinfo._constants.input_types import INPUT_DATA_TYPE
from torchinfo._constants.input_types import INPUT_SIZE_TYPE
from torchinfo._inputs.tensors.get_correct_input_sizes import get_correct_input_sizes
from torchinfo._inputs.tensors.sizes import get_input_data_sizes
from torchinfo._inputs.tensors.get_input_tensor import get_input_tensor
from typing import Any
import numpy as np
import torch


def process_input(
    input_data: INPUT_DATA_TYPE | None,
    input_size: INPUT_SIZE_TYPE | None,
    batch_dim: int | None,
    device: torch.device | None,
    dtypes: list[torch.dtype] | None = None,
) -> tuple[CORRECTED_INPUT_DATA_TYPE, Any]:
    """Reads sample input data to get the input size."""
    if input_size is not None:
        assert device is not None
        dtypes = [torch.float] * len(input_size) if dtypes is None else dtypes
        correct_input_size = get_correct_input_sizes(input_size)
        x = get_input_tensor(correct_input_size, batch_dim, dtypes, device)
        return (x, correct_input_size)
    if input_data is None:
        return (None, [])
    is_array = isinstance(input_data, (torch.Tensor, np.ndarray))
    return ([input_data] if is_array else input_data, get_input_data_sizes(input_data))
