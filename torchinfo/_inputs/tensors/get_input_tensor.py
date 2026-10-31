from __future__ import annotations
from torchinfo._constants.input_types import CORRECTED_INPUT_SIZE_TYPE
import torch


def get_input_tensor(
    input_size: CORRECTED_INPUT_SIZE_TYPE,
    batch_dim: int | None,
    dtypes: list[torch.dtype],
    device: torch.device,
) -> list[torch.Tensor]:
    """Get input_tensor with batch size 1 for use in model.forward()"""
    x = []
    for size, dtype in zip(input_size, dtypes, strict=False):
        input_tensor = torch.rand(*size) if size else torch.rand(())
        if batch_dim is not None:
            input_tensor = input_tensor.unsqueeze(dim=batch_dim)
        x.append(input_tensor.to(device).type(dtype))
    return x
