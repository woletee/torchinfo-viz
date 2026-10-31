from __future__ import annotations
from torch import nn
from torchinfo._constants.input_types import INPUT_DATA_TYPE
import torch


def get_device(
    model: nn.Module, input_data: INPUT_DATA_TYPE | None
) -> torch.device | None:
    """If input_data is given, the device should not be changed"""
    if input_data is None:
        try:
            model_parameter = next(model.parameters())
        except StopIteration:
            model_parameter = None
        if model_parameter is not None and model_parameter.device:
            return model_parameter.device
        if torch.accelerator.is_available():
            return torch.accelerator.current_accelerator()
        return torch.device("cpu")
    return None
