from __future__ import annotations
from torch import nn
from torchinfo._constants.input_types import CORRECTED_INPUT_DATA_TYPE
from torchinfo._enums.run_settings import Mode
from torchinfo._execution.forward.cache import _cached_forward_pass
from torchinfo._execution.forward.collect_layers import collect_layers
from torchinfo.layer_info import LayerInfo
from typing import Any
import torch


def forward_pass(
    model: nn.Module,
    x: CORRECTED_INPUT_DATA_TYPE,
    batch_dim: int | None,
    cache_forward_pass: bool,
    device: torch.device | None,
    mode: Mode,
    **kwargs: Any,
) -> list[LayerInfo]:
    """Run the model and collect its layer statistics."""
    model_name = model.__class__.__name__
    if cache_forward_pass and model_name in _cached_forward_pass:
        return _cached_forward_pass[model_name]
    layers = collect_layers(model, x, batch_dim, device, mode, kwargs)
    if cache_forward_pass and x is not None:
        _cached_forward_pass[model_name] = layers
    return layers
