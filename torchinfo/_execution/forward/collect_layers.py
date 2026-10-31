from __future__ import annotations
from torch import nn
from torchinfo._constants.input_types import CORRECTED_INPUT_DATA_TYPE
from torchinfo._enums.run_settings import Mode
from torchinfo._execution.layer_tree import add_missing_container_layers as missing
from torchinfo._execution.hooks.apply_hooks import apply_hooks
from torchinfo._execution.layer_tree.set_children_layers import set_children_layers
from torchinfo._graph.run_with_graph import run_with_graph
from torchinfo._inputs.device.set_device import set_device
from torchinfo.layer_info import LayerInfo
from typing import Any
import torch


def collect_layers(
    model: nn.Module,
    x: CORRECTED_INPUT_DATA_TYPE,
    batch_dim: int | None,
    device: torch.device | None,
    mode: Mode,
    kwargs: dict[str, Any],
) -> list[LayerInfo]:
    """Hook the model, run it on x (if given) and link the recorded layers."""
    layers, _, hooks = apply_hooks(model.__class__.__name__, model, x, batch_dim)
    if x is not None:
        kwargs = set_device(kwargs, device)
        run_with_graph(model, x, kwargs, device, mode, layers, hooks)
        missing.add_missing_container_layers(layers)
    set_children_layers(layers)
    return layers
