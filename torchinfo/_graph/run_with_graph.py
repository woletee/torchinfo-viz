from __future__ import annotations
from torch import nn
from torchinfo._enums.run_settings import Mode
from torchinfo._execution.forward.execute_forward import execute_forward
from torchinfo._graph.record_graph import record_graph
from typing import Any
import torch


def run_with_graph(
    model: nn.Module,
    x: Any,
    kwargs: dict[str, Any],
    device: torch.device | None,
    mode: Mode,
    layers: list[Any],
    hooks: Any,
) -> None:
    """Run the forward pass while recording its graph onto the root layer."""
    with record_graph(model, x, kwargs, layers) as state:
        execute_forward(model, x, kwargs, device, mode, layers, hooks)
    if layers:
        layers[0].graph = None if state.failed else state
