from __future__ import annotations
from torchinfo._inputs.traverse_input_data import traverse_input_data
from typing import Any
import torch


def set_device(data: Any, device: torch.device | None) -> Any:
    """Sets device for all input types and collections of input types."""
    return (
        data
        if device is None
        else traverse_input_data(
            data,
            action_fn=lambda data: data.to(device, non_blocking=True),
            aggregate_fn=type,
        )
    )
