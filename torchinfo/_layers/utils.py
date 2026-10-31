from __future__ import annotations
from collections.abc import Iterable
from torch import nn
import torch


def prod(num_list: Iterable[int] | torch.Size) -> int:
    result = 1
    if isinstance(num_list, Iterable):
        for item in num_list:
            result *= prod(item) if isinstance(item, Iterable) else item
    return result


def rgetattr(module: nn.Module, attr: str) -> torch.Tensor | None:
    """Get the tensor submodule called attr from module."""
    for attr_i in attr.split("."):
        if not hasattr(module, attr_i):
            return None
        module = getattr(module, attr_i)
    assert isinstance(module, torch.Tensor)
    return module
