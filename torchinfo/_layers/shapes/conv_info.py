from __future__ import annotations
from collections.abc import Iterable
from torch import nn
import torch


def get_kernel_size(module: nn.Module) -> int | list[int] | None:
    if hasattr(module, "kernel_size"):
        k = module.kernel_size
        kernel_size: int | list[int]
        if isinstance(k, Iterable) and (not isinstance(k, torch.Tensor) or k.ndim > 0):
            kernel_size = list(k)
        elif isinstance(k, (int, torch.Tensor)):
            kernel_size = int(k)
        else:
            raise TypeError(f"kernel_size has an unexpected type: {type(k)}")
        return kernel_size
    return None


def get_groups(module: nn.Module) -> int | None:
    if hasattr(module, "groups"):
        return int(module.groups)
    return None
