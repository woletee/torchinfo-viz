from __future__ import annotations
from torch import nn
from torchinfo._layers.utils import rgetattr
import torch


def get_param_count(
    module: nn.Module, name: str, param: torch.Tensor
) -> tuple[int, str]:
    """Get count of number of params, accounting for mask."""
    if name.endswith("_orig"):
        without_suffix = name[:-5]
        pruned_weights = rgetattr(module, f"{without_suffix}_mask")
        if pruned_weights is not None:
            parameter_count = int(torch.sum(pruned_weights))
            return (parameter_count, without_suffix)
    return (param.nelement(), name)
