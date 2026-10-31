from __future__ import annotations
from torch import nn
from torch.jit import ScriptModule


def get_class_name(module: nn.Module) -> str:
    """Return the display class name, unwrapping TorchScript modules."""
    if isinstance(module, ScriptModule):
        return str(module.original_name)
    return str(module._get_name())


def get_layer_name(self, show_var_name: bool, show_depth: bool) -> str:
    layer_name = self.class_name
    if show_var_name and self.var_name:
        layer_name += f" ({self.var_name})"
    if show_depth and self.depth > 0:
        layer_name += f": {self.depth}"
        if self.depth_index is not None:
            layer_name += f"-{self.depth_index}"
    return layer_name
