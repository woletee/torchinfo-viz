from __future__ import annotations
from collections.abc import Callable
from torch import nn
from torchinfo._execution.layer_tree.module_stack import pop_module_stack
from torchinfo._layers.utils import prod
from typing import Any
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def construct_hook(
    global_layer_info: dict[int, LayerInfo],
    module_stack: list[LayerInfo],
    batch_dim: int | None,
) -> Callable[[nn.Module, Any, Any], None]:

    def hook(module: nn.Module, inputs: Any, outputs: Any) -> None:
        info = global_layer_info[id(module)]
        if info.contains_lazy_param:
            info.calculate_num_params()
        info.input_size, _ = info.calculate_size(inputs, batch_dim)
        info.output_size, elem_bytes = info.calculate_size(outputs, batch_dim)
        info.output_bytes = elem_bytes * prod(info.output_size)
        info.executed = True
        info.calculate_macs()
        pop_module_stack(module_stack, module)

    return hook
