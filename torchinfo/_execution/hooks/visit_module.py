from __future__ import annotations
from torch import nn
from torchinfo._constants.input_types import CORRECTED_INPUT_DATA_TYPE
from torchinfo._constants.module_types import WRAPPER_MODULES
from torchinfo._execution.layer_tree.module_stack import record_layer
from torchinfo._execution.hooks.register_hooks import register_hooks
from torchinfo._execution.hooks.state import HookState
from torchinfo.layer_info import LayerInfo


def visit_module(
    state: HookState,
    input_data: CORRECTED_INPUT_DATA_TYPE,
    frame: tuple[str, nn.Module, int, LayerInfo | None],
) -> list[tuple[str, nn.Module, int, LayerInfo]]:
    """Record or hook one module and return its children to visit next."""
    var_name, module, depth, parent = frame
    info = LayerInfo(var_name, module, depth, parent)
    state.global_layer_info[id(module)] = info
    state.contexts.setdefault(id(module), []).append((var_name, depth, parent))
    if input_data is None or isinstance(module, WRAPPER_MODULES):
        record_layer(info, state.summary_list, state.layer_ids, state.global_layer_info)
    else:
        register_hooks(module, state)
    return [
        (name, child, depth + 1, info)
        for name, child in reversed(module._modules.items())
        if child is not None
    ]
