from __future__ import annotations
from collections.abc import Callable
from torch import nn
from torchinfo._execution.layer_tree.module_stack import record_layer
from torchinfo._execution.layer_tree.resolve_layer_context import resolve_layer_context
from torchinfo._execution.hooks.state import HookState
from torchinfo.layer_info import LayerInfo
from typing import Any


def construct_pre_hook(
    state: HookState, module_id: int
) -> Callable[[nn.Module, Any], None]:

    def pre_hook(module: nn.Module, inputs: Any) -> None:
        """Create a LayerInfo object to aggregate layer information."""
        del inputs
        var_name, curr_depth, parent_info = resolve_layer_context(
            state.contexts[module_id], state.module_stack
        )
        info = LayerInfo(var_name, module, curr_depth, parent_info)
        layer_ids, global_layer_info = state.layer_ids, state.global_layer_info
        record_layer(info, state.summary_list, layer_ids, global_layer_info)
        state.module_stack.append(info)

    return pre_hook
