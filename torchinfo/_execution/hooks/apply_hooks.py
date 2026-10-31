from __future__ import annotations
from torch import nn
from torch.utils.hooks import RemovableHandle
from torchinfo._constants.input_types import CORRECTED_INPUT_DATA_TYPE
from torchinfo._execution.hooks.state import HookState
from torchinfo._execution.hooks.visit_module import visit_module
from torchinfo.layer_info import LayerInfo


def apply_hooks(
    model_name: str,
    module: nn.Module,
    input_data: CORRECTED_INPUT_DATA_TYPE,
    batch_dim: int | None,
) -> tuple[
    list[LayerInfo],
    dict[int, LayerInfo],
    dict[int, tuple[RemovableHandle, RemovableHandle]],
]:
    """Traverse modules, collecting contexts and installing forward hooks."""
    state = HookState(batch_dim)
    stack = [(model_name, module, 0, None)]
    while stack:
        stack += visit_module(state, input_data, stack.pop())
    return (state.summary_list, state.global_layer_info, state.hooks)
