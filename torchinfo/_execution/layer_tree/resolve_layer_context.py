from __future__ import annotations
from torchinfo._constants.module_types import LayerContext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def resolve_layer_context(
    contexts: list[LayerContext], module_stack: list[LayerInfo]
) -> LayerContext:
    """Pick the structural context (var_name, depth, parent) matching the module's"""
    if len(contexts) == 1 or not module_stack:
        return contexts[0]
    top = module_stack[-1]
    on_stack = {id(frame.module) for frame in module_stack}
    for context in contexts:
        parent = context[2]
        while parent is not None and id(parent.module) not in on_stack:
            parent = parent.parent_info
        if parent is not None and parent.module is top.module:
            return context
    return contexts[0]
