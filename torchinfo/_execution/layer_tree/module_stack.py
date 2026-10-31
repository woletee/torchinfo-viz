from __future__ import annotations
import warnings


def pop_module_stack(module_stack, module):
    """Pop only the matching frame; retain other frames on unexpected hook order."""
    if module_stack and module_stack[-1].module is module:
        module_stack.pop()
    else:
        top = module_stack[-1].module if module_stack else None
        warnings.warn(
            f"Unexpected forward-hook order: expected {type(module).__name__} on top of the execution stack, found {(type(top).__name__ if top is not None else 'empty stack')}. Skipping the stack pop to stay safe; the layer hierarchy for shared modules may be slightly inaccurate.",
            stacklevel=2,
        )


def record_layer(info, layers, layer_ids, global_layer_info):
    info.calculate_num_params()
    info.check_recursive(layer_ids)
    layers.append(info)
    layer_ids.add(info.layer_id)
    global_layer_info[info.layer_id] = info
