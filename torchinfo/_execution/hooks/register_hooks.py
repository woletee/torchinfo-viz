from __future__ import annotations
from torchinfo._execution.hooks.construct_hook import construct_hook
from torchinfo._execution.hooks.construct_pre_hook import construct_pre_hook


def register_hooks(module, state):
    module_id = id(module)
    for hook in state.hooks.get(module_id, ()):
        hook.remove()
    pre_hook = construct_pre_hook(state, module_id)
    post_hook = construct_hook(
        state.global_layer_info, state.module_stack, state.batch_dim
    )
    state.hooks[module_id] = (
        module.register_forward_pre_hook(pre_hook),
        module.register_forward_hook(post_hook),
    )
