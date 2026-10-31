from __future__ import annotations
from torch import nn
from torch.utils.hooks import RemovableHandle
from typing import Any
from torchinfo._graph.state.tensors import flatten_tensors
from torchinfo._graph.state.graph_state import GraphState
from torchinfo._graph.recording.guard import guard
from torchinfo._graph.recording.module_hooks import module_hooks
from torchinfo._graph.state.tensors import set_producer


def add_inputs(state: GraphState, x: Any, kwargs: Any) -> None:
    """Add one input node per tensor passed to the model."""
    for index, tensor in enumerate(flatten_tensors((x, kwargs))):
        state.nodes[f"in{index}"] = {"kind": "input", "shape": list(tensor.shape)}
        set_producer(state, [tensor], f"in{index}")


def add_hooks(state: GraphState, model: nn.Module) -> list[RemovableHandle]:
    """Attach the recording hooks to every module of the model."""
    pre_hook, post_hook = (guard(state, hook) for hook in module_hooks(state))
    handles = []
    for module in model.modules():
        try:
            handles.append(module.register_forward_pre_hook(pre_hook, with_kwargs=True))
            handles.append(module.register_forward_hook(post_hook, with_kwargs=True))
        except RuntimeError:  # TorchScript modules reject kwargs hooks
            state.failed = True
    return handles
