from __future__ import annotations
from torchinfo._graph.state.tensors import flatten_tensors
from torchinfo._graph.state.graph_state import GraphState
from torchinfo._graph.state.links import link_inputs
from torchinfo._graph.state.links import link_outputs
from torchinfo._graph.state.tensors import set_producer
from typing import Any


def module_hooks(state: GraphState) -> tuple[Any, Any]:
    """Hooks that turn each leaf-module call into a graph node."""

    def pre_hook(module: Any, args: Any, kwargs: Any) -> None:
        # torchinfo's own pre-hook runs first and appends this call's LayerInfo.
        info = state.layers[-1] if state.layers else None
        info = info if info is not None and info.module is module else None
        leaf = info is not None and info.is_leaf_layer
        if leaf:
            state.nodes[f"L{id(info)}"] = {"kind": "layer", "info": info}
            link_inputs(state, flatten_tensors((args, kwargs)), f"L{id(info)}")
        state.stack.append((info, leaf))

    def post_hook(module: Any, args: Any, kwargs: Any, output: Any) -> None:
        info, leaf = state.stack.pop()
        if leaf:
            set_producer(state, flatten_tensors(output), f"L{id(info)}")
        if not state.stack:
            link_outputs(state, output)

    return pre_hook, post_hook
