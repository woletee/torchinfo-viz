from __future__ import annotations
from torchinfo._graph.state.tensors import flatten_tensors
from torchinfo._graph.state.graph_state import GraphState
from torchinfo._graph.state.links import link_inputs
from torchinfo._graph.recording.op_names import PASSTHROUGH, op_name
from torchinfo._graph.state.tensors import set_producer
from typing import Any


def record_op(
    state: GraphState, func: Any, args: Any, kwargs: Any, output: Any
) -> None:
    """Record a tensor op that runs outside leaf modules, e.g. a cat in forward()."""
    if state.inside_leaf():
        return
    outputs = list(flatten_tensors(output))
    inputs = [t for t in flatten_tensors((args, kwargs)) if id(t) in state.producers]
    if not outputs or not inputs:
        return
    name = op_name(func)
    sources = {state.producers[id(t)] for t in inputs}
    if name in PASSTHROUGH and len(sources) == 1:
        set_producer(state, outputs, sources.pop())
        return
    key = f"op{len(state.nodes)}"
    parent = state.stack[-1][0] if state.stack else None
    shape = list(outputs[0].shape)
    state.nodes[key] = {"kind": "op", "name": name, "parent": parent, "shape": shape}
    link_inputs(state, inputs, key)
    set_producer(state, outputs, key)
