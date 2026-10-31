from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from torchinfo._graph.state.tensors import flatten_tensors
from torchinfo._graph.state.graph_state import GraphState
import torch


def link_inputs(state: GraphState, tensors: Iterable[torch.Tensor], key: str) -> None:
    """Add an edge from the producer of each input tensor to the node `key`."""
    for tensor in tensors:
        source = state.producers.get(id(tensor))
        if source is not None and source != key:
            state.edges.setdefault((source, key), list(tensor.shape))


def link_outputs(state: GraphState, output: Any) -> None:
    """Add one output node per tensor the whole model returned."""
    for index, tensor in enumerate(flatten_tensors(output)):
        key = f"out{index}"
        state.nodes[key] = {"kind": "output", "shape": list(tensor.shape)}
        link_inputs(state, [tensor], key)
