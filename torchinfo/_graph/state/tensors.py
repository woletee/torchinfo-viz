from __future__ import annotations
from collections.abc import Iterator, Iterable
from typing import Any
from torchinfo._graph.state.graph_state import GraphState
import torch
import weakref


def flatten_tensors(value: Any) -> Iterator[torch.Tensor]:
    """Yield every tensor nested in tuples, lists and dicts."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from flatten_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from flatten_tensors(item)


def set_producer(state: GraphState, tensors: Iterable[torch.Tensor], key: str) -> None:
    """Remember that `key` produced these tensors, forgetting them once freed."""
    for tensor in tensors:
        tensor_id = id(tensor)
        state.producers[tensor_id] = key
        # Freed tensors drop out, so a recycled id() never links unrelated nodes.
        weakref.finalize(tensor, state.producers.pop, tensor_id, None)
