from __future__ import annotations
from collections.abc import Iterator
from contextlib import contextmanager
from torch import nn
from torchinfo._graph.recording.setup import add_hooks
from torchinfo._graph.recording.setup import add_inputs
from torchinfo._graph.state.graph_state import GraphState
from torchinfo._graph.recording.recording_mode import RecordingMode
from typing import Any


@contextmanager
def record_graph(
    model: nn.Module, x: Any, kwargs: Any, layers: list[Any]
) -> Iterator[GraphState]:
    """Record the data-flow graph of the forward pass run inside this block."""
    state = GraphState(layers)
    add_inputs(state, x, kwargs)
    handles = add_hooks(state, model)
    try:
        with RecordingMode(state):
            yield state
    finally:
        for handle in handles:
            handle.remove()
