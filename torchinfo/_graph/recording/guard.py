from __future__ import annotations
from collections.abc import Callable
from torchinfo._graph.state.graph_state import GraphState
from typing import Any


def guard(state: GraphState, hook: Callable[..., None]) -> Callable[..., None]:
    """Wrap a hook so recording problems never break the user's forward pass."""

    def safe(*args: Any) -> None:
        if state.failed:
            return
        try:
            hook(*args)
        except Exception:
            state.failed = True

    return safe
