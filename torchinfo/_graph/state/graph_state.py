from __future__ import annotations
from typing import Any


class GraphState:
    """Data-flow graph recorded while the model runs its forward pass."""

    def __init__(self, layers: list[Any]) -> None:
        self.layers = layers  # torchinfo's summary_list, filled by its pre-hooks
        self.producers: dict[int, str] = {}  # id(tensor) -> key of its node
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[tuple[str, str], list[int]] = {}
        self.stack: list[tuple[Any, bool]] = []  # (LayerInfo | None, is_leaf)
        self.failed = False

    def inside_leaf(self) -> bool:
        return bool(self.stack) and self.stack[-1][1]
