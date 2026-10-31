from __future__ import annotations
from torchinfo._export.layer_to_dict import layer_to_dict
from torchinfo._export.totals_to_dict import totals_to_dict
from torchinfo._graph.serialize.graph_to_dict import graph_to_dict
from typing import Any


def to_dict(self) -> dict[str, Any]:
    """Return the summary as plain data: totals, a parent-linked tree and a graph."""
    layers, stack, seen = [], [], set()
    called = {id(layer.module) for layer in self.summary_list}
    for index, layer in enumerate(self.summary_list):
        while stack and self.summary_list[stack[-1]].depth >= layer.depth:
            stack.pop()
        layers.append(
            layer_to_dict(layer, index, stack[-1] if stack else None, seen, called)
        )
        stack.append(index)
    root = self.summary_list[0] if self.summary_list else None
    return {
        "model": root.class_name if root is not None else "Model",
        "has_input": bool(self.input_size),
        "input_size": self.input_size,
        "totals": totals_to_dict(self),
        "layers": layers,
        "graph": graph_to_dict(getattr(root, "graph", None), self.summary_list),
    }
