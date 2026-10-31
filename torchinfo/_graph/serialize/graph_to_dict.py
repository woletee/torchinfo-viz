from __future__ import annotations
from torchinfo._graph.state.graph_state import GraphState
from torchinfo._graph.serialize.node_to_dict import node_to_dict
from typing import Any


def graph_to_dict(state: GraphState | None, layers: list[Any]) -> dict[str, Any] | None:
    """Plain-data graph: nodes in execution order and deduplicated edges."""
    if state is None:
        return None
    index = {id(info): row for row, info in enumerate(layers)}
    keys: dict[str, str] = {}
    nodes = []
    for key, node in state.nodes.items():
        data = node_to_dict(key, node, index)
        if data is not None:
            keys[key] = data["key"]
            nodes.append(data)
    edges = [
        {"src": keys[src], "dst": keys[dst], "shape": shape}
        for (src, dst), shape in state.edges.items()
        if src in keys and dst in keys
    ]
    return {"nodes": nodes, "edges": edges}
