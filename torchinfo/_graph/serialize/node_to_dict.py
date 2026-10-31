from __future__ import annotations
from typing import Any


def node_to_dict(
    key: str, node: dict[str, Any], index: dict[int, int]
) -> dict[str, Any] | None:
    """Serialize one recorded node, replacing LayerInfo objects by row indices."""
    if node["kind"] == "layer":
        row = index.get(id(node["info"]))
        if row is None:
            return None
        return {"key": f"L{row}", "kind": "layer", "layer": row}
    data = {name: value for name, value in node.items() if name != "parent"}
    parent = node.get("parent")
    data["parent"] = None if parent is None else index.get(id(parent))
    return {"key": key, **data}
