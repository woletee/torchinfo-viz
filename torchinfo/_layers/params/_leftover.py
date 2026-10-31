from __future__ import annotations
from torchinfo._layers.tree.recursion import skip_subtree
from typing import cast


def _leftover(self, attr: str) -> int:
    """Shared implementation for the leftover_* properties: the amount of `attr`"""
    total = cast("int", getattr(self, attr))
    children = self.children
    seen: set[int] = set()
    i, n = (0, len(children))
    while i < n:
        child = children[i]
        if child.is_recursive:
            if child.layer_id not in seen:
                seen.add(child.layer_id)
                total -= cast("int", getattr(child, attr))
            i = skip_subtree(children, i, child.depth)
            continue
        seen.add(child.layer_id)
        total -= (
            cast("int", getattr(child, attr))
            if child.is_leaf_layer
            else child._leftover(attr)
        )
        i += 1
    return total
