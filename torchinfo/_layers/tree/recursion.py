from __future__ import annotations


def skip_subtree(children, index, depth):
    index += 1
    while index < len(children) and children[index].depth > depth:
        index += 1
    return index


def check_recursive(self, layer_ids: set[int]) -> None:
    """If the current module is already-used, mark as (recursive)."""
    if self.layer_id in layer_ids:
        self.is_recursive = True
