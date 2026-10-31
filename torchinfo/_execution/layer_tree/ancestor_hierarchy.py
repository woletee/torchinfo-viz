from __future__ import annotations


def ancestor_hierarchy(layer):
    hierarchy = {}
    parent = layer.parent_info
    while parent is not None and parent.depth > 0:
        hierarchy[parent.depth] = parent
        parent = parent.parent_info
    return hierarchy
