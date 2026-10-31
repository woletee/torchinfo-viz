from __future__ import annotations
from torchinfo._execution.layer_tree.ancestor_hierarchy import ancestor_hierarchy
from torchinfo._execution.layer_tree.insert_containers import insert_containers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def add_missing_container_layers(summary_list: list[LayerInfo]) -> None:
    """Insert unexecuted containers into the runtime hierarchy."""
    layer_ids = {layer.layer_id for layer in summary_list}
    current_hierarchy = {}
    for idx, layer in enumerate(summary_list):
        hierarchy = ancestor_hierarchy(layer)
        insert_containers(
            summary_list, idx, layer, hierarchy, current_hierarchy, layer_ids
        )
        current_hierarchy[layer.depth] = layer
        depth = layer.depth + 1
        while depth in current_hierarchy:
            current_hierarchy.pop(depth)
            depth += 1
