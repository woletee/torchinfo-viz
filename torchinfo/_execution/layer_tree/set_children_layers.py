from __future__ import annotations
from torchinfo._layers.tree.get_children_layers import get_children_layers
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def set_children_layers(summary_list: list[LayerInfo]) -> None:
    """Populates the children and depth_index fields of all LayerInfo."""
    idx: dict[int, int] = {}
    for i, layer in enumerate(summary_list):
        idx[layer.depth] = idx.get(layer.depth, 0) + 1
        layer.depth_index = idx[layer.depth]
        layer.children = get_children_layers(summary_list, i)
