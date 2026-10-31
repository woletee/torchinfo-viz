from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def get_children_layers(summary_list: list[LayerInfo], index: int) -> list[LayerInfo]:
    """Fetches all of the children of a given layer."""
    num_children = 0
    for layer in summary_list[index + 1 :]:
        if layer.depth <= summary_list[index].depth:
            break
        num_children += 1
    return summary_list[index + 1 : index + 1 + num_children]
