from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def layers_to_str(self, summary_list: list[LayerInfo], total_params: int) -> str:
    """Print each layer of the model using only current layer info."""
    new_str = ""
    for layer_info in summary_list:
        if layer_info.depth > self.max_depth or (
            self.hide_recursive_layers and layer_info.is_recursive
        ):
            continue
        reached_max_depth = layer_info.depth == self.max_depth
        new_str += self.layer_info_to_row(layer_info, reached_max_depth, total_params)
    return new_str
