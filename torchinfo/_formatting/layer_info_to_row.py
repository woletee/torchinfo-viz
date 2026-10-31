from __future__ import annotations
from torchinfo._enums.run_settings import Verbosity
from torchinfo._formatting.layer_values import layer_values
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def layer_info_to_row(
    self, layer_info: LayerInfo, reached_max_depth: bool, total_params: int
) -> str:
    """Convert layer_info to string representation of a row."""
    values_for_row = layer_values(self, layer_info, reached_max_depth, total_params)
    start_str = self.get_start_str(layer_info.depth)
    layer_name = layer_info.get_layer_name(self.show_var_name, self.show_depth)
    new_line = self.format_row(f"{start_str}{layer_name}", values_for_row)
    if self.verbose == Verbosity.VERBOSE:
        for inner_name, inner_layer_info in layer_info.inner_layers.items():
            prefix = self.get_start_str(layer_info.depth + 1)
            new_line += self.format_row(f"{prefix}{inner_name}", inner_layer_info)
    return new_line
