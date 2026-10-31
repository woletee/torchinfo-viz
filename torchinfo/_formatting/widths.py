from __future__ import annotations
from typing import TYPE_CHECKING
import math

if TYPE_CHECKING:
    from torchinfo.layer_info import LayerInfo


def get_total_width(self) -> int:
    """Calculate the total width of all lines in the table."""
    return len(tuple(self.col_names)) * self.col_width + self.layer_name_width


def set_layer_name_width(
    self, summary_list: list[LayerInfo], align_val: int = 5
) -> None:
    """Set layer name width by taking the longest line length and rounding up to"""
    max_length = 0
    for info in summary_list:
        depth_indent = info.depth * align_val + 1
        layer_title = info.get_layer_name(self.show_var_name, self.show_depth)
        max_length = max(max_length, len(layer_title) + depth_indent)
    if max_length >= self.layer_name_width:
        self.layer_name_width = math.ceil(max_length / align_val) * align_val
