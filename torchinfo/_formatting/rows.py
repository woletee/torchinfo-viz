from __future__ import annotations
from torchinfo._constants.display import HEADER_TITLES
from torchinfo._enums.table_settings import ColumnSettings


def format_row(self, layer_name: str, row_values: dict[ColumnSettings, str]) -> str:
    """Get the string representation of a single layer of the model."""
    info_to_use = [row_values.get(row_type, "") for row_type in self.col_names]
    new_line = f"{layer_name:<{self.layer_name_width}} "
    for info in info_to_use:
        new_line += f"{info:<{self.col_width}} "
    return new_line.rstrip() + "\n"


def header_row(self) -> str:
    layer_header = ""
    if self.show_var_name:
        layer_header += " (var_name)"
    if self.show_depth:
        layer_header += ":depth-idx"
    return self.format_row(f"Layer (type{layer_header})", HEADER_TITLES)
