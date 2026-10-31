from __future__ import annotations
from torchinfo._enums.table_settings import ColumnSettings
from torchinfo._enums.table_settings import RowSettings
from torchinfo._enums.Units import Units


def __init__(
    self,
    max_depth: int,
    verbose: int,
    col_names: tuple[ColumnSettings, ...],
    col_width: int,
    row_settings: set[RowSettings],
) -> None:
    self.max_depth = max_depth
    self.verbose = verbose
    self.col_names = col_names
    self.col_width = col_width
    self.row_settings = row_settings
    self.params_count_units = Units.NONE
    self.params_size_units = Units.AUTO
    self.macs_units = Units.AUTO
    self.layer_name_width = 40
    self.ascii_only = RowSettings.ASCII_ONLY in self.row_settings
    self.show_var_name = RowSettings.VAR_NAMES in self.row_settings
    self.show_depth = RowSettings.DEPTH in self.row_settings
    self.hide_recursive_layers = RowSettings.HIDE_RECURSIVE_LAYERS in self.row_settings
