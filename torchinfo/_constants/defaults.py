from __future__ import annotations
from torchinfo._enums.table_settings import ColumnSettings
from torchinfo._enums.table_settings import RowSettings


DEFAULT_COLUMN_NAMES = (ColumnSettings.OUTPUT_SIZE, ColumnSettings.NUM_PARAMS)


DEFAULT_ROW_SETTINGS = {RowSettings.DEPTH}


REQUIRES_INPUT = {
    ColumnSettings.INPUT_SIZE,
    ColumnSettings.OUTPUT_SIZE,
    ColumnSettings.MULT_ADDS,
}
