from __future__ import annotations
from torchinfo._enums.table_settings import ColumnSettings
from torchinfo._enums.Units import Units


HEADER_TITLES = {
    ColumnSettings.KERNEL_SIZE: "Kernel Shape",
    ColumnSettings.GROUPS: "Groups",
    ColumnSettings.INPUT_SIZE: "Input Shape",
    ColumnSettings.OUTPUT_SIZE: "Output Shape",
    ColumnSettings.NUM_PARAMS: "Param #",
    ColumnSettings.PARAMS_PERCENT: "Param %",
    ColumnSettings.MULT_ADDS: "Mult-Adds",
    ColumnSettings.TRAINABLE: "Trainable",
}


CONVERSION_FACTORS = {
    Units.TERABYTES: 1000000000000.0,
    Units.GIGABYTES: 1000000000.0,
    Units.MEGABYTES: 1000000.0,
    Units.KILOBYTES: 1000.0,
    Units.NONE: 1,
}
