from __future__ import annotations
from torchinfo._enums.table_settings import ColumnSettings


def inner_parameter_row(name, param, count):
    kernel = list(param.size())
    if name == "weight" and len(kernel) > 1:
        kernel[0], kernel[1] = (kernel[1], kernel[0])
    return {
        ColumnSettings.KERNEL_SIZE: str(kernel),
        ColumnSettings.NUM_PARAMS: f"├─{count:,}",
    }
